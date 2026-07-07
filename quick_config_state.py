"""
Tokenization + SSE broadcast layer for /quick-config recent transcriptions.

Durable storage lives in `transcriptions_store` (SQLite, WAL). This
module:
  * tokenizes each transcription into single words + adjacent two-word
    phrases (bigrams) for the cb:map autocomplete datalist on
    /quick-config;
  * forwards each finished trace to the SQLite store via record_trace();
  * broadcasts the trace to live SSE subscribers so the UI updates in
    real time without polling.

The bigram pass exists for the common "whisper split a compound" case
— user dictates "Hanspeter", model emits "Hans Peter"; typing "Hans"
should offer both single-word and two-word completions so the user
can pick the phrase as a cb:map key in one step.

The trace + tokens carry literal dictation snippets, which can be
sensitive personal data — DO NOT log buffer contents. The on-disk log
file already holds the same trace via main._format_request_block; the
new SQLite store is the canonical structured durable record.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

logger = logging.getLogger("whisper-api")

# Token shape: at least one letter, then word-chars (letters/digits/-),
# 2 to 64 chars. Drops single-char punctuation noise without aggressively
# splitting compound words (which is the whole point — the user's
# mis-recognition target IS the compound).
_TOKEN_MIN_LEN = 2
_TOKEN_MAX_LEN = 64
_TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß][\w\-äöüÄÖÜß]*")

# Small German stopword set — never useful as autocomplete candidates
# for cb:map keys. Lowercase compare; preserve original casing on display.
_STOPWORDS = frozenset({
    "der", "die", "das", "den", "dem", "und", "oder", "mit", "von", "zu",
    "im", "in", "an", "am", "auf", "für", "fuer", "ist", "sind", "wird",
    "wurde", "werden", "ein", "eine", "einen", "einer", "einem", "auch",
    "sich", "nicht", "nur", "aber", "als", "wie", "so", "bei", "aus",
    "nach", "vor", "über", "unter", "durch",
})

# Per-transcription extraction cap: keep candidates relevant; bound
# memory and SSE payload size.
_TOKEN_CAP = 100

# Same idea for adjacent two-word phrases — kept lower because the
# multiplicative effect (N tokens → ~N-1 bigrams) plus per-trace
# datalist contribution would otherwise dominate the option list.
_BIGRAM_CAP = 50

# Each subscriber is a bounded asyncio.Queue. Bounded so a slow / stuck
# SSE client can't grow memory without limit; on overflow we drop the
# event for that client (the client misses a trace but stays connected).
_subscribers: list[asyncio.Queue] = []
_SUBSCRIBER_QUEUE_MAX = 64


def _tokenize(text: str) -> list[str]:
    """Whitespace + punctuation split. Drop stopwords and out-of-range
    tokens. Preserve original casing of FIRST occurrence per lowercased
    key. Cap at _TOKEN_CAP candidates. Returns insertion-ordered list."""
    seen: dict[str, str] = {}
    for m in _TOKEN_RE.finditer(text or ""):
        tok = m.group(0)
        if not (_TOKEN_MIN_LEN <= len(tok) <= _TOKEN_MAX_LEN):
            continue
        key = tok.lower()
        if key in _STOPWORDS:
            continue
        if key not in seen:
            seen[key] = tok
            if len(seen) >= _TOKEN_CAP:
                break
    return list(seen.values())


def _extract_bigrams(text: str) -> list[str]:
    """Adjacent content-token pairs joined with a single space. A pair
    counts only when BOTH tokens are content (non-stopword, in-range
    length) AND the original text has nothing but whitespace between
    them — a comma or period between two words means they belong to
    different phrases and should NOT form a bigram.

    Insertion-ordered, lowercased de-duped: first-seen casing wins.
    Cap at _BIGRAM_CAP. Returns [] for empty/whitespace input.

    Example: "Hans Peter, und Anna Müller" → ["Hans Peter", "Anna Müller"]
    (the comma blocks the "Peter und" pair; "und" is a stopword
    anyway, which blocks "und Anna")."""
    src = text or ""
    matches = list(_TOKEN_RE.finditer(src))
    seen: dict[str, str] = {}
    for a, b in zip(matches, matches[1:]):
        ta, tb = a.group(0), b.group(0)
        if not (_TOKEN_MIN_LEN <= len(ta) <= _TOKEN_MAX_LEN):
            continue
        if not (_TOKEN_MIN_LEN <= len(tb) <= _TOKEN_MAX_LEN):
            continue
        if ta.lower() in _STOPWORDS or tb.lower() in _STOPWORDS:
            continue
        between = src[a.end():b.start()]
        if between.strip():
            continue
        phrase = f"{ta} {tb}"
        key = phrase.lower()
        if key in seen:
            continue
        seen[key] = phrase
        if len(seen) >= _BIGRAM_CAP:
            break
    return list(seen.values())


def record_trace(
    *,
    request_id: str | None = None,
    model: str,
    raw: str,
    steps: list,
    final: str,
    language: str | None = None,
    source: str = "file",
    user_id: str | None = None,
) -> None:
    """Persist a finished trace to the durable store and broadcast it to
    every live SSE subscriber. Called from main.py's transcribe handler
    after the existing logger.info(_format_request_block(...)) line.

    `request_id` is the uuid4 hex stamped on the request by main.py;
    surfaced on each entry so /quick-config can correlate a user-
    submitted report with the durable text log (`req=<id[:8]>` in the
    per-request block) AND so the later metrics.record_transcription()
    call can patch timing fields onto the same row via UPSERT.

    `steps` is a list of (label, before, after) tuples — same shape
    main.py builds when cfg.TRACE_ENABLED. Pass [] when tracing is off
    so the autocomplete feature still works (raw + final + tokens are
    the only fields the autocomplete needs)."""
    if not request_id:
        # No request_id means we cannot key the UPSERT and the entry
        # cannot be jumped-to from /reports. Skip silently — the live
        # SSE broadcast still fires below.
        pass
    try:
        import api_keys_store
        username = api_keys_store.get_username(user_id)
    except Exception:
        username = None
    final_text = final or ""
    tokens = _tokenize(final_text)
    bigrams = _extract_bigrams(final_text)
    created_ts = time.time()
    entry: dict[str, Any] = {
        "ts": created_ts,
        "created_ts": created_ts,
        "request_id": request_id,
        "model": model,
        "language": language,
        "source": source or "file",
        "raw": raw or "",
        "raw_text": raw or "",
        "steps": [list(s) if isinstance(s, (tuple, list)) else s
                  for s in (steps or [])],
        "final": final_text,
        "final_text": final_text,
        "tokens": tokens,
        "bigrams": bigrams,
        "user_id": user_id,
        "username": username,
        "status": "ok",
    }
    if request_id:
        try:
            import config as cfg
            import transcriptions_store
            transcriptions_store.record_trace(
                request_id=request_id,
                model=model,
                raw=raw or "",
                final=final_text,
                steps=entry["steps"],
                tokens=tokens,
                bigrams=bigrams,
                language=language,
                source=source,
                user_id=user_id,
                username=username,
                created_ts=created_ts,
                prune_every=int(getattr(cfg, "RECENT_TRANSCRIPTIONS_PRUNE_EVERY", 50)),
                max_rows=int(getattr(cfg, "RECENT_TRANSCRIPTIONS_MAX", 500)),
                ttl_days=float(getattr(cfg, "RECENT_TRANSCRIPTIONS_TTL_DAYS", 30)),
            )
        except Exception as e:
            logger.warning("[recent-tx] persist failed: %s", e)
    _broadcast({"event": "trace", "data": entry})


def subscribe() -> asyncio.Queue:
    """Register a new SSE subscriber. Caller MUST call unsubscribe()
    in a finally block when the connection ends."""
    q: asyncio.Queue = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_MAX)
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    try:
        _subscribers.remove(q)
    except ValueError:
        pass


def _broadcast(item: dict[str, Any]) -> None:
    """Push to every subscriber; drop on overflow rather than blocking."""
    for q in list(_subscribers):
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            # Subscriber too slow; drop this event for them. They stay
            # subscribed; next event may succeed if their consumer
            # caught up.
            pass
