"""
In-memory ring buffer for /quick-config trace panel + cb:map autocomplete.

Holds the last N transcription traces (raw whisper output, per-step
before/after, final post-pipeline text) plus pre-tokenized
words + adjacent two-word phrases (bigrams) for the autocomplete
datalist on /quick-config. The bigram pass exists for the common
"whisper split a compound" case — user dictates "Hanspeter", model
emits "Hans Peter"; typing "Hans" should offer both single-word and
two-word completions so the user can pick the phrase as a cb:map
key in one step.

Lost on service restart, capped at _BUFFER_MAX entries. The buffer
contains literal patient dictation snippets on a medical-deployment
host — DO NOT log buffer contents and DO NOT persist them. The
on-disk log file already holds the same trace via main._format_request_block;
that's the canonical durable record. This module exists only to
surface those traces to /quick-config without making the page parse
the log file.
"""
from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from typing import Any

# Capped at 20 to match metrics.recent_tx; small RAM footprint, bounded
# PHI surface, fits on one /quick-config screen without pagination.
_BUFFER_MAX = 20

# Token shape: at least one letter, then word-chars (letters/digits/-),
# 2 to 64 chars. Drops single-char punctuation noise without aggressively
# splitting compound German medical terms (which is the whole point —
# the user's mis-recognition target IS the compound).
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

recent_traces: deque[dict[str, Any]] = deque(maxlen=_BUFFER_MAX)

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
    matches = list(_TOKEN_RE.finditer(text or ""))
    seen: dict[str, str] = {}
    for a, b in zip(matches, matches[1:]):
        ta, tb = a.group(0), b.group(0)
        if not (_TOKEN_MIN_LEN <= len(ta) <= _TOKEN_MAX_LEN):
            continue
        if not (_TOKEN_MIN_LEN <= len(tb) <= _TOKEN_MAX_LEN):
            continue
        if ta.lower() in _STOPWORDS or tb.lower() in _STOPWORDS:
            continue
        between = (text or "")[a.end():b.start()]
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
    user_id: str | None = None,
) -> None:
    """Append a new trace to the ring buffer and broadcast to all SSE
    subscribers. Called from main.py's transcribe handler after the
    existing logger.info(_format_request_block(...)) line.

    `request_id` is the uuid4 hex stamped on the request by main.py.
    Surfaced on each trace entry so /quick-config can correlate a
    user-submitted report with the durable text log (which prints
    `req=<id[:8]>` in the per-request block). None for any pre-feature
    or upstream call that omits it.

    `steps` is a list of (label, before, after) tuples — same shape
    main.py builds when cfg.TRACE_ENABLED. Pass [] when tracing is off
    so the autocomplete feature still works (raw + final + tokens are
    the only fields the autocomplete needs).

    Emits both single-word tokens AND adjacent two-word phrases
    (bigrams). The bigram pass catches the common "whisper split a
    compound" case: user dictates "Hanspeter", model emits "Hans
    Peter"; the datalist on /quick-config will then offer "Hans Peter"
    when the user types "Hans" in a cb:map left field. Bigrams are
    filtered with the same stopword + length rules as tokens to avoid
    noise."""
    try:
        import api_keys_store
        username = api_keys_store.get_username(user_id)
    except Exception:
        username = None
    final_text = final or ""
    entry: dict[str, Any] = {
        "ts": time.time(),
        "request_id": request_id,
        "model": model,
        "raw": raw or "",
        "steps": [list(s) if isinstance(s, (tuple, list)) else s
                  for s in (steps or [])],
        "final": final_text,
        "tokens": _tokenize(final_text),
        "bigrams": _extract_bigrams(final_text),
        "user_id": user_id,
        "username": username,
    }
    recent_traces.append(entry)
    _broadcast({"event": "trace", "data": entry})


def clear() -> None:
    """Wipe the ring buffer and broadcast a clear event so all open
    /quick-config tabs flush their UI."""
    recent_traces.clear()
    _broadcast({"event": "clear", "data": {}})


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
