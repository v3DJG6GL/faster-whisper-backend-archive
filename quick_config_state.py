"""
In-memory ring buffer for /quick-config trace panel + cb:map autocomplete.

Holds the last N transcription traces (raw whisper output, per-step
before/after, final post-pipeline text) plus pre-tokenized
words/bigrams for the autocomplete datalist on /quick-config.

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

# Per-transcription extraction caps: keep candidates relevant; bound
# memory and SSE payload size.
_TOKEN_CAP = 100
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


def _bigrams(text: str) -> list[str]:
    """Adjacent non-stopword pairs from `text`. Skip pairs where either
    side is a stopword. Dedupe case-insensitively. Cap at _BIGRAM_CAP."""
    raw = [m.group(0) for m in _TOKEN_RE.finditer(text or "")]
    out: list[str] = []
    seen: set[str] = set()
    for a, b in zip(raw, raw[1:]):
        if a.lower() in _STOPWORDS or b.lower() in _STOPWORDS:
            continue
        if not (_TOKEN_MIN_LEN <= len(a) <= _TOKEN_MAX_LEN):
            continue
        if not (_TOKEN_MIN_LEN <= len(b) <= _TOKEN_MAX_LEN):
            continue
        bg = f"{a} {b}"
        key = bg.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(bg)
        if len(out) >= _BIGRAM_CAP:
            break
    return out


def record_trace(
    *,
    model: str,
    raw: str,
    steps: list,
    final: str,
) -> None:
    """Append a new trace to the ring buffer and broadcast to all SSE
    subscribers. Called from main.py's transcribe handler after the
    existing logger.info(_format_request_block(...)) line.

    `steps` is a list of (label, before, after) tuples — same shape
    main.py builds when cfg.TRACE_ENABLED. Pass [] when tracing is off
    so the autocomplete feature still works (raw + final + tokens are
    the only fields the autocomplete needs)."""
    entry: dict[str, Any] = {
        "ts": time.time(),
        "model": model,
        "raw": raw or "",
        "steps": [list(s) if isinstance(s, (tuple, list)) else s
                  for s in (steps or [])],
        "final": final or "",
        "tokens": _tokenize(final or ""),
        "bigrams": _bigrams(final or ""),
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
