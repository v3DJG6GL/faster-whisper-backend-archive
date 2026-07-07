"""Durable per-key / per-user usage rollup.

SQLite (stdlib) in WAL mode — single-file, crash-safe, indexed. Lives at
cfg.USAGE_DB (defaults to usage.local.sqlite3 alongside config.local.json).

Why a separate store: the recent-transcriptions table
(transcriptions_store.py) is a pruned rolling window (row-cap + 30-day TTL),
so it cannot back lifetime usage totals. This store keeps a compact HOURLY
ROLLUP — one row per (hour, key_id) — that is never aggressively pruned, so
lifetime totals are a SUM over hours.

Each /transcribe request bumps one row via an UPSERT, called from
metrics.record_transcription (which already runs inside a try/except on the
outer finally of the transcribe handler, on both success and error paths).

**Bucketing is UTC epoch-hour** (`int(ts // 3600)`). Storing in UTC at hour
granularity lets every consumer reckon "days" in whatever timezone it wants by
summing the hours that fall inside that timezone's local day:

  - the per-user /quick-config "today" banner reckons in the VIEWER's local
    timezone (the browser sends its local-midnight epoch);
  - the admin /stats + /api-keys dashboards reckon in the SERVER's local
    timezone (the operator's perspective), via local_day_start_hour() and
    epoch_day_for().

`series()` aggregates hours into server-local days and returns days-since-epoch,
so `day * 86400` is still UTC midnight of that calendar date and the WebUI's
`new Date(day*86400*1000).toISOString().slice(0,10)` renders the right label.

Hour granularity is exact for whole-hour UTC offsets (CET/CEST = +1/+2). For
half-hour-offset zones (e.g. IST +5:30) a transcription in the partial hour at
a day boundary can land in the adjacent day — acceptable at this grain.

user_id is denormalised into every row (not resolved via JOIN) so revoking
a user or key never drops their historical usage, and aggregation needs no
join to the api-keys DB.

Module-level connection: WAL mode lets us share one connection across
threads (check_same_thread=False); _lock serialises writers.

Do not log row content — dictation data can be sensitive. Module log
lines carry only counts and id prefixes.
"""
from __future__ import annotations

import datetime
import logging
import os
import sqlite3
import threading
import time
from typing import Any

logger = logging.getLogger("whisper-api")

_EPOCH = datetime.date(1970, 1, 1)

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

# Sentinel for rows we can't attribute to a real key/user. Kept as a literal
# id string so the NOT NULL columns stay satisfied and aggregation treats it
# as its own bucket. The UI renders it as a plain label.
OPEN_MODE_ID = "(open-mode)"

# ORDER BY column names are interpolated, so they MUST come from this set.
_METRICS: frozenset[str] = frozenset(("requests", "errors", "words", "audio_s"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_hourly (
  hour     INTEGER NOT NULL,
  key_id   TEXT    NOT NULL,
  user_id  TEXT    NOT NULL,
  requests INTEGER NOT NULL DEFAULT 0,
  errors   INTEGER NOT NULL DEFAULT 0,
  words    INTEGER NOT NULL DEFAULT 0,
  audio_s  REAL    NOT NULL DEFAULT 0,
  PRIMARY KEY (hour, key_id)
);
CREATE INDEX IF NOT EXISTS idx_usage_user_hour ON usage_hourly(user_id, hour);
CREATE INDEX IF NOT EXISTS idx_usage_hour      ON usage_hourly(hour);
"""


def init_db(path: str) -> None:
    """Open (or create) the DB at `path` in WAL mode. Idempotent — call
    once on service startup before any other function. Mirrors
    transcriptions_store.init_db."""
    global _conn
    dst_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dst_dir, exist_ok=True)
    _conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL;")
    _conn.execute("PRAGMA synchronous=NORMAL;")
    _conn.execute("PRAGMA temp_store=MEMORY;")
    _conn.executescript(_SCHEMA)


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("usage_store.init_db() was not called before use.")
    return _conn


# --- time helpers ---------------------------------------------------------

def now_hour() -> int:
    return int(time.time() // 3600)


def hour_for_ts(ts: float) -> int:
    """UTC epoch-hour containing `ts`."""
    return int(float(ts) // 3600)


def epoch_day_for(ts: float) -> int:
    """Days-since-epoch of the SERVER-LOCAL calendar date containing `ts`.
    `date.fromtimestamp` resolves in local time (DST-safe — calendar dates,
    not fixed offsets). Used to roll hourly rows into server-local days."""
    return (datetime.date.fromtimestamp(ts) - _EPOCH).days


def local_day_start_hour(days_ago: int = 0) -> int:
    """UTC epoch-hour of SERVER-LOCAL midnight `days_ago` days back (0 =
    today). `datetime(date)` has no tzinfo → its .timestamp() interprets the
    naive value in local time, so this is the correct local-midnight instant
    even across DST. Used by the admin /stats + /api-keys windows."""
    d = datetime.date.today() - datetime.timedelta(days=int(days_ago))
    midnight_ts = datetime.datetime(d.year, d.month, d.day).timestamp()
    return int(midnight_ts // 3600)


def record_usage(
    *,
    key_id: str | None,
    user_id: str | None,
    audio_s: float | None,
    words: int | None,
    status: str,
    hour: int | None = None,
) -> None:
    """Bump the (hour, key_id) rollup row for one transcription. Best-effort:
    any failure is logged, never raised — a usage write must not break a
    transcription. Falsy ids fall back to the open-mode sentinel so the
    NOT NULL columns stay valid."""
    try:
        kid = key_id or OPEN_MODE_ID
        uid = user_id or OPEN_MODE_ID
        h = now_hour() if hour is None else int(hour)
        err = 0 if status == "ok" else 1
        w = int(words or 0)
        a = float(audio_s or 0.0)
        conn = _require_conn()
        with _lock:
            conn.execute(
                "INSERT INTO usage_hourly"
                " (hour, key_id, user_id, requests, errors, words, audio_s)"
                " VALUES (?, ?, ?, 1, ?, ?, ?)"
                " ON CONFLICT(hour, key_id) DO UPDATE SET"
                "  requests = requests + 1,"
                "  errors   = errors + excluded.errors,"
                "  words    = words  + excluded.words,"
                "  audio_s  = audio_s + excluded.audio_s,"
                "  user_id  = excluded.user_id",
                (h, kid, uid, err, w, a),
            )
    except Exception as e:
        logger.warning("[usage] record_usage failed: %s", e)


def _window_clause(start_hour: int | None, end_hour: int | None,
                   ) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if start_hour is not None:
        clauses.append("hour >= ?")
        params.append(int(start_hour))
    if end_hour is not None:
        clauses.append("hour <= ?")
        params.append(int(end_hour))
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def totals_by_key(
    *,
    start_hour: int | None = None,
    end_hour: int | None = None,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Summed usage per key over an optional [start_hour, end_hour] window,
    optionally restricted to one user. Rows survive key/user revocation."""
    conn = _require_conn()
    where, params = _window_clause(start_hour, end_hour)
    if user_id is not None:
        where = (where + " AND user_id = ?") if where else " WHERE user_id = ?"
        params.append(user_id)
    cur = conn.execute(
        "SELECT key_id, user_id,"
        " SUM(requests) AS requests, SUM(errors) AS errors,"
        " SUM(words) AS words, SUM(audio_s) AS audio_s"
        " FROM usage_hourly" + where +
        " GROUP BY key_id, user_id"
        " ORDER BY audio_s DESC",
        params,
    )
    return [dict(r) for r in cur.fetchall()]


def totals_by_user(
    *,
    start_hour: int | None = None,
    end_hour: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Summed usage per user over an optional window, keyed by user_id."""
    conn = _require_conn()
    where, params = _window_clause(start_hour, end_hour)
    cur = conn.execute(
        "SELECT user_id,"
        " SUM(requests) AS requests, SUM(errors) AS errors,"
        " SUM(words) AS words, SUM(audio_s) AS audio_s"
        " FROM usage_hourly" + where +
        " GROUP BY user_id",
        params,
    )
    return {r["user_id"]: dict(r) for r in cur.fetchall()}


def totals_for_user(
    user_id: str,
    *,
    start_hour: int | None = None,
    end_hour: int | None = None,
) -> dict[str, Any]:
    """Summed usage for one user over an optional [start_hour, end_hour]
    window. Returns a zeros dict when the user has no rows (uses
    idx_usage_user_hour). Backs the per-user self-usage banner on
    /quick-config: pass start_hour = hour_for_ts(<viewer local midnight>)
    for a per-viewer-local 'today'."""
    conn = _require_conn()
    where, params = _window_clause(start_hour, end_hour)
    where = (where + " AND user_id = ?") if where else " WHERE user_id = ?"
    params.append(user_id)
    row = conn.execute(
        "SELECT SUM(requests) AS requests, SUM(errors) AS errors,"
        " SUM(words) AS words, SUM(audio_s) AS audio_s"
        " FROM usage_hourly" + where,
        params,
    ).fetchone()
    return {
        "requests": int((row["requests"] if row else 0) or 0),
        "errors": int((row["errors"] if row else 0) or 0),
        "words": int((row["words"] if row else 0) or 0),
        "audio_s": float((row["audio_s"] if row else 0.0) or 0.0),
    }


def series(
    *,
    start_hour: int | None = None,
    end_hour: int | None = None,
    bucket: str = "day",
    user_id: str | None = None,
    key_id: str | None = None,
) -> list[dict[str, Any]]:
    """Time-series of summed usage, one entry per SERVER-LOCAL day (the operator
    dashboard's perspective), ascending. Hours are rolled into local days in
    Python via epoch_day_for(hour*3600) — DST-correct. `day` is days-since-epoch
    so `day*86400` is UTC midnight of that date (the client's label math).
    bucket='week' groups into 7-day blocks (day - day % 7). user_id / key_id
    None => global (all keys)."""
    conn = _require_conn()
    where, params = _window_clause(start_hour, end_hour)
    if user_id is not None:
        where = (where + " AND user_id = ?") if where else " WHERE user_id = ?"
        params.append(user_id)
    if key_id is not None:
        where = (where + " AND key_id = ?") if where else " WHERE key_id = ?"
        params.append(key_id)
    cur = conn.execute(
        "SELECT hour, requests, errors, words, audio_s"
        " FROM usage_hourly" + where,
        params,
    )
    agg: dict[int, dict[str, Any]] = {}
    for r in cur.fetchall():
        day = epoch_day_for(int(r["hour"]) * 3600)
        if bucket == "week":
            day = day - (day % 7)
        cell = agg.get(day)
        if cell is None:
            cell = agg[day] = {"day": day, "requests": 0, "errors": 0,
                               "words": 0, "audio_s": 0.0}
        cell["requests"] += int(r["requests"] or 0)
        cell["errors"] += int(r["errors"] or 0)
        cell["words"] += int(r["words"] or 0)
        cell["audio_s"] += float(r["audio_s"] or 0.0)
    return [agg[d] for d in sorted(agg)]


def leaderboard(
    *,
    start_hour: int | None = None,
    end_hour: int | None = None,
    by: str = "user",
    metric: str = "audio_s",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Top entities for a window, ranked by `metric`. by='user' groups by
    user_id; by='key' groups by key_id (carrying user_id). `metric` is
    validated against the column whitelist (it is interpolated into the
    ORDER BY)."""
    if metric not in _METRICS:
        metric = "audio_s"
    conn = _require_conn()
    where, params = _window_clause(start_hour, end_hour)
    if by == "key":
        group, cols = "key_id", "key_id, user_id"
    else:
        group, cols = "user_id", "user_id"
    cur = conn.execute(
        f"SELECT {cols},"
        " SUM(requests) AS requests, SUM(errors) AS errors,"
        " SUM(words) AS words, SUM(audio_s) AS audio_s"
        " FROM usage_hourly" + where +
        f" GROUP BY {group}"
        f" ORDER BY {metric} DESC"
        " LIMIT ?",
        params + [max(1, int(limit))],
    )
    return [dict(r) for r in cur.fetchall()]


def is_empty() -> bool:
    conn = _require_conn()
    row = conn.execute("SELECT 1 FROM usage_hourly LIMIT 1").fetchone()
    return row is None


def prune(*, retention_days: int) -> int:
    """Drop rollup rows older than the retention cutoff. retention_days <= 0
    is a no-op (the rollup is tiny — unbounded is the default)."""
    if retention_days <= 0:
        return 0
    cutoff = now_hour() - int(retention_days) * 24
    conn = _require_conn()
    with _lock:
        cur = conn.execute("DELETE FROM usage_hourly WHERE hour < ?", (cutoff,))
        return cur.rowcount or 0
