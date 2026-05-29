"""Durable per-key / per-user usage rollup.

SQLite (stdlib) in WAL mode — single-file, crash-safe, indexed. Lives at
cfg.USAGE_DB (defaults to usage.local.sqlite3 alongside config.local.json).

Why a separate store: the recent-transcriptions table
(transcriptions_store.py) is a pruned rolling window (row-cap + 30-day TTL),
so it cannot back lifetime usage totals. This store keeps a compact DAILY
ROLLUP — one row per (day, key_id) — that is never aggressively pruned, so
lifetime totals are a SUM over days and usage-over-time is a GROUP BY day.

Each /transcribe request bumps one row via an UPSERT, called from
metrics.record_transcription (which already runs inside a try/except on the
outer finally of the transcribe handler, on both success and error paths).

Day bucketing is UTC epoch-day (int(ts // 86400)) so the rollup grain is
tz-stable; the WebUI labels buckets in browser-local time.

user_id is denormalised into every row (not resolved via JOIN) so revoking
a user or key never drops their historical usage, and aggregation needs no
join to the api-keys DB.

Module-level connection: WAL mode lets us share one connection across
threads (check_same_thread=False); _lock serialises writers.

Do not log row content — this is a medical deployment. Module log lines
carry only counts and id prefixes.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import Any

logger = logging.getLogger("whisper-api")

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

# Sentinels for rows we can't attribute to a real key/user. Kept as literal
# id strings so the NOT NULL columns stay satisfied and aggregation treats
# them as their own bucket. The UI renders them as plain labels.
OPEN_MODE_ID = "(open-mode)"
BACKFILL_ID = "(backfill)"

# ORDER BY column names are interpolated, so they MUST come from this set.
_METRICS: frozenset[str] = frozenset(("requests", "errors", "words", "audio_s"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_daily (
  day      INTEGER NOT NULL,
  key_id   TEXT    NOT NULL,
  user_id  TEXT    NOT NULL,
  requests INTEGER NOT NULL DEFAULT 0,
  errors   INTEGER NOT NULL DEFAULT 0,
  words    INTEGER NOT NULL DEFAULT 0,
  audio_s  REAL    NOT NULL DEFAULT 0,
  PRIMARY KEY (day, key_id)
);
CREATE INDEX IF NOT EXISTS idx_usage_user_day ON usage_daily(user_id, day);
CREATE INDEX IF NOT EXISTS idx_usage_day      ON usage_daily(day);
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


def today_epoch_day() -> int:
    return int(time.time() // 86400)


def record_usage(
    *,
    key_id: str | None,
    user_id: str | None,
    audio_s: float | None,
    words: int | None,
    status: str,
    day: int | None = None,
) -> None:
    """Bump the (day, key_id) rollup row for one transcription. Best-effort:
    any failure is logged, never raised — a usage write must not break a
    transcription. Falsy ids fall back to the open-mode sentinel so the
    NOT NULL columns stay valid."""
    try:
        kid = key_id or OPEN_MODE_ID
        uid = user_id or OPEN_MODE_ID
        d = today_epoch_day() if day is None else int(day)
        err = 0 if status == "ok" else 1
        w = int(words or 0)
        a = float(audio_s or 0.0)
        conn = _require_conn()
        with _lock:
            conn.execute(
                "INSERT INTO usage_daily"
                " (day, key_id, user_id, requests, errors, words, audio_s)"
                " VALUES (?, ?, ?, 1, ?, ?, ?)"
                " ON CONFLICT(day, key_id) DO UPDATE SET"
                "  requests = requests + 1,"
                "  errors   = errors + excluded.errors,"
                "  words    = words  + excluded.words,"
                "  audio_s  = audio_s + excluded.audio_s,"
                "  user_id  = excluded.user_id",
                (d, kid, uid, err, w, a),
            )
    except Exception as e:
        logger.warning("[usage] record_usage failed: %s", e)


def _window_clause(start_day: int | None, end_day: int | None,
                   ) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if start_day is not None:
        clauses.append("day >= ?")
        params.append(int(start_day))
    if end_day is not None:
        clauses.append("day <= ?")
        params.append(int(end_day))
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def totals_by_key(
    *,
    start_day: int | None = None,
    end_day: int | None = None,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Summed usage per key over an optional [start_day, end_day] window,
    optionally restricted to one user. Rows survive key/user revocation."""
    conn = _require_conn()
    where, params = _window_clause(start_day, end_day)
    if user_id is not None:
        where = (where + " AND user_id = ?") if where else " WHERE user_id = ?"
        params.append(user_id)
    cur = conn.execute(
        "SELECT key_id, user_id,"
        " SUM(requests) AS requests, SUM(errors) AS errors,"
        " SUM(words) AS words, SUM(audio_s) AS audio_s"
        " FROM usage_daily" + where +
        " GROUP BY key_id, user_id"
        " ORDER BY audio_s DESC",
        params,
    )
    return [dict(r) for r in cur.fetchall()]


def totals_by_user(
    *,
    start_day: int | None = None,
    end_day: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Summed usage per user over an optional window, keyed by user_id."""
    conn = _require_conn()
    where, params = _window_clause(start_day, end_day)
    cur = conn.execute(
        "SELECT user_id,"
        " SUM(requests) AS requests, SUM(errors) AS errors,"
        " SUM(words) AS words, SUM(audio_s) AS audio_s"
        " FROM usage_daily" + where +
        " GROUP BY user_id",
        params,
    )
    return {r["user_id"]: dict(r) for r in cur.fetchall()}


def totals_for_user(
    user_id: str,
    *,
    start_day: int | None = None,
    end_day: int | None = None,
) -> dict[str, Any]:
    """Summed usage for one user over an optional [start_day, end_day] window.
    Returns a zeros dict when the user has no rows (uses idx_usage_user_day).
    Backs the per-user self-usage banner on /quick-config."""
    conn = _require_conn()
    where, params = _window_clause(start_day, end_day)
    where = (where + " AND user_id = ?") if where else " WHERE user_id = ?"
    params.append(user_id)
    row = conn.execute(
        "SELECT SUM(requests) AS requests, SUM(errors) AS errors,"
        " SUM(words) AS words, SUM(audio_s) AS audio_s"
        " FROM usage_daily" + where,
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
    start_day: int | None = None,
    end_day: int | None = None,
    bucket: str = "day",
    user_id: str | None = None,
    key_id: str | None = None,
) -> list[dict[str, Any]]:
    """Time-series of summed usage, one entry per bucket, ascending by day.
    bucket='week' groups into 7-day blocks keyed by the block's start day
    (day - day % 7). user_id / key_id None => global (all keys)."""
    conn = _require_conn()
    where, params = _window_clause(start_day, end_day)
    if user_id is not None:
        where = (where + " AND user_id = ?") if where else " WHERE user_id = ?"
        params.append(user_id)
    if key_id is not None:
        where = (where + " AND key_id = ?") if where else " WHERE key_id = ?"
        params.append(key_id)
    day_expr = "(day - (day % 7))" if bucket == "week" else "day"
    cur = conn.execute(
        f"SELECT {day_expr} AS day,"
        " SUM(requests) AS requests, SUM(errors) AS errors,"
        " SUM(words) AS words, SUM(audio_s) AS audio_s"
        " FROM usage_daily" + where +
        f" GROUP BY {day_expr}"
        " ORDER BY day ASC",
        params,
    )
    return [dict(r) for r in cur.fetchall()]


def leaderboard(
    *,
    start_day: int | None = None,
    end_day: int | None = None,
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
    where, params = _window_clause(start_day, end_day)
    if by == "key":
        group, cols = "key_id", "key_id, user_id"
    else:
        group, cols = "user_id", "user_id"
    cur = conn.execute(
        f"SELECT {cols},"
        " SUM(requests) AS requests, SUM(errors) AS errors,"
        " SUM(words) AS words, SUM(audio_s) AS audio_s"
        " FROM usage_daily" + where +
        f" GROUP BY {group}"
        f" ORDER BY {metric} DESC"
        " LIMIT ?",
        params + [max(1, int(limit))],
    )
    return [dict(r) for r in cur.fetchall()]


def is_empty() -> bool:
    conn = _require_conn()
    row = conn.execute("SELECT 1 FROM usage_daily LIMIT 1").fetchone()
    return row is None


def prune(*, retention_days: int) -> int:
    """Drop rollup rows older than the retention cutoff. retention_days <= 0
    is a no-op (the rollup is tiny — unbounded is the default)."""
    if retention_days <= 0:
        return 0
    cutoff = today_epoch_day() - int(retention_days)
    conn = _require_conn()
    with _lock:
        cur = conn.execute("DELETE FROM usage_daily WHERE day < ?", (cutoff,))
        return cur.rowcount or 0


def backfill_from_transcriptions() -> int:
    """One-time seed of usage_daily from the existing recent_transcriptions
    rows, so the feature isn't blank on first deploy. No-op when usage_daily
    already has rows (idempotent across restarts).

    recent_transcriptions has no key_id column, so backfilled rows are
    per-USER only and bucketed under key_id='(backfill)'; real per-key
    history begins at deploy. Best-effort — any failure is logged, not
    raised."""
    try:
        if not is_empty():
            return 0
        import transcriptions_store
        src = transcriptions_store._require_conn()
        rows = src.execute(
            "SELECT CAST(created_ts / 86400 AS INTEGER) AS day,"
            " user_id,"
            " COUNT(*) AS requests,"
            " SUM(CASE WHEN status = 'ok' THEN 0 ELSE 1 END) AS errors,"
            " SUM(COALESCE(words_count, 0)) AS words,"
            " SUM(COALESCE(audio_dur_s, 0)) AS audio_s"
            " FROM recent_transcriptions"
            " GROUP BY day, user_id"
        ).fetchall()
        if not rows:
            return 0
        conn = _require_conn()
        n = 0
        with _lock:
            for r in rows:
                conn.execute(
                    "INSERT INTO usage_daily"
                    " (day, key_id, user_id, requests, errors, words, audio_s)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(day, key_id) DO UPDATE SET"
                    "  requests = requests + excluded.requests,"
                    "  errors   = errors + excluded.errors,"
                    "  words    = words  + excluded.words,"
                    "  audio_s  = audio_s + excluded.audio_s",
                    (int(r["day"]), BACKFILL_ID, r["user_id"] or OPEN_MODE_ID,
                     int(r["requests"] or 0), int(r["errors"] or 0),
                     int(r["words"] or 0), float(r["audio_s"] or 0.0)),
                )
                n += 1
        logger.info("[usage] backfilled %d day/user rows from recent_transcriptions", n)
        return n
    except Exception as e:
        logger.warning("[usage] backfill failed: %s", e)
        return 0
