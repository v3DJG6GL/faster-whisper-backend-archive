"""Durable store for recent /transcribe traces.

SQLite (stdlib) in WAL mode — single-file, crash-safe, indexed. Lives at
cfg.RECENT_TRANSCRIPTIONS_DB (defaults to recent_transcriptions.local.sqlite3
alongside config.local.json).

Replaces the legacy in-memory ring buffers (`quick_config_state.recent_traces`
and `metrics.recent_tx`) so the /quick-config trace panel and /stats
dashboard "Recent transcriptions" widget survive service restarts and
scale beyond a 20-row cap.

Two upsert call sites per /transcribe request, both keyed by request_id:

  1. `record_trace(...)` (success path, inside the inner try) writes the
     rich payload: raw_text, final_text, steps_json, tokens_json,
     bigrams_json, model, language, user_id, username, created_ts.

  2. `record_timing(...)` (outer finally, always) writes proc_dur_s,
     audio_dur_s, words_count, status. On the error path it inserts a
     minimal row (no raw/final/steps) so /stats still counts the request.

Lazy pruning every `cfg.RECENT_TRANSCRIPTIONS_PRUNE_EVERY` inserts: a
single DELETE statement enforces both the row cap and the TTL. A cap of
0 disables the count clause; a TTL of 0 disables the age clause.

Module-level connection: SQLite's WAL mode lets us share one connection
across threads (`check_same_thread=False`). `_lock` serialises writers.

Do not log row content — entries carry literal patient dictation on a
medical deployment. Module log lines carry only counts and request_id
prefixes.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any

logger = logging.getLogger("whisper-api")

_lock = threading.RLock()  # reentrant: record_trace/record_timing hold the lock and may call prune() which re-acquires it
_conn: sqlite3.Connection | None = None
_insert_counter = 0

# Generous field caps — raw/final are audio-driven and can be long
# (a 5-minute dictation block). Steps JSON is bounded by the existing
# reports_store cap so a trace pair (report + recent) doesn't surprise
# anyone with mismatched size limits.
_CAP_RAW = 50_000
_CAP_FINAL = 50_000
_CAP_STEPS_JSON = 200_000
_CAP_TOKEN_FIELD = 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recent_transcriptions (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id    TEXT NOT NULL UNIQUE,
  created_ts    REAL NOT NULL,
  user_id       TEXT,
  username      TEXT,
  model         TEXT NOT NULL,
  language      TEXT,
  source        TEXT NOT NULL DEFAULT 'file',
  status        TEXT NOT NULL DEFAULT 'ok',
  audio_dur_s   REAL,
  proc_dur_s    REAL,
  words_count   INTEGER,
  raw_text      TEXT,
  final_text    TEXT,
  steps_json    TEXT,
  tokens_json   TEXT,
  bigrams_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_rt_created      ON recent_transcriptions(created_ts DESC);
CREATE INDEX IF NOT EXISTS idx_rt_user_created ON recent_transcriptions(user_id, created_ts DESC);
"""


def init_db(path: str) -> None:
    """Open (or create) the DB at `path` in WAL mode. Idempotent — call
    once on service startup before any other function in this module.
    Mirrors reports_store.init_db / captures_store.init pattern."""
    global _conn
    dst_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dst_dir, exist_ok=True)
    _conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL;")
    _conn.execute("PRAGMA synchronous=NORMAL;")
    _conn.execute("PRAGMA temp_store=MEMORY;")
    _conn.executescript(_SCHEMA)
    # Migrate pre-existing DBs (created before the `source` column): add it with
    # the 'file' default so old rows read as batch/file-upload transcriptions.
    cols = {r["name"] for r in _conn.execute("PRAGMA table_info(recent_transcriptions)")}
    if "source" not in cols:
        _conn.execute(
            "ALTER TABLE recent_transcriptions ADD COLUMN source TEXT NOT NULL DEFAULT 'file'")


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError(
            "transcriptions_store.init_db() was not called before use."
        )
    return _conn


def _truncate_steps(steps: list) -> list:
    """Same shape as reports_store._truncate_steps but with this module's
    own caps so the two stores can evolve their limits independently.
    Drops oldest pipeline stages first when over-cap; preserves the
    output-wrapper + terminal-trim trailers (the part admins care about)."""
    out: list = []
    for s in steps:
        if not isinstance(s, (list, tuple)) or len(s) < 3:
            continue
        out.append([str(s[0])[:512],
                    str(s[1])[:_CAP_RAW],
                    str(s[2])[:_CAP_RAW]])
    blob = json.dumps(out, ensure_ascii=False)
    while len(blob) > _CAP_STEPS_JSON and out:
        out.pop(0)
        blob = json.dumps(out, ensure_ascii=False)
    return out


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Materialize a row to the wire shape expected by /quick-config and
    /stats consumers. Decodes the JSON-bearing columns; computes the
    derived `rtf` (audio_dur_s / proc_dur_s) so callers don't repeat the
    formula. Keeps both `ts` (legacy /quick-config field name) and
    `created_ts` (the actual column) so existing JS keeps working."""
    d = dict(row)
    for col, key in (
        ("steps_json", "steps"),
        ("tokens_json", "tokens"),
        ("bigrams_json", "bigrams"),
    ):
        try:
            d[key] = json.loads(d.pop(col, "[]") or "[]")
        except (TypeError, ValueError):
            d[key] = []
    audio = d.get("audio_dur_s")
    proc = d.get("proc_dur_s")
    d["rtf"] = round(audio / proc, 2) if (audio and proc and proc > 0) else None
    d["ts"] = d.get("created_ts")
    # Legacy /stats widget keys — kept for the existing dashboard JS.
    d["audio_dur"] = audio
    d["proc_dur"] = proc
    d["words"] = d.get("words_count")
    # /quick-config renderTrace + _buildReportForm read entry.raw / entry.final;
    # the live SSE event double-keys both names, hydrated rows must match.
    # Error-path rows (record_timing without record_trace) leave the text
    # columns NULL — coerce to '' so the JS string ops never see None.
    d["raw"] = d.get("raw_text") or ""
    d["final"] = d.get("final_text") or ""
    d["username"] = d.get("username") or ""
    d["language"] = d.get("language") or ""
    d["source"] = d.get("source") or "file"
    return d


def _lazy_prune_if_due(prune_every: int, max_rows: int, ttl_days: float) -> None:
    """Bumps an in-process counter; calls prune() every Nth insert."""
    global _insert_counter
    _insert_counter += 1
    if prune_every <= 0:
        return
    if _insert_counter % prune_every != 0:
        return
    try:
        prune(max_rows=max_rows, ttl_days=ttl_days)
    except Exception as e:
        logger.warning("[recent-tx] prune failed: %s", e)


def record_trace(
    *,
    request_id: str,
    model: str,
    raw: str,
    final: str,
    steps: list | None = None,
    tokens: list | None = None,
    bigrams: list | None = None,
    language: str | None = None,
    source: str = "file",
    user_id: str | None = None,
    username: str | None = None,
    created_ts: float | None = None,
    prune_every: int = 50,
    max_rows: int = 500,
    ttl_days: float = 30.0,
) -> None:
    """Insert or update the rich half of a /transcribe row. Called on the
    success path (inside the inner try in main.py). UPSERTs by request_id
    so a later record_timing() call merges the timing fields in.

    All text fields are silently truncated at module caps; over-large
    `steps` lists shed leading entries (terminal trim + wrapper are the
    rows the admin actually wants — see _truncate_steps)."""
    if not request_id:
        return
    raw_s = (raw or "")[:_CAP_RAW]
    final_s = (final or "")[:_CAP_FINAL]
    steps_blob = json.dumps(_truncate_steps(steps or []), ensure_ascii=False)
    tokens_blob = json.dumps([str(t)[:_CAP_TOKEN_FIELD] for t in (tokens or [])],
                             ensure_ascii=False)
    bigrams_blob = json.dumps([str(b)[:_CAP_TOKEN_FIELD * 2] for b in (bigrams or [])],
                              ensure_ascii=False)
    ts = float(created_ts) if created_ts else time.time()
    conn = _require_conn()
    with _lock:
        conn.execute(
            "INSERT INTO recent_transcriptions ("
            "  request_id, created_ts, user_id, username, model, language, source,"
            "  status, raw_text, final_text,"
            "  steps_json, tokens_json, bigrams_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, 'ok', ?, ?, ?, ?, ?)"
            " ON CONFLICT(request_id) DO UPDATE SET"
            "  model        = excluded.model,"
            "  language     = COALESCE(excluded.language, recent_transcriptions.language),"
            "  source       = excluded.source,"
            "  user_id      = COALESCE(excluded.user_id, recent_transcriptions.user_id),"
            "  username     = COALESCE(excluded.username, recent_transcriptions.username),"
            "  raw_text     = excluded.raw_text,"
            "  final_text   = excluded.final_text,"
            "  steps_json   = excluded.steps_json,"
            "  tokens_json  = excluded.tokens_json,"
            "  bigrams_json = excluded.bigrams_json",
            (request_id, ts, user_id, username, model, language, source or "file",
             raw_s, final_s, steps_blob, tokens_blob, bigrams_blob),
        )
        _lazy_prune_if_due(prune_every, max_rows, ttl_days)


def record_timing(
    *,
    request_id: str,
    model: str,
    audio_dur_s: float | None,
    proc_dur_s: float,
    status: str,
    words_count: int,
    user_id: str | None = None,
    created_ts: float | None = None,
    prune_every: int = 50,
    max_rows: int = 500,
    ttl_days: float = 30.0,
) -> None:
    """Insert or update the timing half. Called in the outer finally so
    it runs on BOTH success (after record_trace) and error paths. UPSERT
    by request_id: on success it patches timing fields onto the row
    record_trace already wrote; on error it inserts a minimal row with
    no raw/final/steps."""
    if not request_id:
        return
    ts = float(created_ts) if created_ts else time.time()
    conn = _require_conn()
    with _lock:
        conn.execute(
            "INSERT INTO recent_transcriptions ("
            "  request_id, created_ts, user_id, model, status,"
            "  audio_dur_s, proc_dur_s, words_count"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(request_id) DO UPDATE SET"
            "  status      = excluded.status,"
            "  audio_dur_s = excluded.audio_dur_s,"
            "  proc_dur_s  = excluded.proc_dur_s,"
            "  words_count = excluded.words_count,"
            "  model       = excluded.model",
            (request_id, ts, user_id, model, status,
             audio_dur_s, proc_dur_s, words_count),
        )
        _lazy_prune_if_due(prune_every, max_rows, ttl_days)


def list_recent(
    *,
    before_ts: float | None = None,
    limit: int = 100,
    user_id_filter: str | None = None,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Return up to `limit` rows newer than `before_ts` (or the newest
    `limit` rows when before_ts is None / 0), newest-first. When
    user_id_filter is set, returns only rows for that user — used by
    /quick-config scope='own' to keep one user's traces out of another
    user's view.

    When `query` is set, only rows whose raw_text OR final_text contain
    the substring are returned (case-insensitive substring match). This
    composes with both the before_ts cursor and user_id_filter, so the
    "Load older" pagination walks back through matches only. Note: SQLite
    LIKE is ASCII case-insensitive — non-ASCII (e.g. German umlauts) is
    matched case-sensitively; acceptable for this free-text search."""
    conn = _require_conn()
    where: list[str] = []
    params: list[Any] = []
    if before_ts and before_ts > 0:
        where.append("created_ts < ?")
        params.append(float(before_ts))
    if user_id_filter is not None:
        where.append("user_id = ?")
        params.append(user_id_filter)
    if query:
        # Escape LIKE wildcards so a literal % or _ in the search text is
        # matched literally rather than as a wildcard.
        needle = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = f"%{needle}%"
        where.append(
            "(raw_text LIKE ? ESCAPE '\\' OR final_text LIKE ? ESCAPE '\\')"
        )
        params.append(like)
        params.append(like)
    sql = "SELECT * FROM recent_transcriptions"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_ts DESC LIMIT ?"
    params.append(max(1, int(limit)))
    cur = conn.execute(sql, params)
    return [_row_to_dict(r) for r in cur.fetchall()]


def count() -> int:
    conn = _require_conn()
    row = conn.execute("SELECT COUNT(*) AS n FROM recent_transcriptions").fetchone()
    return int(row["n"]) if row else 0


def prune(*, max_rows: int, ttl_days: float) -> int:
    """Drop rows older than the TTL cutoff AND rows beyond the count cap.
    A max_rows or ttl_days of 0 disables that clause; both 0 makes prune
    a no-op."""
    if max_rows <= 0 and ttl_days <= 0:
        return 0
    conn = _require_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if max_rows > 0:
        clauses.append(
            "id NOT IN (SELECT id FROM recent_transcriptions"
            " ORDER BY created_ts DESC LIMIT ?)"
        )
        params.append(int(max_rows))
    if ttl_days > 0:
        clauses.append("created_ts < ?")
        params.append(time.time() - float(ttl_days) * 86400.0)
    sql = "DELETE FROM recent_transcriptions WHERE " + " OR ".join(clauses)
    with _lock:
        cur = conn.execute(sql, params)
        return cur.rowcount or 0


def clear_all() -> int:
    """Wipe every row. Returns the count deleted. Admin-triggered."""
    conn = _require_conn()
    with _lock:
        cur = conn.execute("DELETE FROM recent_transcriptions")
        return cur.rowcount or 0
