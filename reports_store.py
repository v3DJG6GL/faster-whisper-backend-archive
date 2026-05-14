"""Durable store for user-submitted transcription error reports.

SQLite (stdlib) in WAL mode — single-file, crash-safe, indexed. Lives at
cfg.REPORTS_DB (defaults to reports.local.sqlite3 alongside config.local.json).

This is the only structured, query-able, end-user-editable durable PHI
surface produced by this app. The rotating text logger is also durable
PHI but format-locked; reports are user-curated for triage. Plaintext
on disk; whole-disk encryption is the deployment's responsibility.

Do not log report content. The module's INFO/WARNING lines carry only
counts and report-id prefixes for forensics.

Module-level connection: SQLite's WAL mode lets us share one connection
across threads (with `check_same_thread=False`). All public functions
acquire `_lock` for writes; reads are concurrent and safe.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from typing import Any

import text_corrections

logger = logging.getLogger("whisper-api")

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_db_path: str | None = None

# Field caps — applied server-side before insert. raw/final get the
# generous cap because they're audio-driven and can legitimately be long
# (a 5-min dictation block); intended/comment/notes are human-typed.
_CAP_RAW = 50_000
_CAP_FINAL = 50_000
_CAP_STEPS_JSON = 200_000
_CAP_INTENDED = 2_000
_CAP_COMMENT = 4_000
_CAP_ADMIN_NOTES = 8_000
_CAP_CORRECTIONS = text_corrections.CAP_CORRECTIONS
_CAP_CORRECTION_FIELD = text_corrections.CAP_CORRECTION_FIELD

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
  id              TEXT PRIMARY KEY,
  created_ts      REAL NOT NULL,
  trace_ts        REAL NOT NULL,
  request_id      TEXT,
  model           TEXT NOT NULL,
  raw             TEXT NOT NULL,
  final           TEXT NOT NULL,
  steps_json      TEXT NOT NULL,
  corrections_json TEXT NOT NULL DEFAULT '[]',
  intended_text   TEXT NOT NULL DEFAULT '',
  user_comment    TEXT NOT NULL DEFAULT '',
  reporter_role   TEXT NOT NULL,
  reporter_host   TEXT NOT NULL DEFAULT '',
  status          TEXT NOT NULL DEFAULT 'open',
  admin_notes     TEXT NOT NULL DEFAULT '',
  resolved_ts     REAL,
  snapshot_trusted_client INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_reports_created    ON reports(created_ts DESC);
CREATE INDEX IF NOT EXISTS idx_reports_status     ON reports(status);
CREATE INDEX IF NOT EXISTS idx_reports_request_id ON reports(request_id);
"""

_VALID_STATUS = frozenset({"open", "resolved", "dismissed"})


def init_db(path: str) -> None:
    """Open (or create) the report DB at `path` in WAL mode. Idempotent:
    safe to call on every startup; the schema-CREATE statements use
    IF NOT EXISTS. Call before any other function in this module."""
    global _conn, _db_path
    _db_path = path
    dst_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dst_dir, exist_ok=True)
    # isolation_level=None puts pysqlite in autocommit mode; we issue
    # explicit BEGIN/COMMIT only inside the write helpers. WAL gives us
    # crash-safety; synchronous=NORMAL is the standard WAL recommendation
    # (full durability against power loss is FULL, but NORMAL is fine
    # against process crash and ~10× faster on small writes).
    _conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL;")
    _conn.execute("PRAGMA synchronous=NORMAL;")
    _conn.executescript(_SCHEMA)


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError(
            "reports_store.init_db() was not called before use."
        )
    return _conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Materialize a row, decoding the JSON-bearing columns. Returns
    plain Python types ready for JSON serialization on the wire."""
    d = dict(row)
    try:
        d["steps"] = json.loads(d.pop("steps_json", "[]") or "[]")
    except (TypeError, ValueError):
        d["steps"] = []
    try:
        d["corrections"] = json.loads(d.pop("corrections_json", "[]") or "[]")
    except (TypeError, ValueError):
        d["corrections"] = []
    d["snapshot_trusted_client"] = bool(d.get("snapshot_trusted_client"))
    return d


# ---------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------

def _truncate_steps(steps: list) -> list:
    """Cap the steps list so its JSON serialization stays below
    _CAP_STEPS_JSON. Drops from the tail (oldest pipeline stages first;
    the user-visible final transformation is the last element). Returns
    the original list if already small enough."""
    out: list = []
    for s in steps:
        if not isinstance(s, (list, tuple)) or len(s) < 3:
            continue
        out.append([str(s[0])[:_CAP_CORRECTION_FIELD],
                    str(s[1])[:_CAP_RAW],
                    str(s[2])[:_CAP_RAW]])
    # If the serialized blob is still too big, drop from the END (oldest
    # stages stay; we keep the last few which include the actually-
    # interesting output-wrapper / terminal-trim steps).
    blob = json.dumps(out, ensure_ascii=False)
    while len(blob) > _CAP_STEPS_JSON and out:
        out.pop()
        blob = json.dumps(out, ensure_ascii=False)
    return out


# Delegates to text_corrections so /reports and /captures share one
# definition of the chip shape. Kept here as a module-level name for the
# external callers that already import it (reports_routes.submit_report).
_clean_corrections = text_corrections.clean_corrections


def create_report(
    *,
    request_id: str | None,
    trace_ts: float,
    model: str,
    raw: str,
    final: str,
    steps: list,
    corrections: list,
    intended_text: str,
    user_comment: str,
    reporter_role: str,
    reporter_host: str,
) -> str:
    """Insert a new report row and return its id. Caller has already
    cleaned/trimmed inputs at the route layer; this is the last-line
    enforcer for length caps and the soft-cap eviction policy."""
    rid = uuid.uuid4().hex
    now = time.time()
    raw_t = (raw or "")[:_CAP_RAW]
    final_t = (final or "")[:_CAP_FINAL]
    steps_t = _truncate_steps(steps or [])
    corr_t = _clean_corrections(corrections or [])
    intended_t = (intended_text or "")[:_CAP_INTENDED]
    comment_t = (user_comment or "")[:_CAP_COMMENT]
    role_t = "admin" if reporter_role == "admin" else "user"

    conn = _require_conn()
    with _lock:
        conn.execute(
            "INSERT INTO reports ("
            " id, created_ts, trace_ts, request_id, model,"
            " raw, final, steps_json, corrections_json,"
            " intended_text, user_comment, reporter_role, reporter_host,"
            " status, admin_notes, resolved_ts, snapshot_trusted_client"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rid, now, float(trace_ts or now), request_id, model,
                raw_t, final_t,
                json.dumps(steps_t, ensure_ascii=False),
                json.dumps(corr_t, ensure_ascii=False),
                intended_t, comment_t, role_t, reporter_host or "",
                "open", "", None, 1,
            ),
        )
        _evict_to_cap(conn)

    logger.info("[reports] created id=%s role=%s", rid[:8], role_t)
    return rid


def _evict_to_cap(conn: sqlite3.Connection) -> None:
    """Enforce REPORTS_MAX: when total > cap, delete oldest closed first,
    then oldest open. Single transaction; logs only the count.

    Lazy-imports config so test harnesses that monkey-patch cfg pick up
    the current value. _lock is already held by the caller."""
    try:
        import config as cfg
        cap = int(getattr(cfg, "REPORTS_MAX", 1000))
    except Exception:
        cap = 1000
    if cap < 1:
        return
    row = conn.execute("SELECT COUNT(*) FROM reports").fetchone()
    total = int(row[0]) if row else 0
    excess = total - cap
    if excess <= 0:
        return
    # Closed first: status != 'open', oldest by created_ts.
    closed = conn.execute(
        "DELETE FROM reports WHERE id IN ("
        "  SELECT id FROM reports WHERE status != 'open'"
        "  ORDER BY created_ts ASC LIMIT ?"
        ")",
        (excess,),
    ).rowcount
    remaining = excess - max(0, closed)
    if remaining > 0:
        open_deleted = conn.execute(
            "DELETE FROM reports WHERE id IN ("
            "  SELECT id FROM reports WHERE status = 'open'"
            "  ORDER BY created_ts ASC LIMIT ?"
            ")",
            (remaining,),
        ).rowcount
    else:
        open_deleted = 0
    if closed or open_deleted:
        logger.info(
            "[reports] evicted to cap: %d closed, %d open (cap=%d)",
            closed, open_deleted, cap,
        )


# ---------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------

def list_reports() -> list[dict[str, Any]]:
    """Return all reports, newest first. Client filters/searches in-page;
    the soft cap keeps the row count under what a browser can render."""
    conn = _require_conn()
    cur = conn.execute(
        "SELECT * FROM reports ORDER BY created_ts DESC"
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def get_report(rid: str) -> dict[str, Any] | None:
    conn = _require_conn()
    row = conn.execute("SELECT * FROM reports WHERE id = ?", (rid,)).fetchone()
    return _row_to_dict(row) if row else None


def recent_reported_request_ids(limit: int = 100) -> list[str]:
    """Return up to `limit` distinct request_ids that have at least one
    report. Newest-report-first ordering. Feeds the server-authoritative
    '✓ reported' badge in /quick-config."""
    conn = _require_conn()
    cur = conn.execute(
        "SELECT request_id, MAX(created_ts) AS ts FROM reports"
        " WHERE request_id IS NOT NULL"
        " GROUP BY request_id"
        " ORDER BY ts DESC"
        " LIMIT ?",
        (int(limit),),
    )
    return [r["request_id"] for r in cur.fetchall() if r["request_id"]]


def counts_by_status() -> dict[str, int]:
    """Quick summary for the /reports page toolbar."""
    conn = _require_conn()
    out = {"open": 0, "resolved": 0, "dismissed": 0}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM reports GROUP BY status"
    ):
        if row["status"] in out:
            out[row["status"]] = int(row["n"])
    return out


# ---------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------

def update_report(rid: str, patch: dict[str, Any]) -> dict[str, Any] | None:
    """Apply a partial update. Allowed fields: status, admin_notes.
    Returns the updated row dict or None if not found. Unknown fields
    are ignored silently (the route layer validates the shape; this is
    the last-line guard)."""
    if not patch:
        return get_report(rid)
    sets: list[str] = []
    params: list[Any] = []
    if "status" in patch:
        new_status = str(patch["status"] or "open")
        if new_status not in _VALID_STATUS:
            raise ValueError(f"invalid status: {new_status!r}")
        sets.append("status = ?")
        params.append(new_status)
        if new_status == "open":
            sets.append("resolved_ts = NULL")
        else:
            sets.append("resolved_ts = ?")
            params.append(time.time())
    if "admin_notes" in patch:
        notes = str(patch["admin_notes"] or "")[:_CAP_ADMIN_NOTES]
        sets.append("admin_notes = ?")
        params.append(notes)
    if not sets:
        return get_report(rid)
    params.append(rid)
    conn = _require_conn()
    with _lock:
        cur = conn.execute(
            f"UPDATE reports SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        if cur.rowcount == 0:
            return None
    return get_report(rid)


# ---------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------

def delete_report(rid: str) -> bool:
    """Delete a single report. Returns True if a row was removed."""
    conn = _require_conn()
    with _lock:
        cur = conn.execute("DELETE FROM reports WHERE id = ?", (rid,))
        deleted = cur.rowcount > 0
    if deleted:
        logger.info("[reports] deleted id=%s", rid[:8])
    return deleted


def clear_all(reporter_host: str = "") -> int:
    """Wipe the entire table. Returns the count deleted. WARNING-logs
    the count + caller host for audit."""
    conn = _require_conn()
    with _lock:
        row = conn.execute("SELECT COUNT(*) FROM reports").fetchone()
        n = int(row[0]) if row else 0
        conn.execute("DELETE FROM reports")
        # VACUUM is safe outside an explicit transaction because the
        # connection is in autocommit mode. Reclaims pages so the file
        # shrinks back after a bulk delete.
        conn.execute("VACUUM")
    logger.warning(
        "[reports] admin from %s cleared %d reports",
        reporter_host or "<unknown>", n,
    )
    return n


# ---------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------

def sweep_retention() -> int:
    """Delete rows older than cfg.REPORTS_RETENTION_DAYS. Returns count
    deleted (0 when retention is disabled or nothing's old enough).
    Lazy-imports cfg so admin /config edits take effect on next sweep."""
    try:
        import config as cfg
        days = int(getattr(cfg, "REPORTS_RETENTION_DAYS", 0))
    except Exception:
        return 0
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    conn = _require_conn()
    with _lock:
        cur = conn.execute(
            "DELETE FROM reports WHERE created_ts < ?", (cutoff,)
        )
        n = cur.rowcount
    if n > 0:
        logger.warning(
            "[reports] retention sweep deleted %d rows older than %d days",
            n, days,
        )
    return n
