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
  resolved_ts     REAL
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
    global _conn
    dst_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dst_dir, exist_ok=True)
    # isolation_level=None puts pysqlite in autocommit mode; every
    # statement commits independently. _lock serialises writers (so the
    # COUNT inside _evict_to_cap sees a stable total) but does not make
    # the INSERT + DELETEs atomic vs crash. WAL gives crash-safety per
    # statement; synchronous=NORMAL is the standard WAL recommendation
    # (full durability against power loss is FULL, but NORMAL is fine
    # against process crash and ~10× faster on small writes).
    _conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL;")
    _conn.execute("PRAGMA synchronous=NORMAL;")
    _conn.executescript(_SCHEMA)
    # Idempotent migration: older DBs lack user_id. ALTER errors with
    # OperationalError ("duplicate column name") on a second startup —
    # swallow that one specifically.
    try:
        _conn.execute("ALTER TABLE reports ADD COLUMN user_id TEXT")
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            raise
    _conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_reports_user_request "
        "ON reports(user_id, request_id) WHERE user_id IS NOT NULL"
    )


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
    return d


# ---------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------

def _truncate_steps(steps: list) -> list:
    """Cap the steps list so its JSON serialization stays below
    _CAP_STEPS_JSON. Drops from the front (oldest pipeline stages first)
    so the last entries — output-wrapper + terminal-trim, which are what
    the admin actually wants to see — are preserved."""
    out: list = []
    for s in steps:
        if not isinstance(s, (list, tuple)) or len(s) < 3:
            continue
        out.append([str(s[0])[:_CAP_CORRECTION_FIELD],
                    str(s[1])[:_CAP_RAW],
                    str(s[2])[:_CAP_RAW]])
    blob = json.dumps(out, ensure_ascii=False)
    while len(blob) > _CAP_STEPS_JSON and out:
        out.pop(0)
        blob = json.dumps(out, ensure_ascii=False)
    return out


# Delegates to text_corrections so /reports and /captures share one
# definition of the chip shape. Kept here as a module-level name for the
# external callers that already import it (reports_routes.submit_report).
_clean_corrections = text_corrections.clean_corrections


def find_by_request_user(
    request_id: "str | None", user_id: "str | None",
) -> "dict[str, Any] | None":
    """Return the most recent report row keyed on (user_id, request_id),
    or None. Used by upsert_report (so re-reporting the same trace
    updates the existing row instead of stacking duplicates) and by
    reports_routes.delete_my_report_api to target the caller's own row."""
    if not request_id or not user_id:
        return None
    conn = _require_conn()
    row = conn.execute(
        "SELECT * FROM reports WHERE request_id = ? AND user_id = ?"
        " ORDER BY created_ts DESC LIMIT 1",
        (request_id, user_id),
    ).fetchone()
    return _row_to_dict(row) if row else None


# Resubmit merge reuses text_corrections.three_way_merge_corrections with an
# empty baseline — start from existing, overlay incoming on key match, dedupe
# anchorless on (wrong, correct). That helper is the single source of truth
# for chip-merge keying (anchored on (idx, idx_end), anchorless on
# (wrong, correct)) and the captures-routes group-PATCH path already uses it
# — keep the two stores in lockstep so future chip-merge tweaks land once.


def upsert_report(
    *,
    user_id: "str | None",
    request_id: "str | None",
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
) -> "tuple[str, bool]":
    """Insert a new report or update the existing one keyed on
    (user_id, request_id). Returns (report_id, was_updated). The
    `was_updated` flag is True when an existing row was merged; False
    when a fresh row was inserted.

    On update: corrections go through three_way_merge_corrections —
    keyed on (idx, idx_end) for anchored chips and on (wrong, correct)
    for anchorless ones — intended_text and user_comment overwrite
    (latest submission supersedes), and created_ts bumps to "now" so
    the row re-floats to the top of /reports.
    """
    existing = find_by_request_user(request_id, user_id)
    raw_t = (raw or "")[:_CAP_RAW]
    final_t = (final or "")[:_CAP_FINAL]
    steps_t = _truncate_steps(steps or [])
    corr_in = _clean_corrections(corrections or [])
    intended_t = (intended_text or "")[:_CAP_INTENDED]
    comment_t = (user_comment or "")[:_CAP_COMMENT]
    role_t = "admin" if reporter_role == "admin" else "user"
    now = time.time()

    conn = _require_conn()
    if existing is not None:
        merged = text_corrections.three_way_merge_corrections(
            baseline=[], edited=corr_in, current=existing.get("corrections") or [],
        )
        rid = existing["id"]
        with _lock:
            conn.execute(
                "UPDATE reports SET"
                "  created_ts = ?, trace_ts = ?, model = ?, raw = ?, final = ?,"
                "  steps_json = ?, corrections_json = ?,"
                "  intended_text = ?, user_comment = ?,"
                "  reporter_role = ?, reporter_host = ?,"
                "  status = 'open', resolved_ts = NULL"
                " WHERE id = ?",
                (
                    now, float(trace_ts or now), model, raw_t, final_t,
                    json.dumps(steps_t, ensure_ascii=False),
                    json.dumps(merged, ensure_ascii=False),
                    intended_t, comment_t, role_t, reporter_host or "",
                    rid,
                ),
            )
        logger.info("[reports] upsert-updated id=%s role=%s", rid[:8], role_t)
        return rid, True

    rid = uuid.uuid4().hex
    with _lock:
        conn.execute(
            "INSERT INTO reports ("
            " id, created_ts, trace_ts, request_id, model,"
            " raw, final, steps_json, corrections_json,"
            " intended_text, user_comment, reporter_role, reporter_host,"
            " status, admin_notes, resolved_ts,"
            " user_id"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rid, now, float(trace_ts or now), request_id, model,
                raw_t, final_t,
                json.dumps(steps_t, ensure_ascii=False),
                json.dumps(corr_in, ensure_ascii=False),
                intended_t, comment_t, role_t, reporter_host or "",
                "open", "", None, user_id,
            ),
        )
        _evict_to_cap(conn)

    logger.info("[reports] created id=%s role=%s", rid[:8], role_t)
    return rid, False


def _evict_to_cap(conn: sqlite3.Connection) -> None:
    """Enforce REPORTS_MAX: when total > cap, delete oldest closed first,
    then oldest open. Two autocommit DELETEs (connection is in
    autocommit mode); logs only the count.

    Lazy-imports config so test harnesses that monkey-patch cfg pick up
    the current value. _lock is already held by the caller, so the COUNT
    sees a stable total even though the DELETEs are not transactionally
    atomic with the caller's INSERT."""
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


def list_reports_for_request_id(request_id: str) -> list[dict[str, Any]]:
    """Return every report row sharing this `request_id`. Used by the
    report-delete cascade to compute which chips other surviving reports
    still claim — so deleting one report doesn't strip chips a sibling
    would re-add. Snapshot is taken AFTER the delete (see
    _delete_report_and_cascade) so no caller needs an "exclude" lever."""
    conn = _require_conn()
    cur = conn.execute(
        "SELECT * FROM reports WHERE request_id = ?"
        " ORDER BY created_ts DESC",
        (request_id,),
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def recent_reports_for_user(
    user_id: str, limit: int = 100,
) -> list[dict[str, Any]]:
    """Return up to `limit` of the caller's most-recent open reports as
    full row dicts (includes corrections + intended_text). Newest-
    created-first. Feeds /quick-config so the page can re-render chips
    the user previously submitted, even after a hard reload.

    Filtered by user_id so other users' reports never leak into a
    different user's /quick-config view. status='open' matches the
    badge-id query above (resolved/dismissed reports don't trigger the
    '✓ reported' badge either)."""
    if not user_id:
        return []
    conn = _require_conn()
    cur = conn.execute(
        "SELECT * FROM reports"
        " WHERE user_id = ? AND status = 'open' AND request_id IS NOT NULL"
        " ORDER BY created_ts DESC"
        " LIMIT ?",
        (user_id, int(limit)),
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


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
        n = conn.execute("DELETE FROM reports").rowcount
    # VACUUM rewrites the entire DB file and blocks other writers — run
    # it outside _lock so unrelated writes (e.g. a fresh report submit
    # arriving during the wipe) aren't stalled. Connection is autocommit,
    # no transaction needed. Skip when nothing was deleted — VACUUM on an
    # already-empty table is pure I/O for zero space recovery.
    if n > 0:
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
