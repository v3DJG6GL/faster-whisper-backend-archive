"""Durable store for capture groups — packed ≤28 s training samples
built from 2+ consecutive same-speaker captures.

Lives in the same SQLite DB as `captures` (re-uses the connection from
`captures_store`). The merged WAV files sit under `CAPTURES_DIR/groups/`
with the same 4-char fanout as individual captures.

The schema is structurally minimal: one `capture_samples` row per merged
sample + a `sample_id`/`sample_order` FK on the existing captures table
(added by captures_store's migration). Dissolving a group deletes the
row + the merged WAV; members get their `sample_id` NULL'd out and
return to the flat list.

Member-content drift is recorded via per-member PCM-content hashes
stored in `member_hashes_json` at merge time; downstream consumers
compare against the snapshot when they need to decide whether the
merged WAV is still authoritative.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from typing import Any

logger = logging.getLogger("whisper-api")

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_groups_audio_dir: str | None = None    # e.g. CAPTURES_DIR/groups

# Schema init runs in three phases — the same pattern documented at
# captures_store.py:64-76:
#
#   1. _SCHEMA_CORE     — CREATE TABLE IF NOT EXISTS + indexes that only
#                         reference columns present on the very first
#                         shipped version of the table.
#   2. _MIGRATIONS      — idempotent stmts for upgrading older DBs:
#                         ALTER ADD COLUMN for columns added after the
#                         first ship, ALTER DROP COLUMN for the legacy
#                         corrections_json / corrections_migrated_at
#                         cache, plus a one-shot UPDATE that normalises
#                         transcript_join_strategy='newline' rows to
#                         'space'. Each stmt runs in its own try/except
#                         so fresh DBs swallow "duplicate column …" /
#                         "no such column …" and keep going.
#   3. _SCHEMA_POST_MIGRATIONS — indexes that reference columns added
#                         by phase 2. If we put `CREATE INDEX … ON
#                         capture_samples(status)` in phase 1 alongside
#                         the CREATE TABLE, then on a DB upgraded from
#                         before `status` existed the CREATE TABLE is a
#                         no-op, the CREATE INDEX raises "no such
#                         column: status", and `executescript` aborts —
#                         which skips phase 2 entirely and leaves the
#                         store unable to read its own rows.
_SCHEMA_CORE = """
CREATE TABLE IF NOT EXISTS capture_samples (
  id                          TEXT PRIMARY KEY,
  user_id                     TEXT NOT NULL,
  created_ts                  REAL NOT NULL,
  merged_wav_relpath          TEXT NOT NULL,
  merged_duration_ms          INTEGER NOT NULL,
  transcript                  TEXT NOT NULL,
  transcript_join_strategy    TEXT NOT NULL DEFAULT 'space',
  member_hashes_json          TEXT NOT NULL,
  inter_segment_silence_ms    INTEGER NOT NULL DEFAULT 300,
  is_stale                    INTEGER NOT NULL DEFAULT 0,
  is_locked                   INTEGER NOT NULL DEFAULT 0,
  status                      TEXT NOT NULL DEFAULT 'new',
  admin_notes                 TEXT NOT NULL DEFAULT '',
  language                    TEXT,
  merged_lead_trim_ms         INTEGER NOT NULL DEFAULT 0,
  merged_trail_trim_ms        INTEGER NOT NULL DEFAULT 0,
  member_trims_json           TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_capture_samples_user
  ON capture_samples(user_id, created_ts DESC);
"""

_MIGRATIONS: tuple[str, ...] = (
    # Old DBs still carry these two columns from the one-time-projection
    # cache design. Drop them idempotently — fresh DBs see "no such
    # column" and swallow it; upgraded DBs succeed and reclaim the space.
    "ALTER TABLE capture_samples DROP COLUMN corrections_json",
    "ALTER TABLE capture_samples DROP COLUMN corrections_migrated_at",
    # Live columns that may need adding on older DBs that pre-date them.
    "ALTER TABLE capture_samples ADD COLUMN status TEXT NOT NULL DEFAULT 'new'",
    "ALTER TABLE capture_samples ADD COLUMN admin_notes TEXT NOT NULL DEFAULT ''",
    # `language` per-group (BCP-47-ish, e.g. "de"). Whisper-detected at
    # the first member; emitted in the export manifest so fine-tune
    # loaders force the right language token.
    "ALTER TABLE capture_samples ADD COLUMN language TEXT",
    # Drop the `newline` join strategy: Whisper never emits literal `\n`
    # in continuous speech, so training samples that join members with
    # \n confuse the model. Normalise any historical rows to 'space'.
    "UPDATE capture_samples SET transcript_join_strategy='space' "
    "WHERE transcript_join_strategy='newline'",
    # VAD-trim offsets on the merged WAV. NOT NULL DEFAULT 0 so the
    # `_build_merged_words` math can rely on a numeric value without
    # a COALESCE — un-trimmed groups simply contribute 0.
    "ALTER TABLE capture_samples ADD COLUMN merged_lead_trim_ms "
    "INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE capture_samples ADD COLUMN merged_trail_trim_ms "
    "INTEGER NOT NULL DEFAULT 0",
    # Per-member silence-trim map: {member_id: {lead_ms, new_duration_ms,
    # segments:[[orig_start_ms, orig_end_ms, new_start_ms], ...]}}. Populated
    # when a group is created/re-merged under per-member trimming; empty '{}'
    # for legacy groups, which keep using merged_lead_trim_ms instead.
    "ALTER TABLE capture_samples ADD COLUMN member_trims_json "
    "TEXT NOT NULL DEFAULT '{}'",
)

_SCHEMA_POST_MIGRATIONS = """
CREATE INDEX IF NOT EXISTS idx_capture_samples_status
  ON capture_samples(status);
"""

_VALID_STATUS = frozenset({"new", "reviewed", "ready", "dismissed"})
_CAP_ADMIN_NOTES = 8000


# ---------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------

def init(conn: sqlite3.Connection, captures_audio_root: str) -> None:
    """Reuse the captures DB connection (single SQLite file). The audio
    root mirrors captures_store's fanout but under a `groups/` subtree
    so a directory listing distinguishes singles from packed groups."""
    global _conn, _groups_audio_dir
    _conn = conn
    _groups_audio_dir = os.path.join(captures_audio_root, "groups")
    os.makedirs(_groups_audio_dir, exist_ok=True)
    # One-shot table rename capture_groups → capture_samples (group→sample
    # terminology migration). MUST run before CREATE TABLE so an existing
    # table is renamed in place (data preserved) rather than shadowed by a
    # fresh empty capture_samples. Idempotent: a no-op once renamed.
    try:
        _tables = {
            r[0] for r in _conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "capture_groups" in _tables and "capture_samples" not in _tables:
            _conn.execute("ALTER TABLE capture_groups RENAME TO capture_samples")
    except sqlite3.OperationalError:
        pass
    _conn.executescript(_SCHEMA_CORE)
    for stmt in _MIGRATIONS:
        try:
            _conn.execute(stmt)
        except sqlite3.OperationalError:
            # Column already present (fresh DB or migrated previously).
            pass
    _conn.executescript(_SCHEMA_POST_MIGRATIONS)
    # One-time backfill of group `language` from the first member with
    # a populated language. Rows with `language` already set are left
    # alone. Cheap on small stores; bounded SELECT-per-group on large.
    try:
        # One-shot correlated UPDATE: for each group missing a language,
        # pull the first non-empty language from its member captures in
        # sample_order. After this lands once on a deployed DB the WHERE
        # clause matches zero rows and the statement is a fast no-op.
        cur = _conn.execute(
            "UPDATE capture_samples SET language = ("
            "  SELECT language FROM captures"
            "   WHERE sample_id = capture_samples.id"
            "     AND language IS NOT NULL AND language != ''"
            "   ORDER BY sample_order ASC LIMIT 1"
            ") WHERE (language IS NULL OR language = '')"
            "   AND EXISTS ("
            "     SELECT 1 FROM captures"
            "      WHERE sample_id = capture_samples.id"
            "        AND language IS NOT NULL AND language != ''"
            "   )"
        )
        if cur.rowcount:
            logger.info(
                "[groups] language-backfill set %d groups", cur.rowcount,
            )
    except sqlite3.OperationalError as e:
        # Schema not as expected (e.g. very old DB pre-migration order);
        # log and continue — backfill is best-effort.
        logger.warning("[groups] language-backfill skipped: %s", e)


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("capture_samples_store.init() was not called before use.")
    return _conn


def _require_audio_root() -> str:
    if _groups_audio_dir is None:
        raise RuntimeError("capture_samples_store.init() was not called before use.")
    return _groups_audio_dir


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------

def _relpath_for(sid: str) -> str:
    """`groups/<g0g1>/<g2g3>/<sid>.wav` — mirrors the captures fanout."""
    return os.path.join("groups", sid[0:2], sid[2:4], f"{sid}.wav")


def abs_path_for(relpath: str) -> str:
    """Resolve a relpath (the `groups/...` form) to an absolute path
    rooted at CAPTURES_DIR. Path-traversal defense — anything that
    escapes returns ValueError."""
    # Imported lazily to avoid an import cycle at module load.
    import captures_store
    root = os.path.abspath(captures_store._require_audio_dir())
    abs_p = os.path.abspath(os.path.join(root, relpath))
    try:
        common = os.path.commonpath([abs_p, root])
    except ValueError:
        raise ValueError("group audio path escapes captures dir")
    if common != root:
        raise ValueError("group audio path escapes captures dir")
    return abs_p


# ---------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    # `corrections` is derived from current member chips on every read
    # (see captures_routes._enrich_sample / list_samples_api). The DB row
    # has no chip storage of its own — single source of truth lives on
    # the member captures.
    return {
        "id":                          row["id"],
        "user_id":                     row["user_id"],
        "created_ts":                  float(row["created_ts"]),
        "merged_wav_relpath":          row["merged_wav_relpath"],
        "merged_duration_ms":          int(row["merged_duration_ms"]),
        "transcript":                  row["transcript"] or "",
        "transcript_join_strategy":    row["transcript_join_strategy"] or "space",
        "member_hashes":               json.loads(row["member_hashes_json"] or "{}"),
        "inter_segment_silence_ms":    int(row["inter_segment_silence_ms"]),
        "is_stale":                    bool(row["is_stale"]),
        "is_locked":                   bool(row["is_locked"]),
        "status":                      row["status"] or "new",
        "admin_notes":                 row["admin_notes"] or "",
        "language":                    row["language"] or "",
        "merged_lead_trim_ms":         int(row["merged_lead_trim_ms"] or 0),
        "merged_trail_trim_ms":        int(row["merged_trail_trim_ms"] or 0),
        "member_trims":                json.loads(row["member_trims_json"] or "{}"),
    }


# ---------------------------------------------------------------------
# Create / read / mutate / dissolve
# ---------------------------------------------------------------------

def get_sample(sid: str) -> dict[str, Any] | None:
    conn = _require_conn()
    row = conn.execute(
        "SELECT * FROM capture_samples WHERE id = ?", (sid,),
    ).fetchone()
    return _row_to_dict(row) if row else None


def list_samples(
    *, user_id: str | None = None, status: str | None = None,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if user_id is not None:
        clauses.append("user_id = ?")
        params.append(user_id)
    if status is not None and status in _VALID_STATUS:
        clauses.append("status = ?")
        params.append(status)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = _require_conn().execute(
        f"SELECT * FROM capture_samples{where} ORDER BY created_ts DESC",
        params,
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_members(sid: str) -> list[dict[str, Any]]:
    """Return member captures in their declared sample_order, decoded
    enough for the UI (transcript + duration; no heavy words/segments).
    Includes corrections_json so chip-aware joiners can apply each
    member's corrections to its post-processing text before merging."""
    conn = _require_conn()
    rows = conn.execute(
        "SELECT id, created_ts, duration_seconds, raw, final,"
        " text_for_training, audio_trimmed_relpath,"
        " corrected_text, corrections_json, status, sample_order, user_id,"
        " language"
        " FROM captures WHERE sample_id = ? ORDER BY sample_order ASC",
        (sid,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        try:
            d["corrections"] = json.loads(d.pop("corrections_json", "[]") or "[]")
            if not isinstance(d["corrections"], list):
                d["corrections"] = []
        except (TypeError, ValueError):
            d["corrections"] = []
        out.append(d)
    return out


def update_sample(
    sid: str,
    patch: dict[str, Any],
) -> dict[str, Any] | None:
    """Patch transcript / transcript_join_strategy /
    inter_segment_silence_ms / is_locked / is_stale /
    member_hashes_json / merged_duration_ms / status / admin_notes /
    language / merged_lead_trim_ms / merged_trail_trim_ms /
    member_trims_json. Returns the updated row.

    Chip state (the user-facing "corrections" list) is NOT on the
    group row — it's derived from members on every read. Patch chip
    edits via `captures_store.update_capture` on each member instead."""
    allowed = {
        "transcript", "transcript_join_strategy",
        "inter_segment_silence_ms", "is_locked", "is_stale",
        "member_hashes_json", "merged_duration_ms",
        "status", "admin_notes", "language",
        "merged_lead_trim_ms", "merged_trail_trim_ms",
        "member_trims_json",
    }
    sets: list[str] = []
    args: list[Any] = []
    for k, v in patch.items():
        if k not in allowed:
            raise ValueError(f"field {k!r} not patchable")
        if k == "status":
            if v not in _VALID_STATUS:
                raise ValueError(f"invalid status: {v!r}")
        if k == "admin_notes":
            v = str(v or "")[:_CAP_ADMIN_NOTES]
        sets.append(f"{k} = ?")
        args.append(v)
    if not sets:
        return get_sample(sid)
    args.append(sid)
    conn = _require_conn()
    with _lock:
        conn.execute(
            f"UPDATE capture_samples SET {', '.join(sets)} WHERE id = ?", args,
        )
    return get_sample(sid)


def dissolve_sample(sid: str) -> None:
    """Delete the row, unlink the merged WAV, NULL out members'
    (sample_id, sample_order)."""
    g = get_sample(sid)
    if g is None:
        return
    conn = _require_conn()
    with _lock:
        with conn:
            conn.execute(
                "UPDATE captures SET sample_id = NULL, sample_order = NULL"
                " WHERE sample_id = ?",
                (sid,),
            )
            conn.execute("DELETE FROM capture_samples WHERE id = ?", (sid,))
    try:
        abs_p = abs_path_for(g["merged_wav_relpath"])
        if os.path.exists(abs_p):
            os.unlink(abs_p)
    except (OSError, ValueError) as e:
        logger.warning("[groups] failed to unlink %s: %s",
                       g["merged_wav_relpath"], e)
    try:
        import captures_merge_proposer
        captures_merge_proposer.invalidate(g.get("user_id"))
    except Exception:
        pass
    logger.info("[groups] dissolved sid=%s", sid[:8])


def clear_all_samples() -> int:
    """Drop every row from capture_samples. Caller is responsible for
    removing the merged WAV files (captures_store.clear_all handles
    that as part of its filesystem wipe). Returns count deleted."""
    conn = _require_conn()
    with _lock:
        row = conn.execute(
            "SELECT COUNT(*) FROM capture_samples",
        ).fetchone()
        n = int(row[0]) if row else 0
        conn.execute("DELETE FROM capture_samples")
    return n


def reconcile_on_startup() -> tuple[int, int, int]:
    """Audit the merged-group WAV subtree (`<CAPTURES_DIR>/groups/`)
    AND the captures→sample_id FKs. Mirrors
    `captures_store.reconcile_on_startup` but scoped to this store.

      1. For each group whose merged WAV is missing on disk, count it.
         No column is set — the GET /audio route already returns 404
         and the UI surfaces a Regenerate banner; tracking it twice
         would add no signal.
      2. For each file under `groups/` with no matching capture_samples
         row, unlink (true orphans: dissolved groups whose unlink
         failed, crashed regenerates leaving stale `.tmp`s, etc.).
      3. For each capture whose `sample_id` points to a group row that
         doesn't exist, NULL out `sample_id` + `sample_order`. This catches
         half-committed merges from earlier server versions and
         crash-recovery edge cases; without this, the capture is
         effectively unreachable (UI filters grouped captures from the
         flat list, and merge attempts are rejected with "already in
         a group").

    Returns (rows_with_missing_wav, files_unlinked, orphan_fks_cleared).
    """
    conn = _require_conn()
    groups_dir = _require_audio_root()
    rows_missing = 0
    files_unlinked = 0

    known_paths: set[str] = set()
    with _lock:
        rows = conn.execute(
            "SELECT id, merged_wav_relpath FROM capture_samples",
        ).fetchall()
    for r in rows:
        try:
            abs_p = abs_path_for(r["merged_wav_relpath"])
        except ValueError:
            continue
        if os.path.isfile(abs_p):
            known_paths.add(os.path.abspath(abs_p))
        else:
            rows_missing += 1

    if os.path.isdir(groups_dir):
        for root, _dirs, files in os.walk(groups_dir):
            for name in files:
                full = os.path.join(root, name)
                if name.endswith(".tmp"):
                    try:
                        os.unlink(full)
                    except OSError:
                        continue
                    files_unlinked += 1
                    continue
                p = os.path.abspath(full)
                if p not in known_paths:
                    try:
                        os.unlink(p)
                    except OSError:
                        continue
                    files_unlinked += 1

    # Pass 3: orphan-FK sweep. A capture whose sample_id points to a
    # missing group row gets returned to the flat list.
    orphan_fks_cleared = 0
    with _lock:
        cur = conn.execute(
            "UPDATE captures SET sample_id = NULL, sample_order = NULL"
            " WHERE sample_id IS NOT NULL"
            " AND sample_id NOT IN (SELECT id FROM capture_samples)"
        )
        orphan_fks_cleared = cur.rowcount or 0

    logger.info(
        "[groups] reconcile: %d rows with missing WAV, "
        "%d orphan files removed, %d orphan sample_id FKs cleared",
        rows_missing, files_unlinked, orphan_fks_cleared,
    )
    return rows_missing, files_unlinked, orphan_fks_cleared


