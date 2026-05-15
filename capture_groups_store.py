"""Durable store for capture groups — packed ≤28 s training samples
built from 2+ consecutive same-speaker captures.

Lives in the same SQLite DB as `captures` (re-uses the connection from
`captures_store`). The merged WAV files sit under `CAPTURES_DIR/groups/`
with the same 4-char fanout as individual captures.

The schema is structurally minimal: one `capture_groups` row per merged
sample + a `group_id`/`group_order` FK on the existing captures table
(added by captures_store's migration). Dissolving a group deletes the
row + the merged WAV; members get their `group_id` NULL'd out and
return to the flat list.

Member-content drift is detected via per-member PCM-content hashes
stored in `member_hashes_json`. When an admin edits a member's
transcript, `recompute_stale_for_member` rehashes the file and marks
the group stale if the audio bytes changed — the merged WAV needs a
rebuild.
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
#   2. _MIGRATIONS      — idempotent ALTER TABLE ADD COLUMN for every
#                         column added after the first ship. Each ALTER
#                         is run in its own try/except so a fresh DB
#                         (where the CREATE TABLE already includes the
#                         column) swallows "duplicate column …" and
#                         keeps going.
#   3. _SCHEMA_POST_MIGRATIONS — indexes that reference columns added
#                         by phase 2. If we put `CREATE INDEX … ON
#                         capture_groups(status)` in phase 1 alongside
#                         the CREATE TABLE, then on a DB upgraded from
#                         before `status` existed the CREATE TABLE is a
#                         no-op, the CREATE INDEX raises "no such
#                         column: status", and `executescript` aborts —
#                         which skips phase 2 entirely and leaves the
#                         store unable to read its own rows.
_SCHEMA_CORE = """
CREATE TABLE IF NOT EXISTS capture_groups (
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
  admin_notes                 TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_capture_groups_user
  ON capture_groups(user_id, created_ts DESC);
"""

_MIGRATIONS: tuple[str, ...] = (
    # Old DBs still carry these two columns from the one-time-projection
    # cache design. Drop them idempotently — fresh DBs see "no such
    # column" and swallow it; upgraded DBs succeed and reclaim the space.
    "ALTER TABLE capture_groups DROP COLUMN corrections_json",
    "ALTER TABLE capture_groups DROP COLUMN corrections_migrated_at",
    # Live columns that may need adding on older DBs that pre-date them.
    "ALTER TABLE capture_groups ADD COLUMN status TEXT NOT NULL DEFAULT 'new'",
    "ALTER TABLE capture_groups ADD COLUMN admin_notes TEXT NOT NULL DEFAULT ''",
)

_SCHEMA_POST_MIGRATIONS = """
CREATE INDEX IF NOT EXISTS idx_capture_groups_status
  ON capture_groups(status);
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
    _conn.executescript(_SCHEMA_CORE)
    for stmt in _MIGRATIONS:
        try:
            _conn.execute(stmt)
        except sqlite3.OperationalError:
            # Column already present (fresh DB or migrated previously).
            pass
    _conn.executescript(_SCHEMA_POST_MIGRATIONS)


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("capture_groups_store.init() was not called before use.")
    return _conn


def _require_audio_root() -> str:
    if _groups_audio_dir is None:
        raise RuntimeError("capture_groups_store.init() was not called before use.")
    return _groups_audio_dir


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------

def _relpath_for(gid: str) -> str:
    """`groups/<g0g1>/<g2g3>/<gid>.wav` — mirrors the captures fanout."""
    return os.path.join("groups", gid[0:2], gid[2:4], f"{gid}.wav")


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
    # (see captures_routes._enrich_group / list_groups_api). The DB row
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
    }


# ---------------------------------------------------------------------
# Create / read / mutate / dissolve
# ---------------------------------------------------------------------

def create_group(
    *,
    user_id: str,
    member_ids: list[str],
    transcript: str,
    transcript_join_strategy: str,
    inter_segment_silence_ms: int,
    member_hash_map: dict[str, str],
    merged_duration_ms: int,
) -> str:
    """Insert the group row AND wire members' (group_id, group_order)
    in the same transaction. Returns the new group_id.

    Callers must build the merged WAV BEFORE this call — we record its
    relpath, but don't write the file ourselves.
    """
    if len(member_ids) < 2:
        raise ValueError("group must have at least 2 members")
    gid = uuid.uuid4().hex
    relpath = _relpath_for(gid)
    now = time.time()
    conn = _require_conn()
    with _lock:
        with conn:                          # BEGIN/COMMIT transaction
            conn.execute(
                "INSERT INTO capture_groups"
                " (id, user_id, created_ts, merged_wav_relpath,"
                "  merged_duration_ms, transcript,"
                "  transcript_join_strategy, member_hashes_json,"
                "  inter_segment_silence_ms, is_stale, is_locked)"
                " VALUES (?,?,?,?,?,?,?,?,?,0,0)",
                (
                    gid, user_id, now, relpath, int(merged_duration_ms),
                    transcript, transcript_join_strategy,
                    json.dumps(member_hash_map, sort_keys=True),
                    int(inter_segment_silence_ms),
                ),
            )
            for order, mid in enumerate(member_ids):
                conn.execute(
                    "UPDATE captures SET group_id = ?, group_order = ?"
                    " WHERE id = ? AND group_id IS NULL",
                    (gid, order, mid),
                )
    logger.info(
        "[groups] created gid=%s user=%s n=%d dur=%.1fs",
        gid[:8], user_id[:8] if user_id else "?",
        len(member_ids), merged_duration_ms / 1000.0,
    )
    return gid


def get_group(gid: str) -> dict[str, Any] | None:
    conn = _require_conn()
    row = conn.execute(
        "SELECT * FROM capture_groups WHERE id = ?", (gid,),
    ).fetchone()
    return _row_to_dict(row) if row else None


_VALID_STATUS = {"new", "reviewed", "ready", "dismissed"}


def list_groups(
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
        f"SELECT * FROM capture_groups{where} ORDER BY created_ts DESC",
        params,
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_members(gid: str) -> list[dict[str, Any]]:
    """Return member captures in their declared group_order, decoded
    enough for the UI (transcript + duration; no heavy words/segments).
    Includes corrections_json so chip-aware joiners can apply each
    member's corrections to its post-processing text before merging."""
    conn = _require_conn()
    rows = conn.execute(
        "SELECT id, created_ts, duration_seconds, raw, final,"
        " corrected_text, corrections_json, status, group_order, user_id"
        " FROM captures WHERE group_id = ? ORDER BY group_order ASC",
        (gid,),
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


def update_group(
    gid: str,
    patch: dict[str, Any],
) -> dict[str, Any] | None:
    """Patch transcript / is_locked / inter_segment_silence_ms /
    transcript_join_strategy / is_stale / status / admin_notes /
    member_hashes_json / merged_duration_ms. Returns the updated row.

    Chip state (the user-facing "corrections" list) is NOT on the
    group row — it's derived from members on every read. Patch chip
    edits via `captures_store.update_capture` on each member instead."""
    allowed = {
        "transcript", "transcript_join_strategy",
        "inter_segment_silence_ms", "is_locked", "is_stale",
        "member_hashes_json", "merged_duration_ms",
        "status", "admin_notes",
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
        return get_group(gid)
    args.append(gid)
    conn = _require_conn()
    with _lock:
        conn.execute(
            f"UPDATE capture_groups SET {', '.join(sets)} WHERE id = ?", args,
        )
    return get_group(gid)


def mark_stale(gid: str, *, stale: bool = True) -> None:
    update_group(gid, {"is_stale": 1 if stale else 0})


def dissolve_group(gid: str) -> None:
    """Delete the row, unlink the merged WAV, NULL out members'
    (group_id, group_order)."""
    g = get_group(gid)
    if g is None:
        return
    conn = _require_conn()
    with _lock:
        with conn:
            conn.execute(
                "UPDATE captures SET group_id = NULL, group_order = NULL"
                " WHERE group_id = ?",
                (gid,),
            )
            conn.execute("DELETE FROM capture_groups WHERE id = ?", (gid,))
    try:
        abs_p = abs_path_for(g["merged_wav_relpath"])
        if os.path.exists(abs_p):
            os.unlink(abs_p)
    except (OSError, ValueError) as e:
        logger.warning("[groups] failed to unlink %s: %s",
                       g["merged_wav_relpath"], e)
    logger.info("[groups] dissolved gid=%s", gid[:8])


def clear_all_groups() -> int:
    """Drop every row from capture_groups. Caller is responsible for
    removing the merged WAV files (captures_store.clear_all handles
    that as part of its filesystem wipe). Returns count deleted."""
    conn = _require_conn()
    with _lock:
        row = conn.execute(
            "SELECT COUNT(*) FROM capture_groups",
        ).fetchone()
        n = int(row[0]) if row else 0
        conn.execute("DELETE FROM capture_groups")
    return n


def reconcile_on_startup() -> tuple[int, int, int]:
    """Audit the merged-group WAV subtree (`<CAPTURES_DIR>/groups/`)
    AND the captures→group_id FKs. Mirrors
    `captures_store.reconcile_on_startup` but scoped to this store.

      1. For each group whose merged WAV is missing on disk, count it.
         No column is set — the GET /audio route already returns 404
         and the UI surfaces a Regenerate banner; tracking it twice
         would add no signal.
      2. For each file under `groups/` with no matching capture_groups
         row, unlink (true orphans: dissolved groups whose unlink
         failed, crashed regenerates leaving stale `.tmp`s, etc.).
      3. For each capture whose `group_id` points to a group row that
         doesn't exist, NULL out `group_id` + `group_order`. This catches
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
            "SELECT id, merged_wav_relpath FROM capture_groups",
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
                        files_unlinked += 1
                    except OSError:
                        pass
                    continue
                p = os.path.abspath(full)
                if p not in known_paths:
                    try:
                        os.unlink(p)
                        files_unlinked += 1
                    except OSError:
                        pass

    # Pass 3: orphan-FK sweep. A capture whose group_id points to a
    # missing group row gets returned to the flat list.
    orphan_fks_cleared = 0
    with _lock:
        cur = conn.execute(
            "UPDATE captures SET group_id = NULL, group_order = NULL"
            " WHERE group_id IS NOT NULL"
            " AND group_id NOT IN (SELECT id FROM capture_groups)"
        )
        orphan_fks_cleared = cur.rowcount or 0

    logger.info(
        "[groups] reconcile: %d rows with missing WAV, "
        "%d orphan files removed, %d orphan group_id FKs cleared",
        rows_missing, files_unlinked, orphan_fks_cleared,
    )
    return rows_missing, files_unlinked, orphan_fks_cleared


def find_group_for_member(capture_id: str) -> str | None:
    """Return the group_id that owns this capture, or None."""
    conn = _require_conn()
    row = conn.execute(
        "SELECT group_id FROM captures WHERE id = ?", (capture_id,),
    ).fetchone()
    return row["group_id"] if row and row["group_id"] else None


def recompute_stale_for_member(capture_id: str) -> None:
    """Called from captures_store.update_capture whenever a member's
    audio could have changed (currently: a transcript edit doesn't
    change audio, but future audio re-uploads will). For now we
    invoke it on every member edit as defense-in-depth; the hash
    compare is cheap (~50 ms for 15 s)."""
    gid = find_group_for_member(capture_id)
    if not gid:
        return
    g = get_group(gid)
    if not g:
        return
    # Re-hash the member's current PCM bytes and compare against the
    # snapshot stored at merge time.
    expected = g["member_hashes"].get(capture_id)
    if not expected:
        return
    import audio_merge
    import captures_store
    cap = captures_store.get_capture(capture_id)
    if not cap:
        return
    try:
        abs_p = captures_store.abs_audio_path(cap["audio_relpath"])
        current = audio_merge.hash_wav_pcm(abs_p)
    except Exception as e:
        logger.warning(
            "[groups] hash check failed for %s: %s", capture_id[:8], e,
        )
        return
    if current != expected:
        mark_stale(gid, stale=True)
        logger.info(
            "[groups] gid=%s marked stale (member %s drift)",
            gid[:8], capture_id[:8],
        )
