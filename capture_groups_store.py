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

_SCHEMA = """
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
  corrections_json            TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_capture_groups_user
  ON capture_groups(user_id, created_ts DESC);
"""

# Idempotent ALTER TABLE additions for DBs created under an older schema.
# Run after _SCHEMA so a fresh DB (with the column already present in
# CREATE TABLE) raises OperationalError and we swallow it.
_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE capture_groups ADD COLUMN corrections_json TEXT NOT NULL DEFAULT '[]'",
)


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
    _conn.executescript(_SCHEMA)
    for stmt in _MIGRATIONS:
        try:
            _conn.execute(stmt)
        except sqlite3.OperationalError:
            # Column already present (fresh DB or migrated previously).
            pass


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
    try:
        corrections = json.loads(row["corrections_json"] or "[]")
        if not isinstance(corrections, list):
            corrections = []
    except (TypeError, ValueError):
        corrections = []
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
        "corrections":                 corrections,
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


def list_groups(
    *, user_id: str | None = None,
) -> list[dict[str, Any]]:
    conn = _require_conn()
    if user_id is None:
        rows = conn.execute(
            "SELECT * FROM capture_groups ORDER BY created_ts DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM capture_groups WHERE user_id = ?"
            " ORDER BY created_ts DESC",
            (user_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_members(gid: str) -> list[dict[str, Any]]:
    """Return member captures in their declared group_order, decoded
    enough for the UI (transcript + duration; no heavy words/segments)."""
    conn = _require_conn()
    rows = conn.execute(
        "SELECT id, created_ts, duration_seconds, raw, final,"
        " corrected_text, status, group_order, user_id"
        " FROM captures WHERE group_id = ? ORDER BY group_order ASC",
        (gid,),
    ).fetchall()
    return [dict(r) for r in rows]


def update_group(
    gid: str,
    patch: dict[str, Any],
) -> dict[str, Any] | None:
    """Patch transcript / is_locked / inter_segment_silence_ms /
    transcript_join_strategy / is_stale / corrections_json. Returns the
    updated row."""
    allowed = {
        "transcript", "transcript_join_strategy",
        "inter_segment_silence_ms", "is_locked", "is_stale",
        "member_hashes_json", "merged_duration_ms", "corrections_json",
    }
    sets: list[str] = []
    args: list[Any] = []
    for k, v in patch.items():
        if k not in allowed:
            raise ValueError(f"field {k!r} not patchable")
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


def reconcile_on_startup() -> tuple[int, int]:
    """Audit the merged-group WAV subtree (`<CAPTURES_DIR>/groups/`).
    Mirrors `captures_store.reconcile_on_startup` but scoped to this
    store's files.

      1. For each group whose merged WAV is missing on disk, count it.
         No column is set — the GET /audio route already returns 404
         and the UI surfaces a Regenerate banner; tracking it twice
         would add no signal.
      2. For each file under `groups/` with no matching capture_groups
         row, unlink (true orphans: dissolved groups whose unlink
         failed, crashed regenerates leaving stale `.tmp`s, etc.).

    Returns (rows_with_missing_wav, files_unlinked).
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

    if rows_missing or files_unlinked:
        logger.warning(
            "[groups] reconcile: %d rows with missing WAV, "
            "%d orphan files removed",
            rows_missing, files_unlinked,
        )
    return rows_missing, files_unlinked


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
