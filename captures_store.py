"""Durable store for transcription audio + word-timestamps + admin
corrections — the source-of-truth for Whisper fine-tuning training data.

Storage layout:

  cfg.CAPTURES_DB    — SQLite (WAL) holding the row metadata, word
                       timestamps as JSON, admin corrections, status.
  cfg.CAPTURES_DIR   — filesystem tree with the raw audio files,
                       organized in a 4-char fanout
                       (`<dir>/<id[0:2]>/<id[2:4]>/<id>.<ext>`) to keep
                       any single directory's child count modest even at
                       high row counts.

The two stores can desync (file present, row missing, or vice versa); the
write path uses .tmp + os.replace so a successful insert implies a
complete file, and reconcile_on_startup() handles drift from crashes /
manual cleanup.

PHI hygiene: audio is biometric-grade PHI. The disk format is plaintext;
whole-disk encryption is the deployment's responsibility. INFO/WARNING
log lines carry only counts + id prefixes, never content.

The route layer must NOT trust audio_format from a row directly when
serving — paths must be re-resolved against CAPTURES_DIR and rejected
if they escape it (path traversal defense).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import threading
import time
import uuid
from typing import Any

import text_corrections

logger = logging.getLogger("whisper-api")

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_audio_dir: str | None = None

# Field caps.
_CAP_RAW = 50_000
_CAP_FINAL = 50_000
_CAP_CORRECTED = 100_000  # admin-edited free-form ground truth
_CAP_ADMIN_NOTES = 8_000
_CAP_WORDS_JSON = 1_000_000  # ~10k words at ~100 bytes/word; fine-grained
_CAP_SEGMENTS_JSON = 200_000
_CAP_AUDIO_FORMAT = 16  # "wav", "m4a", "webm" — short extensions only

_VALID_STATUS = frozenset({"new", "reviewed", "ready", "dismissed", "audio_missing"})

# Eviction priority — earlier statuses get evicted first. Training data
# ("ready") wins; unreviewed work ("new") is also protected because the
# admin needs to act on it.
_EVICTION_ORDER = ("dismissed", "audio_missing", "reviewed", "new", "ready")

# Schema is split into THREE phases on purpose, ordered around the
# user_id / sample_id migration:
#
#   1. _SCHEMA_CORE  — table + indexes that reference ONLY original
#      columns. Safe to run against a pre-flag DB.
#   2. _MIGRATIONS   — ALTER TABLE ADD COLUMN for every column added
#      after the first ship (currently: user_id, sample_id, sample_order,
#      text_for_training, audio_trimmed_relpath, audio_trim_lead_ms,
#      audio_trim_trail_ms). Idempotent — each stmt runs in its own
#      try/except so a fresh DB whose CREATE TABLE already includes
#      the column swallows "duplicate column …" and keeps going.
#   3. _SCHEMA_USER_GROUP_INDEXES — indexes that reference user_id /
#      sample_id. MUST run AFTER step 2 — if we packaged them into
#      _SCHEMA_CORE with executescript(), the index creation on a
#      pre-flag DB raised "no such column: user_id" and the executescript
#      bailed before step 2 could fix it, leaving the table missing
#      both the columns AND the indexes.
_SCHEMA_CORE = """
CREATE TABLE IF NOT EXISTS captures (
  id              TEXT PRIMARY KEY,
  created_ts      REAL NOT NULL,
  request_id      TEXT,
  model           TEXT NOT NULL,
  language        TEXT,
  duration_seconds REAL,
  audio_relpath   TEXT NOT NULL,
  audio_format    TEXT NOT NULL,
  raw             TEXT NOT NULL,
  final           TEXT NOT NULL,
  text_for_training TEXT,
  audio_trimmed_relpath TEXT,
  audio_trim_lead_ms INTEGER,
  audio_trim_trail_ms INTEGER,
  words_json      TEXT NOT NULL,
  segments_json   TEXT NOT NULL DEFAULT '[]',
  corrected_text  TEXT NOT NULL DEFAULT '',
  corrections_json TEXT NOT NULL DEFAULT '[]',
  admin_notes     TEXT NOT NULL DEFAULT '',
  status          TEXT NOT NULL DEFAULT 'new',
  reviewed_ts     REAL,
  user_id         TEXT,
  sample_id        TEXT,
  sample_order     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_captures_created    ON captures(created_ts DESC);
CREATE INDEX IF NOT EXISTS idx_captures_status     ON captures(status);
CREATE INDEX IF NOT EXISTS idx_captures_request_id ON captures(request_id);
"""

# Migrations for installs that predate later columns. SQLite ALTER ADD
# COLUMN raises OperationalError("duplicate column …") on a fresh DB
# whose CREATE TABLE already includes them, so we swallow that specific
# error and keep going.
#
# text_for_training: post-processing text built with the captures-specific
#   pipeline-rule exclude set applied (default-skipped: `dictation-map` +
#   `capitalize-after-terminator`). Used by /captures UI + the export
#   manifest so reviewers see — and Whisper trains on — the word-form
#   transcript that matches the model's raw output at inference time
#   under SUPPRESS_CHARS.
# audio_trimmed_relpath: optional separate WAV with leading/trailing
#   silence cut via Silero VAD (per-singleton manual trim). NULL means
#   "use audio_relpath".
_MIGRATIONS = (
    # group→sample terminology rename. RENAME first so existing DBs keep
    # their membership data (the later ADD COLUMN sample_id then duplicate-
    # fails harmlessly); fresh DBs swallow "no such column" and the ADD
    # creates them. SQLite ≥3.25 auto-updates dependent index definitions.
    "ALTER TABLE captures RENAME COLUMN group_id TO sample_id",
    "ALTER TABLE captures RENAME COLUMN group_order TO sample_order",
    "ALTER TABLE captures ADD COLUMN user_id TEXT",
    "ALTER TABLE captures ADD COLUMN sample_id TEXT",
    "ALTER TABLE captures ADD COLUMN sample_order INTEGER",
    "ALTER TABLE captures ADD COLUMN text_for_training TEXT",
    "ALTER TABLE captures ADD COLUMN audio_trimmed_relpath TEXT",
    # VAD silence-trim offset bookkeeping. NULL means "row was never
    # trimmed"; populated values record how many ms were cut from each
    # edge of the original WAV. Words/segments stay in original-audio
    # time in the JSON columns — the route layer applies the offset
    # before responding so the karaoke player aligns with the trimmed
    # audio file served by GET /audio.
    "ALTER TABLE captures ADD COLUMN audio_trim_lead_ms INTEGER",
    "ALTER TABLE captures ADD COLUMN audio_trim_trail_ms INTEGER",
)

_SCHEMA_USER_GROUP_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_captures_user  ON captures(user_id, created_ts DESC);
CREATE INDEX IF NOT EXISTS idx_captures_group ON captures(sample_id, sample_order);
"""


# ---------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------

def init(db_path: str, audio_dir: str) -> None:
    """Open the SQLite DB (WAL) and ensure the audio dir exists. Idempotent:
    safe to call on every startup. Call once before any other function."""
    global _conn, _audio_dir
    _audio_dir = audio_dir
    db_dir = os.path.dirname(os.path.abspath(db_path)) or "."
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    _conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL;")
    _conn.execute("PRAGMA synchronous=NORMAL;")
    _conn.executescript(_SCHEMA_CORE)
    for stmt in _MIGRATIONS:
        try:
            _conn.execute(stmt)
        except sqlite3.OperationalError:
            # Column already present — idempotent.
            pass
    _conn.executescript(_SCHEMA_USER_GROUP_INDEXES)


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("captures_store.init() was not called before use.")
    return _conn


def _require_audio_dir() -> str:
    if _audio_dir is None:
        raise RuntimeError("captures_store.init() was not called before use.")
    return _audio_dir


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------

def _relpath_for(cid: str, ext: str) -> str:
    """Compute the fanned-out audio relpath for a capture id.

    4-char fanout keeps each directory under ~256 entries even when the
    store grows beyond the default 5k cap. Lowercase the extension and
    strip a leading dot so the file name is `<id>.<ext>`."""
    ext = ext.lstrip(".").lower()[:_CAP_AUDIO_FORMAT]
    if not ext:
        ext = "bin"
    return os.path.join(cid[0:2], cid[2:4], f"{cid}.{ext}")


def abs_audio_path(audio_relpath: str) -> str:
    """Resolve relpath to absolute path under CAPTURES_DIR. Includes a
    path-traversal defense — the resolved path MUST stay inside the
    audio root. Raises ValueError if it escapes."""
    root = os.path.abspath(_require_audio_dir())
    abs_p = os.path.abspath(os.path.join(root, audio_relpath))
    # commonpath rejects different drives on Windows by raising
    try:
        common = os.path.commonpath([abs_p, root])
    except ValueError:
        raise ValueError("audio path escapes captures dir")
    if common != root:
        raise ValueError("audio path escapes captures dir")
    return abs_p


def _safe_unlink(abs_path: str) -> bool:
    """Best-effort unlink with Windows AV-lock retry. Returns True if the
    file is gone after the call (including the "never existed" case)."""
    for attempt in range(3):
        try:
            os.unlink(abs_path)
            return True
        except FileNotFoundError:
            return True
        except OSError as e:
            if attempt == 2:
                logger.warning(
                    "[captures] failed to unlink %s: %s",
                    os.path.basename(abs_path), e,
                )
                return False
            time.sleep(0.1)
    return False


# ---------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row, include_words: bool = True) -> dict[str, Any]:
    """Materialize a row, decoding the JSON-bearing columns. With
    include_words=False the heavy fields (words_json, segments_json)
    are dropped — used by /list to keep the wire payload light."""
    d = dict(row)
    try:
        d["corrections"] = json.loads(d.pop("corrections_json", "[]") or "[]")
    except (TypeError, ValueError):
        d["corrections"] = []
    if include_words:
        try:
            d["words"] = json.loads(d.pop("words_json", "[]") or "[]")
        except (TypeError, ValueError):
            d["words"] = []
        try:
            d["segments"] = json.loads(d.pop("segments_json", "[]") or "[]")
        except (TypeError, ValueError):
            d["segments"] = []
    else:
        d.pop("words_json", None)
        d.pop("segments_json", None)
    return d


# ---------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------

def count() -> int:
    """Total row count (any status). Cheap; used by the transcribe handler
    to short-circuit the capture decision when the store is at its cap,
    and by /quick-config to decide whether to surface the reapply-rules
    modal."""
    conn = _require_conn()
    row = conn.execute("SELECT COUNT(*) FROM captures").fetchone()
    return int(row[0]) if row else 0


def create_capture(
    *,
    audio_src_path: str,
    request_id: str | None,
    model: str,
    language: str | None,
    duration_seconds: float | None,
    raw: str,
    final: str,
    text_for_training: str | None = None,
    words: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    user_id: str | None = None,
) -> str:
    """Transcode source audio into a 16 kHz mono WAV at the row's
    audio_relpath, then insert the SQLite row. On row-insert failure
    we unlink the audio file so we never orphan a multi-MB blob.

    Every internal file is RIFF/WAVE — universal browser playback
    (Firefox on Linux ships no AAC decoder, so we can't store .m4a)
    and Whisper's native input rate, so fine-tuning loses no quality."""
    import audio_transcode

    cid = uuid.uuid4().hex
    relpath = _relpath_for(cid, "wav")
    abs_path = abs_audio_path(relpath)

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    tmp_path = abs_path + ".tmp"
    try:
        wav_bytes = audio_transcode.transcode_to_wav_16k_mono(
            audio_src_path, tmp_path,
        )
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise RuntimeError(f"audio transcode failed: {e}") from e

    # Atomic visibility is handled by writing to .tmp + os.replace below;
    # transcode_to_wav_16k_mono already closed the write fd, and re-opening
    # read-only just to fsync is a no-op on Windows (fsync needs write
    # access) and the .tmp+replace dance already guards the
    # crash-between-rename-and-insert window.
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            os.replace(tmp_path, abs_path)
            last_err = None
            break
        except OSError as e:
            last_err = e
            time.sleep(0.1)
    if last_err is not None:
        _safe_unlink(tmp_path)
        raise last_err

    # Insert row. On failure, drop the audio file so we don't orphan it.
    now = time.time()
    raw_t = (raw or "")[:_CAP_RAW]
    final_t = (final or "")[:_CAP_FINAL]
    training_t = (text_for_training or "")[:_CAP_FINAL] if text_for_training is not None else None
    words_t = _truncate_json(words or [], _CAP_WORDS_JSON)
    segments_t = _truncate_json(segments or [], _CAP_SEGMENTS_JSON)

    try:
        conn = _require_conn()
        with _lock:
            conn.execute(
                "INSERT INTO captures ("
                " id, created_ts, request_id, model, language,"
                " duration_seconds, audio_relpath, audio_format,"
                " raw, final, text_for_training, audio_trimmed_relpath,"
                " words_json, segments_json,"
                " corrected_text, corrections_json, admin_notes,"
                " status, reviewed_ts, user_id, sample_id, sample_order"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cid, now, request_id, model or "", language or "",
                    float(duration_seconds or 0.0), relpath, "wav",
                    raw_t, final_t, training_t, None,
                    words_t, segments_t,
                    "", "[]", "",
                    "new", None,
                    user_id, None, None,
                ),
            )
            _evict_to_cap(conn)
    except Exception:
        _safe_unlink(abs_path)
        raise

    logger.info(
        "[captures] created id=%s model=%s dur=%.1fs words=%d wav_bytes=%d",
        cid[:8], model or "?", float(duration_seconds or 0.0),
        len(words or []), int(wav_bytes),
    )
    # Drop the proposer cache so the freshly-recorded clip is eligible on
    # the next Auto-propose-merges call instead of waiting up to TTL_S.
    try:
        import captures_merge_proposer
        captures_merge_proposer.invalidate(user_id)
    except Exception:
        pass
    return cid


def _truncate_json(items: list[Any], cap_bytes: int) -> str:
    """Serialize a list and trim from the tail if the blob is over the
    cap. Falls back to '[]' on serialization failure (defense against
    non-serializable garbage in the words list)."""
    try:
        blob = json.dumps(items, ensure_ascii=False)
    except (TypeError, ValueError):
        return "[]"
    if len(blob) <= cap_bytes:
        return blob
    # Trim from the end — front entries carry more signal for a karaoke
    # review (the start of the clip is the most reviewed part).
    lo, hi = 0, len(items)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        blob = json.dumps(items[:mid], ensure_ascii=False)
        if len(blob) <= cap_bytes:
            lo = mid
        else:
            hi = mid - 1
    return json.dumps(items[:lo], ensure_ascii=False)


def _evict_to_cap(conn: sqlite3.Connection) -> None:
    """Enforce CAPTURES_MAX (row count) and CAPTURES_MAX_MB (total
    audio bytes). Drops rows + their audio files in _EVICTION_ORDER
    priority. _lock is already held by the caller."""
    try:
        import config as cfg
        row_cap = int(getattr(cfg, "CAPTURES_MAX", 5000))
        mb_cap = int(getattr(cfg, "CAPTURES_MAX_MB", 5000))
    except Exception:
        row_cap = 5000
        mb_cap = 5000
    if row_cap < 1 and mb_cap < 1:
        return

    row = conn.execute("SELECT COUNT(*) FROM captures").fetchone()
    total = int(row[0]) if row else 0

    # Row-count overflow first (cheap). Audio-byte overflow is more
    # expensive (os.path.getsize per row), so we walk that pass second.
    excess = max(0, total - row_cap) if row_cap >= 1 else 0
    evicted = 0
    if excess > 0:
        for status in _EVICTION_ORDER:
            if excess <= 0:
                break
            evicted_status = _drop_oldest_with_status(conn, status, excess)
            excess -= evicted_status
            evicted += evicted_status

    bytes_freed = 0
    if mb_cap >= 1:
        # Walk every row to sum audio bytes — there's no cached size
        # column, so this is O(N) getsize calls per insert. Hot-path
        # cost is tolerated only because typical N is in the few-thousand
        # range and the captures dir lives on local disk.
        byte_cap = mb_cap * 1024 * 1024
        total_bytes = _total_audio_bytes(conn)
        if total_bytes > byte_cap:
            for status in _EVICTION_ORDER:
                if total_bytes <= byte_cap:
                    break
                freed = _drop_oldest_by_bytes(
                    conn, status, total_bytes - byte_cap,
                )
                total_bytes -= freed
                bytes_freed += freed

    if evicted or bytes_freed:
        logger.info(
            "[captures] evicted to cap: row_cap=%d, mb_cap=%d,"
            " rows_dropped=%d, bytes_freed=%d",
            row_cap, mb_cap, evicted, bytes_freed,
        )


def _total_audio_bytes(conn: sqlite3.Connection) -> int:
    # Restrict to non-group rows: _drop_oldest_by_bytes excludes group
    # members from eviction (they are owned by capture_samples_store), so
    # counting their bytes here would let the total cross the cap with
    # nothing the evictor can drop, causing a full scan on every insert.
    # Include the VAD-trimmed companion: _drop_oldest_by_bytes credits its
    # size to `freed`, so this counter must also include it or the cap
    # stays loose by total-trimmed-bytes.
    total = 0
    rows = conn.execute(
        "SELECT audio_relpath, audio_trimmed_relpath FROM captures"
        " WHERE sample_id IS NULL"
    )
    for row in rows:
        try:
            total += os.path.getsize(abs_audio_path(row["audio_relpath"]))
        except (OSError, ValueError):
            pass
        trimmed = row["audio_trimmed_relpath"]
        if trimmed:
            try:
                total += os.path.getsize(abs_audio_path(trimmed))
            except (OSError, ValueError):
                pass
    return total


def _drop_oldest_with_status(
    conn: sqlite3.Connection, status: str, limit: int,
) -> int:
    """Delete up to `limit` oldest rows of a given status, also unlinking
    audio files (primary + trimmed companion). Returns the count deleted.
    Group members are excluded from eviction — losing a member silently
    breaks the owning merged-WAV; if the user wants to evict, they must
    dissolve the group first (or use clear_all)."""
    if limit <= 0:
        return 0
    rows = conn.execute(
        "SELECT id, audio_relpath, audio_trimmed_relpath FROM captures"
        " WHERE status = ? AND sample_id IS NULL"
        " ORDER BY created_ts ASC LIMIT ?",
        (status, limit),
    ).fetchall()
    if not rows:
        return 0
    ids = [r["id"] for r in rows]
    placeholders = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM captures WHERE id IN ({placeholders})", ids)
    for r in rows:
        try:
            _safe_unlink(abs_audio_path(r["audio_relpath"]))
        except ValueError:
            pass
        trimmed = r["audio_trimmed_relpath"]
        if trimmed:
            try:
                _safe_unlink(abs_audio_path(trimmed))
            except ValueError:
                pass
    return len(rows)


def _drop_oldest_by_bytes(
    conn: sqlite3.Connection, status: str, bytes_needed: int,
) -> int:
    """Delete oldest rows of a given status until `bytes_needed` bytes
    are freed. Returns bytes actually freed. Group members are
    excluded — see _drop_oldest_with_status for the rationale."""
    if bytes_needed <= 0:
        return 0
    rows = conn.execute(
        "SELECT id, audio_relpath, audio_trimmed_relpath FROM captures"
        " WHERE status = ? AND sample_id IS NULL"
        " ORDER BY created_ts ASC",
        (status,),
    ).fetchall()
    freed = 0
    drop_ids: list[str] = []
    drop_paths: list[str] = []
    for r in rows:
        if freed >= bytes_needed:
            break
        try:
            abs_p = abs_audio_path(r["audio_relpath"])
            sz = os.path.getsize(abs_p)
        except (OSError, ValueError):
            sz = 0
            abs_p = ""
        drop_ids.append(r["id"])
        if abs_p:
            drop_paths.append(abs_p)
        trimmed = r["audio_trimmed_relpath"]
        if trimmed:
            try:
                abs_t = abs_audio_path(trimmed)
                drop_paths.append(abs_t)
                try:
                    sz += os.path.getsize(abs_t)
                except OSError:
                    pass
            except ValueError:
                pass
        freed += sz
    if drop_ids:
        placeholders = ",".join("?" * len(drop_ids))
        conn.execute(
            f"DELETE FROM captures WHERE id IN ({placeholders})", drop_ids,
        )
        for p in drop_paths:
            _safe_unlink(p)
    return freed


# ---------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------

# Column projection for list/find queries that don't need words/segments.
# Hoisted to a constant so list_captures and find_by_request_id can't
# silently drift apart when a new column is added.
_LIST_COLUMNS = (
    "id, created_ts, request_id, model, language,"
    " duration_seconds, audio_relpath, audio_format,"
    " raw, final, text_for_training, audio_trimmed_relpath,"
    " audio_trim_lead_ms, audio_trim_trail_ms,"
    " corrected_text, corrections_json, admin_notes,"
    " status, reviewed_ts, user_id, sample_id, sample_order"
)


def list_captures(
    *,
    status: str | None = None,
    limit: int = 200,
    before_ts: float | None = None,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Newest-first listing, optionally filtered by status and user_id.
    Heavy fields (words / segments) are dropped to keep the wire payload
    light.

    `user_id=None` means "do not filter" — callers in admin contexts pass
    None; per-user contexts pass the caller's id so the user can only see
    their own captures.

    Use `before_ts` for cursor pagination — repeat the call passing the
    oldest created_ts from the previous page."""
    conn = _require_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if status and status != "all":
        clauses.append("status = ?")
        params.append(status)
    if before_ts is not None:
        clauses.append("created_ts < ?")
        params.append(float(before_ts))
    if user_id is not None:
        clauses.append("user_id = ?")
        params.append(user_id)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(int(limit))
    cur = conn.execute(
        f"SELECT {_LIST_COLUMNS} FROM captures{where}"
        f" ORDER BY created_ts DESC LIMIT ?",
        params,
    )
    return [_row_to_dict(r, include_words=False) for r in cur.fetchall()]


def iter_captures_for_export(
    *,
    status: str | None = None,
    user_id: str | None = None,
):
    """Generator yielding capture rows (full payload incl. words) in
    deterministic order. Used by the export endpoint to stream a tarball
    without holding the whole result set in memory.

    `user_id=None` means "no filter" (admin / scope=all). A string
    narrows to a single owner — symmetric to `list_captures`, ready
    for the day non-admin scope=own users get export access (the
    endpoint stays admin-only for now per the plan)."""
    conn = _require_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if status and status != "all":
        clauses.append("status = ?")
        params.append(status)
    if user_id is not None:
        clauses.append("user_id = ?")
        params.append(user_id)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    cur = conn.execute(
        f"SELECT * FROM captures{where} ORDER BY created_ts ASC", params,
    )
    for row in cur:
        yield _row_to_dict(row, include_words=True)


def get_capture(cid: str) -> dict[str, Any] | None:
    conn = _require_conn()
    row = conn.execute(
        "SELECT * FROM captures WHERE id = ?", (cid,),
    ).fetchone()
    return _row_to_dict(row, include_words=True) if row else None


def find_by_request_id(request_id: str) -> list[dict[str, Any]]:
    """Cross-link from /reports → captures matching a request_id. Used
    by the /captures `by-request` jump endpoint to surface every capture
    that shares a transcription request_id."""
    conn = _require_conn()
    cur = conn.execute(
        f"SELECT {_LIST_COLUMNS} FROM captures WHERE request_id = ?"
        f" ORDER BY created_ts DESC",
        (request_id,),
    )
    return [_row_to_dict(r, include_words=False) for r in cur.fetchall()]


def counts_by_status() -> dict[str, int]:
    """Status breakdown for the page toolbar."""
    conn = _require_conn()
    out = {s: 0 for s in _VALID_STATUS}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM captures GROUP BY status"
    ):
        if row["status"] in out:
            out[row["status"]] = int(row["n"])
    return out


# ---------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------

def update_capture(cid: str, patch: dict[str, Any]) -> dict[str, Any] | None:
    """Apply a partial update. Allowed fields: status, corrected_text,
    corrections, admin_notes, final, text_for_training,
    audio_trimmed_relpath, audio_trim_lead_ms, audio_trim_trail_ms.
    Returns the updated row or None if not found."""
    if not patch:
        return get_capture(cid)
    sets: list[str] = []
    params: list[Any] = []
    if "status" in patch:
        new_status = str(patch["status"] or "new")
        if new_status not in _VALID_STATUS:
            raise ValueError(f"invalid status: {new_status!r}")
        sets.append("status = ?")
        params.append(new_status)
        if new_status == "new":
            sets.append("reviewed_ts = NULL")
        else:
            sets.append("reviewed_ts = ?")
            params.append(time.time())
    if "corrected_text" in patch:
        text = str(patch["corrected_text"] or "")[:_CAP_CORRECTED]
        sets.append("corrected_text = ?")
        params.append(text)
    if "corrections" in patch:
        cleaned = text_corrections.clean_corrections(patch["corrections"])
        sets.append("corrections_json = ?")
        params.append(json.dumps(cleaned, ensure_ascii=False))
    if "admin_notes" in patch:
        notes = str(patch["admin_notes"] or "")[:_CAP_ADMIN_NOTES]
        sets.append("admin_notes = ?")
        params.append(notes)
    if "final" in patch:
        sets.append("final = ?")
        params.append(str(patch["final"] or "")[:_CAP_FINAL])
    if "text_for_training" in patch:
        sets.append("text_for_training = ?")
        val = patch["text_for_training"]
        params.append(str(val)[:_CAP_FINAL] if val is not None else None)
    if "audio_trimmed_relpath" in patch:
        sets.append("audio_trimmed_relpath = ?")
        val = patch["audio_trimmed_relpath"]
        params.append(str(val) if val else None)
    if "audio_trim_lead_ms" in patch:
        sets.append("audio_trim_lead_ms = ?")
        val = patch["audio_trim_lead_ms"]
        params.append(int(val) if val is not None else None)
    if "audio_trim_trail_ms" in patch:
        sets.append("audio_trim_trail_ms = ?")
        val = patch["audio_trim_trail_ms"]
        params.append(int(val) if val is not None else None)
    if not sets:
        return get_capture(cid)
    params.append(cid)
    conn = _require_conn()
    with _lock:
        cur = conn.execute(
            f"UPDATE captures SET {', '.join(sets)} WHERE id = ?", params,
        )
        if cur.rowcount == 0:
            return None
    row = get_capture(cid)
    # Lazy import — captures_merge_proposer imports this module; a top-
    # level import would loop. Safe to call with None on a missing row.
    try:
        import captures_merge_proposer
        captures_merge_proposer.invalidate(row.get("user_id") if row else None)
    except Exception:
        pass
    return row


# ---------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------

def delete_capture(cid: str) -> bool:
    """Drop a single capture row + its audio file. If the capture was
    a member of a group, the group is auto-dissolved — a merged
    sample is reconstructable only from its complete member set, so
    losing a member makes the group's WAV a frozen artifact of a
    state that can never be rebuilt. Returns True if the row existed."""
    conn = _require_conn()
    with _lock:
        row = conn.execute(
            "SELECT audio_relpath, audio_trimmed_relpath, sample_id, user_id"
            " FROM captures WHERE id = ?",
            (cid,),
        ).fetchone()
        if row is None:
            return False
        sid = row["sample_id"]
        uid = row["user_id"]
        conn.execute("DELETE FROM captures WHERE id = ?", (cid,))
    try:
        _safe_unlink(abs_audio_path(row["audio_relpath"]))
    except ValueError:
        pass
    # Also clean up the trimmed companion file if one was produced.
    trimmed = row["audio_trimmed_relpath"]
    if trimmed:
        try:
            _safe_unlink(abs_audio_path(trimmed))
        except ValueError:
            pass
    if sid:
        try:
            import capture_samples_store
            capture_samples_store.dissolve_sample(sid)
            logger.info(
                "[captures] auto-dissolved sid=%s after member %s delete",
                sid[:8], cid[:8],
            )
        except Exception as _e:
            logger.warning(
                "[captures] auto-dissolve of sid=%s failed: %s",
                sid[:8], _e,
            )
    try:
        import captures_merge_proposer
        captures_merge_proposer.invalidate(uid)
    except Exception:
        pass
    logger.info("[captures] deleted id=%s", cid[:8])
    return True


def clear_all(reporter_host: str = "") -> int:
    """Wipe every row + every audio file under the captures dir.
    WARNING-logs the count + caller host for audit. Returns the count
    of captures deleted.

    Also wipes capture_samples — the filesystem rmtree below clears
    the groups/ subtree along with the hex fanout dirs, so leaving
    capture_samples rows alive would leave them pointing at vanished
    files (the user's '404 merged audio missing' symptom)."""
    conn = _require_conn()
    audio_dir = _require_audio_dir()
    n_groups = 0
    try:
        import capture_samples_store
        n_groups = capture_samples_store.clear_all_samples()
    except Exception as _e:
        logger.warning(
            "[captures] clear_all_samples failed (continuing): %s", _e,
        )
    with _lock:
        row = conn.execute("SELECT COUNT(*) FROM captures").fetchone()
        n = int(row[0]) if row else 0
        conn.execute("DELETE FROM captures")
        conn.execute("VACUUM")
    if os.path.isdir(audio_dir):
        for sub in os.listdir(audio_dir):
            sub_path = os.path.join(audio_dir, sub)
            if os.path.isdir(sub_path):
                shutil.rmtree(sub_path, ignore_errors=True)
    logger.warning(
        "[captures] admin from %s cleared %d captures + %d groups",
        reporter_host or "<unknown>", n, n_groups,
    )
    return n


# ---------------------------------------------------------------------
# Reconciliation & retention
# ---------------------------------------------------------------------

def reconcile_on_startup() -> tuple[int, int]:
    """Audit the store on startup. Two passes:

      1. For each row whose audio file is missing, set status=
         'audio_missing' (never auto-delete — the admin needs to see the
         loss in the UI).
      2. For each file under CAPTURES_DIR with no matching row, unlink
         (orphaned files from prior crashes / clear-all races).

    Returns (rows_marked_missing, files_unlinked)."""
    conn = _require_conn()
    audio_dir = _require_audio_dir()
    rows_marked = 0
    files_unlinked = 0

    known_paths: set[str] = set()
    with _lock:
        cur = conn.execute(
            "SELECT id, audio_relpath, audio_trimmed_relpath, status"
            " FROM captures",
        )
        for r in cur.fetchall():
            try:
                abs_p = abs_audio_path(r["audio_relpath"])
            except ValueError:
                continue
            if os.path.isfile(abs_p):
                known_paths.add(os.path.abspath(abs_p))
            else:
                if r["status"] != "audio_missing":
                    conn.execute(
                        "UPDATE captures SET status='audio_missing' WHERE id=?",
                        (r["id"],),
                    )
                    rows_marked += 1
            # Trimmed companion (if present) is also a known artifact;
            # without this it gets unlinked as "orphan" on every restart.
            trimmed_rel = r["audio_trimmed_relpath"]
            if trimmed_rel:
                try:
                    t_abs = abs_audio_path(trimmed_rel)
                except ValueError:
                    continue
                if os.path.isfile(t_abs):
                    known_paths.add(os.path.abspath(t_abs))

    # Walk the audio directory and unlink anything not in known_paths.
    # The `groups/` subtree is owned by capture_samples_store and has its
    # own reconcile pass — prune it here so we never see a merged-group
    # WAV as an orphan and delete it on every restart. Hex-fanout dirs
    # are "00".."ff", none of which equal "groups", so this is safe.
    if os.path.isdir(audio_dir):
        for root, dirs, files in os.walk(audio_dir):
            if root == audio_dir and "groups" in dirs:
                dirs.remove("groups")
            for name in files:
                if name.endswith(".tmp"):
                    # Crash mid-write — delete the partial.
                    if _safe_unlink(os.path.join(root, name)):
                        files_unlinked += 1
                    continue
                p = os.path.abspath(os.path.join(root, name))
                if p not in known_paths:
                    if _safe_unlink(p):
                        files_unlinked += 1

    logger.info(
        "[captures] reconcile: %d rows marked audio_missing, "
        "%d orphan files removed",
        rows_marked, files_unlinked,
    )
    return rows_marked, files_unlinked


def sweep_retention() -> int:
    """Delete rows + audio files older than cfg.CAPTURES_RETENTION_DAYS.
    Returns count deleted. Group members are excluded — see
    _drop_oldest_with_status for the rationale."""
    try:
        import config as cfg
        days = int(getattr(cfg, "CAPTURES_RETENTION_DAYS", 0))
    except Exception:
        return 0
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    conn = _require_conn()
    with _lock:
        rows = conn.execute(
            "SELECT id, audio_relpath, audio_trimmed_relpath FROM captures"
            " WHERE created_ts < ? AND sample_id IS NULL",
            (cutoff,),
        ).fetchall()
        if not rows:
            return 0
        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" * len(ids))
        conn.execute(
            f"DELETE FROM captures WHERE id IN ({placeholders})", ids,
        )
    for r in rows:
        try:
            _safe_unlink(abs_audio_path(r["audio_relpath"]))
        except ValueError:
            pass
        trimmed = r["audio_trimmed_relpath"]
        if trimmed:
            try:
                _safe_unlink(abs_audio_path(trimmed))
            except ValueError:
                pass
    logger.warning(
        "[captures] retention sweep deleted %d rows older than %d days",
        len(rows), days,
    )
    return len(rows)
