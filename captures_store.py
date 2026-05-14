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
_db_path: str | None = None
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

_SCHEMA = """
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
  words_json      TEXT NOT NULL,
  segments_json   TEXT NOT NULL DEFAULT '[]',
  corrected_text  TEXT NOT NULL DEFAULT '',
  corrections_json TEXT NOT NULL DEFAULT '[]',
  admin_notes     TEXT NOT NULL DEFAULT '',
  status          TEXT NOT NULL DEFAULT 'new',
  reviewed_ts     REAL,
  user_id         TEXT,
  group_id        TEXT,
  group_order     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_captures_created    ON captures(created_ts DESC);
CREATE INDEX IF NOT EXISTS idx_captures_status     ON captures(status);
CREATE INDEX IF NOT EXISTS idx_captures_request_id ON captures(request_id);
CREATE INDEX IF NOT EXISTS idx_captures_user       ON captures(user_id, created_ts DESC);
CREATE INDEX IF NOT EXISTS idx_captures_group      ON captures(group_id, group_order);
"""

# Migrations for installs that predate the user_id / group_id columns.
# SQLite ALTER ADD COLUMN is cheap and idempotent (we catch OperationalError
# to skip the no-op repeat call). Run after _SCHEMA so the indexes are also
# present even if the table existed before.
_MIGRATIONS = (
    "ALTER TABLE captures ADD COLUMN user_id TEXT",
    "ALTER TABLE captures ADD COLUMN group_id TEXT",
    "ALTER TABLE captures ADD COLUMN group_order INTEGER",
)


# ---------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------

def init(db_path: str, audio_dir: str) -> None:
    """Open the SQLite DB (WAL) and ensure the audio dir exists. Idempotent:
    safe to call on every startup. Call once before any other function."""
    global _conn, _db_path, _audio_dir
    _db_path = db_path
    _audio_dir = audio_dir
    db_dir = os.path.dirname(os.path.abspath(db_path)) or "."
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    _conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL;")
    _conn.execute("PRAGMA synchronous=NORMAL;")
    _conn.executescript(_SCHEMA)
    for stmt in _MIGRATIONS:
        try:
            _conn.execute(stmt)
        except sqlite3.OperationalError:
            # Column already present — idempotent.
            pass


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
    to short-circuit the capture decision when the store is at its cap."""
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

    # fsync the data before swap so a crash between rename and SQLite
    # insert doesn't leave a zero-length file on disk.
    try:
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass
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
    words_t = _truncate_json(words or [], _CAP_WORDS_JSON)
    segments_t = _truncate_json(segments or [], _CAP_SEGMENTS_JSON)

    try:
        conn = _require_conn()
        with _lock:
            conn.execute(
                "INSERT INTO captures ("
                " id, created_ts, request_id, model, language,"
                " duration_seconds, audio_relpath, audio_format,"
                " raw, final, words_json, segments_json,"
                " corrected_text, corrections_json, admin_notes,"
                " status, reviewed_ts, user_id, group_id, group_order"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cid, now, request_id, model or "", language or "",
                    float(duration_seconds or 0.0), relpath, "wav",
                    raw_t, final_t, words_t, segments_t,
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

    if mb_cap >= 1:
        # Compute current total size by walking rows. Cheap because we
        # only call getsize when over the row cap path.
        byte_cap = mb_cap * 1024 * 1024
        total_bytes = _total_audio_bytes(conn)
        if total_bytes > byte_cap:
            for status in _EVICTION_ORDER:
                if total_bytes <= byte_cap:
                    break
                total_bytes -= _drop_oldest_by_bytes(
                    conn, status, total_bytes - byte_cap,
                )
            evicted += 1  # log marker; precise count is in the helper

    if evicted:
        logger.info(
            "[captures] evicted to cap: row_cap=%d, mb_cap=%d", row_cap, mb_cap,
        )


def _total_audio_bytes(conn: sqlite3.Connection) -> int:
    total = 0
    for row in conn.execute("SELECT audio_relpath FROM captures"):
        try:
            total += os.path.getsize(abs_audio_path(row["audio_relpath"]))
        except (OSError, ValueError):
            continue
    return total


def _drop_oldest_with_status(
    conn: sqlite3.Connection, status: str, limit: int,
) -> int:
    """Delete up to `limit` oldest rows of a given status, also unlinking
    audio files. Returns the count deleted."""
    if limit <= 0:
        return 0
    rows = conn.execute(
        "SELECT id, audio_relpath FROM captures WHERE status = ?"
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
    return len(rows)


def _drop_oldest_by_bytes(
    conn: sqlite3.Connection, status: str, bytes_needed: int,
) -> int:
    """Delete oldest rows of a given status until `bytes_needed` bytes
    are freed. Returns bytes actually freed."""
    if bytes_needed <= 0:
        return 0
    rows = conn.execute(
        "SELECT id, audio_relpath FROM captures WHERE status = ?"
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
        f"SELECT id, created_ts, request_id, model, language,"
        f" duration_seconds, audio_relpath, audio_format,"
        f" raw, final, corrected_text, corrections_json, admin_notes,"
        f" status, reviewed_ts, user_id, group_id, group_order"
        f" FROM captures{where}"
        f" ORDER BY created_ts DESC LIMIT ?",
        params,
    )
    return [_row_to_dict(r, include_words=False) for r in cur.fetchall()]


def iter_captures_for_export(
    *, status: str | None = None,
):
    """Generator yielding capture rows (full payload incl. words) in
    deterministic order. Used by the export endpoint to stream a tarball
    without holding the whole result set in memory. Each iteration takes
    a short lock for the SELECT, then releases it; subsequent rows are
    fetched lazily via the cursor."""
    conn = _require_conn()
    where = ""
    params: list[Any] = []
    if status and status != "all":
        where = " WHERE status = ?"
        params.append(status)
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
    """Cross-link from /reports: find captures matching a request_id.
    Returns a list (multiple captures per request_id is rare but allowed
    if sampling+cap-eviction happens to keep multiple)."""
    conn = _require_conn()
    cur = conn.execute(
        "SELECT id, created_ts, model, duration_seconds, status,"
        " audio_format, audio_relpath"
        " FROM captures WHERE request_id = ? ORDER BY created_ts DESC",
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
    """Apply a partial update. Allowed fields: corrected_text, corrections,
    admin_notes, status. Returns the updated row or None if not found."""
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
    # If the capture belongs to a group, recheck the audio-content hash;
    # transcript-only edits don't change audio so the hash will match and
    # nothing flips, but a future audio-edit code path is now wired.
    try:
        import capture_groups_store
        capture_groups_store.recompute_stale_for_member(cid)
    except Exception:
        pass
    return get_capture(cid)


# ---------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------

def delete_capture(cid: str) -> bool:
    """Drop a single capture row + its audio file. Returns True if the
    row existed."""
    conn = _require_conn()
    with _lock:
        row = conn.execute(
            "SELECT audio_relpath FROM captures WHERE id = ?", (cid,),
        ).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM captures WHERE id = ?", (cid,))
    try:
        _safe_unlink(abs_audio_path(row["audio_relpath"]))
    except ValueError:
        pass
    logger.info("[captures] deleted id=%s", cid[:8])
    return True


def clear_all(reporter_host: str = "") -> int:
    """Wipe every row + every audio file under the captures dir.
    WARNING-logs the count + caller host for audit. Returns the count
    deleted."""
    conn = _require_conn()
    audio_dir = _require_audio_dir()
    with _lock:
        row = conn.execute("SELECT COUNT(*) FROM captures").fetchone()
        n = int(row[0]) if row else 0
        conn.execute("DELETE FROM captures")
        conn.execute("VACUUM")
    # Remove every file under the captures dir. We could be more surgical
    # and only delete files we have rows for, but the row table is empty
    # at this point — anything still on disk is orphaned, drop it all.
    if os.path.isdir(audio_dir):
        for sub in os.listdir(audio_dir):
            sub_path = os.path.join(audio_dir, sub)
            if os.path.isdir(sub_path):
                shutil.rmtree(sub_path, ignore_errors=True)
    logger.warning(
        "[captures] admin from %s cleared %d captures",
        reporter_host or "<unknown>", n,
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
            "SELECT id, audio_relpath, status FROM captures",
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

    # Walk the audio directory and unlink anything not in known_paths.
    if os.path.isdir(audio_dir):
        for root, _dirs, files in os.walk(audio_dir):
            for name in files:
                if name.endswith(".tmp"):
                    # Crash mid-write — delete the partial.
                    _safe_unlink(os.path.join(root, name))
                    files_unlinked += 1
                    continue
                p = os.path.abspath(os.path.join(root, name))
                if p not in known_paths:
                    _safe_unlink(p)
                    files_unlinked += 1

    if rows_marked or files_unlinked:
        logger.warning(
            "[captures] reconcile: %d rows marked audio_missing, "
            "%d orphan files removed",
            rows_marked, files_unlinked,
        )
    return rows_marked, files_unlinked


def sweep_retention() -> int:
    """Delete rows + audio files older than cfg.CAPTURES_RETENTION_DAYS.
    Returns count deleted."""
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
            "SELECT id, audio_relpath FROM captures WHERE created_ts < ?",
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
    logger.warning(
        "[captures] retention sweep deleted %d rows older than %d days",
        len(rows), days,
    )
    return len(rows)
