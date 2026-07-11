"""Durable per-account store for the desktop client's synced settings blob.

SQLite (stdlib) in WAL mode — single-file, crash-safe. Lives at
cfg.CLIENT_SETTINGS_DB (defaults to client_settings.local.sqlite3 alongside
config.local.json).

The blob is OPAQUE, SENSITIVE client JSON: the desktop app puts its whole
synced configuration in it, which by user choice may include the client's
own backend API keys. Never log blob contents — INFO/WARNING lines carry
only a shortened user tag, profile, version, byte size, and device label.
Plaintext on disk; whole-disk encryption is the deployment's responsibility.

One row per (user_id, profile). v1 clients always use profile='' — the
column exists so named settings-sets can land later as an additive change.

Concurrency: optimistic versioning. A row's `version` starts at 1 and bumps
on every successful write; writers echo the version they last saw
(`base_version`) and the compare-and-swap is a SINGLE SQL statement
(INSERT-or-IntegrityError for create, UPDATE ... WHERE version=? for
update), so it stays atomic across threads AND across uvicorn workers if an
operator ever sets SERVER_WORKERS>1 — the module `_lock` alone would only
serialise within one process. `_lock` still wraps write+read-back so the
returned row reflects this write within the process.
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

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


class StoreUnavailable(RuntimeError):
    """The store was never initialized (init_db failed or wasn't called).
    Routes map this to a 503 with a pointer at the server log — the
    alternative is a bare 500 that tells the operator nothing."""

# Size caps — applied server-side before insert. The blob is opaque JSON, so
# an over-cap payload is REJECTED (ValueError → route maps to 413), never
# truncated: cutting JSON mid-document would hand every other device a
# corrupt config. 512 KB fits dozens of backends/profiles with headroom.
# The device label is informational metadata → truncation is harmless.
_CAP_BLOB = 512_000
_CAP_DEVICE = 200

_SCHEMA = """
CREATE TABLE IF NOT EXISTS client_settings (
  user_id    TEXT    NOT NULL,
  profile    TEXT    NOT NULL DEFAULT '',
  blob       TEXT    NOT NULL,
  version    INTEGER NOT NULL,
  updated_at REAL    NOT NULL,
  device     TEXT,
  PRIMARY KEY (user_id, profile)
);
"""


def init_db(path: str) -> None:
    """Open (or create) the client-settings DB at `path` in WAL mode.
    Idempotent: safe to call on every startup; the schema-CREATE uses
    IF NOT EXISTS. Call before any other function in this module."""
    global _conn
    dst_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dst_dir, exist_ok=True)
    _conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL;")
    _conn.execute("PRAGMA synchronous=NORMAL;")
    _conn.executescript(_SCHEMA)
    _ensure_columns(_conn)


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Additive column migrations (PRAGMA table_info + ALTER TABLE), the
    same convention as api_keys_store. Nothing to migrate yet — this hook
    exists so a future column (e.g. blob_sha) lands the standard way."""


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise StoreUnavailable(
            "client_settings_store.init_db() was not called before use."
        )
    return _conn


def _uid_tag(user_id: str) -> str:
    """Loggable identity: the open-mode sentinel verbatim (it's not an id),
    otherwise a shortened prefix so full account ids stay out of the log."""
    return user_id if user_id.startswith("(") else user_id[:8]


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    try:
        d["blob"] = json.loads(d["blob"])
    except (TypeError, ValueError):
        # A row we wrote can't be unparseable, but never let a corrupt DB
        # take the endpoint down — surface it as an empty object.
        d["blob"] = {}
    return d


def get(user_id: str, profile: str = "") -> dict[str, Any] | None:
    """Return the row dict {user_id, profile, blob(parsed), version,
    updated_at, device} or None when nothing is stored."""
    conn = _require_conn()
    row = conn.execute(
        "SELECT * FROM client_settings WHERE user_id = ? AND profile = ?",
        (user_id, profile),
    ).fetchone()
    return _row_to_dict(row)


def list_meta() -> list[dict[str, Any]]:
    """Every row's METADATA — {user_id, profile, version, updated_at,
    device, bytes} — deliberately WITHOUT the blob, so admin surfaces
    (the keys page's per-account chip/drawer) can list what's stored
    without the sensitive contents ever leaving this module in bulk.
    `bytes` is the stored UTF-8 size (CAST to BLOB: TEXT length() would
    count characters)."""
    conn = _require_conn()
    rows = conn.execute(
        "SELECT user_id, profile, version, updated_at, device,"
        " length(CAST(blob AS BLOB)) AS bytes FROM client_settings"
    ).fetchall()
    return [dict(r) for r in rows]


def put(
    user_id: str,
    blob: Any,
    base_version: int,
    *,
    device: str | None = None,
    profile: str = "",
) -> tuple[bool, dict[str, Any] | None]:
    """Optimistic-concurrency upsert.

    Returns (ok, state):
      ok=True  — write landed; state is the NEW row (version bumped).
      ok=False — `base_version` was stale; state is the CURRENT row for
                 the 409 body (or None if the row vanished mid-flight).

    Raises ValueError if the serialized blob exceeds _CAP_BLOB.
    Force-push needs no special flag: send the version just fetched and
    the CAS matches unless someone else wrote in between — which is
    exactly the conflict the versioning exists to catch.
    """
    blob_json = json.dumps(blob, ensure_ascii=False, separators=(",", ":"))
    if len(blob_json.encode("utf-8")) > _CAP_BLOB:
        raise ValueError("blob too large")
    dev = str(device)[:_CAP_DEVICE] if device else None
    now = time.time()
    conn = _require_conn()
    with _lock:
        if int(base_version) <= 0:
            # Bootstrap create. A PK collision means a row already exists,
            # i.e. the caller's "nothing stored yet" view is stale → conflict.
            try:
                conn.execute(
                    "INSERT INTO client_settings"
                    " (user_id, profile, blob, version, updated_at, device)"
                    " VALUES (?,?,?,1,?,?)",
                    (user_id, profile, blob_json, now, dev),
                )
            except sqlite3.IntegrityError:
                return False, get(user_id, profile)
            logger.info(
                "[client-settings] created user=%s profile=%r v=1 bytes=%d device=%r",
                _uid_tag(user_id), profile, len(blob_json), dev,
            )
            return True, get(user_id, profile)

        cur = conn.execute(
            "UPDATE client_settings"
            " SET blob = ?, version = version + 1, updated_at = ?, device = ?"
            " WHERE user_id = ? AND profile = ? AND version = ?",
            (blob_json, now, dev, user_id, profile, int(base_version)),
        )
        if cur.rowcount == 0:
            # Stale base_version, or the row was deleted (then the caller's
            # next look shows blob=None via GET / the 409 body's state=None
            # is normalized by the route).
            return False, get(user_id, profile)
        new_row = get(user_id, profile)
        logger.info(
            "[client-settings] updated user=%s profile=%r v=%s bytes=%d device=%r",
            _uid_tag(user_id), profile,
            new_row["version"] if new_row else "?", len(blob_json), dev,
        )
        return True, new_row


def force_put(
    user_id: str,
    blob: Any,
    *,
    device: str | None = None,
    profile: str = "",
) -> dict[str, Any]:
    """Unconditional admin write (the WebUI's import/restore path) — no
    base_version. An atomic upsert bumps `version` past whatever is
    stored, so every device's next CAS push conflicts and its next pull
    sees a newer server copy: the imported settings propagate through
    the devices' normal merge path with no device-side changes.

    Raises ValueError if the serialized blob exceeds _CAP_BLOB (the
    route maps it to 413, same as put())."""
    blob_json = json.dumps(blob, ensure_ascii=False, separators=(",", ":"))
    if len(blob_json.encode("utf-8")) > _CAP_BLOB:
        raise ValueError("blob too large")
    dev = str(device)[:_CAP_DEVICE] if device else None
    now = time.time()
    conn = _require_conn()
    with _lock:
        conn.execute(
            "INSERT INTO client_settings"
            " (user_id, profile, blob, version, updated_at, device)"
            " VALUES (?,?,?,1,?,?)"
            " ON CONFLICT(user_id, profile) DO UPDATE SET"
            " blob = excluded.blob,"
            " version = client_settings.version + 1,"
            " updated_at = excluded.updated_at,"
            " device = excluded.device",
            (user_id, profile, blob_json, now, dev),
        )
        row = get(user_id, profile)
    logger.info(
        "[client-settings] imported user=%s profile=%r v=%s bytes=%d device=%r",
        _uid_tag(user_id), profile,
        row["version"] if row else "?", len(blob_json), dev,
    )
    assert row is not None  # the upsert we just did can't vanish under _lock
    return row


def delete(user_id: str, profile: str = "") -> bool:
    """Remove the row. Returns True if one was deleted. After a delete the
    store reads as version 0 again; a device still holding version N gets a
    409 on its next PUT (base N no longer matches), correctly surfacing the
    deletion instead of silently resurrecting the blob."""
    conn = _require_conn()
    with _lock:
        cur = conn.execute(
            "DELETE FROM client_settings WHERE user_id = ? AND profile = ?",
            (user_id, profile),
        )
        deleted = cur.rowcount > 0
    if deleted:
        logger.info("[client-settings] deleted user=%s profile=%r",
                    _uid_tag(user_id), profile)
    return deleted
