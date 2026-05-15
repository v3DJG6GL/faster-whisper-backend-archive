"""Durable store for API keys + users — the identity layer that gates
the transcription endpoint and the WebUI.

Storage layout:

  cfg.API_KEYS_DB — SQLite (WAL) with two tables:
    users     — { id, username, is_admin, created_ts, revoked_ts }
    api_keys  — { id, user_id, key_hash, key_prefix, key_last4,
                  label, created_ts, revoked_ts, last_used_ts }

Keys at rest are SHA-256(raw_key) hex. Raw keys are shown ONCE on
creation and never retrievable. High-entropy random keys (256-bit) make
slow password hashes pointless — OWASP guidance.

Lookup is O(1) via an in-memory `_KEY_INDEX: dict[hash_hex, user_row]`
rebuilt on any mutation. For ≤10 users this is trivial; if the design
ever scales we can swap to a per-request DB query without changing the
public surface.

`last_used_ts` is debounced 60 s per key (one DB write/min/key max).

`get_open_mode_user()` returns a synthetic admin used when no real
admin key exists — the server starts unprotected with a prominent
WARNING banner so the operator can bootstrap.
"""
from __future__ import annotations

import logging
import os
import secrets
import sqlite3
import threading
import time
import uuid
from hashlib import sha256
from typing import Any

logger = logging.getLogger("whisper-api")

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_db_path: str | None = None

# In-memory hash → user row index. Rebuilt on mutation.
_KEY_INDEX: dict[str, dict[str, Any]] = {}

# Debounce window for last_used_ts writes (seconds).
_LAST_USED_DEBOUNCE_S = 60.0

# Raw-key format. Prefix is cosmetic but aids leak-detection (Stripe pattern).
KEY_PREFIX = "wk_"
_KEY_BYTES = 32   # secrets.token_urlsafe(32) → 43-char base64url, 256-bit entropy

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id          TEXT PRIMARY KEY,
  username    TEXT NOT NULL UNIQUE,
  is_admin    INTEGER NOT NULL DEFAULT 0,
  created_ts  REAL NOT NULL,
  revoked_ts  REAL
);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(revoked_ts);

CREATE TABLE IF NOT EXISTS api_keys (
  id            TEXT PRIMARY KEY,
  user_id       TEXT NOT NULL REFERENCES users(id),
  key_hash      TEXT NOT NULL UNIQUE,
  key_prefix    TEXT NOT NULL,
  key_last4     TEXT NOT NULL,
  label         TEXT NOT NULL DEFAULT '',
  created_ts    REAL NOT NULL,
  revoked_ts    REAL,
  last_used_ts  REAL
);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(revoked_ts);
CREATE INDEX IF NOT EXISTS idx_api_keys_user   ON api_keys(user_id);
"""


# ---------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------

def init_db(db_path: str) -> None:
    """Open the SQLite DB (WAL) and ensure the schema exists. Idempotent.
    Builds the in-memory key index from active rows."""
    global _conn, _db_path
    _db_path = db_path
    db_dir = os.path.dirname(os.path.abspath(db_path)) or "."
    os.makedirs(db_dir, exist_ok=True)
    _conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL;")
    _conn.execute("PRAGMA synchronous=NORMAL;")
    _conn.executescript(_SCHEMA)
    _rebuild_index_locked()


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("api_keys_store.init_db() was not called before use.")
    return _conn


# ---------------------------------------------------------------------
# Hash + key generation
# ---------------------------------------------------------------------

def hash_key(raw_key: str) -> str:
    """SHA-256 hex of the raw key bytes (UTF-8)."""
    return sha256(raw_key.encode("utf-8")).hexdigest()


def generate_raw_key() -> str:
    """Generate a fresh `wk_<43-char base64url>` key (256-bit entropy)."""
    return KEY_PREFIX + secrets.token_urlsafe(_KEY_BYTES)


def _split_display_parts(raw_key: str) -> tuple[str, str]:
    """Return (key_prefix, key_last4) for display.

    prefix is the first 8 chars (e.g. `wk_a1b2`); last4 is the trailing
    4 chars. Stored in plaintext so the admin UI can render
    `wk_a1b2…d4e5` without retaining the raw key."""
    prefix = raw_key[:8]
    last4 = raw_key[-4:] if len(raw_key) >= 4 else raw_key
    return prefix, last4


# ---------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------

def _row_to_user_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "username": row["username"],
        "is_admin": bool(row["is_admin"]),
        "created_ts": float(row["created_ts"]),
        "revoked_ts": float(row["revoked_ts"]) if row["revoked_ts"] is not None else None,
    }


def _row_to_key_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "key_prefix": row["key_prefix"],
        "key_last4": row["key_last4"],
        "label": row["label"] or "",
        "created_ts": float(row["created_ts"]),
        "revoked_ts": float(row["revoked_ts"]) if row["revoked_ts"] is not None else None,
        "last_used_ts": float(row["last_used_ts"]) if row["last_used_ts"] is not None else None,
    }


# ---------------------------------------------------------------------
# In-memory index
# ---------------------------------------------------------------------

def _rebuild_index_locked() -> None:
    """Rebuild _KEY_INDEX from the DB. Caller holds _lock (or is in init)."""
    global _KEY_INDEX
    conn = _require_conn()
    rows = conn.execute(
        "SELECT k.key_hash, k.id AS key_id, u.id AS user_id, u.username, u.is_admin"
        " FROM api_keys k JOIN users u ON u.id = k.user_id"
        " WHERE k.revoked_ts IS NULL AND u.revoked_ts IS NULL"
    ).fetchall()
    idx: dict[str, dict[str, Any]] = {}
    for r in rows:
        idx[r["key_hash"]] = {
            "key_id": r["key_id"],
            "user_id": r["user_id"],
            "username": r["username"],
            "is_admin": bool(r["is_admin"]),
        }
    _KEY_INDEX = idx


def _invalidate_index() -> None:
    """Mark the index for rebuild; called after any mutation."""
    with _lock:
        _rebuild_index_locked()


# ---------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------

def create_user(username: str, is_admin: bool) -> str:
    """Create a user record. Returns the new user_id. Raises ValueError
    if username is blank or a duplicate."""
    username = (username or "").strip()
    if not username:
        raise ValueError("username is required")
    if len(username) > 128:
        raise ValueError("username too long (max 128 chars)")
    uid = uuid.uuid4().hex
    now = time.time()
    conn = _require_conn()
    try:
        with _lock:
            conn.execute(
                "INSERT INTO users (id, username, is_admin, created_ts, revoked_ts)"
                " VALUES (?,?,?,?,NULL)",
                (uid, username, 1 if is_admin else 0, now),
            )
            _rebuild_index_locked()
    except sqlite3.IntegrityError as e:
        if "UNIQUE" in str(e):
            raise ValueError(f"username {username!r} already exists") from e
        raise
    logger.info(
        "[auth] user created id=%s username=%s admin=%s",
        uid[:8], username, is_admin,
    )
    return uid


def list_users(*, include_revoked: bool = False) -> list[dict[str, Any]]:
    conn = _require_conn()
    if include_revoked:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY created_ts DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM users WHERE revoked_ts IS NULL ORDER BY created_ts DESC"
        ).fetchall()
    return [_row_to_user_dict(r) for r in rows]


def get_user(user_id: str) -> dict[str, Any] | None:
    conn = _require_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,),
    ).fetchone()
    return _row_to_user_dict(row) if row else None


def get_username(user_id: "str | None") -> "str | None":
    """Return the configured username for a user_id, or None if the user
    no longer exists / has been revoked / wasn't authenticated (open
    mode). Cheap render-time lookup — the captures, reports, and
    quick-config routes all call this to swap the truncated user_id
    pill for the readable username."""
    if not user_id or user_id == "(open-mode)":
        return None
    u = get_user(user_id)
    return u.get("username") if u else None


def revoke_user(user_id: str) -> None:
    """Soft-delete a user and revoke all their keys. Last-admin guard
    must be checked by the caller — this function unconditionally
    revokes."""
    now = time.time()
    conn = _require_conn()
    with _lock:
        conn.execute(
            "UPDATE users SET revoked_ts = ? WHERE id = ? AND revoked_ts IS NULL",
            (now, user_id),
        )
        conn.execute(
            "UPDATE api_keys SET revoked_ts = ?"
            " WHERE user_id = ? AND revoked_ts IS NULL",
            (now, user_id),
        )
        _rebuild_index_locked()
    logger.info("[auth] user revoked id=%s", user_id[:8])


def count_active_admins() -> int:
    """Return the number of users with is_admin=1 AND revoked_ts IS NULL."""
    conn = _require_conn()
    row = conn.execute(
        "SELECT COUNT(*) FROM users WHERE is_admin = 1 AND revoked_ts IS NULL"
    ).fetchone()
    return int(row[0]) if row else 0


def count_active_admin_keys() -> int:
    """Active admin keys = keys whose user is_admin AND both rows are active.
    Used by the last-admin-key guard on revoke."""
    conn = _require_conn()
    row = conn.execute(
        "SELECT COUNT(*) FROM api_keys k JOIN users u ON u.id = k.user_id"
        " WHERE k.revoked_ts IS NULL AND u.revoked_ts IS NULL AND u.is_admin = 1"
    ).fetchone()
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------
# Key CRUD
# ---------------------------------------------------------------------

def create_key(user_id: str, *, label: str = "") -> tuple[str, dict[str, Any]]:
    """Generate a new raw key, store its hash + display parts, return
    `(raw_key, record)`. The raw key is the ONLY way the caller will
    ever see it — display it once to the operator and discard."""
    label = (label or "").strip()
    if len(label) > 128:
        raise ValueError("label too long (max 128 chars)")
    user = get_user(user_id)
    if user is None or user["revoked_ts"] is not None:
        raise ValueError("user not found or revoked")
    raw_key = generate_raw_key()
    h = hash_key(raw_key)
    kp, k4 = _split_display_parts(raw_key)
    kid = uuid.uuid4().hex
    now = time.time()
    conn = _require_conn()
    with _lock:
        conn.execute(
            "INSERT INTO api_keys"
            " (id, user_id, key_hash, key_prefix, key_last4, label,"
            "  created_ts, revoked_ts, last_used_ts)"
            " VALUES (?,?,?,?,?,?,?,NULL,NULL)",
            (kid, user_id, h, kp, k4, label, now),
        )
        _rebuild_index_locked()
    logger.info(
        "[auth] key created kid=%s user=%s prefix=%s label=%s",
        kid[:8], user_id[:8], kp, label or "(no label)",
    )
    rec = {
        "id": kid,
        "user_id": user_id,
        "label": label,
        "key_prefix": kp,
        "key_last4": k4,
        "created_ts": now,
        "revoked_ts": None,
        "last_used_ts": None,
    }
    return raw_key, rec


def list_keys(user_id: str | None = None, *, include_revoked: bool = False) -> list[dict[str, Any]]:
    conn = _require_conn()
    clauses = []
    args: list[Any] = []
    if user_id is not None:
        clauses.append("user_id = ?")
        args.append(user_id)
    if not include_revoked:
        clauses.append("revoked_ts IS NULL")
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM api_keys{where} ORDER BY created_ts DESC", args,
    ).fetchall()
    return [_row_to_key_dict(r) for r in rows]


def get_key(key_id: str) -> dict[str, Any] | None:
    conn = _require_conn()
    row = conn.execute(
        "SELECT * FROM api_keys WHERE id = ?", (key_id,),
    ).fetchone()
    return _row_to_key_dict(row) if row else None


def revoke_key(key_id: str) -> None:
    """Soft-revoke. Last-admin-key guard is the caller's responsibility."""
    now = time.time()
    conn = _require_conn()
    with _lock:
        conn.execute(
            "UPDATE api_keys SET revoked_ts = ?"
            " WHERE id = ? AND revoked_ts IS NULL",
            (now, key_id),
        )
        _rebuild_index_locked()
    logger.info("[auth] key revoked kid=%s", key_id[:8])


# ---------------------------------------------------------------------
# Auth lookup
# ---------------------------------------------------------------------

def lookup_by_raw_key(raw_key: str) -> dict[str, Any] | None:
    """Resolve a raw bearer key to its active user record. Returns
    `{key_id, user_id, username, is_admin}` or None on miss.

    O(1) via in-memory hash index. Touches last_used_ts (debounced)."""
    if not raw_key:
        return None
    h = hash_key(raw_key)
    rec = _KEY_INDEX.get(h)
    if rec is None:
        return None
    _touch_last_used_debounced(rec["key_id"])
    return dict(rec)


def _touch_last_used_debounced(key_id: str) -> None:
    """Update last_used_ts at most once per _LAST_USED_DEBOUNCE_S per key.
    Read of last_used_ts goes to the index (which doesn't track it) → we
    cache a separate small dict here."""
    now = time.time()
    last = _LAST_USED_CACHE.get(key_id, 0.0)
    if now - last < _LAST_USED_DEBOUNCE_S:
        return
    _LAST_USED_CACHE[key_id] = now
    conn = _require_conn()
    try:
        with _lock:
            conn.execute(
                "UPDATE api_keys SET last_used_ts = ? WHERE id = ?",
                (now, key_id),
            )
    except sqlite3.Error:
        # Non-fatal: last_used is a stats field, not auth-critical.
        pass


_LAST_USED_CACHE: dict[str, float] = {}


# ---------------------------------------------------------------------
# Open-mode synthetic admin
# ---------------------------------------------------------------------

OPEN_MODE_USER: dict[str, Any] = {
    "key_id": "(open-mode)",
    "user_id": "(open-mode)",
    "username": "(open mode)",
    "is_admin": True,
}


def is_locked_down() -> bool:
    """Server is locked down iff at least one active admin key exists.
    Open mode (return False) lets every request through as the synthetic
    admin so the operator can bootstrap. A periodic WARNING is logged
    by the auth module while open."""
    try:
        return count_active_admin_keys() >= 1
    except Exception:
        # If the DB hasn't been opened yet treat as open — main.py
        # decides ordering.
        return False
