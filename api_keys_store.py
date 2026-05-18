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

import json
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


class LastAdminError(Exception):
    """Raised when revoke_user / revoke_key would leave zero active
    admins (or active admin keys). The guard is enforced atomically
    under _lock so two concurrent revokes can't both pass a count==2
    check and double-revoke into a lockout."""


_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

# In-memory hash → user row index. Rebuilt on mutation.
_KEY_INDEX: dict[str, dict[str, Any]] = {}

# Cached "an active admin key exists" — refreshed by _rebuild_index_locked()
# on every user/key mutation. Default False = open mode (matches the original
# pre-init behaviour where the SQL fallback raised and was caught).
_IS_LOCKED_DOWN: bool = False

# Debounce window for last_used_ts writes (seconds).
_LAST_USED_DEBOUNCE_S = 60.0

# Raw-key format. Prefix is cosmetic but aids leak-detection (Stripe pattern).
KEY_PREFIX = "wk_"
_KEY_BYTES = 32   # secrets.token_urlsafe(32) → 43-char base64url, 256-bit entropy

# ---------------------------------------------------------------------
# Permission model — per-user page access + own/all scope
# ---------------------------------------------------------------------

# Tri-state per page: "none" | "own" | "all".
#   "none" → 403 on the page (and its API sub-routes).
#   "own"  → page visible, store layer filters by caller's user_id.
#   "all"  → page visible, no user_id filter (admin-equivalent visibility).
# Admins (is_admin=1) bypass this entirely via the super-role short-circuit.
# /config and /config/api-keys are admin-only by definition and never appear.
PAGES: tuple[str, ...] = ("logs", "stats", "quick_config", "reports", "captures")

# Per-user data pages — support the full none|own|all triple.
SCOPED_PAGES: frozenset[str] = frozenset(
    ("logs", "quick_config", "reports", "captures")
)

# Pages without a per-user notion — only none|all is meaningful.
# (stats is server-wide aggregate metrics; "own" would be nonsensical.)
ACCESS_ONLY_PAGES: frozenset[str] = frozenset(("stats",))

# Allowed scope values per page, used by set_user_permissions() validation.
_ALLOWED_SCOPES: dict[str, tuple[str, ...]] = {
    p: (("none", "all") if p in ACCESS_ONLY_PAGES else ("none", "own", "all"))
    for p in PAGES
}

# Safe-by-default for new + existing non-admin users:
#   per-data pages → "own" (see their own records),
#   stats          → "none" (no leak of server health).
DEFAULT_NONADMIN_PERMS: dict[str, Any] = {
    "pages": {
        "logs":         "own",
        "stats":        "none",
        "quick_config": "own",
        "reports":      "own",
        "captures":     "own",
    },
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id           TEXT PRIMARY KEY,
  username     TEXT NOT NULL UNIQUE,
  is_admin     INTEGER NOT NULL DEFAULT 0,
  created_ts   REAL NOT NULL,
  revoked_ts   REAL,
  permissions  TEXT NOT NULL DEFAULT '{}'
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
    global _conn
    db_dir = os.path.dirname(os.path.abspath(db_path)) or "."
    os.makedirs(db_dir, exist_ok=True)
    _conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL;")
    _conn.execute("PRAGMA synchronous=NORMAL;")
    _conn.executescript(_SCHEMA)
    _migrate_add_permissions_locked()
    _rebuild_index_locked()


def _migrate_add_permissions_locked() -> None:
    """One-shot migration for DBs that predate the `permissions` column.
    Idempotent: ALTER fails-silent if the column already exists, and the
    backfill UPDATE only touches rows still on the empty '{}' default.

    Called from init_db() so a fresh deployment (which gets `permissions`
    via _SCHEMA) and an upgrade (which needs the ALTER) end on the same
    final shape."""
    conn = _require_conn()
    try:
        conn.execute(
            "ALTER TABLE users ADD COLUMN permissions TEXT NOT NULL DEFAULT '{}'"
        )
    except sqlite3.OperationalError:
        pass  # column already exists — fresh DB, or second-boot upgrade
    cur = conn.execute(
        "UPDATE users SET permissions = ?"
        " WHERE permissions = '{}' AND is_admin = 0",
        (json.dumps(DEFAULT_NONADMIN_PERMS),),
    )
    if cur.rowcount:
        logger.info(
            "[auth] backfilled default permissions for %d non-admin user(s)",
            cur.rowcount,
        )


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
    last4 = raw_key[-4:]
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
        "permissions": _parse_permissions(row["permissions"] if "permissions" in row.keys() else None),
    }


def _parse_permissions(raw: "str | None") -> dict[str, Any]:
    """Decode the JSON-encoded permissions column. Returns an empty dict on
    null/blank/invalid input — the policy object treats missing pages as
    'none', which is the safe default for an unknown shape."""
    if not raw:
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


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
    """Rebuild _KEY_INDEX from the DB. Caller holds _lock (or is in init).
    Also refresh _IS_LOCKED_DOWN — the answer only changes when an admin
    key/user is created or revoked, all of which already pass through
    this function, so the hot is_locked_down() check on every
    authenticated request avoids a JOIN COUNT(*) SQL roundtrip."""
    global _KEY_INDEX, _IS_LOCKED_DOWN
    conn = _require_conn()
    rows = conn.execute(
        "SELECT k.key_hash, k.id AS key_id, u.id AS user_id, u.username,"
        " u.is_admin, u.permissions"
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
            "permissions_raw": _parse_permissions(r["permissions"]),
        }
    _KEY_INDEX = idx
    _IS_LOCKED_DOWN = any(v["is_admin"] for v in idx.values())


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
    # Admin users get '{}' — they bypass the policy anyway. Non-admins
    # get the safe default so they can immediately review their own data
    # on /captures, /reports, /quick-config, /logs (stats stays off).
    perms_json = "{}" if is_admin else json.dumps(DEFAULT_NONADMIN_PERMS)
    conn = _require_conn()
    try:
        with _lock:
            conn.execute(
                "INSERT INTO users (id, username, is_admin, created_ts,"
                " revoked_ts, permissions)"
                " VALUES (?,?,?,?,NULL,?)",
                (uid, username, 1 if is_admin else 0, now, perms_json),
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


def list_users() -> list[dict[str, Any]]:
    conn = _require_conn()
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


def get_usernames(user_ids: "list[str | None]") -> "dict[str, str | None]":
    """Batched companion to get_username: one SQL roundtrip for many ids.
    Returns a dict keyed by every truthy input id mapping to its
    username (None for unknown / sentinel ids). The empty-id and
    "(open-mode)" sentinel are not represented in the dict; callers
    should still default to None for those.

    Use this on listing routes that would otherwise call get_username
    once per row — a /reports list with 500 rows = 500 SQL hits where
    one is enough."""
    real_ids = {
        uid for uid in user_ids
        if isinstance(uid, str) and uid and uid != "(open-mode)"
    }
    out: dict[str, "str | None"] = {uid: None for uid in real_ids}
    if not real_ids:
        return out
    conn = _require_conn()
    placeholders = ",".join("?" * len(real_ids))
    cur = conn.execute(
        f"SELECT id, username FROM users WHERE id IN ({placeholders})",
        list(real_ids),
    )
    for row in cur:
        out[row["id"]] = row["username"]
    return out


def revoke_user(user_id: str) -> None:
    """Soft-delete a user and revoke all their keys. Raises
    LastAdminError if the target is an active admin and revoking would
    leave zero active admins — checked atomically under _lock so a
    concurrent second revoke can't slip through."""
    now = time.time()
    conn = _require_conn()
    with _lock:
        row = conn.execute(
            "SELECT is_admin FROM users WHERE id = ? AND revoked_ts IS NULL",
            (user_id,),
        ).fetchone()
        if row and int(row["is_admin"]) == 1:
            # Count remaining active admin KEYS (not users) — is_locked_down
            # is driven by active admin keys, so revoking an admin user with
            # the only admin keys flips the server to OPEN mode even if
            # another keyless admin user still exists. Mirror revoke_key's
            # JOIN count and exclude the keys we're about to cascade-revoke.
            row2 = conn.execute(
                "SELECT COUNT(*) FROM api_keys k"
                " JOIN users u ON u.id = k.user_id"
                " WHERE k.user_id != ? AND k.revoked_ts IS NULL"
                " AND u.revoked_ts IS NULL AND u.is_admin = 1",
                (user_id,),
            ).fetchone()
            if int(row2[0]) == 0:
                raise LastAdminError(
                    "refusing to revoke the last admin user"
                )
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


def list_keys(user_id: str) -> list[dict[str, Any]]:
    conn = _require_conn()
    rows = conn.execute(
        "SELECT * FROM api_keys WHERE user_id = ? AND revoked_ts IS NULL"
        " ORDER BY created_ts DESC",
        (user_id,),
    ).fetchall()
    return [_row_to_key_dict(r) for r in rows]


def active_key_counts() -> dict[str, int]:
    """Return {user_id: active_key_count} in one SQL roundtrip — the batched
    companion to len(list_keys(user_id=u)) over a /api/users response."""
    conn = _require_conn()
    cur = conn.execute(
        "SELECT user_id, COUNT(*) AS n FROM api_keys"
        " WHERE revoked_ts IS NULL GROUP BY user_id"
    )
    return {r["user_id"]: int(r["n"]) for r in cur.fetchall()}


def get_key(key_id: str) -> dict[str, Any] | None:
    conn = _require_conn()
    row = conn.execute(
        "SELECT * FROM api_keys WHERE id = ?", (key_id,),
    ).fetchone()
    return _row_to_key_dict(row) if row else None


def revoke_key(key_id: str) -> None:
    """Soft-revoke a key. Raises LastAdminError if this is the last
    active admin key — checked atomically under _lock so two concurrent
    revokes of admin keys can't both pass a count==2 check."""
    now = time.time()
    conn = _require_conn()
    with _lock:
        row = conn.execute(
            "SELECT u.is_admin FROM api_keys k JOIN users u ON u.id = k.user_id"
            " WHERE k.id = ? AND k.revoked_ts IS NULL AND u.revoked_ts IS NULL",
            (key_id,),
        ).fetchone()
        if row and int(row["is_admin"]) == 1:
            row2 = conn.execute(
                "SELECT COUNT(*) FROM api_keys k JOIN users u ON u.id = k.user_id"
                " WHERE k.id != ? AND k.revoked_ts IS NULL"
                " AND u.revoked_ts IS NULL AND u.is_admin = 1",
                (key_id,),
            ).fetchone()
            if int(row2[0]) == 0:
                raise LastAdminError(
                    "refusing to revoke the last active admin key"
                )
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
    # is_admin=True short-circuits the policy object anyway, but populate
    # the raw shape so a hypothetical demote-to-non-admin path doesn't see
    # a missing key.
    "permissions_raw": {},
}


# ---------------------------------------------------------------------
# Permission helpers (user-level, JSON column on users)
# ---------------------------------------------------------------------

def get_user_permissions(user_id: str) -> dict[str, Any]:
    """Return the JSON-decoded permissions dict for `user_id`, or an
    empty dict for unknown / revoked / sentinel ids. Cheap render-time
    lookup used by the admin matrix UI."""
    if not user_id or user_id == "(open-mode)":
        return {}
    conn = _require_conn()
    row = conn.execute(
        "SELECT permissions FROM users WHERE id = ?", (user_id,),
    ).fetchone()
    if row is None:
        return {}
    return _parse_permissions(row["permissions"])


def set_user_permissions(user_id: str, perms: dict[str, Any]) -> dict[str, Any]:
    """PATCH-merge a permissions update onto the user's stored shape.
    Returns the (validated + canonical) dict that landed in the DB so
    the caller can echo it back to the UI.

    Merge semantics: incoming `perms["pages"]` is merged with whatever
    is currently stored; pages absent from the payload retain their
    existing scope. The matrix UI sends the full set on each Save, but
    partial PATCH calls (CLI, future bulk-edit) only need to send the
    cells they want to change.

    Validation rules:
      - `perms["pages"]` must be a dict (other top-level keys ignored).
      - Page names must be in api_keys_store.PAGES.
      - Scope values must be in _ALLOWED_SCOPES[page] — i.e., "none"|"own"|
        "all" for scoped pages, "none"|"all" for access-only pages.
      - Pages missing from BOTH incoming and stored shapes default to
        "none" — the persisted JSON is always canonical (all PAGES keys
        present), which keeps the matrix UI simple.

    Raises ValueError on any invalid input. Rebuilds the in-memory index
    so live key holders see the new scope on their next request (no
    logout required)."""
    if not isinstance(perms, dict):
        raise ValueError("permissions must be a JSON object")
    incoming_pages = perms.get("pages", {})
    if not isinstance(incoming_pages, dict):
        raise ValueError("permissions.pages must be a JSON object")
    for page, scope in incoming_pages.items():
        if page not in PAGES:
            raise ValueError(f"unknown page: {page!r}")
        allowed = _ALLOWED_SCOPES[page]
        if scope not in allowed:
            raise ValueError(
                f"invalid scope {scope!r} for {page!r}"
                f" (allowed: {', '.join(allowed)})"
            )
    # Merge: incoming wins, otherwise keep what's stored, otherwise "none".
    existing = get_user_permissions(user_id).get("pages") or {}
    merged_pages: dict[str, str] = {}
    for page in PAGES:
        if page in incoming_pages:
            merged_pages[page] = incoming_pages[page]
        elif page in existing and existing[page] in _ALLOWED_SCOPES[page]:
            merged_pages[page] = existing[page]
        else:
            merged_pages[page] = "none"
    clean = {"pages": merged_pages}
    conn = _require_conn()
    with _lock:
        cur = conn.execute(
            "UPDATE users SET permissions = ? WHERE id = ? AND revoked_ts IS NULL",
            (json.dumps(clean), user_id),
        )
        if cur.rowcount == 0:
            raise ValueError("user not found or revoked")
        _rebuild_index_locked()
    logger.info(
        "[auth] permissions updated user=%s pages=%s",
        user_id[:8], merged_pages,
    )
    return clean


def is_locked_down() -> bool:
    """Server is locked down iff at least one active admin key exists.
    Open mode (return False) lets every request through as the synthetic
    admin so the operator can bootstrap. A periodic WARNING is logged
    by the auth module while open. Reads the cache populated by
    _rebuild_index_locked() — no SQL on the per-request hot path."""
    return _IS_LOCKED_DOWN
