"""Durable browser-session store — the cookie layer on top of api_keys_store.

The WebUI exchanges a pasted API key at /auth/login for an HttpOnly session
cookie. This module persists those sessions so they survive a browser/computer
restart (unlike the old tab-scoped sessionStorage). Non-browser clients (curl,
SDKs) never touch this — they keep sending `Authorization: Bearer`.

Storage layout:

  cfg.SESSIONS_DB — SQLite (WAL) with one table:
    sessions — { token_hash, user_id, csrf_token, created_ts,
                 expires_ts, revoked_ts }

Session tokens at rest are SHA-256(raw_token) hex — same rationale as
api_keys_store: a high-entropy random token (256-bit) makes slow password
hashes pointless. The raw token lives only in the HttpOnly cookie.

`csrf_token` is the double-submit pairing value. It is delivered to JS via a
readable cookie (and /auth/whoami) and echoed back as X-CSRF-Token on
cookie-authenticated mutations; the CSRF middleware compares it against this
stored value. Stored plaintext because it is, by design, handed to the client.

Lookup is O(1) via an in-memory `_SESSION_INDEX: dict[token_hash, row]` rebuilt
on mutation, mirroring api_keys_store. The sliding-TTL refresh on each lookup
is debounced so an active session does not write to disk on every request.
"""
from __future__ import annotations

import logging
import os
import secrets
import sqlite3
import threading
import time
from hashlib import sha256
from typing import Any

logger = logging.getLogger("whisper-api")

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

# In-memory token_hash → session row index. Rebuilt on mutation.
_SESSION_INDEX: dict[str, dict[str, Any]] = {}

# Debounce window for sliding-expiry writes (seconds): an active session
# refreshes its expiry at most once per this interval.
_SLIDE_DEBOUNCE_S = 300.0
_SLIDE_CACHE: dict[str, float] = {}

_TOKEN_BYTES = 32   # secrets.token_urlsafe(32) → 43-char base64url, 256-bit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  token_hash  TEXT PRIMARY KEY,
  user_id     TEXT NOT NULL,
  key_id      TEXT,
  csrf_token  TEXT NOT NULL,
  created_ts  REAL NOT NULL,
  expires_ts  REAL NOT NULL,
  revoked_ts  REAL
);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_ts);
"""


# ---------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------

def init_db(db_path: str) -> None:
    """Open the SQLite DB (WAL) and ensure the schema exists. Idempotent.
    Purges expired rows and builds the in-memory index from active ones."""
    global _conn
    db_dir = os.path.dirname(os.path.abspath(db_path)) or "."
    os.makedirs(db_dir, exist_ok=True)
    _conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL;")
    _conn.execute("PRAGMA synchronous=NORMAL;")
    _conn.executescript(_SCHEMA)
    _migrate_add_key_id(_conn)
    with _lock:
        _purge_expired_locked()
        _rebuild_index_locked()


def _migrate_add_key_id(conn: sqlite3.Connection) -> None:
    """Add the `key_id` column to a sessions table created before login began
    stamping the key. Pre-migration sessions keep `key_id` NULL, so they resolve
    to the `(session)` sentinel (no key layer) — the prior behaviour — until the
    user next logs in and a fresh, key-stamped session is issued."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
    if "key_id" not in cols:
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN key_id TEXT")
        except sqlite3.Error:
            pass  # best-effort; a fresh DB already has the column


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("sessions_store.init_db() was not called before use.")
    return _conn


# ---------------------------------------------------------------------
# Token + index helpers
# ---------------------------------------------------------------------

def hash_token(raw_token: str) -> str:
    """SHA-256 hex of the raw session token (UTF-8)."""
    return sha256(raw_token.encode("utf-8")).hexdigest()


def _rebuild_index_locked() -> None:
    """Rebuild _SESSION_INDEX from the DB. Caller holds _lock (or is in
    init). Only non-revoked, non-expired rows are indexed."""
    global _SESSION_INDEX
    conn = _require_conn()
    now = time.time()
    rows = conn.execute(
        "SELECT token_hash, user_id, key_id, csrf_token, created_ts, expires_ts"
        " FROM sessions WHERE revoked_ts IS NULL AND expires_ts > ?",
        (now,),
    ).fetchall()
    _SESSION_INDEX = {
        r["token_hash"]: {
            "user_id": r["user_id"],
            "key_id": r["key_id"],
            "csrf_token": r["csrf_token"],
            "created_ts": float(r["created_ts"]),
            "expires_ts": float(r["expires_ts"]),
        }
        for r in rows
    }


def _purge_expired_locked() -> None:
    """Best-effort delete of revoked/expired rows. Caller holds _lock."""
    conn = _require_conn()
    try:
        conn.execute(
            "DELETE FROM sessions WHERE revoked_ts IS NOT NULL OR expires_ts <= ?",
            (time.time(),),
        )
    except sqlite3.Error:
        pass  # cleanup is non-fatal


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def create_session(user_id: str, ttl_s: float,
                   key_id: str | None = None) -> tuple[str, str]:
    """Create a session for `user_id` valid for `ttl_s` seconds. Returns
    `(raw_token, csrf_token)`. The raw token is the ONLY way the caller
    will see it — set it in the HttpOnly cookie and discard.

    `key_id` is the API key exchanged at login. Stamping it lets per-key
    overrides/locks bind on the resulting cookie-authenticated requests too
    (without it the session resolves to the `(session)` sentinel = no key
    layer, so per-key restrictions would silently stop applying)."""
    raw_token = secrets.token_urlsafe(_TOKEN_BYTES)
    csrf_token = secrets.token_urlsafe(_TOKEN_BYTES)
    th = hash_token(raw_token)
    now = time.time()
    expires = now + float(ttl_s)
    conn = _require_conn()
    with _lock:
        conn.execute(
            "INSERT INTO sessions"
            " (token_hash, user_id, key_id, csrf_token, created_ts,"
            " expires_ts, revoked_ts)"
            " VALUES (?,?,?,?,?,?,NULL)",
            (th, user_id, key_id, csrf_token, now, expires),
        )
        _rebuild_index_locked()
    logger.info("[auth] session created user=%s ttl=%.0fs", user_id[:8], ttl_s)
    return raw_token, csrf_token


def lookup_session(raw_token: str) -> dict[str, Any] | None:
    """Resolve a raw session token to its active record. Returns
    `{user_id, key_id, csrf_token, created_ts, expires_ts}` (key_id may be
    None for pre-migration sessions) or None if missing / revoked / expired.
    On a hit, slides the expiry forward (debounced).

    Sliding window: each successful lookup extends expires_ts to
    now + (original lifetime), so an actively-used session never lapses.
    """
    if not raw_token:
        return None
    th = hash_token(raw_token)
    rec = _SESSION_INDEX.get(th)
    if rec is None:
        return None
    now = time.time()
    if rec["expires_ts"] <= now:
        # Lazily evict an index entry that lapsed since the last rebuild.
        with _lock:
            _SESSION_INDEX.pop(th, None)
        return None
    _slide_expiry_debounced(th, rec)
    return dict(rec)


def _slide_expiry_debounced(token_hash: str, rec: dict[str, Any]) -> None:
    """Refresh expires_ts at most once per _SLIDE_DEBOUNCE_S per session.
    Preserves the original lifetime (expires - created) as the window."""
    now = time.time()
    last = _SLIDE_CACHE.get(token_hash, 0.0)
    if now - last < _SLIDE_DEBOUNCE_S:
        return
    _SLIDE_CACHE[token_hash] = now
    lifetime = rec["expires_ts"] - rec["created_ts"]
    new_expires = now + lifetime
    rec["expires_ts"] = new_expires  # keep the in-memory index current
    conn = _require_conn()
    try:
        with _lock:
            conn.execute(
                "UPDATE sessions SET expires_ts = ?"
                " WHERE token_hash = ? AND revoked_ts IS NULL",
                (new_expires, token_hash),
            )
    except sqlite3.Error:
        pass  # non-fatal: the in-memory expiry was already bumped


def revoke_session(raw_token: str) -> None:
    """Soft-revoke a session (used by /auth/logout). No-op if unknown."""
    if not raw_token:
        return
    th = hash_token(raw_token)
    now = time.time()
    conn = _require_conn()
    with _lock:
        conn.execute(
            "UPDATE sessions SET revoked_ts = ?"
            " WHERE token_hash = ? AND revoked_ts IS NULL",
            (now, th),
        )
        _SLIDE_CACHE.pop(th, None)
        _rebuild_index_locked()


def purge_expired() -> None:
    """Public best-effort cleanup of revoked/expired rows + index rebuild."""
    with _lock:
        _purge_expired_locked()
        _rebuild_index_locked()


def _reset_for_tests() -> None:
    """Drop the in-memory caches so the autouse test fixture starts clean."""
    global _SESSION_INDEX
    _SESSION_INDEX = {}
    _SLIDE_CACHE.clear()
