"""FastAPI auth dependencies — the bearer-token gate that resolves an
incoming `Authorization: Bearer <api_key>` header to a `UserRecord`.

Two-level dependency layering (FastAPI idiomatic):

  - `get_current_user`: hashes the bearer, looks up in api_keys_store,
    raises 401 on miss. In OPEN mode (no admin key configured yet)
    returns the synthetic `OPEN_MODE_USER` so existing checks keep
    working while the operator bootstraps.
  - `require_admin`: depends on `get_current_user` and raises 403 if
    `is_admin=False`.

Open-mode logging: a background asyncio task emits a WARNING every 60
seconds while the server has no admin keys, so the operator sees the
nag on every page-load AND in the logs.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import api_keys_store

logger = logging.getLogger("whisper-api")

# auto_error=False so we can return our own 401 body with WWW-Authenticate.
_bearer = HTTPBearer(auto_error=False, description="API key as bearer token")

_UNAUTH = HTTPException(
    status.HTTP_401_UNAUTHORIZED,
    "invalid or missing API key",
    headers={"WWW-Authenticate": "Bearer"},
)

_FORBIDDEN = HTTPException(
    status.HTTP_403_FORBIDDEN, "admin privileges required",
)


# ---------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------

def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    """Resolve the bearer to a user record. Raises 401 on miss (locked
    down) or returns the synthetic admin (open mode).

    Returns a dict with keys: key_id, user_id, username, is_admin.
    """
    if not api_keys_store.is_locked_down():
        return dict(api_keys_store.OPEN_MODE_USER)

    if creds is None or (creds.scheme or "").lower() != "bearer":
        raise _UNAUTH
    rec = api_keys_store.lookup_by_raw_key(creds.credentials or "")
    if rec is None:
        raise _UNAUTH
    return rec


def require_admin(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Same as get_current_user, then 403 if not admin."""
    if not user.get("is_admin"):
        raise _FORBIDDEN
    return user


# ---------------------------------------------------------------------
# Open-mode nag loop
# ---------------------------------------------------------------------

_OPEN_MODE_LOG_INTERVAL_S = 60.0


async def open_mode_warning_loop() -> None:
    """Background task: emit WARNING every 60 s while in open mode.
    Stops nagging once an admin key is created."""
    logged_locked_down = False
    while True:
        try:
            if api_keys_store.is_locked_down():
                if not logged_locked_down:
                    logger.info("[auth] admin key present — server is locked down")
                    logged_locked_down = True
            else:
                logger.warning(
                    "[auth] no admin key configured — running in OPEN mode."
                    " Anyone reachable on this server can use it. Generate"
                    " an admin key in /config/api-keys now."
                )
                logged_locked_down = False
        except Exception as e:
            logger.error("[auth] open-mode loop error: %s", e)
        try:
            await asyncio.sleep(_OPEN_MODE_LOG_INTERVAL_S)
        except asyncio.CancelledError:
            return
