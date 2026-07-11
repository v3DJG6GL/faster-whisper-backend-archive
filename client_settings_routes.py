"""Per-account settings-sync endpoints for the desktop client.

Mounted always-on in main.py (like the pipeline-rules /v1 router), so a
route-level 404 keeps its client-side meaning of "this backend build
doesn't have the endpoint". Endpoints:

  GET    /v1/client-settings   Current blob (200 {version:0, blob:null} when empty)
  PUT    /v1/client-settings   Optimistic write; 409 carries the current state
  DELETE /v1/client-settings   Drop the stored blob

Security model:
  - User-tier bearer auth ONLY: Depends(get_current_user). Deliberately NO
    require_page gate (unlike quick_config's v1_router) — settings sync is
    account infrastructure, not a page permission; a key that may transcribe
    must be able to sync its client's settings. No host allowlist for the
    same reason /v1/usage has none: remote desktop clients must reach it.
  - Open mode: every caller resolves to the synthetic "(open-mode)" user,
    so ALL open-mode devices share one row. Acceptable for the single-user
    self-host case; documented in the README.
  - CSRF: bearer clients are exempt from the double-submit middleware; a
    cookie-authenticated browser PUT/DELETE needs X-CSRF-Token (main.py).

The blob is opaque, SENSITIVE client JSON (may contain the client's own
backend API keys). Never log or interpret its contents — the store logs
only sizes/versions, and this module logs nothing about the payload.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import client_settings_store
from auth import get_current_user

router = APIRouter(prefix="/v1")


class PutClientSettings(BaseModel):
    """PUT body. `blob` must be a JSON OBJECT (not any scalar) so a stored
    blob can never be confused with the `blob: null` zero-state on GET.
    The server never interprets its keys. `base_version` echoes the version
    the client last saw; 0 (or absent row semantics) means "create"."""
    model_config = {"extra": "forbid"}
    blob: dict[str, Any]
    base_version: int
    device: str | None = None


def _store_unavailable() -> HTTPException:
    """503 for StoreUnavailable: init_db failed at startup (the reason is in
    the server log — typically a CLIENT_SETTINGS_DB path that isn't writable
    in this deployment). A bare 500 here told the operator nothing."""
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "client-settings store unavailable on this server — check the server "
        "log for the startup error (CLIENT_SETTINGS_DB path/permissions)",
    )


def _state_body(row: dict[str, Any] | None) -> dict[str, Any]:
    """The one wire shape shared by GET, PUT-200, and PUT-409 (so a client
    parses a single schema everywhere): absent row = the version-0 zero-state."""
    if row is None:
        return {"version": 0, "blob": None, "updated_at": None, "device": None}
    return {
        "version": row["version"],
        "blob": row["blob"],
        "updated_at": row["updated_at"],
        "device": row["device"],
    }


@router.get("/client-settings")
async def get_client_settings(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """The caller's stored settings blob. Empty store is 200 with
    `{version: 0, blob: null}` — NOT 404/204, because the desktop client
    reads a route-level 404 as "backend too old for sync" and a 204 would
    force a bodyless special case."""
    try:
        return _state_body(client_settings_store.get(user["user_id"]))
    except client_settings_store.StoreUnavailable:
        raise _store_unavailable() from None


@router.put("/client-settings")
async def put_client_settings(
    payload: PutClientSettings,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Optimistic write. 200 = stored (body carries the new version);
    409 = `base_version` is stale — the body carries the CURRENT state so
    the client can 3-way merge and re-PUT without another GET round-trip.
    413 = blob over the server cap. Force-push is just a PUT echoing the
    version fetched a moment ago; there is no bypass flag."""
    try:
        ok, row = client_settings_store.put(
            user["user_id"],
            payload.blob,
            payload.base_version,
            device=payload.device,
        )
    except client_settings_store.StoreUnavailable:
        raise _store_unavailable() from None
    except ValueError:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            "client settings blob too large",
        )
    if not ok:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={**_state_body(row), "detail": "version conflict"},
        )
    return JSONResponse(status_code=status.HTTP_200_OK, content=_state_body(row))


@router.delete("/client-settings")
async def delete_client_settings(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Drop the stored blob. After this, GET reads the version-0 zero-state
    and a device still holding version N gets a 409 on its next PUT — the
    deletion surfaces instead of being silently overwritten."""
    try:
        removed = client_settings_store.delete(user["user_id"])
    except client_settings_store.StoreUnavailable:
        raise _store_unavailable() from None
    return {"ok": True, "deleted": removed}
