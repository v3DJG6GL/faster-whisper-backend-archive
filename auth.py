"""FastAPI auth dependencies — the bearer-token gate that resolves an
incoming `Authorization: Bearer <api_key>` header to a `UserRecord`.

Three-level dependency layering (FastAPI idiomatic):

  - `get_current_user`: hashes the bearer, looks up in api_keys_store,
    raises 401 on miss. Attaches a `Permissions` policy object so
    callers can ask "can this user reach page X?" / "what data scope?".
    In OPEN mode (no admin key configured yet) returns the synthetic
    `OPEN_MODE_USER` so existing checks keep working while the operator
    bootstraps.
  - `require_admin`: depends on `get_current_user` and raises 403 if
    `is_admin=False`. Used for system-mutation endpoints (/settings,
    /settings/api-keys, delete/clear/reapply-rules).
  - `require_page(name)`: dependency factory — raises 403 if the user
    has no access to the named page. Mounted on the per-data routers
    (/captures, /reports, /quick-config, /logs, /stats) at the
    APIRouter constructor so every sub-route inherits the check.

Open-mode logging: a background asyncio task emits a WARNING every 60
seconds while the server has no admin keys, so the operator sees the
nag on every page-load AND in the logs.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import api_keys_store
import config as cfg
import sessions_store

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
# Permissions — policy object
# ---------------------------------------------------------------------

class Permissions:
    """Read-side wrapper around a user's stored permissions JSON.

    Built once per request by `get_current_user`. Routes ask four
    questions:
      - `can(page)`        — page-access gate (admin bypass).
      - `scope(page)`      — "own" or "all" for visible pages.
      - `effective_user_id_for(page, uid)` — store-layer filter contract:
                              None  → no filter
                              str   → WHERE user_id = ?
      - `can_see_rule(rule)` — visibility for a single PIPELINE_RULES row
                               via the user's quick_config_tags.
    Plus `assert_can_read_row(row, page, uid)` for detail-by-id endpoints
    — returns 404 (not 403) on scope miss to avoid leaking existence
    (OWASP IDOR Prevention Cheat Sheet guidance).
    """

    __slots__ = ("_pages", "_is_admin", "_tags")

    def __init__(self, raw: dict[str, Any] | None, is_admin: bool) -> None:
        pages = (raw or {}).get("pages", {})
        self._pages: dict[str, str] = pages if isinstance(pages, dict) else {}
        self._is_admin = bool(is_admin)
        # Per-user quick-config tags. Asymmetric semantics: an empty
        # list means the user sees ONLY untagged rules (prevents quiet
        # widening when a new user is added without tags). Admin
        # bypasses this entirely via can_see_rule.
        raw_tags = (raw or {}).get("quick_config_tags") or []
        self._tags: list[str] = (
            list(raw_tags) if isinstance(raw_tags, list) else []
        )

    def can(self, page: str) -> bool:
        if self._is_admin:
            return True
        return self._pages.get(page, "none") != "none"

    def scope(self, page: str) -> str:
        if self._is_admin:
            return "all"
        return self._pages.get(page, "none")

    def effective_user_id_for(self, page: str, caller_uid: str) -> str | None:
        """Return the value to pass to store layer as `user_id=...`.
        None = "no filter" (admin-equivalent visibility); str = the
        caller's own id (own-only filter). Store helpers (e.g.,
        `captures_store.list_captures`) already treat None as no-filter
        — this method is the one place that contract is encoded."""
        return None if self.scope(page) == "all" else caller_uid

    def assert_can_read_row(
        self, row: dict[str, Any] | None, page: str, caller_uid: str,
    ) -> None:
        """Detail-endpoint guard. 404 (not 403) on cross-user reads —
        a 403 would confirm the row exists, which is itself a leak."""
        if self.scope(page) == "all":
            return
        if (row or {}).get("user_id") != caller_uid:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, "not found",
            )

    def quick_config_tags(self) -> list[str]:
        """The caller's per-user tag list. Admins return the empty list
        here (they bypass tag filtering — see can_see_rule); the UI
        should special-case admins separately rather than treating an
        empty tag list as a permission signal."""
        return list(self._tags)

    def can_see_rule(self, rule: dict[str, Any]) -> bool:
        """Visibility check for a single PIPELINE_RULES entry on
        /quick-config.

        Everyone (including admins) is filtered by `exposed=True` —
        /quick-config is the curated end-user view, not a mirror of
        /settings. Admins manage the full unfiltered list at /settings.

        Within exposed rules, admins bypass the tag filter (they see
        every exposed rule). Non-admins additionally need rule.tags
        to intersect user.tags — except when rule.tags is empty,
        which is the "visible to everyone" migration default.
        """
        if not (rule or {}).get("exposed"):
            return False
        if self._is_admin:
            return True  # exposed + admin = see it regardless of tags
        rule_tags = set(rule.get("tags") or [])
        if not rule_tags:
            return True  # untagged rule = permissive (migration path)
        return bool(rule_tags & set(self._tags))

    def to_dict(self) -> dict[str, Any]:
        """Serializable view for `/auth/whoami`."""
        return {
            "pages": dict(self._pages),
            "quick_config_tags": list(self._tags),
        }


# ---------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------

def user_from_session_cookie(request: Request) -> dict[str, Any] | None:
    """Resolve the HttpOnly session cookie to a LIVE user record, or None.

    Shared by get_current_user and the SSE auth variants. The session
    stores only a user_id, so the record (permissions, admin status) is
    re-derived per request via api_keys_store.get_user_record — permission
    edits and user revocation take effect without re-login. On a hit the
    session's CSRF token is stashed on request.state for the CSRF
    middleware to verify against the X-CSRF-Token header."""
    raw = request.cookies.get(cfg.SESSION_COOKIE_NAME, "")
    if not raw:
        return None
    sess = sessions_store.lookup_session(raw)
    if sess is None:
        return None
    rec = api_keys_store.get_user_record(sess["user_id"])
    if rec is None:
        return None
    request.state.session_csrf = sess["csrf_token"]
    return rec


def get_current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    """Resolve the caller to a user record. Raises 401 on miss (locked
    down) or returns the synthetic admin (open mode).

    Two credential carriers in locked-down mode, tried in order:
      1. `Authorization: Bearer <api_key>` — API clients (Vowen, curl)
         and any direct API use.
      2. The HttpOnly session cookie issued by /auth/login — the WebUI.
    Bearer wins when present so API clients stay deterministic.

    Returns a dict with keys: key_id, user_id, username, is_admin,
    permissions (a `Permissions` policy object).
    """
    rec = _resolve_user(request, creds)
    if rec is None:
        raise _UNAUTH
    return rec


def _resolve_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None,
) -> dict[str, Any] | None:
    """Non-raising core of `get_current_user`: returns the resolved record or
    None (locked down + no/invalid credential). In OPEN mode returns the
    synthetic admin. Shared with `require_host_or_auth` so the host-OR-key gates
    can fall back to auth without catching exceptions.
    """
    if not api_keys_store.is_locked_down():
        rec = dict(api_keys_store.OPEN_MODE_USER)
    else:
        rec = None
        if creds is not None and (creds.scheme or "").lower() == "bearer":
            rec = api_keys_store.lookup_by_raw_key(creds.credentials or "")
        if rec is None:
            rec = user_from_session_cookie(request)
        if rec is None:
            return None
    # Attach the policy object so routes can ask can(page)/scope(page)
    # without each one re-parsing the JSON.
    rec["permissions"] = Permissions(
        rec.get("permissions_raw") or {},
        bool(rec.get("is_admin")),
    )
    return rec


def require_host_or_auth(
    allowlist_ref: "Callable[[], list[str]]",
    *,
    admin: bool = False,
):
    """Dependency factory: pass if the client host is in `allowlist_ref()`
    (loopback always allowed) OR the caller presents a valid API key (bearer
    header or session cookie). When `admin=True` the key must belong to an admin.

    Used to gate endpoints that should be reachable from trusted hosts without a
    key but otherwise require one — e.g. /docs (any key) and /sev, /logs (admin
    key). OPEN mode → synthetic admin → always passes, so bootstrapping is
    unaffected. Returns 401 when locked down with no/invalid key, 403 when a
    valid non-admin key hits an admin-only gate.
    """
    import web_common  # local import avoids any module-load ordering concerns

    def _dep(
        request: Request,
        creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> None:
        if web_common.host_in_allowlist(request, allowlist_ref()):
            return
        rec = _resolve_user(request, creds)
        if rec is None:
            raise _UNAUTH
        if admin and not rec.get("is_admin"):
            raise _FORBIDDEN

    return _dep


def require_admin(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Same as get_current_user, then 403 if not admin."""
    if not user.get("is_admin"):
        raise _FORBIDDEN
    return user


def require_page(name: str):
    """Dependency factory: 403 if the user has no access to the named
    page. Admins bypass via Permissions.can(). Mount on the APIRouter
    constructor so every present + future sub-route inherits the check
    — closes the "forgot to gate this endpoint" hole.

    Usage:
        router = APIRouter(
            prefix="/captures",
            dependencies=[
                Depends(require_admin_host),
                Depends(require_page("captures")),
            ],
        )
    """
    def _dep(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        perms: Permissions = user["permissions"]
        if not perms.can(name):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"no access to /{name.replace('_', '-')}",
            )
        return user
    return _dep


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
                    " an admin key in /settings/api-keys now."
                )
                logged_locked_down = False
        except Exception as e:
            logger.error("[auth] open-mode loop error: %s", e)
        try:
            await asyncio.sleep(_OPEN_MODE_LOG_INTERVAL_S)
        except asyncio.CancelledError:
            return
