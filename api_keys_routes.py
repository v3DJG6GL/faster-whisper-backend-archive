"""/settings/api-keys — admin UI for per-user API key management.

Endpoints (all admin-only):

  GET    /settings/api-keys                       HTML page
  GET    /settings/api-keys/api/users             list users + key counts
  POST   /settings/api-keys/api/users             { username, is_admin }
  DELETE /settings/api-keys/api/users/{uid}       soft-revoke (cascades to keys)
  GET    /settings/api-keys/api/users/{uid}/keys  list keys for one user
  POST   /settings/api-keys/api/users/{uid}/keys  { label }   -> show-once raw key
  DELETE /settings/api-keys/api/users/{uid}/keys/{kid}        soft-revoke

Last-admin guard: revoking the only admin key (or only admin user)
returns 409. Prevents accidental lockout.

The HTML page is a single-file React-less app: vanilla JS + an HttpOnly
session cookie (same pattern as /settings). Generates keys with a show-once modal —
the raw key is copied to the clipboard, then never retrievable.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import api_keys_store
import web_common
from web_common import require_admin_webui_host
from auth import require_admin

logger = logging.getLogger("whisper-api")

router = APIRouter(prefix="/settings/api-keys")


# ---------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------

class CreateUserIn(BaseModel):
    model_config = {"extra": "forbid"}
    username: str = Field(min_length=1, max_length=128)
    is_admin: bool = False


class CreateKeyIn(BaseModel):
    model_config = {"extra": "forbid"}
    label: str = Field(default="", max_length=128)


class RenameKeyIn(BaseModel):
    """Wire shape for PATCH /api/users/{uid}/keys/{kid}/label. A label is
    required (the store rejects blank) — renaming only touches the display
    label, never the secret."""
    model_config = {"extra": "forbid"}
    label: str = Field(max_length=128)


class ConfigBindingIn(BaseModel):
    """Wire shape for a per-identity config binding — shared by the per-user
    `config` field on PatchPermissionsIn and the per-key /config endpoint.
    `overrides` is a flat OverrideProfile-shaped dict (decode/streaming fields
    + PIPELINE_RULES_EXCLUDE/INCLUDE); `profiles` is the ordered FORCED-applied
    profile list (earlier wins); `locks` names the fields the client may not
    override.

    Request-gate fields (all optional; None = inherit the next scope → global,
    and they can only NARROW the global gate, never widen it):
    `allow_request_override_profile` — may this identity NAME a profile per
    request; `allow_request_decode_overrides` — may it send inline decode
    tweaks; `allowed_override_profiles` — the request allowlist (distinct from
    `profiles`): None = all, ["*"] = all, an explicit list restricts, [] = none.

    `apply_no_profiles` is an ADMIN FORCE, not a request gate (set per-key in the
    WebUI): True suppresses every bound + requested profile for the identity, so
    it resolves to plain defaults. It does NOT inherit and is NOT bound by
    ALLOW_REQUEST_OVERRIDE_PROFILE; None/absent = off.
    """
    model_config = {"extra": "forbid"}
    overrides: dict[str, Any] = Field(default_factory=dict)
    profiles: list[str] = Field(default_factory=list)
    locks: list[str] = Field(default_factory=list)
    allow_request_override_profile: bool | None = None
    allow_request_decode_overrides: bool | None = None
    allowed_override_profiles: list[str] | None = None
    apply_no_profiles: bool | None = None


class PatchPermissionsIn(BaseModel):
    """Payload for PATCH /api/users/{uid}/permissions.

    `pages` is a partial map — only the cells the admin changed need
    appear; the store's `set_user_permissions` merges with the existing
    shape (omitted pages keep their stored scope). Cell-by-cell save
    is also fine; full-row save (what the matrix UI sends) just lists
    every page.

    `quick_config_tags` is the user's per-rule tag set for the new
    tag-based /quick-config visibility filter. `None` means "leave the
    stored value untouched" — useful for cell-level saves that only
    touched pages. An empty list `[]` is explicit "clear all tags"
    (user sees only untagged rules)."""
    model_config = {"extra": "forbid"}
    pages: dict[str, str] = Field(default_factory=dict)
    quick_config_tags: list[str] | None = None
    # Per-user config binding (override profiles + direct override blob + per-
    # field locks). `None` = leave the stored value untouched; an explicit empty
    # binding ({} overrides, [] profiles) clears it. Validated by
    # config_store.validate_binding inside set_user_permissions.
    config: ConfigBindingIn | None = None


# ---------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------

@router.get(
    "",
    dependencies=[Depends(require_admin_webui_host)],
    response_class=HTMLResponse,
)
async def api_keys_page() -> HTMLResponse:
    return HTMLResponse(
        web_common.render_page(_API_KEYS_HTML, current="api-keys"),
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------
# JSON APIs
# ---------------------------------------------------------------------

@router.get(
    "/api/users",
    dependencies=[Depends(require_admin_webui_host), Depends(require_admin)],
)
async def list_users_api() -> JSONResponse:
    import config as cfg
    import config_store
    # Snapshot of every exposed (non-terminal) rule's tag list. Lets
    # the matrix UI render the "Will see: N of M rules" preview live
    # as the admin edits a user's tags — no extra roundtrip.
    exposed_rule_tags: list[list[str]] = []
    for r in (cfg.PIPELINE_RULES or []):
        rd = r.model_dump() if hasattr(r, "model_dump") else r
        if not isinstance(rd, dict):
            continue
        if rd.get("type") == "terminal":
            continue
        if not rd.get("exposed"):
            continue
        exposed_rule_tags.append(list(rd.get("tags") or []))
    users = api_keys_store.list_users()
    # Annotate each user with their active key count for the card header.
    # Batched: one GROUP BY query instead of N list_keys() roundtrips.
    counts = api_keys_store.active_key_counts()
    # Newest last_used_ts across each user's non-revoked keys — feeds the
    # header's "last active" timestamp + activity dot (renderUser builds the
    # header synchronously, before the per-user /keys fetch, so this can't be
    # derived client-side from the rendered key cards). Same batched shape as
    # the counts above.
    last_used = api_keys_store.last_used_by_user()
    out = [
        {
            **u,
            "active_key_count": counts.get(u["id"], 0),
            "last_used_ts": last_used.get(u["id"]),
            # permissions is already in `u` via _row_to_user_dict — keep
            # the canonical key name so the matrix UI can read it
            # directly without a second roundtrip.
        }
        for u in users
    ]
    return JSONResponse({
        "users": out,
        "open_mode": not api_keys_store.is_locked_down(),
        # Surface the page model so the front-end matrix can render
        # column headers without hardcoding them — keeps server +
        # client in sync when a new page is added.
        "pages": list(api_keys_store.PAGES),
        "scoped_pages": sorted(api_keys_store.SCOPED_PAGES),
        "access_only_pages": sorted(api_keys_store.ACCESS_ONLY_PAGES),
        # Union of every tag currently used by any rule. The matrix's
        # tag-picker uses this for autocomplete + "tags actually in
        # use" hints. Empty list means no rule is tagged yet, which
        # is the day-0 migration state.
        "available_tags": config_store.pipeline_rule_tags(cfg.PIPELINE_RULES),
        # Tag list per exposed rule — for the "Will see: N of M" preview.
        "exposed_rule_tags": exposed_rule_tags,
    })


@router.post(
    "/api/users",
    dependencies=[Depends(require_admin_webui_host), Depends(require_admin)],
)
async def create_user_api(payload: CreateUserIn) -> JSONResponse:
    try:
        uid = api_keys_store.create_user(payload.username, payload.is_admin)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return JSONResponse({"user_id": uid})


@router.delete(
    "/api/users/{uid}",
    dependencies=[Depends(require_admin_webui_host), Depends(require_admin)],
)
async def revoke_user_api(uid: str) -> JSONResponse:
    user = api_keys_store.get_user(uid)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    if user["revoked_ts"] is not None:
        return JSONResponse({"ok": True, "already_revoked": True})
    try:
        api_keys_store.revoke_user(uid)
    except api_keys_store.LastAdminError as e:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{e} — create another admin first",
        )
    return JSONResponse({"ok": True})


@router.get(
    "/api/users/{uid}/keys",
    dependencies=[Depends(require_admin_webui_host), Depends(require_admin)],
)
async def list_user_keys_api(uid: str) -> JSONResponse:
    if api_keys_store.get_user(uid) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    return JSONResponse({"keys": api_keys_store.list_keys(user_id=uid)})


@router.get(
    "/api/usage",
    dependencies=[Depends(require_admin_webui_host), Depends(require_admin)],
)
async def usage_api(days: int = 0) -> JSONResponse:
    """Per-user and per-key usage rollup for the cards. `days=0` (default)
    is lifetime; `days=N` is the trailing N-day window. Returned as id-keyed
    maps so the front-end can join onto the users + keys it already renders
    (no server-side name resolution needed). Per-user totals include every
    one of that user's keys plus any pre-feature backfilled usage; per-key
    totals cover only real keys (backfill has no key id)."""
    import usage_store
    # Window in UTC epoch-hours; the N-day window is reckoned in the server's
    # local timezone (admin/operator perspective). days=0 => lifetime.
    start_hour = None
    if days and days > 0:
        start_hour = usage_store.local_day_start_hour(days_ago=int(days) - 1)
    by_user = usage_store.totals_by_user(start_hour=start_hour)
    by_key = {
        r["key_id"]: r
        for r in usage_store.totals_by_key(start_hour=start_hour)
    }
    return JSONResponse({"by_user": by_user, "by_key": by_key, "days": days})


@router.post(
    "/api/users/{uid}/keys",
    dependencies=[Depends(require_admin_webui_host), Depends(require_admin)],
)
async def create_user_key_api(uid: str, payload: CreateKeyIn) -> JSONResponse:
    """Show-once raw key on creation. Subsequent reads via list_user_keys
    never return the raw value. A label is mandatory at this boundary (the
    store stays lenient for internal/test callers); blank/whitespace -> 400."""
    label = payload.label.strip()
    if not label:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "label is required")
    try:
        raw_key, rec = api_keys_store.create_key(uid, label=label)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return JSONResponse({"key": raw_key, "record": rec})


@router.delete(
    "/api/users/{uid}/keys/{kid}",
    dependencies=[Depends(require_admin_webui_host), Depends(require_admin)],
)
async def revoke_key_api(uid: str, kid: str) -> JSONResponse:
    key = api_keys_store.get_key(kid)
    if key is None or key["user_id"] != uid:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "key not found")
    if key["revoked_ts"] is not None:
        return JSONResponse({"ok": True, "already_revoked": True})
    try:
        api_keys_store.revoke_key(kid)
    except api_keys_store.LastAdminError as e:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{e} — generate another admin key first",
        )
    return JSONResponse({"ok": True})


@router.patch(
    "/api/users/{uid}/permissions",
    dependencies=[Depends(require_admin_webui_host), Depends(require_admin)],
)
async def patch_user_permissions_api(
    uid: str, payload: PatchPermissionsIn,
) -> JSONResponse:
    """PATCH-merge per-page permissions onto a user. Returns the
    canonical post-merge shape so the matrix UI can echo it back into
    its rendered state without a second GET.

    Admins still validate + persist (so a future demote-to-non-admin
    path picks up the saved defaults) but their `is_admin` flag
    short-circuits all page checks at request time — the matrix UI
    greys their row out for clarity."""
    target = api_keys_store.get_user(uid)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    if target["revoked_ts"] is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "user is revoked")
    try:
        merge_payload: dict[str, Any] = {"pages": payload.pages}
        if payload.quick_config_tags is not None:
            merge_payload["quick_config_tags"] = payload.quick_config_tags
        if payload.config is not None:
            merge_payload["config"] = payload.config.model_dump()
        merged = api_keys_store.set_user_permissions(uid, merge_payload)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return JSONResponse({"ok": True, "permissions": merged})


@router.patch(
    "/api/users/{uid}/keys/{kid}/label",
    dependencies=[Depends(require_admin_webui_host), Depends(require_admin)],
)
async def rename_key_api(
    uid: str, kid: str, payload: RenameKeyIn,
) -> JSONResponse:
    """Rename an existing key's display label. 404 if the key isn't this
    user's; 409 if revoked (revoked keys are read-only); 400 on a blank or
    over-long label. Renaming never exposes or rotates the secret."""
    key = api_keys_store.get_key(kid)
    if key is None or key["user_id"] != uid:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "key not found")
    if key["revoked_ts"] is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "key is revoked")
    try:
        rec = api_keys_store.update_key_label(kid, payload.label)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    if rec is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "key not found")
    return JSONResponse({"ok": True, "record": rec})


@router.patch(
    "/api/users/{uid}/keys/{kid}/config",
    dependencies=[Depends(require_admin_webui_host), Depends(require_admin)],
)
async def patch_key_config_api(
    uid: str, kid: str, payload: ConfigBindingIn,
) -> JSONResponse:
    """Validate + persist a per-key config binding (override profiles + direct
    overrides + locks). Returns the stored {"direct": …, "profiles": …} so the
    drawer can echo it back. 404 if the key isn't this user's; 409 if revoked."""
    key = api_keys_store.get_key(kid)
    if key is None or key["user_id"] != uid:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "key not found")
    if key["revoked_ts"] is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "key is revoked")
    try:
        binding = api_keys_store.set_key_config(uid, kid, payload.model_dump())
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return JSONResponse({"ok": True, "config": binding})


# ---------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------

_API_KEYS_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{HEADER_TITLE}}</title>
{{PAGE_META}}
{{SCALE_BOOTSTRAP_HEAD}}
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --fg: #c9d1d9; --dim: #6e7681;
    --cyan: #79c0ff; --green: #7ee787; --yellow: #f2cc60;
    --red: #ff7b72; --bold: #f0f6fc; --border: #30363d; --input-bg: #0d1117;
    --help: #8b949e;
  }
  html, body { background: var(--bg); color: var(--fg);
    font-family: var(--font-sans); font-size: var(--fs-base); margin: 0; }
  a { color: var(--cyan); }
  header button { background: var(--panel); border: 1px solid var(--border);
    color: var(--fg); padding: 0.25rem 0.625rem; border-radius: 4px;
    cursor: pointer; font: inherit; font-size: var(--fs-sm); }
  header button.primary { color: var(--green); border-color: var(--green); }
  /* Widened from 56rem to claim the unused right gutter on the user-key
     cards — admin keys + permissions matrix benefit from extra horizontal
     space and the per-row layout reads better without forced wrapping. */
  main { padding: 1rem; max-width: 72rem; margin: 0 auto; }
  .banner-open {
    background: #5a2424; color: #fff; padding: 0.6rem 1rem;
    text-align: center; font-weight: 600;
  }
  .card { background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.75rem 1rem; margin-bottom: 0.75rem; }
  .card h3 { margin: 0 0 0.25rem 0; font-size: var(--fs-lg);
    color: var(--bold); display: flex; align-items: center; gap: 0.6rem;
    flex-wrap: wrap; }
  .pill { font-size: var(--fs-xs); padding: 0.075rem 0.5rem;
    border-radius: 999px; border: 1px solid var(--border); color: var(--dim);
    font-weight: normal; }
  .pill.admin { color: var(--yellow); border-color: #4d3e1f; }
  .pill.revoked { color: var(--red); border-color: #5a2424;
    background: #2d1414; }
  .pill.live { color: var(--green); border-color: #1d4f2c; }
  /* ---- collapsible per-user section ---- */
  .card.user { padding: 0; overflow: hidden; }
  .card.user .user-head { margin: 0; padding: 0.7rem 1rem; cursor: pointer;
    user-select: none; }
  .card.user .user-head:hover { background: #10151c; }
  .card.user .user-head .uchev { color: var(--dim); font-size: 0.85em;
    display: inline-block; transition: transform 150ms ease; }
  .card.user.open .user-head .uchev { transform: rotate(90deg); }
  .card.user .user-body { display: none; padding: 0 1rem 0.85rem;
    border-top: 1px solid var(--border); }
  .card.user.open .user-body { display: block; }

  /* ---- rule-card key row (mirrors the /settings/pipeline rule cards) ---- */
  .kcard { display: grid; grid-template-columns: 2.4rem 1fr auto;
    border: 1px solid var(--border); border-radius: 9px; background: var(--panel);
    margin-top: 0.55rem; overflow: hidden;
    transition: border-color 120ms ease, box-shadow 120ms ease; }
  .kcard:hover { border-color: var(--border2);
    box-shadow: 0 6px 18px -12px rgba(0, 0, 0, 0.85); }
  .kcard.revoked { opacity: 0.6; }
  .krail { display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 0.3rem; font-size: 1rem; color: var(--dim);
    background: linear-gradient(180deg, #1b222c, var(--panel));
    border-right: 1px solid var(--border); }
  /* Status dot: green = used within the recent window, dim-grey = idle,
     red = revoked. Decays green->grey live via _refreshApiKeyDots on a timer.
     Colour is a secondary cue only — the active/revoked pill + tooltip carry
     the meaning (WCAG 1.4.1). */
  .kdot { width: 0.5rem; height: 0.5rem; border-radius: 50%;
    background: var(--dim); }
  .kdot.live { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .kdot.idle { background: var(--dim); box-shadow: none; }
  .kdot.dead { background: var(--red); box-shadow: none; }
  .kbody { padding: 0.55rem 0.2rem 0.55rem 0.7rem; min-width: 0; }
  .ktitle { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
  .ktitle .kname { color: var(--bold); font-weight: 600; word-break: break-word; }
  .ktitle .kname.none { color: var(--dim); font-style: italic;
    font-weight: normal; }
  /* Meta stacks vertically: key id, then a created/used (or created/revoked)
     timestamp grid so the two dates line up column-for-column and are easy to
     compare. Mono values keep the digits tabular. */
  .kmeta { display: flex; flex-direction: column; gap: 0.3rem; margin-top: 0.3rem;
    font-size: var(--fs-sm); color: var(--dim); }
  .kmeta .kid { color: var(--fg); font-family: var(--font-mono); }
  .kmeta .renote { color: var(--yellow); }
  .ktimes { display: grid; grid-template-columns: auto auto;
    justify-content: start; gap: 0.15rem 0.6rem; align-items: baseline; }
  .ktimes .kt-lbl { color: var(--dim); }
  .ktimes .kt-val { color: var(--fg); font-family: var(--font-mono); }
  .kacts { display: flex; align-items: center; gap: 0.4rem;
    padding: 0.55rem 0.6rem; flex-wrap: wrap; justify-content: flex-end; }
  /* inline rename editor under the key title */
  .krename { display: flex; align-items: center; gap: 0.4rem;
    margin-top: 0.35rem; flex-wrap: wrap; }
  .krename input[type=text] { width: auto; flex: 1; min-width: 8rem;
    padding: 0.3rem 0.5rem; font-size: var(--fs-sm); }
  /* Phone: drop the action column under the body so nothing is cramped. */
  @media (max-width: 40em) {
    .kcard { grid-template-columns: 2.4rem 1fr; }
    .kacts { grid-column: 1 / -1; justify-content: flex-start;
      border-top: 1px solid var(--border); }
  }
  /* Per-key usage: a row of compact stat chips (value over caption). Mono
     values so digits stay tabular; dim captions. Wraps gracefully on narrow
     viewports. */
  /* Two-row grid: every stat's value lands on row 1, its caption on row 2,
     so values align across stats no matter their text width/content (a
     plain flex row left the % stat baseline-drifting). .stat is display:
     contents so its value+caption become direct grid items. */
  .usage-cell { display: inline-grid; grid-auto-flow: column;
    grid-template-rows: auto auto; align-items: start; justify-content: start;
    gap: 0.05rem 0.9rem; line-height: 1.15; }
  .usage-cell .stat { display: contents; }
  .usage-cell .stat .v { grid-row: 1; color: var(--fg);
    font-family: var(--font-mono); font-size: var(--fs-sm); white-space: nowrap; }
  .usage-cell .stat .k { grid-row: 2; color: var(--dim); font-size: var(--fs-xs);
    text-transform: lowercase; }
  .usage-cell .stat.err .v { color: var(--yellow); }
  .usage-cell.empty { display: block; color: var(--dim); font-style: italic;
    font-size: var(--fs-xs); }
  /* Per-user summary strip under the card header: one mono line of the
     user's lifetime totals across all their keys. Sits in the card header,
     pushed right (margin-left:auto); doesn't wrap internally. */
  .user-usage { display: flex; flex-wrap: wrap; gap: 0.15rem 1.1rem;
    margin-left: auto; }
  .user-usage .stat { display: flex; align-items: baseline; gap: 0.3rem; }
  .user-usage .stat .v { color: var(--cyan); font-family: var(--font-mono);
    font-size: var(--fs-md); }
  .user-usage .stat .k { color: var(--dim); font-size: var(--fs-xs);
    text-transform: lowercase; }
  /* "last active" leads the strip: neutral mono timestamp (the cyan stays a
     cue for the numeric aggregates), label first so it reads "last active
     <when>". Updated live by timeTick() via the [data-ts] span metaWhen emits. */
  .user-usage .stat.lastact .v { color: var(--fg); font-size: var(--fs-sm); }
  /* User-level activity dot in the header: reuses .kdot (green = any non-
     revoked key used in the last 15 min, grey = idle), decayed by the same
     30s _refreshApiKeyDots timer via data-used-ts. flex:none keeps its size
     as a flex child of the h3 header row (.card h3 is display:flex). */
  .user-head .kdot { flex: none; }
  /* Buttons inside <main>: a quiet ghost baseline so nothing is ever browser-
     default, plus filled .primary and red .danger — mirrors the /pipeline
     button language. (Modal + header + matrix buttons carry their own,
     higher-specificity rules and are unaffected.) */
  main button { font: inherit; font-size: var(--fs-sm); cursor: pointer;
    line-height: 1.3; white-space: nowrap; border-radius: 6px;
    padding: 0.3rem 0.65rem; display: inline-flex; align-items: center;
    gap: 0.35rem; background: #21262d; color: var(--fg);
    border: 1px solid var(--border);
    transition: background 120ms ease, border-color 120ms ease, color 120ms ease; }
  main button:hover:not(:disabled) { background: #30363d; border-color: #484f58; }
  main button:disabled { opacity: 0.45; cursor: not-allowed; }
  main button.primary { background: #238636; border-color: #2ea043;
    color: var(--bold); }
  main button.primary:hover:not(:disabled) { background: #2ea043; }
  main button.ghost { background: transparent; color: var(--dim);
    border-color: transparent; }
  main button.ghost:hover:not(:disabled) { background: #1f2630;
    border-color: var(--border2); color: var(--fg); }
  main button.ghost.on { background: #1f2630; border-color: var(--border2);
    color: var(--bold); }
  main button.danger { background: transparent; color: var(--red);
    border-color: transparent; }
  main button.danger:hover:not(:disabled) { background: #3a0d0d;
    border-color: #5a2424; }
  /* The override-drawer micro-buttons (lock / remove) opt out of the padded
     baseline so they stay glyph-sized. */
  .cfg-ovr-row .lk, .cfg-ovr-row .rm { padding: 0 0.15rem; }
  .toolbar { display: flex; gap: 0.5rem; margin: 0.5rem 0; flex-wrap: wrap; }
  input[type=text], input[type=password] { box-sizing: border-box;
    width: 100%;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.5rem 0.65rem; font: inherit; font-size: var(--fs-md);
    line-height: 1.4; }
  input[type=text]:focus, input[type=password]:focus {
    outline: none; border-color: var(--cyan);
  }
  label.row { display: flex; gap: 0.5rem; align-items: center;
    margin: 0.3rem 0; font-size: var(--fs-sm); }
  .modal { position: fixed; inset: 0; display: none;
    align-items: center; justify-content: center;
    background: rgba(0,0,0,0.7); z-index: 100; }
  .modal.show { display: flex; }
  .modal .box { background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.4rem 1.5rem 1.2rem;
    width: 30rem; max-width: 92vw;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5); }
  .modal h3 { margin: 0 0 0.5rem 0; color: var(--bold);
    font-size: var(--fs-xl); display: flex; align-items: center;
    gap: 0.5rem; flex-wrap: wrap; }
  .modal h3 .pill { color: var(--cyan); border-color: #1f4a6b;
    font-size: var(--fs-sm); font-weight: normal; }
  .modal p { margin: 0 0 0.9rem 0; line-height: 1.45;
    color: var(--help); font-size: var(--fs-sm); }
  .modal p strong, .modal p code { color: var(--fg); }
  .modal p:last-of-type { margin-bottom: 1rem; }
  .modal input[type=text], .modal input[type=password] {
    /* Slightly taller inputs in modals (better tap target + reads
       cleaner next to the buttons below). */
    padding: 0.55rem 0.7rem; font-size: var(--fs-md);
  }
  .modal .keybox { position: relative; }
  .modal .raw-key {
    font-family: var(--font-mono); font-size: var(--fs-sm);
    word-break: break-all; padding: 0.65rem 3rem 0.65rem 0.75rem;
    background: var(--input-bg); color: var(--bold);
    border: 1px solid var(--border); border-radius: 4px;
    margin: 0.65rem 0 0; user-select: all;
    line-height: 1.5;
  }
  /* Inline copy affordance pinned to the key box — copying is always one
     click away regardless of which footer button the operator uses. */
  .modal .copy-icon { position: absolute; top: 0.95rem; right: 0.5rem;
    background: #1f2630; border: 1px solid var(--border); color: var(--dim);
    border-radius: 6px; padding: 0.2rem 0.5rem; cursor: pointer; font: inherit;
    font-size: var(--fs-sm); line-height: 1.3; }
  .modal .copy-icon:hover { color: var(--fg); border-color: var(--border2); }
  /* Callouts: amber = always-on "store it now"; red guard = shown only when
     the operator tries to close before copying. */
  .modal .warn { display: flex; gap: 0.5rem; align-items: flex-start;
    margin: 0.7rem 0 0; padding: 0.55rem 0.7rem; border-radius: 8px;
    font-size: var(--fs-sm); line-height: 1.4;
    background: rgba(242, 204, 96, 0.08); border: 1px solid #4d3e1f;
    color: var(--yellow); }
  .modal .warn strong { color: inherit; }
  .modal .warn.guard { display: none; background: rgba(255, 123, 114, 0.08);
    border-color: #5a2424; color: var(--red); }
  .modal .warn.guard.on { display: flex; }
  /* Uniform action-button sizing: same line-height, same padding, same
     min-height — so Cancel and Save render identical regardless of which
     border colour they carry. Avoid `padding: …` redeclaration in
     button.primary etc. so heights stay in sync. */
  .modal .actions {
    display: flex; gap: 0.6rem; align-items: center; justify-content: flex-end;
    margin-top: 1.1rem; padding-top: 0.85rem;
    border-top: 1px solid var(--border);
  }
  /* Copy-state note sits on the left of the footer; turns green once copied. */
  .modal .actions .copystate { margin-right: auto; color: var(--dim);
    font-size: var(--fs-xs); }
  .modal .actions .copystate.done { color: var(--green); }
  .modal .actions button {
    font: inherit; font-size: var(--fs-md);
    line-height: 1.4;
    padding: 0.45rem 1rem;
    min-height: 2.25rem;
    border-radius: 4px;
    cursor: pointer;
    background: var(--input-bg);
    color: var(--fg);
    border: 1px solid var(--border);
  }
  .modal .actions button:hover { background: #21262d; color: var(--bold); }
  .modal .actions button:disabled {
    opacity: 0.45; cursor: not-allowed; background: var(--input-bg);
  }
  .modal .actions button.primary {
    background: #238636; border-color: #2ea043; color: var(--bold);
  }
  .modal .actions button.primary:hover { background: #2ea043; color: var(--bold); }
  .modal .actions button.danger {
    color: var(--red); border-color: #5a2424;
  }
  .err { color: var(--red); font-size: var(--fs-sm); margin: 0.4rem 0 0 0; }
  .modal .err { margin: 0.55rem 0 0 0; }
  .modal .err:empty { display: none; }
  .hint { color: var(--help); font-size: var(--fs-sm); }

  /* Per-user × per-page permission matrix.
     User-rows × page-columns, each cell a tri-state select (none/own/all).
     Admin rows are greyed out and disabled — they bypass page checks at
     request time but we still render the row so a future demote path
     stays predictable. Dirty rows tint yellow until Save flushes. */
  .perm-matrix { width: 100%; border-collapse: collapse;
    font-size: var(--fs-sm); margin-top: 0.35rem; }
  .perm-matrix th, .perm-matrix td {
    padding: 0.35rem 0.5rem; border-bottom: 1px solid var(--border);
    text-align: left; vertical-align: middle; }
  .perm-matrix th { color: var(--dim); font-weight: 500;
    font-size: var(--fs-xs); text-transform: uppercase;
    letter-spacing: 0.04em; white-space: nowrap; }
  .perm-matrix th .tip { color: var(--dim); cursor: help;
    margin-left: 0.25rem; border-bottom: 1px dotted var(--dim); }
  .perm-matrix tbody td:first-child {
    color: var(--bold); font-weight: 500; }
  .perm-matrix tbody tr.admin-row { color: var(--dim); }
  .perm-matrix tbody tr.admin-row td:first-child { color: var(--dim); }
  .perm-matrix tbody tr.dirty td {
    background: rgba(242, 204, 96, 0.07); }
  /* Per-cell scope is a shared .status-btn-group (none/own/all) from NAV_CSS —
     keep adjacent cells from crowding, and trim the segment padding so the
     5-column matrix fits a normal screen without the wrap's scrollbar. */
  .perm-matrix td .status-btn-group { white-space: nowrap; }
  .perm-matrix td .status-btn { padding: 0.2rem 0.45rem; }
  .perm-matrix .admin-cell {
    color: var(--dim); font-style: italic; }
  /* The matrix is an inherently 2-D users×pages grid that doesn't stack into
     cards meaningfully — the documented exception to the .rcards pattern.
     Let it scroll horizontally and pin the username column so the row stays
     identifiable while scanning page permissions on a phone. */
  #perm-matrix-wrap { overflow-x: auto; }
  @media (max-width: 40em) {
    .perm-matrix th:first-child, .perm-matrix td:first-child {
      position: sticky; left: 0; z-index: 1; background: var(--panel); }
    .perm-matrix tbody tr.dirty td:first-child { background: #2a2718; }
  }
  .perm-matrix button.save-perms {
    background: var(--input-bg); border: 1px solid var(--border);
    color: var(--fg); padding: 0.2rem 0.65rem; border-radius: 3px;
    font: inherit; font-size: var(--fs-sm); cursor: pointer;
    opacity: 0.5; transition: opacity 0.15s ease; }
  .perm-matrix tbody tr.dirty button.save-perms {
    opacity: 1; color: var(--green); border-color: var(--green); }
  .perm-matrix button.save-perms:disabled {
    opacity: 0.2; cursor: not-allowed; }
  .perm-empty {
    color: var(--dim); padding: 0.6rem 0; font-style: italic;
    font-size: var(--fs-sm); }
  /* Visibility preview under each user's tag picker: "Will see: N of M".
     Caption-sized, dim. Goes muted when there are no exposed rules at
     all so the admin understands the preview is unactionable. */
  .perm-matrix .tag-preview {
    margin-top: 0.2rem; font-size: var(--fs-xs);
    color: var(--dim); font-family: var(--font-sans);
  }
  .perm-matrix .tag-preview.muted { font-style: italic; opacity: 0.7; }
  /* Per-identity config binding drawer (overrides) — three titled cards in a
     bordered container + a footer holding the actions. */
  .cfg-drawer { border: 1px solid var(--border); border-radius: 0.5rem;
    background: var(--bg); margin: 0.5rem 0; overflow: hidden; }
  .cfg-drawer .lbl { font-size: var(--fs-xs); color: var(--dim); }
  .ov-card { padding: 0.85rem 0.95rem; border-bottom: 1px solid var(--border); }
  .ov-card .ohead { display: flex; align-items: baseline; gap: 0.5rem; }
  .ov-card .ohead .t { font-size: var(--fs-md); font-weight: 600; color: var(--bold); }
  .ov-card .ohead .c { font-size: var(--fs-xs); color: var(--dim); }
  .ov-card .ohint { font-size: var(--fs-xs); color: var(--help);
    margin: 0.15rem 0 0.6rem; }
  /* one request gate = label left, segmented control right; uniform rows with
     hairlines so the set reads as a group (not a raw table, not loose selects) */
  .gaterow { display: grid; grid-template-columns: 1fr auto; align-items: center;
    gap: 0.8rem; padding: 0.4rem 0; border-top: 1px solid rgba(48, 54, 61, 0.55); }
  .gaterow:first-of-type { border-top: 0; }
  .gaterow .gl { font-size: var(--fs-sm); color: var(--fg); }
  .gaterow .gl .sub { display: block; font-family: var(--font-mono);
    font-size: var(--fs-xs); color: var(--dim); }
  .gaterow .eff { font-size: var(--fs-xs); color: var(--dim); justify-self: end; }
  /* restrict-to checkbox grid (replaces the former inline-style px) */
  .ov-checks { display: flex; flex-wrap: wrap; gap: 0.5rem 0.9rem; margin-top: 0.5rem; }
  .ov-checks label { display: inline-flex; align-items: center; gap: 0.25rem;
    font-size: var(--fs-sm); }
  .cfg-chips { display: flex; flex-wrap: wrap; gap: 0.3rem; align-items: center;
    margin: 0.2rem 0; }
  .cfg-chip { display: inline-flex; align-items: center; gap: 0.25rem;
    border: 1px solid var(--border); border-radius: 999px;
    padding: 0.1rem 0.5rem; font-size: var(--fs-sm); }
  .cfg-chip .num { font-family: var(--font-mono); color: var(--dim);
    font-size: var(--fs-xs); }
  .cfg-chip button { background: none; border: 0; color: var(--dim);
    cursor: pointer; font: inherit; padding: 0; }
  .cfg-chip button:hover { color: var(--fg); }
  /* "Apply no profiles" admin force (per-key only): a switch row above the chip
     list; when ON the list is dimmed + non-interactive (data kept, not cleared). */
  .cfg-noprof-row { display: flex; align-items: center; gap: 0.5rem;
    margin: 0.1rem 0 0.2rem; font-size: var(--fs-sm); }
  .cfg-noprof-row label { color: var(--fg); cursor: pointer; }
  .cfg-prof-dim { opacity: 0.45; pointer-events: none; }
  .cfg-ovr-row { display: grid;
    grid-template-columns: minmax(10rem, 1fr) minmax(7rem, 1fr) auto auto;
    gap: 0.4rem; align-items: center; padding: 0.12rem 0; }
  .cfg-ovr-row .nm { font-family: var(--font-mono); font-size: var(--fs-sm); }
  .cfg-ovr-row input[type=text], .cfg-ovr-row input[type=number],
  .cfg-ovr-row select { background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px; font: inherit;
    font-size: var(--fs-sm); padding: 0.1rem 0.3rem; width: 100%;
    box-sizing: border-box; }
  .cfg-ovr-row .lk { background: none; border: 0; cursor: pointer; color: var(--dim); }
  .cfg-ovr-row .lk.on { color: var(--yellow); }
  .cfg-ovr-row .rm { background: none; border: 0; cursor: pointer; color: var(--dim); }
  /* footer: exactly one primary (Save) on the right, Preview ghost on the left */
  .ov-footer { display: flex; align-items: center; gap: 0.5rem;
    padding: 0.65rem 0.95rem; background: var(--panel); }
  .ov-footer .spacer { flex: 1; }
  .cfg-eff { font-family: var(--font-mono); font-size: var(--fs-xs);
    border-top: 1px solid var(--border); padding: 0.5rem 0.95rem; }
  .cfg-eff .ef { display: flex; gap: 0.5rem; }
  .cfg-eff .ef .v { color: var(--green); } .cfg-eff .ef .src { color: var(--dim); }
  .cfg-eff .ef .lk { color: var(--yellow); }
  {{NAV_CSS}}
</style></head>
<body>

<div id="open-banner" class="banner-open" style="display:none">
  &#9888; No admin key set &mdash; the server is in OPEN mode and anyone who can
  reach it can use it. Generate the first admin key below.
</div>

<header>
  <div class="header-inner">
    <span class="title">{{HEADER_BRAND}}</span>
    <span class="brand-sep" aria-hidden="true"></span>
    {{NAV}}
    <span class="spacer"></span>
    <span class="hdr-right">{{SEV_PILLS}}{{SCALE_PICKER}}{{RELOAD}}{{LOGOUT}}</span>
  </div>
  <div class="subbar">
    <span class="subbar-title">API keys</span>
  </div>
</header>

<main>
  <div class="card">
    <h3>Add user</h3>
    <div class="toolbar">
      <input id="new-username" type="text" placeholder="username (e.g., Dr. Mueller)"
             style="flex: 1; max-width: 18rem;">
      <label class="row" style="margin: 0;">
        <input id="new-is-admin" type="checkbox" class="switch" role="switch"> admin
      </label>
      <button id="add-user-btn" class="primary">+ add user</button>
    </div>
    <p class="hint">
      Usernames are display names only &mdash; nothing about login. Each user
      can hold any number of API keys (one per device is the standard
      pattern).
    </p>
  </div>

  <div class="card" id="perm-matrix-card" style="display:none">
    <h3>Page permissions</h3>
    <p class="hint">
      Per-user access to each admin page. <strong>none</strong> hides
      the page entirely (403 on its API). <strong>own</strong> shows
      the page but filters records to the user's own; <strong>all</strong>
      shows every user's data. <code>/settings</code> and <code>/settings/api-keys</code>
      are always admin-only and never appear here.
    </p>
    <div id="perm-matrix-wrap"></div>
  </div>

  <div id="users-container"></div>
</main>

<!-- Show-once raw key modal -->
<div id="key-modal" class="modal">
  <div class="box">
    <h3>&#128273; New API key <span class="pill" id="key-modal-label"></span></h3>
    <p>New key for <strong id="key-modal-user"></strong>. This is the only time
    the full key is shown &mdash; anyone who has it gets the same access.</p>
    <div class="keybox">
      <div class="raw-key" id="key-modal-raw"></div>
      <button id="key-modal-copyicon" class="copy-icon" type="button"
              title="Copy to clipboard">&#10697; copy</button>
    </div>
    <div class="warn">&#9888; Store it in a password manager now. To replace it,
    generate a new key &mdash; it can't be shown again.</div>
    <div class="warn guard" id="key-modal-guard">You haven't copied the key yet
    &mdash; closing now means it's gone for good. Copy it first, or press
    <strong>Close</strong> again to dismiss anyway.</div>
    <div class="actions">
      <span class="copystate" id="key-modal-copystate">not copied yet</span>
      <button id="key-modal-close" type="button">Close</button>
      <button id="key-modal-copyclose" class="primary" type="button">&#10697; Copy &amp; close</button>
    </div>
  </div>
</div>


{{SCALE_PICKER_JS}}
{{SEV_POLLER_JS}}
{{TIME_HELPERS_JS}}
{{TAG_PICKER_JS}}
<script>
(function() {
  'use strict';

  // Leading glyph for every key card's rail. A line-art key whose stroke uses
  // currentColor, so it inherits the rail's var(--dim) tone and sits at the
  // same weight as the page's other glyphs (✎ ⚙ ▸) — unlike a colour emoji.
  var KEY_ICON_SVG =
    '<svg viewBox="0 0 24 24" width="1.15em" height="1.15em" fill="none"'
    + ' stroke="currentColor" stroke-width="1.7" stroke-linecap="round"'
    + ' stroke-linejoin="round" aria-hidden="true">'
    + '<path d="M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1'
    + 'v-1a1 1 0 0 1 1-1h1a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 1.414-.586'
    + 'l.814-.814a6.5 6.5 0 1 0-4-4z"/>'
    + '<circle cx="16.5" cy="7.5" r=".6" fill="currentColor"/></svg>';

  // Sign-in is handled by the shared full-screen login gate (web_common):
  // on a 401 we call window._showLoginGate(), which prompts + reloads.
  async function api(method, path, body) {
    // The HttpOnly session cookie is sent automatically; mutations also
    // carry the double-submit CSRF token.
    var h = { Accept: 'application/json' };
    if (method !== 'GET' && method !== 'HEAD') {
      h['X-CSRF-Token'] = window._csrfToken ? window._csrfToken() : '';
    }
    var opts = { method: method, headers: h };
    if (body !== undefined) {
      h['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    return fetch(path, opts);
  }

  {{NOT_ADMIN_LANDING_JS}}

  async function _check403(r) {
    if (!r || r.status !== 403) return false;
    try {
      var who = await fetch('/auth/whoami');
      if (who.ok) {
        var j = await who.json();
        // Cache whoami so the landing can list pages the caller CAN reach.
        try { window.__whoami = j; } catch(_) {}
        if (j && j.is_admin === false) {
          _renderNoAccessLanding({ page: 'api-keys' });
          return true;
        }
      }
    } catch (_) {}
    return false;
  }

  function showToast(msg, kind) {
    var el = document.getElementById('toast');
    if (!el) {
      el = document.createElement('div');
      el.id = 'toast';
      el.style.position = 'fixed';
      el.style.bottom = '1rem';
      el.style.right = '1rem';
      el.style.padding = '0.6rem 1rem';
      el.style.borderRadius = '4px';
      el.style.zIndex = '200';
      el.style.fontSize = 'var(--fs-sm)';
      document.body.appendChild(el);
    }
    el.textContent = msg;
    el.style.background = kind === 'err' ? '#5a2424'
                       : kind === 'ok'  ? '#1d4f2c'
                       : '#21262d';
    el.style.color = '#fff';
    el.style.display = 'block';
    setTimeout(function(){ el.style.display = 'none'; }, 3000);
  }


  function showKeyModal(rawKey, username, label) {
    var m = document.getElementById('key-modal');
    document.getElementById('key-modal-user').textContent = username;
    var lp = document.getElementById('key-modal-label');
    lp.textContent = label || '';
    lp.style.display = label ? '' : 'none';
    document.getElementById('key-modal-raw').textContent = rawKey;

    // Two independent guards against losing the one-time secret:
    //  • `copied` flips true on any successful copy (icon or Copy & close);
    //  • `closeArmed` makes a bare "Close" warn once, then close on a second
    //    press — so a stray click can't silently discard an uncopied key.
    var copied = false, closeArmed = false;
    var guard = document.getElementById('key-modal-guard');
    var state = document.getElementById('key-modal-copystate');
    guard.classList.remove('on');
    state.textContent = 'not copied yet';
    state.classList.remove('done');

    function doCopy() {
      // navigator.clipboard needs a secure context (https / localhost). Over
      // LAN HTTP it's undefined, so fall back to a hidden textarea +
      // execCommand('copy'); if both fail, select the .raw-key span for Ctrl+C.
      function copyFallback(text) {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.top = '-1000px';
        ta.setAttribute('readonly', '');
        document.body.appendChild(ta);
        ta.select();
        ta.setSelectionRange(0, ta.value.length);
        var ok = false;
        try { ok = document.execCommand('copy'); } catch (_) {}
        document.body.removeChild(ta);
        return ok;
      }
      function onSuccess() {
        copied = true;
        guard.classList.remove('on');
        state.textContent = '✓ copied to clipboard';
        state.classList.add('done');
        showToast('Copied to clipboard', 'ok');
      }
      function onFailure() {
        var span = document.getElementById('key-modal-raw');
        var sel = window.getSelection();
        var range = document.createRange();
        range.selectNodeContents(span);
        sel.removeAllRanges();
        sel.addRange(range);
        showToast('Auto-copy blocked — press Ctrl+C', 'err');
      }
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(rawKey).then(onSuccess, function () {
          if (copyFallback(rawKey)) onSuccess(); else onFailure();
        });
      } else {
        if (copyFallback(rawKey)) onSuccess(); else onFailure();
      }
    }
    function doClose() {
      m.classList.remove('show');
      document.removeEventListener('keydown', onKey);
      load();  // rebuilds the user cards, so the label input also resets
    }
    function tryClose() {
      if (copied || closeArmed) { doClose(); return; }
      closeArmed = true;
      guard.classList.add('on');
    }
    function onKey(e) { if (e.key === 'Escape') tryClose(); }

    document.getElementById('key-modal-copyicon').onclick = doCopy;
    document.getElementById('key-modal-copyclose').onclick = function () {
      doCopy(); doClose();
    };
    document.getElementById('key-modal-close').onclick = tryClose;
    document.addEventListener('keydown', onKey);
    m.classList.add('show');
  }

  // Tooltips describe what "own" means per page. Kept short — full
  // help lives in the page card's intro paragraph.
  var PAGE_TIPS = {
    quick_config: 'own = this user’s submitted chips + their own recent traces',
    captures:     'own = only audio + transcripts dictated under this user’s key',
    reports:      'own = only reports this user submitted',
    stats:        'system aggregates — no per-user view (none|all only)',
    logs:         'server-wide log stream — no per-user view (none|all only)'
  };

  // Build the segmented none/own/all (or none/all for access-only pages)
  // control for one page cell — replaces the old <select>. role=radiogroup +
  // role=radio/aria-checked; the active segment carries a semantic data-tone
  // fill (none=grey, own=blue, all=amber). Clicking sets active + marks the
  // row dirty (same effect the old select.change had). Shared .status-btn-group
  // styling comes from web_common.NAV_CSS.
  // Saved-state signature for one permission row: the segmented page scopes
  // plus the (order-insensitive) quick-config tag set, canonicalised to a
  // string. recomputePermDirty() compares the live row against the baseline
  // captured at render / after a successful save — so reverting an edit back
  // to the saved value clears the dirty tint + Save affordance, instead of
  // latching dirty on the first change (which left Save lit even when nothing
  // actually differed from what was saved).
  function permRowState(tr) {
    var pages = {};
    tr.querySelectorAll('.status-btn-group[data-page]').forEach(function(g) {
      var act = g.querySelector('.status-btn.active');
      pages[g.dataset.page] = act ? act.dataset.val : 'none';
    });
    var canon = {};
    Object.keys(pages).sort().forEach(function(k) { canon[k] = pages[k]; });
    var tags = tr._tagPicker ? tr._tagPicker.getTags().slice().sort() : [];
    return JSON.stringify({ pages: canon, tags: tags });
  }
  function recomputePermDirty(tr) {
    tr.classList.toggle('dirty', permRowState(tr) !== tr._permBaseline);
  }

  var SCOPE_LABEL = { none: 'None', own: 'Own', all: 'All' };
  function _permSeg(page, allowed, cur, tr) {
    var grp = document.createElement('span');
    grp.className = 'status-btn-group';
    grp.dataset.page = page;
    grp.setAttribute('role', 'radiogroup');
    allowed.forEach(function(s) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'status-btn' + (s === cur ? ' active' : '');
      b.dataset.val = s;
      b.dataset.tone = s;
      b.textContent = SCOPE_LABEL[s] || s;
      b.setAttribute('role', 'radio');
      b.setAttribute('aria-checked', s === cur ? 'true' : 'false');
      b.addEventListener('click', function() {
        if (b.classList.contains('active')) return;
        grp.querySelectorAll('.status-btn').forEach(function(x) {
          x.classList.remove('active');
          x.setAttribute('aria-checked', 'false');
        });
        b.classList.add('active');
        b.setAttribute('aria-checked', 'true');
        recomputePermDirty(tr);
      });
      grp.appendChild(b);
    });
    return grp;
  }

  function renderMatrix(j) {
    var card = document.getElementById('perm-matrix-card');
    var wrap = document.getElementById('perm-matrix-wrap');
    wrap.innerHTML = '';
    var pages = j.pages || [];
    if (!pages.length || !j.users || !j.users.length) {
      card.style.display = 'none';
      return;
    }
    card.style.display = 'block';

    var accessOnly = new Set(j.access_only_pages || []);
    var nonAdminUsers = j.users.filter(function(u){ return !u.is_admin; });
    if (!nonAdminUsers.length) {
      wrap.innerHTML =
        '<div class="perm-empty">No non-admin users yet. Permissions only ' +
        'apply to non-admin users — admins bypass all page checks.</div>';
      return;
    }

    // Tag-picker autocomplete source: union of every tag currently set
    // on any pipeline rule. Computed server-side (list_users_api) so
    // an admin tagging a rule on /settings sees the new tag in the
    // matrix autocomplete after a reload.
    var availableTags = j.available_tags || [];
    // Per-exposed-rule tag list, used by the per-row "Will see: N of M"
    // visibility preview. Computed live as the admin edits a user's
    // tags — no extra roundtrip.
    var exposedRuleTags = j.exposed_rule_tags || [];

    var table = document.createElement('table');
    table.className = 'perm-matrix';
    var thead = document.createElement('thead');
    var headRow = document.createElement('tr');
    var thUser = document.createElement('th');
    thUser.textContent = 'User';
    thUser.style.minWidth = '10rem';
    headRow.appendChild(thUser);
    pages.forEach(function(p) {
      var th = document.createElement('th');
      th.textContent = '/' + p.replace(/_/g, '-');
      var tip = document.createElement('span');
      tip.className = 'tip';
      tip.textContent = '?';
      tip.title = PAGE_TIPS[p] || '';
      th.appendChild(tip);
      headRow.appendChild(th);
    });
    // Quick-config tag picker column. Tooltip explains the asymmetric
    // empty-list semantics (user with no tags sees only untagged rules).
    var thTags = document.createElement('th');
    thTags.textContent = 'quick-config tags';
    var tagsTip = document.createElement('span');
    tagsTip.className = 'tip';
    tagsTip.textContent = '?';
    tagsTip.title = 'Per-user filter for /quick-config rules. A user '
      + 'sees a rule when their tags intersect the rule’s tags '
      + '— OR when the rule has no tags (visible to everyone). '
      + 'A user with no tags sees only untagged rules. Admins see all.';
    thTags.appendChild(tagsTip);
    headRow.appendChild(thTags);
    var thSave = document.createElement('th');
    thSave.textContent = '';
    headRow.appendChild(thSave);
    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = document.createElement('tbody');
    j.users.forEach(function(u) {
      var tr = document.createElement('tr');
      tr.dataset.uid = u.id;
      if (u.is_admin) tr.classList.add('admin-row');

      var tdUser = document.createElement('td');
      tdUser.textContent = u.username + (u.is_admin ? ' (admin)' : '');
      tr.appendChild(tdUser);

      var currentPages = (u.permissions && u.permissions.pages) || {};

      pages.forEach(function(page) {
        var td = document.createElement('td');
        if (u.is_admin) {
          td.className = 'admin-cell';
          td.textContent = '—';
          td.title = 'admin bypasses all page + scope checks';
        } else {
          var allowed = accessOnly.has(page)
            ? ['none', 'all']
            : ['none', 'own', 'all'];
          var cur = allowed.indexOf(currentPages[page]) >= 0
            ? currentPages[page] : 'none';
          td.appendChild(_permSeg(page, allowed, cur, tr));
        }
        tr.appendChild(td);
      });

      // Per-user quick-config tag picker. Stash the controller on the
      // row so savePerms() can read the current tag list at submit time
      // without scanning DOM. Admins get a greyed-out "—" cell.
      var tdTags = document.createElement('td');
      if (u.is_admin) {
        tdTags.className = 'admin-cell';
        tdTags.textContent = '—';
        tdTags.title = 'admins see every rule regardless of tags';
      } else {
        var preview = document.createElement('div');
        preview.className = 'tag-preview';
        function _refreshPreview(userTags) {
          var total = exposedRuleTags.length;
          var seen = 0;
          var hasTag = function(t) { return userTags.indexOf(t) !== -1; };
          for (var i = 0; i < exposedRuleTags.length; i++) {
            var rt = exposedRuleTags[i];
            if (!rt.length) { seen++; continue; }   // untagged = visible
            for (var k = 0; k < rt.length; k++) {
              if (hasTag(rt[k])) { seen++; break; }
            }
          }
          if (!total) {
            preview.textContent = 'No rules exposed yet.';
            preview.className = 'tag-preview muted';
          } else if (userTags.length === 0) {
            preview.textContent = 'Will see: ' + seen + ' of ' + total
              + ' (untagged only)';
            preview.className = 'tag-preview';
          } else {
            preview.textContent = 'Will see: ' + seen + ' of ' + total
              + ' exposed rule' + (total === 1 ? '' : 's');
            preview.className = 'tag-preview';
          }
        }
        var picker = window._renderTagPicker({
          initial: (u.permissions && u.permissions.quick_config_tags) || [],
          available: availableTags,
          placeholder: '+ tag',
          onChange: function(newTags) {
            recomputePermDirty(tr);
            _refreshPreview(newTags);
          },
        });
        tr._tagPicker = picker;
        tdTags.appendChild(picker.el);
        tdTags.appendChild(preview);
        _refreshPreview(picker.getTags());
      }
      tr.appendChild(tdTags);

      var tdSave = document.createElement('td');
      if (!u.is_admin) {
        var btn = document.createElement('button');
        btn.className = 'save-perms';
        btn.textContent = 'Save';
        btn.title = 'Save this row';
        btn.addEventListener('click', function() {
          savePerms(u.id, tr, btn);
        });
        tdSave.appendChild(btn);
      }
      tr.appendChild(tdSave);

      // Snapshot the saved state now that every cell (segments + tag picker)
      // is attached; subsequent edits diff against this.
      tr._permBaseline = permRowState(tr);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
  }

  function savePerms(uid, tr, btn) {
    var pages = {};
    tr.querySelectorAll('.status-btn-group[data-page]').forEach(function(g) {
      var act = g.querySelector('.status-btn.active');
      pages[g.dataset.page] = act ? act.dataset.val : 'none';
    });
    var body = { pages: pages };
    // Tag picker is row-scoped; the controller was stashed at render.
    if (tr._tagPicker) body.quick_config_tags = tr._tagPicker.getTags();
    btn.disabled = true;
    api('PATCH',
        '/settings/api-keys/api/users/' + encodeURIComponent(uid) + '/permissions',
        body)
      .then(function(r) {
        if (!r.ok) return r.text().then(function(t) { throw new Error(t); });
        return r.json();
      })
      .then(function() {
        // The just-saved state is the new baseline; recompute clears dirty.
        tr._permBaseline = permRowState(tr);
        recomputePermDirty(tr);
        showToast('Permissions saved', 'ok');
      })
      .catch(function(e) {
        showToast(String(e.message || e), 'err');
      })
      .finally(function() {
        btn.disabled = false;
      });
  }

  // ---- small render helpers (status dot + relative-time + collapse state) ----
  function metaWhen(ts) {
    if (!ts) return '—';
    return '<span data-ts="' + ts + '" title="' + escapeHtml(absTime(ts)) + '">'
      + escapeHtml(fmtWhen(ts)) + '</span>';
  }
  var RECENT_USE_MS = 15 * 60 * 1000;  // green dot = used within this window
  function dotClassFor(lastUsedTs) {
    if (!lastUsedTs) return 'idle';
    return (Date.now() - lastUsedTs * 1000) <= RECENT_USE_MS ? 'live' : 'idle';
  }
  // Recompute the live/idle dots in place so a key crossing the recent-use
  // window decays green->grey without a reload. Revoked dots carry no
  // data-used-ts and keep their static red.
  function refreshKeyDots() {
    var now = Date.now();
    document.querySelectorAll('.kdot[data-used-ts]').forEach(function (d) {
      var ts = parseFloat(d.getAttribute('data-used-ts')) || 0;
      var live = ts && (now - ts * 1000) <= RECENT_USE_MS;
      d.classList.toggle('live', !!live);
      d.classList.toggle('idle', !live);
    });
  }
  window._refreshApiKeyDots = refreshKeyDots;

  function userCollapseKey(uid) { return 'apikeys.collapsed.' + uid; }
  // First visit (no stored pref): expand when the list is short, collapse when
  // it's long. Stored '0' = expanded, '1' = collapsed.
  function isUserOpen(uid, nUsers) {
    var v = null;
    try { v = localStorage.getItem(userCollapseKey(uid)); } catch (_) {}
    if (v === '0') return true;
    if (v === '1') return false;
    return nUsers <= 5;
  }
  function setUserOpen(uid, open) {
    try { localStorage.setItem(userCollapseKey(uid), open ? '0' : '1'); } catch (_) {}
  }

  function renderUser(u, nUsers) {
    var card = document.createElement('div');
    card.className = 'card user';
    if (isUserOpen(u.id, nUsers)) card.classList.add('open');

    // ---- clickable collapsible header ----
    var head = document.createElement('h3');
    head.className = 'user-head';
    var chev = document.createElement('span');
    chev.className = 'uchev'; chev.textContent = '▸';
    head.appendChild(chev);
    // User-level activity dot: green if ANY non-revoked key was used in the
    // last 15 min (max last_used_ts over the user's keys, server-computed).
    // data-used-ts opts it into the same 30s refreshKeyDots() decay timer as
    // the per-key dots; dotClassFor() never returns 'dead' here (no ts → idle).
    var udotCls = dotClassFor(u.last_used_ts);
    var udot = document.createElement('span');
    udot.className = 'kdot ' + udotCls;
    if (u.last_used_ts) udot.setAttribute('data-used-ts', u.last_used_ts);
    udot.title = udotCls === 'live'
      ? 'a key was used in the last 15 min' : 'no recent key use';
    head.appendChild(udot);
    var nameEl = document.createElement('span');
    nameEl.textContent = u.username;
    head.appendChild(nameEl);
    var pill = document.createElement('span');
    pill.className = 'pill ' + (u.is_admin ? 'admin' : '');
    pill.textContent = u.is_admin ? 'admin' : 'user';
    head.appendChild(pill);
    var keyCount = document.createElement('span');
    keyCount.className = 'pill';
    keyCount.textContent = u.active_key_count + ' active key' +
      (u.active_key_count === 1 ? '' : 's');
    head.appendChild(keyCount);
    // Lifetime usage strip, pushed right (margin-left:auto) so it fills the
    // space beside the name instead of crowding a line below.
    var usageStrip = document.createElement('div');
    usageStrip.innerHTML = userUsageHtml(
      (window.__usage && window.__usage.by_user || {})[u.id], u.last_used_ts);
    head.appendChild(usageStrip.firstChild);
    head.onclick = function () {
      var open = !card.classList.contains('open');
      card.classList.toggle('open', open);
      setUserOpen(u.id, open);
    };
    card.appendChild(head);

    var userBody = document.createElement('div');
    userBody.className = 'user-body';
    card.appendChild(userBody);

    // ---- generate / overrides / revoke-user toolbar ----
    var tb = document.createElement('div');
    tb.className = 'toolbar';
    var labelInp = document.createElement('input');
    labelInp.type = 'text';
    labelInp.placeholder = 'key label — required';
    labelInp.maxLength = 128;
    labelInp.style.maxWidth = '14rem';
    labelInp.style.flex = '1';
    var addBtn = document.createElement('button');
    addBtn.className = 'primary';
    addBtn.textContent = '+ generate key';
    addBtn.disabled = true;                 // labels are mandatory
    addBtn.title = 'Enter a label first';
    labelInp.oninput = function () {
      var has = !!labelInp.value.trim();
      addBtn.disabled = !has;
      addBtn.title = has ? '' : 'Enter a label first';
    };
    addBtn.onclick = function () {
      var label = labelInp.value.trim();
      if (!label) { showToast('Label is required', 'err'); labelInp.focus(); return; }
      addBtn.disabled = true;
      api('POST', '/settings/api-keys/api/users/' + encodeURIComponent(u.id) + '/keys',
          { label: label })
        .then(function (r) {
          if (!r.ok) return r.text().then(function (t) { throw new Error(t); });
          return r.json();
        })
        .then(function (j) { showKeyModal(j.key, u.username, label); })
        .catch(function (e) {
          showToast(String(e.message || e), 'err');
          addBtn.disabled = !labelInp.value.trim();
        });
    };
    labelInp.onkeydown = function (e) {
      if (e.key === 'Enter' && !addBtn.disabled) addBtn.onclick();
    };
    tb.appendChild(labelInp);
    tb.appendChild(addBtn);

    // Per-user config overrides (profiles + direct overrides + locks).
    var ovrBtn = document.createElement('button');
    ovrBtn.className = 'ghost';
    ovrBtn.textContent = '⚙ Overrides';
    var ovrWrap = document.createElement('div');
    ovrBtn.onclick = function () {
      if (ovrWrap.firstChild) { ovrWrap.innerHTML = ''; ovrBtn.classList.remove('on'); return; }
      ovrBtn.classList.add('on');
      ovrWrap.appendChild(buildBindingDrawer({
        scope: 'user', binding: bindingFromConfig((u.permissions || {}).config),
        previewQuery: 'user_id=' + encodeURIComponent(u.id),
        save: function (body) {
          return api('PATCH', '/settings/api-keys/api/users/' + encodeURIComponent(u.id) + '/permissions',
                     { config: body })
            .then(function (r) { if (!r.ok) return r.text().then(function (t) { throw new Error(t); }); return r.json(); });
        },
      }));
    };
    tb.appendChild(ovrBtn);

    var revBtn = document.createElement('button');
    revBtn.className = 'danger';
    revBtn.textContent = 'Revoke user';
    revBtn.style.marginLeft = 'auto';       // separate the destructive action
    revBtn.onclick = function () {
      if (!confirm('Revoke user "' + u.username +
          '"? This will also revoke all of their keys.')) return;
      api('DELETE', '/settings/api-keys/api/users/' + encodeURIComponent(u.id))
        .then(function (r) {
          if (r.status === 409) {
            return r.json().then(function (j) { throw new Error(j.detail || 'cannot revoke last admin'); });
          }
          if (!r.ok) throw new Error('HTTP ' + r.status);
          showToast('User revoked', 'ok');
          load();
        })
        .catch(function (e) { showToast(String(e.message || e), 'err'); });
    };
    tb.appendChild(revBtn);

    userBody.appendChild(tb);
    userBody.appendChild(ovrWrap);

    // ---- key list (rule cards) ----
    var listEl = document.createElement('div');
    userBody.appendChild(listEl);
    api('GET', '/settings/api-keys/api/users/' + encodeURIComponent(u.id) + '/keys')
      .then(function (r) { return r.ok ? r.json() : { keys: [] }; })
      .then(function (j) {
        if (!j.keys || j.keys.length === 0) {
          listEl.innerHTML = '<p class="hint">No keys yet — generate one above.</p>';
          return;
        }
        j.keys.forEach(function (k) { listEl.appendChild(renderKeyCard(u, k)); });
        refreshKeyDots();
      });

    return card;
  }

  // One rule-style card for a single API key (returned wrapped so the
  // overrides drawer can sit full-width below the card grid).
  function renderKeyCard(u, k) {
    var revoked = !!k.revoked_ts;
    var hasLabel = !!(k.label && k.label.trim());
    var keyUsage = (window.__usage && window.__usage.by_key || {})[k.id];

    var kc = document.createElement('div');
    kc.className = 'kcard' + (revoked ? ' revoked' : '');

    // rail with status dot
    var dotCls = revoked ? 'dead' : dotClassFor(k.last_used_ts);
    var dotTitle = revoked ? 'revoked'
      : (dotCls === 'live' ? 'used in the last 15 min' : 'no recent use');
    var rail = document.createElement('div');
    rail.className = 'krail';
    rail.innerHTML = KEY_ICON_SVG
      + '<span class="kdot ' + dotCls + '"'
      + (revoked ? '' : ' data-used-ts="' + (k.last_used_ts || 0) + '"')
      + ' title="' + dotTitle + '"></span>';
    kc.appendChild(rail);

    // body: title + meta
    var body = document.createElement('div');
    body.className = 'kbody';
    body.innerHTML =
      '<div class="ktitle">'
        + '<span class="kname' + (hasLabel ? '' : ' none') + '">'
          + escapeHtml(hasLabel ? k.label : '(no label)') + '</span>'
        + '<span class="pill ' + (revoked ? 'revoked' : 'live') + '">'
          + (revoked ? 'revoked' : 'active') + '</span>'
      + '</div>'
      + '<div class="kmeta">'
        + '<span class="kid">' + escapeHtml(k.key_prefix) + '&hellip;' + escapeHtml(k.key_last4) + '</span>'
        + '<div class="ktimes">'
          + '<span class="kt-lbl">created</span><span class="kt-val">' + metaWhen(k.created_ts) + '</span>'
          + (revoked
              ? '<span class="kt-lbl">revoked</span><span class="kt-val">' + metaWhen(k.revoked_ts) + '</span>'
              : '<span class="kt-lbl">used</span><span class="kt-val">' + metaWhen(k.last_used_ts) + '</span>')
        + '</div>'
        + (hasLabel ? '' : '<span class="renote">✎ rename to label this key</span>')
      + '</div>';
    var usageWrap = document.createElement('div');
    usageWrap.innerHTML = usageCellHtml(keyUsage);
    body.querySelector('.kmeta').appendChild(usageWrap.firstChild);
    kc.appendChild(body);

    // actions
    var acts = document.createElement('div');
    acts.className = 'kacts';
    var kdraw = document.createElement('div');  // overrides drawer holder
    if (!revoked) {
      // Rename — inline editor under the title
      var renameBtn = document.createElement('button');
      renameBtn.className = 'ghost';
      renameBtn.textContent = '✎ Rename';
      var renameRow = null;
      renameBtn.onclick = function () {
        if (renameRow) { renameRow.remove(); renameRow = null; return; }
        renameRow = document.createElement('div');
        renameRow.className = 'krename';
        var inp = document.createElement('input');
        inp.type = 'text'; inp.maxLength = 128;
        inp.value = hasLabel ? k.label : '';
        inp.placeholder = 'label (e.g., desktop)';
        var saveB = document.createElement('button');
        saveB.className = 'primary'; saveB.textContent = 'Save';
        var cancelB = document.createElement('button');
        cancelB.className = 'ghost'; cancelB.textContent = 'Cancel';
        function closeRename() { if (renameRow) { renameRow.remove(); renameRow = null; } }
        function doSave() {
          var nl = inp.value.trim();
          if (!nl) { showToast('Label is required', 'err'); inp.focus(); return; }
          saveB.disabled = true;
          api('PATCH', '/settings/api-keys/api/users/' + encodeURIComponent(u.id) +
              '/keys/' + encodeURIComponent(k.id) + '/label', { label: nl })
            .then(function (r) { if (!r.ok) return r.text().then(function (t) { throw new Error(t); }); return r.json(); })
            .then(function () { showToast('Key renamed', 'ok'); load(); })
            .catch(function (e) { showToast(String(e.message || e), 'err'); saveB.disabled = false; });
        }
        saveB.onclick = doSave;
        cancelB.onclick = closeRename;
        inp.onkeydown = function (e) {
          if (e.key === 'Enter') doSave();
          else if (e.key === 'Escape') closeRename();
        };
        renameRow.appendChild(inp);
        renameRow.appendChild(saveB);
        renameRow.appendChild(cancelB);
        body.appendChild(renameRow);
        inp.focus();
      };
      acts.appendChild(renameBtn);

      // Overrides — same drawer + label as the per-user button
      var kcfgBtn = document.createElement('button');
      kcfgBtn.className = 'ghost';
      kcfgBtn.textContent = '⚙ Overrides';
      kcfgBtn.onclick = function () {
        if (kdraw.firstChild) { kdraw.innerHTML = ''; kcfgBtn.classList.remove('on'); return; }
        kcfgBtn.classList.add('on');
        kdraw.appendChild(buildBindingDrawer({
          scope: 'key', binding: bindingFromConfig(k.config),
          previewQuery: 'user_id=' + encodeURIComponent(u.id) + '&key_id=' + encodeURIComponent(k.id),
          save: function (body2) {
            return api('PATCH', '/settings/api-keys/api/users/' + encodeURIComponent(u.id) +
                       '/keys/' + encodeURIComponent(k.id) + '/config', body2)
              .then(function (r) { if (!r.ok) return r.text().then(function (t) { throw new Error(t); }); return r.json(); });
          },
        }));
      };
      acts.appendChild(kcfgBtn);

      // Revoke (destructive)
      var b = document.createElement('button');
      b.className = 'danger';
      b.textContent = 'Revoke';
      b.onclick = function () {
        if (!confirm('Revoke key "' + (hasLabel ? k.label : k.key_prefix + '…' + k.key_last4) +
            '"? Apps using it will stop working immediately.')) return;
        api('DELETE', '/settings/api-keys/api/users/' + encodeURIComponent(u.id) +
            '/keys/' + encodeURIComponent(k.id))
          .then(function (r) {
            if (r.status === 409) {
              return r.json().then(function (j) { throw new Error(j.detail || 'cannot revoke last admin key'); });
            }
            if (!r.ok) throw new Error('HTTP ' + r.status);
            showToast('Key revoked', 'ok');
            load();
          })
          .catch(function (e) { showToast(String(e.message || e), 'err'); });
      };
      acts.appendChild(b);
    }
    kc.appendChild(acts);

    var wrap = document.createElement('div');
    wrap.appendChild(kc);
    wrap.appendChild(kdraw);
    return wrap;
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  // --- usage formatting + rendering helpers --------------------------
  function fmtCount(n) {
    n = Number(n || 0);
    if (n >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, '') + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'k';
    return String(n);
  }
  function fmtDuration(sec) {
    sec = Number(sec || 0);
    if (sec < 60) return sec.toFixed(sec < 10 ? 1 : 0) + 's';
    if (sec < 3600) return (sec / 60).toFixed(1).replace(/\.0$/, '') + 'm';
    if (sec < 86400) return (sec / 3600).toFixed(1).replace(/\.0$/, '') + 'h';
    return (sec / 86400).toFixed(1).replace(/\.0$/, '') + 'd';
  }
  function fmtErrRate(reqs, errs) {
    reqs = Number(reqs || 0); errs = Number(errs || 0);
    if (!reqs) return '0%';
    var p = (errs / reqs) * 100;
    return (p < 10 ? p.toFixed(1) : p.toFixed(0)) + '%';
  }
  function _stat(value, caption, cls) {
    return '<div class="stat' + (cls ? ' ' + cls : '') + '">'
         + '<span class="v">' + value + '</span>'
         + '<span class="k">' + caption + '</span></div>';
  }
  // Compact per-key stat block for a key card's meta row.
  function usageCellHtml(stat) {
    if (!stat || !stat.requests) {
      return '<div class="usage-cell empty">no usage yet</div>';
    }
    return '<div class="usage-cell">'
      + _stat(fmtCount(stat.requests), 'requests')
      + _stat(fmtCount(stat.words), 'words')
      + _stat(fmtDuration(stat.audio_s), 'audio')
      + _stat(fmtErrRate(stat.requests, stat.errors), 'errors',
              stat.errors ? 'err' : '')
      + '</div>';
  }
  // One-line activity + lifetime summary strip under a user card header.
  // "last active" leads (always shown, even before any usage is rolled up);
  // the lifetime aggregates follow when present. lastUsedTs is the newest
  // last_used_ts across the user's non-revoked keys (server-computed).
  function userUsageHtml(stat, lastUsedTs) {
    var lastact = '<span class="stat lastact"><span class="k">last active</span>'
      + '<span class="v">' + metaWhen(lastUsedTs) + '</span></span>';
    var rest = (!stat || !stat.requests)
      ? '<span class="hint">No usage recorded yet.</span>'
      : (_stat(fmtCount(stat.requests), 'requests')
         + _stat(fmtCount(stat.words), 'words')
         + _stat(fmtDuration(stat.audio_s), 'audio')
         + _stat(fmtErrRate(stat.requests, stat.errors), 'errors'));
    return '<div class="user-usage">' + lastact + rest + '</div>';
  }

  // ---- per-identity config binding drawer (overrides) ----
  function ovState() { return window.__ovstate || { profiles: {}, field_meta: {}, rules: [] }; }

  function bindingFromConfig(cfg) {
    cfg = cfg || {};
    var direct = Object.assign({}, cfg.direct || {});
    var locks = (direct.locks || []).slice();
    delete direct.locks;
    return {
      overrides: direct,
      profiles: (cfg.profiles || []).slice(),
      locks: locks,
      allow_request_override_profile:
        typeof cfg.allow_request_override_profile === 'boolean'
          ? cfg.allow_request_override_profile : null,
      allow_request_decode_overrides:
        typeof cfg.allow_request_decode_overrides === 'boolean'
          ? cfg.allow_request_decode_overrides : null,
      allowed_override_profiles:
        Array.isArray(cfg.allowed_override_profiles)
          ? cfg.allowed_override_profiles.slice() : null,
      apply_no_profiles:
        typeof cfg.apply_no_profiles === 'boolean'
          ? cfg.apply_no_profiles : null,
    };
  }

  function _cfgAvailFields() {
    var fm = ovState().field_meta || {};
    return Object.keys(fm).filter(function (k) { return fm[k].kind !== 'rulelist'; }).sort();
  }
  function _cfgDefault(name) {
    var m = (ovState().field_meta || {})[name] || { kind: 'str' };
    if (m.kind === 'bool') return true;
    if (m.kind === 'int' || m.kind === 'float') return m.min != null ? m.min : 0;
    if (m.kind === 'enum') return (m.opts || [''])[0];
    return '';
  }
  function _cfgWidget(name, val, onchange) {
    var meta = (ovState().field_meta || {})[name] || { kind: 'str' };
    var el;
    if (meta.kind === 'bool') {
      el = document.createElement('input'); el.type = 'checkbox';
      el.className = 'switch'; el.setAttribute('role', 'switch'); el.checked = !!val;
      el.onchange = function () { onchange(el.checked); };
    } else if (meta.kind === 'enum') {
      el = document.createElement('select');
      (meta.opts || []).forEach(function (o) {
        var op = document.createElement('option'); op.value = o; op.textContent = o;
        if (String(o) === String(val)) op.selected = true; el.appendChild(op);
      });
      el.onchange = function () { onchange(el.value); };
    } else if (meta.kind === 'int' || meta.kind === 'float') {
      el = document.createElement('input'); el.type = 'number';
      if (meta.min != null) el.min = meta.min;
      if (meta.max != null) el.max = meta.max;
      if (meta.kind === 'float') el.step = 'any';
      el.value = val;
      el.onchange = function () {
        var n = meta.kind === 'int' ? parseInt(el.value, 10) : parseFloat(el.value);
        onchange(isNaN(n) ? null : n);
      };
    } else {
      el = document.createElement('input'); el.type = 'text'; el.value = val == null ? '' : val;
      if (meta.maxlen) el.maxLength = meta.maxlen;
      el.onchange = function () { onchange(el.value); };
    }
    return el;
  }

  // Segmented control for the drawer's tri/quad-state gates. `options` is a
  // list of [value, label, tone]; clicking a segment updates the active
  // visuals (so it works whether or not the callback rerenders) then calls
  // onpick(value). Shared .status-btn-group styling from web_common.NAV_CSS.
  function _ovSeg(options, current, onpick) {
    var grp = document.createElement('span');
    grp.className = 'status-btn-group';
    grp.setAttribute('role', 'radiogroup');
    options.forEach(function (o) {
      var val = o[0], label = o[1], tone = o[2];
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'status-btn' + (val === current ? ' active' : '');
      btn.dataset.val = val;
      if (tone) btn.dataset.tone = tone;
      btn.textContent = label;
      btn.setAttribute('role', 'radio');
      btn.setAttribute('aria-checked', val === current ? 'true' : 'false');
      btn.onclick = function () {
        if (btn.classList.contains('active')) return;
        grp.querySelectorAll('.status-btn').forEach(function (x) {
          x.classList.remove('active'); x.setAttribute('aria-checked', 'false');
        });
        btn.classList.add('active'); btn.setAttribute('aria-checked', 'true');
        onpick(val);
      };
      grp.appendChild(btn);
    });
    return grp;
  }

  function buildBindingDrawer(opts) {
    var b = opts.binding;
    var root = document.createElement('div'); root.className = 'cfg-drawer';
    function rerender() { root.innerHTML = ''; draw(); }
    // Dirty-gating: snapshot the binding on open and keep "Save overrides"
    // disabled until the live binding actually differs from it (mirrors the
    // /settings/overrides page's snapshot-compare). Arrays whose order is
    // immaterial (locks, the requestable allowlist) are sorted before
    // comparison; the profiles order is meaningful (earlier wins) and kept
    // as-is. allowed_override_profiles keeps null (=inherit) distinct from []
    // (=none). Every rerender() rebuilds the footer and re-evaluates this;
    // the few mutation paths that DON'T rerender call _syncSave() directly.
    function _sig(x) {
      var ov = {};
      Object.keys(x.overrides || {}).sort().forEach(function (k) { ov[k] = x.overrides[k]; });
      return JSON.stringify({
        overrides: ov,
        profiles: (x.profiles || []),
        locks: (x.locks || []).slice().sort(),
        arop: x.allow_request_override_profile,
        ardo: x.allow_request_decode_overrides,
        aop: x.allowed_override_profiles == null
          ? null : x.allowed_override_profiles.slice().sort(),
        anp: x.apply_no_profiles,
      });
    }
    var _baseline = _sig(b);
    var _saveBtn = null;
    function _syncSave() { if (_saveBtn) _saveBtn.disabled = (_sig(b) === _baseline); }
    // UI intent: a freshly-chosen but still-empty "Restrict to…" allowlist has
    // the same data ([]) as "None". Without this sticky flag the dropdown would
    // snap back to "None" on rerender and the checkbox grid (the only way to add
    // names) would never show — making restrict mode unreachable from the UI.
    var restrictOpen = false;

    // Per-identity REQUEST GATES — whether this identity may request override-
    // profiles / decode tweaks, and which profiles it may name. These can only
    // NARROW the global gates (set on /settings); the server enforces this.
    // Build a card shell: title (+ optional caption) header and optional hint.
    function _ovCard(title, caption, hintText) {
      var card = document.createElement('div'); card.className = 'ov-card';
      var head = document.createElement('div'); head.className = 'ohead';
      var t = document.createElement('span'); t.className = 't'; t.textContent = title;
      head.appendChild(t);
      if (caption) {
        var c = document.createElement('span'); c.className = 'c'; c.textContent = caption;
        head.appendChild(c);
      }
      card.appendChild(head);
      if (hintText) {
        var hn = document.createElement('div'); hn.className = 'ohint';
        hn.textContent = hintText; card.appendChild(hn);
      }
      return card;
    }

    function _triRow(label, slug, hint, val, set) {
      var row = document.createElement('div'); row.className = 'gaterow';
      var gl = document.createElement('span'); gl.className = 'gl';
      gl.title = hint; gl.appendChild(document.createTextNode(label));
      var sub = document.createElement('span'); sub.className = 'sub'; sub.textContent = slug;
      gl.appendChild(sub); row.appendChild(gl);
      var cur = val === true ? 'allow' : (val === false ? 'deny' : 'inherit');
      row.appendChild(_ovSeg(
        [['inherit', 'Inherit', 'inherit'], ['allow', 'Allow', 'allow'], ['deny', 'Deny', 'deny']],
        cur,
        function (v) { set(v === 'allow' ? true : (v === 'deny' ? false : null)); }));
      return row;
    }

    function requestGatesSection(b) {
      var sec = _ovCard('Request gates', null,
        'Narrow the global gates for this identity only — can never widen them.');
      sec.appendChild(_triRow('Allow requesting override-profiles', 'override_profile',
        'May this identity name a profile in a per-request override_profile?',
        b.allow_request_override_profile,
        function (v) { b.allow_request_override_profile = v; rerender(); }));
      sec.appendChild(_triRow('Allow custom decode params', 'decode_overrides',
        'May this identity send inline per-request decode_overrides?',
        b.allow_request_decode_overrides,
        function (v) { b.allow_request_decode_overrides = v; _syncSave(); }));

      // Allowlist of requestable profiles: null=inherit(all), ['*']=all, []=none,
      // [names]=restrict.
      var al = b.allowed_override_profiles;
      var dataMode = al == null ? 'inherit'
        : (al.indexOf('*') >= 0 ? 'all' : (al.length === 0 ? 'none' : 'restrict'));
      if (dataMode === 'inherit' || dataMode === 'all') restrictOpen = false;
      else if (dataMode === 'restrict') restrictOpen = true;
      // Keep the grid open when restrict was chosen but the list is still empty.
      var mode = (restrictOpen && dataMode === 'none') ? 'restrict' : dataMode;
      var arow = document.createElement('div'); arow.className = 'gaterow';
      var agl = document.createElement('span'); agl.className = 'gl';
      agl.appendChild(document.createTextNode('Requestable profiles'));
      var asub = document.createElement('span'); asub.className = 'sub';
      asub.textContent = 'allowed_override_profiles'; agl.appendChild(asub);
      arow.appendChild(agl);
      arow.appendChild(_ovSeg(
        [['inherit', 'Inherit', 'inherit'], ['all', 'All', 'all'],
         ['restrict', 'Restrict…', 'own'], ['none', 'None', 'deny']],
        mode,
        function (v) {
          if (v === 'inherit') { b.allowed_override_profiles = null; restrictOpen = false; }
          else if (v === 'all') { b.allowed_override_profiles = ['*']; restrictOpen = false; }
          else if (v === 'none') { b.allowed_override_profiles = []; restrictOpen = false; }
          else {
            b.allowed_override_profiles =
              (Array.isArray(al) ? al.filter(function (n) { return n !== '*'; }) : []);
            restrictOpen = true;
          }
          rerender();
        }));
      sec.appendChild(arow);

      if (mode === 'restrict') {
        var names = Object.keys(ovState().profiles || {}).sort();
        var box = document.createElement('div'); box.className = 'ov-checks';
        if (!names.length) {
          var e = document.createElement('span'); e.className = 'lbl';
          e.textContent = '(no profiles defined — create them on the Overrides page)';
          box.appendChild(e);
        }
        names.forEach(function (n) {
          var lab = document.createElement('label');
          var cb = document.createElement('input'); cb.type = 'checkbox';
          cb.checked = al.indexOf(n) >= 0;
          cb.onchange = function () {
            var i = b.allowed_override_profiles.indexOf(n);
            if (cb.checked && i < 0) b.allowed_override_profiles.push(n);
            else if (!cb.checked && i >= 0) b.allowed_override_profiles.splice(i, 1);
            _syncSave();
          };
          lab.appendChild(cb);
          lab.appendChild(document.createTextNode(' ' + n));
          box.appendChild(lab);
        });
        sec.appendChild(box);
      }
      return sec;
    }

    function draw() {
      // --- Profiles card ---
      var pcard = _ovCard('Profiles', 'ordered · earlier wins',
        'Applied in order, before the direct overrides below.');
      // Admin force, per-KEY only: suppress all profiles → plain defaults. Not a
      // request gate (lives here, not in the gates card) and not gated globally.
      var noProf = (opts.scope === 'key' && b.apply_no_profiles === true);
      if (opts.scope === 'key') {
        var npRow = document.createElement('div'); npRow.className = 'cfg-noprof-row';
        var npCb = document.createElement('input');
        npCb.type = 'checkbox'; npCb.className = 'switch'; npCb.setAttribute('role', 'switch');
        npCb.id = 'cfg-apply-no-profiles';
        npCb.checked = noProf;
        npCb.onchange = function () { b.apply_no_profiles = npCb.checked ? true : null; rerender(); };
        var npLbl = document.createElement('label');
        npLbl.setAttribute('for', 'cfg-apply-no-profiles');
        npLbl.textContent = 'Apply no profiles';
        npRow.appendChild(npCb); npRow.appendChild(npLbl);
        pcard.appendChild(npRow);
        var npHint = document.createElement('div'); npHint.className = 'ohint';
        npHint.textContent = noProf
          ? 'On — this key ignores every profile (key, user-level, and per-request) and resolves to plain defaults.'
          : 'Force plain defaults for this key: ignore every profile (key, user-level, and per-request).';
        pcard.appendChild(npHint);
      }
      var chips = document.createElement('div');
      chips.className = noProf ? 'cfg-chips cfg-prof-dim' : 'cfg-chips';
      b.profiles.forEach(function (name, i) {
        var c = document.createElement('span'); c.className = 'cfg-chip';
        c.innerHTML = '<span class="num">' + (i + 1) + '</span>' + escapeHtml(name);
        var up = document.createElement('button'); up.textContent = '↑'; up.title = 'earlier';
        up.onclick = function () { if (i > 0) { var t = b.profiles[i - 1]; b.profiles[i - 1] = b.profiles[i]; b.profiles[i] = t; rerender(); } };
        var dn = document.createElement('button'); dn.textContent = '↓'; dn.title = 'later';
        dn.onclick = function () { if (i < b.profiles.length - 1) { var t = b.profiles[i + 1]; b.profiles[i + 1] = b.profiles[i]; b.profiles[i] = t; rerender(); } };
        var x = document.createElement('button'); x.textContent = '×';
        x.onclick = function () { b.profiles.splice(i, 1); rerender(); };
        c.appendChild(up); c.appendChild(dn); c.appendChild(x); chips.appendChild(c);
      });
      var avail = Object.keys(ovState().profiles || {}).filter(function (n) { return b.profiles.indexOf(n) < 0; }).sort();
      if (avail.length) {
        var sel = document.createElement('select'); sel.innerHTML = '<option value="">+ add profile…</option>';
        avail.forEach(function (n) { var o = document.createElement('option'); o.value = n; o.textContent = n; sel.appendChild(o); });
        sel.onchange = function () { if (sel.value) { b.profiles.push(sel.value); rerender(); } };
        chips.appendChild(sel);
      } else if (!b.profiles.length) {
        var e = document.createElement('span'); e.className = 'lbl';
        e.textContent = '(no profiles defined — create them on the Overrides page)';
        chips.appendChild(e);
      }
      pcard.appendChild(chips);
      root.appendChild(pcard);

      // --- Request gates card ---
      root.appendChild(requestGatesSection(b));

      // --- Direct overrides card ---
      var dcard = _ovCard('Direct overrides', 'pin a field value',
        'Force a config value for this identity; lock to stop clients overriding it.');
      Object.keys(b.overrides).sort().forEach(function (name) {
        var row = document.createElement('div'); row.className = 'cfg-ovr-row';
        var nm = document.createElement('span'); nm.className = 'nm'; nm.textContent = name; row.appendChild(nm);
        var vc = document.createElement('span');
        vc.appendChild(_cfgWidget(name, b.overrides[name], function (v) { if (v !== null) b.overrides[name] = v; _syncSave(); }));
        row.appendChild(vc);
        var on = b.locks.indexOf(name) >= 0;
        var lk = document.createElement('button'); lk.className = 'lk' + (on ? ' on' : '');
        lk.innerHTML = on ? '\u{1F512}' : '\u{1F513}'; lk.title = 'lock — client cannot override this field';
        lk.onclick = function () { var i = b.locks.indexOf(name); if (i >= 0) b.locks.splice(i, 1); else b.locks.push(name); rerender(); };
        row.appendChild(lk);
        var rm = document.createElement('button'); rm.className = 'rm'; rm.textContent = '×';
        rm.onclick = function () { delete b.overrides[name]; var i = b.locks.indexOf(name); if (i >= 0) b.locks.splice(i, 1); rerender(); };
        row.appendChild(rm); dcard.appendChild(row);
      });
      var avf = _cfgAvailFields().filter(function (n) { return !(n in b.overrides); });
      var addwrap = document.createElement('div'); addwrap.className = 'cfg-chips';
      var addsel = document.createElement('select'); addsel.innerHTML = '<option value="">+ add field…</option>';
      avf.forEach(function (n) { var o = document.createElement('option'); o.value = n; o.textContent = n; addsel.appendChild(o); });
      addsel.onchange = function () { if (addsel.value) { b.overrides[addsel.value] = _cfgDefault(addsel.value); rerender(); } };
      addwrap.appendChild(addsel); dcard.appendChild(addwrap);
      root.appendChild(dcard);

      // --- footer: Preview (ghost, left) + Save (primary, right) ---
      var foot = document.createElement('div'); foot.className = 'ov-footer';
      var prev = document.createElement('button'); prev.className = 'ghost'; prev.textContent = '⟲ Preview effective';
      var spacer = document.createElement('span'); spacer.className = 'spacer';
      var save = document.createElement('button'); save.className = 'primary'; save.textContent = 'Save overrides';
      _saveBtn = save;
      save.disabled = (_sig(b) === _baseline);   // nothing changed yet → off
      save.onclick = function () {
        save.disabled = true;
        opts.save({
          overrides: b.overrides, profiles: b.profiles, locks: b.locks,
          allow_request_override_profile: b.allow_request_override_profile,
          allow_request_decode_overrides: b.allow_request_decode_overrides,
          allowed_override_profiles: b.allowed_override_profiles,
          apply_no_profiles: b.apply_no_profiles,
        })
          .then(function () { showToast('Overrides saved', 'ok'); load(); })
          .catch(function (er) { showToast(String(er.message || er), 'err'); })
          // On success load() tears the drawer down; on failure the binding is
          // still dirty, so re-enable only if it genuinely differs.
          .finally(function () { _syncSave(); });
      };
      foot.appendChild(prev); foot.appendChild(spacer); foot.appendChild(save);
      root.appendChild(foot);
      var eff = document.createElement('div'); eff.className = 'cfg-eff'; eff.style.display = 'none'; root.appendChild(eff);
      prev.onclick = function () {
        eff.style.display = '';
        eff.innerHTML = '<span class="lbl">resolving…</span>';
        fetch('/settings/overrides/resolve?' + opts.previewQuery)
          .then(function (r) { return r.ok ? r.json() : null; })
          .then(function (j) {
            if (!j) { eff.innerHTML = '<span class="lbl">preview unavailable</span>'; return; }
            eff.innerHTML = ''; var any = false;
            Object.keys(j.fields || {}).sort().forEach(function (f) {
              var fr = j.fields[f];
              if (!fr.winner_layer || fr.winner_layer === 'global' || fr.winner_layer === 'per-model') return;
              any = true;
              var row = document.createElement('div'); row.className = 'ef';
              row.innerHTML = '<span class="nm">' + escapeHtml(f) + '</span>'
                + '<span class="v">' + escapeHtml(JSON.stringify(fr.winner_value)) + '</span>'
                + '<span class="src">' + escapeHtml(fr.winner_layer) + '</span>'
                + (fr.locked ? '<span class="lk">\u{1F512}</span>' : '');
              eff.appendChild(row);
            });
            if (!any) eff.innerHTML = '<span class="lbl">no identity overrides take effect (all inherit global / per-model)</span>';
          })
          .catch(function () { eff.innerHTML = '<span class="lbl">preview unavailable</span>'; });
      };
    }
    draw();
    return root;
  }

  async function load() {
    var r = await api('GET', '/settings/api-keys/api/users');
    if (r.status === 401) {
      // Shared login gate prompts + reloads on success (which re-runs load()).
      if (window._showLoginGate) window._showLoginGate();
      return;
    }
    if (await _check403(r)) return;
    if (!r.ok) {
      showToast('Load failed: HTTP ' + r.status, 'err');
      return;
    }
    var j = await r.json();
    // role-admin is set by OPEN_MODE_BANNER_JS (single source of truth)
    // when whoami.is_admin=true. This page used to add it here
    // unconditionally; redundant now that the central script owns it.
    var banner = document.getElementById('open-banner');
    banner.style.display = j.open_mode ? 'block' : 'none';
    var ct = document.getElementById('users-container');
    ct.innerHTML = '';
    if (!j.users.length) {
      document.getElementById('perm-matrix-card').style.display = 'none';
      ct.innerHTML = '<p class="hint">No users yet. Add one above to create the first admin.</p>';
      return;
    }
    // Pull the usage rollup once (lifetime) so per-user strips and per-key
    // cells can join against it client-side. Best-effort — a usage failure
    // must never blank the key-management UI.
    window.__usage = { by_user: {}, by_key: {} };
    try {
      var ur = await api('GET', '/settings/api-keys/api/usage');
      if (ur.ok) window.__usage = await ur.json();
    } catch (_) {}
    // Override profiles + field metadata for the per-identity binding drawers.
    window.__ovstate = { profiles: {}, field_meta: {}, rules: [] };
    try {
      var os = await api('GET', '/settings/overrides/state');
      if (os.ok) window.__ovstate = await os.json();
    } catch (_) {}
    renderMatrix(j);
    var nUsers = j.users.length;
    j.users.forEach(function(u) { ct.appendChild(renderUser(u, nUsers)); });
  }

  document.getElementById('add-user-btn').onclick = function() {
    var username = document.getElementById('new-username').value.trim();
    var is_admin = document.getElementById('new-is-admin').checked;
    if (!username) { showToast('Enter a username', 'err'); return; }
    api('POST', '/settings/api-keys/api/users', { username: username, is_admin: is_admin })
      .then(function(r) {
        if (!r.ok) return r.text().then(function(t) { throw new Error(t); });
        return r.json();
      })
      .then(function() {
        document.getElementById('new-username').value = '';
        document.getElementById('new-is-admin').checked = false;
        showToast('User created', 'ok');
        load();
      })
      .catch(function(e) { showToast(String(e.message || e), 'err'); });
  };

  // #logout-btn + #reload-btn are wired globally in OPEN_MODE_BANNER_JS;
  // expose this page's soft refresh (re-fetch key list) as the reload hook.
  window._pageReload = load;

  load();
})();
</script>
<script>
// Runs AFTER TIME_HELPERS_JS defines timeTick. Ages the relative suffix on the
// per-key created/last-used cells in place; the cells already carry static
// fmtWhen() text at render time, and timeTick re-queries [data-ts] each tick so
// it also catches cells added by later list reloads.
timeTick();
// Decay the per-key status dots (green -> grey) as keys cross the recent-use
// window, without a reload. _refreshApiKeyDots re-queries [data-used-ts] each
// tick so it also catches dots added by later list reloads.
if (window._refreshApiKeyDots) {
  window._refreshApiKeyDots();
  setInterval(window._refreshApiKeyDots, 30000);
}
</script>
</body></html>
"""
