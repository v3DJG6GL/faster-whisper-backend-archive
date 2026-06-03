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
from admin_routes import require_admin_host
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


# ---------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------

@router.get(
    "",
    dependencies=[Depends(require_admin_host)],
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
    dependencies=[Depends(require_admin_host), Depends(require_admin)],
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
    out = [
        {
            **u,
            "active_key_count": counts.get(u["id"], 0),
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
    dependencies=[Depends(require_admin_host), Depends(require_admin)],
)
async def create_user_api(payload: CreateUserIn) -> JSONResponse:
    try:
        uid = api_keys_store.create_user(payload.username, payload.is_admin)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return JSONResponse({"user_id": uid})


@router.delete(
    "/api/users/{uid}",
    dependencies=[Depends(require_admin_host), Depends(require_admin)],
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
    dependencies=[Depends(require_admin_host), Depends(require_admin)],
)
async def list_user_keys_api(uid: str) -> JSONResponse:
    if api_keys_store.get_user(uid) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    return JSONResponse({"keys": api_keys_store.list_keys(user_id=uid)})


@router.get(
    "/api/usage",
    dependencies=[Depends(require_admin_host), Depends(require_admin)],
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
    dependencies=[Depends(require_admin_host), Depends(require_admin)],
)
async def create_user_key_api(uid: str, payload: CreateKeyIn) -> JSONResponse:
    """Show-once raw key on creation. Subsequent reads via list_user_keys
    never return the raw value."""
    try:
        raw_key, rec = api_keys_store.create_key(uid, label=payload.label)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return JSONResponse({"key": raw_key, "record": rec})


@router.delete(
    "/api/users/{uid}/keys/{kid}",
    dependencies=[Depends(require_admin_host), Depends(require_admin)],
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
    dependencies=[Depends(require_admin_host), Depends(require_admin)],
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
        merged = api_keys_store.set_user_permissions(uid, merge_payload)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return JSONResponse({"ok": True, "permissions": merged})


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
  .key-row { display: grid;
    /* label / id / usage stats / activity (created+used stacked) / action.
       The usage block carries the new per-key counters; the two timestamps
       collapse into one stacked "activity" cell so the previously crammed
       created/used pair reads cleanly and the freed left-side width goes to
       the stats that admins actually scan. */
    grid-template-columns: minmax(6rem,0.9fr) 8.5rem minmax(11rem,1.2fr) minmax(8rem,auto) auto;
    gap: 0.6rem; align-items: center; padding: 0.4rem 0;
    border-top: 1px solid var(--border); font-size: var(--fs-sm); }
  .key-row:first-child { border-top: none; }
  /* Phone: the 5-column key row can't hold its widths on a ~360px screen —
     stack the cells into a single column so each (label / id / usage /
     activity / action) gets the full width. */
  @media (max-width: 40em) {
    .key-row { grid-template-columns: 1fr; gap: 0.3rem; align-items: start; }
    .key-row .action { justify-self: start; }
  }
  .key-row .label { color: var(--fg); word-break: break-word; }
  .key-row .id { color: var(--dim); font-family: var(--font-mono); }
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
  /* The stacked activity cell reuses the .ts two-line pattern but holds
     both created and used, one above the other. */
  .key-row .activity { display: flex; flex-direction: column; gap: 0.2rem; }
  .key-row .ts { display: flex; flex-direction: row; align-items: baseline;
    gap: 0.35rem; line-height: 1.2; }
  .key-row .ts .ts-label { color: var(--dim); font-size: var(--fs-xs);
    text-transform: lowercase; min-width: 3.2rem; }
  .key-row .ts .ts-value { color: var(--fg); font-size: var(--fs-sm);
    font-family: var(--font-mono); white-space: nowrap; }
  .key-row .ts.empty .ts-value { color: var(--dim); font-style: italic; }
  button.danger { color: var(--red); border-color: #5a2424; }
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
    font-size: var(--fs-xl); }
  .modal p { margin: 0 0 0.9rem 0; line-height: 1.45;
    color: var(--help); font-size: var(--fs-sm); }
  .modal p strong, .modal p code { color: var(--fg); }
  .modal p:last-of-type { margin-bottom: 1rem; }
  .modal input[type=text], .modal input[type=password] {
    /* Slightly taller inputs in modals (better tap target + reads
       cleaner next to the buttons below). */
    padding: 0.55rem 0.7rem; font-size: var(--fs-md);
  }
  .modal .raw-key {
    font-family: var(--font-mono); font-size: var(--fs-sm);
    word-break: break-all; padding: 0.65rem 0.75rem;
    background: var(--input-bg); color: var(--bold);
    border: 1px solid var(--border); border-radius: 4px;
    margin: 0.65rem 0 0.85rem; user-select: all;
    line-height: 1.5;
  }
  /* Uniform action-button sizing: same line-height, same padding, same
     min-height — so Cancel and Save render identical regardless of which
     border colour they carry. Avoid `padding: …` redeclaration in
     button.primary etc. so heights stay in sync. */
  .modal .actions {
    display: flex; gap: 0.6rem; justify-content: flex-end;
    margin-top: 1.1rem; padding-top: 0.85rem;
    border-top: 1px solid var(--border);
  }
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
    color: var(--green); border-color: var(--green);
  }
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
  .perm-matrix select {
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 0.15rem 0.3rem; font: inherit; font-size: var(--fs-sm);
    font-family: var(--font-sans);
  }
  .perm-matrix select:disabled {
    opacity: 0.35; cursor: not-allowed; }
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
        <input id="new-is-admin" type="checkbox"> admin
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
    <h3>New API key</h3>
    <p>Save this key now &mdash; it will not be shown again. Anyone with the key
    has the same access as <strong id="key-modal-user"></strong>.</p>
    <div class="raw-key" id="key-modal-raw"></div>
    <div class="actions">
      <button id="key-modal-copy">Copy</button>
      <button id="key-modal-done" class="primary">I've saved it</button>
    </div>
  </div>
</div>

<!-- API key prompt -->
<div id="token-modal" class="modal">
  <div class="box">
    <h3>Admin API key</h3>
    <p>Paste your <code>wk_&hellip;</code> admin key to manage users and
    keys. In OPEN mode (no admin key configured yet) any value works.</p>
    <input id="token-input" type="password" placeholder="wk_&hellip;">
    <p id="token-err" class="err"></p>
    <div class="actions">
      <button id="token-cancel">Cancel</button>
      <button id="token-save" class="primary">Save</button>
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

  // Exchange a pasted key for an HttpOnly session cookie. Returns true on
  // success. Dispatches whisper:auth-changed so the shared chrome refreshes.
  async function doLogin(key) {
    try {
      var r = await fetch('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: key }),
      });
      if (!r.ok) return false;
      try { window.dispatchEvent(new Event('whisper:auth-changed')); } catch(_) {}
      return true;
    } catch (_) { return false; }
  }

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


  function showTokenModal() {
    return new Promise(function(resolve){
      var m = document.getElementById('token-modal');
      var inp = document.getElementById('token-input');
      var err = document.getElementById('token-err');
      err.textContent = '';
      inp.value = '';
      m.classList.add('show');
      setTimeout(function(){ inp.focus(); }, 50);
      function done(v) {
        m.classList.remove('show');
        document.getElementById('token-save').onclick = null;
        document.getElementById('token-cancel').onclick = null;
        inp.onkeydown = null;
        resolve(v);
      }
      document.getElementById('token-save').onclick = function() {
        var v = inp.value.trim();
        if (!v) { err.textContent = 'Empty value'; return; }
        done(v);
      };
      document.getElementById('token-cancel').onclick = function() {
        done(null);
      };
      inp.onkeydown = function(e) {
        if (e.key === 'Enter') document.getElementById('token-save').click();
        if (e.key === 'Escape') document.getElementById('token-cancel').click();
      };
    });
  }

  function showKeyModal(rawKey, username) {
    document.getElementById('key-modal-user').textContent = username;
    document.getElementById('key-modal-raw').textContent = rawKey;
    var m = document.getElementById('key-modal');
    m.classList.add('show');
    document.getElementById('key-modal-copy').onclick = function() {
      // navigator.clipboard requires a secure context (https / localhost).
      // Over LAN HTTP it's undefined, so fall back to a hidden textarea +
      // document.execCommand('copy'). If both fail, select the visible
      // .raw-key span so the user can ctrl-c manually.
      function copyFallback(text) {
        var ta = document.createElement('textarea');
        ta.value = text;
        // Hide off-screen but keep selectable.
        ta.style.position = 'fixed';
        ta.style.top = '-1000px';
        ta.setAttribute('readonly', '');
        document.body.appendChild(ta);
        ta.select();
        ta.setSelectionRange(0, ta.value.length);
        var ok = false;
        try { ok = document.execCommand('copy'); } catch(_) {}
        document.body.removeChild(ta);
        return ok;
      }
      function selectRawSpan() {
        var span = document.getElementById('key-modal-raw');
        var sel = window.getSelection();
        var range = document.createRange();
        range.selectNodeContents(span);
        sel.removeAllRanges();
        sel.addRange(range);
      }
      function onSuccess() { showToast('Copied to clipboard', 'ok'); }
      function onFailure() {
        selectRawSpan();
        showToast('Auto-copy blocked — press Ctrl+C', 'err');
      }
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(rawKey).then(onSuccess, function() {
          if (!copyFallback(rawKey)) onFailure();
          else onSuccess();
        });
      } else {
        if (copyFallback(rawKey)) onSuccess();
        else onFailure();
      }
    };
    document.getElementById('key-modal-done').onclick = function() {
      m.classList.remove('show');
      load();
    };
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
          var sel = document.createElement('select');
          sel.dataset.page = page;
          var allowed = accessOnly.has(page)
            ? ['none', 'all']
            : ['none', 'own', 'all'];
          allowed.forEach(function(s) {
            var opt = document.createElement('option');
            opt.value = s;
            opt.textContent = s;
            sel.appendChild(opt);
          });
          sel.value = allowed.indexOf(currentPages[page]) >= 0
            ? currentPages[page] : 'none';
          sel.addEventListener('change', function() {
            tr.classList.add('dirty');
          });
          td.appendChild(sel);
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
            tr.classList.add('dirty');
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

      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
  }

  function savePerms(uid, tr, btn) {
    var pages = {};
    tr.querySelectorAll('select').forEach(function(sel) {
      pages[sel.dataset.page] = sel.value;
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
        tr.classList.remove('dirty');
        showToast('Permissions saved', 'ok');
      })
      .catch(function(e) {
        showToast(String(e.message || e), 'err');
      })
      .finally(function() {
        btn.disabled = false;
      });
  }

  function renderUser(u) {
    var card = document.createElement('div');
    card.className = 'card';
    var h = document.createElement('h3');
    h.innerHTML = '<span>' + escapeHtml(u.username) + '</span>';
    var pill = document.createElement('span');
    pill.className = 'pill ' + (u.is_admin ? 'admin' : '');
    pill.textContent = u.is_admin ? 'admin' : 'user';
    h.appendChild(pill);
    var keyCount = document.createElement('span');
    keyCount.className = 'pill';
    keyCount.textContent = u.active_key_count + ' active key' +
      (u.active_key_count === 1 ? '' : 's');
    h.appendChild(keyCount);

    // Lifetime usage strip — sums every one of this user's keys (plus any
    // pre-feature backfilled usage) from the rollup fetched in load().
    // Lives INSIDE the header, pushed to the right (margin-left:auto) so it
    // reuses the empty space beside the name instead of crowding a line below.
    var usageStrip = document.createElement('div');
    usageStrip.innerHTML = userUsageHtml(
      (window.__usage && window.__usage.by_user || {})[u.id]);
    h.appendChild(usageStrip.firstChild);
    card.appendChild(h);

    var tb = document.createElement('div');
    tb.className = 'toolbar';
    {
      var labelInp = document.createElement('input');
      labelInp.type = 'text';
      labelInp.placeholder = 'label (e.g., desktop)';
      labelInp.style.maxWidth = '14rem';
      labelInp.style.flex = '1';
      var addBtn = document.createElement('button');
      addBtn.className = 'primary';
      addBtn.textContent = '+ generate key';
      addBtn.onclick = function() {
        var label = labelInp.value.trim();
        api('POST', '/settings/api-keys/api/users/' + encodeURIComponent(u.id) + '/keys',
            { label: label })
          .then(function(r) {
            if (!r.ok) return r.text().then(function(t) { throw new Error(t); });
            return r.json();
          })
          .then(function(j) {
            showKeyModal(j.key, u.username);
          })
          .catch(function(e) {
            showToast(String(e.message || e), 'err');
          });
      };
      tb.appendChild(labelInp);
      tb.appendChild(addBtn);

      var revBtn = document.createElement('button');
      revBtn.className = 'danger';
      revBtn.textContent = 'revoke user';
      revBtn.onclick = function() {
        if (!confirm('Revoke user "' + u.username +
            '"? This will also revoke all of their keys.')) return;
        api('DELETE', '/settings/api-keys/api/users/' + encodeURIComponent(u.id))
          .then(function(r) {
            if (r.status === 409) {
              return r.json().then(function(j) {
                throw new Error(j.detail || 'cannot revoke last admin');
              });
            }
            if (!r.ok) throw new Error('HTTP ' + r.status);
            showToast('User revoked', 'ok');
            load();
          })
          .catch(function(e) {
            showToast(String(e.message || e), 'err');
          });
      };
      tb.appendChild(revBtn);
    }
    card.appendChild(tb);

    // Fetch + render keys
    var listEl = document.createElement('div');
    card.appendChild(listEl);
    api('GET', '/settings/api-keys/api/users/' + encodeURIComponent(u.id) + '/keys')
      .then(function(r) { return r.ok ? r.json() : { keys: [] }; })
      .then(function(j) {
        if (!j.keys || j.keys.length === 0) {
          listEl.innerHTML = '<p class="hint">No keys yet — generate one above.</p>';
          return;
        }
        j.keys.forEach(function(k) {
          var row = document.createElement('div');
          row.className = 'key-row';
          function _tsCell(label, ts) {
            // Caption + value on one line; created and used stack inside the
            // shared .activity cell. Empty ts (never-used key) renders an
            // em-dash so the cell keeps its height and the grid stays aligned.
            var hasValue = !!ts;
            var v = hasValue
              ? '<span data-ts="' + ts + '" title="' + escapeHtml(absTime(ts)) + '">'
                  + escapeHtml(fmtWhen(ts)) + '</span>'
              : '—';
            return '<div class="ts' + (hasValue ? '' : ' empty') + '">'
                 + '<div class="ts-label">' + label + '</div>'
                 + '<div class="ts-value">' + v + '</div>'
                 + '</div>';
          }
          var keyUsage = (window.__usage && window.__usage.by_key || {})[k.id];
          row.innerHTML =
            '<div class="label">' + escapeHtml(k.label || '(no label)') + '</div>' +
            '<div class="id">' + escapeHtml(k.key_prefix) + '&hellip;' + escapeHtml(k.key_last4) + '</div>' +
            usageCellHtml(keyUsage) +
            '<div class="activity">' +
              _tsCell('created', k.created_ts) +
              _tsCell('used',    k.last_used_ts) +
            '</div>';
          var actionCell = document.createElement('div');
          if (k.revoked_ts) {
            var rp = document.createElement('span');
            rp.className = 'pill revoked';
            rp.textContent = 'revoked';
            actionCell.appendChild(rp);
          } else {
            var b = document.createElement('button');
            b.className = 'danger';
            b.textContent = 'revoke';
            b.onclick = function() {
              if (!confirm('Revoke key ' + k.key_prefix + '…' + k.key_last4 + '?')) return;
              api('DELETE', '/settings/api-keys/api/users/' + encodeURIComponent(u.id) +
                  '/keys/' + encodeURIComponent(k.id))
                .then(function(r) {
                  if (r.status === 409) {
                    return r.json().then(function(j) {
                      throw new Error(j.detail || 'cannot revoke last admin key');
                    });
                  }
                  if (!r.ok) throw new Error('HTTP ' + r.status);
                  showToast('Key revoked', 'ok');
                  load();
                })
                .catch(function(e) {
                  showToast(String(e.message || e), 'err');
                });
            };
            actionCell.appendChild(b);
          }
          row.appendChild(actionCell);
          listEl.appendChild(row);
        });
      });

    return card;
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
  // Compact per-key stat block for a .key-row usage cell.
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
  // One-line lifetime summary strip under a user card header.
  function userUsageHtml(stat) {
    if (!stat || !stat.requests) {
      return '<div class="user-usage"><span class="hint">No usage recorded yet.</span></div>';
    }
    return '<div class="user-usage">'
      + _stat(fmtCount(stat.requests), 'requests')
      + _stat(fmtCount(stat.words), 'words')
      + _stat(fmtDuration(stat.audio_s), 'audio')
      + _stat(fmtErrRate(stat.requests, stat.errors), 'errors')
      + '</div>';
  }

  async function load() {
    var r = await api('GET', '/settings/api-keys/api/users');
    if (r.status === 401) {
      var v = await showTokenModal();
      if (!v) return;
      // The typed key wasn't recognised → the login request itself fails.
      if (!(await doLogin(v))) {
        document.getElementById('token-err').textContent = 'invalid key';
        return;
      }
      r = await api('GET', '/settings/api-keys/api/users');
      // 403 after a valid session = "valid key, no admin scope" — render
      // the no-access landing rather than re-prompting.
      if (r.status === 403 && await _check403(r)) return;
      // Other non-2xx (5xx, network): surface the status without
      // nuking the bearer.
      if (!r.ok) {
        showToast('Load failed: HTTP ' + r.status, 'err');
        return;
      }
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
    renderMatrix(j);
    j.users.forEach(function(u) { ct.appendChild(renderUser(u)); });
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
</script>
</body></html>
"""
