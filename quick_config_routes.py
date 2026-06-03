"""
End-user simple-config WebUI for faster-whisper-backend.

Mounted at /quick-config. Endpoints:

  GET  /quick-config                      HTML page (loopback / ADMIN_ALLOWED_HOSTS)
  GET  /quick-config/state                Returns ONLY the rules an admin marked exposed
  POST /quick-config/state                Patch enabled / body fields on exposed rules
  POST /quick-config/reapply-rules        Kick off bulk-reapply job over existing captures
  GET  /quick-config/reapply-rules/status Poll bulk-reapply job state
  GET  /quick-config/recent               Snapshot of the recent-traces ring buffer
  GET  /quick-config/stream               SSE stream of recent traces (live updates)

Security model:
  1. IP gate:           require_admin_host (loopback always permitted)
  2. API key:           Depends(get_current_user) — bearer must resolve to
                        an active key. Admin = is_admin=True.
  3. Rule allow-list:   POST enforces `exposed == True` AND a per-type
                        field allow-list, regardless of caller role. Defends
                        against `{"locked": false}`-style bypass attempts —
                        the filter runs BEFORE the merge into PIPELINE_RULES.

The recent-traces ring buffer holds literal patient dictation snippets on
a medical deployment. RAM-only, capped, lost on restart. Don't log
buffer contents and don't persist them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError

import config as cfg
import config_store
import quick_config_state
import transcriptions_store
import web_common
from admin_routes import (
    _apply_hot_changes,
    _canon_rules,
    require_admin_host,
)
from auth import Permissions, get_current_user, require_page, user_from_session_cookie

logger = logging.getLogger("whisper-api")

router = APIRouter(prefix="/quick-config")


# Per-type allow-list of fields an end-user is allowed to patch on an
# already-existing exposed rule. Anything else in a patch dict triggers a
# 400. `name`, `label`, `type`, `exposed`, `locked`, `seeded` are NEVER
# editable from /quick-config (admin-only). Adding a new rule, deleting a
# rule, or reordering is also admin-only.
_PATCH_ALLOWED_FIELDS: dict[str, frozenset[str]] = {
    "regex":                       frozenset({"enabled", "pattern", "replacement"}),
    "callback:map":                frozenset({"enabled", "map"}),
    "callback:lowercase-wordlist": frozenset({"enabled", "pattern", "wordlist"}),
    "callback:dedup":              frozenset({"enabled", "pattern"}),
    "callback:upper":              frozenset({"enabled", "pattern"}),
}


# SSE endpoint compatibility: EventSource has no way to attach an
# Authorization header, so the /stream endpoint accepts ?key=<raw_key>
# as a fallback.

def require_user_or_admin_sse(request: Request) -> dict[str, Any]:
    """SSE-aware variant of get_current_user + require_page("quick_config").
    Resolves auth from (in order) the bearer header, the HttpOnly session
    cookie (EventSource sends it automatically), then `?key=` — the last is
    the legacy fallback for non-browser SSE clients that can set neither a
    header nor a cookie. Attaches the Permissions policy object and rejects
    callers whose quick_config scope is "none"."""
    import api_keys_store
    if not api_keys_store.is_locked_down():
        rec = dict(api_keys_store.OPEN_MODE_USER)
    else:
        auth_header = request.headers.get("authorization") or ""
        raw = ""
        if auth_header.lower().startswith("bearer "):
            raw = auth_header.split(" ", 1)[1].strip()
        rec = api_keys_store.lookup_by_raw_key(raw) if raw else None
        if rec is None:
            rec = user_from_session_cookie(request)
        if rec is None:
            key = request.query_params.get("key") or ""
            rec = api_keys_store.lookup_by_raw_key(key) if key else None
        if rec is None:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    rec["permissions"] = Permissions(
        rec.get("permissions_raw") or {}, bool(rec.get("is_admin")),
    )
    if not rec["permissions"].can("quick_config"):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "no access to /quick-config",
        )
    return rec


class QuickPatchPayload(BaseModel):
    """POST body shape:
        {"rules_patch": {slug: {field: value, ...}, ...},
         "fingerprints": {slug: <hex>, ...}}      # optional, recommended

    `fingerprints` carries the per-rule hash each slug had when the
    client loaded /state. Server compares against the current rule's
    fingerprint and reports a conflict for any mismatch. Without it,
    last-writer-wins (legacy behavior; clients are expected to send it
    now). See _rule_fingerprint()."""
    model_config = {"extra": "forbid"}
    rules_patch: dict[str, dict[str, Any]]
    fingerprints: dict[str, str] | None = None


def _rule_fingerprint(rule: dict[str, Any]) -> str:
    """Stable short hash of a rule dict for optimistic concurrency
    control (HTTP-ETag style). Order-insensitive on dict fields. Uses
    sha1 because it's fast and we don't need cryptographic strength —
    the worst case of a collision is "two different rule states hash
    the same" which is statistically irrelevant at our buffer cap. Cut
    to 12 hex chars (~48 bits) to keep the wire payload small."""
    canonical = json.dumps(
        rule, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


@router.get(
    "",
    # HTML page is host-only — the login modal runs in this page's
    # own JS, so the bearer isn't available on the initial navigation.
    # The per-page permission check lives on each API route below; if
    # the user lacks /quick-config access, the page's first state fetch
    # 403s and the JS renders a "no access" landing.
    dependencies=[Depends(require_admin_host)],
)
async def get_quick_config_page() -> HTMLResponse:
    return HTMLResponse(
        web_common.render_page(_QUICK_CONFIG_HTML, current="quick-config"),
        media_type="text/html",
    )


@router.get(
    "/state",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_page("quick_config")),
    ],
)
async def get_state(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Return rules the caller can see on /quick-config.

    Visibility is a triple-AND: (a) the rule must not be the terminal
    sentinel, (b) it must be `exposed=True`, (c) the caller's
    `quick_config_tags` must intersect `rule.tags` (or `rule.tags` must
    be empty = visible-to-all). Admins bypass (b)+(c).

    All three checks live behind `perms.can_see_rule()` (auth.py) so the
    policy lives in exactly one place — same pattern as the existing
    page-permission gates."""
    perms = user["permissions"]
    canonical = [
        r for r in _canon_rules(list(cfg.PIPELINE_RULES))
        if isinstance(r, dict)
        and r.get("type") != "terminal"
        and perms.can_see_rule(r)
    ]
    # Tag each rule with a fingerprint of its canonical form. Client
    # echoes back per-rule on save; server uses it to detect concurrent
    # edits. Fingerprint is computed AFTER _canon_rules so client and
    # server hash the same canonical bytes.
    for rd in canonical:
        rd["_fp"] = _rule_fingerprint(rd)
    # Server-authoritative "this transcription has been reported" set
    # + the user's previously-submitted chip corrections per request_id.
    # The /quick-config page reads both: badges sync from the id list;
    # the chip map seeds form._corrections so re-opening a reported
    # trace shows what was submitted, instead of an empty form. Capped
    # at 100 newest per user. Failure (init_db never ran, etc.) is
    # non-fatal — the client falls back to its localStorage hint and
    # an empty chip map.
    reported_chips: dict[str, dict[str, Any]] = {}
    try:
        import reports_store
        uid = user.get("user_id") or ""
        if uid:
            my_reports = reports_store.recent_reports_for_user(uid, limit=100)
            for rep in my_reports:
                rid = rep.get("request_id")
                if not rid:
                    continue
                # Newest-first iteration above; first occurrence wins on
                # duplicate request_id (the upsert keeps a single row
                # per (request_id, user_id), so duplicates are rare).
                if rid not in reported_chips:
                    reported_chips[rid] = {
                        "corrections": rep.get("corrections") or [],
                    }
        # No fallback when uid is empty: reports are per-user; open-mode
        # callers see no reported-badge state at all.
    except Exception:
        pass
    return {
        "rules": canonical,
        "role": "admin" if user.get("is_admin") else "user",
        "reported_chips": reported_chips,
    }


@router.get(
    "/usage",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_page("quick_config")),
    ],
)
async def get_my_usage(
    tz_midnight: float | None = None,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """The caller's OWN transcription usage (today + lifetime) for the
    personal banner in the /quick-config subbar. Self-scoped: reads only the
    authenticated user's user_id, so no admin scope is needed and no other
    user's numbers are ever returned. Best-effort — any failure yields zeros
    so the page never breaks.

    'today' resets at the VIEWER's local midnight: the browser passes
    `tz_midnight` (epoch seconds of its local 00:00), and we sum the UTC hours
    since then. When the param is absent/invalid we fall back to the server's
    local day."""
    zero = {"requests": 0, "errors": 0, "words": 0, "audio_s": 0.0}
    uid = user.get("user_id") or ""
    today = dict(zero)
    total = dict(zero)
    if uid:
        try:
            import usage_store
            if tz_midnight and tz_midnight > 0:
                start_hour = usage_store.hour_for_ts(float(tz_midnight))
            else:
                start_hour = usage_store.local_day_start_hour()
            total = usage_store.totals_for_user(uid)
            today = usage_store.totals_for_user(uid, start_hour=start_hour)
        except Exception:
            today, total = dict(zero), dict(zero)
    return {
        "username": user.get("username") or "",
        "today": today,
        "total": total,
    }


@router.post(
    "/state",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_page("quick_config")),
    ],
)
async def post_state(
    payload: QuickPatchPayload,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Apply a per-rule patch. Rejects:
       - patches against rules that aren't currently exposed
       - patches against terminal rules
       - any field outside the per-type allow-list (including admin-only
         fields like `locked`, `name`, `label`, `exposed`, `seeded`)

    Defense order matters: validate + filter the patch BEFORE merging into
    PIPELINE_RULES, so an end-user can't sneak `locked: false` past the
    Pydantic schema (which accepts `locked` as a real field) by routing it
    through the merge step.
    """
    rules_patch = payload.rules_patch
    fingerprints = payload.fingerprints or {}
    if not rules_patch:
        return JSONResponse({
            "saved": [], "conflicts": [],
            "hot_applied": [], "cold_pending": [],
            "env_pinned_ignored": [], "evicted": [],
            "requires_restart": False,
        })

    # Snapshot the current PIPELINE_RULES as plain dicts so we can overlay
    # patches deterministically. Keep order; the merged list will replace
    # cfg.PIPELINE_RULES verbatim (count + order preserved → terminal rule
    # stays last).
    current_rules: list[dict[str, Any]] = []
    for r in cfg.PIPELINE_RULES:
        if hasattr(r, "model_dump"):
            current_rules.append(r.model_dump())
        else:
            current_rules.append(dict(r))
    by_slug = {r.get("name"): i for i, r in enumerate(current_rules)}
    # Canonicalize for fingerprint comparison — must hash the same shape
    # /state served. _canon_rules drops None fields and sorts dict keys.
    canonical_now = {r["name"]: r for r in _canon_rules(current_rules)
                     if isinstance(r, dict) and r.get("name")}

    saved: list[str] = []
    conflicts: list[dict[str, Any]] = []
    for slug, patch in rules_patch.items():
        if not isinstance(patch, dict):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"rules_patch['{slug}'] must be an object",
            )
        idx = by_slug.get(slug)
        if idx is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"unknown rule slug: '{slug}' "
                f"(adding rules from /quick-config is not allowed)",
            )
        target = current_rules[idx]
        # Defense-in-depth: must pass the same can_see_rule check the GET
        # uses, so a user can't PATCH a rule their tag set forbids them
        # from seeing (which would have been hidden in the UI anyway).
        # The exposed + terminal + admin checks all live inside
        # can_see_rule, so this single call replaces the bare
        # `target.get("exposed")` test from before.
        if not user["permissions"].can_see_rule(target):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"rule '{slug}' is not visible to your user",
            )
        rtype = target.get("type", "?")
        if rtype == "terminal":
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "the terminal rule cannot be edited from /quick-config",
            )
        allowed = _PATCH_ALLOWED_FIELDS.get(rtype, frozenset())
        for field in patch.keys():
            if field not in allowed:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"field '{field}' is not editable on rule '{slug}' "
                    f"(type={rtype}) from /quick-config",
                )
        # Optimistic-concurrency check: if the client included a
        # fingerprint for this rule, it must match what's on disk now.
        # Mismatch → another writer (admin or another /quick-config tab)
        # changed this rule between load and save. Skip the patch and
        # report conflict rather than silently overwrite. Patches without
        # a fingerprint fall through to the legacy last-writer-wins path
        # (kept so older clients / curl scripts still work).
        client_fp = fingerprints.get(slug)
        if client_fp:
            current = canonical_now.get(slug, {})
            current_fp = _rule_fingerprint(current) if current else None
            if current_fp != client_fp:
                conflicts.append({"slug": slug, "current_fp": current_fp})
                continue
        # Server-owned map_meta: stamp added/value-changed cb:map entries with
        # the current epoch (added-or-last-updated), drop meta for removed keys.
        # Done here — after the conflict check, before the overlay — so the
        # client can't forge timestamps and the fingerprint above still compared
        # against the shape /state served. The mutation lands on `target` (an
        # element of current_rules) and so reaches save_overrides below.
        if rtype == "callback:map" and "map" in patch:
            old_map = target.get("map") or {}
            new_map = patch["map"] or {}
            meta = dict(target.get("map_meta") or {})
            now = int(time.time())
            for k, v in new_map.items():
                if k not in old_map or old_map[k] != v:
                    meta[k] = now
            target["map_meta"] = {k: meta[k] for k in new_map if k in meta}
        target.update(patch)
        saved.append(slug)

    # If every patch conflicted, skip the save+rebuild entirely — nothing
    # to write. Return the conflict list so the client can refetch.
    if not saved:
        return JSONResponse({
            "saved": [],
            "conflicts": conflicts,
            "hot_applied": [], "cold_pending": [],
            "env_pinned_ignored": [], "evicted": [],
            "requires_restart": False,
        })

    # Hand off the merged list to the same save path the admin uses. This
    # re-runs the full Pydantic validation including the 2 s ReDoS guard;
    # an error may name a rule the user didn't touch — the client surfaces
    # this gracefully ("admin's pipeline has an error").
    try:
        written = config_store.save_overrides({"PIPELINE_RULES": current_rules})
    except ValidationError as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"errors": config_store.format_validation_errors(e)},
        )
    except OSError as e:
        logger.error("[quick-config] save failed: %s", e)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"could not write config.local.json: {e}",
        )

    applied = await _apply_hot_changes(written)

    client_host = request.client.host if request.client else "?"
    logger.info(
        "[quick-config] update from=%s user=%s admin=%s saved=%s conflicts=%s",
        client_host, user.get("username"), user.get("is_admin"),
        saved, [c["slug"] for c in conflicts],
    )

    captures_count = 0
    if getattr(cfg, "CAPTURE_RECORDINGS_ENABLED", False):
        try:
            import captures_store
            captures_count = captures_store.count()
        except Exception as _e:
            logger.warning("[quick-config] capture count lookup failed: %s", _e)

    return JSONResponse({
        "saved": saved,
        "conflicts": conflicts,
        **applied,
        "requires_restart": bool(applied["cold_pending"]),
        "captures_count": captures_count,
    })


# --- Re-apply current pipeline rules to existing captures ------------
#
# Quick-config rules are only baked into a capture's `final` at
# transcription time. After a rule edit, historical captures are
# frozen against the rule set at their time of capture. These two
# endpoints drive a background backfill job that re-runs the current
# pipeline over every capture's raw text, updates `final`, and
# rebuilds affected (unlocked) group transcript snapshots from the
# now-refreshed member text. `corrected_text` and chip corrections
# are admin-authoritative — preserved untouched.

@router.post(
    "/reapply-rules",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_page("quick_config")),
    ],
)
async def post_reapply_rules(
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    if not user.get("is_admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    if not getattr(cfg, "CAPTURE_RECORDINGS_ENABLED", False):
        return JSONResponse({"status": "idle", "note": "captures disabled"})
    import captures_reapply
    return JSONResponse(captures_reapply.start())


@router.get(
    "/reapply-rules/status",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_page("quick_config")),
    ],
)
async def get_reapply_rules_status(
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    if not user.get("is_admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    import captures_reapply
    return JSONResponse(captures_reapply.status())


# --- Recent transcription traces (panel + autocomplete source) -------------
#
# The ring buffer lives in quick_config_state. main.py's transcribe handler
# appends an entry per completed transcription. Both endpoints below are
# token-gated so end-users without a valid token can't enumerate recent
# patient dictation snippets.

@router.get(
    "/recent",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_page("quick_config")),
    ],
)
async def get_recent(
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Newest-first slice of the durable recent-transcriptions store.
    The /stream endpoint replays the same slice on connect; /recent
    additionally supports the "Load older" pagination cursor.

    Query params:
      before_ts (float, optional) — cursor: return rows STRICTLY older
        than this created_ts. Omit / 0 to fetch the freshest slice.
      limit (int, optional)       — clamped to cfg.RECENT_TRANSCRIPTIONS_PAGE_SIZE.

    Response:
      {recent: [...], next_before_ts: <float|null>}
        next_before_ts is the oldest entry's created_ts when the slice
        was fully filled (caller can re-query with that as before_ts);
        null when this is the last batch.

    Scope-aware: scope=own users see only their own rows; scope=all
    users (admins) see every row. The username column is materialized
    at write so the read path doesn't hit api_keys_store.

    q (str, optional) — free-text filter: only rows whose raw/final text
    contain the substring (case-insensitive). Pagination via before_ts
    stays within the matching set."""
    perms = user["permissions"]
    caller_uid = user.get("user_id") or ""
    sees_all = perms.scope("quick_config") == "all"

    page_size = int(getattr(cfg, "RECENT_TRANSCRIPTIONS_PAGE_SIZE", 100))
    try:
        q_before = float(request.query_params.get("before_ts", "") or 0.0)
    except (TypeError, ValueError):
        q_before = 0.0
    try:
        q_limit = int(request.query_params.get("limit", "") or page_size)
    except (TypeError, ValueError):
        q_limit = page_size
    q_limit = max(1, min(q_limit, page_size))
    q_search = (request.query_params.get("q") or "").strip()

    user_filter = None if sees_all else caller_uid
    traces = transcriptions_store.list_recent(
        before_ts=q_before if q_before > 0 else None,
        limit=q_limit,
        user_id_filter=user_filter,
        query=q_search or None,
    )
    next_before = traces[-1]["created_ts"] if len(traces) >= q_limit else None
    return {"recent": traces, "next_before_ts": next_before}


@router.get(
    "/stream",
    dependencies=[Depends(require_admin_host)],
)
async def stream_recent(
    request: Request,
    user: dict[str, Any] = Depends(require_user_or_admin_sse),
) -> StreamingResponse:
    """Server-sent events stream of recent transcriptions.

    On connect, replays the current buffer (`event: trace` for each entry,
    oldest first). After the replay, pushes any new transcription as
    another `event: trace`. Sends a `: keepalive` SSE comment line every
    15 s so reverse proxies don't kill an idle connection.

    Scope-aware: `scope=own` filters replay AND live items to the
    caller's user_id; `scope=all` lets everything through. Live items
    flow through quick_config_state.subscribe()'s shared queue — every
    subscriber gets every event — so filtering happens here per
    subscriber rather than at the publisher (no extra queue infra)."""
    perms = user["permissions"]
    caller_uid = user.get("user_id") or ""
    sees_all = perms.scope("quick_config") == "all"

    def _visible(entry: dict[str, Any] | None) -> bool:
        if sees_all:
            return True
        # caller_uid is already coerced from None to "" above; coerce the
        # entry side too so a persisted row with user_id=NULL doesn't get
        # silently excluded for a caller whose own user_id is missing.
        return bool(entry) and (entry.get("user_id") or "") == caller_uid

    async def gen():
        q = quick_config_state.subscribe()
        try:
            # Replay the freshest page from the durable store (oldest-
            # first so the client receives them in chronological order,
            # matching the prior in-memory deque iteration semantics).
            page_size = int(getattr(cfg, "RECENT_TRANSCRIPTIONS_PAGE_SIZE", 100))
            replay = transcriptions_store.list_recent(
                limit=page_size,
                user_id_filter=None if sees_all else caller_uid,
            )
            for entry in reversed(replay):
                if _visible(entry):
                    yield f"event: trace\ndata: {json.dumps(entry)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=15.0)
                    ev = item.get("event", "trace")
                    payload = item.get("data") or {}
                    if ev == "trace" and not _visible(payload):
                        continue
                    yield f"event: {ev}\ndata: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            quick_config_state.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


_QUICK_CONFIG_HTML = r"""<!doctype html>
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
    --red: #ff7b72; --magenta: #d2a8ff; --bold: #f0f6fc;
    --border: #30363d; --input-bg: #0d1117;
  }
  html, body { background: var(--bg); color: var(--fg);
    font-family: var(--font-sans); font-size: var(--fs-base); margin: 0; }
  a { color: var(--cyan); }
  /* header / .header-inner / .title / page-toolbar controls (buttons,
     pills, the #status text) are all centralized in NAV_CSS. */
  main { padding: 1rem; max-width: 60rem; margin: 0 auto; }
  /* Personal self-usage banner in the subbar's empty middle. Grows to fill,
     truncates on narrow screens; the subbar's own flex-wrap drops it to its
     own line when space is tight. */
  header .subbar .qc-usage { flex: 1 1 auto; min-width: 0; text-align: center;
    font-size: var(--fs-sm); color: var(--dim); white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; }
  header .subbar .qc-usage:empty { display: none; }
  .qc-usage .u-name { color: var(--bold); font-weight: 600; }
  .qc-usage .u-sep { color: var(--dim); margin: 0 0.5rem; }
  .qc-usage .u-num { font-family: var(--font-mono); }
  .qc-usage .u-today { color: var(--cyan); }
  .qc-usage .u-total { color: var(--fg); }
  .card { background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.75rem 1rem; margin-bottom: 0.75rem; }
  .card h3 { margin: 0 0 0.25rem 0; font-size: var(--fs-lg); color: var(--bold);
    display: flex; align-items: baseline; gap: 0.5rem; }
  .card .type-pill { display: inline-block; padding: 0 0.375rem;
    border-radius: 3px; font-size: var(--fs-xs); background: #21262d;
    color: var(--cyan); font-weight: normal; }
  .card .enabled-row { display: flex; align-items: center; gap: 0.4rem;
    margin: 0.4rem 0; font-size: var(--fs-sm); color: var(--dim); }
  .card .enabled-row input { margin: 0; }
  .card .help { color: var(--help); font-size: var(--fs-sm);
    margin-top: 0.375rem; }
  .card .rule-editor { margin-top: 0.4rem; display: flex;
    flex-direction: column; gap: 0.25rem; font-family: var(--font-mono);
    font-size: var(--fs-sm); }
  .card .rule-editor input, .card .rule-editor textarea {
    box-sizing: border-box;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 0.25rem 0.4rem; font: inherit; font-family: var(--font-mono);
    font-size: var(--fs-sm); }
  /* Browser-default focus outline draws OUTSIDE the input's box, so when
     the input fills its td (width:100%) the blue ring overflows the
     table cell on the right. Negative outline-offset overlays the
     existing 1px border instead. */
  .card .rule-editor input:focus-visible,
  .card .rule-editor textarea:focus-visible {
    outline: 2px solid var(--cyan); outline-offset: -2px;
  }
  .card .rule-editor textarea { width: 100%; resize: vertical; }
  .card .rule-editor .map-table { width: 100%; }
  .card .rule-editor .map-table input { width: 100%; }
  /* Inline "added / last-updated" date column (last cell). Dim + mono, fixed
     width so the key/value inputs keep their space. */
  .card .rule-editor .map-table td.map-date-cell {
    width: 11rem; padding-left: 0.4rem; text-align: right;
    white-space: nowrap; vertical-align: middle; }
  /* Phone: a fixed 11rem date column eats ~half a 360px screen, crushing the
     key/value inputs — let it shrink to its content and wrap under them. */
  @media (max-width: 40em) {
    .card .rule-editor .map-table td.map-date-cell {
      width: auto; white-space: normal; }
  }
  .card .rule-editor .map-date {
    color: var(--dim); font-size: var(--fs-xs); font-family: var(--font-mono); }
  /* Collapse the oldest entries behind the toggle. _readMap still reads them
     (display:none keeps them in the DOM). */
  .card .rule-editor .map-table:not(.show-all) tr.map-row-collapsed {
    display: none; }
  .card .rule-editor button.map-toggle {
    align-self: flex-start; background: none; border: none; padding: 0.2rem 0;
    color: var(--dim); font-size: var(--fs-sm); cursor: pointer; }
  .card .rule-editor button.map-toggle:hover:not(:disabled) {
    background: none; color: var(--cyan); }
  .card .rule-editor button { background: var(--panel);
    border: 1px solid var(--border); color: var(--fg);
    padding: 0.4rem 0.9rem; border-radius: 3px; cursor: pointer;
    font: inherit; font-size: var(--fs-md); line-height: 1.4; }
  .card .rule-editor button:hover:not(:disabled) {
    background: #21262d; color: var(--bold); }
  .card .rule-editor button:disabled {
    opacity: 0.4; cursor: not-allowed; }
  .card .rule-editor button.primary:not(:disabled) {
    color: var(--green); border-color: var(--green); }
  .card .rule-editor button.primary:hover:not(:disabled) {
    background: rgba(46, 160, 67, 0.12); }
  .empty-state { text-align: center; color: var(--dim);
    padding: 4rem 1rem; font-size: var(--fs-md); }
  .empty-state h2 { color: var(--fg); font-size: var(--fs-lg);
    margin: 0 0 0.5rem 0; }
  #toast { position: fixed; bottom: 1rem; right: 1rem; padding: 0.5rem 1rem;
    background: var(--panel); border: 1px solid var(--border); border-radius: 4px;
    color: var(--fg); font-size: var(--fs-sm); display: none; z-index: 10; }
  #toast.err { border-color: var(--red); color: var(--red); }
  #toast.ok { border-color: var(--green); color: var(--green); }
  /* API-key login modal — matches /captures, /reports, /settings */
  #token-modal { position: fixed; inset: 0; background: rgba(0,0,0,0.65);
    display: none; align-items: center; justify-content: center; z-index: 8; }
  #token-modal.show { display: flex; }
  #token-modal .box {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.4rem 1.5rem 1.2rem;
    width: 30rem; max-width: 92vw;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
  }
  #token-modal h3 {
    margin: 0 0 0.5rem 0; color: var(--bold); font-size: var(--fs-xl);
  }
  #token-modal p {
    margin: 0 0 0.9rem 0; line-height: 1.45;
    color: var(--help, var(--dim)); font-size: var(--fs-sm);
  }
  #token-modal p code { color: var(--fg); font-family: var(--font-mono); }
  #token-modal input {
    box-sizing: border-box; width: 100%;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.55rem 0.7rem; font-family: var(--font-mono);
    font-size: var(--fs-md); line-height: 1.4;
  }
  #token-modal input:focus { outline: none; border-color: var(--cyan); }
  #token-modal .actions {
    display: flex; gap: 0.6rem; justify-content: flex-end;
    margin-top: 1.1rem; padding-top: 0.85rem;
    border-top: 1px solid var(--border);
  }
  #token-modal .actions button {
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
  #token-modal .actions button:hover { background: #21262d; color: var(--bold); }
  #token-modal .actions button.primary {
    color: var(--green); border-color: var(--green);
  }

  /* Recent transcriptions panel */
  #recent-panel { margin-top: 1.5rem; border-top: 1px solid var(--border);
    padding-top: 1rem; }
  .recent-header { display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 0.5rem; flex-wrap: wrap; }
  .recent-header h2 { font-size: var(--fs-lg); margin: 0; color: var(--bold); }
  .recent-header .recent-label { font-size: var(--fs-xs); color: var(--dim);
    font-style: italic; }
  .recent-header .spacer { flex: 1; }
  /* Free-text search for the recent list — mirrors the /captures subbar
     search. margin-left:auto pushes it to the right edge of the header;
     the magnifier replaces a "search" label to save width. Sized in
     em/rem so it tracks the scale picker. */
  .recent-header .subbar-search { flex: 1 1 auto; min-width: 8rem;
    max-width: 28rem; margin-left: auto; display: inline-flex;
    align-items: center; gap: 0.35rem; }
  .recent-header .subbar-search .search-ico { flex: 0 0 auto;
    width: 0.95em; height: 0.95em; stroke: var(--help); fill: none;
    stroke-width: 2; }
  .recent-header .subbar-search input[type="text"] { flex: 1 1 auto;
    min-width: 4rem; max-width: none; }
  .recent-pager { display: flex; justify-content: center; padding: 0.5rem 0 1rem; }
  .recent-pager button { background: transparent; color: var(--cyan);
    border: 1px solid var(--border); padding: 0.4rem 0.9rem;
    border-radius: 4px; font: inherit; cursor: pointer; }
  .recent-pager button:hover { background: var(--panel); }
  .recent-pager button[disabled] { opacity: 0.5; cursor: default; }
  .empty-recent { color: var(--dim); font-style: italic; padding: 1rem 0;
    font-size: var(--fs-sm); }
  .trace-item { background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem; }
  .trace-meta { display: flex; gap: 0.5rem; color: var(--dim);
    font-size: var(--fs-xs); margin-bottom: 0.375rem; align-items: center; }
  .trace-meta .pill { background: var(--panel-alt, rgba(255,255,255,0.04));
    border: 1px solid var(--border); border-radius: 999rem;
    padding: 0.05rem 0.5rem; color: var(--fg); font-family: var(--font-sans);
    font-size: var(--fs-xs); }
  .trace-text { font-family: var(--font-mono); font-size: var(--fs-sm);
    word-wrap: break-word; }
  .trace-raw { color: var(--dim); margin-bottom: 0.25rem; }
  .trace-tag { display: inline-block; min-width: 3rem; color: var(--dim);
    font-family: var(--font-sans); font-size: var(--fs-xs);
    margin-right: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;
    /* Keep double-click word-selection off the tag so selecting the first
       word of the adjacent text doesn't also grab "RAW"/"FINAL". */
    -webkit-user-select: none; user-select: none; }
  .trace-final { color: var(--bold); }
  .trace-final .trace-tag { color: var(--green); }
  details.trace-steps { margin-top: 0.375rem; }
  details.trace-steps > summary { cursor: pointer; font-size: var(--fs-xs);
    color: var(--cyan); list-style: revert; user-select: none; }
  .trace-step { padding: 0.25rem 0 0.25rem 0.75rem; font-size: var(--fs-xs);
    border-left: 2px solid var(--border); margin-top: 0.25rem; }
  .trace-step.skipped { opacity: 0.55; }
  .trace-step .step-label { color: var(--dim); display: block;
    font-family: var(--font-sans); margin-bottom: 0.125rem;
    -webkit-user-select: none; user-select: none; }
  .trace-step .step-label .skipped-tag { color: var(--yellow);
    margin-left: 0.25rem; }
  .trace-step .step-before, .trace-step .step-after {
    font-family: var(--font-mono); display: block; word-wrap: break-word; }
  .trace-step .step-before { color: var(--dim); }
  .trace-step .step-after { color: var(--fg); }
  .trace-step .step-arrow { color: var(--green); margin-right: 0.25rem; }

  /* Per-trace report row + inline form */
  .trace-actions { display: flex; align-items: center; gap: 0.5rem;
    margin-top: 0.5rem; flex-wrap: wrap; }
  .trace-report-btn { background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 0.125rem 0.5rem; font: inherit; font-size: var(--fs-xs);
    cursor: pointer; }
  .trace-report-btn:hover { background: #21262d; color: var(--bold); }
  .trace-reported-badge { color: var(--green); font-size: var(--fs-xs);
    border: 1px solid var(--green); border-radius: 3px;
    padding: 0.05rem 0.4rem; font-family: var(--font-sans); }
  .trace-report-form { display: none; margin-top: 0.5rem;
    border-top: 1px dashed var(--border); padding-top: 0.5rem; }
  .trace-report-form.open { display: block; }
  .trace-report-form .form-label { font-size: var(--fs-sm);
    color: var(--fg); margin: 0.5rem 0 0.25rem; display: block;
    font-family: var(--font-sans); }
  .trace-report-form .form-label .help { color: var(--help);
    font-size: var(--fs-xs); font-weight: normal; margin-left: 0.25rem; }
  .rep-final-words { font-family: var(--font-mono); font-size: var(--fs-sm);
    line-height: 1.6; padding: 0.375rem 0.5rem; background: var(--input-bg);
    border: 1px solid var(--border); border-radius: 3px;
    word-wrap: break-word; }
  .rep-final-words .word { cursor: pointer; padding: 0 0.05rem;
    border-radius: 2px; }
  .rep-final-words .word:hover { background: #21262d; }
  .rep-final-words .word.selected { color: var(--red);
    text-decoration: line-through; background: rgba(255, 123, 114, 0.12); }
  /* Inline replacement next to a struck-through word — mirrors the
     /captures word strip and /reports' .diff-ins green look so the
     correction reads as track-changes inline, not just in the chip
     panel below. */
  .rep-final-words .word-replacement {
    color: var(--green); font-weight: 600;
    background: rgba(126, 231, 135, 0.10);
    padding: 0 0.25rem; margin-left: 0.25rem;
    border-radius: 2px;
    font-family: var(--font-mono);
  }
  .rep-corrections { display: flex; flex-wrap: wrap; gap: 0.375rem;
    margin-top: 0.375rem; }
  .rep-corrections:empty { display: none; }
  .rep-correction { display: inline-flex; align-items: center;
    gap: 0.25rem; background: var(--panel); border: 1px solid var(--border);
    border-radius: 3px; padding: 0.125rem 0.4rem; font-family: var(--font-mono);
    font-size: var(--fs-sm); }
  .rep-correction .rep-wrong { color: var(--red);
    text-decoration: line-through; }
  .rep-correction .rep-arrow { color: var(--dim); }
  .rep-correction .rep-correct { background: var(--input-bg);
    color: var(--green); border: 1px solid var(--border); border-radius: 3px;
    padding: 0 0.25rem; font: inherit; font-family: var(--font-mono);
    font-size: var(--fs-sm); min-width: 6rem; }
  .rep-correction .rep-correct:focus-visible { outline: 1px solid var(--cyan);
    outline-offset: 0; }
  .rep-correction .rep-correction-remove { background: transparent;
    border: none; color: var(--dim); cursor: pointer;
    font-size: var(--fs-md); padding: 0 0.25rem; line-height: 1; }
  .rep-correction .rep-correction-remove:hover { color: var(--red); }
  .trace-report-form textarea { box-sizing: border-box; width: 100%;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 0.25rem 0.4rem; font-family: var(--font-sans);
    font-size: var(--fs-sm); resize: vertical; }
  .trace-report-form textarea:focus-visible {
    outline: 2px solid var(--cyan); outline-offset: -2px; }
  .rep-actions { display: flex; gap: 0.5rem; margin-top: 0.5rem;
    justify-content: flex-end; }
  .rep-actions button { background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 0.125rem 0.625rem; font: inherit; font-size: var(--fs-sm);
    cursor: pointer; }
  .rep-actions button.primary { color: var(--green); border-color: var(--green); }
  .rep-actions button:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Non-modal silent-reapply strip — shown under the header bar after
     Save kicks off the reapply job automatically. Fades out 3s after
     completion. */
  #reapply-strip {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.35rem 0.75rem;
    background: var(--input-bg); border-bottom: 1px solid var(--border);
    font-family: var(--font-sans); font-size: var(--fs-sm);
    color: var(--fg);
    transition: opacity 0.4s ease-out;
  }
  #reapply-strip[hidden] { display: none; }
  #reapply-strip.fade-out { opacity: 0; }
  #reapply-strip.err { background: rgba(220, 80, 80, 0.12);
    border-bottom-color: var(--red, #c84343); }
  #reapply-strip .r-label { color: var(--dim); flex-shrink: 0; }
  #reapply-strip .r-bar {
    display: inline-block; height: 0.5rem; flex: 1; min-width: 4rem;
    background: var(--border); border-radius: 2px; overflow: hidden;
  }
  #reapply-strip .r-fill {
    display: block; height: 100%; width: 0%; background: var(--green);
    transition: width 0.3s ease-out;
  }
  #reapply-strip .r-stats { font-family: var(--font-mono);
    font-size: var(--fs-xs); color: var(--dim); flex-shrink: 0; }

  {{NAV_CSS}}
</style>
</head>
<body>

<header>
  <div class="header-inner">
    <span class="title">{{HEADER_BRAND}}</span>
    <span class="brand-sep" aria-hidden="true"></span>
    {{NAV}}
    <span class="spacer"></span>
    <span class="hdr-right">{{SEV_PILLS}}{{SCALE_PICKER}}{{RELOAD}}{{LOGOUT}}</span>
  </div>
  <div class="subbar">
    <span class="subbar-title">Quick config</span>
    <div class="qc-usage" id="qc-usage"></div>
    <div class="subbar-right">
      <button id="discard-btn" disabled>discard</button>
      <button id="save-btn" class="primary" disabled>save</button>
      <span id="status">loading…</span>
    </div>
  </div>
</header>

<div id="reapply-strip" hidden>
  <span class="r-label">Re-applying…</span>
  <span class="r-bar"><span class="r-fill"></span></span>
  <span class="r-stats">0 / 0</span>
</div>

<datalist id="recent-words"></datalist>

<main>
  <section id="cards"></section>
  <section id="recent-panel">
    <div class="recent-header">
      <h2>Recent transcriptions</h2>
      <span class="recent-label" title="Persisted in SQLite (WAL). Caps and TTL are configurable at /settings.">persistent · paginated</span>
      <label class="subbar-search" title="search recent transcriptions">
        <svg class="search-ico" viewBox="0 0 24 24" aria-hidden="true" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg>
        <input id="recent-search" type="text" aria-label="search recent transcriptions" placeholder="text in raw / final">
      </label>
    </div>
    <div id="recent-list">
      <div class="empty-recent">No transcriptions yet — they'll appear here as you dictate.</div>
    </div>
    <div id="recent-pager" class="recent-pager">
      <button type="button" id="btn-load-older" class="ghost" style="display:none;">Load older</button>
    </div>
  </section>
</main>

<div id="toast"></div>

<div id="token-modal">
  <div class="box">
    <h3>API key</h3>
    <p style="color:var(--dim);font-size:var(--fs-sm);margin:0 0 0.5rem 0;">
      Paste your <code>wk_…</code> API key. Your admin issues one per user in
      <code>/settings/api-keys</code>.
    </p>
    <input id="token-input" type="password" autocomplete="off" spellcheck="false"
           placeholder="wk_…">
    <div class="actions">
      <button id="token-cancel">cancel</button>
      <button id="token-ok" class="primary">ok</button>
    </div>
  </div>
</div>

{{RULE_EDITOR_JS}}

<script>
(() => {
'use strict';

let initialRules = [];      // last-loaded rules from server (deep-copy snapshot)
let liveRules = [];         // editable rules — diffed against initialRules to build patch
let dirty = new Set();      // slugs with changes

// Exchange a pasted key for the HttpOnly session cookie. Returns true on
// success. Dispatches whisper:auth-changed so the shared chrome refreshes.
async function doLogin(key) {
  try {
    const r = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key }),
    });
    if (!r.ok) return false;
    try { window.dispatchEvent(new Event('whisper:auth-changed')); } catch(_) {}
    return true;
  } catch (_) { return false; }
}

async function api(method, path, body) {
  // Session cookie sent automatically; mutations carry the CSRF token.
  const h = { 'Accept': 'application/json' };
  if (method !== 'GET' && method !== 'HEAD') {
    h['X-CSRF-Token'] = window._csrfToken ? window._csrfToken() : '';
  }
  const opts = { method, headers: h };
  if (body !== undefined) {
    h['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  return fetch(path, opts);
}

function showToast(msg, kind) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = kind || '';
  el.style.display = 'block';
  setTimeout(() => { el.style.display = 'none'; }, 3500);
}

function setStatus(s) {
  document.getElementById('status').textContent = s;
}

function promptToken() {
  return new Promise((resolve) => {
    const m = document.getElementById('token-modal');
    const inp = document.getElementById('token-input');
    inp.value = '';
    m.classList.add('show');
    inp.focus();
    function done(val) {
      m.classList.remove('show');
      document.getElementById('token-ok').onclick = null;
      document.getElementById('token-cancel').onclick = null;
      inp.onkeydown = null;
      resolve(val);
    }
    document.getElementById('token-ok').onclick = () => done(inp.value);
    document.getElementById('token-cancel').onclick = () => done(null);
    inp.onkeydown = (e) => { if (e.key === 'Enter') done(inp.value); };
  });
}

async function ensureToken() {
  // Probe /quick-config/state. On 401 prompt; retry once. On 403
  // (valid bearer, no quick_config scope) render the shared no-access
  // landing and keep the bearer so the user can navigate elsewhere
  // without re-logging in.
  let r = await api('GET', '/quick-config/state');
  if (r.status === 401) {
    const t = await promptToken();
    if (!t) return null;
    if (!(await doLogin(t))) {
      showToast('invalid key', 'err');
      return null;
    }
    r = await api('GET', '/quick-config/state');
  }
  if (r.status === 403 && typeof _renderNoAccessLanding === 'function') {
    // Refresh whoami so the landing can list reachable pages.
    try {
      const w = await fetch('/auth/whoami');
      if (w.ok) window.__whoami = await w.json();
    } catch (_) {}
    _renderNoAccessLanding({ page: 'quick_config' });
    return null;
  }
  return r;
}

function commitData(slug) {
  dirty.add(slug);
  updateButtons();
}
// Per-card Save buttons, keyed by rule slug. Rebuilt on every renderCards()
// so an entry only ever refers to a button currently in the DOM.
let _perCardSaveBtns = new Map();
function _buildPerCardSaveBtn(slug) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'primary';
  btn.textContent = 'save';
  btn.disabled = !dirty.has(slug);
  btn.addEventListener('click', doSave);
  _perCardSaveBtns.set(slug, btn);
  return btn;
}
function updateButtons() {
  const has = dirty.size > 0;
  document.getElementById('save-btn').disabled = !has;
  document.getElementById('discard-btn').disabled = !has;
  setStatus(has ? (dirty.size + ' rule' + (dirty.size === 1 ? '' : 's') + ' modified')
                : (initialRules.length + ' rule' + (initialRules.length === 1 ? '' : 's') + ' loaded'));
  for (const [slug, btn] of _perCardSaveBtns) {
    btn.disabled = !dirty.has(slug);
  }
}

function renderCards() {
  const root = document.getElementById('cards');
  root.innerHTML = '';
  _perCardSaveBtns = new Map();  // detach stale refs; GC takes the old DOM
  if (!liveRules.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.innerHTML = '<h2>Nothing to configure here yet</h2>'
      + '<p>No rules have been exposed by your admin yet.</p>';
    root.appendChild(empty);
    return;
  }
  for (const rule of liveRules) {
    const card = document.createElement('div');
    card.className = 'card';
    const h3 = document.createElement('h3');
    const labelText = document.createTextNode(rule.label || rule.name);
    h3.appendChild(labelText);
    const pill = document.createElement('span');
    pill.className = 'type-pill';
    pill.textContent = _typePill(rule.type);
    h3.appendChild(pill);
    card.appendChild(h3);

    const enRow = document.createElement('label');
    enRow.className = 'enabled-row';
    const enCb = document.createElement('input');
    enCb.type = 'checkbox';
    enCb.checked = !!rule.enabled;
    enCb.addEventListener('change', () => {
      rule.enabled = enCb.checked;
      commitData(rule.name);
    });
    enRow.appendChild(enCb);
    enRow.appendChild(document.createTextNode(' enabled'));
    card.appendChild(enRow);

    const slug = rule.name;
    const editor = renderTypeEditor(rule, () => commitData(slug), {
      datalistId: 'recent-words',
      makeSaveBtn: () => _buildPerCardSaveBtn(slug),
      commitOnEnter: doSave,
      showMapDates: true,
      collapseMapAfter: 15,
    });
    card.appendChild(editor);

    root.appendChild(card);
  }
}

// ---------- Recent transcriptions panel + autocomplete -------------------
//
// SSE-driven for the freshest page; durable store backs the "Load older"
// pagination cursor. /quick-config/stream replays the freshest server-
// configured page on connect (oldest-first), then pushes new traces as
// `event: trace`. EventSource auto-reconnects with a 3 s backoff per
// the WHATWG spec — no custom retry logic needed. _oldestLoadedTs tracks
// the bottom edge of the in-DOM list so "Load older" can issue a
// before_ts cursor.
const _MAX_DATALIST = 200;
let _es = null;
let _oldestLoadedTs = null;
let _loadOlderBusy = false;
// request_ids already rendered, so the SSE on-connect replay (which always
// re-sends the freshest page) doesn't duplicate rows already placed by the
// initial /quick-config/recent fetch or by "Load older".
let _seenReqIds = new Set();
// Active free-text filter (server-side substring over raw/final). Empty =
// unfiltered newest-first view.
let _searchQuery = '';
// Active stream-recovery poller (mirrors /stats); non-null while reconnecting.
let _recoveryTimer = null;

function _entryKey(entry) {
  if (!entry) return '';
  if (entry.request_id) return 'r:' + entry.request_id;
  if (entry.id != null) return 'i:' + entry.id;
  return '';
}
// Returns true if this entry was already rendered (and should be skipped);
// otherwise records it as seen and returns false. Entries without any id are
// never deduped (always allowed through).
function _seenTrace(entry) {
  const k = _entryKey(entry);
  if (!k) return false;
  if (_seenReqIds.has(k)) return true;
  _seenReqIds.add(k);
  return false;
}
function _matchesSearch(entry) {
  if (!_searchQuery) return true;
  const q = _searchQuery.toLowerCase();
  return (((entry && entry.raw) || '') + ' ' + ((entry && entry.final) || ''))
    .toLowerCase().indexOf(q) !== -1;
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s == null ? '' : String(s);
  return div.innerHTML;
}
// absTime is injected via TIME_HELPERS_JS.

function renderTrace(entry) {
  const item = document.createElement('div');
  item.className = 'trace-item';
  // Stash the entry so rebuildDatalist() can read tokens/bigrams without
  // a separate cache. JSON encoding is fine — tokens/bigrams are small.
  try { item.dataset.entry = JSON.stringify(entry); } catch (_) {}

  const meta = document.createElement('div');
  meta.className = 'trace-meta';
  const ts = document.createElement('span');
  ts.className = 'trace-ts';
  const _ets = entry.ts || 0;
  ts.textContent = fmtWhen(_ets);
  if (_ets) { ts.dataset.ts = String(_ets); ts.title = absTime(_ets); }
  meta.appendChild(ts);
  if (entry.model) {
    const mdl = document.createElement('span');
    mdl.textContent = entry.model;
    meta.appendChild(mdl);
  }
  if (entry.username) {
    const spk = document.createElement('span');
    spk.className = 'pill';
    spk.title = 'speaker';
    spk.textContent = entry.username;
    meta.appendChild(spk);
  } else if (entry.user_id) {
    const spk = document.createElement('span');
    spk.className = 'pill';
    spk.title = 'speaker (unknown user)';
    spk.textContent = String(entry.user_id).slice(0, 6);
    meta.appendChild(spk);
  }
  item.appendChild(meta);

  const raw = document.createElement('div');
  raw.className = 'trace-text trace-raw';
  raw.innerHTML = '<span class="trace-tag">raw</span>'
    + '<span class="ws-region">' + escapeHtml(entry.raw || '') + '</span>';
  item.appendChild(raw);

  const steps = entry.steps || [];
  const changed = steps.filter(s =>
    Array.isArray(s) && s.length >= 3 && s[1] !== s[2]);
  if (steps.length) {
    const det = document.createElement('details');
    det.className = 'trace-steps';
    if (changed.length) det.open = true;  // open by default if anything changed
    const sum = document.createElement('summary');
    sum.textContent = 'Pipeline steps (' + changed.length + ' changed text'
      + (steps.length > changed.length
          ? ', ' + (steps.length - changed.length) + ' unchanged'
          : '')
      + ')';
    det.appendChild(sum);
    for (const s of steps) {
      if (!Array.isArray(s) || s.length < 3) continue;
      const [label, before, after] = s;
      const stepEl = document.createElement('div');
      stepEl.className = 'trace-step' + (before === after ? ' skipped' : '');
      const lblHtml = escapeHtml(label || '?');
      const skippedTag = before === after
        ? ' <span class="skipped-tag">[unchanged]</span>'
        : '';
      stepEl.innerHTML =
        '<span class="step-label">▸ ' + lblHtml + skippedTag + '</span>'
        + '<span class="step-before"><span class="ws-region">'
        + escapeHtml(before || '') + '</span></span>'
        + '<span class="step-after"><span class="step-arrow">→</span>'
        + '<span class="ws-region">' + escapeHtml(after || '') + '</span></span>';
      det.appendChild(stepEl);
    }
    item.appendChild(det);
  }

  const final = document.createElement('div');
  final.className = 'trace-text trace-final';
  final.innerHTML = '<span class="trace-tag">final</span>'
    + '<span class="ws-region">' + escapeHtml(entry.final || '') + '</span>';
  item.appendChild(final);

  // Action row: "Report" button + (conditional) "✓ reported" badge.
  // The inline form is lazy — built on first click, kept in DOM after
  // submit so subsequent visits see the badge.
  const actions = document.createElement('div');
  actions.className = 'trace-actions';
  const reportBtn = document.createElement('button');
  reportBtn.type = 'button';
  reportBtn.className = 'trace-report-btn';
  reportBtn.textContent = 'Report a problem';
  reportBtn.title = 'For single-word fixes, edit the pipeline rules above. '
                  + 'Use this form for issues larger than a single word.';
  reportBtn.addEventListener('click', () => toggleReportForm(item, entry));
  actions.appendChild(reportBtn);
  const badge = document.createElement('span');
  badge.className = 'trace-reported-badge';
  badge.textContent = '✓ reported';
  badge.style.display = isReported(entry.request_id) ? 'inline-flex' : 'none';
  actions.appendChild(badge);
  item.appendChild(actions);

  return item;
}

// ---------- "Report a problem" inline form -------------------------------
//
// Each trace gets a lazy form: built on the first "Report" click and
// kept in DOM thereafter. The form supports three complementary inputs:
//   (a) per-word corrections — click a word chip in `final` to mark it
//       wrong, then type the replacement. Multiple corrections allowed.
//   (b) "what you meant to say" — free-text rewrite for full rephrases.
//   (c) free-text comment — anything else for the admin.
// At least one of (a-with-non-empty-correct), (b), or (c) must be
// non-empty for the server to accept the submission.

const _REPORTED_KEY = 'whisper-reported-ids';
const _REPORTED_MAX = 200;
// Server-authoritative set, refreshed on every load() of /state.
// Survives across browsers/devices. localStorage is a per-tab UX hint
// that lets a freshly submitted badge show instantly without waiting
// for the next /state poll.
let _serverReportedSet = new Set();
// Per-request_id chip corrections previously submitted by THIS user,
// keyed on request_id. Populated from /state.reported_chips. Lets the
// report form re-render chips on form-open instead of starting empty.
// Filtered server-side to caller's user_id; never contains other
// users' chips.
let _serverReportedChips = {};

function getReportedIds() {
  try { return JSON.parse(localStorage.getItem(_REPORTED_KEY) || '[]'); }
  catch (_) { return []; }
}
function isReported(rid) {
  if (!rid) return false;
  if (_serverReportedSet.has(rid)) return true;
  const ids = getReportedIds();
  return ids.indexOf(rid) !== -1;
}
function markReported(rid) {
  if (!rid) return;
  const ids = getReportedIds();
  if (ids.indexOf(rid) !== -1) return;
  ids.unshift(rid);
  if (ids.length > _REPORTED_MAX) ids.length = _REPORTED_MAX;
  try { localStorage.setItem(_REPORTED_KEY, JSON.stringify(ids)); }
  catch (_) { /* quota — accept that the badge won't survive a reload */ }
}
function unmarkReported(rid) {
  // Called when the user explicitly removes their report. Drops the rid
  // from the localStorage hint AND the in-memory server set so the
  // badge clears immediately, without waiting for the next /state poll.
  if (!rid) return;
  const ids = getReportedIds().filter((x) => x !== rid);
  try { localStorage.setItem(_REPORTED_KEY, JSON.stringify(ids)); }
  catch (_) {}
  _serverReportedSet.delete(rid);
}

function syncReportedBadges() {
  // Called after /state load to flip badges visible/hidden on every
  // trace item currently in the DOM, using the freshly fetched server
  // list. Cheap; the recent panel holds at most 20 items. Also flips
  // the "Remove report" button visibility per open report form.
  document.querySelectorAll('.trace-item').forEach((item) => {
    let rid = '';
    try { rid = (JSON.parse(item.dataset.entry || '{}') || {}).request_id || ''; }
    catch (_) {}
    const reported = isReported(rid);
    const badge = item.querySelector('.trace-reported-badge');
    if (badge) badge.style.display = reported ? 'inline-flex' : 'none';
    const removeBtn = item.querySelector('.rep-remove');
    if (removeBtn) removeBtn.style.display = reported ? 'inline-flex' : 'none';
  });
}

// Tokenization for the clickable-chip layer. We anchor on any word-char
// (`\w` = [A-Za-z0-9_]) plus the Latin-Extended range À-ɏ so numeric
// tokens like "123" and digit-leading mixed tokens ("3D", "5mg") are
// also selectable — matching /captures' "every Whisper token is
// clickable" behaviour. Punctuation runs are emitted as plain-text
// nodes between word spans.
//
// We deliberately diverge from quick_config_state.py:_TOKEN_RE here.
// That server-side regex extracts autocomplete candidates and
// intentionally rejects digit-only tokens (rule candidates shouldn't
// be raw numbers). The clickable-chip layer has a different purpose:
// the user must be able to point at any visible token to correct it.
const _WORD_RE = /[\wÀ-ɏ][\wÀ-ɏ\-]*/g;

function _wordifyFinal(form, finalText) {
  // Tokenize `final` into clickable word spans + track char offsets so a
  // multi-word chip can slice the original string and preserve any
  // inter-word punctuation/whitespace.
  const container = form.querySelector('.rep-final-words');
  container.innerHTML = '';
  form._finalText = finalText || '';
  form._wordPositions = [];   // [{idx, start, end, text}]
  if (!finalText) {
    container.innerHTML = '<span class="empty-recent">(no final text)</span>';
    return;
  }
  let last = 0;
  let wordIdx = 0;
  _WORD_RE.lastIndex = 0;
  let m;
  while ((m = _WORD_RE.exec(finalText)) !== null) {
    if (m.index > last) {
      container.appendChild(
        document.createTextNode(finalText.slice(last, m.index))
      );
    }
    const span = document.createElement('span');
    span.className = 'word';
    span.dataset.idx = String(wordIdx);
    span.textContent = m[0];
    span.addEventListener('click', (e) => {
      toggleCorrection(form, parseInt(span.dataset.idx, 10),
                       span.textContent, !!e.shiftKey);
    });
    container.appendChild(span);
    form._wordPositions.push({
      idx: wordIdx, start: m.index, end: m.index + m[0].length, text: m[0],
    });
    last = m.index + m[0].length;
    wordIdx++;
  }
  if (last < finalText.length) {
    container.appendChild(document.createTextNode(finalText.slice(last)));
  }
}

function _buildReportForm(entry) {
  const form = document.createElement('div');
  form.className = 'trace-report-form';
  form.innerHTML =
    '<div class="help rep-scope-hint">'
    + 'For a single wrong word, prefer editing the pipeline rules above. '
    + 'Use this report when the issue is larger than a single word.'
    + '</div>'
    + '<div class="form-label">Mark wrong words'
    + ' <span class="help">(click a word; shift-click another to extend'
    + ' the range; type the correction; press Enter to add another)</span>'
    + '</div>'
    + '<div class="rep-final-words"></div>'
    + '<div class="rep-corrections"></div>'
    + '<label class="form-label">Comment'
    + ' <span class="help">(optional)</span>'
    + '</label>'
    + '<textarea class="rep-comment" rows="3"'
    + ' placeholder="What went wrong? Anything else the admin should know?"></textarea>'
    + '<div class="rep-actions">'
    + '  <button type="button" class="rep-cancel">Cancel</button>'
    + '  <button type="button" class="rep-remove">Remove report</button>'
    + '  <button type="button" class="rep-submit primary">Submit report</button>'
    + '</div>';
  // Seed chips from the user's previously-submitted report (if any).
  // Deep-clone so user edits don't mutate the shared server snapshot;
  // _serverReportedChips is rewritten on every /state load.
  const _seed = _serverReportedChips[entry.request_id || ''];
  form._corrections = (_seed && Array.isArray(_seed.corrections))
    ? JSON.parse(JSON.stringify(_seed.corrections))
    : [];
  // idx_end is the inclusive range end; single-word chips set
  // idx_end === idx so range-aware code doesn't need to special-case.
  form._entry = entry;
  _wordifyFinal(form, entry.final || '');
  // Render chips + struck-through + inline replacements immediately so
  // the user sees what they previously submitted. Skipped when there
  // are no seeded chips — empty render is a no-op but a needless
  // reflow.
  if (form._corrections.length) {
    _renderCorrections(form);
  }
  form.querySelector('.rep-cancel').addEventListener('click', () => {
    form.classList.remove('open');
  });
  form.querySelector('.rep-submit').addEventListener('click', () => {
    submitReport(form);
  });
  const removeBtn = form.querySelector('.rep-remove');
  // Visible only when the trace has already been reported. Lives in
  // the actions row so its placement matches Cancel/Submit.
  removeBtn.style.display =
    isReported(entry.request_id) ? 'inline-flex' : 'none';
  removeBtn.addEventListener('click', () => {
    removeReport(form);
  });
  return form;
}

function toggleReportForm(item, entry) {
  let form = item.querySelector('.trace-report-form');
  if (!form) {
    form = _buildReportForm(entry);
    item.appendChild(form);
  }
  form.classList.toggle('open');
  if (form.classList.contains('open')) {
    // Focus the first correction input if any chips are present (e.g.
    // rehydrated from a previous submission or freshly added). With
    // no chips yet the user's next action is clicking a word in the
    // strip, so leave focus where it is.
    const firstCorr = form.querySelector('.rep-correct');
    if (firstCorr) firstCorr.focus();
  }
}

// Range-aware chip helpers ------------------------------------------------
//
// A chip spans [chip.idx … chip.idx_end] inclusive. Single-word chips set
// idx_end === idx so every helper can iterate the range uniformly.
function _chipCovers(chip, idx) {
  return chip.idx <= idx && idx <= chip.idx_end;
}

function _selectWord(form, idx, on) {
  const span = form.querySelector(
    '.rep-final-words .word[data-idx="' + idx + '"]'
  );
  if (span) span.classList.toggle('selected', !!on);
}

// Inline replacement next to a struck-through word — mirrors the
// .diff-ins green look from /reports' renderDiff so the user sees
// the correction inline in the words strip, not just in the chip
// panel below.
function _setReplacementInline(form, chip) {
  if (typeof chip.idx !== 'number') return;
  const lastIdx = (typeof chip.idx_end === 'number') ? chip.idx_end : chip.idx;
  const anchor = form.querySelector(
    '.rep-final-words .word[data-idx="' + lastIdx + '"]'
  );
  if (!anchor) return;
  let existing = anchor.nextSibling;
  if (!existing || !existing.classList ||
      !existing.classList.contains('word-replacement')) {
    existing = null;
  }
  const text = (chip.correct || '').trim();
  if (!text) {
    if (existing) existing.parentNode.removeChild(existing);
    return;
  }
  if (!existing) {
    existing = document.createElement('span');
    existing.className = 'word-replacement';
    anchor.parentNode.insertBefore(existing, anchor.nextSibling);
  }
  existing.textContent = text;
}

function _clearReplacementInline(form, chip) {
  if (typeof chip.idx !== 'number') return;
  const lastIdx = (typeof chip.idx_end === 'number') ? chip.idx_end : chip.idx;
  const anchor = form.querySelector(
    '.rep-final-words .word[data-idx="' + lastIdx + '"]'
  );
  if (!anchor) return;
  const nxt = anchor.nextSibling;
  if (nxt && nxt.classList && nxt.classList.contains('word-replacement')) {
    nxt.parentNode.removeChild(nxt);
  }
}

function _spanText(form, a, b) {
  // Slice the original final string so inter-word punctuation/whitespace
  // is preserved (e.g. "A, B C" stays "A, B C", not "A B C").
  const pos = form._wordPositions || [];
  const finalText = form._finalText || '';
  if (!pos[a] || !pos[b]) {
    // Fallback: join textContent of the matching spans.
    const parts = [];
    for (let i = a; i <= b; i++) {
      const s = form.querySelector(
        '.rep-final-words .word[data-idx="' + i + '"]'
      );
      if (s) parts.push(s.textContent);
    }
    return parts.join(' ');
  }
  return finalText.slice(pos[a].start, pos[b].end);
}

function _focusLastInput(form) {
  const inputs = form.querySelectorAll('.rep-correction .rep-correct');
  const last = inputs[inputs.length - 1];
  if (last) last.focus();
}

function _removeChip(form, chipIdx) {
  const chip = form._corrections[chipIdx];
  if (!chip) return;
  for (let j = chip.idx; j <= chip.idx_end; j++) _selectWord(form, j, false);
  _clearReplacementInline(form, chip);
  form._corrections.splice(chipIdx, 1);
  _renderCorrections(form);
}

function _extendLastChip(form, idx) {
  // Grow the most-recent chip to include `idx`, then absorb any earlier
  // chip whose range overlaps the new one. The user-typed `correct`
  // value on the last chip is preserved; absorbed chips lose theirs.
  const lastI = form._corrections.length - 1;
  const last = form._corrections[lastI];
  if (!last) return;
  const newStart = Math.min(last.idx, idx);
  const newEnd   = Math.max(last.idx_end, idx);
  form._corrections = form._corrections.filter((c, i) => {
    if (i === lastI) return true;
    const overlaps = !(c.idx_end < newStart || c.idx > newEnd);
    if (overlaps) {
      for (let j = c.idx; j <= c.idx_end; j++) _selectWord(form, j, false);
    }
    return !overlaps;
  });
  last.idx = newStart;
  last.idx_end = newEnd;
  last.wrong = _spanText(form, newStart, newEnd);
  for (let j = newStart; j <= newEnd; j++) _selectWord(form, j, true);
  _renderCorrections(form);
  _focusLastInput(form);
}

function toggleCorrection(form, idx, word, shiftKey) {
  const last = form._corrections[form._corrections.length - 1];
  if (shiftKey && last) {
    _extendLastChip(form, idx);
    return;
  }
  // Click on a word that's already inside any chip removes that chip.
  // Splitting a multi-word chip in two is intentionally out of scope —
  // simpler to nuke and re-create.
  const existingI = form._corrections.findIndex(c => _chipCovers(c, idx));
  if (existingI >= 0) {
    _removeChip(form, existingI);
    return;
  }
  form._corrections.push({ wrong: word, correct: '', idx: idx, idx_end: idx });
  _selectWord(form, idx, true);
  _renderCorrections(form);
  _focusLastInput(form);
}

function _renderCorrections(form) {
  const box = form.querySelector('.rep-corrections');
  box.innerHTML = '';
  // Refresh inline replacement siblings for all current chips so
  // re-renders keep the strip in sync with the chip-panel state.
  form._corrections.forEach(c => _setReplacementInline(form, c));
  form._corrections.forEach((c, i) => {
    const chip = document.createElement('div');
    chip.className = 'rep-correction';
    chip.dataset.idx = String(c.idx);
    const w = document.createElement('span');
    w.className = 'rep-wrong';
    w.textContent = c.wrong;
    const a = document.createElement('span');
    a.className = 'rep-arrow';
    a.textContent = '→';
    const inp = document.createElement('input');
    inp.type = 'text';
    inp.className = 'rep-correct';
    inp.placeholder = 'correct word';
    inp.value = c.correct;
    inp.addEventListener('input', () => {
      c.correct = inp.value;
      _setReplacementInline(form, c);
    });
    inp.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        // Move focus to the next chip input if any, else to the
        // comment textarea so the user can add context. No more
        // chips + no comment yet = a deliberate done signal; we
        // don't auto-submit here, the user still clicks the button.
        const inputs = Array.from(form.querySelectorAll('.rep-correct'));
        const next = inputs[i + 1];
        if (next) next.focus();
        else {
          const c = form.querySelector('.rep-comment');
          if (c) c.focus();
        }
      }
    });
    const x = document.createElement('button');
    x.type = 'button';
    x.className = 'rep-correction-remove';
    x.textContent = '×';
    x.setAttribute('aria-label', 'remove correction');
    x.addEventListener('click', () => {
      // Find the chip's current index — _renderCorrections rebuilds
      // the list, so the closure-captured `i` may be stale.
      const idx = form._corrections.indexOf(c);
      if (idx >= 0) _removeChip(form, idx);
    });
    chip.appendChild(w);
    chip.appendChild(a);
    chip.appendChild(inp);
    chip.appendChild(x);
    box.appendChild(chip);
  });
}

async function submitReport(form) {
  const entry = form._entry || {};
  const comment = form.querySelector('.rep-comment').value.trim();
  const corrections = form._corrections
    .map(c => {
      const out = {
        wrong: c.wrong,
        correct: (c.correct || '').trim(),
        idx: c.idx,
      };
      if (typeof c.idx_end === 'number' && c.idx_end !== c.idx) {
        out.idx_end = c.idx_end;
      }
      return out;
    })
    .filter(c => c.correct.length > 0);
  if (!corrections.length && !comment) {
    showToast('Mark a wrong word or leave a comment.', 'err');
    return;
  }
  const submitBtn = form.querySelector('.rep-submit');
  submitBtn.disabled = true;
  try {
    const r = await api('POST', '/quick-config/reports/api/submit', {
      trace_ts: entry.ts || 0,
      request_id: entry.request_id || null,
      model: entry.model || '',
      raw: entry.raw || '',
      final: entry.final || '',
      steps: entry.steps || [],
      corrections: corrections,
      user_comment: comment,
    });
    if (!r.ok) {
      let msg = 'Report failed (' + r.status + ')';
      try { const j = await r.json(); if (j && j.detail) msg = j.detail; } catch (_) {}
      showToast(msg, 'err');
      return;
    }
    if (entry.request_id) {
      markReported(entry.request_id);
      _serverReportedSet.add(entry.request_id);
      // Snapshot the user's just-submitted chips so any re-render that
      // calls _buildReportForm(entry) again (e.g. a future SSE-driven
      // refresh) re-seeds correctly. Slight skew if the server merged
      // with pre-existing chips; the next /state load reconciles.
      _serverReportedChips[entry.request_id] = {
        corrections: JSON.parse(JSON.stringify(form._corrections || [])),
      };
    }
    // Update the badge + collapse the form. Keep the form's filled-in
    // values intact so the user can re-open it if they want to add more.
    const item = form.closest('.trace-item');
    if (item) {
      const badge = item.querySelector('.trace-reported-badge');
      if (badge) badge.style.display = 'inline-flex';
    }
    form.classList.remove('open');
    let body = {};
    try { body = await r.json(); } catch (_) {}
    const wasUpdated = !!body.was_updated;
    showToast(wasUpdated ? 'Report updated — thank you.' : 'Report sent — thank you.', 'ok');
  } catch (e) {
    showToast('Report failed: ' + e, 'err');
  } finally {
    submitBtn.disabled = false;
  }
}

async function removeReport(form) {
  // Withdraw the user's report for this trace. Reports are an
  // independent store: removing one does NOT touch any capture's chip
  // corrections (those are managed at /captures via batch-review).
  const entry = form._entry || {};
  const rid = entry.request_id || '';
  if (!rid) {
    showToast('No request_id for this trace — nothing to remove.', 'err');
    return;
  }
  if (!confirm('Remove your report for this trace?')) return;
  const removeBtn = form.querySelector('.rep-remove');
  removeBtn.disabled = true;
  try {
    const r = await api(
      'DELETE',
      '/quick-config/reports/api/by-request/' + encodeURIComponent(rid),
    );
    if (!r.ok) {
      let msg = 'Remove failed (' + r.status + ')';
      try { const j = await r.json(); if (j && j.detail) msg = j.detail; } catch (_) {}
      showToast(msg, 'err');
      return;
    }
    unmarkReported(rid);
    // Drop the seed entry so a fresh _buildReportForm call after this
    // remove starts empty (matches the server-side state). The form
    // we hold open here keeps form._corrections in-memory; that's
    // fine — closing the form below means the next open re-uses this
    // form node, which still has its in-memory chips. The user can
    // re-submit those by re-clicking Submit. They can also click
    // each chip's × to clear them.
    delete _serverReportedChips[rid];
    syncReportedBadges();
    form.classList.remove('open');
    showToast('Report removed.', 'ok');
  } catch (e) {
    showToast('Remove failed: ' + e, 'err');
  } finally {
    removeBtn.disabled = false;
  }
}

function pushTrace(entry) {
  const list = document.getElementById('recent-list');
  if (!list) return;
  // Skip rows already on screen (the SSE replay re-sends the freshest page
  // that the initial fetch already rendered) and, while a search is active,
  // live rows that don't match the filter (so they don't pollute the view).
  if (_seenTrace(entry)) return;
  if (!_matchesSearch(entry)) return;
  const empty = list.querySelector('.empty-recent');
  if (empty) empty.remove();
  list.insertBefore(renderTrace(entry), list.firstChild);
  // Server caps row count via RECENT_TRANSCRIPTIONS_MAX + TTL prune; no
  // client-side trim. The "Load older" cursor walks back through the
  // store from whatever the user has on screen.
  if (_oldestLoadedTs == null || (entry.ts && entry.ts < _oldestLoadedTs)) {
    // First SSE event after a fresh connect seeds the cursor; subsequent
    // pushes are newer than the current bottom and don't move it.
    if (_oldestLoadedTs == null) _oldestLoadedTs = entry.ts || null;
  }
  rebuildDatalist();
}

function appendTraceAtBottom(entry) {
  const list = document.getElementById('recent-list');
  if (!list) return;
  if (_seenTrace(entry)) return;
  // #recent-pager is a SIBLING of #recent-list (it sits below the list as a
  // separate footer bar), not a child — so insertBefore(node, pager) throws
  // NotFoundError. Older entries belong at the end of the list, so just
  // append them there (they land above the pager, which follows the list).
  list.appendChild(renderTrace(entry));
}

// Clear the list and (re)load the freshest page from the durable store for
// the current _searchQuery. Used for the initial population (so the panel no
// longer depends on the SSE replay succeeding), on every search change, and
// to catch up after a stream reconnect. Resets the dedupe set + pagination
// cursor so "Load older" walks back through the (optionally filtered) set.
async function reloadRecent() {
  const list = document.getElementById('recent-list');
  const btn = document.getElementById('btn-load-older');
  if (!list) return;
  _seenReqIds = new Set();
  _oldestLoadedTs = null;
  const q = _searchQuery;
  const url = '/quick-config/recent' + (q ? ('?q=' + encodeURIComponent(q)) : '');
  let j = null;
  try {
    const r = await api('GET', url);
    if (r.ok) j = await r.json();
    else showToast('Recent load failed (' + r.status + ')', 'err');
  } catch (e) {
    showToast('Recent load failed: ' + e, 'err');
  }
  const rows = (j && j.recent) || [];
  // Server returns newest-first; appending in order leaves newest at the top.
  list.innerHTML = '';
  for (const entry of rows) appendTraceAtBottom(entry);
  _oldestLoadedTs = j && j.next_before_ts ? j.next_before_ts : null;
  if (!rows.length) {
    const d = document.createElement('div');
    d.className = 'empty-recent';
    d.textContent = q
      ? 'No transcriptions match your search.'
      : "No transcriptions yet — they'll appear here as you dictate.";
    list.appendChild(d);
  }
  if (btn) btn.style.display = _oldestLoadedTs ? '' : 'none';
  rebuildDatalist();
}

async function loadOlder() {
  if (_loadOlderBusy) return;
  const list = document.getElementById('recent-list');
  const btn = document.getElementById('btn-load-older');
  if (!list || !btn) return;
  if (_oldestLoadedTs == null) {
    // Bootstrap: derive cursor from the oldest .trace-item currently rendered.
    const items = list.querySelectorAll('.trace-item');
    if (items.length) {
      try {
        const data = JSON.parse(items[items.length - 1].dataset.entry || '{}');
        _oldestLoadedTs = data.ts || null;
      } catch (_) {}
    }
  }
  if (_oldestLoadedTs == null) { btn.style.display = 'none'; return; }
  _loadOlderBusy = true;
  btn.disabled = true;
  const prevLabel = btn.textContent;
  btn.textContent = 'Loading…';
  try {
    const r = await api('GET',
      '/quick-config/recent?before_ts=' + encodeURIComponent(_oldestLoadedTs)
      + (_searchQuery ? '&q=' + encodeURIComponent(_searchQuery) : ''));
    if (!r.ok) { showToast('Load older failed (' + r.status + ')', 'err'); return; }
    const j = await r.json();
    const rows = (j && j.recent) || [];
    for (const entry of rows) appendTraceAtBottom(entry);
    _oldestLoadedTs = j && j.next_before_ts ? j.next_before_ts : null;
    if (rows.length) rebuildDatalist();
    btn.style.display = _oldestLoadedTs ? '' : 'none';
  } catch (e) {
    showToast('Load older failed: ' + e, 'err');
  } finally {
    _loadOlderBusy = false;
    btn.disabled = false;
    btn.textContent = prevLabel;
  }
}

function rebuildDatalist() {
  const dl = document.getElementById('recent-words');
  if (!dl) return;
  // Iterate trace items newest → oldest. First-seen wins (= newest casing).
  const seen = new Map();
  document.querySelectorAll('.trace-item').forEach(item => {
    let data;
    try { data = JSON.parse(item.dataset.entry || '{}'); }
    catch (_) { return; }
    for (const t of data.tokens || []) {
      const k = String(t).toLowerCase();
      if (!seen.has(k)) seen.set(k, t);
    }
    for (const b of data.bigrams || []) {
      const k = String(b).toLowerCase();
      if (!seen.has(k)) seen.set(k, b);
    }
  });
  dl.innerHTML = '';
  let n = 0;
  for (const v of seen.values()) {
    const opt = document.createElement('option');
    opt.value = v;
    dl.appendChild(opt);
    if (++n >= _MAX_DATALIST) break;
  }
}

function _setRecentLabel(text) {
  const el = document.querySelector('.recent-header .recent-label');
  if (el) el.textContent = text;
}

function startStream() {
  if (_es) { try { _es.close(); } catch (_) {} _es = null; }
  // EventSource sends the HttpOnly session cookie automatically (same-origin),
  // so the SSE auth dep resolves it without the legacy ?key= query param.
  _es = new EventSource('/quick-config/stream');
  _es.addEventListener('trace', (e) => {
    try { pushTrace(JSON.parse(e.data)); } catch (_) {}
  });
  _es.onopen = () => {
    // A successful (re)connect ends any recovery polling and restores the label.
    if (_recoveryTimer) { clearInterval(_recoveryTimer); _recoveryTimer = null; }
    _setRecentLabel('persistent · paginated');
  };
  _es.onerror = () => {
    // EventSource does NOT auto-reconnect after an HTTP error (e.g. an
    // intermittent 401 where the browser didn't attach the session cookie to
    // the SSE handshake) — it fails the connection permanently. Mirror the
    // /stats recovery: poll a cheap cookie-authed endpoint until it 200s,
    // then catch up via reloadRecent() (deduped) and reopen the stream.
    _setRecentLabel('reconnecting…');
    if (_recoveryTimer) return;
    _recoveryTimer = setInterval(async () => {
      try {
        const r = await api('GET', '/quick-config/recent?limit=1');
        if (r.ok) {
          clearInterval(_recoveryTimer);
          _recoveryTimer = null;
          await reloadRecent();
          startStream();
        }
      } catch (_) { /* keep polling */ }
    }, 3000);
  };
}

// --- Field-level diff helpers (used for both build-patch and conflict re-merge) ---
//
// For each editable field we represent the user's change as a typed diff
// vs. their original baseline (initialRules at load time):
//   dict   (cb:map.map)                    → {kind:'dict', added, changed, removed[]}
//   list   (cb:lowercase-wordlist.wordlist)→ {kind:'list', added[], removed[]}
//   scalar (everything else)               → {kind:'scalar', value}
// On a conflict we re-apply the diff to the FRESH server state, so the
// other user's entries survive — only the cells the user actually
// touched get overlaid.
const _DICT_FIELDS = new Set(['map']);
const _LIST_FIELDS = new Set(['wordlist']);

function _stringify(x) {
  // Stable order for objects so two equal-content objects compare equal.
  if (x !== null && typeof x === 'object' && !Array.isArray(x)) {
    const out = {};
    for (const k of Object.keys(x).sort()) out[k] = x[k];
    return JSON.stringify(out);
  }
  return JSON.stringify(x);
}

function _fieldDiff(baseline, edited, kind) {
  if (kind === 'dict') {
    const base = baseline || {};
    const cur = edited || {};
    const added = {};
    const changed = {};
    const removed = [];
    for (const k of Object.keys(cur)) {
      if (!(k in base)) added[k] = cur[k];
      else if (_stringify(base[k]) !== _stringify(cur[k])) changed[k] = cur[k];
    }
    for (const k of Object.keys(base)) if (!(k in cur)) removed.push(k);
    if (!Object.keys(added).length && !Object.keys(changed).length && !removed.length) {
      return null;
    }
    return { kind: 'dict', added, changed, removed };
  }
  if (kind === 'list') {
    const base = (baseline || []).map(_stringify);
    const baseSet = new Set(base);
    const cur = (edited || []).map(_stringify);
    const curSet = new Set(cur);
    const added = (edited || []).filter(x => !baseSet.has(_stringify(x)));
    const removed = (baseline || []).filter(x => !curSet.has(_stringify(x)));
    if (!added.length && !removed.length) return null;
    return { kind: 'list', added, removed };
  }
  // scalar
  if (_stringify(baseline) === _stringify(edited)) return null;
  return { kind: 'scalar', value: edited };
}

function _applyFieldDiff(serverValue, diff) {
  if (!diff) return serverValue;
  if (diff.kind === 'dict') {
    const out = { ...(serverValue || {}) };
    for (const k of diff.removed) delete out[k];
    Object.assign(out, diff.added, diff.changed);
    return out;
  }
  if (diff.kind === 'list') {
    const removedKeys = new Set(diff.removed.map(_stringify));
    const out = (serverValue || []).filter(x => !removedKeys.has(_stringify(x)));
    for (const item of diff.added) out.push(item);
    return out;
  }
  return diff.value;
}

function _diffKindFor(field) {
  if (_DICT_FIELDS.has(field)) return 'dict';
  if (_LIST_FIELDS.has(field)) return 'list';
  return 'scalar';
}

// Build the rules_patch dict + per-rule fingerprints by diffing live vs initial.
// The fingerprint is the `_fp` field the server stamped on each rule at /state
// load — sending it back lets the server detect that another writer changed
// this rule since we loaded, instead of silently overwriting their edit.
function buildPatchAndFingerprints() {
  const patch = {};
  const fingerprints = {};
  const byName = new Map(initialRules.map(r => [r.name, r]));
  for (const slug of dirty) {
    const live = liveRules.find(r => r.name === slug);
    const orig = byName.get(slug);
    if (!live || !orig) continue;
    const allowed = _ALLOWED_BY_TYPE[live.type] || new Set(['enabled']);
    const delta = {};
    for (const k of allowed) {
      const a = live[k], b = orig[k];
      if (JSON.stringify(a) !== JSON.stringify(b)) {
        delta[k] = a;
      }
    }
    if (Object.keys(delta).length) {
      patch[slug] = delta;
      if (orig._fp) fingerprints[slug] = orig._fp;
    }
  }
  return { patch, fingerprints };
}

const _ALLOWED_BY_TYPE = {
  'regex':                       new Set(['enabled', 'pattern', 'replacement']),
  'callback:map':                new Set(['enabled', 'map']),
  'callback:lowercase-wordlist': new Set(['enabled', 'pattern', 'wordlist']),
  'callback:dedup':              new Set(['enabled', 'pattern']),
  'callback:upper':              new Set(['enabled', 'pattern']),
};

async function load() {
  const r = await ensureToken();
  if (!r) { setStatus('not authenticated'); return; }
  if (!r.ok) {
    setStatus('load failed (' + r.status + ')');
    showToast('load failed: ' + r.status, 'err');
    return;
  }
  const j = await r.json();
  initialRules = j.rules || [];
  liveRules = JSON.parse(JSON.stringify(initialRules));
  dirty = new Set();
  // role: "admin" reveals .admin-only nav links + sev pills. Non-admin keys
  // sessions never get the class so admin chrome stays hidden.
  if (j.role === 'admin') document.body.classList.add('role-admin');
  else document.body.classList.remove('role-admin');
  // Server-authoritative "reported" set + chip-corrections map. The
  // ids feed the badge visibility; the chips feed form rehydration on
  // form-open. Badges in already-rendered .trace-item nodes update
  // in-place via syncReportedBadges; chip rehydration is lazy (only
  // applied when the user opens a report form).
  _serverReportedChips = j.reported_chips || {};
  _serverReportedSet = new Set(Object.keys(_serverReportedChips));
  syncReportedBadges();
  renderCards();
  updateButtons();
  loadMyUsage();   // fire-and-forget; populates the subbar self-usage banner
}

// --- personal self-usage banner (subbar middle) ---------------------------
function _uCount(n) {
  n = Number(n || 0);
  if (n >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, '') + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'k';
  return String(Math.round(n));
}
function _uDur(sec) {
  sec = Number(sec || 0);
  if (sec < 60) return sec.toFixed(sec < 10 ? 1 : 0) + 's';
  if (sec < 3600) return (sec / 60).toFixed(1).replace(/\.0$/, '') + 'm';
  if (sec < 86400) return (sec / 3600).toFixed(1).replace(/\.0$/, '') + 'h';
  return (sec / 86400).toFixed(1).replace(/\.0$/, '') + 'd';
}
function _uEsc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
async function loadMyUsage() {
  const el = document.getElementById('qc-usage');
  if (!el) return;
  try {
    // Pass this browser's local midnight (epoch seconds) so "today" resets at
    // the viewer's own midnight, not the server's.
    var _mid = new Date(); _mid.setHours(0, 0, 0, 0);
    var tzMidnight = Math.floor(_mid.getTime() / 1000);
    const r = await api('GET', '/quick-config/usage?tz_midnight=' + tzMidnight);
    if (!r.ok) { el.innerHTML = ''; return; }
    const j = await r.json();
    const name = '<span class="u-name">' + _uEsc(j.username || 'there') + '</span>';
    const today = j.today || {}, total = j.total || {};
    if (!total.requests) {
      el.innerHTML = 'Hi ' + name + '<span class="u-sep">·</span>no transcriptions yet';
      return;
    }
    const seg = (cls, n) =>
      '<span class="u-num ' + cls + '">' + _uCount(n.words) + '</span> words '
      + '<span class="u-num ' + cls + '">' + _uDur(n.audio_s) + '</span> audio';
    el.innerHTML = 'Hi ' + name
      + '<span class="u-sep">·</span>today ' + seg('u-today', today)
      + '<span class="u-sep">·</span>total ' + seg('u-total', total);
  } catch (_) {
    el.innerHTML = '';
  }
}

async function doSave() {
  const { patch, fingerprints } = buildPatchAndFingerprints();
  if (!Object.keys(patch).length) return;
  setStatus('saving…');
  const r = await api('POST', '/quick-config/state',
                      { rules_patch: patch, fingerprints });
  if (r.status === 422) {
    // Validation error — likely involves a rule the user can't see (admin
    // pipeline has bad regex etc.). Surface a generic message rather than
    // confusing field-bound errors pointing at hidden rules.
    showToast('admin pipeline has a validation error — please contact admin', 'err');
    setStatus('save failed');
    return;
  }
  if (!r.ok) {
    let msg = 'save failed (' + r.status + ')';
    try { const j = await r.json(); if (j && j.detail) msg = j.detail; } catch (_) {}
    showToast(msg, 'err');
    setStatus('save failed');
    return;
  }
  const result = await r.json();
  const conflicts = result.conflicts || [];
  const saved = result.saved || [];
  if (conflicts.length) {
    // Another writer changed one or more rules between our load and save.
    // We preserve the user's intent by computing their per-field DIFF
    // against the baseline they saw at load time, refetching the fresh
    // server state, and re-applying that diff on top. For cb:map this
    // means only the keys the user added/changed/removed get overlaid —
    // the other user's keys survive. For scalars (regex pattern,
    // replacement, enabled) the user's new value still wins (no way to
    // merge two scalar edits), but the user reviews before re-saving.
    const conflictSlugs = conflicts.map(c => c.slug);

    // Snapshot diffs BEFORE await load() — load() resets initialRules.
    const diffsBySlug = new Map();
    for (const slug of conflictSlugs) {
      const live = liveRules.find(r => r.name === slug);
      const orig = initialRules.find(r => r.name === slug);
      if (!live || !orig) continue;
      const allowed = _ALLOWED_BY_TYPE[live.type] || new Set(['enabled']);
      const fieldDiffs = {};
      for (const k of allowed) {
        const d = _fieldDiff(orig[k], live[k], _diffKindFor(k));
        if (d) fieldDiffs[k] = d;
      }
      if (Object.keys(fieldDiffs).length) diffsBySlug.set(slug, fieldDiffs);
    }

    await load();   // refetches; clears dirty; renders fresh server state

    // Re-apply each user diff onto the fresh server state. dirty gets
    // re-populated so the next Save sends an updated patch with the
    // FRESH fingerprint (which now matches the server) plus the
    // user's intent overlaid.
    let merged = 0;
    for (const [slug, fieldDiffs] of diffsBySlug) {
      const idx = liveRules.findIndex(r => r.name === slug);
      if (idx < 0) continue;  // rule was deleted / un-exposed server-side
      for (const [field, diff] of Object.entries(fieldDiffs)) {
        liveRules[idx][field] = _applyFieldDiff(liveRules[idx][field], diff);
      }
      dirty.add(slug);
      merged++;
    }

    if (merged) {
      renderCards();
      updateButtons();
      const prefix = saved.length ? 'Saved ' + saved.length + '. ' : '';
      showToast(prefix + 'Auto-merged your changes for '
                + conflictSlugs.join(', ')
                + ' on top of the latest version — review and Save again.',
                'err');
    } else {
      // The conflicted rule was deleted server-side or no diff survived.
      const prefix = saved.length ? 'Saved ' + saved.length + '. ' : '';
      showToast(prefix + 'Conflict on ' + conflictSlugs.join(', ')
                + ' could not be auto-merged (rule changed or removed).',
                'err');
    }
    return;
  }
  showToast(saved.length
    ? ('saved ' + saved.length + ' rule' + (saved.length === 1 ? '' : 's'))
    : 'nothing to save', 'ok');
  if (result.captures_count > 0) {
    // Auto-trigger the reapply job silently — admins shouldn't have to
    // click twice. The header strip shows progress; there's no manual
    // re-apply button anymore.
    startReapplyJobSilent(result.captures_count);
  }
  await load();
}


// --- Silent reapply strip (auto-triggered after Save) ---
//
// Same /reapply-rules POST + status-polling as the manual modal, but
// rendered as an inline header strip so the admin keeps editing.
// Fades out 3s after completion; sticks (red-tinted) on error so the
// failure isn't missed.
let _stripPoll = null;
let _stripFadeTimer = null;

function _showStrip(n) {
  const strip = document.getElementById('reapply-strip');
  strip.hidden = false;
  strip.classList.remove('fade-out');
  strip.classList.remove('err');
  strip.querySelector('.r-label').textContent =
    'Re-applying to ' + (n || '?') + ' captures…';
  strip.querySelector('.r-fill').style.width = '0%';
  strip.querySelector('.r-stats').textContent = '0 / ' + (n || 0);
  if (_stripFadeTimer) { clearTimeout(_stripFadeTimer); _stripFadeTimer = null; }
}
function _hideStripSoon() {
  const strip = document.getElementById('reapply-strip');
  strip.classList.add('fade-out');
  if (_stripFadeTimer) clearTimeout(_stripFadeTimer);
  _stripFadeTimer = setTimeout(() => {
    strip.hidden = true;
    strip.classList.remove('fade-out');
  }, 3000);
}
async function _pollStripOnce() {
  let r;
  try { r = await api('GET', '/quick-config/reapply-rules/status'); }
  catch (_) { return null; }
  if (!r.ok) return null;
  const s = await r.json();
  const strip = document.getElementById('reapply-strip');
  const fill = strip.querySelector('.r-fill');
  const stats = strip.querySelector('.r-stats');
  const label = strip.querySelector('.r-label');
  const pct = s.total > 0
    ? Math.round((s.processed / s.total) * 100)
    : (s.status === 'done' ? 100 : 0);
  fill.style.width = pct + '%';
  stats.textContent = (s.processed || 0) + ' / ' + (s.total || 0)
    + ' · ' + (s.captures_updated || 0) + ' updated · '
    + (s.groups_updated || 0) + ' groups';
  if (s.status === 'running') {
    label.textContent = 'Re-applying rules…';
  } else if (s.status === 'done') {
    label.textContent = 'Re-applied ' + (s.captures_updated || 0)
      + ' captures · ' + (s.groups_updated || 0) + ' groups';
    if (_stripPoll) { clearInterval(_stripPoll); _stripPoll = null; }
    _hideStripSoon();
  } else if (s.status === 'error') {
    label.textContent = 'Re-apply failed: ' + (s.error || 'unknown');
    strip.classList.add('err');
    if (_stripPoll) { clearInterval(_stripPoll); _stripPoll = null; }
    // Don't auto-hide on error — the user needs to see it.
  }
  return s.status || null;
}
async function startReapplyJobSilent(n) {
  _showStrip(n);
  const r = await api('POST', '/quick-config/reapply-rules');
  if (!r.ok) {
    const strip = document.getElementById('reapply-strip');
    strip.classList.add('err');
    strip.querySelector('.r-label').textContent =
      'Re-apply failed: HTTP ' + r.status;
    return;
  }
  // Reset any prior interval before scheduling — guards against a double
  // click triggering two concurrent polls.
  if (_stripPoll) { clearInterval(_stripPoll); _stripPoll = null; }
  const status = await _pollStripOnce();
  // Only schedule the recurring poll while the job is actually running;
  // status==='done'|'error' already cleared the strip via _pollStripOnce.
  if (status === 'running') {
    _stripPoll = setInterval(_pollStripOnce, 1500);
  }
}

function doDiscard() {
  liveRules = JSON.parse(JSON.stringify(initialRules));
  dirty = new Set();
  renderCards();
  updateButtons();
}

document.getElementById('save-btn').addEventListener('click', doSave);
document.getElementById('discard-btn').addEventListener('click', doDiscard);
const _loadOlderBtn = document.getElementById('btn-load-older');
if (_loadOlderBtn) _loadOlderBtn.addEventListener('click', loadOlder);

// Recent-transcriptions search: debounced server-side substring filter over
// raw/final. Mirrors the /captures search (150 ms debounce); whole-DB so
// matches are found regardless of how far the list has been paginated.
const _recentSearchInput = document.getElementById('recent-search');
let _recentSearchTimer = null;
if (_recentSearchInput) {
  _recentSearchInput.addEventListener('input', () => {
    if (_recentSearchTimer) clearTimeout(_recentSearchTimer);
    _recentSearchTimer = setTimeout(() => {
      _recentSearchTimer = null;
      const next = (_recentSearchInput.value || '').trim();
      if (next === _searchQuery) return;
      _searchQuery = next;
      reloadRecent();
    }, 150);
  });
}

load().then(() => {
  // Populate the recent list via a plain fetch FIRST so the panel no longer
  // depends on the SSE replay succeeding (a single failed /stream handshake
  // used to leave it empty). reloadRecent() also seeds the dedupe set + the
  // "Load older" cursor before the stream opens, so the on-connect replay is
  // skipped as already-seen rather than duplicated.
  return reloadRecent();
}).then(() => {
  startStream();
});
})();
</script>

{{SCALE_PICKER_JS}}
{{SEV_POLLER_JS}}
{{TIME_HELPERS_JS}}

<script>
// Must run AFTER TIME_HELPERS_JS defines timeTick. Default [data-ts] selector
// refreshes both the cb:map date cells created by later renderCards() calls and
// the Recent-transcriptions .trace-ts nodes; all carry static fmtWhen() text at
// creation, this just ages the relative suffix in place.
timeTick();
</script>

</body>
</html>
"""
