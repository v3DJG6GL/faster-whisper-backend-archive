"""
End-user simple-config WebUI for faster-whisper-backend.

Mounted at /quick-config. Endpoints:

  GET  /quick-config              HTML page (loopback / ADMIN_ALLOWED_HOSTS)
  GET  /quick-config/state        Returns ONLY the rules an admin marked exposed
  POST /quick-config/state        Patch enabled / body fields on exposed rules
  GET  /quick-config/recent       Snapshot of the recent-traces ring buffer
  GET  /quick-config/stream       SSE stream of recent traces (live updates)
  POST /quick-config/recent/clear Wipe the recent-traces buffer

Security model:
  1. IP gate:           require_admin_host (loopback always permitted)
  2. Bearer token:      USER_TOKEN OR ADMIN_TOKEN accepted (TokenWithGrace).
                        Header preferred; ?token=<...> query string also
                        accepted because EventSource has no Authorization
                        header support.
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
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ValidationError

import config as cfg
import config_store
import quick_config_state
import web_common
from admin_routes import (
    ADMIN_TOKEN_GUARD,
    USER_TOKEN_GUARD,
    _apply_hot_changes,
    _canon_rules,
    require_admin_host,
)

logger = logging.getLogger("whisper-api")

router = APIRouter(prefix="/quick-config")
_bearer = HTTPBearer(auto_error=False)


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
    # terminal: NEVER patchable from /quick-config (defense in depth — the
    # admin UI already hides the expose checkbox for terminal rules).
    "terminal":                    frozenset(),
}


def require_user_or_admin_token(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Literal["admin", "user"]:
    """Bearer-gate the /quick-config endpoints. Accepts either USER_TOKEN
    or ADMIN_TOKEN. If both tokens are unset, the loopback check alone is
    the gate (returns "admin" for telemetry purposes).

    Header `Authorization: Bearer <token>` is preferred. Query parameter
    `?token=<token>` is also accepted because the browser's EventSource
    API has no way to attach custom headers — the SSE stream endpoint
    needs a query-string fallback.

    Returns the matched role for logging — NOT for authorization. The patch
    endpoint enforces `exposed == True` uniformly regardless of role."""
    admin_set = ADMIN_TOKEN_GUARD.is_set()
    user_set = USER_TOKEN_GUARD.is_set()
    if not admin_set and not user_set:
        # No token configured anywhere → loopback-only access. Treat as admin
        # for the role label since loopback callers are trusted.
        return "admin"
    presented: "str | None" = None
    if creds is not None and creds.scheme.lower() == "bearer":
        presented = creds.credentials
    elif "token" in request.query_params:
        presented = request.query_params.get("token")
    if not presented:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bearer token required")
    if admin_set and ADMIN_TOKEN_GUARD.matches(presented):
        return "admin"
    if user_set and USER_TOKEN_GUARD.matches(presented):
        return "user"
    logger.warning("[quick-config] rejected bad bearer token")
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")


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


@router.get("", dependencies=[Depends(require_admin_host)])
async def get_quick_config_page() -> HTMLResponse:
    return HTMLResponse(
        web_common.render_page(_QUICK_CONFIG_HTML, current="quick-config"),
        media_type="text/html",
    )


@router.get(
    "/state",
    dependencies=[Depends(require_admin_host), Depends(require_user_or_admin_token)],
)
async def get_state(
    role: str = Depends(require_user_or_admin_token),
) -> dict[str, Any]:
    """Return ONLY the rules currently flagged exposed=True. The terminal
    rule is filtered out unconditionally (admin UI already hides its expose
    toggle, but enforce here too as defense in depth)."""
    exposed: list[dict[str, Any]] = []
    for r in cfg.PIPELINE_RULES:
        # cfg.PIPELINE_RULES holds raw dicts (post-coercion) — getattr won't
        # work on plain dicts, so handle both shapes.
        if isinstance(r, dict):
            rd = r
        else:
            rd = r.model_dump() if hasattr(r, "model_dump") else dict(r)
        if rd.get("type") == "terminal":
            continue
        if rd.get("exposed"):
            exposed.append(rd)
    canonical = _canon_rules(exposed)
    # Tag each rule with a fingerprint of its canonical form. Client
    # echoes back per-rule on save; server uses it to detect concurrent
    # edits. Fingerprint is computed AFTER _canon_rules so client and
    # server hash the same canonical bytes.
    for rd in canonical:
        rd["_fp"] = _rule_fingerprint(rd)
    return {
        "rules": canonical,
        "token_required": bool(
            getattr(cfg, "USER_TOKEN", None) or getattr(cfg, "ADMIN_TOKEN", None)
        ),
        "service_name": "WhisperAPI",
        "role": role,
    }


@router.post(
    "/state",
    dependencies=[Depends(require_admin_host), Depends(require_user_or_admin_token)],
)
async def post_state(
    payload: QuickPatchPayload,
    request: Request,
    role: str = Depends(require_user_or_admin_token),
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
    rules_patch = payload.rules_patch or {}
    fingerprints = payload.fingerprints or {}
    if not rules_patch:
        return JSONResponse(
            {"saved": [], "conflicts": [], "requires_restart": False}
        )

    # Snapshot the current PIPELINE_RULES as plain dicts so we can overlay
    # patches deterministically. Keep order; the merged list will replace
    # cfg.PIPELINE_RULES verbatim (count + order preserved → terminal rule
    # stays last).
    current_rules: list[dict[str, Any]] = []
    for r in cfg.PIPELINE_RULES:
        if isinstance(r, dict):
            current_rules.append(dict(r))
        elif hasattr(r, "model_dump"):
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
        if not target.get("exposed"):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"rule '{slug}' is not exposed for end-user editing",
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
        prev_admin_token = getattr(cfg, "ADMIN_TOKEN", None)
        prev_user_token = getattr(cfg, "USER_TOKEN", None)
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

    applied = await _apply_hot_changes(
        written,
        prev_admin_token=prev_admin_token,
        prev_user_token=prev_user_token,
    )

    client_host = request.client.host if request.client else "?"
    logger.info(
        "[quick-config] update from=%s role=%s saved=%s conflicts=%s",
        client_host, role, saved, [c["slug"] for c in conflicts],
    )

    return JSONResponse({
        "saved": saved,
        "conflicts": conflicts,
        **applied,
        "requires_restart": bool(applied["cold_pending"]),
    })


# --- Recent transcription traces (panel + autocomplete source) -------------
#
# The ring buffer lives in quick_config_state. main.py's transcribe handler
# appends an entry per completed transcription. Both endpoints below are
# token-gated so end-users without a valid token can't enumerate recent
# patient dictation snippets.

@router.get(
    "/recent",
    dependencies=[Depends(require_admin_host), Depends(require_user_or_admin_token)],
)
async def get_recent() -> dict[str, Any]:
    """Snapshot of the recent-traces buffer. The /stream endpoint replays
    the snapshot on connect, so most clients won't need /recent — it
    exists as a cheap polling fallback for environments without
    EventSource."""
    return {"recent": list(quick_config_state.recent_traces)}


@router.get(
    "/stream",
    dependencies=[Depends(require_admin_host), Depends(require_user_or_admin_token)],
)
async def stream_recent(request: Request) -> StreamingResponse:
    """Server-sent events stream of recent transcriptions.

    On connect, replays the current buffer (`event: trace` for each entry,
    oldest first). After the replay, pushes any new transcription as
    another `event: trace`. Sends an empty `event: clear` when the admin
    wipes the buffer. Sends a `: keepalive` SSE comment line every 15 s
    so reverse proxies don't kill an idle connection."""
    async def gen():
        q = quick_config_state.subscribe()
        try:
            for entry in list(quick_config_state.recent_traces):
                yield f"event: trace\ndata: {json.dumps(entry)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=15.0)
                    ev = item.get("event", "trace")
                    payload = item.get("data") or {}
                    yield f"event: {ev}\ndata: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            quick_config_state.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post(
    "/recent/clear",
    dependencies=[Depends(require_admin_host), Depends(require_user_or_admin_token)],
)
async def clear_recent(role: str = Depends(require_user_or_admin_token)) -> JSONResponse:
    """Manual kill-switch for the recent-traces buffer. Broadcasts a
    `clear` event so all open /quick-config tabs flush their UI."""
    quick_config_state.clear()
    logger.info("[quick-config] recent-traces buffer cleared by role=%s", role)
    return JSONResponse({"ok": True})


_QUICK_CONFIG_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Quick config — WhisperAPI</title>
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
  header { position: sticky; top: 0; background: var(--bg); z-index: 5;
    border-bottom: 1px solid var(--border); padding: 0.5rem 1rem;
    display: block; }
  header > .header-inner { display: flex; align-items: center;
    gap: 0.625rem; flex-wrap: wrap; }
  header .title { font-size: var(--fs-lg); font-weight: 600; color: var(--bold); }
  header .spacer { flex: 1; }
  header button { background: var(--panel); border: 1px solid var(--border);
    color: var(--fg); padding: 0.25rem 0.625rem; border-radius: 4px;
    cursor: pointer; font: inherit; font-size: var(--fs-sm);
    flex-shrink: 0; white-space: nowrap; }
  header button:disabled { opacity: 0.4; cursor: not-allowed; }
  header button.primary { color: var(--green); border-color: var(--green); }
  header button.danger { color: var(--red); }
  header #status { color: var(--dim); font-size: var(--fs-sm);
    flex-shrink: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
  main { padding: 1rem; max-width: 60rem; margin: 0 auto; }
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
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 0.25rem 0.4rem; font: inherit; font-family: var(--font-mono);
    font-size: var(--fs-sm); }
  .card .rule-editor textarea { width: 100%; resize: vertical; }
  .card .rule-editor .map-table { width: 100%; }
  .card .rule-editor .map-table input { width: 100%; }
  .card .rule-editor button { background: var(--panel);
    border: 1px solid var(--border); color: var(--fg);
    padding: 0.125rem 0.5rem; border-radius: 3px; cursor: pointer;
    font: inherit; font-size: var(--fs-sm); }
  .empty-state { text-align: center; color: var(--dim);
    padding: 4rem 1rem; font-size: var(--fs-md); }
  .empty-state h2 { color: var(--fg); font-size: var(--fs-lg);
    margin: 0 0 0.5rem 0; }
  #toast { position: fixed; bottom: 1rem; right: 1rem; padding: 0.5rem 1rem;
    background: var(--panel); border: 1px solid var(--border); border-radius: 4px;
    color: var(--fg); font-size: var(--fs-sm); display: none; z-index: 10; }
  #toast.err { border-color: var(--red); color: var(--red); }
  #toast.ok { border-color: var(--green); color: var(--green); }
  #token-modal { position: fixed; inset: 0; background: rgba(0,0,0,0.6);
    display: none; align-items: center; justify-content: center; z-index: 8; }
  #token-modal.show { display: flex; }
  #token-modal .box { background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.25rem; min-width: 18rem; }
  #token-modal h3 { margin: 0 0 0.5rem 0; }
  #token-modal input { width: 100%; padding: 0.4rem; background: var(--input-bg);
    color: var(--fg); border: 1px solid var(--border); border-radius: 3px;
    font: inherit; }
  #token-modal .actions { display: flex; gap: 0.5rem; margin-top: 0.75rem;
    justify-content: flex-end; }

  /* Recent transcriptions panel */
  #recent-panel { margin-top: 1.5rem; border-top: 1px solid var(--border);
    padding-top: 1rem; }
  .recent-header { display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 0.5rem; flex-wrap: wrap; }
  .recent-header h2 { font-size: var(--fs-lg); margin: 0; color: var(--bold); }
  .recent-header .recent-label { font-size: var(--fs-xs); color: var(--dim);
    font-style: italic; }
  .recent-header .spacer { flex: 1; }
  .empty-recent { color: var(--dim); font-style: italic; padding: 1rem 0;
    font-size: var(--fs-sm); }
  .trace-item { background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem; }
  .trace-meta { display: flex; gap: 0.5rem; color: var(--dim);
    font-size: var(--fs-xs); margin-bottom: 0.375rem; }
  .trace-text { font-family: var(--font-mono); font-size: var(--fs-sm);
    word-wrap: break-word; }
  .trace-raw { color: var(--dim); margin-bottom: 0.25rem; }
  .trace-tag { display: inline-block; min-width: 3rem; color: var(--dim);
    font-family: var(--font-sans); font-size: var(--fs-xs);
    margin-right: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .trace-final { color: var(--bold); }
  .trace-final .trace-tag { color: var(--green); }
  details.trace-steps { margin-top: 0.375rem; }
  details.trace-steps > summary { cursor: pointer; font-size: var(--fs-xs);
    color: var(--cyan); list-style: revert; user-select: none; }
  .trace-step { padding: 0.25rem 0 0.25rem 0.75rem; font-size: var(--fs-xs);
    border-left: 2px solid var(--border); margin-top: 0.25rem; }
  .trace-step.skipped { opacity: 0.55; }
  .trace-step .step-label { color: var(--dim); display: block;
    font-family: var(--font-sans); margin-bottom: 0.125rem; }
  .trace-step .step-label .skipped-tag { color: var(--yellow);
    margin-left: 0.25rem; }
  .trace-step .step-before, .trace-step .step-after {
    font-family: var(--font-mono); display: block; word-wrap: break-word; }
  .trace-step .step-before { color: var(--dim); }
  .trace-step .step-after { color: var(--fg); }
  .trace-step .step-arrow { color: var(--green); margin-right: 0.25rem; }

  {{NAV_CSS}}
</style>
</head>
<body>

<header>
  <div class="header-inner">
    <span class="title">Quick config</span>
    {{NAV}}
    <span class="spacer"></span>
    {{SCALE_PICKER}}
    <button id="discard-btn" disabled>discard</button>
    <button id="save-btn" class="primary" disabled>save</button>
    <span id="status">loading…</span>
  </div>
</header>

<datalist id="recent-words"></datalist>

<main>
  <section id="cards"></section>
  <section id="recent-panel">
    <div class="recent-header">
      <h2>Recent transcriptions</h2>
      <span class="recent-label">in-memory only, cleared on page reload</span>
      <span class="spacer"></span>
      <button id="clear-recent">clear recent</button>
    </div>
    <div id="recent-list">
      <div class="empty-recent">No transcriptions yet — they'll appear here as you dictate.</div>
    </div>
  </section>
</main>

<div id="toast"></div>

<div id="token-modal">
  <div class="box">
    <h3>Bearer token</h3>
    <p style="color:var(--dim);font-size:var(--fs-sm);margin:0 0 0.5rem 0;">
      Your admin gave you a USER_TOKEN to access this page.
    </p>
    <input id="token-input" type="password" autocomplete="off" spellcheck="false">
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

const TOKEN_KEY = 'whisper_user_token';
let initialRules = [];      // last-loaded rules from server (deep-copy snapshot)
let liveRules = [];         // editable rules — diffed against initialRules to build patch
let dirty = new Set();      // slugs with changes

function getToken() { return sessionStorage.getItem(TOKEN_KEY) || ''; }
function setToken(t) {
  if (t) sessionStorage.setItem(TOKEN_KEY, t);
  else sessionStorage.removeItem(TOKEN_KEY);
}

async function api(method, path, body) {
  const h = { 'Accept': 'application/json' };
  const t = getToken();
  if (t) h['Authorization'] = 'Bearer ' + t;
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
  // Probe /quick-config/state. On 401 prompt; retry once.
  let r = await api('GET', '/quick-config/state');
  if (r.status === 401) {
    const t = await promptToken();
    if (!t) return null;
    setToken(t);
    r = await api('GET', '/quick-config/state');
    if (r.status === 401) {
      setToken('');
      showToast('invalid token', 'err');
      return null;
    }
  }
  return r;
}

function commitData(slug) {
  dirty.add(slug);
  updateButtons();
}
function updateButtons() {
  const has = dirty.size > 0;
  document.getElementById('save-btn').disabled = !has;
  document.getElementById('discard-btn').disabled = !has;
  setStatus(has ? (dirty.size + ' rule' + (dirty.size === 1 ? '' : 's') + ' modified')
                : (initialRules.length + ' rule' + (initialRules.length === 1 ? '' : 's') + ' loaded'));
}

function renderCards() {
  const root = document.getElementById('cards');
  root.innerHTML = '';
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

    const editor = renderTypeEditor(rule, () => commitData(rule.name),
                                    { datalistId: 'recent-words' });
    card.appendChild(editor);

    root.appendChild(card);
  }
}

// ---------- Recent transcriptions panel + autocomplete -------------------
//
// SSE-driven. /quick-config/stream replays the buffer on connect (oldest
// first) then pushes new traces as `event: trace`. A `clear` event wipes
// the panel + datalist. EventSource auto-reconnects with a 3 s backoff
// per the WHATWG spec — no custom retry logic needed.
const _BUFFER_MAX = 20;
const _MAX_DATALIST = 200;
let _es = null;

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s == null ? '' : String(s);
  return div.innerHTML;
}
function relTime(ts) {
  const sec = Math.max(0, (Date.now() / 1000) - ts);
  if (sec < 5) return 'just now';
  if (sec < 60) return Math.floor(sec) + 's ago';
  if (sec < 3600) return Math.floor(sec / 60) + ' min ago';
  return Math.floor(sec / 3600) + ' h ago';
}

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
  ts.textContent = relTime(entry.ts || 0);
  meta.appendChild(ts);
  if (entry.model) {
    const mdl = document.createElement('span');
    mdl.textContent = entry.model;
    meta.appendChild(mdl);
  }
  item.appendChild(meta);

  const raw = document.createElement('div');
  raw.className = 'trace-text trace-raw';
  raw.innerHTML = '<span class="trace-tag">raw</span>'
    + escapeHtml(entry.raw || '');
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
        + '<span class="step-before">' + escapeHtml(before || '') + '</span>'
        + '<span class="step-after"><span class="step-arrow">→</span>'
        + escapeHtml(after || '') + '</span>';
      det.appendChild(stepEl);
    }
    item.appendChild(det);
  }

  const final = document.createElement('div');
  final.className = 'trace-text trace-final';
  final.innerHTML = '<span class="trace-tag">final</span>'
    + escapeHtml(entry.final || '');
  item.appendChild(final);

  return item;
}

function pushTrace(entry) {
  const list = document.getElementById('recent-list');
  if (!list) return;
  const empty = list.querySelector('.empty-recent');
  if (empty) empty.remove();
  list.insertBefore(renderTrace(entry), list.firstChild);
  while (list.children.length > _BUFFER_MAX) {
    list.removeChild(list.lastChild);
  }
  rebuildDatalist();
}

function showEmptyRecent() {
  const list = document.getElementById('recent-list');
  if (!list) return;
  list.innerHTML = '<div class="empty-recent">No transcriptions yet '
    + "&mdash; they'll appear here as you dictate.</div>";
}

function clearTraces() {
  showEmptyRecent();
  rebuildDatalist();
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

function startStream() {
  if (_es) { try { _es.close(); } catch (_) {} _es = null; }
  // EventSource has no Authorization-header support; the server's auth
  // dep accepts ?token=<...> as a fallback specifically for SSE.
  const tok = getToken();
  const url = '/quick-config/stream'
    + (tok ? '?token=' + encodeURIComponent(tok) : '');
  _es = new EventSource(url);
  _es.addEventListener('trace', (e) => {
    try { pushTrace(JSON.parse(e.data)); } catch (_) {}
  });
  _es.addEventListener('clear', () => clearTraces());
  _es.onerror = () => {
    // EventSource auto-reconnects; nothing to do here. If the token is
    // truly bad the server will keep returning 401 and EventSource will
    // give up after a few retries — at which point the page is stale
    // until the user reloads. Acceptable.
  };
}

async function doClearRecent() {
  if (!confirm('Clear all recent transcriptions from memory?')) return;
  try {
    const r = await api('POST', '/quick-config/recent/clear', {});
    if (!r.ok) {
      showToast('clear failed (' + r.status + ')', 'err');
      return;
    }
    // Server broadcasts a 'clear' SSE event; our own listener flushes
    // the panel. Don't double-clear locally — that'd race the broadcast.
  } catch (e) {
    showToast('clear failed: ' + e, 'err');
  }
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
  // role: "admin" reveals .admin-only nav links + sev pills. USER_TOKEN
  // sessions never get the class so admin chrome stays hidden.
  if (j.role === 'admin') document.body.classList.add('role-admin');
  else document.body.classList.remove('role-admin');
  renderCards();
  updateButtons();
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
    // The server applied the non-conflicting patches and rejected the
    // conflicted ones. We DON'T auto-replay the conflicted edits onto the
    // fresh state because dict/list fields (cb:map's map, wordlists)
    // would overwrite the other user's entries again — that's the
    // lost-update bug we're trying to prevent. Instead: discard the
    // conflicted edits, refetch, and ask the user to re-do them on top
    // of the new state so they consciously merge with what's there.
    const conflictSlugs = conflicts.map(c => c.slug);
    const msg = (saved.length
        ? 'Saved ' + saved.length + '. '
        : '')
      + 'Someone else just changed: ' + conflictSlugs.join(', ')
      + '. Your edit was discarded; the page now shows their version — '
      + 'please review and re-apply if still needed.';
    showToast(msg, 'err');
    await load();
    return;
  }
  showToast(saved.length
    ? ('saved ' + saved.length + ' rule' + (saved.length === 1 ? '' : 's'))
    : 'nothing to save', 'ok');
  await load();
}

function doDiscard() {
  liveRules = JSON.parse(JSON.stringify(initialRules));
  dirty = new Set();
  renderCards();
  updateButtons();
}

document.getElementById('save-btn').addEventListener('click', doSave);
document.getElementById('discard-btn').addEventListener('click', doDiscard);
document.getElementById('clear-recent').addEventListener('click', doClearRecent);

load().then(() => {
  // Only open the SSE stream after the rules state has loaded — avoids
  // racing the token prompt with the EventSource connection.
  startStream();
});
})();
</script>

{{SCALE_PICKER_JS}}
{{SEV_POLLER_JS}}

</body>
</html>
"""
