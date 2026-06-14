"""
Admin API for the layered per-identity config-override feature.

Serves the dedicated /settings/overrides page's JSON contract:
  GET  /settings/overrides/state    profiles + field metadata + groups + rules + usage
  POST /settings/overrides/state    save the OVERRIDE_PROFILES dirty diff (hot-applied)
  GET  /settings/overrides/resolve  the effective-config WATERFALL for a user/key/model
                                    (drives the Explorer + the in-context preview)

The HTML page + nav wiring live in this module too once the UI lands (phase 6);
for now this is the backend the binding editors and the Explorer call. All
endpoints are admin-only (host allowlist + admin key).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError

import api_keys_store
import config as cfg
import config_store
import effective_config
import web_common
from auth import require_admin

logger = logging.getLogger("whisper.overrides")

require_admin_webui_host = web_common.require_admin_webui_host

router = APIRouter(prefix="/settings/overrides")

_TYPE_KIND = {
    "integer": "int", "number": "float", "boolean": "bool",
    "string": "str", "array": "list",
}


def _build_field_meta() -> dict[str, dict[str, Any]]:
    """Widget metadata (kind / min / max / opts) for every overridable field,
    derived from the OverrideProfile JSON schema so it can never drift from the
    Pydantic bounds. Drives the profile editor + direct-override sub-editor."""
    schema = config_store.OverrideProfile.model_json_schema()
    out: dict[str, dict[str, Any]] = {}
    for name, spec in schema.get("properties", {}).items():
        if name == "locks":
            continue
        variants = spec.get("anyOf") or [spec]
        v = next((x for x in variants if x.get("type") != "null"), variants[0])
        info: dict[str, Any] = {}
        if name in ("PIPELINE_RULES_EXCLUDE", "PIPELINE_RULES_INCLUDE"):
            info["kind"] = "rulelist"
        elif "enum" in v:
            info["kind"] = "enum"
            info["opts"] = v["enum"]
        else:
            info["kind"] = _TYPE_KIND.get(v.get("type"), "str")
        if "minimum" in v:
            info["min"] = v["minimum"]
        if "maximum" in v:
            info["max"] = v["maximum"]
        if "maxLength" in v:
            info["maxlen"] = v["maxLength"]
        out[name] = info
    return out


def _build_groups() -> list[dict[str, Any]]:
    """Section layout for the profile editor: the global /settings field groups
    filtered to the per-identity overridable scalars, so section names + order
    match the rest of the admin UI (Decode / Advanced / VAD / Live streaming /
    Output …). Load-time + server sections drop out entirely."""
    import admin_routes
    target = config_store.LOCKABLE_FIELDS
    out: list[dict[str, Any]] = []
    for section, subs in admin_routes._FIELD_GROUPS:
        subgroups = []
        for sub_title, names in subs:
            fields = [n for n in names if n in target]
            if fields:
                subgroups.append({"title": sub_title, "fields": fields})
        if subgroups:
            out.append({"title": section, "subgroups": subgroups})
    return out


def _build_rules() -> list[dict[str, Any]]:
    """Non-terminal pipeline rules (name/label/enabled) for the per-profile
    force-on/off checklist."""
    out = []
    for r in (getattr(cfg, "PIPELINE_RULES", None) or []):
        if not isinstance(r, dict) or r.get("type") == "terminal":
            continue
        out.append({
            "name": r.get("name"),
            "label": r.get("label") or r.get("name"),
            "enabled": bool(r.get("enabled", True)),
        })
    return out


def _build_usage() -> dict[str, dict[str, list[str]]]:
    """Reverse index: profile name → {users:[id…], keys:[id…]} that reference
    it. Powers the sidebar usage counts and the usage-aware delete guard."""
    usage: dict[str, dict[str, list[str]]] = {
        name: {"users": [], "keys": []} for name in (getattr(cfg, "OVERRIDE_PROFILES", None) or {})
    }
    for u in api_keys_store.list_users():
        uid = u["id"]
        for p in api_keys_store.get_user_config(uid).get("profiles", []):
            usage.setdefault(p, {"users": [], "keys": []})["users"].append(uid)
        for k in api_keys_store.list_keys(uid):
            for p in (k.get("config") or {}).get("profiles", []):
                usage.setdefault(p, {"users": [], "keys": []})["keys"].append(k["id"])
    return usage


def _models() -> list[str]:
    """Model ids for the Explorer's model picker (allowlist, else default)."""
    allowed = sorted(getattr(cfg, "ALLOWED_MODELS", None) or [])
    if allowed:
        return allowed
    default = getattr(cfg, "DEFAULT_MODEL", "") or ""
    return [default] if default else []


@router.get("/state",
            dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def get_state() -> dict[str, Any]:
    """Profiles + the metadata the editors need."""
    return {
        "profiles": dict(getattr(cfg, "OVERRIDE_PROFILES", None) or {}),
        "field_meta": _build_field_meta(),
        "groups": _build_groups(),
        "rules": _build_rules(),
        "usage": _build_usage(),
        "models": _models(),
    }


@router.get("", response_class=HTMLResponse,
            dependencies=[Depends(require_admin_webui_host)])
async def overrides_page() -> HTMLResponse:
    return HTMLResponse(
        web_common.render_page(_OVERRIDES_HTML, current="overrides"),
        headers={"Cache-Control": "no-store"},
    )


@router.post("/state",
             dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def post_state(payload: dict[str, Any], request: Request) -> JSONResponse:
    """Persist the OVERRIDE_PROFILES dirty diff (the client sends the full
    profiles dict under that key). Same validate → save → hot-apply contract as
    /settings/state; OVERRIDE_PROFILES is a hot field (resolved per-request, so
    no cache rebuild / model eviction needed)."""
    # Only the OVERRIDE_PROFILES key is accepted here — this page never edits
    # any other global setting.
    unknown = set(payload) - {"OVERRIDE_PROFILES"}
    if unknown:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"unexpected fields for this page: {sorted(unknown)}")
    try:
        written = config_store.save_overrides(payload)
    except ValidationError as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"errors": config_store.format_validation_errors(e)},
        )
    except OSError as e:
        logger.error("[overrides] save failed: %s", e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"could not write config.local.json: {e}")

    import admin_routes
    applied = await admin_routes._apply_hot_changes(written)
    client_host = request.client.host if request.client else "?"
    logger.info("[overrides] profiles update from=%s saved=%s",
                client_host, sorted(written.keys()))
    return JSONResponse({
        "saved": sorted(written.keys()),
        **applied,
        "requires_restart": bool(applied["cold_pending"]),
    })


def _resolve_model(model: str | None) -> str | None:
    model = (model or "").strip()
    if not model or model == "whisper-1":
        return getattr(cfg, "DEFAULT_MODEL", "") or None
    return model


@router.get("/resolve",
            dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def resolve(user_id: str = "", key_id: str = "", model: str = "",
                  sim: str = "") -> dict[str, Any]:
    """Effective-config waterfall for (user_id [, key_id], model), optionally
    simulating a client per-request decode_override (`sim` = JSON object of
    lowercase decode keys). Returns, per field, the ordered layer stack with the
    winner, lock state, and the simulated client outcome."""
    sim_dict: dict[str, Any] = {}
    if sim.strip():
        try:
            parsed = json.loads(sim)
            if isinstance(parsed, dict):
                sim_dict = parsed
        except json.JSONDecodeError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                f"sim is not valid JSON: {e}")

    r = effective_config.resolve(
        _resolve_model(model),
        user_id=user_id or None, key_id=key_id or None,
        request_overrides=sim_dict, with_provenance=True,
    )

    fields: dict[str, Any] = {}
    for fname, stack in (r.provenance or {}).items():
        winner = next((h for h in stack if h["is_winner"]), None)
        client_sim = None
        ck = effective_config._CONFIG_TO_CLIENT_KEY.get(fname)
        if ck and ck in sim_dict:
            client_sim = {
                "value": sim_dict[ck],
                "outcome": "ignored_locked" if fname in r.locked else "applied",
            }
        fields[fname] = {
            "winner_value": winner["value"] if winner else None,
            "winner_layer": winner["layer_id"] if winner else None,
            "locked": fname in r.locked,
            "client_sim": client_sim,
            "layers": stack,
        }
    return {
        "fields": fields,
        "rules": r.rule_provenance or {},
        "profiles_applied": r.profiles_applied,
    }


# ---------------------------------------------------------------------
# HTML page — Profiles manager (master-detail) + effective-config Explorer
# ---------------------------------------------------------------------

_OVERRIDES_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{HEADER_TITLE}}</title>
{{PAGE_META}}
{{SCALE_BOOTSTRAP_HEAD}}
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --fg: #c9d1d9; --dim: #6e7681;
    --cyan: #79c0ff; --green: #7ee787; --yellow: #f2cc60; --magenta: #d2a8ff;
    --red: #ff7b72; --bold: #f0f6fc; --border: #30363d; --input-bg: #0d1117;
    --help: #8b949e;
  }
  html, body { background: var(--bg); color: var(--fg);
    font-family: var(--font-sans); font-size: var(--fs-base); margin: 0; }
  a { color: var(--cyan); }
  code, .mono { font-family: var(--font-mono); }
  header button { background: var(--panel); border: 1px solid var(--border);
    color: var(--fg); padding: 0.25rem 0.625rem; border-radius: 4px;
    cursor: pointer; font: inherit; font-size: var(--fs-sm); }
  header button.primary { color: var(--green); border-color: var(--green); }
  header button:disabled { opacity: 0.45; cursor: default; }
  main { padding: 1rem; max-width: 78rem; margin: 0 auto; }
  .hint { color: var(--help); font-size: var(--fs-sm); margin: 0.3rem 0 0; }
  .status { font-size: var(--fs-sm); color: var(--dim); margin-left: 0.5rem; }
  .status.ok { color: var(--green); } .status.err { color: var(--red); }
  /* subbar tabs */
  .ov-tab { background: transparent; border: 1px solid var(--border);
    color: var(--dim); padding: 0.2rem 0.7rem; border-radius: 999px;
    cursor: pointer; font: inherit; font-size: var(--fs-sm); }
  .ov-tab.active { color: var(--bold); border-color: var(--cyan); }
  /* master-detail */
  .ov-wrap { display: grid; grid-template-columns: 16rem 1fr; gap: 0.75rem; }
  .ov-side { display: flex; flex-direction: column; gap: 0.25rem;
    border: 1px solid var(--border); border-radius: 4px; padding: 0.5rem;
    background: var(--panel); align-self: start; }
  .ov-item { display: flex; align-items: center; gap: 0.4rem; cursor: pointer;
    padding: 0.3rem 0.4rem; border-radius: 4px; border: 1px solid transparent; }
  .ov-item:hover { background: #1c2230; }
  .ov-item.active { border-color: var(--cyan); background: #1c2230; }
  .ov-item .nm { flex: 1; font-family: var(--font-mono);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .ov-dot { width: 0.6rem; flex: 0 0 auto; }
  .ov-dot.on { color: var(--green); } .ov-dot.off { color: var(--dim); }
  .ov-count { font-size: var(--fs-xs); color: var(--dim);
    font-family: var(--font-mono); }
  .ov-usage { font-size: var(--fs-xs); color: var(--dim);
    padding: 0.2rem 0.4rem; border-top: 1px solid var(--border);
    margin-top: 0.3rem; }
  .ov-side-actions { display: flex; gap: 0.3rem; flex-wrap: wrap;
    margin-top: 0.3rem; }
  .ov-side-actions button, .ov-new-input { font: inherit;
    font-size: var(--fs-sm); }
  .ov-main { border: 1px solid var(--border); border-radius: 4px;
    background: var(--panel); padding: 0.6rem 0.85rem; min-height: 8rem; }
  .ov-empty { color: var(--dim); padding: 2rem 1rem; text-align: center; }
  .ov-sec > h4 { margin: 0.8rem 0 0.3rem; font-size: var(--fs-sm);
    color: var(--bold); border-bottom: 1px solid var(--border);
    padding-bottom: 0.2rem; }
  .ov-sec:first-child > h4 { margin-top: 0.2rem; }
  /* field row: dot | name | value | lock | ctrl */
  .ov-row { display: grid;
    grid-template-columns: 0.8rem minmax(11rem, 1fr) minmax(8rem, 1.4fr) 1.6rem 5.2rem;
    align-items: center; gap: 0.4rem; padding: 0.18rem 0; }
  .ov-row .ov-name { font-family: var(--font-mono); font-size: var(--fs-sm);
    color: var(--fg); overflow: hidden; text-overflow: ellipsis; }
  .ov-row.is-set .ov-name { color: var(--bold); }
  .ov-val input[type=text], .ov-val input[type=number], .ov-val select {
    width: 100%; box-sizing: border-box; background: var(--input-bg);
    color: var(--fg); border: 1px solid var(--border); border-radius: 4px;
    padding: 0.15rem 0.35rem; font-family: var(--font-mono);
    font-size: var(--fs-sm); }
  .ov-inherits { color: var(--dim); font-size: var(--fs-xs);
    font-family: var(--font-mono); }
  .ov-lock { background: none; border: 0; cursor: pointer; font-size: var(--fs-sm);
    color: var(--dim); padding: 0; }
  .ov-lock.locked { color: var(--yellow); }
  .ov-lock:disabled { opacity: 0.3; cursor: default; }
  .ov-ctrl button { background: none; border: 0; cursor: pointer;
    font-size: var(--fs-xs); color: var(--cyan); padding: 0; }
  .ov-ctrl .reset { color: var(--dim); }
  /* pipeline tri-state */
  .ov-rule { display: grid; grid-template-columns: 1fr 9rem; align-items: center;
    gap: 0.4rem; padding: 0.15rem 0; }
  .ov-rule .rl { font-size: var(--fs-sm); }
  .ov-rule .rl .slug { font-family: var(--font-mono); color: var(--dim);
    font-size: var(--fs-xs); }
  .ov-rule select { background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px; font: inherit;
    font-size: var(--fs-sm); padding: 0.1rem 0.25rem; }
  /* explorer */
  .ov-expl-bar { display: flex; flex-wrap: wrap; gap: 0.5rem 0.75rem;
    align-items: center; margin-bottom: 0.6rem; }
  .ov-expl-bar label { font-size: var(--fs-sm); color: var(--dim); }
  .ov-expl-bar select, .ov-sim { background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px; font: inherit;
    font-size: var(--fs-sm); padding: 0.15rem 0.35rem; }
  .ov-sim { font-family: var(--font-mono); width: 22rem; max-width: 100%; }
  .ov-sim.err { border-color: var(--red); }
  /* waterfall */
  .ov-wf { border: 1px solid var(--border); border-radius: 4px;
    margin: 0.4rem 0; background: var(--bg); }
  .ov-wf-head { display: flex; align-items: baseline; gap: 0.5rem;
    padding: 0.3rem 0.5rem; border-bottom: 1px solid var(--border);
    flex-wrap: wrap; }
  .ov-wf-name { font-family: var(--font-mono); color: var(--bold); }
  .ov-wf-win { font-size: var(--fs-sm); color: var(--dim); }
  .ov-wf-win code { color: var(--green); }
  .ov-wf-lock { color: var(--yellow); }
  .ov-wf-layers { list-style: none; margin: 0; padding: 0.2rem 0.3rem; }
  .ov-wf-layer { display: grid;
    grid-template-columns: 1rem minmax(9rem, 1fr) minmax(5rem, 1fr) auto;
    align-items: baseline; gap: 0.4rem; padding: 0.12rem 0.2rem;
    border-bottom: 1px dashed #21262d; }
  .ov-wf-layer:last-child { border-bottom: 0; }
  .ov-wf-layer .lbl { font-size: var(--fs-sm); }
  .ov-wf-layer .val { font-family: var(--font-mono); font-size: var(--fs-sm); }
  .ov-wf-layer .flag { font-size: var(--fs-xs); }
  .ov-wf-layer.win { border-left: 3px solid var(--green);
    background: #11271a; padding-left: 0.3rem; }
  .ov-wf-layer.win .lbl { color: var(--bold); } .ov-wf-layer.win .val { color: var(--green); }
  .ov-wf-layer.over .val { opacity: 0.5; text-decoration: line-through; }
  .ov-wf-layer.over .lbl { opacity: 0.6; }
  .ov-wf-layer.unset .val { color: var(--dim); font-style: italic; }
  .ov-wf-layer.sim .lbl { color: var(--cyan); }
  .ov-wf-layer.sim.ignored .flag { color: var(--red); }
  .ov-wf-layer .flag.locked { color: var(--yellow); }
  .ov-wf-tick { color: var(--green); }
  @media (prefers-reduced-motion: no-preference) {
    .ov-wf-layer, .ov-lock { transition: background-color .16s ease, color .16s ease; }
  }
  @media (max-width: 56rem) { .ov-wrap { grid-template-columns: 1fr; }
    .ov-row { grid-template-columns: 0.8rem 1fr; }
    .ov-row .ov-val, .ov-row .ov-lock, .ov-row .ov-ctrl { grid-column: 2; } }
  {{NAV_CSS}}
</style></head>
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
    <span class="subbar-title">Overrides</span>
    <span class="subbar-left">
      <button id="tab-profiles" class="ov-tab active">Profiles</button>
      <button id="tab-explorer" class="ov-tab">Explorer</button>
    </span>
    <span class="subbar-right" id="profiles-actions">
      <button id="discard-btn" disabled>Discard</button>
      <button id="save-btn" class="primary" disabled>Save</button>
      <span id="save-status" class="status"></span>
    </span>
  </div>
</header>

<main>
  <section id="panel-profiles">
    <p class="hint">Reusable config profiles. Assign them (ordered, earlier
      wins) to users &amp; API keys on the <a href="/settings/api-keys">API keys</a>
      page. A &#128274; field cannot be overridden per-request by the client.</p>
    <div class="ov-wrap">
      <div class="ov-side" id="profile-list"></div>
      <div class="ov-main" id="profile-main"></div>
    </div>
  </section>
  <section id="panel-explorer" hidden>
    <p class="hint">What-if: pick a user (and optionally one of their keys) and a
      model to see how every setting resolves &mdash; which layer wins, what is
      overridden, what is locked, and how a simulated client override fares.</p>
    <div class="ov-expl-bar">
      <label>user <select id="ex-user"></select></label>
      <label>key <select id="ex-key"></select></label>
      <label>model <select id="ex-model"></select></label>
      <label>sim client override
        <input id="ex-sim" class="ov-sim" placeholder='{"beam_size": 8}'></label>
    </div>
    <div id="ex-out"></div>
  </section>
</main>

{{SCALE_PICKER_JS}}
{{SEV_POLLER_JS}}
<script>
/* Standalone resolution-waterfall renderer — its own esc() scope so both the
   Explorer and (later) the api-keys inline preview can call it. Pure: takes a
   per-field /resolve object and returns a DOM node. */
window._renderWaterfall = (function () {
  'use strict';
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }
  function fmt(v) {
    if (v === null || v === undefined) return '—';
    if (typeof v === 'boolean') return v ? 'true' : 'false';
    if (typeof v === 'object') return JSON.stringify(v);
    return String(v);
  }
  return function (name, fr) {
    var el = document.createElement('div');
    el.className = 'ov-wf';
    var locked = fr.locked
      ? ' <span class="ov-wf-lock" title="locked — client cannot override">&#128274;</span>'
      : '';
    var rows = (fr.layers || []).map(function (h) {
      var cls = 'ov-wf-layer';
      if (h.is_winner) cls += ' win';
      else if (h.is_set) cls += ' over';
      else cls += ' unset';
      var flag = h.locked ? '<span class="flag locked">&#128274; locked</span>' : '';
      return '<li class="' + cls + '">'
        + '<span class="ov-wf-tick">' + (h.is_winner ? '✓' : '') + '</span>'
        + '<span class="lbl">' + esc(h.label) + '</span>'
        + '<span class="val">' + esc(fmt(h.is_set ? h.value : undefined)) + '</span>'
        + flag + '</li>';
    });
    if (fr.client_sim) {
      var ig = fr.client_sim.outcome === 'ignored_locked';
      rows.push('<li class="ov-wf-layer sim' + (ig ? ' ignored' : '') + '">'
        + '<span class="ov-wf-tick"></span>'
        + '<span class="lbl">client (sim)</span>'
        + '<span class="val">' + esc(fmt(fr.client_sim.value)) + '</span>'
        + '<span class="flag">' + (ig ? '⊘ ignored — locked' : 'applied') + '</span></li>');
    }
    el.innerHTML =
      '<div class="ov-wf-head"><span class="ov-wf-name">' + esc(name) + '</span>'
      + '<span class="ov-wf-win">winner <code>' + esc(fmt(fr.winner_value))
      + '</code> &middot; ' + esc(fr.winner_layer || 'library') + '</span>' + locked + '</div>'
      + '<ol class="ov-wf-layers">' + rows.join('') + '</ol>';
    return el;
  };
})();
</script>
<script>
(function () {
  'use strict';
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }
  var coll = new Intl.Collator('de', { sensitivity: 'base', numeric: true });

  async function api(method, path, body) {
    var h = { Accept: 'application/json' };
    if (method !== 'GET' && method !== 'HEAD') {
      h['X-CSRF-Token'] = window._csrfToken ? window._csrfToken() : '';
    }
    var opts = { method: method, headers: h };
    if (body !== undefined) { h['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
    return fetch(path, opts);
  }
  {{NOT_ADMIN_LANDING_JS}}
  async function guard403(r) {
    if (r && r.status === 403) {
      try { _renderNoAccessLanding({ page: 'overrides' }); } catch (_) {}
      return true;
    }
    return false;
  }

  // ---- state ----
  var S = null;                 // /state payload
  var profiles = {};            // editable working copy
  var snapshot = '{}';          // JSON snapshot for dirty compare
  var sel = null;               // selected profile name
  var PIPE_FIELDS = ['PIPELINE_RULES_EXCLUDE', 'PIPELINE_RULES_INCLUDE'];

  // ---- dom refs ----
  var listEl, mainEl, saveBtn, discardBtn, statusEl;

  function dirty() { return JSON.stringify(profiles) !== snapshot; }
  function refreshButtons() {
    var d = dirty();
    saveBtn.disabled = !d; discardBtn.disabled = !d;
  }
  function setStatus(msg, kind) {
    statusEl.textContent = msg || '';
    statusEl.className = 'status' + (kind ? ' ' + kind : '');
  }

  function profileNames() { return Object.keys(profiles).sort(coll.compare); }
  function overrideCount(p) {
    var n = 0;
    Object.keys(p || {}).forEach(function (k) {
      if (k === 'locks') return;
      if (PIPE_FIELDS.indexOf(k) >= 0) { n += (p[k] || []).length; return; }
      if (p[k] !== null && p[k] !== undefined) n += 1;
    });
    // Value-less locks (lock-to-inherited) carry no value entry but still do
    // something, so count them — otherwise a locks-only profile looks empty.
    ((p && p.locks) || []).forEach(function (f) {
      if (p[f] === null || p[f] === undefined) n += 1;
    });
    return n;
  }

  // ---- sidebar ----
  function renderSide() {
    listEl.innerHTML = '';
    var names = profileNames();
    names.forEach(function (name) {
      var p = profiles[name];
      var item = document.createElement('div');
      item.className = 'ov-item' + (name === sel ? ' active' : '');
      var cnt = overrideCount(p);
      item.innerHTML = '<span class="ov-dot ' + (cnt ? 'on' : 'off') + '">'
        + (cnt ? '●' : '○') + '</span>'
        + '<span class="nm">' + esc(name) + '</span>'
        + '<span class="ov-count">' + cnt + '</span>';
      item.onclick = function () { sel = name; render(); };
      listEl.appendChild(item);
      if (name === sel) {
        var u = (S.usage || {})[name] || { users: [], keys: [] };
        var us = document.createElement('div');
        us.className = 'ov-usage';
        us.textContent = 'used by ' + u.users.length + ' user'
          + (u.users.length === 1 ? '' : 's') + ' · ' + u.keys.length + ' key'
          + (u.keys.length === 1 ? '' : 's');
        listEl.appendChild(us);
      }
    });
    if (!names.length) {
      var e = document.createElement('div');
      e.className = 'ov-usage'; e.textContent = 'no profiles yet';
      listEl.appendChild(e);
    }
    var acts = document.createElement('div');
    acts.className = 'ov-side-actions';
    var nb = document.createElement('button'); nb.textContent = '+ new';
    nb.onclick = newProfile;
    acts.appendChild(nb);
    if (sel) {
      var db = document.createElement('button'); db.textContent = '⧉ duplicate';
      db.onclick = duplicateProfile; acts.appendChild(db);
      var xb = document.createElement('button'); xb.textContent = '× delete';
      xb.onclick = deleteProfile; acts.appendChild(xb);
    }
    listEl.appendChild(acts);
  }

  function newProfile() {
    var name = (prompt('New profile name (a-z0-9-, max 32):') || '').trim().toLowerCase();
    if (!name) return;
    if (!/^[a-z0-9][a-z0-9-]{0,31}$/.test(name)) { alert('invalid profile name'); return; }
    if (profiles[name]) { alert('profile already exists'); return; }
    profiles[name] = {}; sel = name; render(); refreshButtons();
  }
  function duplicateProfile() {
    if (!sel) return;
    var base = sel + '-copy'; var name = base; var i = 2;
    while (profiles[name]) { name = base + i; i++; }
    profiles[name] = JSON.parse(JSON.stringify(profiles[sel]));
    sel = name; render(); refreshButtons();
  }
  function deleteProfile() {
    if (!sel) return;
    var u = (S.usage || {})[sel] || { users: [], keys: [] };
    if (u.users.length || u.keys.length) {
      alert('Profile "' + sel + '" is in use by ' + u.users.length + ' user(s) and '
        + u.keys.length + ' key(s). Unbind it on the API keys page first.');
      return;
    }
    if (!confirm('Delete profile "' + sel + '"?')) return;
    delete profiles[sel]; sel = null; render(); refreshButtons();
  }

  // ---- main pane ----
  function render() { renderSide(); renderMain(); }

  function renderMain() {
    mainEl.innerHTML = '';
    if (!sel) {
      mainEl.innerHTML = '<div class="ov-empty">Select a profile, or create one with <b>+ new</b>.</div>';
      return;
    }
    var p = profiles[sel];
    (S.groups || []).forEach(function (g) {
      var fields = [];
      (g.subgroups || []).forEach(function (sg) {
        (sg.fields || []).forEach(function (f) { fields.push(f); });
      });
      if (!fields.length) return;
      var sec = document.createElement('div');
      sec.className = 'ov-sec';
      sec.innerHTML = '<h4>' + esc(g.title) + '</h4>';
      fields.forEach(function (f) { sec.appendChild(fieldRow(p, f)); });
      mainEl.appendChild(sec);
    });
    mainEl.appendChild(pipelineSection(p));
  }

  function isSet(p, f) { return p[f] !== null && p[f] !== undefined; }

  function fieldRow(p, name) {
    var meta = (S.field_meta || {})[name] || { kind: 'str' };
    var row = document.createElement('div');
    row.className = 'ov-row' + (isSet(p, name) ? ' is-set' : '');
    var dot = '<span class="ov-dot ' + (isSet(p, name) ? 'on' : 'off') + '">'
      + (isSet(p, name) ? '●' : '○') + '</span>';
    row.innerHTML = dot + '<span class="ov-name" title="' + esc(name) + '">' + esc(name) + '</span>';
    var valCell = document.createElement('span'); valCell.className = 'ov-val';
    var lockCell = document.createElement('button'); lockCell.className = 'ov-lock';
    var ctrl = document.createElement('span'); ctrl.className = 'ov-ctrl';

    var locks = p.locks || [];
    var locked = locks.indexOf(name) >= 0;
    lockCell.innerHTML = locked ? '\u{1F512}' : '\u{1F513}';
    if (locked) lockCell.classList.add('locked');
    lockCell.setAttribute('role', 'switch');
    lockCell.setAttribute('aria-checked', locked ? 'true' : 'false');
    // Available even when the field is unset: a value-less lock pins the
    // inherited (per-model/global) value and still blocks client overrides.
    lockCell.title = isSet(p, name)
      ? 'lock — client cannot override this field'
      : 'lock to inherited — client cannot override this field';
    lockCell.onclick = function () { toggleLock(p, name); };

    if (isSet(p, name)) {
      valCell.appendChild(makeWidget(name, meta, p[name], function (v) { setVal(p, name, v); }));
      var rb = document.createElement('button'); rb.className = 'reset';
      rb.textContent = '↶ reset';
      rb.onclick = function () { clearVal(p, name); };
      ctrl.appendChild(rb);
    } else {
      valCell.innerHTML = '<span class="ov-inherits">inherits</span>';
      var ab = document.createElement('button'); ab.textContent = '+ override';
      ab.onclick = function () { setVal(p, name, defaultFor(meta)); };
      ctrl.appendChild(ab);
    }
    row.appendChild(valCell); row.appendChild(lockCell); row.appendChild(ctrl);
    return row;
  }

  function defaultFor(meta) {
    if (meta.kind === 'bool') return true;
    if (meta.kind === 'int') return meta.min != null ? meta.min : 0;
    if (meta.kind === 'float') return meta.min != null ? meta.min : 0;
    if (meta.kind === 'enum') return (meta.opts || [''])[0];
    return '';
  }

  function makeWidget(name, meta, val, onchange) {
    var el;
    if (meta.kind === 'bool') {
      el = document.createElement('input'); el.type = 'checkbox';
      el.className = 'switch'; el.setAttribute('role', 'switch');
      el.checked = !!val;
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

  function setVal(p, name, v) {
    if (v === null) { clearVal(p, name); return; }
    p[name] = v; render(); refreshButtons();
  }
  function clearVal(p, name) {
    delete p[name];
    if (p.locks) { var i = p.locks.indexOf(name); if (i >= 0) p.locks.splice(i, 1); }
    render(); refreshButtons();
  }
  function toggleLock(p, name) {
    if (!p.locks) p.locks = [];
    var i = p.locks.indexOf(name);
    if (i >= 0) p.locks.splice(i, 1); else p.locks.push(name);
    render(); refreshButtons();
  }

  // ---- pipeline tri-state ----
  function pipelineSection(p) {
    var sec = document.createElement('div'); sec.className = 'ov-sec';
    sec.innerHTML = '<h4>Pipeline rules</h4>';
    var exc = p.PIPELINE_RULES_EXCLUDE || [];
    var inc = p.PIPELINE_RULES_INCLUDE || [];
    (S.rules || []).forEach(function (r) {
      var row = document.createElement('div'); row.className = 'ov-rule';
      var state = exc.indexOf(r.name) >= 0 ? 'off' : (inc.indexOf(r.name) >= 0 ? 'on' : 'inherit');
      row.innerHTML = '<span class="rl">' + esc(r.label)
        + ' <span class="slug">' + esc(r.name) + (r.enabled ? '' : ' (off)') + '</span></span>';
      var seln = document.createElement('select');
      [['inherit', 'inherit (' + (r.enabled ? 'on' : 'off') + ')'],
       ['on', 'force on'], ['off', 'force off']].forEach(function (o) {
        var op = document.createElement('option'); op.value = o[0]; op.textContent = o[1];
        if (o[0] === state) op.selected = true; seln.appendChild(op);
      });
      seln.onchange = function () { setRule(p, r.name, seln.value); };
      row.appendChild(seln); sec.appendChild(row);
    });
    if (!(S.rules || []).length) {
      sec.innerHTML += '<p class="hint">No pipeline rules configured.</p>';
    }
    return sec;
  }
  function setRule(p, slug, state) {
    function pull(arr) { var i = arr.indexOf(slug); if (i >= 0) arr.splice(i, 1); }
    p.PIPELINE_RULES_EXCLUDE = p.PIPELINE_RULES_EXCLUDE || [];
    p.PIPELINE_RULES_INCLUDE = p.PIPELINE_RULES_INCLUDE || [];
    pull(p.PIPELINE_RULES_EXCLUDE); pull(p.PIPELINE_RULES_INCLUDE);
    if (state === 'off') p.PIPELINE_RULES_EXCLUDE.push(slug);
    else if (state === 'on') p.PIPELINE_RULES_INCLUDE.push(slug);
    if (!p.PIPELINE_RULES_EXCLUDE.length) delete p.PIPELINE_RULES_EXCLUDE;
    if (!p.PIPELINE_RULES_INCLUDE.length) delete p.PIPELINE_RULES_INCLUDE;
    refreshButtons();
  }

  // ---- save / discard ----
  async function save() {
    setStatus('saving…');
    var r = await api('POST', '/settings/overrides/state', { OVERRIDE_PROFILES: profiles });
    if (await guard403(r)) return;
    if (r.status === 422) {
      var j = await r.json();
      setStatus('invalid: ' + (j.errors || []).map(function (e) { return e.loc + ' ' + e.msg; }).join('; '), 'err');
      return;
    }
    if (!r.ok) { setStatus('save failed (' + r.status + ')', 'err'); return; }
    snapshot = JSON.stringify(profiles);
    await loadState(true);    // refresh usage + canonical values
    setStatus('saved', 'ok'); refreshButtons();
  }
  function discard() {
    profiles = JSON.parse(snapshot);
    if (sel && !profiles[sel]) sel = null;
    render(); refreshButtons(); setStatus('');
  }

  // ---- explorer ----
  var exUser, exKey, exModel, exSim, exOut, usersCache = [];
  function buildExplorer() {
    exUser = document.getElementById('ex-user');
    exKey = document.getElementById('ex-key');
    exModel = document.getElementById('ex-model');
    exSim = document.getElementById('ex-sim');
    exOut = document.getElementById('ex-out');
    (S.models || []).forEach(function (m) {
      var o = document.createElement('option'); o.value = m; o.textContent = m || '(default)';
      exModel.appendChild(o);
    });
    exModel.onchange = doResolve;
    exUser.onchange = function () { fillKeys(); doResolve(); };
    exKey.onchange = doResolve;
    var t = null;
    exSim.oninput = function () { clearTimeout(t); t = setTimeout(doResolve, 250); };
    fillUsers();
  }
  async function fillUsers() {
    var r = await fetch('/settings/api-keys/api/users');
    if (!r.ok) { return; }
    var j = await r.json();
    usersCache = (j.users || []);
    usersCache.sort(function (a, b) { return coll.compare(a.username, b.username); });
    exUser.innerHTML = '<option value="">—</option>';
    usersCache.forEach(function (u) {
      var o = document.createElement('option'); o.value = u.id; o.textContent = u.username;
      exUser.appendChild(o);
    });
    fillKeys();
  }
  async function fillKeys() {
    exKey.innerHTML = '<option value="">(no specific key)</option>';
    var uid = exUser.value; if (!uid) return;
    var r = await fetch('/settings/api-keys/api/users/' + encodeURIComponent(uid) + '/keys');
    if (!r.ok) return;
    var j = await r.json();
    (j.keys || []).forEach(function (k) {
      var o = document.createElement('option'); o.value = k.id;
      o.textContent = (k.label || '(no label)') + ' · ' + k.key_prefix + '…' + k.key_last4;
      exKey.appendChild(o);
    });
  }
  async function doResolve() {
    if (!exUser.value) { exOut.innerHTML = '<p class="hint">Pick a user to resolve.</p>'; return; }
    var sim = exSim.value.trim();
    if (sim) { try { JSON.parse(sim); exSim.classList.remove('err'); }
               catch (e) { exSim.classList.add('err'); return; } }
    var qs = new URLSearchParams({ user_id: exUser.value, model: exModel.value || '' });
    if (exKey.value) qs.set('key_id', exKey.value);
    if (sim) qs.set('sim', sim);
    var r = await fetch('/settings/overrides/resolve?' + qs.toString());
    if (await guard403(r)) return;
    if (!r.ok) { exOut.innerHTML = '<p class="status err">resolve failed (' + r.status + ')</p>'; return; }
    var j = await r.json();
    exOut.innerHTML = '';
    if (j.profiles_applied && j.profiles_applied.length) {
      var pa = document.createElement('p'); pa.className = 'hint';
      pa.textContent = 'profiles applied (earlier wins): ' + j.profiles_applied.join(' → ');
      exOut.appendChild(pa);
    }
    var names = Object.keys(j.fields || {}).sort(coll.compare);
    names.forEach(function (f) { exOut.appendChild(window._renderWaterfall(f, j.fields[f])); });
    if (!names.length) exOut.appendChild(document.createTextNode('no resolvable fields'));
  }

  // ---- tabs ----
  function showTab(which) {
    var prof = which === 'profiles';
    document.getElementById('panel-profiles').hidden = !prof;
    document.getElementById('panel-explorer').hidden = prof;
    document.getElementById('profiles-actions').style.visibility = prof ? '' : 'hidden';
    document.getElementById('tab-profiles').classList.toggle('active', prof);
    document.getElementById('tab-explorer').classList.toggle('active', !prof);
    try { localStorage.setItem('ov.tab', which); } catch (_) {}
    if (!prof && !exUser) buildExplorer();
  }

  // ---- boot ----
  async function loadState(keepSel) {
    var r = await fetch('/settings/overrides/state');
    if (await guard403(r)) return false;
    if (!r.ok) { setStatus('load failed (' + r.status + ')', 'err'); return false; }
    S = await r.json();
    profiles = JSON.parse(JSON.stringify(S.profiles || {}));
    snapshot = JSON.stringify(profiles);
    if (!keepSel) { var ns = profileNames(); sel = ns.length ? ns[0] : null; }
    else if (sel && !profiles[sel]) sel = null;
    return true;
  }
  async function boot() {
    listEl = document.getElementById('profile-list');
    mainEl = document.getElementById('profile-main');
    saveBtn = document.getElementById('save-btn');
    discardBtn = document.getElementById('discard-btn');
    statusEl = document.getElementById('save-status');
    saveBtn.onclick = save; discardBtn.onclick = discard;
    document.getElementById('tab-profiles').onclick = function () { showTab('profiles'); };
    document.getElementById('tab-explorer').onclick = function () { showTab('explorer'); };
    var ok = await loadState(false);
    if (!ok) return;
    render(); refreshButtons();
    var t = 'profiles';
    try { t = localStorage.getItem('ov.tab') || 'profiles'; } catch (_) {}
    showTab(t);
  }
  boot();
})();
</script>
</body></html>
"""
