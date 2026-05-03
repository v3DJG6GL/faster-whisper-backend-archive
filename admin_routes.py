"""
Admin WebUI for faster-whisper-backend.

Mounted at /config when WHISPER_ADMIN_UI=1. Endpoints:

  GET  /config            HTML page (loopback only)
  GET  /config/state      Resolved config + provenance + warm/cold tags
  POST /config/state      Save overrides (validation errors -> 422)
  POST /config/restart    Detach a self-restart helper (Windows only)

Security model (layered):
  1. Allowlist gate:   require_admin_host rejects callers not in
                       cfg.ADMIN_ALLOWED_HOSTS (loopback always permitted)
  2. Bearer token:     if WHISPER_ADMIN_TOKEN is set, mutating endpoints
                       require Authorization: Bearer <token>
  3. Pydantic schema:  AdminConfig validates body shape, types, bounds
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError

import config as cfg
import config_store
import web_common

logger = logging.getLogger("whisper-api")

# Fields the WebUI is allowed to surface. Keep this as the single source of
# truth for the form layout — drives section grouping in the HTML and the
# /config/state endpoint's provenance map.
_FIELD_GROUPS: list[tuple[str, list[str]]] = [
    ("Models", [
        "DEFAULT_MODEL", "ALLOWED_MODELS", "PRELOAD_MODELS", "MAX_LOADED_MODELS",
        "MODEL_DEVICE", "MODEL_COMPUTE_TYPE",
        "MODEL_DEVICE_FALLBACK", "MODEL_COMPUTE_TYPE_FALLBACK",
    ]),
    ("Locale", [
        "DEFAULT_LANGUAGE", "DEFAULT_PROMPT", "CHARACTER_REPLACEMENTS",
    ]),
    ("Pipeline", [
        "DICTATION_ENABLED", "TRUST_MODEL_PUNCTUATION", "TRACE_ENABLED",
        "PUNCTUATION_TO_KEEP", "DICTATION_MAP",
        "LOWERCASE_AFTER_STRIPPED_TERMINATOR",
    ]),
    ("Whisper transcribe", [
        "BEAM_SIZE", "BEST_OF",
        "VAD_FILTER", "VAD_MIN_SILENCE_MS", "VAD_SPEECH_PAD_MS", "VAD_THRESHOLD",
        "CONDITION_ON_PREVIOUS_TEXT", "WORD_TIMESTAMPS_ENABLED",
        "NO_SPEECH_THRESHOLD", "LOG_PROB_THRESHOLD", "COMPRESSION_RATIO_THRESHOLD",
    ]),
    ("Logging", [
        "LOG_FILE", "LOG_MAX_BYTES", "LOG_BACKUP_COUNT",
    ]),
    ("Server (uvicorn)", [
        "SERVER_HOST", "SERVER_PORT", "SERVER_WORKERS", "SERVER_LOG_LEVEL",
    ]),
    ("Access (allowlists)", [
        "ADMIN_ALLOWED_HOSTS", "STATS_ALLOWED_HOSTS",
    ]),
]


# --- auth deps ---------------------------------------------------------------
#
# /config is gated by an IP/CIDR allowlist (cfg.ADMIN_ALLOWED_HOSTS). Loopback
# is always implicitly permitted — even a misconfigured allowlist can never
# lock the local operator out, since they can still curl from the box itself.
require_admin_host = web_common.require_allowed_host(lambda: cfg.ADMIN_ALLOWED_HOSTS)


_bearer = HTTPBearer(auto_error=False)


def require_admin_token(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """If cfg.ADMIN_TOKEN is set, require a matching bearer token. If unset,
    this is a no-op — the loopback check alone is the gate."""
    expected = cfg.ADMIN_TOKEN
    if not expected:
        return
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bearer token required")
    if not secrets.compare_digest(creds.credentials, expected):
        logger.warning("[config] rejected bad bearer token from a loopback caller")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")


# --- router ------------------------------------------------------------------

router = APIRouter(prefix="/config")


def _resolved_value(field: str) -> Any:
    """Read the current effective value of a config field by attribute name."""
    val = getattr(cfg, field, None)
    # Convert un-JSON-able types so the WebUI gets clean data.
    if isinstance(val, (set, frozenset)):
        return sorted(val)
    if isinstance(val, tuple):
        return [list(p) if isinstance(p, tuple) else p for p in val]
    return val


def _provenance(field: str, env_pinned: dict[str, str], saved: dict[str, Any]) -> str:
    """Where the current effective value came from: 'env', 'local.json', or 'default'."""
    if field in env_pinned:
        return "env"
    if field in saved:
        return "local.json"
    return "default"


@router.get("", response_class=HTMLResponse, dependencies=[Depends(require_admin_host)])
async def config_page() -> HTMLResponse:
    """The admin HTML page. Allowlist-gated (loopback always allowed) — no
    token required to LOAD the page; the page itself collects the token and
    attaches it on every fetch. `no-store` so browsers never serve a stale
    build after a service restart."""
    return HTMLResponse(
        web_common.render_page(_CONFIG_VIEWER_HTML, current="config"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/state", dependencies=[Depends(require_admin_host), Depends(require_admin_token)])
async def get_state() -> dict[str, Any]:
    """Return the resolved config (current effective values) plus provenance
    flags so the WebUI can render badges. Does NOT include the saved-only
    overrides — the form fills from effective values; the badge tells the
    user where the value is coming from."""
    saved = config_store.load_overrides()
    env_pinned = config_store.env_pinned_fields()

    fields: dict[str, dict[str, Any]] = {}
    for _section, names in _FIELD_GROUPS:
        for name in names:
            fields[name] = {
                "value": _resolved_value(name),
                "provenance": _provenance(name, env_pinned, saved),
                "env_var": env_pinned.get(name),
                "restart_required": name in config_store.RESTART_REQUIRED_FIELDS,
            }

    return {
        "fields": fields,
        "groups": [{"title": title, "fields": names} for title, names in _FIELD_GROUPS],
        "token_required": bool(cfg.ADMIN_TOKEN),
        "service_name": "WhisperAPI",
    }


@router.post("/state", dependencies=[Depends(require_admin_host), Depends(require_admin_token)])
async def post_state(payload: dict[str, Any], request: Request) -> JSONResponse:
    """Validate and persist overrides. Returns the diff (which fields were
    saved) plus a `requires_restart` flag for the WebUI to act on. Hot fields
    are applied to the running cfg module immediately and any derived caches
    are rebuilt; cold fields stick around in the JSON file for the restart to
    pick up."""
    try:
        written = config_store.save_overrides(payload)
    except ValidationError as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"errors": config_store.format_validation_errors(e)},
        )
    except OSError as e:
        logger.error("[config] save failed: %s", e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"could not write config.local.json: {e}")

    # Apply hot edits to the running cfg module so the next request sees them.
    # We re-load from disk so the in-memory values get the same coercions
    # (set/frozenset/tuple) load_overrides applies.
    coerced = config_store.load_overrides()
    hot_changed: list[str] = []
    cold_changed: list[str] = []
    needs_cache_rebuild = False
    env_pinned = config_store.env_pinned_fields()

    for name in written:
        if name in env_pinned:
            # Save persists, but the running cfg won't change until the env
            # var is unset. Don't include in `hot_changed` — nothing changed
            # in memory.
            continue
        new_val = coerced.get(name, getattr(cfg, name, None))
        setattr(cfg, name, new_val)
        if name in config_store.CACHE_REBUILD_FIELDS:
            needs_cache_rebuild = True
        if name in config_store.RESTART_REQUIRED_FIELDS:
            cold_changed.append(name)
        else:
            hot_changed.append(name)

    if needs_cache_rebuild:
        try:
            import main as _main
            _main.rebuild_caches()
            logger.info("[config] rebuilt pipeline caches after admin update")
        except Exception as e:
            logger.error("[config] cache rebuild failed: %s", e)

    client_host = request.client.host if request.client else "?"
    logger.info(
        "[config] admin update from=%s saved=%d hot=%s cold=%s pinned=%s",
        client_host, len(written), hot_changed, cold_changed,
        [n for n in written if n in env_pinned],
    )

    return JSONResponse({
        "saved": sorted(written.keys()),
        "hot_applied": hot_changed,
        "cold_pending": cold_changed,
        "env_pinned_ignored": sorted(n for n in written if n in env_pinned),
        "requires_restart": bool(cold_changed),
    })


@router.post("/restart", dependencies=[Depends(require_admin_host), Depends(require_admin_token)])
async def post_restart(request: Request) -> JSONResponse:
    """Trigger a self-restart of the WhisperAPI Windows Service.

    Spawns `WhisperAPI.exe restart!` (WinSW's documented self-restart
    command) and schedules `os._exit(0)` ~1.5 s in the future, then
    returns 200. WinSW relaunches the wrapper after we die, surviving
    the SCM child-tree kill. The 1.5 s delay gives this response time
    to flush over loopback before uvicorn dies. End-to-end downtime is
    ~3-4 s for a no-preload deployment.

    See restart_service.py for why we use WinSW's explicit `restart!`
    rather than relying on <onfailure action="restart"/> on exit 0
    (v2's onfailure semantics on graceful exits are unreliable).
    """
    try:
        from restart_service import trigger_self_restart
    except ImportError as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"restart_service module unavailable: {e}")

    import sys as _sys
    if _sys.platform != "win32":
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            "self-restart is Windows-only; restart manually with "
            "`Restart-Service WhisperAPI`",
        )

    client_host = request.client.host if request.client else "?"
    logger.info("[config] admin restart requested from=%s", client_host)
    try:
        method = trigger_self_restart()
    except Exception as e:
        logger.error("[config] self-restart scheduling failed: %s", e)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"could not schedule self-restart: {e}",
        )
    delay_sec = 1.5
    logger.info("[config] self-restart scheduled via method=%s; "
                "process will exit in %.1f s", method, delay_sec)

    return JSONResponse({
        "status": "restarting",
        "method": method,
        "delay_sec": delay_sec,
    })


# --- HTML template ------------------------------------------------------------
# Vanilla JS, no build step. Mirrors the /logs viewer styling. Sections, table
# editor for DICTATION_MAP, textarea-per-line editors for list/set fields, save
# flow with restart modal + post-restart polling.

_CONFIG_VIEWER_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>faster-whisper-backend · config</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --fg: #c9d1d9; --dim: #6e7681;
    --cyan: #79c0ff; --green: #7ee787; --yellow: #f2cc60;
    --red: #ff7b72; --magenta: #d2a8ff; --bold: #f0f6fc;
    --border: #30363d; --input-bg: #0d1117;
  }
  html, body { background: var(--bg); color: var(--fg);
    font: 13px/1.45 ui-monospace, "Cascadia Code", Menlo, Consolas, monospace;
    margin: 0; padding: 0; min-height: 100%; }
  header { position: sticky; top: 0; background: var(--panel); border-bottom: 1px solid var(--border);
    padding: 8px 14px; display: flex; gap: 12px; align-items: center; z-index: 10; }
  header .title { font-weight: 600; color: var(--bold); }
  header .pill { padding: 2px 8px; border-radius: 999px; background: #21262d; color: var(--dim);
    font-size: 11px; }
  header .pill.ok { color: var(--green); border: 1px solid #1f4d2a; }
  header .pill.warn { color: var(--yellow); border: 1px solid #4d3e1f; }
  header .pill.err { color: var(--red); border: 1px solid #5a2424; }
  header button { background: #21262d; color: var(--fg); border: 1px solid var(--border);
    padding: 4px 10px; border-radius: 4px; cursor: pointer; font: inherit; }
  header button:hover { background: #30363d; }
  header button.primary { background: #238636; border-color: #2ea043; color: var(--bold); }
  header button.primary:hover { background: #2ea043; }
  header button:disabled { opacity: 0.4; cursor: not-allowed; }
  main { padding: 14px; max-width: 1100px; }
  section { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 10px 14px 12px; margin-bottom: 14px; }
  h2 { color: var(--bold); font-size: 14px; margin: 0 0 8px; padding-bottom: 6px;
    border-bottom: 1px solid var(--border); }
  .field { display: grid; grid-template-columns: 230px 1fr; gap: 10px;
    align-items: start; padding: 6px 0; border-bottom: 1px dashed #21262d; }
  .field:last-child { border-bottom: none; }
  .label-col { display: flex; flex-direction: column; gap: 4px; }
  .label-col .name { color: var(--bold); }
  .badges { display: flex; gap: 4px; flex-wrap: wrap; }
  .badge { font-size: 10px; padding: 1px 6px; border-radius: 999px;
    border: 1px solid var(--border); color: var(--dim); }
  .badge.live { color: var(--green); border-color: #1f4d2a; }
  .badge.restart { color: var(--yellow); border-color: #4d3e1f; }
  .badge.env { color: var(--magenta); border-color: #4a2e6f; }
  .badge.local { color: var(--cyan); border-color: #194f73; }
  .input-col input, .input-col textarea, .input-col select {
    width: 100%; box-sizing: border-box;
    background: var(--input-bg); color: var(--fg); border: 1px solid var(--border);
    padding: 5px 8px; border-radius: 4px; font: inherit; }
  .input-col input[type=checkbox] { width: auto; }
  .input-col textarea { font-family: inherit; min-height: 60px; resize: vertical; }
  .input-col .help { color: var(--dim); font-size: 11px; margin-top: 3px; }
  .err { color: var(--red); font-size: 11px; margin-top: 3px; }
  /* Field row dimming when a parent toggle makes this row irrelevant.
     pointer-events stays alive so the user can still see the contents and
     edit if they want — we just signal "this is currently unused". */
  .field.dep-irrelevant { opacity: 0.45; }
  .field.dep-irrelevant .input-col { filter: grayscale(0.6); }
  .field .dep-note { color: var(--dim); font-size: 11px; margin-top: 3px;
    font-style: italic; }
  /* Nullable-number editor in its disabled (null) state — greyed input,
     enable/disable button labelled accordingly. */
  .nullable-wrap input:disabled { opacity: 0.4; cursor: not-allowed; }
  table.dict { width: 100%; border-collapse: collapse; }
  table.dict th, table.dict td { border: 1px solid var(--border); padding: 4px 8px;
    text-align: left; }
  table.dict th { background: #1c2129; color: var(--dim); font-weight: 500; font-size: 11px; }
  table.dict td input { width: 100%; box-sizing: border-box; background: transparent;
    color: var(--fg); border: none; padding: 0; font: inherit; }
  table.dict td input:focus { outline: 1px solid var(--cyan); outline-offset: -1px; }
  table.dict button.del { background: transparent; border: 1px solid var(--border);
    color: var(--red); padding: 2px 6px; border-radius: 3px; cursor: pointer; font-size: 11px; }
  .add-row { margin-top: 6px; }
  .add-row button { background: #21262d; border: 1px solid var(--border); color: var(--fg);
    padding: 3px 10px; border-radius: 4px; cursor: pointer; font: inherit; }
  .modal { position: fixed; inset: 0; background: rgba(0,0,0,0.7); display: none;
    align-items: center; justify-content: center; z-index: 100; }
  .modal.show { display: flex; }
  .modal-box { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 16px 20px; max-width: 520px; }
  .modal-box h3 { margin: 0 0 10px; color: var(--bold); }
  .modal-box ul { margin: 6px 0 12px; padding-left: 20px; }
  .modal-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 14px; }
  .modal-actions button { padding: 5px 12px; }
  .login { max-width: 480px; margin: 60px auto; background: var(--panel);
    border: 1px solid var(--border); border-radius: 6px; padding: 20px 24px; }
  .login h1 { color: var(--bold); margin: 0 0 6px; font-size: 18px; }
  .login p { color: var(--dim); margin: 6px 0 12px; }
  .login input { width: 100%; box-sizing: border-box; background: var(--input-bg);
    color: var(--fg); border: 1px solid var(--border); padding: 8px;
    border-radius: 4px; font: inherit; }
  .login button { margin-top: 10px; background: #238636; border: 1px solid #2ea043;
    color: var(--bold); padding: 7px 14px; border-radius: 4px; cursor: pointer;
    font: inherit; }
  #toast { position: fixed; bottom: 16px; right: 16px; background: var(--panel);
    border: 1px solid var(--border); border-radius: 4px; padding: 8px 12px;
    color: var(--fg); display: none; }
  #toast.show { display: block; }
  #toast.err { border-color: #5a2424; color: var(--red); }
  {{NAV_CSS}}
</style></head>
<body>

<div id="login-wrap" class="login" style="display:none">
  <h1>faster-whisper-backend · admin</h1>
  <p>Enter the value of <code>WHISPER_ADMIN_TOKEN</code> to continue. The token
  stays in your browser's <code>sessionStorage</code> until you close the tab.</p>
  <input id="login-token" type="password" autocomplete="off" placeholder="bearer token">
  <button id="login-btn">Unlock</button>
  <p id="login-err" class="err"></p>
</div>

<div id="app-wrap" style="display:none">
  <header>
    <span class="title">faster-whisper-backend · config</span>
    {{NAV}}
    <span class="spacer"></span>
    <button id="logout-btn" title="forget token in this tab">logout</button>
    <button id="reload-btn">reload</button>
    <button id="restart-btn" title="restart the WhisperAPI Windows Service">restart</button>
    <button id="save-btn" class="primary" disabled>save</button>
    <span id="status" class="pill">loading…</span>
  </header>
  <main id="main"></main>
</div>

<div id="restart-modal" class="modal">
  <div class="modal-box">
    <h3 id="restart-modal-title">Restart required</h3>
    <p id="restart-modal-body">These changes need a service restart to take effect:</p>
    <ul id="restart-fields"></ul>
    <p>The page will reload once the service is back up.</p>
    <div class="modal-actions">
      <button id="restart-cancel">Cancel</button>
      <button id="restart-now" class="primary">Restart now</button>
    </div>
  </div>
</div>

<div id="restart-progress" class="modal">
  <div class="modal-box">
    <h3>Restarting…</h3>
    <p id="restart-progress-msg">Waiting for the service to come back up.</p>
  </div>
</div>

<div id="toast"></div>

<script>
(() => {
'use strict';

const TOKEN_KEY = 'whisper_admin_token';
let state = null;          // last server state
let dirty = {};            // field -> new value (only changed)

const $ = (id) => document.getElementById(id);

function authHeaders() {
  const t = sessionStorage.getItem(TOKEN_KEY);
  return t ? { 'Authorization': 'Bearer ' + t } : {};
}

function toast(msg, isErr) {
  const el = $('toast');
  el.textContent = msg;
  el.className = isErr ? 'show err' : 'show';
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.className = ''; }, 3500);
}

async function api(method, path, body) {
  const opts = { method, headers: { ...authHeaders() } };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  if (r.status === 401) {
    sessionStorage.removeItem(TOKEN_KEY);
    showLogin('token rejected');
    throw new Error('unauthorized');
  }
  return r;
}

function showLogin(errMsg) {
  $('login-wrap').style.display = '';
  $('app-wrap').style.display = 'none';
  $('login-err').textContent = errMsg || '';
}

function showApp() {
  $('login-wrap').style.display = 'none';
  $('app-wrap').style.display = '';
}

async function loadState() {
  const r = await api('GET', '/config/state');
  if (r.status === 401) return;  // showLogin already called
  if (!r.ok) {
    toast('failed to load state: ' + r.status, true);
    return;
  }
  state = await r.json();
  dirty = {};
  $('save-btn').disabled = true;
  $('status').textContent = 'loaded ' + Object.keys(state.fields).length + ' fields';
  $('status').className = 'pill ok';
  render();
}

function fieldDef(name) { return state.fields[name]; }
function isEnvPinned(name) { return fieldDef(name).provenance === 'env'; }

// Read the in-progress value if dirty, else the server-side saved value.
// Used by editors that want to react to OTHER fields' live edits — e.g.
// the DEFAULT_MODEL dropdown and PRELOAD_MODELS multi-select source their
// option set from the current ALLOWED_MODELS, including unsaved edits.
function currentValue(name) {
  return Object.prototype.hasOwnProperty.call(dirty, name)
    ? dirty[name]
    : fieldDef(name).value;
}

function setDirty(name, value) {
  // Equality vs. server value — if user reverts manually we drop the entry
  const cur = JSON.stringify(fieldDef(name).value);
  const nxt = JSON.stringify(value);
  if (cur === nxt) {
    delete dirty[name];
  } else {
    dirty[name] = value;
  }
  $('save-btn').disabled = Object.keys(dirty).length === 0;
  // Notify dependent editors when the model lists change. The DEFAULT_MODEL
  // dropdown + PRELOAD_MODELS multi-select listen for this and re-render.
  if (name === 'ALLOWED_MODELS' || name === 'PRELOAD_MODELS') {
    document.dispatchEvent(new CustomEvent('admin:model-lists-changed'));
  }
  // Re-evaluate "is row X irrelevant given the current state of toggle Y?"
  // Cheap (handful of fields), runs after every edit so the UI tracks live.
  applyFieldDependencies();
}

// Map of field → ("irrelevant" predicate, reason shown to the user when the
// row is dimmed). Predicates read currentValue() so they pick up dirty edits
// before save. Keep entries here whenever a new "parent" toggle is added that
// makes another field's value not-actually-used.
const _FIELD_DEPS = {
  DICTATION_MAP: {
    irrelevant: () => currentValue('DICTATION_ENABLED') === false,
    note: 'unused while DICTATION_ENABLED is off',
  },
  LOWERCASE_AFTER_STRIPPED_TERMINATOR: {
    irrelevant: () => currentValue('TRUST_MODEL_PUNCTUATION') === true,
    note: 'unused while TRUST_MODEL_PUNCTUATION is on (Step 3 STRIP TERMS is skipped)',
  },
};

function applyFieldDependencies() {
  for (const [name, { irrelevant, note }] of Object.entries(_FIELD_DEPS)) {
    const row = document.querySelector(`.field[data-field="${name}"]`);
    if (!row) continue;
    const dim = irrelevant();
    row.classList.toggle('dep-irrelevant', dim);
    let n = row.querySelector('.dep-note');
    if (dim && !n) {
      n = document.createElement('div');
      n.className = 'dep-note';
      n.textContent = note;
      row.querySelector('.input-col').appendChild(n);
    } else if (!dim && n) {
      n.remove();
    }
  }
}

function makeBadges(name) {
  const d = fieldDef(name);
  const wrap = document.createElement('div');
  wrap.className = 'badges';
  if (d.restart_required) {
    wrap.innerHTML += '<span class="badge restart">restart</span>';
  } else {
    wrap.innerHTML += '<span class="badge live">live</span>';
  }
  if (d.provenance === 'env') {
    wrap.innerHTML += '<span class="badge env">env: ' + d.env_var + '</span>';
  } else if (d.provenance === 'local.json') {
    wrap.innerHTML += '<span class="badge local">local.json</span>';
  }
  return wrap;
}

function fieldRow(name) {
  const row = document.createElement('div');
  row.className = 'field';
  row.dataset.field = name;   // used by applyFieldDependencies()
  const labelCol = document.createElement('div');
  labelCol.className = 'label-col';
  const nameEl = document.createElement('div');
  nameEl.className = 'name';
  nameEl.textContent = name;
  labelCol.appendChild(nameEl);
  labelCol.appendChild(makeBadges(name));
  if (isEnvPinned(name)) {
    const note = document.createElement('div');
    note.className = 'help';
    note.textContent = 'Currently overridden by env var; saves persist but '
      + 'only take effect when the env var is unset.';
    labelCol.appendChild(note);
  }
  row.appendChild(labelCol);

  const inputCol = document.createElement('div');
  inputCol.className = 'input-col';
  inputCol.appendChild(makeEditor(name));
  row.appendChild(inputCol);
  return row;
}

function makeEditor(name) {
  const v = fieldDef(name).value;
  // Type dispatch — keep this strict. Order matters: check shape (object vs.
  // array vs. boolean vs. number) BEFORE name-based heuristics, otherwise
  // misses like MAX_LOADED_MODELS routing to a list editor sneak in.
  if (typeof v === 'boolean') return boolEditor(name, v);
  if (typeof v === 'number') return numberEditor(name, v);
  if (name === 'DICTATION_MAP') return dictTableEditor(name, v || {});
  if (name === 'CHARACTER_REPLACEMENTS') return tupleListEditor(name, v || []);
  // Model-aware editors (must precede generic Array/list dispatch). Source
  // their options from the current ALLOWED_MODELS state — typing in the
  // allowlist textarea live-updates these.
  if (name === 'DEFAULT_MODEL') return modelDropdownEditor(name, v);
  if (name === 'PRELOAD_MODELS') return modelMultiSelectEditor(name, v);
  if (Array.isArray(v)) return linesEditor(name, v);
  // Empty/missing array-shaped fields fall through here; only force a list
  // editor when we know the field is a collection by name.
  if (name === 'ALLOWED_MODELS'
      || name === 'LOWERCASE_AFTER_STRIPPED_TERMINATOR'
      || name === 'ADMIN_ALLOWED_HOSTS' || name === 'STATS_ALLOWED_HOSTS') {
    return linesEditor(name, []);
  }
  if (name === 'SERVER_LOG_LEVEL') return selectEditor(name, v, ['debug','info','warning','error','critical']);
  if (name === 'MODEL_DEVICE' || name === 'MODEL_DEVICE_FALLBACK') return selectEditor(name, v, ['cuda','cpu']);
  if (name === 'MODEL_COMPUTE_TYPE' || name === 'MODEL_COMPUTE_TYPE_FALLBACK') {
    return selectEditor(name, v, ['float16','int8_float16','int8','float32','bfloat16']);
  }
  if (name === 'DEFAULT_PROMPT') return textareaEditor(name, v || '');
  // Numeric fields that can be null ("disabled"). Render as number input with
  // a "(disable)" button next to it so the user can clear → None.
  if (name === 'NO_SPEECH_THRESHOLD' || name === 'LOG_PROB_THRESHOLD'
      || name === 'COMPRESSION_RATIO_THRESHOLD') {
    return nullableNumberEditor(name, v);
  }
  return stringEditor(name, v == null ? '' : v);
}

function modelDropdownEditor(name, v) {
  // Single-select dropdown for DEFAULT_MODEL. Options come from the current
  // ALLOWED_MODELS (including unsaved edits in the textarea above), with a
  // "(preloaded)" suffix on entries that are also in PRELOAD_MODELS so the
  // user can see which choices avoid the cold-start cost. Re-renders on
  // any change to either list.
  //
  // Empty ALLOWED_MODELS means "any model passes" (per config.py): falls
  // back to a free-text input — a dropdown of nothing is useless.
  // A current value not in ALLOWED_MODELS is preserved as a "(custom)"
  // option so opening this page after editing the allowlist doesn't
  // silently drop a deliberate choice.
  const wrap = document.createElement('div');
  function render() {
    wrap.innerHTML = '';
    const allowed = Array.isArray(currentValue('ALLOWED_MODELS'))
      ? currentValue('ALLOWED_MODELS') : [];
    const preload = new Set(Array.isArray(currentValue('PRELOAD_MODELS'))
      ? currentValue('PRELOAD_MODELS') : []);
    const cur = currentValue(name) || '';

    if (allowed.length === 0) {
      const i = document.createElement('input');
      i.type = 'text'; i.value = cur;
      i.placeholder = 'any model id (ALLOWED_MODELS is empty)';
      i.addEventListener('input', () => setDirty(name, i.value));
      wrap.appendChild(i);
      const help = document.createElement('div');
      help.className = 'help';
      help.textContent = 'ALLOWED_MODELS is empty — free-form. Add entries above for a dropdown.';
      wrap.appendChild(help);
      return;
    }

    const sel = document.createElement('select');
    let curInList = false;
    if (!cur) {
      // Empty default — show a placeholder "Select model..." option.
      const ph = document.createElement('option');
      ph.value = ''; ph.textContent = '— select a model —';
      ph.selected = true; ph.disabled = true;
      sel.appendChild(ph);
    }
    for (const m of allowed) {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = preload.has(m) ? (m + '  (preloaded)') : m;
      if (m === cur) { opt.selected = true; curInList = true; }
      sel.appendChild(opt);
    }
    if (cur && !curInList) {
      const opt = document.createElement('option');
      opt.value = cur;
      opt.textContent = cur + '  (NOT in ALLOWED_MODELS — request will fail)';
      opt.selected = true;
      sel.insertBefore(opt, sel.firstChild);
    }
    sel.addEventListener('change', () => setDirty(name, sel.value));
    wrap.appendChild(sel);
  }
  render();
  document.addEventListener('admin:model-lists-changed', render);
  return wrap;
}

function modelMultiSelectEditor(name, v) {
  // Checkbox list for PRELOAD_MODELS. Universe = ALLOWED_MODELS ∪ current
  // PRELOAD entries (so a stale entry no longer in the allowlist still
  // shows, with a warning, instead of silently disappearing). Empty
  // ALLOWED_MODELS falls back to a textarea — same idea as the dropdown.
  const wrap = document.createElement('div');
  function render() {
    wrap.innerHTML = '';
    const allowed = Array.isArray(currentValue('ALLOWED_MODELS'))
      ? currentValue('ALLOWED_MODELS') : [];
    const preload = Array.isArray(currentValue(name))
      ? currentValue(name) : [];
    const preloadSet = new Set(preload);

    if (allowed.length === 0) {
      const t = document.createElement('textarea');
      t.value = preload.join('\n');
      t.rows = Math.min(Math.max(preload.length, 2), 8);
      t.placeholder = 'one model id per line';
      t.addEventListener('input', () => {
        const lines = t.value.split('\n').map(s => s.trim()).filter(Boolean);
        setDirty(name, lines);
      });
      wrap.appendChild(t);
      const help = document.createElement('div');
      help.className = 'help';
      help.textContent = 'ALLOWED_MODELS is empty — free-form. Add entries above for checkboxes.';
      wrap.appendChild(help);
      return;
    }

    // Universe = allowed + any stale preload entries (preserved so the user
    // can see + uncheck them deliberately).
    const universe = [...allowed];
    for (const p of preload) {
      if (!universe.includes(p)) universe.push(p);
    }

    const list = document.createElement('div');
    list.style.display = 'flex';
    list.style.flexDirection = 'column';
    list.style.gap = '4px';
    for (const m of universe) {
      const lbl = document.createElement('label');
      lbl.style.display = 'flex';
      lbl.style.gap = '6px';
      lbl.style.alignItems = 'center';
      lbl.style.cursor = 'pointer';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = preloadSet.has(m);
      cb.addEventListener('change', () => {
        const next = new Set(Array.isArray(currentValue(name)) ? currentValue(name) : []);
        if (cb.checked) next.add(m); else next.delete(m);
        setDirty(name, [...next]);
      });
      lbl.appendChild(cb);
      const txt = document.createElement('span');
      txt.textContent = m;
      if (!allowed.includes(m)) {
        const warn = document.createElement('em');
        warn.textContent = '  (not in ALLOWED_MODELS)';
        warn.style.color = '#f2cc60';
        warn.style.fontStyle = 'normal';
        txt.appendChild(warn);
      }
      lbl.appendChild(txt);
      list.appendChild(lbl);
    }
    wrap.appendChild(list);
    const help = document.createElement('div');
    help.className = 'help';
    help.textContent = 'preload at startup — first request to each is hot. '
      + 'keep <= MAX_LOADED_MODELS to avoid LRU evicting your own preloads.';
    wrap.appendChild(help);
  }
  render();
  document.addEventListener('admin:model-lists-changed', render);
  return wrap;
}

function nullableNumberEditor(name, v) {
  // Toggle between "active number" and "disabled (null)" states. When null:
  //   - input is HTMLDisabled (greyed out, can't focus or type)
  //   - button label flips to "enable" so the user knows it's a toggle
  // When non-null:
  //   - input is editable
  //   - button reads "disable" (sets value back to null)
  // Re-enabling restores the last-known number (or 0 if there was none).
  const wrap = document.createElement('span');
  wrap.className = 'nullable-wrap';
  wrap.style.display = 'flex'; wrap.style.gap = '6px'; wrap.style.alignItems = 'center';

  const i = document.createElement('input');
  i.type = 'number'; i.step = 'any';
  const btn = document.createElement('button');
  btn.style.padding = '2px 8px';

  // Last non-null value the user typed, used to restore on "enable" click.
  let lastVal = (v == null) ? 0 : v;

  function paint() {
    const cur = currentValue(name);
    const disabled = (cur == null);
    i.disabled = disabled;
    i.value = disabled ? '' : cur;
    i.placeholder = disabled ? '(disabled — null)' : '';
    btn.textContent = disabled ? 'enable' : 'disable';
    btn.title = disabled
      ? 'Restore numeric value (run this quality check again)'
      : 'Set to null (skip this quality check)';
  }

  i.addEventListener('input', () => {
    if (i.value === '') { setDirty(name, null); paint(); return; }
    const n = Number(i.value);
    if (!Number.isNaN(n)) { lastVal = n; setDirty(name, n); }
  });
  btn.addEventListener('click', () => {
    const cur = currentValue(name);
    if (cur == null) {
      // Re-enable with the last known value.
      setDirty(name, lastVal);
    } else {
      // Disable. Stash the current value so a later "enable" restores it.
      lastVal = cur;
      setDirty(name, null);
    }
    paint();
  });
  paint();
  wrap.appendChild(i);
  wrap.appendChild(btn);
  return wrap;
}

function stringEditor(name, v) {
  const i = document.createElement('input');
  i.type = 'text'; i.value = v;
  i.addEventListener('input', () => setDirty(name, i.value));
  return i;
}
function textareaEditor(name, v) {
  const t = document.createElement('textarea');
  t.value = v;
  t.rows = 4;
  t.addEventListener('input', () => setDirty(name, t.value));
  return t;
}
function numberEditor(name, v) {
  const i = document.createElement('input');
  i.type = 'number'; i.value = v;
  i.addEventListener('input', () => {
    const n = i.value === '' ? null : Number(i.value);
    if (n !== null && !Number.isNaN(n)) setDirty(name, n);
  });
  return i;
}
function boolEditor(name, v) {
  const i = document.createElement('input');
  i.type = 'checkbox'; i.checked = !!v;
  i.addEventListener('change', () => setDirty(name, i.checked));
  return i;
}
function selectEditor(name, v, opts) {
  const s = document.createElement('select');
  for (const o of opts) {
    const opt = document.createElement('option');
    opt.value = o; opt.textContent = o;
    if (o === v) opt.selected = true;
    s.appendChild(opt);
  }
  s.addEventListener('change', () => setDirty(name, s.value));
  return s;
}
function linesEditor(name, v) {
  const t = document.createElement('textarea');
  t.value = v.join('\n');
  t.rows = Math.min(Math.max(v.length, 2), 10);
  t.placeholder = 'one entry per line';
  const help = document.createElement('div');
  help.className = 'help';
  help.textContent = 'one entry per line. blank lines ignored.';
  const update = () => {
    const lines = t.value.split('\n').map(s => s.trim()).filter(Boolean);
    setDirty(name, lines);
  };
  t.addEventListener('input', update);
  const wrap = document.createElement('div');
  wrap.appendChild(t); wrap.appendChild(help);
  return wrap;
}
function tupleListEditor(name, v) {
  // Simple table for [from,to] string pairs.
  const wrap = document.createElement('div');
  const rows = v.map(p => Array.isArray(p) ? [...p] : [p, '']);
  const tbl = document.createElement('table');
  tbl.className = 'dict';
  tbl.innerHTML = '<thead><tr><th>from</th><th>to</th><th></th></tr></thead>';
  const body = document.createElement('tbody');
  tbl.appendChild(body);
  function emit() {
    const pairs = [];
    for (const tr of body.children) {
      const a = tr.children[0].firstChild.value;
      const b = tr.children[1].firstChild.value;
      if (a !== '' || b !== '') pairs.push([a, b]);
    }
    setDirty(name, pairs);
  }
  function addRow(a, b) {
    const tr = document.createElement('tr');
    for (const cell of [a, b]) {
      const td = document.createElement('td');
      const i = document.createElement('input');
      i.type = 'text'; i.value = cell;
      i.addEventListener('input', emit);
      td.appendChild(i);
      tr.appendChild(td);
    }
    const td = document.createElement('td');
    const del = document.createElement('button');
    del.className = 'del'; del.textContent = '×';
    del.addEventListener('click', () => { tr.remove(); emit(); });
    td.appendChild(del);
    tr.appendChild(td);
    body.appendChild(tr);
  }
  for (const [a, b] of rows) addRow(a, b);
  wrap.appendChild(tbl);
  const addWrap = document.createElement('div');
  addWrap.className = 'add-row';
  const add = document.createElement('button');
  add.textContent = '+ add';
  add.addEventListener('click', () => { addRow('', ''); });
  addWrap.appendChild(add);
  wrap.appendChild(addWrap);
  return wrap;
}
// `<input type="text">` strips newlines on get/set, so values like "\n" or
// "\n\n" (DICTATION_MAP entries for "neue Zeile" / "neuer Absatz") render
// as blank cells and would be silently lost on save. We round-trip control
// characters as escape sequences so the user can see and edit them. Order
// matters in escapeForInput: backslash first so we don't double-escape.
function escapeForInput(s) {
  return String(s).replace(/\\/g, '\\\\')
                  .replace(/\n/g, '\\n')
                  .replace(/\r/g, '\\r')
                  .replace(/\t/g, '\\t');
}
function unescapeFromInput(s) {
  // Single-pass so "\\n" round-trips to "\n" (literal backslash + n), not a newline.
  return String(s).replace(/\\([nrt\\])/g, (_, c) => (
    { n: '\n', r: '\r', t: '\t', '\\': '\\' }[c]
  ));
}

function dictTableEditor(name, dict) {
  const wrap = document.createElement('div');
  const tbl = document.createElement('table');
  tbl.className = 'dict';
  tbl.innerHTML = '<thead><tr><th>spoken word</th><th>symbol</th><th></th></tr></thead>';
  const body = document.createElement('tbody');
  tbl.appendChild(body);
  function emit() {
    const out = {};
    for (const tr of body.children) {
      const k = tr.children[0].firstChild.value.trim();
      const v = unescapeFromInput(tr.children[1].firstChild.value);
      if (k) out[k] = v;
    }
    setDirty(name, out);
  }
  function addRow(k, v) {
    const tr = document.createElement('tr');
    const td1 = document.createElement('td');
    const i1 = document.createElement('input');
    i1.type = 'text'; i1.value = k; i1.placeholder = 'e.g. Punkt';
    i1.addEventListener('input', emit);
    td1.appendChild(i1);
    tr.appendChild(td1);
    const td2 = document.createElement('td');
    const i2 = document.createElement('input');
    i2.type = 'text';
    i2.value = escapeForInput(v);
    i2.placeholder = 'e.g. .  (use \\n for newline, \\t for tab)';
    i2.addEventListener('input', emit);
    td2.appendChild(i2);
    tr.appendChild(td2);
    const td3 = document.createElement('td');
    const del = document.createElement('button');
    del.className = 'del'; del.textContent = '×';
    del.addEventListener('click', () => { tr.remove(); emit(); });
    td3.appendChild(del);
    tr.appendChild(td3);
    body.appendChild(tr);
  }
  for (const k of Object.keys(dict)) addRow(k, dict[k]);
  wrap.appendChild(tbl);
  const help = document.createElement('div');
  help.className = 'help';
  help.textContent = 'Symbol column: use \\n for newline, \\n\\n for paragraph break, '
    + '\\t for tab, \\\\ for a literal backslash.';
  wrap.appendChild(help);
  const addWrap = document.createElement('div');
  addWrap.className = 'add-row';
  const add = document.createElement('button');
  add.textContent = '+ add row';
  add.addEventListener('click', () => { addRow('', ''); });
  addWrap.appendChild(add);
  wrap.appendChild(addWrap);
  return wrap;
}

function render() {
  const main = $('main');
  main.innerHTML = '';
  for (const g of state.groups) {
    const sec = document.createElement('section');
    const h = document.createElement('h2');
    h.textContent = g.title;
    sec.appendChild(h);
    for (const fname of g.fields) {
      try {
        sec.appendChild(fieldRow(fname));
      } catch (err) {
        console.error('failed to render field', fname, err);
        const errRow = document.createElement('div');
        errRow.className = 'field';
        errRow.innerHTML = '<div class="label-col"><div class="name">' + fname
          + '</div></div><div class="input-col"><div class="err">'
          + 'render failed: ' + (err.message || err) + '</div></div>';
        sec.appendChild(errRow);
      }
    }
    main.appendChild(sec);
  }
  // Run dependency dimming once after the form is built; subsequent updates
  // are driven by setDirty().
  applyFieldDependencies();
}

async function save() {
  if (Object.keys(dirty).length === 0) return;
  const r = await api('POST', '/config/state', dirty);
  if (r.status === 422) {
    const j = await r.json();
    const msg = (j.errors || [])
      .map(e => e.loc + ': ' + e.msg).join('  /  ');
    toast('validation: ' + msg, true);
    return;
  }
  if (!r.ok) {
    toast('save failed: ' + r.status, true);
    return;
  }
  const result = await r.json();
  dirty = {};
  $('save-btn').disabled = true;

  if (result.requires_restart && result.cold_pending.length > 0) {
    showRestartModal(result.cold_pending);
  } else {
    toast('saved ' + result.saved.length + ' field(s); ' +
          result.hot_applied.length + ' applied live');
    await loadState();
  }
}

function showRestartModal(fields, opts) {
  opts = opts || {};
  $('restart-modal-title').textContent = opts.title || 'Restart required';
  $('restart-modal-body').textContent = opts.body
    || 'These changes need a service restart to take effect:';
  const ul = $('restart-fields');
  ul.innerHTML = '';
  if (fields && fields.length) {
    ul.style.display = '';
    for (const f of fields) {
      const li = document.createElement('li');
      li.textContent = f;
      ul.appendChild(li);
    }
  } else {
    ul.style.display = 'none';
  }
  $('restart-modal').classList.add('show');
}

async function doRestart() {
  $('restart-modal').classList.remove('show');
  $('restart-progress').classList.add('show');
  $('restart-progress-msg').textContent = 'Asking the server to spawn the restart helper.';
  const r = await api('POST', '/config/restart', {});
  if (!r.ok) {
    $('restart-progress').classList.remove('show');
    let detail = r.status;
    try { detail = (await r.json()).detail || detail; } catch {}
    toast('restart failed: ' + detail, true);
    return;
  }
  const result = await r.json();
  $('restart-progress-msg').textContent =
    'Helper spawned (' + (result.method || 'unknown method') + '). '
    + 'Service will stop in ' + (result.delay_sec || 3) + ' s.';

  // Poll /v1/models. We need to see the service genuinely go DOWN first
  // (failed fetch), then come back UP. Until we see a down poll we never
  // declare success — otherwise an unchanged service satisfies the check.
  // Deadline is generous because PRELOAD_MODELS blocks uvicorn's lifespan
  // startup — N preloads × ~5-10 s per large-v3 = up to several minutes.
  const RESTART_TIMEOUT_MS = 5 * 60 * 1000;
  const deadline = Date.now() + RESTART_TIMEOUT_MS;
  let sawDown = false;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 1000));
    try {
      const m = await fetch('/v1/models', { cache: 'no-store' });
      if (m.ok && sawDown) {
        $('restart-progress').classList.remove('show');
        toast('service is back; reloading');
        setTimeout(() => location.reload(), 600);
        return;
      }
      if (!m.ok) sawDown = true;
    } catch {
      sawDown = true;
      $('restart-progress-msg').textContent =
        'Service is down; waiting for it to come back '
        + '(may take a few minutes if PRELOAD_MODELS is large).';
    }
  }
  $('restart-progress-msg').textContent =
    "Service didn't come back within " + (RESTART_TIMEOUT_MS / 60000) + " min. "
    + "Check `Get-Service WhisperAPI` and the log viewer at /logs.";
}

document.addEventListener('DOMContentLoaded', async () => {
  $('login-btn').addEventListener('click', async () => {
    const t = $('login-token').value.trim();
    if (!t) return;
    sessionStorage.setItem(TOKEN_KEY, t);
    showApp();
    await loadState();
  });
  $('login-token').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') $('login-btn').click();
  });
  $('logout-btn').addEventListener('click', () => {
    sessionStorage.removeItem(TOKEN_KEY);
    showLogin();
  });
  $('reload-btn').addEventListener('click', loadState);
  $('save-btn').addEventListener('click', save);
  $('restart-btn').addEventListener('click', () => {
    // Manual-restart entry point. Same flow as the post-save modal but with
    // generic copy and no field list — useful when an earlier save's cold
    // changes never got applied (e.g., the auto-restart flow misfired).
    if (Object.keys(dirty).length > 0) {
      if (!confirm('You have unsaved changes that will be lost on restart. Continue?')) {
        return;
      }
    }
    showRestartModal(null, {
      title: 'Restart service',
      body: 'Restart the WhisperAPI service now? Any pending overrides in '
        + 'config.local.json will be applied on next start.',
    });
  });
  $('restart-cancel').addEventListener('click',
    () => $('restart-modal').classList.remove('show'));
  $('restart-now').addEventListener('click', doRestart);

  // Probe the state endpoint to figure out whether a token is required.
  const probe = await fetch('/config/state', {
    headers: authHeaders(),
    cache: 'no-store',
  });
  if (probe.status === 401) {
    showLogin('Bearer token required.');
    return;
  }
  if (!probe.ok) {
    document.body.innerHTML = '<main style="padding:20px;color:#ff7b72">'
      + 'Could not load /config/state (' + probe.status + '). '
      + 'Check service logs.</main>';
    return;
  }
  showApp();
  await loadState();
});

})();
</script>
</body></html>"""
