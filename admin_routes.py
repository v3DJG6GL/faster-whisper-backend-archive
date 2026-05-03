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
# Section groups: each section can have one or more SUB-groups. A subgroup
# title of None means "no subheader" — fields render directly under the
# section. Section titles mirror the per-request log block phases (Decode
# params / Pipeline / …) so an operator reading a log can find the matching
# config knobs by section name with no translation.
_FIELD_GROUPS: list[tuple[str, list[tuple[str | None, list[str]]]]] = [
    ("Models", [(None, [
        "DEFAULT_MODEL", "ALLOWED_MODELS", "PRELOAD_MODELS", "MAX_LOADED_MODELS",
        "MODEL_DEVICE", "MODEL_COMPUTE_TYPE",
        "MODEL_DEVICE_FALLBACK", "MODEL_COMPUTE_TYPE_FALLBACK",
    ])]),
    ("Decode params", [(None, [
        "DEFAULT_LANGUAGE", "DEFAULT_PROMPT",
        "BEAM_SIZE", "BEST_OF",
        "VAD_FILTER", "VAD_MIN_SILENCE_MS", "VAD_SPEECH_PAD_MS", "VAD_THRESHOLD",
        "CONDITION_ON_PREVIOUS_TEXT", "WORD_TIMESTAMPS_ENABLED",
        "NO_SPEECH_THRESHOLD", "LOG_PROB_THRESHOLD", "COMPRESSION_RATIO_THRESHOLD",
    ])]),
    ("Pipeline", [
        ("Step 0 — character replacements", ["CHARACTER_REPLACEMENTS"]),
        ("Step 1 — punctuation strip",      ["PUNCTUATION_TO_KEEP"]),
        ("Step 3 — Whisper noise strip",    [
            "STRIP_REGEX_DISABLE",
            "STRIP_AND_LOWERCASE_REGEX",
            "STRIP_AND_LOWERCASE_WORDS",
            "STRIP_ONLY_REGEX",
        ]),
        ("Steps 4-8 — dictation pipeline",  [
            "DICTATION_ENABLED", "DICTATION_MAP", "TRACE_ENABLED",
        ]),
    ]),
    ("Logging", [(None, [
        "LOG_FILE", "LOG_MAX_BYTES", "LOG_BACKUP_COUNT",
    ])]),
    ("Server (uvicorn)", [(None, [
        "SERVER_HOST", "SERVER_PORT", "SERVER_WORKERS", "SERVER_LOG_LEVEL",
    ])]),
    ("Access (allowlists)", [(None, [
        "ADMIN_ALLOWED_HOSTS", "STATS_ALLOWED_HOSTS",
    ])]),
]


def _all_fields() -> list[str]:
    """Flat list of every field name across all sections + subgroups, in
    display order. Used by the /state endpoint and post_state echo paths."""
    out: list[str] = []
    for _section, subs in _FIELD_GROUPS:
        for _sub_title, names in subs:
            out.extend(names)
    return out


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
    baseline = getattr(cfg, "_BASELINE", {}) or {}
    field_descs = config_store.FIELD_DESCRIPTIONS
    pyd_fields = config_store.AdminConfig.model_fields

    def _baseline_value(name: str) -> Any:
        # Used by the WebUI's "↺ Reset" button. Returns the in-repo default
        # captured in cfg._BASELINE before local.json + env overrides apply.
        # Convert non-JSON-serializable types (set, frozenset, tuple of
        # tuples) the same way _resolved_value does so the round-trip is clean.
        v = baseline.get(name)
        if isinstance(v, (set, frozenset)):
            return sorted(v)
        if isinstance(v, tuple):
            return [list(p) if isinstance(p, tuple) else p for p in v]
        return v

    fields: dict[str, dict[str, Any]] = {}
    for name in _all_fields():
        # Description preference: Pydantic schema > FIELD_DESCRIPTIONS dict
        # (they're the same string in practice; schema wins so reload picks
        # up live edits to FIELD_DESCRIPTIONS without a service restart).
        desc = ""
        if name in pyd_fields and pyd_fields[name].description:
            desc = pyd_fields[name].description
        elif name in field_descs:
            desc = field_descs[name]
        fields[name] = {
            "value": _resolved_value(name),
            "default_value": _baseline_value(name),
            "description": desc,
            "provenance": _provenance(name, env_pinned, saved),
            "env_var": env_pinned.get(name),
            "restart_required": name in config_store.RESTART_REQUIRED_FIELDS,
        }

    # Surface the nested group structure to the client. Each group has a list
    # of subgroups: {title, subgroups: [{title: str | None, fields: [...]}]}.
    groups_payload = [
        {
            "title": section,
            "subgroups": [
                {"title": sub_title, "fields": names}
                for sub_title, names in subs
            ],
        }
        for section, subs in _FIELD_GROUPS
    ]

    return {
        "fields": fields,
        "groups": groups_payload,
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


@router.post("/test-regex",
             dependencies=[Depends(require_admin_host), Depends(require_admin_token)])
async def test_regex(payload: dict[str, Any]) -> JSONResponse:
    """Validate + dry-run the Step-3 regex pair against a sample.

    Used by the WebUI's regex-editor live-validation badge AND the inline
    test panel. Each non-empty regex is compiled and run under a 2 s
    timeout against the supplied sample; the response carries per-pass
    diff data so the UI can highlight matches and show the final result.

    Payload shape: { sample: str, regex_a: str, regex_b: str, words: list[str] }
    Response shape: {
      pass_a: { compiled, error, matches, slow, result, lowercased },
      pass_b: { compiled, error, matches, slow,  result },
      final: str,
    }
    """
    import threading

    sample = str(payload.get("sample") or "")
    regex_a = str(payload.get("regex_a") or "")
    regex_b = str(payload.get("regex_b") or "")
    words = {str(w).lower() for w in (payload.get("words") or [])}

    def _compile_and_run(pattern: str, replacer):
        """Returns dict with compiled / error / matches / slow / result."""
        if not pattern:
            return {"compiled": False, "error": None, "matches": [],
                    "slow": False, "result": sample, "skipped": True}
        try:
            cre = re.compile(pattern)
        except re.error as e:
            return {"compiled": False, "error": str(e), "matches": [],
                    "slow": False, "result": sample, "skipped": False}

        # Timeout-guarded run on a worker thread. The `re` module has no
        # native timeout, so we just don't wait past 2 s — the work itself
        # may continue in the daemon thread but we return early.
        out: dict[str, Any] = {"done": False, "matches": [], "result": sample,
                               "lowercased": []}
        def _run() -> None:
            try:
                # Collect match positions for highlighting before sub() runs.
                out["matches"] = [
                    {"start": m.start(), "end": m.end(), "text": m.group(0)}
                    for m in cre.finditer(sample)
                ]
                if replacer is None:
                    out["result"] = cre.sub("", sample)
                else:
                    lc: list[str] = []
                    def _wrap(m: "re.Match[str]") -> str:
                        return replacer(m, lc)
                    out["result"] = cre.sub(_wrap, sample)
                    out["lowercased"] = lc
                out["done"] = True
            except Exception as e:    # noqa: BLE001 — surface any error
                out["error"] = str(e)
                out["done"] = True
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=2.0)
        if not out["done"]:
            return {"compiled": True, "error": None, "matches": [],
                    "slow": True, "result": sample, "skipped": False,
                    "timeout": "regex did not complete in 2 s on the sample"}
        return {"compiled": True, "error": out.get("error"),
                "matches": out["matches"], "slow": False,
                "result": out["result"], "skipped": False,
                "lowercased": out.get("lowercased", [])}

    # Pass A replacer: strip terminator, conditionally lowercase next word.
    # Mirrors main.py's _step3_pass_a.replace exactly.
    def _pass_a_replacer(m: "re.Match[str]", lc_log: list[str]) -> str:
        try:
            ws, first, rest = m.group(1), m.group(2), m.group(3)
        except IndexError:
            # Custom regex didn't produce 3 groups — degrade to plain strip.
            return ""
        if (first + rest).lower() in words:
            lc_log.append(first + rest)
            return ws + first.lower() + rest
        return ws + first + rest

    pa = _compile_and_run(regex_a, _pass_a_replacer)
    pb = _compile_and_run(regex_b, None)
    return JSONResponse({
        "pass_a": pa,
        "pass_b": pb,
        "final": pb.get("result") if pb.get("compiled") and not pb.get("error") else pa.get("result"),
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
  /* Subgroup heading inside a section: smaller than h2, lighter weight,
     small dividing line so it's visibly distinct from the section header
     but doesn't draw the eye like a top-level section change. */
  h3.subgroup { color: var(--dim); font-size: 12px; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 14px 0 6px 0; padding-bottom: 3px;
    border-bottom: 1px solid var(--border); }
  /* Reset-to-default link button — small, italic, only visible when the
     current value differs from the in-repo default. Sits below the help
     text, so it doesn't crowd the editor itself. */
  .reset-wrap { margin-top: 4px; }
  .reset-link { background: none; border: none; padding: 0;
    color: var(--cyan); cursor: pointer; font: inherit; font-size: 11px;
    font-style: italic; text-decoration: underline; text-underline-offset: 2px; }
  .reset-link:hover { color: var(--bold); }
  /* Regex editor + status badge */
  .regex-wrap { display: flex; flex-direction: column; gap: 4px; }
  .regex-status { font-size: 11px; font-family: ui-monospace, Menlo, monospace; }
  .regex-status.ok { color: var(--green); }
  .regex-status.err { color: var(--red); }
  .regex-status.warn { color: var(--yellow); }
  .regex-status.empty { color: var(--dim); font-style: italic; }
  /* Advanced warning banner above the Step 3 fields */
  .advanced-warn { background: #2d1f0a; color: #f2cc60; border-left: 3px solid #f2cc60;
    padding: 6px 10px; margin: 8px 0; border-radius: 3px; font-size: 12px; }
  /* Test panel */
  .regex-test-panel { background: #161b22; border: 1px solid var(--border);
    border-radius: 4px; padding: 10px 12px; margin: 8px 0 14px 0; }
  .regex-test-out { margin-top: 10px; }
  .regex-test-pass { margin: 6px 0; }
  .regex-test-head { font-family: ui-monospace, Menlo, monospace; font-size: 12px;
    color: var(--dim); }
  .regex-test-head .tag { display: inline-block; padding: 0 6px; margin-left: 6px;
    border-radius: 3px; font-size: 11px; }
  .regex-test-head .tag.ok { background: #033a16; color: #7ee787; }
  .regex-test-head .tag.err { background: #3a0d0d; color: #ff7b72; }
  .regex-test-head .tag.warn { background: #2d1f0a; color: #f2cc60; }
  .regex-test-head .tag.empty { background: #21262d; color: var(--dim); font-style: italic; }
  .regex-test-result { background: #0d1117; border: 1px solid var(--border);
    padding: 4px 8px; border-radius: 3px; margin: 4px 0;
    font-family: ui-monospace, Menlo, monospace; font-size: 12px;
    white-space: pre-wrap; word-break: break-word; max-height: 120px; overflow: auto; }
  .regex-test-final { margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border); }
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
  // Notify per-field listeners (currently the ↺ Reset button) so they can
  // refresh their "value differs from default?" display.
  document.dispatchEvent(new CustomEvent('admin:dirty', { detail: { name } }));
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
  STRIP_AND_LOWERCASE_REGEX: {
    irrelevant: () => currentValue('STRIP_REGEX_DISABLE') === true,
    note: 'unused while STRIP_REGEX_DISABLE is on (both Step 3 passes skipped)',
  },
  STRIP_AND_LOWERCASE_WORDS: {
    irrelevant: () =>
      currentValue('STRIP_REGEX_DISABLE') === true
      || currentValue('STRIP_AND_LOWERCASE_REGEX') === '',
    note: 'unused when STRIP_REGEX_DISABLE is on or STRIP_AND_LOWERCASE_REGEX is empty (Pass A skipped)',
  },
  STRIP_ONLY_REGEX: {
    irrelevant: () => currentValue('STRIP_REGEX_DISABLE') === true,
    note: 'unused while STRIP_REGEX_DISABLE is on (both Step 3 passes skipped)',
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

  // Inline description from FIELD_DESCRIPTIONS (single source of truth).
  // Surfaced via /config/state's per-field payload.
  const desc = fieldDef(name).description;
  if (desc) {
    const help = document.createElement('div');
    help.className = 'help';
    help.textContent = desc;
    inputCol.appendChild(help);
  }

  // ↺ Reset link button — appears whenever the current value differs from
  // the in-repo baseline (cfg._BASELINE). Clicking sets the field to the
  // baseline value, marking it dirty so the save button enables. Recovery
  // path for "I broke a regex / typed the wrong number".
  const resetWrap = document.createElement('div');
  resetWrap.className = 'reset-wrap';
  const resetBtn = document.createElement('button');
  resetBtn.type = 'button';
  resetBtn.className = 'reset-link';
  resetBtn.textContent = '↺ Reset to default';
  resetBtn.title = 'Restore the in-repo default value';
  resetBtn.addEventListener('click', () => {
    const def = fieldDef(name).default_value;
    setDirty(name, def);
    // Re-render this row so the editor reflects the new value.
    const newRow = fieldRow(name);
    row.replaceWith(newRow);
  });
  resetWrap.appendChild(resetBtn);
  inputCol.appendChild(resetWrap);
  // Toggle reset visibility on every input event by checking dirty + current.
  function refreshReset() {
    const cur = currentValue(name);
    const def = fieldDef(name).default_value;
    const same = JSON.stringify(cur) === JSON.stringify(def);
    resetWrap.style.display = same ? 'none' : '';
  }
  refreshReset();
  // Subscribe to dirty changes for this field via a custom event we'll fire
  // from setDirty(). Simpler than wiring per-editor change listeners.
  document.addEventListener('admin:dirty', (e) => {
    if (!e.detail || e.detail.name === name) refreshReset();
  });

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
  // Regex fields — special editor with live validation badge + per-step
  // test panel (rendered once per "Whisper noise strip" subgroup).
  if (name === 'STRIP_AND_LOWERCASE_REGEX' || name === 'STRIP_ONLY_REGEX') {
    return regexEditor(name, v == null ? '' : v);
  }
  if (Array.isArray(v)) return linesEditor(name, v);
  // Empty/missing array-shaped fields fall through here; only force a list
  // editor when we know the field is a collection by name.
  if (name === 'ALLOWED_MODELS'
      || name === 'STRIP_AND_LOWERCASE_WORDS'
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

function regexEditor(name, v) {
  // Mono-font input with a live status badge below it.
  // Validation states:
  //   ✓ valid · matches N in sample · M chars   (green)
  //   ✗ <re.error>                              (red, blocks save via 422 on save)
  //   ⚠ slow (timed out at 2 s)                 (yellow, allows save with warn)
  //   ∅ empty — pass skipped                    (grey, italic)
  // Status updates on every keystroke via debounced POST /config/test-regex.
  const wrap = document.createElement('div');
  wrap.className = 'regex-wrap';

  const ta = document.createElement('input');
  ta.type = 'text';
  ta.spellcheck = false;
  ta.autocomplete = 'off';
  ta.value = v || '';
  ta.style.fontFamily = 'ui-monospace, Menlo, Consolas, monospace';
  ta.style.fontSize = '12px';
  ta.style.width = '100%';
  ta.style.boxSizing = 'border-box';

  const status = document.createElement('div');
  status.className = 'regex-status';
  status.textContent = '∅ empty — pass skipped';

  // Debounced live test against the default sample (the test panel
  // overrides the sample if open). 250 ms is comfortable for typing.
  let timer = null;
  async function refreshStatus() {
    if (timer) clearTimeout(timer);
    timer = setTimeout(async () => {
      const cur = currentValue(name) || '';
      if (!cur) {
        status.className = 'regex-status empty';
        status.textContent = '∅ empty — pass skipped';
        return;
      }
      // Use the test panel's current sample if visible, else the default.
      const panelSample = document.getElementById('regex-test-sample');
      const sample = panelSample ? panelSample.value : DEFAULT_REGEX_SAMPLE;
      const r = await api('POST', '/config/test-regex', {
        sample,
        regex_a: name === 'STRIP_AND_LOWERCASE_REGEX' ? cur : '',
        regex_b: name === 'STRIP_ONLY_REGEX'          ? cur : '',
        words: currentValue('STRIP_AND_LOWERCASE_WORDS') || [],
      });
      if (!r.ok) {
        status.className = 'regex-status err';
        status.textContent = '✗ test endpoint error';
        return;
      }
      const j = await r.json();
      const pass = name === 'STRIP_AND_LOWERCASE_REGEX' ? j.pass_a : j.pass_b;
      if (pass.error) {
        status.className = 'regex-status err';
        status.textContent = '✗ ' + pass.error;
      } else if (pass.slow) {
        status.className = 'regex-status warn';
        status.textContent = '⚠ slow — exceeded 2 s on sample (catastrophic backtracking?)';
      } else {
        status.className = 'regex-status ok';
        const n = (pass.matches || []).length;
        status.textContent = '✓ valid · ' + n + ' match' + (n === 1 ? '' : 'es')
                             + ' in sample · ' + cur.length + ' chars';
      }
    }, 250);
  }

  ta.addEventListener('input', () => {
    setDirty(name, ta.value);
    refreshStatus();
  });
  // Refresh once on initial render (so the badge isn't blank).
  // Run after the row is in the DOM — defer with rAF.
  requestAnimationFrame(refreshStatus);

  wrap.appendChild(ta);
  wrap.appendChild(status);
  return wrap;
}

const DEFAULT_REGEX_SAMPLE =
  "Hallo. Wie geht's? 10.23 Uhr! Bitte Frau, Müller. neuer Absatz. 1,000 EUR.";

function regexTestPanel() {
  // One panel per Step-3 subgroup. Inserted once via the toggle in the
  // <h3 class="subgroup"> banner. Editable sample, run-against-both-passes
  // button, per-pass diff render, final output row.
  const wrap = document.createElement('div');
  wrap.className = 'regex-test-panel';

  const sampleLbl = document.createElement('div');
  sampleLbl.className = 'help';
  sampleLbl.textContent = 'Test sample (edit to try your own):';
  wrap.appendChild(sampleLbl);

  const sample = document.createElement('textarea');
  sample.id = 'regex-test-sample';
  sample.value = DEFAULT_REGEX_SAMPLE;
  sample.rows = 2;
  sample.style.width = '100%';
  sample.style.boxSizing = 'border-box';
  sample.style.fontFamily = 'ui-monospace, Menlo, Consolas, monospace';
  sample.style.fontSize = '12px';
  wrap.appendChild(sample);

  const runBtn = document.createElement('button');
  runBtn.type = 'button';
  runBtn.textContent = 'Run test';
  runBtn.style.marginTop = '6px';
  wrap.appendChild(runBtn);

  const out = document.createElement('div');
  out.className = 'regex-test-out';
  wrap.appendChild(out);

  async function run() {
    out.innerHTML = '<em>running…</em>';
    const r = await api('POST', '/config/test-regex', {
      sample: sample.value,
      regex_a: currentValue('STRIP_AND_LOWERCASE_REGEX') || '',
      regex_b: currentValue('STRIP_ONLY_REGEX') || '',
      words: currentValue('STRIP_AND_LOWERCASE_WORDS') || [],
    });
    if (!r.ok) { out.innerHTML = '<em class="err">test endpoint error</em>'; return; }
    const j = await r.json();
    const block = (label, pass) => {
      const p = document.createElement('div');
      p.className = 'regex-test-pass';
      const head = document.createElement('div');
      head.className = 'regex-test-head';
      head.textContent = label + ': ';
      const tag = document.createElement('span');
      if (pass.skipped) {
        tag.className = 'tag empty'; tag.textContent = 'skipped';
      } else if (pass.error) {
        tag.className = 'tag err'; tag.textContent = '✗ ' + pass.error;
      } else if (pass.slow) {
        tag.className = 'tag warn'; tag.textContent = '⚠ slow';
      } else {
        tag.className = 'tag ok'; tag.textContent = '✓ ' + (pass.matches || []).length + ' matches';
      }
      head.appendChild(tag);
      p.appendChild(head);

      if (!pass.skipped && !pass.error) {
        const res = document.createElement('pre');
        res.className = 'regex-test-result';
        res.textContent = pass.result;
        p.appendChild(res);
        if (pass.lowercased && pass.lowercased.length) {
          const lc = document.createElement('div');
          lc.className = 'help';
          lc.textContent = 'lowercased: ' + pass.lowercased.join(', ');
          p.appendChild(lc);
        }
      }
      return p;
    };
    out.innerHTML = '';
    out.appendChild(block('Pass A', j.pass_a));
    out.appendChild(block('Pass B', j.pass_b));
    const finalRow = document.createElement('div');
    finalRow.className = 'regex-test-final';
    finalRow.innerHTML = '<strong>Final →</strong> ';
    const fpre = document.createElement('pre');
    fpre.className = 'regex-test-result';
    fpre.textContent = j.final;
    finalRow.appendChild(fpre);
    out.appendChild(finalRow);
  }
  runBtn.addEventListener('click', run);
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
    // Each group now has subgroups; iterate them. A subgroup with title===null
    // emits no subheader (back-compat with old single-list layout).
    for (const sub of (g.subgroups || [{ title: null, fields: g.fields || [] }])) {
      if (sub.title) {
        const h3 = document.createElement('h3');
        h3.className = 'subgroup';
        h3.textContent = sub.title;
        sec.appendChild(h3);
        // Step 3 subgroup gets the regex test panel + advanced badge.
        if (/Step 3/.test(sub.title)) {
          const adv = document.createElement('div');
          adv.className = 'advanced-warn';
          adv.innerHTML = '⚠ <strong>advanced</strong> — incorrect regex breaks transcription. '
            + 'Use the test panel below to dry-run before saving. ↺ Reset to default if you get stuck.';
          sec.appendChild(adv);
          sec.appendChild(regexTestPanel());
        }
      }
      for (const fname of sub.fields) {
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
