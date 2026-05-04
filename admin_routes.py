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
import os
import re
import secrets
import time
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
    ("Models", [
        (None, [
            "DEFAULT_MODEL", "ALLOWED_MODELS", "PRELOAD_MODELS",
            "MAX_LOADED_MODELS", "MODEL_IDLE_TIMEOUT_S",
            "MODEL_DEVICE", "MODEL_COMPUTE_TYPE",
            "MODEL_DEVICE_FALLBACK", "MODEL_COMPUTE_TYPE_FALLBACK",
        ]),
        ("Advanced — load-time hardware", [
            "DOWNLOAD_ROOT", "LOCAL_FILES_ONLY", "USE_AUTH_TOKEN",
            "CPU_THREADS", "NUM_WORKERS", "DEVICE_INDEX",
        ]),
    ]),
    ("Decode params", [
        (None, [
            "DEFAULT_LANGUAGE", "DEFAULT_PROMPT", "DEFAULT_HOTWORDS",
            "BEAM_SIZE", "BEST_OF",
            "VAD_FILTER", "VAD_MIN_SILENCE_MS", "VAD_SPEECH_PAD_MS", "VAD_THRESHOLD",
            "CONDITION_ON_PREVIOUS_TEXT", "WORD_TIMESTAMPS_ENABLED",
            "NO_SPEECH_THRESHOLD", "LOG_PROB_THRESHOLD", "COMPRESSION_RATIO_THRESHOLD",
        ]),
        ("Advanced — beam & sampling", [
            "TEMPERATURE", "PATIENCE", "LENGTH_PENALTY",
            "REPETITION_PENALTY", "NO_REPEAT_NGRAM_SIZE",
            "PROMPT_RESET_ON_TEMPERATURE",
        ]),
        ("Advanced — language detection (active when DEFAULT_LANGUAGE empty)", [
            "MULTILINGUAL", "LANGUAGE_DETECTION_THRESHOLD",
            "LANGUAGE_DETECTION_SEGMENTS",
        ]),
        ("Advanced — anti-hallucination & token control", [
            "HALLUCINATION_SILENCE_THRESHOLD", "SUPPRESS_BLANK", "SUPPRESS_TOKENS",
            "PREPEND_PUNCTUATIONS", "APPEND_PUNCTUATIONS",
        ]),
    ]),
    ("Output wrappers", [(None, [
        "OUTPUT_PREFIX", "OUTPUT_SUFFIX",
    ])]),
    ("Per-model overrides", [(None, ["MODEL_OVERRIDES"])]),
    ("Pipeline", [(None, ["PIPELINE_RULES"])]),
    ("Logging", [(None, [
        "LOG_FILE", "LOG_MAX_BYTES", "LOG_BACKUP_COUNT", "TRACE_ENABLED",
    ])]),
    ("Server (uvicorn)", [(None, [
        "SERVER_HOST", "SERVER_PORT", "SERVER_WORKERS", "SERVER_LOG_LEVEL",
    ])]),
    ("Access (allowlists + token)", [(None, [
        "ADMIN_ALLOWED_HOSTS", "STATS_ALLOWED_HOSTS", "ADMIN_TOKEN",
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

# --- Token rotation with 60 s grace window ----------------------------------
#
# When an admin rotates ADMIN_TOKEN (or clears it), the previous value stays
# valid for 60 s so the editing session can update its stored token without
# getting locked out mid-flight. After the grace window expires, only the
# current cfg.ADMIN_TOKEN is accepted. Loopback bypass is unaffected — the
# loopback caller can always edit/clear the token without auth.
_TOKEN_GRACE_S = 60.0
_previous_token: "str | None" = None
_previous_token_expires_at: float = 0.0   # time.monotonic()


def _record_previous_token(old: "str | None") -> None:
    """Stash the pre-rotate token + expiry. Called from post_state when
    ADMIN_TOKEN changes. Empty old token is recorded as None (== bypass)."""
    global _previous_token, _previous_token_expires_at
    _previous_token = old or None
    _previous_token_expires_at = time.monotonic() + _TOKEN_GRACE_S


def _previous_token_valid() -> "str | None":
    """Return the previous token IF it's still inside the grace window,
    else clear it and return None."""
    global _previous_token, _previous_token_expires_at
    if not _previous_token:
        return None
    if time.monotonic() > _previous_token_expires_at:
        _previous_token = None
        _previous_token_expires_at = 0.0
        return None
    return _previous_token


def require_admin_token(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """If cfg.ADMIN_TOKEN is set, require a matching bearer token. If unset,
    this is a no-op — the loopback check alone is the gate.

    During a 60 s grace window after rotate (see _record_previous_token), the
    pre-rotate token is also accepted so the editing session can update its
    stored token without disruption."""
    expected = cfg.ADMIN_TOKEN
    if not expected:
        return
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bearer token required")
    presented = creds.credentials
    if secrets.compare_digest(presented, expected):
        return
    grace = _previous_token_valid()
    if grace and secrets.compare_digest(presented, grace):
        return
    logger.warning("[config] rejected bad bearer token")
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")


# --- router ------------------------------------------------------------------

router = APIRouter(prefix="/config")


def _resolved_value(field: str) -> Any:
    """Read the current effective value of a config field by attribute name.
    For ADMIN_TOKEN, return only a presence sentinel (the UI never needs the
    raw value, only whether one is set)."""
    if field == "ADMIN_TOKEN":
        return "***" if getattr(cfg, "ADMIN_TOKEN", None) else ""
    val = getattr(cfg, field, None)
    # Convert un-JSON-able types so the WebUI gets clean data.
    if isinstance(val, (set, frozenset)):
        return sorted(val)
    if isinstance(val, tuple):
        return [list(p) if isinstance(p, tuple) else p for p in val]
    return val


# Pydantic re-validates each rule so model_dump() emits keys in the
# discriminated-union's declaration order — same on both `value` and
# `default_value` so JSON.stringify on each yields identical strings
# when the rule contents match. Without this, _BASELINE keeps source
# order from config.py while the resolved value (after a local.json
# overlay) carries Pydantic's parent-first MRO order, and the WebUI's
# _isRuleDirty() always reports dirty on first paint.
def _canon_rules(rules: Any) -> Any:
    if not isinstance(rules, list):
        return rules
    from pydantic import TypeAdapter
    adapter = TypeAdapter(config_store.PipelineRule)
    out: list[Any] = []
    for r in rules:
        try:
            dumped = adapter.validate_python(r).model_dump(exclude_none=True)
            out.append(_sort_dicts(dumped))
        except Exception:
            out.append(r)  # malformed — pass through; save-time validator catches it
    return out


# `model_dump()` preserves insertion order on nested dict fields (e.g. the
# `map` on a callback:map rule). The resolved value (after a local.json
# overlay) and the baseline `default_value` (from cfg._BASELINE) can carry
# different insertion orders even when contents are equal — which makes
# JSON.stringify(value) !== JSON.stringify(default_value) and the WebUI's
# _isRuleDirty() falsely reports dirty on first paint, AND clicking reset
# visibly re-sorts the rows. Recursively sorting nested dict keys (applied
# identically to value AND default_value) makes the equality check reliable.
# Forced alphabetical is the right canonical order for `cb:map` rules: the
# longest-first word-bounded regex is rebuilt server-side from these keys,
# so display order has no functional meaning.
def _sort_dicts(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_dicts(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_sort_dicts(x) for x in obj]
    return obj


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

    # PIPELINE_RULES: canonicalize key order on both sides of the wire so
    # the WebUI's deep-equal compare (JSON.stringify) is reliable on first
    # paint. See _canon_rules() for the why.
    if "PIPELINE_RULES" in fields:
        fields["PIPELINE_RULES"]["value"] = _canon_rules(fields["PIPELINE_RULES"]["value"])
        fields["PIPELINE_RULES"]["default_value"] = _canon_rules(fields["PIPELINE_RULES"]["default_value"])

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

    # Capture the pre-save ADMIN_TOKEN so we can install the 60 s grace window
    # if it's about to change. Loopback callers don't need this, but token-
    # gated remote sessions would lock themselves out otherwise.
    _prev_admin_token = getattr(cfg, "ADMIN_TOKEN", None)

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

    # Token rotate: install grace window when ADMIN_TOKEN actually changed.
    # Empty / unset old token → no grace needed (no one was authenticating
    # with a token anyway). Loopback bypass means the local admin always
    # has access regardless.
    if "ADMIN_TOKEN" in written and _prev_admin_token \
            and getattr(cfg, "ADMIN_TOKEN", None) != _prev_admin_token:
        _record_previous_token(_prev_admin_token)
        logger.info("[config] ADMIN_TOKEN rotated — previous token valid for "
                    "%d s grace window", int(_TOKEN_GRACE_S))

    if needs_cache_rebuild:
        try:
            import main as _main
            _main.rebuild_caches()
            logger.info("[config] rebuilt pipeline caches after admin update")
        except Exception as e:
            logger.error("[config] cache rebuild failed: %s", e)

    # Eviction-on-edit: when a load-time field changed (globally or per-model),
    # drop the affected loaded model(s) from the cache so the next request
    # reloads them with the new settings. In-flight transcribes finish on the
    # old WhisperModel instance via Python ref-counting (drain-then-evict).
    evicted: list[str] = []
    try:
        import main as _main
        load_time_changed_globally = bool(
            set(written.keys()) & config_store.LOAD_TIME_FIELDS
        )
        if load_time_changed_globally:
            # Affects every loaded model that doesn't have a per-model
            # override winning over the changed global field. Conservative
            # fallback: evict ALL models. They reload lazily so this is cheap.
            ev = await _main.drain_then_evict(None)
            evicted.extend(ev)
        if "MODEL_OVERRIDES" in written:
            # Per-model override changed for one or more model ids — figure
            # out which ones touched a load-time field and evict only those.
            new_overrides = coerced.get("MODEL_OVERRIDES") or {}
            old_overrides = (
                # The pre-save effective value of MODEL_OVERRIDES. We don't
                # have a clean snapshot, but `coerced` is post-save and
                # `written` only contains diffs vs disk; combined with the
                # cfg.MODEL_OVERRIDES that was active before setattr ran above
                # we can't reconstruct cleanly. Easiest: any model id that
                # appears in the new dict and has a load-time field set
                # gets evicted (idempotent for unchanged models — they just
                # reload once).
                {}
            )
            for model_id, ovr in new_overrides.items():
                if not isinstance(ovr, dict):
                    continue
                if set(ovr.keys()) & config_store.LOAD_TIME_FIELDS:
                    ev = await _main.drain_then_evict(model_id)
                    evicted.extend(ev)
    except Exception as e:
        # Never let eviction failure break the save response. The user's
        # change still persisted; worst case they restart manually.
        logger.error("[config] eviction-on-edit failed: %s", e)

    # Re-sync os.environ["HF_TOKEN"] whenever USE_AUTH_TOKEN changed. The
    # token is set process-wide at startup (main.py) so non-WhisperModel HF
    # calls (Silero VAD, tokenizer fetches) inherit it; live edits via the
    # admin UI need to re-set the env var or those callers stay on the old
    # value until next service restart.
    if "USE_AUTH_TOKEN" in written:
        new_token = getattr(cfg, "USE_AUTH_TOKEN", None) or ""
        if new_token:
            os.environ["HF_TOKEN"] = new_token
            logger.info("[config] HF_TOKEN updated from USE_AUTH_TOKEN edit")
        else:
            os.environ.pop("HF_TOKEN", None)
            logger.info("[config] HF_TOKEN cleared (USE_AUTH_TOKEN unset)")

    client_host = request.client.host if request.client else "?"
    logger.info(
        "[config] admin update from=%s saved=%d hot=%s cold=%s pinned=%s evicted=%s",
        client_host, len(written), hot_changed, cold_changed,
        [n for n in written if n in env_pinned], evicted,
    )

    return JSONResponse({
        "saved": sorted(written.keys()),
        "hot_applied": hot_changed,
        "cold_pending": cold_changed,
        "env_pinned_ignored": sorted(n for n in written if n in env_pinned),
        "evicted": evicted,
        "requires_restart": bool(cold_changed),
    })


@router.post("/test-pipeline",
             dependencies=[Depends(require_admin_host), Depends(require_admin_token)])
async def test_pipeline(payload: dict[str, Any]) -> JSONResponse:
    """Dry-run the full pipeline-rules list against a sample. Used by the
    WebUI's per-row live-validation badge AND the inline test panel.

    Payload: { sample: str, rules: list[dict] }
    `rules` is the live (dirty+saved) overlay from the WebUI so unsaved edits
    are testable. Each rule is the same dict shape as in cfg.PIPELINE_RULES.

    Response:
      {
        "steps": [
          { ordinal, label, type, before, after, matches, skipped, error, slow }, ...
          { ordinal, label: "Trim edges", type: "terminal", ... }
        ],
        "final": str,
      }

    Each rule is compiled and run under a 2 s threading-timer guard against
    the sample. Disabled rules render as `skipped: true` (not run). Rules
    with empty patterns also `skipped: true`. Compile errors → `error: "<msg>"`
    and the pipeline continues with the un-modified text. The terminal trim
    is appended at the end; if no terminal row is present, the trim is
    still applied (matching main.py behaviour).
    """
    import threading

    sample = str(payload.get("sample") or "")
    rules = payload.get("rules") or []
    if not isinstance(rules, list):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "rules must be a list of rule dicts"},
        )

    def _run_rule(text: str, rule: dict) -> dict[str, Any]:
        """Apply one rule to `text`. Returns the step dict for the response."""
        rtype = rule.get("type", "?")
        label = rule.get("label", rule.get("name", "?"))
        common = {"label": label, "type": rtype, "before": text, "matches": 0,
                  "skipped": False, "error": None, "slow": False}

        if rtype == "terminal":
            after = text.lstrip(" \t\r").rstrip(" \t\r")
            return {**common, "after": after}

        if not rule.get("enabled", True):
            return {**common, "after": text, "skipped": True}

        try:
            if rtype == "callback:map":
                m = rule.get("map", {}) or {}
                if not m:
                    return {**common, "after": text, "skipped": True}
                alternation = "|".join(re.escape(k) for k in sorted(m, key=len, reverse=True))
                cre = re.compile(r"\b(" + alternation + r")\b", re.IGNORECASE)
                lookup = {k.lower(): v for k, v in m.items()}
                replacer = lambda mt: lookup.get(mt.group(0).lower(), mt.group(0))
            else:
                pattern = rule.get("pattern", "") or ""
                if not pattern:
                    return {**common, "after": text, "skipped": True}
                cre = re.compile(pattern)
                if rtype == "regex":
                    repl = rule.get("replacement", "") or ""
                    replacer = lambda mt, _r=repl: mt.expand(_r) if "\\" in _r else _r
                elif rtype == "callback:lowercase-wordlist":
                    wordlist = frozenset(w.lower() for w in (rule.get("wordlist") or []))
                    def replacer(mt, _wl=wordlist):
                        try:
                            ws, first, rest = mt.group(1), mt.group(2), mt.group(3)
                        except IndexError:
                            return ""
                        if (first + rest).lower() in _wl:
                            return ws + first.lower() + rest
                        return ws + first + rest
                elif rtype == "callback:dedup":
                    def replacer(mt):
                        run = mt.group(0)
                        non_comma = [c for c in run if c != ","]
                        return non_comma[-1] if non_comma else ","
                elif rtype == "callback:upper":
                    def replacer(mt):
                        try:
                            return mt.group(1) + mt.group(2).upper()
                        except IndexError:
                            return mt.group(0).upper()
                else:
                    return {**common, "after": text, "skipped": True,
                            "error": f"unknown rule type: {rtype}"}
        except re.error as e:
            return {**common, "after": text, "error": str(e)}

        out: dict[str, Any] = {"done": False, "after": text, "matches": 0}
        def _do() -> None:
            try:
                out["matches"] = sum(1 for _ in cre.finditer(text))
                out["after"] = cre.sub(replacer, text)
                out["done"] = True
            except Exception as e:  # noqa: BLE001
                out["err"] = str(e)
                out["done"] = True
        t = threading.Thread(target=_do, daemon=True)
        t.start()
        t.join(timeout=2.0)
        if not out["done"]:
            return {**common, "after": text, "slow": True}
        if "err" in out:
            return {**common, "after": text, "error": out["err"]}
        return {**common, "after": out["after"], "matches": out["matches"]}

    text = sample
    steps: list[dict[str, Any]] = []
    saw_terminal = False
    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        step = _run_rule(text, rule)
        step["ordinal"] = idx + 1
        steps.append(step)
        text = step["after"]
        if rule.get("type") == "terminal":
            saw_terminal = True
    if not saw_terminal:
        # No terminal row in the payload — apply the implicit trim.
        before = text
        text = text.lstrip(" \t\r").rstrip(" \t\r")
        if before != text:
            steps.append({
                "ordinal": len(steps) + 1,
                "label": "Trim edges (always-last)",
                "type": "terminal",
                "before": before, "after": text, "matches": 0,
                "skipped": False, "error": None, "slow": False,
            })

    return JSONResponse({"steps": steps, "final": text})


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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>faster-whisper-backend · config</title>
{{SCALE_BOOTSTRAP_HEAD}}
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --fg: #c9d1d9; --dim: #6e7681;
    --cyan: #79c0ff; --green: #7ee787; --yellow: #f2cc60;
    --red: #ff7b72; --magenta: #d2a8ff; --bold: #f0f6fc;
    --border: #30363d; --input-bg: #0d1117;
  }
  /* Font-size tokens, --font-sans, --font-mono and html { font-size: var(--fs-base) }
     live in NAV_CSS (injected further down, just before the body markup)
     so all pages share one scaling knob. Important: never embed the NAV_CSS
     template placeholder inside another comment block — render_page() does
     a naive string replace and would inject NAV_CSS into this comment,
     prematurely closing it (NAV_CSS contains its own internal comments)
     and silently dropping every CSS rule that follows.
     Chrome (titles, labels, descriptions, buttons, badges) uses --font-sans;
     code-y contexts (input/textarea values, log lines, regex panels, the
     dictation-map key/value cells) opt into --font-mono via the rules below. */
  html, body { background: var(--bg); color: var(--fg);
    font: 1rem/1.5 var(--font-sans);
    margin: 0; padding: 0; min-height: 100%; }
  input, textarea, select, kbd, code, pre { font-family: var(--font-mono); }
  header { position: sticky; top: 0; background: var(--panel); border-bottom: 1px solid var(--border);
    z-index: 10; padding: 0; }
  header > .header-inner { display: flex; gap: 0.75rem; align-items: center;
    max-width: 1100px; margin: 0 auto; width: 100%; padding: 0.5rem 0.875rem;
    box-sizing: border-box; }
  header .title { font-weight: 600; color: var(--bold);
    white-space: nowrap; flex-shrink: 0; }
  header .pill { padding: 0.125rem 0.5rem; border-radius: 4px; background: #21262d; color: var(--dim);
    font-size: var(--fs-xs); white-space: nowrap; flex-shrink: 0; }
  header .pill.ok { color: var(--green); border: 1px solid #1f4d2a; }
  header .pill.warn { color: var(--yellow); border: 1px solid #4d3e1f; }
  header .pill.err { color: var(--red); border: 1px solid #5a2424; }
  header button { background: #21262d; color: var(--fg); border: 1px solid var(--border);
    padding: 0.25rem 0.625rem; border-radius: 4px; cursor: pointer; font: inherit;
    flex-shrink: 0; }
  header button:hover { background: #30363d; }
  header button.primary { background: #238636; border-color: #2ea043; color: var(--bold); }
  header button.primary:hover { background: #2ea043; }
  /* Discard button: red-tinted "warning" feel when there are unsaved edits.
     Disabled state inherits the generic header button:disabled (opacity 0.4). */
  header button#discard-btn:not(:disabled) { background: #3a0d0d;
    border-color: #5a2424; color: var(--red); }
  header button#discard-btn:not(:disabled):hover { background: #531f1f;
    border-color: #7d2d2d; }
  header button:disabled { opacity: 0.4; cursor: not-allowed; }
  main { padding: 0.875rem; max-width: 1100px; margin: 0 auto; }
  section { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 0.625rem 0.875rem 0.75rem; margin-bottom: 0.875rem; }
  h2 { color: var(--bold); font-size: 0.933rem; margin: 0 0 0.5rem; padding-bottom: 0.375rem;
    border-bottom: 1px solid var(--border); }
  /* Subgrid: each subgroup wraps its fields in .group-fields so all rows
     share one column track. The label column auto-sizes to whatever the
     longest label in THAT section needs; the value column gets the rest.
     Sections are independent — a tight section won't widen because of a
     wide one elsewhere. Subgrid: Chrome 117+, Firefox 71+, Safari 16+. */
  .group-fields { display: grid;
    grid-template-columns: minmax(min-content, max-content) 1fr;
    column-gap: 0.625rem; }
  .field { display: grid; grid-template-columns: subgrid;
    grid-column: 1 / -1; gap: 0.625rem; align-items: start;
    padding: 0.375rem 0; border-bottom: 1px dashed #21262d; }
  .field:last-child { border-bottom: none; }
  .label-col { display: flex; flex-direction: column; gap: 0.25rem; }
  .label-col .name { color: var(--bold); }
  .badges { display: flex; gap: 0.25rem; flex-wrap: wrap; }
  .badge { font-size: 0.667rem; padding: 0.0625rem 0.375rem; border-radius: 999px;
    border: 1px solid var(--border); color: var(--dim); }
  .badge.live { color: var(--green); border-color: #1f4d2a; }
  .badge.restart { color: var(--yellow); border-color: #4d3e1f; }
  .badge.env { color: var(--magenta); border-color: #4a2e6f; }
  .badge.local { color: var(--cyan); border-color: #194f73; }
  /* font-size/line-height inherit from body (=1rem/1.5); font-family
     comes from the generic `input,textarea,select { font-family: var(--font-mono) }`
     rule in the body block. Avoid `font: inherit` here — that's a shorthand
     that pulls in font-family from the parent (which is now sans), defeating
     the chrome/code split. */
  .input-col input, .input-col textarea, .input-col select {
    width: 100%; box-sizing: border-box;
    background: var(--input-bg); color: var(--fg); border: 1px solid var(--border);
    padding: 0.3125rem 0.5rem; border-radius: 4px;
    font-size: inherit; line-height: inherit; }
  .input-col input[type=checkbox] { width: auto; }
  .input-col textarea { min-height: 4rem; resize: vertical; }
  /* --- Native widget overrides — bring all browser-default controls
     (checkbox, number-spinner, select, textarea, unclassed buttons)
     into the GitHub-dark palette. ----------------------------------- */
  /* Checkbox: replace native white square with a dark themed one. */
  input[type="checkbox"] {
    appearance: none; -webkit-appearance: none;
    width: 16px; height: 16px;
    border: 1px solid var(--border); border-radius: 3px;
    background: var(--input-bg); cursor: pointer;
    position: relative; vertical-align: middle;
    transition: background 120ms ease, border-color 120ms ease;
  }
  input[type="checkbox"]:hover { border-color: var(--cyan); }
  input[type="checkbox"]:checked { background: #1f6feb; border-color: #388bfd; }
  input[type="checkbox"]:checked::after {
    content: ""; position: absolute; left: 4px; top: 0;
    width: 4px; height: 9px;
    border: solid var(--bold); border-width: 0 2px 2px 0;
    transform: rotate(45deg);
  }
  input[type="checkbox"]:focus-visible {
    outline: 2px solid var(--cyan); outline-offset: 1px;
  }
  /* Number-input: hide native spinner buttons (Firefox + WebKit/Blink).
     Field looks like a plain text input; user types or pastes numbers. */
  input[type="number"] { -moz-appearance: textfield; }
  input[type="number"]::-webkit-inner-spin-button,
  input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none; margin: 0;
  }
  /* Select: replace native white triangle with a custom dim-grey SVG arrow.
     Scoped to .input-col so the header scale-picker (which has its own
     style) isn't double-overridden. */
  .input-col select {
    appearance: none; -webkit-appearance: none;
    background: var(--input-bg) url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'><path fill='%236e7681' d='M0 0l5 6 5-6z'/></svg>") no-repeat right 0.5rem center;
    padding: 0.3125rem 1.5rem 0.3125rem 0.5rem;
    cursor: pointer;
  }
  .input-col select:focus { outline: 1px solid var(--cyan); outline-offset: -1px; }
  /* Generic dark button styling for unclassed <button> elements (Add /
     Cancel in custom-rule dialog, "+ Add custom rule", etc.). Scoped to
     not override existing classed buttons (.reset-link, .delete-btn,
     .expand-btn, .add-row button, header button, .modal button, etc.). */
  main button:not([class]) {
    background: #21262d; color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.25rem 0.75rem; font: inherit; cursor: pointer;
    transition: background 120ms ease, border-color 120ms ease;
  }
  main button:not([class]):hover { background: #30363d; border-color: #484f58; }
  main button:not([class]):active { background: #161b22; }
  main button:not([class]):disabled { opacity: 0.45; cursor: not-allowed; }
  /* white-space: pre-line preserves \n in description strings (so a
     description that enumerates values can render each on its own line)
     while still collapsing other whitespace and wrapping long lines. */
  .input-col .help { color: var(--help); font-size: var(--fs-sm);
    margin-top: 0.1875rem; white-space: pre-line; }
  /* Vertical checkbox list (PRELOAD_MODELS, etc.). Both gaps in rem so they
     scale with --fs-base — fixes "compressed checklist" at higher scales. */
  .cb-list { display: flex; flex-direction: column; gap: 0.4rem; }
  .cb-row  { display: flex; gap: 0.5rem; align-items: center; cursor: pointer; }
  .err { color: var(--red); font-size: var(--fs-xs); margin-top: 0.1875rem; }
  /* Field row dimming when a parent toggle makes this row irrelevant.
     pointer-events stays alive so the user can still see the contents and
     edit if they want — we just signal "this is currently unused". */
  .field.dep-irrelevant { opacity: 0.45; }
  .field.dep-irrelevant .input-col { filter: grayscale(0.6); }
  /* Subgroup heading inside a section: smaller than h2, lighter weight,
     small dividing line so it's visibly distinct from the section header
     but doesn't draw the eye like a top-level section change. */
  h3.subgroup { color: var(--dim); font-size: var(--fs-sm); font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 0.875rem 0 0.375rem 0; padding-bottom: 0.1875rem;
    border-bottom: 1px solid var(--border); }
  /* Collapsible subgroup: <summary> mirrors h3.subgroup styling, plus a
     leading chevron that flips when the <details> is open. Native
     disclosure-triangle is suppressed — list-style: none on summary +
     ::-webkit-details-marker for older WebKit. State persists in
     localStorage (see render() in JS). */
  details.subgroup-details { margin: 0.875rem 0 0.375rem 0; }
  details.subgroup-details > summary.subgroup-summary {
    color: var(--dim); font-size: var(--fs-sm); font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.06em;
    padding-bottom: 0.1875rem; border-bottom: 1px solid var(--border);
    cursor: pointer; user-select: none; list-style: none;
    display: flex; align-items: center; gap: 0.4rem; }
  details.subgroup-details > summary.subgroup-summary::-webkit-details-marker { display: none; }
  details.subgroup-details > summary.subgroup-summary::before {
    content: '▸'; color: var(--dim); font-size: 0.8em;
    transition: transform 120ms ease; display: inline-block; }
  details.subgroup-details[open] > summary.subgroup-summary::before {
    transform: rotate(90deg); }
  details.subgroup-details > summary.subgroup-summary:hover { color: var(--fg); }
  details.subgroup-details > .group-fields { margin-top: 0.375rem; }
  /* Reset-to-default link button — small, italic, only visible when the
     current value differs from the in-repo default. Sits below the help
     text, so it doesn't crowd the editor itself. */
  .reset-wrap { margin-top: 0.25rem; }
  .reset-link { background: none; border: none; padding: 0;
    color: var(--cyan); cursor: pointer; font: inherit; font-size: var(--fs-xs);
    font-style: italic; text-decoration: underline; text-underline-offset: 2px; }
  .reset-link:hover { color: var(--bold); }
  /* Regex editor + status badge */
  .regex-wrap { display: flex; flex-direction: column; gap: 0.25rem; }
  .regex-status { font-size: var(--fs-xs); font-family: var(--font-mono); }
  .regex-status.ok { color: var(--green); }
  .regex-status.err { color: var(--red); }
  .regex-status.warn { color: var(--yellow); }
  .regex-status.empty { color: var(--dim); font-style: italic; }
  /* Advanced warning banner above the Step 3 fields */
  .advanced-warn { background: #2d1f0a; color: #f2cc60; border-left: 3px solid #f2cc60;
    padding: 0.375rem 0.625rem; margin: 0.5rem 0; border-radius: 3px; font-size: var(--fs-sm); }
  /* Test panel */
  .regex-test-panel { background: #161b22; border: 1px solid var(--border);
    border-radius: 4px; padding: 0.625rem 0.75rem; margin: 0.5rem 0 0.875rem 0; }
  .regex-test-panel textarea { resize: vertical; max-width: 100%; }
  .regex-test-out { margin-top: 0.625rem; }
  .regex-test-pass { margin: 0.375rem 0; }
  .regex-test-head { font-family: var(--font-mono); font-size: var(--fs-sm);
    color: var(--dim); }
  .regex-test-head .tag { display: inline-block; padding: 0 0.375rem; margin-left: 0.375rem;
    border-radius: 3px; font-size: var(--fs-xs); }
  .regex-test-head .tag.ok { background: #033a16; color: #7ee787; }
  .regex-test-head .tag.err { background: #3a0d0d; color: #ff7b72; }
  .regex-test-head .tag.warn { background: #2d1f0a; color: #f2cc60; }
  .regex-test-head .tag.empty { background: #21262d; color: var(--dim); font-style: italic; }
  .regex-test-result { background: #0d1117; border: 1px solid var(--border);
    padding: 0.25rem 0.5rem; border-radius: 3px; margin: 0.25rem 0;
    font-family: var(--font-mono); font-size: var(--fs-sm);
    white-space: pre-wrap; word-break: break-word; max-height: 7.5rem; overflow: auto; }
  .regex-test-final { margin-top: 0.625rem; padding-top: 0.625rem; border-top: 1px solid var(--border); }
  .field .dep-note { color: var(--dim); font-size: var(--fs-xs); margin-top: 0.1875rem;
    font-style: italic; }
  /* Pipeline rules editor */
  .pipeline-rules-wrap { display: flex; flex-direction: column; gap: 0.375rem; }
  .rule-list { display: flex; flex-direction: column; gap: 0.25rem; }
  .rule-row { background: #161b22; border: 1px solid var(--border); border-radius: 4px;
    padding: 0.375rem 0.625rem; }
  .rule-row.locked { border-left: 3px solid #f2cc60; }
  .rule-row.terminal { border-left: 3px solid var(--dim); opacity: 0.85; }
  /* Button-like interactive feel on every rule row. :active propagates
     up from descendants so pressing the drag-handle still tints the row. */
  .rule-row { transition: background-color 120ms ease; cursor: default; }
  .rule-row:not(.terminal):hover { background: #1c2230; }
  .rule-row:not(.terminal):active { background: #232a36; }
  .rule-row[tabindex]:focus-visible { outline: 2px solid var(--cyan);
    outline-offset: -1px; }
  /* Suppress :hover noise while a drag is in flight — otherwise every
     row the cursor crosses pulses. */
  .rule-list.dnd-active .rule-row:hover { background: #161b22; }
  .rule-row.dragging { opacity: 0.3; outline: 2px dashed var(--cyan); }
  /* Placeholder slot — the empty cyan-bordered space the dragged row
     will land in. Height set inline at dragstart to match source row. */
  .rule-placeholder {
    border: 1px dashed var(--cyan);
    background: rgba(56, 189, 248, 0.08);
    border-radius: 4px;
    margin-bottom: 0.25rem;
    transition: height 120ms ease;
  }
  .rule-row.disabled { opacity: 0.55; }
  .rule-row .row-header { display: flex; align-items: center; gap: 0.5rem;
    font-family: var(--font-mono); font-size: var(--fs-sm); }
  .rule-row .drag-handle { cursor: grab; user-select: none; color: var(--dim);
    padding: 0.125rem 0.25rem; }
  .rule-row .drag-handle:active { cursor: grabbing; }
  .rule-row.locked .drag-handle { cursor: not-allowed; }
  .rule-row .ordinal { color: var(--dim); min-width: 1.5rem; text-align: right; }
  .rule-row .rule-label { flex: 1; color: var(--fg); }
  .rule-row .rule-slug { color: var(--dim); font-size: var(--fs-xs); font-style: italic; }
  .rule-row .type-pill { display: inline-block; padding: 0 0.375rem; border-radius: 3px;
    font-size: var(--fs-xs); background: #21262d; color: var(--cyan); }
  .rule-row .expand-btn, .rule-row .delete-btn {
    background: transparent; border: 1px solid var(--border);
    color: var(--fg); padding: 0.125rem 0.375rem; border-radius: 3px; cursor: pointer;
    font: inherit; font-size: var(--fs-xs); }
  .rule-row .delete-btn { color: var(--red); }
  .rule-row .row-body { padding-left: 2rem; padding-top: 0.375rem; display: none; }
  .rule-row.expanded .row-body { display: block; }
  .rule-row.terminal .row-body { display: block; padding-top: 0.25rem; }
  .rule-editor { display: flex; flex-direction: column; gap: 0.25rem; }
  .rule-editor .map-table { font-family: var(--font-mono); font-size: var(--fs-sm); }
  .rule-editor .map-table input { background: transparent; color: var(--fg);
    border: 1px solid var(--border); padding: 0.125rem 0.25rem; }
  /* Full-pipeline test table */
  .pipeline-test-table { width: 100%; border-collapse: collapse; font-size: var(--fs-sm);
    margin-top: 0.375rem; }
  .pipeline-test-table th, .pipeline-test-table td {
    border-bottom: 1px solid var(--border); padding: 0.25rem 0.375rem;
    text-align: left; vertical-align: top; }
  .pipeline-test-table th { color: var(--dim); font-weight: 500; }
  .pipeline-test-table tr:nth-child(even) { background: #0d1117; }
  .pipeline-test-table .out { font-family: var(--font-mono);
    white-space: pre-wrap; word-break: break-word; }
  .pipeline-test-table .out mark { background: #033a16; color: #7ee787; }
  .pipeline-test-table .nochange { color: var(--dim); font-style: italic; }
  .pipeline-test-table .tag { display: inline-block; padding: 0 0.375rem;
    border-radius: 3px; font-size: var(--fs-xs); }
  .pipeline-test-table .tag.ok { background: #033a16; color: #7ee787; }
  .pipeline-test-table .tag.err { background: #3a0d0d; color: #ff7b72; }
  .pipeline-test-table .tag.warn { background: #2d1f0a; color: #f2cc60; }
  .pipeline-test-table .tag.empty { background: #21262d; color: var(--dim); font-style: italic; }
  .preset-select { margin-bottom: 0.375rem; }
  .preset-select select { font-family: var(--font-mono); font-size: var(--fs-sm); }
  /* Nullable-number editor in its disabled (null) state — greyed input,
     enable/disable button labelled accordingly. */
  .nullable-wrap input:disabled { opacity: 0.4; cursor: not-allowed; }
  table.dict { width: 100%; border-collapse: collapse; }
  table.dict th, table.dict td { border: 1px solid var(--border); padding: 0.25rem 0.5rem;
    text-align: left; }
  table.dict th { background: #1c2129; color: var(--dim); font-weight: 500; font-size: var(--fs-xs); }
  table.dict td input { width: 100%; box-sizing: border-box; background: transparent;
    color: var(--fg); border: none; padding: 0;
    font-size: inherit; line-height: inherit; }
  table.dict td input:focus { outline: 1px solid var(--cyan); outline-offset: -1px; }
  table.dict button.del { background: transparent; border: 1px solid var(--border);
    color: var(--red); padding: 0.125rem 0.375rem; border-radius: 3px; cursor: pointer; font-size: var(--fs-xs); }
  .add-row { margin-top: 0.375rem; }
  .add-row button { background: #21262d; border: 1px solid var(--border); color: var(--fg);
    padding: 0.1875rem 0.625rem; border-radius: 4px; cursor: pointer; font: inherit; }
  .modal { position: fixed; inset: 0; background: rgba(0,0,0,0.7); display: none;
    align-items: center; justify-content: center; z-index: 100; }
  .modal.show { display: flex; }
  .modal-box { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 1rem 1.25rem; max-width: 32.5rem; }
  .modal-box h3 { margin: 0 0 0.625rem; color: var(--bold); }
  .modal-box ul { margin: 0.375rem 0 0.75rem; padding-left: 1.25rem; }
  .modal-actions { display: flex; gap: 0.5rem; justify-content: flex-end; margin-top: 0.875rem; }
  .modal-actions button { padding: 0.3125rem 0.75rem; }
  .login { max-width: 30rem; margin: 3.75rem auto; background: var(--panel);
    border: 1px solid var(--border); border-radius: 6px; padding: 1.25rem 1.5rem; }
  .login h1 { color: var(--bold); margin: 0 0 0.375rem; font-size: var(--fs-xl); }
  .login p { color: var(--dim); margin: 0.375rem 0 0.75rem; }
  .login input { width: 100%; box-sizing: border-box; background: var(--input-bg);
    color: var(--fg); border: 1px solid var(--border); padding: 0.5rem;
    border-radius: 4px; font-size: inherit; line-height: inherit; }
  .login button { margin-top: 0.625rem; background: #238636; border: 1px solid #2ea043;
    color: var(--bold); padding: 0.4375rem 0.875rem; border-radius: 4px; cursor: pointer;
    font: inherit; }
  #toast { position: fixed; bottom: 1rem; right: 1rem; background: var(--panel);
    border: 1px solid var(--border); border-radius: 4px; padding: 0.5rem 0.75rem;
    color: var(--fg); display: none; }
  #toast.show { display: block; }
  #toast.err { border-color: #5a2424; color: var(--red); }
  /* ============================================================ */
  /* MODEL_OVERRIDES master-detail editor                          */
  /* ============================================================ */
  /* Two-column layout: sidebar of models + main pane of fields.
     Below 56rem the sidebar collapses to a horizontal scroller above
     the pane (KDE Linux narrow-viewport requirement).               */
  .mo-wrap { display: grid; grid-template-columns: 16rem 1fr;
    gap: 0.875rem; align-items: start; }
  .mo-sidebar { display: flex; flex-direction: column; gap: 0.25rem;
    background: #0d1117; border: 1px solid var(--border); border-radius: 4px;
    padding: 0.5rem; min-width: 0; }
  .mo-list-item { padding: 0.4rem 0.6rem; border-radius: 4px; cursor: pointer;
    display: flex; align-items: center; gap: 0.5rem;
    border-left: 3px solid transparent; min-width: 0; }
  .mo-list-item:hover { background: #1c2230; }
  .mo-list-item.active { background: #1c2230; border-left-color: var(--cyan); }
  .mo-list-item .mo-list-label { flex: 1; min-width: 0;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .mo-list-item .mo-count { color: var(--dim); font-size: var(--fs-xs);
    flex-shrink: 0; }
  .mo-list-add { margin-top: 0.4rem; padding-top: 0.4rem;
    border-top: 1px solid var(--border); }
  .mo-list-add select { width: 100%; box-sizing: border-box;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.25rem 0.5rem; font: inherit; font-size: var(--fs-sm); }
  .mo-diff-toggle { margin-top: 0.5rem; padding-top: 0.5rem;
    border-top: 1px solid var(--border); font-size: var(--fs-sm); }
  .mo-remove-btn { margin-top: 0.5rem; color: var(--red) !important;
    text-align: left; }
  .mo-mainpane { display: flex; flex-direction: column; gap: 0.5rem;
    min-width: 0; }
  .mo-edit-head { padding: 0.4rem 0.6rem; background: #0d1117;
    border: 1px solid var(--border); border-radius: 4px;
    font-size: var(--fs-sm); }
  .mo-edit-head .mo-name { font-family: var(--font-mono); color: var(--cyan); }
  .mo-section { padding: 0.625rem 0.75rem; background: #0d1117;
    border: 1px solid var(--border); border-radius: 4px; }
  .mo-section > h4 { margin: 0 0 0.5rem 0; font-size: var(--fs-sm);
    color: var(--bold); display: flex; align-items: center; flex-wrap: wrap; }
  /* Per-field row: ● label | input | ↶ button. Grid keeps columns aligned
     across rows of the same section. Min-content on the label column so
     long field names size the column to fit; rest goes to the input cell. */
  .mo-row { display: grid;
    grid-template-columns: 0.75rem minmax(min-content, max-content) 1fr auto;
    gap: 0.5rem; align-items: center;
    padding: 0.25rem 0; border-bottom: 1px dashed #21262d; min-width: 0; }
  .mo-row:last-child { border-bottom: none; }
  .mo-row .mo-name { font-family: var(--font-mono); color: var(--bold);
    font-size: var(--fs-sm); white-space: nowrap; }
  .mo-row .mo-value-cell { display: flex; align-items: center;
    gap: 0.5rem; min-width: 0; }
  .mo-row .mo-value-cell input[type="text"],
  .mo-row .mo-value-cell input[type="number"],
  .mo-row .mo-value-cell select,
  .mo-row .mo-value-cell textarea {
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.25rem 0.4rem;
    font: inherit; font-family: var(--font-mono); font-size: var(--fs-sm);
    width: 100%; box-sizing: border-box; }
  .mo-row .mo-value-cell select { appearance: none; -webkit-appearance: none;
    background: var(--input-bg) url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'><path fill='%236e7681' d='M0 0l5 6 5-6z'/></svg>") no-repeat right 0.5rem center;
    padding-right: 1.5rem; cursor: pointer; }
  .mo-row .mo-value-cell textarea { min-height: 2.5rem; resize: vertical; }
  .mo-row .mo-value-cell input[type="number"]::-webkit-inner-spin-button,
  .mo-row .mo-value-cell input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none; margin: 0; }
  .mo-row .mo-value-cell input[type="number"] { -moz-appearance: textfield; }
  .mo-row .mo-row-ctrl { display: flex; gap: 0.4rem; align-items: center;
    flex-shrink: 0; }
  .mo-row.mo-row-ref .mo-value-cell { font-family: var(--font-mono);
    color: var(--fg); font-size: var(--fs-sm); }
  /* Inherit/override dot indicator. Filled cyan = override active, dim
     outline = inherits global. */
  .mo-dot { width: 0.5rem; height: 0.5rem; border-radius: 50%;
    flex-shrink: 0; }
  .mo-dot.override { background: var(--cyan);
    box-shadow: 0 0 4px rgba(121, 192, 255, 0.5); }
  .mo-dot.inherit { background: transparent; border: 1px solid var(--dim);
    box-sizing: border-box; }
  .mo-inherits { color: var(--dim); font-style: italic;
    font-size: var(--fs-sm); }
  /* Diff-to-global mode: dim rows whose value matches the resolved global.
     Reuses the same opacity/grayscale recipe as .field.dep-irrelevant. */
  .mo-mainpane.diff-mode .mo-row[data-matches-global="true"] {
    opacity: 0.45; filter: grayscale(0.6); }
  /* Per-model pipeline rules checklist: makeRuleListEditor renders each
     rule with .rule-row; checklist mode adds .checklist-mode on the wrap
     and per-row .excluded marker. Compact layout: checkbox | label | pill | view.   */
  .pipeline-rules-wrap.checklist-mode .rule-row {
    padding: 0.25rem 0.5rem; }
  .pipeline-rules-wrap.checklist-mode .rule-row .row-header {
    display: flex; align-items: center; gap: 0.5rem; }
  .pipeline-rules-wrap.checklist-mode .rule-row.excluded .rule-label {
    color: var(--dim); text-decoration: line-through; }
  .pipeline-rules-wrap.checklist-mode .rule-row .rule-checklist-status {
    font-size: var(--fs-xs); color: var(--yellow); font-style: italic; }
  .pipeline-rules-wrap.checklist-mode .rule-row.terminal {
    border-left-color: var(--dim); }
  /* Globally-disabled, NOT force-included → dim the row so the admin sees
     "this is dormant unless I act". On hover it brightens slightly so the
     row remains scannable. Once force-included, the .globally-disabled
     class is removed → row returns to full opacity. */
  .pipeline-rules-wrap.checklist-mode .rule-row.globally-disabled {
    opacity: 0.5; filter: grayscale(0.6); }
  .pipeline-rules-wrap.checklist-mode .rule-row.globally-disabled:hover {
    opacity: 0.85; filter: grayscale(0.3); }
  /* Tags: small italic badges sitting after the rule label/pill. */
  .pipeline-rules-wrap.checklist-mode .rule-globally-disabled-tag,
  .pipeline-rules-wrap.checklist-mode .rule-force-included-tag {
    font-size: var(--fs-xs); padding: 0 0.4rem;
    border-radius: 3px; font-style: italic; margin-left: 0.25rem; }
  .pipeline-rules-wrap.checklist-mode .rule-globally-disabled-tag {
    color: var(--dim); border: 1px solid var(--border); }
  .pipeline-rules-wrap.checklist-mode .rule-force-included-tag {
    color: var(--cyan); border: 1px solid #194f73; }
  .rule-checklist-footer { color: var(--dim); margin-top: 0.4rem;
    font-size: var(--fs-sm); }
  /* Narrow viewport: stack sidebar above main pane; let it scroll
     horizontally so long lists stay accessible. Threshold 56rem matches
     the existing scale-aware breakpoint convention elsewhere. */
  @media (max-width: 56rem) {
    .mo-wrap { grid-template-columns: 1fr; }
    .mo-sidebar { flex-direction: row; overflow-x: auto;
      align-items: stretch; }
    .mo-list-item { flex: 0 0 auto; border-left: none;
      border-bottom: 3px solid transparent; }
    .mo-list-item.active { border-left: none;
      border-bottom-color: var(--cyan); }
    .mo-list-add, .mo-diff-toggle, .mo-remove-btn { flex: 0 0 auto; }
  }
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
  <header><div class="header-inner">
    <span class="title">faster-whisper-backend · config</span>
    {{NAV}}
    <span class="spacer"></span>
    {{SCALE_PICKER}}
    <button id="logout-btn" title="forget token in this tab">logout</button>
    <button id="reload-btn">reload</button>
    <button id="restart-btn" title="restart the WhisperAPI Windows Service">restart</button>
    <button id="discard-btn" title="discard all unsaved changes" disabled>discard</button>
    <button id="save-btn" class="primary" disabled>save</button>
    <span id="status" class="pill">loading…</span>
  </div></header>
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
  $('discard-btn').disabled = true;
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
  const empty = Object.keys(dirty).length === 0;
  $('save-btn').disabled = empty;
  $('discard-btn').disabled = empty;
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

// Field dependencies are now handled inside makeRuleListEditor itself
// (per-row enabled/locked/seeded state). No top-level field-to-field
// dimming rules survive the PIPELINE_RULES unification.
function applyFieldDependencies() { /* no-op */ }

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
  row.appendChild(labelCol);

  const inputCol = document.createElement('div');
  inputCol.className = 'input-col';
  inputCol.appendChild(makeEditor(name));
  // Env-override warning: lives in the input column (not the label column)
  // so the long line wraps inside the value-column width. If it sits in
  // .label-col, CSS subgrid sizes the column to this prose's max-content
  // and blows the whole section's label track wide. .input-col .help
  // already has the right styling.
  if (isEnvPinned(name)) {
    const note = document.createElement('div');
    note.className = 'help';
    note.textContent = 'Currently overridden by env var; saves persist but '
      + 'only take effect when the env var is unset.';
    inputCol.appendChild(note);
  }

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
  // PIPELINE_RULES gets its own list-of-rules editor (mixed row types,
  // drag-to-reorder, per-row test badge). Routed by name BEFORE shape checks
  // since the value is a list.
  if (name === 'PIPELINE_RULES') return makeRuleListEditor(name, v || [], 'full', {});
  // MODEL_OVERRIDES is a dict[model_id, dict[field, value]] — too freeform
  // for the standard editors. Render as a JSON textarea with parse-validation
  // on every input. Save sends the parsed object; pydantic validates server-
  // side. Future polish: master-detail UI per the original design.
  if (name === 'MODEL_OVERRIDES') return modelOverridesEditor(name, v || {});
  // ADMIN_TOKEN: never render the raw value; show "Set ✓ / Not set" with
  // explicit Rotate / Clear actions. Confirm dialogs remind the admin
  // about the 60 s grace window after rotate.
  if (name === 'ADMIN_TOKEN') return adminTokenEditor(name, v);
  // Type dispatch — keep this strict. Order matters: check shape (object vs.
  // array vs. boolean vs. number) BEFORE name-based heuristics, otherwise
  // misses like MAX_LOADED_MODELS routing to a list editor sneak in.
  if (typeof v === 'boolean') return boolEditor(name, v);
  if (typeof v === 'number') return numberEditor(name, v);
  // Model-aware editors (must precede generic Array/list dispatch). Source
  // their options from the current ALLOWED_MODELS state — typing in the
  // allowlist textarea live-updates these.
  if (name === 'DEFAULT_MODEL') return modelDropdownEditor(name, v);
  if (name === 'PRELOAD_MODELS') return modelMultiSelectEditor(name, v);
  if (Array.isArray(v)) return linesEditor(name, v);
  // Empty/missing array-shaped fields fall through here; only force a list
  // editor when we know the field is a collection by name.
  if (name === 'ALLOWED_MODELS'
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

function adminTokenEditor(name, v) {
  // ADMIN_TOKEN is sensitive — never render the raw value.
  // Status pill + [Rotate] [Clear] buttons. Rotate opens a modal-style inline
  // form for the new token (typed twice to catch typos). Clear sets to empty.
  // After save, the previous token has 60 s grace at the server side.
  const wrap = document.createElement('div');

  function statusPill(set) {
    const span = document.createElement('span');
    span.className = 'badge ' + (set ? 'live' : '');
    span.textContent = set ? 'Set ✓' : 'Not set (loopback-only)';
    return span;
  }

  function render() {
    wrap.innerHTML = '';
    const cur = currentValue(name);
    const isSet = !!cur;

    const top = document.createElement('div');
    top.style.display = 'flex';
    top.style.gap = '0.5rem';
    top.style.alignItems = 'center';
    top.style.flexWrap = 'wrap';
    top.appendChild(statusPill(isSet));

    const rotateBtn = document.createElement('button');
    rotateBtn.type = 'button';
    rotateBtn.textContent = isSet ? '↻ Rotate' : '+ Set token';
    rotateBtn.addEventListener('click', () => {
      const t1 = prompt('Enter the new ADMIN_TOKEN (32+ chars recommended):');
      if (t1 === null) return;
      const t1s = t1.trim();
      if (!t1s) {
        alert('Empty token — use Clear to disable token auth.');
        return;
      }
      const t2 = prompt('Re-enter the new ADMIN_TOKEN to confirm:');
      if (t2 === null) return;
      if (t1s !== t2.trim()) {
        alert('Tokens do not match. No change made.');
        return;
      }
      const grace = isSet
        ? '\n\nAfter save, the previous token stays valid for 60 s as a grace '
          + 'window so this session can update its stored token.'
        : '';
      if (!confirm('Set ADMIN_TOKEN to the new value?' + grace)) return;
      setDirty(name, t1s);
      render();
    });
    top.appendChild(rotateBtn);

    if (isSet) {
      const clearBtn = document.createElement('button');
      clearBtn.type = 'button';
      clearBtn.textContent = '⌫ Clear';
      clearBtn.title = 'Disable token auth (loopback bypass remains)';
      clearBtn.addEventListener('click', () => {
        if (!confirm('Disable ADMIN_TOKEN auth?\n\n'
            + 'Loopback (127.0.0.1, ::1) remains the only gate.\n'
            + 'After save, the previous token stays valid for 60 s.')) return;
        setDirty(name, '');
        render();
      });
      top.appendChild(clearBtn);
    }

    // If the user has typed a new token but not saved yet, show a tiny hint.
    if (Object.prototype.hasOwnProperty.call(dirty, name)) {
      const pending = document.createElement('span');
      pending.className = 'badge';
      pending.style.color = 'var(--yellow, #f2cc60)';
      pending.textContent = 'pending — click Save to apply';
      top.appendChild(pending);
    }

    wrap.appendChild(top);
  }

  render();
  document.addEventListener('admin:dirty', (e) => {
    if (!e.detail || e.detail.name === name) render();
  });
  return wrap;
}

// =============================================================================
// MODEL_OVERRIDES editor — master-detail UI
// =============================================================================
// Sidebar lists [Global (read-only ref), per-model entries with override
// counts, + add model]. Main pane switches between a Global ref view (resolved
// global values, jump-links to the in-page edit rows) and a per-model edit
// view (6 sections + Pipeline rules checklist). Per-field rows show ●/○
// inherit/override dots, an input widget when overridden, "inherits {global}"
// + "+ override" when not. Diff-to-global toggle dims rows whose value
// matches the global. Advanced expanders persist open/closed state in
// localStorage. Pipeline rules render via makeRuleListEditor(mode="checklist").
function modelOverridesEditor(name, v) {
  // Authoritative state. Local mutable copy of the saved overrides; full
  // snapshot pushed via setDirty(name, snapshot) on every edit so the existing
  // /config/state save flow sees one consistent payload.
  let overrides = JSON.parse(JSON.stringify(v || {}));
  let selectedId = null;     // null = Global view; else a model id
  let diffMode = false;

  // Section / field grouping. Mirrors the plan's master-detail layout.
  const SECTIONS = [
    { id: 'hardware', title: 'Hardware', advTitle: 'load-time hardware',
      basic: ['MODEL_DEVICE','MODEL_COMPUTE_TYPE','MODEL_DEVICE_FALLBACK','MODEL_COMPUTE_TYPE_FALLBACK'],
      adv:   ['REVISION','NUM_WORKERS','DEVICE_INDEX'] },
    { id: 'decode', title: 'Decode params', advTitle: 'beam & sampling',
      basic: ['DEFAULT_LANGUAGE','DEFAULT_PROMPT','DEFAULT_HOTWORDS',
              'BEAM_SIZE','BEST_OF',
              'CONDITION_ON_PREVIOUS_TEXT','WORD_TIMESTAMPS_ENABLED',
              'NO_SPEECH_THRESHOLD','LOG_PROB_THRESHOLD','COMPRESSION_RATIO_THRESHOLD'],
      adv:   ['TEMPERATURE','PATIENCE','LENGTH_PENALTY','REPETITION_PENALTY',
              'NO_REPEAT_NGRAM_SIZE','PROMPT_RESET_ON_TEMPERATURE'] },
    { id: 'vad', title: 'VAD', advTitle: null,
      basic: ['VAD_FILTER','VAD_MIN_SILENCE_MS','VAD_SPEECH_PAD_MS','VAD_THRESHOLD'],
      adv:   [] },
    { id: 'langdet', title: 'Language detection (active when DEFAULT_LANGUAGE empty)',
      advTitle: 'all detection knobs',
      basic: [],
      adv:   ['MULTILINGUAL','LANGUAGE_DETECTION_THRESHOLD','LANGUAGE_DETECTION_SEGMENTS'] },
    { id: 'antihalluc', title: 'Anti-hallucination & token control',
      advTitle: 'all anti-hallucination knobs',
      basic: [],
      adv:   ['HALLUCINATION_SILENCE_THRESHOLD','SUPPRESS_BLANK','SUPPRESS_TOKENS',
              'PREPEND_PUNCTUATIONS','APPEND_PUNCTUATIONS'] },
    { id: 'output', title: 'Output wrappers', advTitle: null,
      basic: ['OUTPUT_PREFIX','OUTPUT_SUFFIX'], adv: [] },
  ];

  // LOAD_TIME_FIELDS subset that overlaps with ModelOverride. Editing any of
  // these triggers drain-then-evict on save (no service restart).
  // Mirror of config_store.LOAD_TIME_FIELDS minus globals-only entries.
  const LOAD_TIME_FIELDS = new Set([
    'MODEL_DEVICE','MODEL_COMPUTE_TYPE',
    'MODEL_DEVICE_FALLBACK','MODEL_COMPUTE_TYPE_FALLBACK',
    'REVISION','NUM_WORKERS','DEVICE_INDEX',
  ]);

  // Per-field widget metadata. Mirrors ModelOverride pydantic constraints
  // (config_store.py:481+). Kept compact — extend a row only if behavior
  // differs from a generic input.
  const FIELD_META = {
    MODEL_DEVICE:                { kind: 'enum', opts: ['cuda','cpu'] },
    MODEL_COMPUTE_TYPE:          { kind: 'enum', opts: ['float16','int8_float16','int8','float32','bfloat16'] },
    MODEL_DEVICE_FALLBACK:       { kind: 'enum', opts: ['cuda','cpu'] },
    MODEL_COMPUTE_TYPE_FALLBACK: { kind: 'enum', opts: ['float16','int8_float16','int8','float32','bfloat16'] },
    REVISION:                    { kind: 'string', placeholder: 'main | <git-sha>' },
    NUM_WORKERS:                 { kind: 'int',   min: 1, max: 8 },
    DEVICE_INDEX:                { kind: 'int',   min: 0, max: 15 },
    DEFAULT_LANGUAGE:            { kind: 'string', placeholder: 'e.g. en, de (empty = auto)' },
    DEFAULT_PROMPT:              { kind: 'textarea' },
    DEFAULT_HOTWORDS:            { kind: 'textarea' },
    BEAM_SIZE:                   { kind: 'int',   min: 1, max: 20 },
    BEST_OF:                     { kind: 'int',   min: 1, max: 20 },
    CONDITION_ON_PREVIOUS_TEXT:  { kind: 'bool' },
    WORD_TIMESTAMPS_ENABLED:     { kind: 'bool' },
    NO_SPEECH_THRESHOLD:         { kind: 'nullable_float', min: 0, max: 1, step: 0.05 },
    LOG_PROB_THRESHOLD:          { kind: 'nullable_float', min: -10, max: 0, step: 0.1 },
    COMPRESSION_RATIO_THRESHOLD: { kind: 'nullable_float', min: 0, max: 10, step: 0.1 },
    TEMPERATURE:                 { kind: 'string', placeholder: '0.0,0.2,0.4,0.6,0.8,1.0' },
    PATIENCE:                    { kind: 'float', min: 0.5, max: 5,   step: 0.1 },
    LENGTH_PENALTY:              { kind: 'float', min: 0.1, max: 5,   step: 0.1 },
    REPETITION_PENALTY:          { kind: 'float', min: 0.5, max: 5,   step: 0.05 },
    NO_REPEAT_NGRAM_SIZE:        { kind: 'int',   min: 0, max: 10 },
    PROMPT_RESET_ON_TEMPERATURE: { kind: 'float', min: 0, max: 1,     step: 0.05 },
    VAD_FILTER:                  { kind: 'bool' },
    VAD_MIN_SILENCE_MS:          { kind: 'int',   min: 0, max: 10000 },
    VAD_SPEECH_PAD_MS:           { kind: 'int',   min: 0, max: 2000 },
    VAD_THRESHOLD:               { kind: 'float', min: 0, max: 1,     step: 0.05 },
    MULTILINGUAL:                { kind: 'bool' },
    LANGUAGE_DETECTION_THRESHOLD:{ kind: 'float', min: 0, max: 1,     step: 0.05 },
    LANGUAGE_DETECTION_SEGMENTS: { kind: 'int',   min: 1, max: 10 },
    HALLUCINATION_SILENCE_THRESHOLD: { kind: 'nullable_float', min: 0, max: 60, step: 0.5 },
    SUPPRESS_BLANK:              { kind: 'bool' },
    SUPPRESS_TOKENS:             { kind: 'string', placeholder: '-1 | comma-ints | (empty = none)' },
    PREPEND_PUNCTUATIONS:        { kind: 'string' },
    APPEND_PUNCTUATIONS:         { kind: 'string' },
    OUTPUT_PREFIX:               { kind: 'string' },
    OUTPUT_SUFFIX:               { kind: 'string' },
  };

  // -------- helpers ------------------------------------------------------
  function getOverrideValue(modelId, field) {
    const m = overrides[modelId];
    if (!m) return undefined;
    const val = m[field];
    return (val === null || val === undefined) ? undefined : val;
  }
  function setOverrideValue(modelId, field, value) {
    if (!overrides[modelId]) overrides[modelId] = {};
    if (value === undefined) {
      delete overrides[modelId][field];
    } else {
      overrides[modelId][field] = value;
    }
    persist();
  }
  function clearOverride(modelId, field) {
    if (overrides[modelId]) delete overrides[modelId][field];
    persist();
  }
  function countOverrides(modelId) {
    const m = overrides[modelId];
    if (!m) return 0;
    return Object.keys(m).filter(k => m[k] !== undefined && m[k] !== null).length;
  }
  function persist() {
    setDirty(name, JSON.parse(JSON.stringify(overrides)));
  }
  // Resolve the global value for a given field. Reads via currentValue() so
  // unsaved edits in the surrounding global field rows propagate live (the
  // "inherits {global}" hint reflects the in-progress global edit).
  function globalValue(field) {
    try { return currentValue(field); } catch (_) { return null; }
  }
  function fmtValue(v) {
    if (v === null || v === undefined) return '∅';
    if (typeof v === 'boolean') return v ? '☑ on' : '☐ off';
    if (Array.isArray(v))       return '[' + v.length + ' item' + (v.length === 1 ? '' : 's') + ']';
    if (typeof v === 'string')  return v === '' ? '""' : v;
    return String(v);
  }

  // -------- DOM scaffolding ----------------------------------------------
  const wrap = document.createElement('div');
  wrap.className = 'mo-wrap';
  const sidebar = document.createElement('aside');
  sidebar.className = 'mo-sidebar';
  const mainpane = document.createElement('div');
  mainpane.className = 'mo-mainpane';
  wrap.appendChild(sidebar);
  wrap.appendChild(mainpane);

  // -------- sidebar -------------------------------------------------------
  function renderSidebar() {
    sidebar.innerHTML = '';
    const allowed = Array.isArray(currentValue('ALLOWED_MODELS'))
      ? currentValue('ALLOWED_MODELS') : [];
    // List every key in overrides, even those with 0 fields (admin just added
    // an entry). Sort alphabetically for deterministic order.
    const ids = Object.keys(overrides).sort();

    // Item #1: Global (read-only ref). Dot is filled if ANY model has any override.
    const gItem = document.createElement('div');
    gItem.className = 'mo-list-item';
    if (selectedId === null) gItem.classList.add('active');
    const anyOverridden = ids.some(id => countOverrides(id) > 0);
    gItem.innerHTML = '<span class="mo-dot ' + (anyOverridden ? 'override' : 'inherit') + '"></span>'
      + '<span class="mo-list-label">Global (read-only ref)</span>';
    gItem.addEventListener('click', () => { selectedId = null; renderAll(); });
    sidebar.appendChild(gItem);

    // Per-model entries.
    for (const id of ids) {
      const item = document.createElement('div');
      item.className = 'mo-list-item';
      if (id === selectedId) item.classList.add('active');
      const n = countOverrides(id);
      const dot = '<span class="mo-dot ' + (n > 0 ? 'override' : 'inherit') + '"></span>';
      // Long HF-style ids would blow the sidebar width — let CSS truncate.
      const lbl = '<span class="mo-list-label" title="' + id + '">' + id + '</span>';
      const cnt = '<span class="mo-count">(' + n + ')</span>';
      item.innerHTML = dot + lbl + cnt;
      item.addEventListener('click', () => { selectedId = id; renderAll(); });
      sidebar.appendChild(item);
    }

    // + add model — inline dropdown of allowed models without an entry yet.
    const addRow = document.createElement('div');
    addRow.className = 'mo-list-add';
    const free = allowed.filter(m => !overrides[m]);
    if (free.length > 0) {
      const sel = document.createElement('select');
      const ph = document.createElement('option');
      ph.value = ''; ph.textContent = '+ add model override...';
      ph.selected = true; ph.disabled = true;
      sel.appendChild(ph);
      for (const m of free) {
        const o = document.createElement('option'); o.value = m; o.textContent = m;
        sel.appendChild(o);
      }
      sel.addEventListener('change', () => {
        if (!sel.value) return;
        overrides[sel.value] = {};
        selectedId = sel.value;
        persist();
        renderAll();
      });
      addRow.appendChild(sel);
    } else if (allowed.length === 0) {
      const note = document.createElement('div');
      note.className = 'help';
      note.textContent = 'ALLOWED_MODELS is empty — add models in the Models section above.';
      addRow.appendChild(note);
    } else {
      const note = document.createElement('div');
      note.className = 'help';
      note.textContent = 'Every allowed model already has an entry.';
      addRow.appendChild(note);
    }
    sidebar.appendChild(addRow);

    // Diff-to-global toggle — dims rows whose value matches the global.
    const diffWrap = document.createElement('label');
    diffWrap.className = 'cb-row mo-diff-toggle';
    const diffCb = document.createElement('input');
    diffCb.type = 'checkbox'; diffCb.checked = diffMode;
    diffCb.addEventListener('change', () => {
      diffMode = diffCb.checked;
      mainpane.classList.toggle('diff-mode', diffMode);
    });
    diffWrap.appendChild(diffCb);
    const diffLbl = document.createElement('span');
    diffLbl.textContent = 'Compare diff to global';
    diffLbl.title = 'Dim rows whose value equals the resolved global default';
    diffWrap.appendChild(diffLbl);
    sidebar.appendChild(diffWrap);

    // Remove-current-model affordance — only when a model is selected.
    if (selectedId) {
      const rmBtn = document.createElement('button');
      rmBtn.type = 'button';
      rmBtn.className = 'reset-link mo-remove-btn';
      rmBtn.textContent = '× remove model entry';
      rmBtn.title = 'Delete all overrides for ' + selectedId;
      rmBtn.addEventListener('click', () => {
        if (!confirm('Delete all overrides for ' + selectedId + '?')) return;
        delete overrides[selectedId];
        selectedId = null;
        persist();
        renderAll();
      });
      sidebar.appendChild(rmBtn);
    }
  }

  // -------- main pane: dispatcher ----------------------------------------
  function renderMain() {
    mainpane.innerHTML = '';
    mainpane.classList.toggle('diff-mode', diffMode);
    if (selectedId === null) {
      renderGlobalRefView();
    } else {
      renderModelEditView();
    }
  }

  // -------- main pane: Global ref view -----------------------------------
  function renderGlobalRefView() {
    const intro = document.createElement('div');
    intro.className = 'help';
    intro.textContent = 'Resolved global defaults (read-only). Edit them in their normal sections elsewhere on this page.';
    mainpane.appendChild(intro);
    for (const sec of SECTIONS) {
      const secEl = document.createElement('div');
      secEl.className = 'mo-section';
      const h4 = document.createElement('h4');
      h4.textContent = sec.title;
      secEl.appendChild(h4);
      const allFields = [...sec.basic, ...sec.adv];
      for (const f of allFields) {
        const row = document.createElement('div');
        row.className = 'mo-row mo-row-ref';
        const dot = document.createElement('span');
        dot.className = 'mo-dot inherit';
        const lbl = document.createElement('span');
        lbl.className = 'mo-name';
        lbl.textContent = f;
        const val = document.createElement('span');
        val.className = 'mo-value-cell mo-inherits';
        val.textContent = fmtValue(globalValue(f));
        const jump = document.createElement('button');
        jump.type = 'button';
        jump.className = 'reset-link';
        jump.textContent = '↑ edit';
        jump.title = 'Scroll to the global field row';
        jump.addEventListener('click', () => jumpToField(f));
        row.appendChild(dot);
        row.appendChild(lbl);
        row.appendChild(val);
        row.appendChild(jump);
        secEl.appendChild(row);
      }
      mainpane.appendChild(secEl);
    }
  }

  // -------- main pane: Model edit view -----------------------------------
  function renderModelEditView() {
    const head = document.createElement('div');
    head.className = 'mo-edit-head';
    head.innerHTML = '<strong>Editing: </strong><span class="mo-name">' + selectedId + '</span>';
    mainpane.appendChild(head);
    for (const sec of SECTIONS) {
      mainpane.appendChild(renderSection(sec));
    }
    mainpane.appendChild(renderPipelineSection());
  }

  function renderSection(sec) {
    const secEl = document.createElement('div');
    secEl.className = 'mo-section';
    const h4 = document.createElement('h4');
    h4.textContent = sec.title;
    if (sec.id === 'hardware') {
      const reload = document.createElement('span');
      reload.className = 'badge restart';
      reload.textContent = 'drain-then-evict on edit';
      reload.style.marginLeft = '0.5rem';
      h4.appendChild(reload);
    }
    secEl.appendChild(h4);
    for (const f of sec.basic) {
      secEl.appendChild(renderFieldRow(f));
    }
    if (sec.adv.length) {
      const det = document.createElement('details');
      det.className = 'subgroup-details mo-advanced';
      const sum = document.createElement('summary');
      sum.className = 'subgroup-summary';
      sum.textContent = 'Advanced — ' + (sec.advTitle || 'more knobs');
      det.appendChild(sum);
      const lsKey = 'mo.adv.' + sec.id;
      det.open = localStorage.getItem(lsKey) === '1';
      det.addEventListener('toggle', () => {
        try { localStorage.setItem(lsKey, det.open ? '1' : '0'); } catch (_) {}
      });
      const advWrap = document.createElement('div');
      for (const f of sec.adv) advWrap.appendChild(renderFieldRow(f));
      det.appendChild(advWrap);
      secEl.appendChild(det);
    }
    return secEl;
  }

  function renderFieldRow(field) {
    const meta = FIELD_META[field] || { kind: 'string' };
    const overrideVal = getOverrideValue(selectedId, field);
    const isOverridden = overrideVal !== undefined;
    const globalVal = globalValue(field);

    const row = document.createElement('div');
    row.className = 'mo-row mo-row-edit';
    // Use a distinct attribute (not data-field) so the diff dim selector
    // doesn't collide with the global fieldRow's data-field. The global
    // editor uses data-field on its rows; we use data-mo-field to keep the
    // jump-link selectors unambiguous.
    row.dataset.moField = field;
    row.dataset.overridden = isOverridden ? 'true' : 'false';
    // matches-global: not overridden, OR overridden but the value happens to
    // equal the global. The latter is degenerate but cheap to detect, and
    // useful — the row is functionally inherited.
    const matches = !isOverridden || JSON.stringify(overrideVal) === JSON.stringify(globalVal);
    row.dataset.matchesGlobal = matches ? 'true' : 'false';

    const dot = document.createElement('span');
    dot.className = 'mo-dot ' + (isOverridden ? 'override' : 'inherit');
    row.appendChild(dot);

    const labelWrap = document.createElement('span');
    labelWrap.className = 'mo-name';
    labelWrap.textContent = field;
    if (LOAD_TIME_FIELDS.has(field)) {
      const reload = document.createElement('span');
      reload.className = 'badge restart';
      reload.textContent = '⚠ reload';
      reload.title = 'Editing this field evicts and reloads the model on save';
      reload.style.marginLeft = '0.4rem';
      labelWrap.appendChild(reload);
    }
    row.appendChild(labelWrap);

    const valueCell = document.createElement('span');
    valueCell.className = 'mo-value-cell';
    if (isOverridden) {
      valueCell.appendChild(makeInputWidget(field, meta, overrideVal));
    } else {
      const inh = document.createElement('span');
      inh.className = 'mo-inherits';
      inh.textContent = 'inherits ' + fmtValue(globalVal);
      valueCell.appendChild(inh);
    }
    row.appendChild(valueCell);

    const ctrl = document.createElement('span');
    ctrl.className = 'mo-row-ctrl';
    if (isOverridden) {
      const reset = document.createElement('button');
      reset.type = 'button';
      reset.className = 'reset-link';
      reset.textContent = '↶ reset';
      reset.title = 'Restore inherited value (' + fmtValue(globalVal) + ')';
      reset.addEventListener('click', () => {
        clearOverride(selectedId, field);
        renderMain();
        renderSidebar();
      });
      ctrl.appendChild(reset);
    } else {
      const add = document.createElement('button');
      add.type = 'button';
      add.className = 'reset-link';
      add.textContent = '+ override';
      add.title = 'Set a per-model override starting from the inherited value';
      add.addEventListener('click', () => {
        // Seed with the current global value so the admin starts from the
        // inherited value (not from a zero/empty default that overrides it).
        const seed = (globalVal === null || globalVal === undefined)
          ? defaultForKind(meta.kind) : globalVal;
        setOverrideValue(selectedId, field, seed);
        renderMain();
        renderSidebar();
      });
      ctrl.appendChild(add);
    }
    row.appendChild(ctrl);
    return row;
  }

  function defaultForKind(kind) {
    if (kind === 'bool') return false;
    if (kind === 'int' || kind === 'float' || kind === 'nullable_float') return 0;
    return '';
  }

  function makeInputWidget(field, meta, currentVal) {
    if (meta.kind === 'bool') {
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = !!currentVal;
      cb.addEventListener('change', () => setOverrideValue(selectedId, field, cb.checked));
      return cb;
    }
    if (meta.kind === 'enum') {
      const sel = document.createElement('select');
      for (const o of (meta.opts || [])) {
        const opt = document.createElement('option');
        opt.value = o; opt.textContent = o;
        if (o === currentVal) opt.selected = true;
        sel.appendChild(opt);
      }
      sel.addEventListener('change', () => setOverrideValue(selectedId, field, sel.value));
      return sel;
    }
    if (meta.kind === 'int' || meta.kind === 'float') {
      const inp = document.createElement('input');
      inp.type = 'number';
      if (meta.min !== undefined) inp.min = meta.min;
      if (meta.max !== undefined) inp.max = meta.max;
      if (meta.step !== undefined) inp.step = meta.step;
      else if (meta.kind === 'int') inp.step = 1;
      inp.value = currentVal == null ? '' : currentVal;
      inp.addEventListener('input', () => {
        const raw = inp.value;
        if (raw === '') return;     // wait for a complete number
        const n = meta.kind === 'int' ? parseInt(raw, 10) : parseFloat(raw);
        if (Number.isFinite(n)) setOverrideValue(selectedId, field, n);
      });
      return inp;
    }
    if (meta.kind === 'nullable_float') {
      // Empty input = null = "explicitly disabled" override (different from
      // not-overridden, which is "use global"). Send null on empty.
      const inp = document.createElement('input');
      inp.type = 'number';
      if (meta.min !== undefined) inp.min = meta.min;
      if (meta.max !== undefined) inp.max = meta.max;
      if (meta.step !== undefined) inp.step = meta.step;
      inp.value = currentVal == null ? '' : currentVal;
      inp.addEventListener('input', () => {
        const raw = inp.value;
        if (raw === '') {
          setOverrideValue(selectedId, field, null);
          return;
        }
        const n = parseFloat(raw);
        if (Number.isFinite(n)) setOverrideValue(selectedId, field, n);
      });
      return inp;
    }
    if (meta.kind === 'textarea') {
      const ta = document.createElement('textarea');
      ta.rows = 3;
      ta.value = currentVal == null ? '' : currentVal;
      ta.addEventListener('input', () => setOverrideValue(selectedId, field, ta.value));
      return ta;
    }
    // string fallback (and TEMPERATURE / SUPPRESS_TOKENS / PUNCTUATIONS)
    const inp = document.createElement('input');
    inp.type = 'text';
    if (meta.placeholder) inp.placeholder = meta.placeholder;
    inp.value = currentVal == null ? '' : currentVal;
    inp.addEventListener('input', () => setOverrideValue(selectedId, field, inp.value));
    return inp;
  }

  // -------- main pane: pipeline rules checklist --------------------------
  function renderPipelineSection() {
    const secEl = document.createElement('div');
    secEl.className = 'mo-section';
    const h4 = document.createElement('h4');
    h4.textContent = 'Pipeline rules (scoping only)';
    secEl.appendChild(h4);
    const note = document.createElement('div');
    note.className = 'help';
    note.textContent = 'Per-model pipeline scoping. Rule bodies are edited globally — '
      + 'the checkbox here decides whether each rule runs in THIS model\'s pipeline. '
      + 'Globally-enabled rules can be force-disabled by unchecking; globally-disabled '
      + 'rules can be force-enabled by checking. The two modes are mutually exclusive '
      + 'per slug (a rule cannot be both force-disabled and force-enabled).';
    secEl.appendChild(note);

    let rules = [];
    try { rules = currentValue('PIPELINE_RULES') || []; } catch (_) { rules = []; }
    const curEx = (overrides[selectedId] && overrides[selectedId].PIPELINE_RULES_EXCLUDE) || [];
    const curIn = (overrides[selectedId] && overrides[selectedId].PIPELINE_RULES_INCLUDE) || [];
    const excludeSet = new Set(curEx);
    const includeSet = new Set(curIn);
    const ruleOpts = {
      excludeSet,
      includeSet,
      // Single callback for both lists. globallyEnabled is the rule's
      // current global state; we mutate the appropriate list:
      //   globallyEnabled=true  → toggle EXCLUDE (uncheck adds to EXCLUDE)
      //   globallyEnabled=false → toggle INCLUDE (check adds to INCLUDE)
      // The pydantic validator forbids same-slug overlap so we never
      // need to worry about a slug ending up in both.
      onToggle: (slug, wantActive, globallyEnabled) => {
        if (globallyEnabled) {
          if (wantActive) excludeSet.delete(slug);
          else            excludeSet.add(slug);
        } else {
          if (wantActive) includeSet.add(slug);
          else            includeSet.delete(slug);
        }
        if (!overrides[selectedId]) overrides[selectedId] = {};
        const ex = [...excludeSet];
        const inc = [...includeSet];
        if (ex.length)  overrides[selectedId].PIPELINE_RULES_EXCLUDE = ex;
        else            delete overrides[selectedId].PIPELINE_RULES_EXCLUDE;
        if (inc.length) overrides[selectedId].PIPELINE_RULES_INCLUDE = inc;
        else            delete overrides[selectedId].PIPELINE_RULES_INCLUDE;
        persist();
        renderSidebar();   // override count changed
      },
      onJumpToGlobal: (slug) => jumpToRule(slug),
    };
    secEl.appendChild(makeRuleListEditor('PIPELINE_RULES', rules, 'checklist', ruleOpts));
    return secEl;
  }

  // -------- jump-link helpers --------------------------------------------
  // The global field row uses `data-field="X"`; per-model rows here use
  // `data-mo-field="X"` to avoid clashing. Scope the global selector to
  // .field (the existing per-row class) so we always land on the right node.
  function jumpToField(field) {
    const target = document.querySelector('main .field[data-field="' + field + '"]');
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
  function jumpToRule(slug) {
    const fieldRow = document.querySelector('main .field[data-field="PIPELINE_RULES"]');
    if (!fieldRow) return;
    const ruleRow = fieldRow.querySelector('.rule-row[data-slug="' + slug + '"]');
    if (ruleRow) {
      ruleRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
      // Auto-expand the rule body if collapsed, so the admin lands directly
      // in its editor rather than just at the row header.
      if (!ruleRow.classList.contains('expanded')) {
        const btn = ruleRow.querySelector(':scope > .row-header > .expand-btn');
        if (btn) btn.click();
      }
    } else {
      fieldRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  // Initial selection: the alphabetically-first model with overrides, else
  // Global. Avoids a blank main pane on first paint when overrides exist.
  (function() {
    const ids = Object.keys(overrides).filter(id => countOverrides(id) > 0).sort();
    if (ids.length > 0) selectedId = ids[0];
  })();

  function renderAll() {
    renderSidebar();
    renderMain();
  }
  renderAll();
  // Re-render the sidebar on ALLOWED_MODELS changes (the "+ add model"
  // dropdown sources from there). Same event the DEFAULT_MODEL dropdown
  // and PRELOAD_MODELS multi-select listen for.
  document.addEventListener('admin:model-lists-changed', renderSidebar);
  return wrap;
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
      t.placeholder = 'One model id per line';
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
    list.className = 'cb-list';
    for (const m of universe) {
      const lbl = document.createElement('label');
      lbl.className = 'cb-row';
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

// =============================================================================
// PIPELINE_RULES editor — single ordered list of mixed-type rules
// =============================================================================
// Rendered as a stack of .rule-row cards. Each row: drag-handle | enabled
// checkbox | ordinal | name+label | type-pill | edit toggle | reset (seeded
// only) | delete (custom only). Body (visible when expanded) holds a
// type-specific sub-editor + per-row live status badge.
//
// Drag-to-reorder uses HTML5 native DnD on the .drag-handle. Locked rules
// (e.g. dictation-map → tidies → capitalize chain) trigger a confirm()
// dialog when dropped to a position that breaks an ordering edge.

const TEST_PRESETS = {
  'default':              "Hallo. Wie geht's? 10.23 Uhr! Bitte Frau, Müller. neuer Absatz. 1,000 EUR.",
  'german verbatim':      "Hallo. Wie geht's? 10.23 Uhr. Bitte Frau Müller. 1,000 EUR.",
  'german + dictation':   "Hallo Punkt neue Zeile Frau Komma Müller. 1,000 EUR neue Zeile Bitte fragen.",
  'english':              "Hello. How are you? It is 10:30 AM. Please ask Mrs. Smith.",
  'numbers + commas':     "Total 1,000 EUR. Range 10-23 cm. 10.23 Uhr ist 10:23. Schritt 1, 2, 3.",
};

function _slugify(s) {
  // kebab-case, ASCII-ish, drop diacritics, collapse runs of -.
  return (s || '').toLowerCase()
    .normalize('NFD').replace(/[̀-ͯ]/g, '')   // strip combining marks
    .replace(/ß/g, 'ss').replace(/[äöü]/g, m => ({'ä':'a','ö':'o','ü':'u'}[m]))
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 64) || 'rule';
}

// Display \n, \r, \t, \\ as literal 2-char escape sequences in <input>
// cells. Single-line inputs strip newlines per WHATWG spec (the value
// sanitization algorithm), so without this the user sees an empty field
// for any value containing a newline and would silently overwrite it
// with "" on save.
function _esc(s) {
  if (s == null) return '';
  return String(s)
    .replace(/\\/g, '\\\\')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t');
}
function _unesc(s) {
  if (s == null) return '';
  let out = '';
  for (let i = 0; i < s.length; i++) {
    if (s[i] === '\\' && i + 1 < s.length) {
      const nxt = s[++i];
      if (nxt === 'n') out += '\n';
      else if (nxt === 'r') out += '\r';
      else if (nxt === 't') out += '\t';
      else if (nxt === '\\') out += '\\';
      else out += nxt;       // unknown escape: pass through
    } else {
      out += s[i];
    }
  }
  return out;
}

function _ensureUniqueSlug(slug, existing) {
  if (!existing.has(slug)) return slug;
  let n = 2;
  while (existing.has(`${slug}-${n}`)) n++;
  return `${slug}-${n}`;
}

const _PIPELINE_TYPES = [
  { type: 'regex',                       pill: 'regex' },
  { type: 'callback:lowercase-wordlist', pill: 'cb:wordlist' },
  { type: 'callback:map',                pill: 'cb:map' },
  { type: 'callback:dedup',              pill: 'cb:dedup' },
  { type: 'callback:upper',              pill: 'cb:upper' },
  { type: 'terminal',                    pill: 'terminal' },
];
const _typePill = (t) => (_PIPELINE_TYPES.find(x => x.type === t) || {}).pill || t;

// Live status check for one rule against the current test panel sample.
// Hits POST /config/test-pipeline with a single-rule list. Returns the step
// dict or null on transport error.
async function _testOneRule(rule) {
  const panelSample = document.getElementById('pipeline-test-sample');
  const sample = panelSample ? panelSample.value : TEST_PRESETS['default'];
  const r = await api('POST', '/config/test-pipeline', {
    sample, rules: [rule],
  });
  if (!r.ok) return null;
  const j = await r.json();
  return (j.steps && j.steps[0]) || null;
}

function makeRuleListEditor(name, initialRules, mode, opts) {
  // mode: "full" (default) → editable list with drag-reorder, add/delete,
  //         per-row body editor, per-row test badge. Used for the global
  //         PIPELINE_RULES editor.
  //       "checklist" → compact read-only-ish list with one checkbox per
  //         rule (writes to MODEL_OVERRIDES[id].PIPELINE_RULES_EXCLUDE) and
  //         a "↑ view" jump-link to the global editor. Used by the per-model
  //         override pane. opts.excludeSet (Set<slug>) drives initial state;
  //         opts.onToggle(slug, excluded) and opts.onJumpToGlobal(slug) are
  //         the two callbacks.
  // initialRules: list of rule dicts as stored in PIPELINE_RULES.
  // Local mutable copy lives in a closure; in "full" mode setDirty(name, snapshot)
  // on every edit so save tracks the whole list as one entry. In "checklist"
  // mode the list is purely read-only — toggles call opts.onToggle directly.
  mode = mode || 'full';
  opts = opts || {};
  const isChecklist = mode === 'checklist';
  let rules = JSON.parse(JSON.stringify(initialRules || []));
  const wrap = document.createElement('div');
  wrap.className = 'pipeline-rules-wrap';
  if (isChecklist) wrap.classList.add('checklist-mode');

  if (!isChecklist) {
    const advWarn = document.createElement('div');
    advWarn.className = 'advanced-warn';
    advWarn.innerHTML = '⚠ <strong>advanced</strong> — incorrect regex breaks transcription. '
      + 'Use the test panel below to dry-run before saving. ↺ Reset to default if you get stuck.';
    wrap.appendChild(advWarn);
  }

  const list = document.createElement('div');
  list.className = 'rule-list';
  wrap.appendChild(list);

  // Drag state across rule rows.
  let dragSrcIdx = null;
  let dragSrcEl = null;     // the <div class="rule-row"> currently dragging
  // One shared placeholder div — HTML5 DnD allows only one drag at a time
  // per browser tab, so a single node is sufficient. Inserted on dragstart
  // (after a setTimeout(0) so Chrome's drag-image snapshot captures the
  // full row), moves around on dragenter, removed on dragend.
  const placeholder = document.createElement('div');
  placeholder.className = 'rule-placeholder';
  // Expanded-row state, keyed by rule.name (slug). Survives full repaints.
  const expandedNames = new Set();

  // List-level (delegated) drag handlers. dragenter moves the placeholder
  // around as the cursor crosses rows; drop fires once at the placeholder's
  // final position. Per-row handlers only fire dragstart/dragend.
  // Checklist mode is read-only — drag-to-reorder is meaningless there.
  if (!isChecklist) {
  list.addEventListener('dragover', (e) => {
    if (!dragSrcEl) return;
    e.preventDefault();                          // required for drop to fire
    e.dataTransfer.dropEffect = 'move';
  });
  list.addEventListener('dragenter', (e) => {
    if (!dragSrcEl) return;
    const targetRow = e.target.closest && e.target.closest('.rule-row');
    if (!targetRow || targetRow === dragSrcEl) return;
    // Terminal row stays last — placeholder snaps just above it.
    if (targetRow.classList.contains('terminal')) {
      list.insertBefore(placeholder, targetRow);
      return;
    }
    const rect = targetRow.getBoundingClientRect();
    const before = (e.clientY - rect.top) < rect.height / 2;
    list.insertBefore(placeholder, before ? targetRow : targetRow.nextSibling);
  });
  list.addEventListener('drop', (e) => {
    if (!dragSrcEl) return;
    e.preventDefault();
    // Convert placeholder DOM position → rules-array index by counting
    // visible .rule-row siblings before it (source is hidden, excluded).
    let newIdx = 0;
    for (const child of list.children) {
      if (child === placeholder) break;
      if (child.classList.contains('rule-row') && child !== dragSrcEl) newIdx++;
    }
    const oldIdx = dragSrcIdx;
    if (newIdx === oldIdx) return;     // dropped in place
    const src = rules[oldIdx];
    const targetEl = placeholder.nextElementSibling;
    const movingLocked = src.locked
      || (targetEl && targetEl.classList && targetEl.classList.contains('locked'));
    if (movingLocked) {
      const ok = confirm(
        'Reordering ' + src.name + ' near a locked rule may break the\n' +
        'pipeline (e.g. dictation must run before its tidy rules).\n\n' +
        'Proceed anyway?'
      );
      if (!ok) return;                 // dragend cleans up
    }
    const [moved] = rules.splice(oldIdx, 1);
    rules.splice(newIdx, 0, moved);
    // Defensive: keep terminal last.
    const tIdx = rules.findIndex(r => r.type === 'terminal');
    if (tIdx >= 0 && tIdx !== rules.length - 1) {
      const [tr] = rules.splice(tIdx, 1);
      rules.push(tr);
    }
    commitFull();
  });
  }  // end if (!isChecklist) — drag handlers

  // --- baseline / dirtiness helpers (drive reset-button visibility) ----
  function _baselineList() { return fieldDef(name).default_value || []; }
  function _baselineByName() {
    const m = new Map();
    _baselineList().forEach(b => m.set(b.name, b));
    return m;
  }
  function _isRuleDirty(rule) {
    if (!rule || !rule.seeded) return false;
    const baseline = _baselineByName().get(rule.name);
    if (!baseline) return false;
    return JSON.stringify(rule) !== JSON.stringify(baseline);
  }
  function _seededOrderDirty() {
    const baseSeeded = _baselineList().filter(b => b.seeded).map(b => b.name);
    const curSeeded = rules.filter(r => r.seeded).map(r => r.name);
    return JSON.stringify(curSeeded) !== JSON.stringify(baseSeeded);
  }
  function _anyRuleDirty() { return rules.some(r => _isRuleDirty(r)); }

  function refreshControlsVisibility() {
    if (isChecklist) return;   // checklist rows have no reset/dirty controls
    // Per-row reset buttons: hide when rule matches its baseline.
    list.querySelectorAll('.rule-row').forEach(r => {
      const idx = parseInt(r.dataset.idx, 10);
      const rule = rules[idx];
      if (!rule) return;
      const btn = r.querySelector(':scope > .row-header > .reset-link');
      if (btn) btn.style.display = _isRuleDirty(rule) ? '' : 'none';
    });
    // List-wide controls: hide when nothing to reset.
    if (resetOrderBtn) {
      resetOrderBtn.style.display = _seededOrderDirty() ? '' : 'none';
    }
    if (resetAllBtn) {
      resetAllBtn.style.display = (_anyRuleDirty() || _seededOrderDirty())
        ? '' : 'none';
    }
  }

  // --- commit helpers --------------------------------------------------
  // Inline edits (typing into pattern/replacement/wordlist/map) MUST NOT
  // rebuild the DOM — that would steal focus mid-keystroke and collapse
  // any other expanded rows. Structural changes (add/delete/reorder/reset/
  // toggle-enabled) DO rebuild because the visible row layout changes.
  function commitData() {
    if (isChecklist) return;   // checklist mode is read-only — no setDirty
    setDirty(name, JSON.parse(JSON.stringify(rules)));
    refreshControlsVisibility();
  }
  function commitFull() {
    if (isChecklist) {
      paintAll();
      return;
    }
    setDirty(name, JSON.parse(JSON.stringify(rules)));
    paintAll();
  }

  function paintAll() {
    list.innerHTML = '';
    rules.forEach((rule, idx) => list.appendChild(renderRow(rule, idx)));
    refreshControlsVisibility();
  }

  function renderRow(rule, idx) {
    const row = document.createElement('div');
    row.className = 'rule-row';
    row.dataset.idx = idx;
    row.dataset.slug = rule.name || '';

    // -------- Checklist mode: compact one-checkbox row + jump-link --------
    // No drag handle, no body, no test badge, no add/delete/reset. Bodies
    // are still edited globally; this row is just an inclusion/exclusion
    // toggle for the model selected in the master-detail UI.
    //
    // Effective state: (rule.enabled && !forcedOut) || forcedIn.
    //   forcedOut = slug in PIPELINE_RULES_EXCLUDE → force-disable
    //   forcedIn  = slug in PIPELINE_RULES_INCLUDE → force-enable a globally-
    //               disabled rule (the only way an off rule can run for one model)
    // The pydantic validator forbids same-slug overlap so the two are
    // mutually exclusive on the wire.
    if (isChecklist) {
      const globallyEnabled = !!rule.enabled;
      const forcedOut = !!(opts.excludeSet && opts.excludeSet.has(rule.name));
      const forcedIn  = !!(opts.includeSet && opts.includeSet.has(rule.name));
      const effective = (globallyEnabled && !forcedOut) || forcedIn;
      const isTerminal = rule.type === 'terminal';

      if (isTerminal) row.classList.add('terminal');
      if (forcedOut) row.classList.add('excluded');
      // Globally-disabled rules render dimmed UNTIL the admin force-includes
      // them. Once force-included the row is "alive" — undim it so the
      // effective state is visible at a glance.
      if (!globallyEnabled && !forcedIn) row.classList.add('globally-disabled');
      if (!globallyEnabled && forcedIn) row.classList.add('force-included');

      const head = document.createElement('div');
      head.className = 'row-header';

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = effective;
      if (isTerminal) {
        cb.disabled = true;
        cb.title = 'Terminal trim — always runs, cannot be excluded per model';
      } else {
        // Tooltip combines all the state so the admin sees WHY the box is
        // checked or not at a glance.
        if (!globallyEnabled && forcedIn) {
          cb.title = 'Globally disabled, force-included for this model. '
                   + 'Uncheck to fall back to the global default (off).';
        } else if (!globallyEnabled) {
          cb.title = 'Globally disabled. Check to force-enable for this model.';
        } else if (forcedOut) {
          cb.title = 'Force-disabled for this model. Check to inherit the '
                   + 'global default (on).';
        } else {
          cb.title = 'Active for this model (inherits global enabled).';
        }
        cb.addEventListener('change', () => {
          // Hand the global state to onToggle so the parent picks the right
          // list to mutate (EXCLUDE vs INCLUDE). Single callback, two paths.
          if (opts.onToggle) opts.onToggle(rule.name, cb.checked, globallyEnabled);
          // Local re-class without a full repaint — keeps focus / scroll.
          const newForcedOut = globallyEnabled && !cb.checked;
          const newForcedIn  = !globallyEnabled && cb.checked;
          row.classList.toggle('excluded', newForcedOut);
          row.classList.toggle('globally-disabled', !globallyEnabled && !newForcedIn);
          row.classList.toggle('force-included', newForcedIn);
          // Update tags + status text.
          status.textContent = newForcedOut ? 'EXCLUDED' : '';
          if (gdTag) gdTag.style.display = (!globallyEnabled && !newForcedIn) ? '' : 'none';
          if (fiTag) fiTag.style.display = (!globallyEnabled && newForcedIn) ? '' : 'none';
          if (footer) footer.textContent = _footerText();
        });
      }
      head.appendChild(cb);

      const lbl = document.createElement('span');
      lbl.className = 'rule-label';
      lbl.textContent = rule.label || rule.name;
      head.appendChild(lbl);

      const pill = document.createElement('span');
      pill.className = 'type-pill';
      pill.textContent = _typePill(rule.type);
      head.appendChild(pill);

      if (!isTerminal) {
        const view = document.createElement('button');
        view.type = 'button';
        view.className = 'reset-link';
        view.textContent = '↑ view';
        view.title = 'Scroll to global PIPELINE_RULES editor and expand this rule';
        view.addEventListener('click', () => {
          if (opts.onJumpToGlobal) opts.onJumpToGlobal(rule.name);
        });
        head.appendChild(view);
      }

      // Tags: surface global state alongside per-model overrides. Visible
      // only when the relevant condition holds; toggled live by the change
      // handler above.
      let gdTag = null;
      let fiTag = null;
      if (!isTerminal) {
        gdTag = document.createElement('span');
        gdTag.className = 'rule-globally-disabled-tag';
        gdTag.textContent = 'globally disabled';
        gdTag.title = 'This rule is disabled in the global pipeline. '
                    + 'Check the box to force-enable it for this model only.';
        gdTag.style.display = (!globallyEnabled && !forcedIn) ? '' : 'none';
        head.appendChild(gdTag);

        fiTag = document.createElement('span');
        fiTag.className = 'rule-force-included-tag';
        fiTag.textContent = 'force-included';
        fiTag.title = 'Globally disabled but force-enabled for this model.';
        fiTag.style.display = (!globallyEnabled && forcedIn) ? '' : 'none';
        head.appendChild(fiTag);
      }

      const status = document.createElement('span');
      status.className = 'rule-checklist-status';
      status.textContent = forcedOut ? 'EXCLUDED' : '';
      head.appendChild(status);

      row.appendChild(head);
      return row;
    }
    // ----------------------------------------------------------------------

    row.tabIndex = 0;     // keyboard-focusable; CSS :focus-visible draws ring
    if (rule.locked) row.classList.add('locked');
    if (!rule.enabled) row.classList.add('disabled');
    if (rule.type === 'terminal') row.classList.add('terminal');
    // Restore expansion across repaints (state lives in expandedNames Set).
    if (expandedNames.has(rule.name)) row.classList.add('expanded');

    // Header row.
    const head = document.createElement('div');
    head.className = 'row-header';

    const drag = document.createElement('span');
    drag.className = 'drag-handle';
    drag.textContent = rule.locked ? '🔒' : '⋮⋮';
    drag.title = rule.locked
      ? 'Locked — reorder will warn before applying'
      : 'Drag to reorder';
    if (rule.type !== 'terminal') {
      drag.draggable = true;
      drag.addEventListener('dragstart', (e) => {
        dragSrcIdx = idx;
        dragSrcEl = row;
        // Firefox requires setData() — without it, the drag never fires.
        try { e.dataTransfer.setData('text/plain', String(idx)); } catch (_) {}
        e.dataTransfer.effectAllowed = 'move';
        // Whole row as the drag ghost (not just the handle glyph) so
        // the user sees what they're moving. Must run synchronously.
        try { e.dataTransfer.setDragImage(row, 12, 12); } catch (_) {}
        // Match placeholder height to source so layout doesn't jump
        // when we hide the source on the next tick.
        placeholder.style.height = row.offsetHeight + 'px';
        list.classList.add('dnd-active');
        // CRITICAL: defer hide + class-add so Chrome's drag-image
        // snapshot (taken at end of the dragstart tick) captures the
        // full row, not a blank ghost.
        setTimeout(() => {
          row.classList.add('dragging');
          row.parentNode.insertBefore(placeholder, row.nextSibling);
          row.style.display = 'none';
        }, 0);
      });
      drag.addEventListener('dragend', () => {
        // Always cleanup here — fires even on cancelled drops (Esc, off-screen).
        row.style.display = '';
        row.classList.remove('dragging');
        if (placeholder.parentNode) placeholder.remove();
        list.classList.remove('dnd-active');
        dragSrcEl = null;
        dragSrcIdx = null;
      });
    }
    head.appendChild(drag);

    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = !!rule.enabled;
    cb.title = 'Enable / disable this rule';
    if (rule.type === 'terminal') cb.disabled = true;
    cb.addEventListener('change', () => {
      rule.enabled = cb.checked;
      commitFull();
    });
    head.appendChild(cb);

    const ord = document.createElement('span');
    ord.className = 'ordinal';
    ord.textContent = String(idx + 1);
    head.appendChild(ord);

    const lbl = document.createElement('span');
    lbl.className = 'rule-label';
    lbl.textContent = rule.label || rule.name;
    head.appendChild(lbl);

    const slug = document.createElement('span');
    slug.className = 'rule-slug';
    slug.textContent = '(' + (rule.name || '?') + ')';
    head.appendChild(slug);

    const pill = document.createElement('span');
    pill.className = 'type-pill';
    pill.textContent = _typePill(rule.type);
    head.appendChild(pill);

    const expandBtn = document.createElement('button');
    expandBtn.type = 'button';
    expandBtn.className = 'expand-btn';
    const _setExpandLabel = () => {
      expandBtn.textContent = row.classList.contains('expanded') ? 'edit ▴' : 'edit ▾';
    };
    _setExpandLabel();
    expandBtn.addEventListener('click', () => {
      row.classList.toggle('expanded');
      if (row.classList.contains('expanded')) expandedNames.add(rule.name);
      else expandedNames.delete(rule.name);
      _setExpandLabel();
    });
    head.appendChild(expandBtn);

    if (rule.seeded) {
      const reset = document.createElement('button');
      reset.type = 'button';
      reset.className = 'reset-link';
      reset.textContent = '↺ reset';
      reset.title = 'Restore this rule to its in-repo default';
      reset.addEventListener('click', () => {
        const baseline = _baselineByName().get(rule.name);
        if (!baseline) return;
        rules[idx] = JSON.parse(JSON.stringify(baseline));
        commitFull();
      });
      head.appendChild(reset);
    } else {
      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'delete-btn';
      del.textContent = '× delete';
      del.title = 'Remove this custom rule';
      del.addEventListener('click', () => {
        expandedNames.delete(rule.name);
        rules.splice(idx, 1);
        commitFull();
      });
      head.appendChild(del);
    }
    row.appendChild(head);

    // Drop logic lives at list level (see top of makeRuleListEditor) —
    // the shared placeholder follows the cursor between rows there.

    // Body (collapsed by default).
    const body = document.createElement('div');
    body.className = 'row-body';
    body.appendChild(renderTypeEditor(rule, idx));

    if (rule.type !== 'terminal') {
      const status = document.createElement('div');
      status.className = 'regex-status empty';
      status.textContent = '∅ click "Run test" in the panel below, or edit any field to refresh';
      body.appendChild(status);

      let timer = null;
      async function refresh() {
        if (timer) clearTimeout(timer);
        timer = setTimeout(async () => {
          const step = await _testOneRule(rule);
          if (!step) {
            status.className = 'regex-status err';
            status.textContent = '✗ test endpoint error';
            return;
          }
          if (step.skipped) {
            status.className = 'regex-status empty';
            status.textContent = rule.enabled
              ? '∅ empty pattern — rule skipped'
              : '∅ disabled';
          } else if (step.error) {
            status.className = 'regex-status err';
            status.textContent = '✗ ' + step.error;
          } else if (step.slow) {
            status.className = 'regex-status warn';
            status.textContent = '⚠ slow — exceeded 2 s on sample (catastrophic backtracking?)';
          } else {
            status.className = 'regex-status ok';
            const n = step.matches || 0;
            status.textContent = '✓ valid · ' + n + ' match' + (n === 1 ? '' : 'es') + ' in sample';
          }
        }, 250);
      }
      // Refresh on any input change inside the body.
      body.addEventListener('input', refresh);
      body.addEventListener('change', refresh);
      requestAnimationFrame(refresh);
    }

    row.appendChild(body);
    return row;
  }

  function renderTypeEditor(rule, idx) {
    const box = document.createElement('div');
    box.className = 'rule-editor';

    if (rule.type === 'terminal') {
      const note = document.createElement('div');
      note.className = 'help';
      note.textContent = 'Hardcoded terminal step: lstrip(" \\t\\r") + rstrip(" \\t\\r"). '
        + 'Always runs last. Preserves a leading or trailing newline ("\\n") emitted by '
        + '"neue Zeile" / "neuer Absatz" at the edges of the utterance.';
      box.appendChild(note);
      return box;
    }

    if (rule.type === 'regex') {
      box.appendChild(_makeMonoLabeledInput('pattern', rule.pattern, (v) => {
        rule.pattern = v; commitData();
      }));
      box.appendChild(_makeMonoLabeledInput('replacement', rule.replacement, (v) => {
        rule.replacement = v; commitData();
      }, 'escape'));
      return box;
    }

    if (rule.type === 'callback:lowercase-wordlist') {
      box.appendChild(_makeMonoLabeledInput('pattern', rule.pattern, (v) => {
        rule.pattern = v; commitData();
      }));
      const wlLbl = document.createElement('div');
      wlLbl.className = 'help';
      wlLbl.textContent = 'Wordlist (one entry per line, case-insensitive):';
      box.appendChild(wlLbl);
      const ta = document.createElement('textarea');
      ta.value = (rule.wordlist || []).join('\n');
      ta.rows = 6;
      ta.addEventListener('input', () => {
        rule.wordlist = ta.value.split('\n').map(s => s.trim()).filter(Boolean);
        commitData();
      });
      box.appendChild(ta);
      return box;
    }

    if (rule.type === 'callback:map') {
      const note = document.createElement('div');
      note.className = 'help';
      note.textContent = 'Pattern auto-built from map keys (longest-first, '
        + 'word-bounded, case-insensitive). Edit entries below.';
      box.appendChild(note);
      const tbl = document.createElement('table');
      tbl.className = 'map-table';
      tbl.style.width = '100%';
      const rows = Object.entries(rule.map || {});
      rows.forEach(([k, v]) => tbl.appendChild(_makeMapRow(rule, k, v)));
      box.appendChild(tbl);
      const addBtn = document.createElement('button');
      addBtn.type = 'button';
      addBtn.textContent = '+ add entry';
      addBtn.style.marginTop = '0.4rem';
      addBtn.addEventListener('click', () => {
        // Append a new <tr> directly so the surrounding row body stays
        // expanded and other expanded rows keep their input state.
        if (!rule.map) rule.map = {};
        const k = '_new_' + Object.keys(rule.map).length;
        rule.map[k] = '';
        const newTr = _makeMapRow(rule, k, '');
        tbl.appendChild(newTr);
        commitData();
        // Focus the new key cell so the user can start typing immediately.
        const ki = newTr.querySelector('td:first-child input');
        if (ki) { ki.focus(); ki.select(); }
      });
      box.appendChild(addBtn);
      return box;
    }

    if (rule.type === 'callback:dedup' || rule.type === 'callback:upper') {
      box.appendChild(_makeMonoLabeledInput('pattern', rule.pattern, (v) => {
        rule.pattern = v; commitData();
      }));
      const note = document.createElement('div');
      note.className = 'help';
      note.textContent = rule.type === 'callback:dedup'
        ? 'Callback: collapse each match — last non-comma wins; pure-comma run → single comma.'
        : 'Callback: uppercase group(2) (or whole match if pattern has fewer than 2 groups).';
      box.appendChild(note);
      return box;
    }

    return box;
  }

  function _makeMonoLabeledInput(label, val, onInput, kind) {
    // kind === 'escape' → display \n/\r/\t/\\ as literal 2-char escapes,
    // decode on input. Required for fields like regex `replacement` that
    // can hold real newlines (single-line <input> strips them otherwise).
    const lbl = document.createElement('div');
    lbl.className = 'help';
    lbl.textContent = label + ':';
    const inp = document.createElement('input');
    inp.type = 'text';
    inp.spellcheck = false;
    inp.autocomplete = 'off';
    const raw = val == null ? '' : val;
    inp.value = (kind === 'escape') ? _esc(raw) : raw;
    inp.addEventListener('input', () => onInput(
      kind === 'escape' ? _unesc(inp.value) : inp.value
    ));
    const wrap = document.createElement('div');
    wrap.appendChild(lbl); wrap.appendChild(inp);
    return wrap;
  }

  function _makeMapRow(rule, key, val) {
    const tr = document.createElement('tr');
    const td1 = document.createElement('td');
    const td2 = document.createElement('td');
    const td3 = document.createElement('td');
    td3.style.width = '2.5rem';
    const ki = document.createElement('input');
    ki.type = 'text'; ki.value = _esc(key);
    const vi = document.createElement('input');
    vi.type = 'text'; vi.value = _esc(val);
    // Map keys/values may contain \n etc.; <input> strips real newlines,
    // so we display \n as literal 2-char escape and decode on read.
    function _readMap(parent) {
      const m = {};
      parent.querySelectorAll('tr').forEach(r => {
        const k = _unesc(r.querySelector('td:first-child input').value);
        const v = _unesc(r.querySelector('td:nth-child(2) input').value);
        if (k) m[k] = v;
      });
      return m;
    }
    function rebuild() {
      const parent = tr.parentNode;
      if (!parent) return;
      rule.map = _readMap(parent);
      commitData();
    }
    ki.addEventListener('input', rebuild);
    vi.addEventListener('input', rebuild);
    const del = document.createElement('button');
    del.type = 'button'; del.textContent = '×';
    del.addEventListener('click', () => {
      const parent = tr.parentNode;
      tr.remove();
      if (parent) {
        rule.map = _readMap(parent);
        commitData();
      }
    });
    td1.appendChild(ki); td2.appendChild(vi); td3.appendChild(del);
    tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3);
    return tr;
  }

  // Footer + bottom controls. In checklist mode we ONLY show the footer
  // count; full mode adds add-rule + reset-order + reset-all buttons.
  // Footer reflects EFFECTIVE state: globally-disabled rules don't count
  // unless the model force-includes them; globally-enabled rules count
  // unless the model force-excludes them.
  let footer = null;
  function _footerText() {
    const ex = (opts.excludeSet || new Set());
    const inc = (opts.includeSet || new Set());
    let total = 0;
    let active = 0;
    for (const r of rules) {
      if (r.type === 'terminal') continue;
      total++;
      const forcedOut = ex.has(r.name);
      const forcedIn  = inc.has(r.name);
      const effective = (r.enabled && !forcedOut) || forcedIn;
      if (effective) active++;
    }
    return total + ' rule' + (total === 1 ? '' : 's') + ' total · '
      + active + ' active for this model';
  }
  if (isChecklist) {
    footer = document.createElement('div');
    footer.className = 'help rule-checklist-footer';
    footer.textContent = _footerText();
    wrap.appendChild(footer);
    paintAll();
    return wrap;
  }

  // Bottom controls.
  const ctrls = document.createElement('div');
  ctrls.className = 'rule-list-controls';
  ctrls.style.marginTop = '8px';
  ctrls.style.display = 'flex';
  ctrls.style.gap = '8px';

  const addBtn = document.createElement('button');
  addBtn.type = 'button';
  addBtn.textContent = '+ Add custom rule';
  addBtn.addEventListener('click', () => _openAddCustomDialog());
  ctrls.appendChild(addBtn);

  const resetOrderBtn = document.createElement('button');
  resetOrderBtn.type = 'button';
  resetOrderBtn.className = 'reset-link';
  resetOrderBtn.textContent = '↺ Reset order';
  resetOrderBtn.title = 'Restore canonical seeded order; custom rules append before terminal';
  resetOrderBtn.addEventListener('click', () => {
    const baseline = fieldDef(name).default_value || [];
    const baseOrder = baseline.map(b => b.name);
    const seeded = [];
    const customs = [];
    let terminal = null;
    rules.forEach(r => {
      if (r.type === 'terminal') terminal = r;
      else if (r.seeded) seeded.push(r);
      else customs.push(r);
    });
    seeded.sort((a, b) => baseOrder.indexOf(a.name) - baseOrder.indexOf(b.name));
    rules = [...seeded, ...customs];
    if (terminal) rules.push(terminal);
    commitFull();
  });
  ctrls.appendChild(resetOrderBtn);

  const resetAllBtn = document.createElement('button');
  resetAllBtn.type = 'button';
  resetAllBtn.className = 'reset-link';
  resetAllBtn.textContent = '↺ Reset all to defaults';
  resetAllBtn.title = 'Restore the 13 seeded rules to their in-repo defaults; custom rules untouched';
  resetAllBtn.addEventListener('click', () => {
    const baseline = fieldDef(name).default_value || [];
    const customs = rules.filter(r => !r.seeded && r.type !== 'terminal');
    const ok = confirm(
      'Reset 13 seeded rules to their in-repo defaults.\n' +
      (customs.length ? `Your ${customs.length} custom rule(s) will be kept at their current positions.\n\n` : '\n') +
      'Continue?'
    );
    if (!ok) return;
    // Replace seeded rules with baseline copy (order preserved by name).
    const baseByName = new Map(baseline.map(b => [b.name, b]));
    rules = rules.map(r => r.seeded || r.type === 'terminal'
      ? JSON.parse(JSON.stringify(baseByName.get(r.name) || r))
      : r);
    commitFull();
  });
  ctrls.appendChild(resetAllBtn);

  wrap.appendChild(ctrls);

  function _openAddCustomDialog() {
    // Lightweight inline form, appended at the bottom of the rules list.
    const form = document.createElement('div');
    form.className = 'rule-row';
    form.style.borderColor = '#7ee787';
    const head = document.createElement('div');
    head.className = 'row-header';
    head.innerHTML = '<strong>+ New custom rule</strong>';
    form.appendChild(head);
    const body = document.createElement('div');
    body.className = 'row-body';
    body.style.display = 'block';

    const typeSel = document.createElement('select');
    _PIPELINE_TYPES.filter(t => t.type !== 'terminal').forEach(t => {
      const o = document.createElement('option');
      o.value = t.type; o.textContent = t.type + '  (' + t.pill + ')';
      typeSel.appendChild(o);
    });
    const labelInp = document.createElement('input');
    labelInp.type = 'text';
    labelInp.placeholder = 'Friendly label (e.g. "Expand Uhr to :00")';
    labelInp.style.width = '100%'; labelInp.style.marginTop = '4px';
    const patInp = document.createElement('input');
    patInp.type = 'text'; patInp.placeholder = 'pattern';
    patInp.style.width = '100%'; patInp.style.marginTop = '4px';
    patInp.style.fontFamily = 'ui-monospace, monospace';
    const replInp = document.createElement('input');
    replInp.type = 'text'; replInp.placeholder = 'replacement (regex only)';
    replInp.style.width = '100%'; replInp.style.marginTop = '4px';
    replInp.style.fontFamily = 'ui-monospace, monospace';

    const ok = document.createElement('button');
    ok.type = 'button'; ok.textContent = 'Add'; ok.style.marginTop = '6px';
    const cancel = document.createElement('button');
    cancel.type = 'button'; cancel.textContent = 'Cancel'; cancel.style.marginLeft = '6px';

    body.appendChild(_labeledRow('Type', typeSel));
    body.appendChild(_labeledRow('Label', labelInp));
    body.appendChild(_labeledRow('Pattern', patInp));
    body.appendChild(_labeledRow('Replacement', replInp));
    body.appendChild(ok);
    body.appendChild(cancel);
    form.appendChild(body);
    list.appendChild(form);

    cancel.addEventListener('click', () => form.remove());
    ok.addEventListener('click', () => {
      const lbl = labelInp.value.trim() || 'Custom rule';
      const slugSet = new Set(rules.map(r => r.name));
      const slug = _ensureUniqueSlug(_slugify(lbl), slugSet);
      const t = typeSel.value;
      const newRule = {
        name: slug, label: lbl, type: t,
        enabled: true, locked: false, seeded: false,
      };
      if (t === 'regex') {
        newRule.pattern = patInp.value;
        newRule.replacement = _unesc(replInp.value);
      } else if (t === 'callback:map') {
        newRule.map = {};
      } else {
        newRule.pattern = patInp.value;
        if (t === 'callback:lowercase-wordlist') newRule.wordlist = [];
      }
      // Insert just before the terminal row, or at the end if no terminal.
      const tIdx = rules.findIndex(r => r.type === 'terminal');
      if (tIdx >= 0) rules.splice(tIdx, 0, newRule);
      else rules.push(newRule);
      form.remove();
      // Auto-expand the newly added rule so the user lands directly in
      // its editor body and can fill in pattern/map/etc.
      expandedNames.add(slug);
      commitFull();
    });
  }

  function _labeledRow(label, el) {
    const wr = document.createElement('div');
    wr.style.marginTop = '4px';
    const l = document.createElement('div');
    l.className = 'help'; l.textContent = label + ':';
    wr.appendChild(l); wr.appendChild(el);
    return wr;
  }

  paintAll();
  return wrap;
}


function pipelineTestPanel() {
  // Full-pipeline test panel — preset dropdown + editable textarea + run button.
  // Inserted under the Pipeline section heading. Output renders as an ordered
  // table mirroring the trace block: ordinal | label | type-pill | output.
  const wrap = document.createElement('div');
  wrap.className = 'regex-test-panel';

  const presetWrap = document.createElement('div');
  presetWrap.className = 'preset-select';
  presetWrap.innerHTML = '<span class="help" style="margin-right:6px">preset:</span>';
  const sel = document.createElement('select');
  for (const k of Object.keys(TEST_PRESETS)) {
    const o = document.createElement('option');
    o.value = k; o.textContent = k;
    sel.appendChild(o);
  }
  presetWrap.appendChild(sel);
  wrap.appendChild(presetWrap);

  const sampleLbl = document.createElement('div');
  sampleLbl.className = 'help';
  sampleLbl.textContent = 'Test sample (edit to try your own):';
  wrap.appendChild(sampleLbl);

  const sample = document.createElement('textarea');
  sample.id = 'pipeline-test-sample';
  sample.value = TEST_PRESETS['default'];
  sample.rows = 2;
  sample.style.width = '100%';
  sample.style.boxSizing = 'border-box';
  sample.style.fontFamily = 'ui-monospace, Menlo, Consolas, monospace';
  sample.style.fontSize = '12px';
  wrap.appendChild(sample);

  sel.addEventListener('change', () => {
    sample.value = TEST_PRESETS[sel.value] || '';
  });

  const runBtn = document.createElement('button');
  runBtn.type = 'button';
  runBtn.textContent = 'Run all enabled rules';
  runBtn.style.marginTop = '6px';
  wrap.appendChild(runBtn);

  const out = document.createElement('div');
  out.className = 'regex-test-out';
  wrap.appendChild(out);

  async function run() {
    out.innerHTML = '<em>running…</em>';
    const r = await api('POST', '/config/test-pipeline', {
      sample: sample.value,
      rules: currentValue('PIPELINE_RULES') || [],
    });
    if (!r.ok) { out.innerHTML = '<em class="err">test endpoint error</em>'; return; }
    const j = await r.json();
    const tbl = document.createElement('table');
    tbl.className = 'pipeline-test-table';
    const thead = document.createElement('tr');
    thead.innerHTML = '<th>#</th><th>label</th><th>type</th><th>output</th>';
    tbl.appendChild(thead);
    (j.steps || []).forEach(step => {
      const tr = document.createElement('tr');
      let badge = '';
      if (step.skipped) badge = ' <span class="tag empty">skipped</span>';
      else if (step.error) badge = ' <span class="tag err">✗</span>';
      else if (step.slow) badge = ' <span class="tag warn">⚠ slow</span>';
      else if (step.matches) badge = ' <span class="tag ok">' + step.matches + ' matches</span>';
      const changed = step.before !== step.after;
      const outCell = document.createElement('td');
      outCell.className = 'out';
      if (step.error) {
        outCell.innerHTML = '<span class="err">' + step.error + '</span>';
      } else if (!changed) {
        outCell.innerHTML = '<span class="nochange">(no change)</span>';
      } else {
        outCell.textContent = step.after;
      }
      tr.innerHTML = '<td>' + step.ordinal + '</td>'
        + '<td>' + (step.label || '?') + badge + '</td>'
        + '<td><span class="type-pill">' + _typePill(step.type) + '</span></td>';
      tr.appendChild(outCell);
      tbl.appendChild(tr);
    });
    const finalRow = document.createElement('tr');
    const finalCell = document.createElement('td');
    finalCell.colSpan = 3;
    finalCell.innerHTML = '<strong>Final →</strong>';
    finalRow.appendChild(finalCell);
    const finalOut = document.createElement('td');
    finalOut.className = 'out';
    finalOut.textContent = j.final;
    finalRow.appendChild(finalOut);
    tbl.appendChild(finalRow);
    out.innerHTML = '';
    out.appendChild(tbl);
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
  t.placeholder = 'One entry per line';
  const help = document.createElement('div');
  help.className = 'help';
  help.textContent = 'One entry per line. Blank lines ignored.';
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
    // emits no subheader (back-compat with old single-list layout). A subgroup
    // WITH a title is wrapped in <details>/<summary> so the admin can collapse
    // long-tail "Advanced —" sections; open/closed state persists in
    // localStorage keyed by `adv.global.{group}.{sub}`.
    for (const sub of (g.subgroups || [{ title: null, fields: g.fields || [] }])) {
      const fieldsWrap = document.createElement('div');
      fieldsWrap.className = 'group-fields';
      for (const fname of sub.fields) {
        try {
          fieldsWrap.appendChild(fieldRow(fname));
        } catch (err) {
          console.error('failed to render field', fname, err);
          const errRow = document.createElement('div');
          errRow.className = 'field';
          errRow.innerHTML = '<div class="label-col"><div class="name">' + fname
            + '</div></div><div class="input-col"><div class="err">'
            + 'render failed: ' + (err.message || err) + '</div></div>';
          fieldsWrap.appendChild(errRow);
        }
      }
      if (sub.title) {
        const det = document.createElement('details');
        det.className = 'subgroup-details';
        const sum = document.createElement('summary');
        sum.className = 'subgroup-summary';
        sum.textContent = sub.title;
        det.appendChild(sum);
        det.appendChild(fieldsWrap);
        const lsKey = 'adv.global.' + g.title + '.' + sub.title;
        det.open = localStorage.getItem(lsKey) === '1';
        det.addEventListener('toggle', () => {
          try { localStorage.setItem(lsKey, det.open ? '1' : '0'); } catch (_) {}
        });
        sec.appendChild(det);
      } else {
        sec.appendChild(fieldsWrap);
      }
    }
    // The Pipeline section gets the full-pipeline test panel appended at the
    // bottom (after PIPELINE_RULES renders). Single panel — runs the whole
    // ordered list against the editable sample.
    if (g.title === 'Pipeline') {
      sec.appendChild(pipelineTestPanel());
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
  $('discard-btn').disabled = true;

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
  $('discard-btn').addEventListener('click', () => {
    if (Object.keys(dirty).length === 0) return;
    dirty = {};
    $('save-btn').disabled = true;
    $('discard-btn').disabled = true;
    render();    // re-render every field from state.fields (server-side values)
  });
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
    document.body.innerHTML = '<main style="padding:1.25rem;color:#ff7b72">'
      + 'Could not load /config/state (' + probe.status + '). '
      + 'Check service logs.</main>';
    return;
  }
  showApp();
  await loadState();
});

})();
</script>
{{SCALE_PICKER_JS}}
</body></html>"""
