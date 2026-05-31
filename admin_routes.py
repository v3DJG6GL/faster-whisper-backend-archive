"""
Admin WebUI for faster-whisper-backend.

Mounted at /settings when WHISPER_ADMIN_UI=1. Endpoints:

  GET  /settings               HTML page (loopback / ADMIN_ALLOWED_HOSTS)
  GET  /settings/state         Resolved config + provenance + hot/cold tags
  POST /settings/state         Save overrides (validation errors -> 422)
  POST /settings/test-pipeline Dry-run PIPELINE_RULES against a sample
  POST /settings/restart       Detach a self-restart helper (Windows only)

Security model (layered):
  1. Allowlist gate:   require_admin_host rejects callers not in
                       cfg.ADMIN_ALLOWED_HOSTS (loopback always permitted)
  2. API key:          Depends(require_admin) — bearer must resolve to a
                       user with is_admin=True. In OPEN mode (no admin
                       key exists yet) the dep yields a synthetic admin
                       so the operator can bootstrap.
  3. Pydantic schema:  AdminConfig validates body shape, types, bounds
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import TypeAdapter, ValidationError

import config as cfg
import config_store
import web_common
from auth import require_admin

logger = logging.getLogger("whisper-api")

# Discriminated-union adapter for PIPELINE_RULES canonicalization. Built once
# at import time — TypeAdapter construction walks every rule subclass and is
# the dominant cost of _canon_rules, called twice per /settings/state request.
_PIPELINE_RULE_ADAPTER: TypeAdapter = TypeAdapter(config_store.PipelineRule)

# Fields the WebUI is allowed to surface. Keep this as the single source of
# truth for the form layout — drives section grouping in the HTML and the
# /settings/state endpoint's provenance map.
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
            "AUTO_CONVERT_HF_MODELS", "CONVERT_QUANTIZATION", "CONVERTED_MODELS_DIR",
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
            "SUPPRESS_CHARS",
            "PREPEND_PUNCTUATIONS", "APPEND_PUNCTUATIONS",
        ]),
    ]),
    ("Output wrappers", [(None, [
        "OUTPUT_PREFIX", "OUTPUT_SUFFIX",
    ])]),
    ("Per-model overrides", [(None, ["MODEL_OVERRIDES"])]),
    ("Pipeline", [(None, ["PIPELINE_RULES"])]),
    ("Logging", [(None, [
        "LOG_FILE", "LOG_MAX_BYTES", "LOG_BACKUP_COUNT",
        "LOG_VIEWER_INITIAL_LINES", "LOG_VIEWER_DOM_MAX",
        "TRACE_ENABLED",
    ])]),
    ("Server (uvicorn)", [(None, [
        "SERVER_HOST", "SERVER_PORT", "SERVER_WORKERS", "SERVER_LOG_LEVEL",
    ])]),
    ("Access (allowlists)", [
        (None, [
            "ADMIN_ALLOWED_HOSTS", "STATS_ALLOWED_HOSTS",
        ]),
        ("Browser sessions (cookie auth)", [
            "SESSION_COOKIE_SECURE", "SESSION_TTL_SECONDS",
            "SESSION_COOKIE_NAME", "SESSION_CSRF_COOKIE_NAME",
        ]),
    ]),
    ("Reports", [(None, [
        "REPORTS_DB", "REPORTS_MAX", "REPORTS_RETENTION_DAYS",
        "REPORTS_ALLOW_USER_SUBMIT",
    ])]),
    ("Recent transcriptions", [(None, [
        "RECENT_TRANSCRIPTIONS_DB",
        "RECENT_TRANSCRIPTIONS_MAX",
        "RECENT_TRANSCRIPTIONS_TTL_DAYS",
        "RECENT_TRANSCRIPTIONS_PAGE_SIZE",
        "RECENT_TRANSCRIPTIONS_PRUNE_EVERY",
        "STATS_RECENT_TX_DISPLAY",
    ])]),
    ("Capture & fine-tuning", [
        (None, [
            "CAPTURE_RECORDINGS_ENABLED",
            "CAPTURE_RECORDINGS_SAMPLE_RATE",
            "CAPTURES_RETENTION_DAYS",
        ]),
        ("Storage", [
            "CAPTURES_DB", "CAPTURES_DIR",
            "CAPTURES_MAX", "CAPTURES_MAX_MB",
        ]),
        ("Duration & size guards", [
            "CAPTURE_RECORDINGS_MIN_DURATION_SEC",
            "CAPTURE_RECORDINGS_MAX_DURATION_SEC",
            "CAPTURE_RECORDINGS_AUDIO_BYTES_HARD_LIMIT",
        ]),
        ("Sample sizing", [
            "CAPTURES_SAMPLE_MIN_DURATION_S",
            "CAPTURES_SAMPLE_MAX_DURATION_S",
            "CAPTURES_SAMPLE_JOIN_STRATEGY",
            "CAPTURES_PROPOSER_TARGET_S",
            "CAPTURES_PROPOSER_SESSION_GAP_S",
            "CAPTURES_PROPOSER_DUP_THRESHOLD",
            "CAPTURES_PROPOSER_MAX_PROPOSALS",
        ]),
        ("Training-form pipeline", [
            "CAPTURES_PIPELINE_RULES_EXCLUDE",
        ]),
        ("Silence trim (Silero VAD)", [
            "CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS",
            "CAPTURES_VAD_MARGIN_GROUP_EDGE_MS",
            "CAPTURES_VAD_MARGIN_GROUP_INTERNAL_MS",
        ]),
    ]),
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
# /settings is gated by an IP/CIDR allowlist (cfg.ADMIN_ALLOWED_HOSTS, loopback
# always implicit) AND by `Depends(require_admin)` — an API key resolving to
# is_admin=True. In open mode (no admin key configured yet) require_admin
# yields the synthetic admin so the operator can bootstrap.
require_admin_host = web_common.require_allowed_host(lambda: cfg.ADMIN_ALLOWED_HOSTS)


# --- router ------------------------------------------------------------------

router = APIRouter(prefix="/settings")


def _resolved_value(field: str) -> Any:
    """Read the current effective value of a config field by attribute name."""
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
# order while the resolved value (after a local.json overlay) carries
# Pydantic's parent-first MRO order, and the WebUI's dirty / origin-badge
# comparisons would report a spurious diff on first paint.
def _canon_rules(rules: Any) -> Any:
    if not isinstance(rules, list):
        return rules
    out: list[Any] = []
    for r in rules:
        try:
            dumped = _PIPELINE_RULE_ADAPTER.validate_python(r).model_dump(exclude_none=True)
            out.append(_sort_dicts(dumped))
        except Exception:
            out.append(r)  # malformed — pass through; save-time validator catches it
    return out


# `model_dump()` preserves insertion order on nested dict fields (e.g. the
# `map` on a callback:map rule). The resolved value (after a local.json
# overlay) and the baseline `default_value` (from cfg._BASELINE) can carry
# different insertion orders even when contents are equal — which makes
# JSON.stringify(value) !== JSON.stringify(default_value) so the WebUI's
# dirty / origin-badge checks falsely report a diff, AND clicking reset
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
async def settings_page() -> HTMLResponse:
    """The admin HTML page. Allowlist-gated (loopback always allowed) — no
    token required to LOAD the page; the page itself collects the token and
    attaches it on every fetch. `no-store` so browsers never serve a stale
    build after a service restart."""
    return HTMLResponse(
        web_common.render_page(_SETTINGS_VIEWER_HTML, current="settings"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/state", dependencies=[Depends(require_admin_host), Depends(require_admin)])
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
        "service_name": "WhisperAPI",
    }


@router.post("/state", dependencies=[Depends(require_admin_host), Depends(require_admin)])
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

    applied = await _apply_hot_changes(written)

    client_host = request.client.host if request.client else "?"
    logger.info(
        "[config] admin update from=%s saved=%d hot=%s cold=%s pinned=%s evicted=%s",
        client_host, len(written), applied["hot_applied"], applied["cold_pending"],
        applied["env_pinned_ignored"], applied["evicted"],
    )

    return JSONResponse({
        "saved": sorted(written.keys()),
        **applied,
        "requires_restart": bool(applied["cold_pending"]),
    })


async def _apply_hot_changes(written: dict[str, Any]) -> dict[str, Any]:
    """Apply hot edits from a config save to the running cfg module, rebuild
    caches, and evict load-time-affected models.

    Shared by /settings/state (admin) and /quick-config/state (end-user). For
    /quick-config only PIPELINE_RULES can change, so most branches here
    simply skip — but the helper handles all cases uniformly so the two
    paths stay in lockstep.

    Returns a dict suitable to splat into the JSON response envelope:
      hot_applied, cold_pending, env_pinned_ignored, evicted.
    """
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

    return {
        "hot_applied": hot_changed,
        "cold_pending": cold_changed,
        "env_pinned_ignored": sorted(n for n in written if n in env_pinned),
        "evicted": evicted,
    }


@router.get("/factory-rules",
            dependencies=[Depends(require_admin_host), Depends(require_admin)])
async def get_factory_rules() -> dict[str, Any]:
    """Return the committed factory pipeline rules (config.json).

    The WebUI fetches this just before a "promote" so the diff dialog compares
    against the truly-current config.json. Distinct from GET /settings/state,
    which returns the EFFECTIVE rules (config.json overlaid by config.local.json).
    """
    try:
        rules = config_store.load_factory_rules()
    except RuntimeError as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))
    return {"PIPELINE_RULES": _canon_rules(rules)}


@router.post("/factory-rules",
             dependencies=[Depends(require_admin_host), Depends(require_admin)])
async def post_factory_rules(payload: dict[str, Any], request: Request) -> JSONResponse:
    """Validate and persist the factory pipeline rules to the committed
    config.json. Whole-list replace — the WebUI's promote actions send the full
    intended config.json array (one rule spliced in, or the whole effective list
    for "promote all").

    config.json is git-tracked, so the save surfaces as a working-tree change
    the admin then commits + pushes to ship the fix to every deployment.

    After the write: refresh cfg._BASELINE (the "reset to default" baseline),
    recompute the effective cfg.PIPELINE_RULES (config.local.json still wins if
    it carries its own PIPELINE_RULES — unchanged local-override behaviour),
    and rebuild the pipeline cache. The response carries the canonicalized saved
    rules so the editor can refresh its in-memory `factoryRules` snapshot.
    """
    rules = payload.get("PIPELINE_RULES")
    if not isinstance(rules, list):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "payload must contain a 'PIPELINE_RULES' array",
        )
    try:
        saved = config_store.save_factory_rules(rules)
    except ValidationError as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"errors": config_store.format_validation_errors(e)},
        )
    except OSError as e:
        logger.error("[config] factory-rules save failed: %s", e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"could not write config.json: {e}")

    # config.json IS the factory baseline — refresh the in-memory snapshot the
    # WebUI's "↺ reset to default" and /settings/state `default_value` rely on.
    if isinstance(getattr(cfg, "_BASELINE", None), dict):
        cfg._BASELINE["PIPELINE_RULES"] = [dict(r) for r in saved]

    # Recompute the EFFECTIVE rule list. config.local.json's PIPELINE_RULES
    # still wins if present (per-deployment local override — unchanged); only
    # when there is no local override does the factory list run directly.
    overrides = config_store.load_overrides()
    local_rules = overrides.get("PIPELINE_RULES")
    shadowed = isinstance(local_rules, list)
    cfg.PIPELINE_RULES = local_rules if shadowed else [dict(r) for r in saved]

    try:
        import main as _main
        _main.rebuild_caches()
        logger.info("[config] rebuilt pipeline caches after factory-rules save")
    except Exception as e:
        logger.error("[config] cache rebuild failed after factory save: %s", e)

    client_host = request.client.host if request.client else "?"
    logger.info("[config] factory-rules update from=%s rules=%d shadowed_by_local=%s",
                client_host, len(saved), shadowed)

    return JSONResponse({
        "saved": len(saved),
        "shadowed_by_local": shadowed,
        "rules": _canon_rules(saved),
    })


@router.post("/factory-rules/clear-local-override",
             dependencies=[Depends(require_admin_host), Depends(require_admin)])
async def clear_local_pipeline_override(request: Request) -> JSONResponse:
    """Remove the PIPELINE_RULES override from config.local.json so the committed
    config.json becomes the live pipeline on this deployment.

    Offered after a "promote all": once the factory file holds everything, the
    local snapshot is redundant and only shadows config.json. Clearing it makes
    config.json the runtime source here too.

    Done as a dedicated route (not POST /settings/state with a None sentinel)
    because _apply_hot_changes, on override removal, falls back to the stale
    in-memory value rather than the baseline — so it would not take effect
    until restart. Here we explicitly reload config.json into cfg.
    """
    try:
        config_store.save_overrides({"PIPELINE_RULES": None})
        factory = config_store.load_factory_rules()
    except (ValidationError, RuntimeError, OSError) as e:
        logger.error("[config] clear-local-override failed: %s", e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"could not clear local override: {e}")

    cfg.PIPELINE_RULES = factory
    try:
        import main as _main
        _main.rebuild_caches()
        logger.info("[config] rebuilt pipeline caches after clearing local override")
    except Exception as e:
        logger.error("[config] cache rebuild failed after clear-local-override: %s", e)

    client_host = request.client.host if request.client else "?"
    logger.info("[config] local PIPELINE_RULES override cleared from=%s — "
                "config.json (%d rules) is now live", client_host, len(factory))
    return JSONResponse({"ok": True, "rules": len(factory)})


@router.post("/test-pipeline",
             dependencies=[Depends(require_admin_host), Depends(require_admin)])
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


@router.post("/restart", dependencies=[Depends(require_admin_host), Depends(require_admin)])
async def post_restart(request: Request) -> JSONResponse:
    """Trigger a self-restart of the backend (cross-platform).

    Windows: spawns `WhisperAPI.exe restart!` (WinSW's documented self-restart
    command) and schedules `os._exit(0)` ~1.5 s out. Other OSes: re-execs the
    process in place (os.execv) ~1.5 s out — works bare, under systemd, or in
    a container. Either way this returns 200 first; the 1.5 s delay lets the
    response flush over loopback before the process restarts. End-to-end
    downtime is ~3-4 s for a no-preload deployment.

    See restart_service.py for the per-platform mechanics (and why Windows
    uses WinSW's explicit `restart!` rather than <onfailure>).
    """
    try:
        from restart_service import trigger_self_restart
    except ImportError as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"restart_service module unavailable: {e}")

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
# Vanilla JS, no build step. Mirrors the /logs viewer styling. Sections,
# per-rule PIPELINE_RULES editor, textarea-per-line editors for list/set
# fields, save flow with restart modal + post-restart polling.

_SETTINGS_VIEWER_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
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
  /* header / .header-inner / .title / page-toolbar controls (buttons,
     pills, the red-tinted #discard-btn) are all centralized in NAV_CSS. */
  main { padding: 0.875rem; max-width: 68.75rem; margin: 0 auto;
    container-type: inline-size; container-name: form; }
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
  /* Narrow form: stack label-col ABOVE input-col instead of beside it.
     Wide layout wastes vertical space below short labels; narrow layout
     wastes HORIZONTAL space (the empty area under "PIPELINE_RULES" while
     the rule editor gets squeezed). Container query measures `main`'s
     rendered width in rem so it tracks the --fs-base scale. */
  @container form (max-width: 46rem) {
    .group-fields { grid-template-columns: 1fr; row-gap: 0.25rem; }
    .field { row-gap: 0.25rem; padding: 0.5rem 0; }
    .label-col { flex-direction: row; flex-wrap: wrap; align-items: baseline;
      gap: 0.5rem; }
  }
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
    width: 1rem; height: 1rem;
    border: 1px solid var(--border); border-radius: 3px;
    background: var(--input-bg); cursor: pointer;
    position: relative; vertical-align: middle;
    transition: background 120ms ease, border-color 120ms ease;
  }
  input[type="checkbox"]:hover { border-color: var(--cyan); }
  input[type="checkbox"]:checked { background: #1f6feb; border-color: #388bfd; }
  input[type="checkbox"]:checked::after {
    content: ""; position: absolute; left: 0.25rem; top: 0;
    width: 0.25rem; height: 0.5625rem;
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
     .expand-btn, header button, .modal button, etc.). */
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
  /* Collapsible subgroup heading: smaller than h2, lighter weight, with
     a small dividing line so it's visibly distinct from the section
     header. Native disclosure-triangle is suppressed — list-style: none
     on summary + ::-webkit-details-marker for older WebKit. State
     persists in localStorage (see render() in JS). */
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
  .regex-status { font-size: var(--fs-xs); font-family: var(--font-mono); }
  .regex-status.ok { color: var(--green); }
  .regex-status.err { color: var(--red); }
  .regex-status.warn { color: var(--yellow); }
  .regex-status.empty { color: var(--dim); font-style: italic; }
  /* Advanced warning banner above the Step 3 fields */
  .advanced-warn { background: #2d1f0a; color: var(--yellow); border-left: 3px solid var(--yellow);
    padding: 0.375rem 0.625rem; margin: 0.5rem 0; border-radius: 3px; font-size: var(--fs-sm); }
  /* Pipeline rule origin badges (factory / edited / local-only vs config.json) */
  .rule-origin-badge { font-size: var(--fs-xs); font-family: var(--font-mono);
    white-space: nowrap; }
  .rule-origin-badge.factory { color: var(--dim); }
  .rule-origin-badge.edited { color: var(--yellow); }
  .rule-origin-badge.local-only { color: var(--green); }
  /* Per-row + list-wide "promote to config.json" affordances */
  .promote-btn { background: none; border: 1px solid var(--cyan); color: var(--cyan);
    border-radius: 3px; padding: 0 0.375rem; cursor: pointer; font: inherit;
    font-size: var(--fs-xs); }
  .promote-btn:hover { background: #161b22; }
  .promote-all-btn { background: none; border: none; padding: 0; cursor: pointer;
    font: inherit; font-size: var(--fs-xs); color: var(--cyan);
    text-decoration: underline; text-underline-offset: 2px; }
  /* Promote diff / confirm modal */
  .rule-modal-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.6);
    display: flex; align-items: center; justify-content: center; z-index: 1000; }
  .rule-modal { background: #161b22; border: 1px solid var(--border);
    border-radius: 6px; padding: 1rem 1.25rem; max-width: 46rem; width: 90%;
    max-height: 80vh; overflow: auto; }
  .rule-modal-title { font-weight: 600; margin-bottom: 0.625rem; }
  .rule-modal-buttons { display: flex; gap: 0.5rem; justify-content: flex-end;
    margin-top: 0.875rem; flex-wrap: wrap; }
  .rule-modal-buttons button.primary { color: var(--cyan); border-color: var(--cyan); }
  .promote-diff { margin-top: 0.5rem; }
  .promote-diff-row { display: grid; grid-template-columns: 7rem 1fr 1fr;
    gap: 0.5rem; font-family: var(--font-mono); font-size: var(--fs-xs);
    padding: 0.25rem 0; border-bottom: 1px solid var(--border); }
  .promote-diff-key { color: var(--dim); }
  .promote-diff-old { color: var(--red); white-space: pre-wrap; word-break: break-all; }
  .promote-diff-new { color: var(--green); white-space: pre-wrap; word-break: break-all; }
  .promote-change-line { font-size: var(--fs-sm); margin-top: 0.25rem; }
  .promote-unsaved-note { color: var(--yellow); font-size: var(--fs-xs);
    margin-top: 0.5rem; }
  .rule-toast { position: fixed; bottom: 1.25rem; left: 50%;
    transform: translateX(-50%); background: #161b22; color: var(--fg);
    border: 1px solid var(--cyan); border-radius: 4px;
    padding: 0.5rem 0.875rem; font-size: var(--fs-sm); z-index: 1001; }
  /* Test panel */
  .regex-test-panel { background: #161b22; border: 1px solid var(--border);
    border-radius: 4px; padding: 0.625rem 0.75rem; margin: 0.5rem 0 0.875rem 0; }
  .regex-test-panel textarea { resize: vertical; max-width: 100%; }
  .regex-test-out { margin-top: 0.625rem; }
  /* Pipeline rules editor */
  .pipeline-rules-wrap { display: flex; flex-direction: column; gap: 0.375rem; }
  .rule-list { display: flex; flex-direction: column; gap: 0.25rem; }
  /* Button-like interactive feel on every rule row. :active propagates
     up from descendants so pressing the drag-handle still tints the row. */
  .rule-row { background: #161b22; border: 1px solid var(--border); border-radius: 4px;
    padding: 0.375rem 0.625rem; transition: background-color 120ms ease;
    cursor: default; }
  .rule-row.locked { border-left: 3px solid var(--yellow); }
  .rule-row.terminal { border-left: 3px solid var(--dim); opacity: 0.85; }
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
  /* Two-line head: identity (drag, #, enabled, label, slug, type, edit)
     on line 1 — line 2 carries metadata (tag picker, user-editable
     toggle, untagged badge). Splits the cognitive load so the load-
     bearing field (rule-label) actually has room to breathe. */
  .rule-row .row-header { display: flex; flex-direction: column;
    gap: 0.35rem; padding: 0.45rem 0.6rem; }
  .rule-row .row-header-line1,
  .rule-row .row-header-line2 {
    display: flex; align-items: center; gap: 0.5rem;
    flex-wrap: wrap; row-gap: 0.25rem;
    font-family: var(--font-mono); font-size: var(--fs-sm);
  }
  /* Line 2 is metadata — slightly smaller + dim, indented under the
     drag handle so the visual hierarchy reads "this is detail about
     the row above". */
  .rule-row .row-header-line2 {
    padding-left: 1.75rem;
    font-size: var(--fs-xs);
    color: var(--dim);
  }
  /* Hide line 2 for terminal rules (no tags / no exposed flag apply). */
  .rule-row.terminal .row-header-line2 { display: none; }
  /* Line 2's "tags:" prefix sits before the picker. Fixed-width so
     pills always start at the same x across rows. */
  .rule-row .row-header-line2 .meta-label {
    color: var(--dim); user-select: none; min-width: 2.5rem;
  }
  /* The rule label is the load-bearing field — it gets every spare
     pixel of line 1 and ellipses on the right when truly narrow. The
     full text lives in title= so hover always reveals it. */
  .rule-row .rule-label {
    flex: 1; min-width: 0; color: var(--fg);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  /* Phone: the load-bearing name is the only flex item that shrinks, so on a
     crowded row it collapsed to "R…". Drop it onto its own full-width line and
     let it wrap in full instead of truncating (flex-basis:100% breaks the flex
     line; the slug/pills/edit wrap below it). Desktop keeps the single line. */
  @media (max-width: 40em) {
    .rule-row .rule-label {
      flex-basis: 100%;
      white-space: normal; overflow: visible; text-overflow: clip;
    }
  }
  .rule-row .drag-handle { cursor: grab; user-select: none; color: var(--dim);
    padding: 0.125rem 0.25rem; }
  .rule-row .drag-handle:active { cursor: grabbing; }
  .rule-row.locked .drag-handle { cursor: not-allowed; }
  .rule-row .ordinal { color: var(--dim); min-width: 1.5rem; text-align: right; }
  /* Generalised labelled-toggle widget used for BOTH `enabled` (line 1)
     and `user-editable` (line 2). Same shape, same green-when-on cue,
     same dim-when-off. Renamed from `.expose-toggle` — that name lied
     about the toggle below it (the user thought "simple" meant a
     /simple page somewhere). */
  .rule-row .toggle { display: inline-flex; gap: 0.3rem;
    align-items: center; user-select: none; cursor: pointer;
    padding: 0 0.25rem; border-radius: 3px;
    color: var(--dim); }
  .rule-row .toggle.on { color: var(--green); }
  .rule-row .toggle input { margin: 0; }
  .rule-row .rule-slug { color: var(--dim); font-size: var(--fs-xs); font-style: italic; }
  /* Type pill — fixed min-width so the label column starts at a
     predictable x across rows (the previous shrink-to-fit made the
     label jump left-right between regex / cb:wordlist / cb:dedup). */
  .rule-row .type-pill { display: inline-block; padding: 0 0.375rem; border-radius: 3px;
    font-size: var(--fs-xs); background: #21262d; color: var(--cyan);
    min-width: 4rem; text-align: center; }
  .rule-row .expand-btn, .rule-row .delete-btn {
    background: transparent; border: 1px solid var(--border);
    color: var(--fg); padding: 0.125rem 0.375rem; border-radius: 3px; cursor: pointer;
    font: inherit; font-size: var(--fs-xs); }
  .rule-row .delete-btn { color: var(--red); }
  .rule-row .row-body { padding-left: 2rem; padding-top: 0.375rem; display: none; }
  .rule-row.expanded .row-body { display: block; }
  .rule-row.terminal .row-body { display: block; padding-top: 0.25rem; }
  /* Destructive-actions footer inside the expanded body. Holds reset /
     delete — keeps them out of the scannable head row (PatternFly
     "destructive actions outside scannable rows" guidance). */
  .rule-row .row-body-actions {
    display: flex; gap: 0.5rem; margin-top: 0.6rem;
    padding-top: 0.5rem; border-top: 1px solid var(--border);
    justify-content: flex-end;
  }
  .rule-row .row-body-actions .reset-link,
  .rule-row .row-body-actions .delete-btn {
    font-size: var(--fs-xs);
  }
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
  .pipeline-test-table .out mark { background: #033a16; color: var(--green); }
  .pipeline-test-table .nochange { color: var(--dim); font-style: italic; }
  .pipeline-test-table .tag { display: inline-block; padding: 0 0.375rem;
    border-radius: 3px; font-size: var(--fs-xs); }
  .pipeline-test-table .tag.ok { background: #033a16; color: var(--green); }
  .pipeline-test-table .tag.err { background: #3a0d0d; color: #ff7b72; }
  .pipeline-test-table .tag.warn { background: #2d1f0a; color: var(--yellow); }
  .pipeline-test-table .tag.empty { background: #21262d; color: var(--dim); font-style: italic; }
  .preset-select { margin-bottom: 0.375rem; }
  .preset-select select { font-family: var(--font-mono); font-size: var(--fs-sm); }
  /* Nullable-number editor in its disabled (null) state — greyed input,
     enable/disable button labelled accordingly. */
  .nullable-wrap input:disabled { opacity: 0.4; cursor: not-allowed; }
  /* Env-pinned fields: the env var wins at runtime, so the GUI editor is
     disabled and greyed. The label/badges stay full-strength so the field is
     still scannable; only the value controls are dimmed. */
  .field.env-pinned .input-col { opacity: 0.55; }
  .field.env-pinned .env-pinned-editor input:disabled,
  .field.env-pinned .env-pinned-editor select:disabled,
  .field.env-pinned .env-pinned-editor textarea:disabled,
  .field.env-pinned .env-pinned-editor button:disabled { cursor: not-allowed; }
  .help.help-env-pinned { color: var(--magenta); opacity: 1; }
  .modal { position: fixed; inset: 0; background: rgba(0,0,0,0.7); display: none;
    align-items: center; justify-content: center; z-index: 100; }
  .modal.show { display: flex; }
  .modal-box { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 1rem 1.25rem; max-width: min(32.5rem, 95vw); }
  .modal-box h3 { margin: 0 0 0.625rem; color: var(--bold); }
  .modal-box ul { margin: 0.375rem 0 0.75rem; padding-left: 1.25rem; }
  .modal-actions { display: flex; gap: 0.5rem; justify-content: flex-end;
    margin-top: 0.875rem; flex-wrap: wrap; }
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
  /* Full-row .field (no label-col / help text). Inside, two columns:
     a sidebar of models on the left and the detail pane on the right.
     Below 56rem the sidebar collapses to a horizontal scroller above
     the pane (KDE Linux narrow-viewport requirement).               */
  .field-fullrow { grid-column: 1 / -1;
    grid-template-columns: none !important; }
  .mo-wrap { display: grid;
    grid-template-columns: minmax(13rem, 18rem) 1fr;
    gap: 0.875rem; align-items: start; width: 100%; }
  .mo-sidebar { display: flex; flex-direction: column; gap: 0.25rem;
    background: #0d1117; border: 1px solid var(--border); border-radius: 4px;
    padding: 0.5rem; min-width: 0; }
  .mo-list-item { padding: 0.4rem 0.6rem; border-radius: 4px; cursor: pointer;
    display: flex; align-items: center; gap: 0.5rem;
    border-left: 3px solid transparent; min-width: 0; }
  .mo-list-item:hover { background: #1c2230; }
  .mo-list-item.active { background: #1c2230; border-left-color: var(--cyan); }
  /* Model name wraps instead of truncating: in the vertical sidebar it spills
     to a second line (readable); in the ≤56rem horizontal-scroller it just
     shows in full (scroll to reach). Was ellipsis-on-nowrap, which clipped
     longer names to "distil-large-v…". */
  .mo-list-item .mo-list-label { flex: 1; min-width: 0;
    white-space: normal; overflow-wrap: anywhere; }
  .mo-list-item .mo-count { color: var(--dim); font-size: var(--fs-xs);
    flex-shrink: 0; }
  /* Collapse indicator on the active sidebar entry. ▾ when the detail
     pane is expanded, ▸ when collapsed. Hidden on inactive entries. */
  .mo-list-item .mo-collapse-ind { color: var(--dim);
    font-size: var(--fs-xs); flex-shrink: 0; opacity: 0; }
  .mo-list-item.active .mo-collapse-ind { opacity: 0.85; }
  /* Top sidebar row: replaces the old "Global (read-only ref)" entry.
     Whole row toggles diff-mode dim. Looks like a list item but is a label
     wrapping a checkbox. */
  .mo-sidebar-toggle { display: flex; align-items: center; gap: 0.5rem;
    padding: 0.4rem 0.6rem; border-radius: 4px; cursor: pointer;
    border-left: 3px solid transparent; user-select: none;
    border-bottom: 1px solid var(--border); padding-bottom: 0.55rem;
    margin-bottom: 0.25rem; }
  .mo-sidebar-toggle:hover { background: #1c2230; }
  .mo-sidebar-toggle input[type="checkbox"] { flex-shrink: 0; }
  .mo-sidebar-toggle .mo-toggle-label { color: var(--bold);
    font-size: var(--fs-sm); flex: 1; min-width: 0; }
  .mo-list-add { margin-top: 0.4rem; padding-top: 0.4rem;
    border-top: 1px solid var(--border); }
  .mo-list-add select { width: 100%; box-sizing: border-box;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.25rem 0.5rem; font: inherit; font-size: var(--fs-sm); }
  .mo-remove-btn { margin-top: 0.5rem; color: var(--red) !important;
    text-align: left; }
  .mo-mainpane { display: flex; flex-direction: column; gap: 0.5rem;
    min-width: 0; }
  /* Collapsed detail pane: form body hidden, sidebar still scrollable.
     State stored in localStorage['mo.detail.<id>'] = '0' for collapsed. */
  .mo-mainpane.collapsed > .mo-detail-body { display: none; }
  .mo-mainpane.collapsed::before {
    content: 'detail collapsed — click the active model in the sidebar to expand';
    display: block; padding: 0.6rem 0.75rem;
    color: var(--dim); font-style: italic; font-size: var(--fs-sm);
    background: #0d1117; border: 1px dashed var(--border); border-radius: 4px; }
  .mo-detail-body { display: flex; flex-direction: column; gap: 0.5rem; }
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
  /* Diff-to-global mode: dim rows whose value matches the resolved global. */
  .mo-mainpane.diff-mode .mo-row[data-matches-global="true"] {
    opacity: 0.45; filter: grayscale(0.6); }
  /* Pipeline rules (scoping only) inherit the same diff-mode treatment as
     .mo-row: a rule whose slug is in neither EXCLUDE nor INCLUDE for this
     model is "functionally inherited" and dims identically. Independent
     selector because .rule-row lives in its own class namespace. */
  .mo-mainpane.diff-mode .pipeline-rules-wrap.checklist-mode
    .rule-row[data-matches-global="true"] {
    opacity: 0.45; filter: grayscale(0.6); }
  /* Per-model pipeline rules checklist: makeRuleListEditor renders each
     rule with .rule-row; checklist mode adds .checklist-mode on the wrap
     and per-row .excluded marker. Compact layout: checkbox | label | pill | view.   */
  .pipeline-rules-wrap.checklist-mode .rule-row {
    padding: 0.25rem 0.5rem; }
  .pipeline-rules-wrap.checklist-mode .rule-row .row-header {
    /* Override the two-line column flow used in the global rule editor —
       per-model checklist rows are intentionally single-line + compact. */
    display: flex; flex-direction: row; align-items: center;
    gap: 0.5rem; padding: 0; }
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
    .mo-sidebar-toggle { border-left: none; border-bottom: none;
      margin-bottom: 0; padding-bottom: 0.4rem;
      border-right: 1px solid var(--border); padding-right: 0.55rem;
      flex: 0 0 auto; }
    .mo-list-item { flex: 0 0 auto; border-left: none;
      border-bottom: 3px solid transparent; }
    .mo-list-item.active { border-left: none;
      border-bottom-color: var(--cyan); }
    .mo-list-add, .mo-remove-btn { flex: 0 0 auto; }
  }
  {{NAV_CSS}}
</style></head>
<body>

<div id="login-wrap" class="login" style="display:none">
  <h1>faster-whisper-backend · admin</h1>
  <p>Paste your <strong>API key</strong> to continue. Keys are issued in
  <code>/settings/api-keys</code> by an admin. You'll stay signed in on this
  browser until you sign out.</p>
  <input id="login-token" type="password" autocomplete="off" placeholder="wk_…">
  <button id="login-btn">Unlock</button>
  <p id="login-err" class="err"></p>
</div>

<div id="app-wrap" style="display:none">
  <header>
    <div class="header-inner">
      <span class="title">{{HEADER_BRAND}}</span>
      <span class="brand-sep" aria-hidden="true"></span>
      {{NAV}}
      <span class="spacer"></span>
      <span class="hdr-right">{{SEV_PILLS}}{{SCALE_PICKER}}{{RELOAD}}{{LOGOUT}}</span>
    </div>
    <div class="subbar">
      <span class="subbar-title">Settings</span>
      <div class="subbar-right">
        <button id="restart-btn" title="restart the backend service">restart</button>
        <button id="discard-btn" title="discard all unsaved changes" disabled>discard</button>
        <button id="save-btn" class="primary" disabled>save</button>
        <span id="status" class="pill">loading…</span>
      </div>
    </div>
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

{{RULE_EDITOR_JS}}

<script>
(() => {
'use strict';

let state = null;          // last server state
let dirty = {};            // field -> new value (only changed)

const $ = (id) => document.getElementById(id);

function toast(msg, isErr) {
  const el = $('toast');
  el.textContent = msg;
  el.className = isErr ? 'show err' : 'show';
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.className = ''; }, 3500);
}

async function api(method, path, body) {
  // Auth rides the HttpOnly session cookie (sent automatically). Mutations
  // also carry the double-submit CSRF token.
  const opts = { method, headers: {} };
  if (method !== 'GET' && method !== 'HEAD') {
    opts.headers['X-CSRF-Token'] = window._csrfToken ? window._csrfToken() : '';
  }
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  if (r.status === 401) {
    window.dispatchEvent(new Event('whisper:auth-changed'));
    showLogin('session expired — sign in again');
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
  const r = await api('GET', '/settings/state');
  if (r.status === 401) return;  // showLogin already called
  if (r.status === 403 && typeof _renderNoAccessLanding === 'function') {
    // Valid bearer but no admin scope — render the shared landing
    // instead of falling through to the generic toast (which would
    // leave the page blank). Refresh whoami so the landing can list
    // pages this user CAN reach. showApp() is mandatory: <main> lives
    // inside #app-wrap which is display:none until showApp flips it;
    // rendering into a hidden subtree leaves the user with a blank
    // page on F5 reload.
    try {
      const w = await fetch('/auth/whoami');
      if (w.ok) window.__whoami = await w.json();
    } catch (_) {}
    showApp();
    _renderNoAccessLanding({ page: 'config' });
    return;
  }
  if (!r.ok) {
    toast('failed to load state: ' + r.status, true);
    return;
  }
  state = await r.json();
  // role-admin is set by OPEN_MODE_BANNER_JS (single source of truth)
  // when whoami.is_admin=true. /settings/state requires admin so callers
  // who reach this point ARE admin, but the central handler already
  // covered the body-class add — keep this comment as a breadcrumb.
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

// Disable every form control inside an env-pinned field's editor so the value
// can't be edited from the GUI (the env var wins at runtime). Belt-and-braces
// with the setDirty() guard below + the greyed .env-pinned row styling.
function disableEnvPinnedEditor(el) {
  el.classList.add('env-pinned-editor');
  el.querySelectorAll('input, select, textarea, button').forEach(c => { c.disabled = true; });
}

function setDirty(name, value) {
  // Env-pinned fields are read-only in the GUI — the env var takes precedence,
  // so never record a pending edit for them.
  if (isEnvPinned(name)) return;
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
}

// Re-render listeners on the model-overrides + model-list editors used to
// document.addEventListener() on every render — loadState() rebuilds those
// editors on every reload / save / login, leaking one listener per render.
// Register once per (event, key) instead; subsequent registrations remove
// the prior one so old closures (and their pinned DOM) can be GC'd.
const _adminListeners = {};
function _registerAdminListener(event, key, fn) {
  const k = event + '|' + key;
  if (_adminListeners[k]) document.removeEventListener(event, _adminListeners[k]);
  _adminListeners[k] = fn;
  document.addEventListener(event, fn);
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

// Field rows like MODEL_OVERRIDES are master-detail widgets that have their
// own visual structure (sidebar + detail pane) and do not benefit from the
// shared label-col / +reset / help-text chrome — the section header above
// already conveys what the widget is, and the widget renders its own legend.
// Listed here so fieldRow() can render them edge-to-edge in the section's
// .group-fields grid (grid-column: 1 / -1, no label-col, no description).
const FULLROW_FIELDS = new Set(['MODEL_OVERRIDES']);

function fieldRow(name) {
  const row = document.createElement('div');
  row.className = 'field';
  row.dataset.field = name;   // used by jumpToRule() + the global admin:dirty handler

  if (FULLROW_FIELDS.has(name)) {
    row.classList.add('field-fullrow');
    const fullEd = makeEditor(name);
    if (isEnvPinned(name)) { row.classList.add('env-pinned'); disableEnvPinnedEditor(fullEd); }
    row.appendChild(fullEd);
    return row;
  }

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
  const editor = makeEditor(name);
  if (isEnvPinned(name)) {
    row.classList.add('env-pinned');
    disableEnvPinnedEditor(editor);
  }
  inputCol.appendChild(editor);
  // Env-override warning: lives in the input column (not the label column)
  // so the long line wraps inside the value-column width. If it sits in
  // .label-col, CSS subgrid sizes the column to this prose's max-content
  // and blows the whole section's label track wide. .input-col .help
  // already has the right styling.
  if (isEnvPinned(name)) {
    const note = document.createElement('div');
    note.className = 'help help-env-pinned';
    note.textContent = 'Set by ' + fieldDef(name).env_var
      + ' — read-only here; unset the environment variable to edit it in the UI.';
    inputCol.appendChild(note);
  }

  // Inline description from FIELD_DESCRIPTIONS (single source of truth).
  // Surfaced via /settings/state's per-field payload.
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
  // Initial visibility uses the local resetWrap (the row isn't in the DOM
  // yet during render()); subsequent updates flow through the single
  // delegated 'admin:dirty' listener installed at DOMContentLoaded
  // (refreshFieldReset, by data-field lookup). Previously each fieldRow()
  // call registered its own document-level listener; row.replaceWith() and
  // full re-renders never removed them, so listeners (and pinned DOM nodes)
  // accumulated unboundedly across reload/discard/reset cycles.
  (function paintInitialResetVisibility() {
    const cur = currentValue(name);
    const def = fieldDef(name).default_value;
    const same = JSON.stringify(cur) === JSON.stringify(def);
    resetWrap.style.display = same ? 'none' : '';
  })();

  row.appendChild(inputCol);
  return row;
}

// Refresh one field's "↺ Reset to default" visibility by looking up the
// current row in the DOM. Called from the delegated admin:dirty listener.
function refreshFieldReset(name) {
  const row = document.querySelector('main .field[data-field="' + name + '"]');
  if (!row) return;
  const wrap = row.querySelector('.reset-wrap');
  if (!wrap) return;
  const cur = currentValue(name);
  const def = fieldDef(name).default_value;
  const same = JSON.stringify(cur) === JSON.stringify(def);
  wrap.style.display = same ? 'none' : '';
}

function makeEditor(name) {
  // Read through the dirty overlay so re-renders triggered by setDirty
  // (e.g. "↺ Reset to default" or list-editor mutations) reflect the
  // pending value immediately, not the stale server value.
  const v = currentValue(name);
  // PIPELINE_RULES gets its own list-of-rules editor (mixed row types,
  // drag-to-reorder, per-row test badge). Routed by name BEFORE shape checks
  // since the value is a list.
  if (name === 'PIPELINE_RULES') return makeRuleListEditor(name, v || [], 'full', {});
  // Captures-specific exclude list — same checklist widget as per-model
  // PIPELINE_RULES_EXCLUDE, sourced from the live PIPELINE_RULES. The
  // checklist passes `cb.checked` (= "rule will run") to onToggle, so the
  // semantic here is inverted from the field name: checked → REMOVE from
  // EXCLUDE, unchecked → ADD to EXCLUDE. `terminalToggleable: true` lets
  // the trainer drop "Trim edges" too — captures may want to preserve
  // trailing whitespace that the live /transcribe path strips.
  if (name === 'CAPTURES_PIPELINE_RULES_EXCLUDE') {
    const allRules = currentValue('PIPELINE_RULES') || [];
    return makeRuleListEditor(name, allRules, 'checklist', {
      excludeSet: new Set(v || []),
      includeSet: new Set(),
      terminalToggleable: true,
      onToggle: (slug, wantActive) => {
        const cur = currentValue(name) || [];
        const next = wantActive
          ? cur.filter(s => s !== slug)
          : Array.from(new Set([...cur, slug]));
        setDirty(name, next);
      },
    });
  }
  // MODEL_OVERRIDES is a dict[model_id, dict[field, value]] — too freeform
  // for the standard editors. Render as a JSON textarea with parse-validation
  // on every input. Save sends the parsed object; pydantic validates server-
  // side. Future polish: master-detail UI per the original design.
  if (name === 'MODEL_OVERRIDES') return modelOverridesEditor(name, v || {});
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

// =============================================================================
// MODEL_OVERRIDES editor — master-detail UI
// =============================================================================
// Sidebar: [Global ☑ dim toggle] + per-model entries (with override counts +
// collapse indicator on the active row) + [+ add model dropdown] +
// [× remove model entry]. The toggle row replaces the old "Global (read-only
// ref)" navigation entry AND the standalone "Compare diff to global"
// checkbox. Main pane: per-model edit view (6 field sections + Pipeline
// rules checklist) — no top header bar, no chevron. Clicking the active
// sidebar entry collapses/expands the detail pane (state persisted in
// localStorage['mo.detail.<id>']). Per-field rows show ●/○ inherit/override
// dots, an input widget when overridden, "inherits {global}" + "+ override"
// when not. Diff-to-global toggle dims rows whose value matches the resolved
// global. Live: every input change refreshes dot/match/count/inherits text
// in place via refreshMarkers(field) — no input rebuild, focus preserved.
// Advanced subgroups persist their own open/closed state under mo.adv.<id>.
function modelOverridesEditor(name, v) {
  // Authoritative state. Local mutable copy of the saved overrides; full
  // snapshot pushed via setDirty(name, snapshot) on every edit so the existing
  // /settings/state save flow sees one consistent payload.
  let overrides = JSON.parse(JSON.stringify(v || {}));
  // Diff-mode default ON: most useful first state — the user immediately
  // sees which fields differ from global. Toggleable via the top sidebar row.
  let selectedId = null;     // chosen during initial paint below
  let diffMode = true;
  // Live handle to the per-model pipeline section so refreshPipelineSection()
  // can replaceWith() it on global PIPELINE_RULES changes.
  let pipelineSectionEl = null;

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
              'SUPPRESS_CHARS','PREPEND_PUNCTUATIONS','APPEND_PUNCTUATIONS'] },
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

  // Per-field widget metadata. Mirrors ModelOverride pydantic constraints.
  // Kept compact — extend a row only if behavior differs from a generic
  // input.
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
    SUPPRESS_CHARS:              { kind: 'string', placeholder: 'chars to mask, e.g. .,?!:;' },
    PREPEND_PUNCTUATIONS:        { kind: 'string' },
    APPEND_PUNCTUATIONS:         { kind: 'string' },
    OUTPUT_PREFIX:               { kind: 'string' },
    OUTPUT_SUFFIX:               { kind: 'string' },
    CAPTURES_SAMPLE_MIN_DURATION_S:  { kind: 'float', min: 0, max: 30, step: 0.1 },
    CAPTURES_SAMPLE_MAX_DURATION_S:  { kind: 'float', min: 1, max: 30, step: 0.1 },
    CAPTURES_SAMPLE_JOIN_STRATEGY:   { kind: 'enum', opts: ['space','period_space'] },
    CAPTURES_PROPOSER_TARGET_S:      { kind: 'float', min: 1, max: 30, step: 0.5 },
    CAPTURES_PROPOSER_SESSION_GAP_S: { kind: 'int',   min: 1, max: 86400 },
    CAPTURES_PROPOSER_DUP_THRESHOLD: { kind: 'float', min: 0, max: 1, step: 0.01 },
    CAPTURES_PROPOSER_MAX_PROPOSALS: { kind: 'int',   min: 1, max: 200 },
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
    // Live update: dot, data-matches-global, sidebar count, "inherits" text
    // refresh in place — no full re-render means typing focus is preserved.
    refreshMarkers(field);
  }
  function clearOverride(modelId, field) {
    if (overrides[modelId]) delete overrides[modelId][field];
    persist();
    refreshMarkers(field);
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

  // -------- live marker refresh ------------------------------------------
  // Recompute the per-row dot / data-matches-global / "inherits {global}"
  // text and the sidebar override count + dot, WITHOUT rebuilding any input
  // widgets. Cheap (handful of querySelectorAll), preserves typing focus.
  // `field` (optional): only that field's row gets updated. Otherwise all.
  function refreshMarkers(field) {
    // 1. Sidebar: per-model entry override count + dot.
    for (const li of sidebar.querySelectorAll('.mo-list-item')) {
      const id = li.dataset.modelId;
      if (!id) continue;
      const n = countOverrides(id);
      const cnt = li.querySelector('.mo-count');
      if (cnt) cnt.textContent = n ? '(' + n + ')' : '';
      const dot = li.querySelector('.mo-dot');
      if (dot) {
        dot.classList.toggle('override', n > 0);
        dot.classList.toggle('inherit',  n === 0);
      }
    }

    // 2. Per-row markers in the detail pane.
    const rows = mainpane.querySelectorAll('.mo-row[data-mo-field]');
    for (const row of rows) {
      const f = row.dataset.moField;
      if (field && f !== field) continue;
      const overrideVal = getOverrideValue(selectedId, f);
      const isOverridden = overrideVal !== undefined;
      const globalVal = globalValue(f);
      const matches = !isOverridden ||
        JSON.stringify(overrideVal) === JSON.stringify(globalVal);
      row.dataset.matchesGlobal = matches ? 'true' : 'false';
      const dot = row.querySelector('.mo-dot');
      if (dot) {
        dot.classList.toggle('override', isOverridden);
        dot.classList.toggle('inherit',  !isOverridden);
      }
      // Only the not-overridden branch renders a .mo-inherits hint; if the
      // global value just changed, that text needs to follow. (When the row
      // IS overridden, the input widget shows the override value — already
      // live via the input element itself; no DOM patch needed.)
      const inh = row.querySelector('.mo-inherits');
      if (inh) inh.textContent = 'inherits ' + fmtValue(globalVal);
    }
  }

  // Re-render the per-model pipeline checklist in place. The checklist has
  // no typing/drag/scroll/expanded-body state, so a clean rebuild is cheaper
  // and more correct than per-row patching across add/remove/rename/reorder.
  function refreshPipelineSection() {
    if (!pipelineSectionEl || !pipelineSectionEl.parentElement) return;
    if (selectedId === null) return;        // empty-state pane has no section
    const fresh = renderPipelineSection();
    pipelineSectionEl.replaceWith(fresh);
    pipelineSectionEl = fresh;
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
    // an entry). German-aware, case-insensitive order (model IDs are mixed-case).
    const ids = Object.keys(overrides).sort(new Intl.Collator('de', { sensitivity: 'base', numeric: true }).compare);

    // Top row: diff-mode toggle. Replaces the old "Global (read-only ref)"
    // entry AND the standalone "Compare diff to global" checkbox. The whole
    // row toggles diffMode; clicking anywhere flips the checkbox.
    const toggleWrap = document.createElement('label');
    toggleWrap.className = 'mo-sidebar-toggle';
    toggleWrap.title = 'Dim per-model rows whose value equals the resolved global';
    const toggleCb = document.createElement('input');
    toggleCb.type = 'checkbox';
    toggleCb.checked = diffMode;
    toggleCb.addEventListener('change', () => {
      diffMode = toggleCb.checked;
      mainpane.classList.toggle('diff-mode', diffMode);
    });
    toggleWrap.appendChild(toggleCb);
    const toggleLbl = document.createElement('span');
    toggleLbl.className = 'mo-toggle-label';
    toggleLbl.textContent = 'Global — dim what matches';
    toggleWrap.appendChild(toggleLbl);
    sidebar.appendChild(toggleWrap);

    // Per-model entries.
    for (const id of ids) {
      const item = document.createElement('div');
      item.className = 'mo-list-item';
      item.dataset.modelId = id;
      if (id === selectedId) item.classList.add('active');
      const n = countOverrides(id);
      const dot = '<span class="mo-dot ' + (n > 0 ? 'override' : 'inherit') + '"></span>';
      // Long HF-style ids would blow the sidebar width — let CSS truncate.
      const lbl = '<span class="mo-list-label" title="' + id + '">' + id + '</span>';
      const cnt = '<span class="mo-count">' + (n ? '(' + n + ')' : '') + '</span>';
      // Collapse indicator: ▾ when expanded, ▸ when collapsed. Only the
      // active row reveals it via .active CSS opacity.
      const collapsed = (localStorage.getItem('mo.detail.' + id) === '0');
      const ind = '<span class="mo-collapse-ind">' + (collapsed ? '▸' : '▾') + '</span>';
      item.innerHTML = dot + lbl + cnt + ind;
      item.addEventListener('click', () => {
        if (selectedId === id) {
          // Active row → toggle collapse for this model.
          const isCollapsed = (localStorage.getItem('mo.detail.' + id) === '0');
          try {
            localStorage.setItem('mo.detail.' + id, isCollapsed ? '1' : '0');
          } catch (_) {}
          applyCollapseState();
          // Refresh the indicator on this entry only.
          const indEl = item.querySelector('.mo-collapse-ind');
          if (indEl) indEl.textContent = isCollapsed ? '▾' : '▸';
        } else {
          selectedId = id;
          renderAll();
        }
      });
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
        // Pick a sensible new selection: first remaining override-bearing
        // model, else first allowed model, else null.
        const remaining = Object.keys(overrides).sort(new Intl.Collator('de', { sensitivity: 'base', numeric: true }).compare);
        if (remaining.length > 0) selectedId = remaining[0];
        else selectedId = allowed[0] || null;
        persist();
        renderAll();
      });
      sidebar.appendChild(rmBtn);
    }
  }

  // -------- collapse state ------------------------------------------------
  // Apply the saved collapsed/expanded state of the currently-selected
  // model to the mainpane. localStorage['mo.detail.<id>'] === '0' means
  // explicitly collapsed; anything else (including absence) means expanded.
  function applyCollapseState() {
    if (!selectedId) {
      mainpane.classList.remove('collapsed');
      return;
    }
    const isCollapsed = (localStorage.getItem('mo.detail.' + selectedId) === '0');
    mainpane.classList.toggle('collapsed', isCollapsed);
  }

  // -------- main pane: dispatcher ----------------------------------------
  function renderMain() {
    mainpane.innerHTML = '';
    mainpane.classList.toggle('diff-mode', diffMode);
    if (selectedId === null) {
      renderEmptyState();
    } else {
      renderModelEditView();
    }
    applyCollapseState();
  }

  // -------- main pane: empty state (no allowed models) -------------------
  function renderEmptyState() {
    const note = document.createElement('div');
    note.className = 'help';
    note.textContent = 'No model selected. Add a model to ALLOWED_MODELS in the Models section above, '
      + 'then add an override entry from the sidebar.';
    mainpane.appendChild(note);
  }

  // -------- main pane: Model edit view -----------------------------------
  function renderModelEditView() {
    const body = document.createElement('div');
    body.className = 'mo-detail-body';
    for (const sec of SECTIONS) {
      body.appendChild(renderSection(sec));
    }
    pipelineSectionEl = renderPipelineSection();
    body.appendChild(pipelineSectionEl);
    mainpane.appendChild(body);
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
    row.className = 'mo-row';
    // Use a distinct attribute (not data-field) so the diff dim selector
    // doesn't collide with the global fieldRow's data-field. The global
    // editor uses data-field on its rows; we use data-mo-field to keep the
    // jump-link selectors unambiguous.
    row.dataset.moField = field;
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
      // Empty input clears the override (cfg_for treats null and missing
      // identically — there is no "explicitly disabled" state to preserve).
      const inp = document.createElement('input');
      inp.type = 'number';
      if (meta.min !== undefined) inp.min = meta.min;
      if (meta.max !== undefined) inp.max = meta.max;
      if (meta.step !== undefined) inp.step = meta.step;
      inp.value = currentVal == null ? '' : currentVal;
      inp.addEventListener('input', () => {
        const raw = inp.value;
        if (raw === '') {
          setOverrideValue(selectedId, field, undefined);
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
        refreshMarkers();  // sidebar count + dot, in place (parity with field rows)
      },
      onJumpToGlobal: (slug) => jumpToRule(slug),
    };
    secEl.appendChild(makeRuleListEditor('PIPELINE_RULES', rules, 'checklist', ruleOpts));
    return secEl;
  }

  // -------- jump-link helpers --------------------------------------------
  // The global field row uses `data-field="X"`; per-model rows here use
  // `data-mo-field="X"` to avoid clashing.
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
  // Initial selection: the alphabetically-first model with overrides;
  // otherwise null (renders the empty-state hint in the detail pane). The
  // Global ref view is gone — when no model has an override yet the user
  // is told to use the "+ add model override" dropdown.
  (function() {
    const overridden = Object.keys(overrides).filter(id => countOverrides(id) > 0).sort(new Intl.Collator('de', { sensitivity: 'base', numeric: true }).compare);
    if (overridden.length > 0) selectedId = overridden[0];
  })();

  function renderAll() {
    renderSidebar();
    renderMain();
  }
  renderAll();
  // Re-render the sidebar on ALLOWED_MODELS changes (the "+ add model"
  // dropdown sources from there). Same event the DEFAULT_MODEL dropdown
  // and PRELOAD_MODELS multi-select listen for.
  _registerAdminListener('admin:model-lists-changed', 'modelOverrides:sidebar', renderSidebar);
  // Live diff: when a global field changes (in any other editor on the
  // page) the per-model rows' "matches global" / "inherits {x}" / dot
  // states must follow. Filter our own MODEL_OVERRIDES dirty-events out
  // (we already updated locally). PIPELINE_RULES routes to the checklist
  // refresher instead of refreshMarkers — the per-model rule rows live
  // outside the .mo-row[data-mo-field] world refreshMarkers walks.
  _registerAdminListener('admin:dirty', 'modelOverrides:dirty', (e) => {
    if (!e.detail) { refreshMarkers(); refreshPipelineSection(); return; }
    const n = e.detail.name;
    if (n === name) return;
    if (n === 'PIPELINE_RULES') { refreshPipelineSection(); return; }
    refreshMarkers(n);
  });
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
  _registerAdminListener('admin:model-lists-changed', 'modelDropdown:' + name, render);
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
        warn.style.color = 'var(--yellow)';
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
  _registerAdminListener('admin:model-lists-changed', 'modelMultiSelect:' + name, render);
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

// _esc, _unesc, _PIPELINE_TYPES, _typePill, renderTypeEditor,
// _makeMonoLabeledInput, _makeMapRow live in web_common.RULE_EDITOR_JS,
// rendered into a sibling script tag above so /quick-config can reuse
// them. Don't redefine them here.

function _ensureUniqueSlug(slug, existing) {
  if (!existing.has(slug)) return slug;
  let n = 2;
  while (existing.has(`${slug}-${n}`)) n++;
  return `${slug}-${n}`;
}

// Live status check for one rule against the current test panel sample.
// Hits POST /settings/test-pipeline with a single-rule list. Returns the step
// dict or null on transport error.
async function _testOneRule(rule) {
  const panelSample = document.getElementById('pipeline-test-sample');
  const sample = panelSample ? panelSample.value : TEST_PRESETS['default'];
  const r = await api('POST', '/settings/test-pipeline', {
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

  // --- factory baseline (config.json) ---------------------------------
  // `factoryRules` mirrors the committed config.json: seeded from the page's
  // default_value (cfg._BASELINE) and refreshed from every successful
  // POST /settings/factory-rules response, so origin badges, the promote
  // affordances and reset buttons stay correct after a promote.
  let factoryRules = JSON.parse(JSON.stringify(
    (fieldDef(name) && fieldDef(name).default_value) || []));
  function _baselineList() { return factoryRules; }
  function _factoryRule(slug) {
    for (const b of factoryRules) if (b.name === slug) return b;
    return null;
  }
  function _factoryHas(slug) { return _factoryRule(slug) !== null; }
  // Deep key-sort so two rules with equal content but different key order
  // (a freshly-added rule vs the server-canonicalized copy) compare equal.
  function _sortDeep(o) {
    if (Array.isArray(o)) return o.map(_sortDeep);
    if (o && typeof o === 'object') {
      const out = {};
      for (const k of Object.keys(o).sort()) out[k] = _sortDeep(o[k]);
      return out;
    }
    return o;
  }
  // Content equality ignoring the vestigial `seeded` flag (config.json is
  // always seeded:true; a local-only rule is seeded:false — not a real diff)
  // and the server-owned `map_meta` timestamps (present on disk once a cb:map
  // entry is touched via /quick-config, absent from the config.json baseline —
  // not a functional diff, so it must not flip the origin badge to 'edited').
  function _ruleContentEqual(a, b) {
    const strip = (r) => {
      const c = Object.assign({}, r);
      delete c.seeded;
      delete c.map_meta;
      return _sortDeep(c);
    };
    return JSON.stringify(strip(a)) === JSON.stringify(strip(b));
  }
  // 'factory'    — in config.json, identical
  // 'edited'     — in config.json, content differs (local edit)
  // 'local-only' — not in config.json
  function _ruleStatus(rule) {
    if (!rule || rule.type === 'terminal') return 'factory';
    const base = _factoryRule(rule.name);
    if (!base) return 'local-only';
    return _ruleContentEqual(rule, base) ? 'factory' : 'edited';
  }
  function _seededOrderDirty() {
    const baseOrder = factoryRules.filter(b => b.type !== 'terminal').map(b => b.name);
    const curOrder = rules.filter(r => r.type !== 'terminal' && _factoryHas(r.name))
                          .map(r => r.name);
    return JSON.stringify(curOrder)
         !== JSON.stringify(baseOrder.filter(n => curOrder.indexOf(n) >= 0));
  }
  function _paintBadge(el, st) {
    el.className = 'rule-origin-badge ' + st;
    el.textContent = st === 'edited' ? '◆ edited'
                   : st === 'local-only' ? '✚ local-only'
                   : '● factory';
    el.title = st === 'edited'
        ? 'Differs from config.json — local edit not yet promoted'
      : st === 'local-only'
        ? 'Not in config.json — local-only rule'
        : 'Matches config.json';
  }

  function refreshControlsVisibility() {
    if (isChecklist) return;   // checklist rows have no reset/dirty controls
    // Per row: repaint the origin badge, toggle the ⇪ promote button, and
    // show the per-row reset button only when the rule differs from config.json.
    let anyDirty = false;        // some rule is 'edited'
    let anyPromotable = false;   // some rule is 'edited' or 'local-only'
    list.querySelectorAll('.rule-row').forEach(r => {
      const idx = parseInt(r.dataset.idx, 10);
      const rule = rules[idx];
      if (!rule) return;
      const st = _ruleStatus(rule);
      const promotable = st !== 'factory' && rule.type !== 'terminal';
      if (st === 'edited') anyDirty = true;
      if (promotable) anyPromotable = true;
      const badge = r.querySelector('.rule-origin-badge');
      if (badge) _paintBadge(badge, st);
      const pb = r.querySelector('.promote-btn');
      if (pb) pb.style.display = promotable ? '' : 'none';
      const btn = r.querySelector('.reset-link');
      if (btn) btn.style.display = (st === 'edited') ? '' : 'none';
    });
    // List-wide controls: hide when there is nothing to act on.
    const orderDirty = _seededOrderDirty();
    if (resetOrderBtn) {
      resetOrderBtn.style.display = orderDirty ? '' : 'none';
    }
    if (resetAllBtn) {
      resetAllBtn.style.display = (anyDirty || orderDirty) ? '' : 'none';
    }
    if (promoteAllBtn) {
      promoteAllBtn.style.display = (anyPromotable || orderDirty) ? '' : 'none';
    }
  }

  // --- commit helpers --------------------------------------------------
  // Inline edits (typing into pattern/replacement/wordlist/map) MUST NOT
  // rebuild the DOM — that would steal focus mid-keystroke and collapse
  // any other expanded rows. Structural changes (add/delete/reorder/reset/
  // toggle-enabled) DO rebuild because the visible row layout changes.
  // Both push the whole rule list into the global dirty overlay via setDirty
  // so the page-level Save → /settings/state persists it to config.local.json.
  // Promoting (a separate action) writes config.json instead.
  function commitData() {
    if (isChecklist) return;   // checklist mode is read-only — no setDirty
    setDirty(name, JSON.parse(JSON.stringify(rules)));
    refreshControlsVisibility();
  }
  function commitFull() {
    setDirty(name, JSON.parse(JSON.stringify(rules)));
    paintAll();
  }

  function paintAll() {
    list.innerHTML = '';
    rules.forEach((rule, idx) => list.appendChild(renderRow(rule, idx)));
    refreshControlsVisibility();
  }

  // Union of every tag currently set on any rule in this list — used as
  // the tag-picker autocomplete source so admins don't have to retype
  // an existing tag's spelling. Computed at render time so newly-added
  // tags propagate without a reload.
  function _allRuleTags() {
    const seen = {};
    for (const r of rules) {
      for (const t of (r.tags || [])) {
        if (typeof t === 'string' && t) seen[t] = true;
      }
    }
    return Object.keys(seen).sort();
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
      // Diff-to-global hook: a rule whose slug is in neither EXCLUDE nor
      // INCLUDE is "functionally inherited" — same contract as .mo-row's
      // data-matches-global. The diff-mode CSS dims rows tagged 'true'.
      row.dataset.matchesGlobal = (!forcedOut && !forcedIn) ? 'true' : 'false';

      const head = document.createElement('div');
      head.className = 'row-header';

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = effective;
      // Per-model PIPELINE_RULES_EXCLUDE: the live /transcribe path
      // re-trims after wrappers unconditionally, so a per-model exclude
      // on `trim-edges` is a no-op — the checkbox stays locked to avoid
      // misleading the admin. CAPTURES_PIPELINE_RULES_EXCLUDE opts in to
      // toggling via `terminalToggleable: true`, since the captures
      // pipeline honors the exclude end-to-end.
      if (isTerminal && !opts.terminalToggleable) {
        cb.disabled = true;
        cb.title = 'Terminal trim — always runs, cannot be excluded per model';
      } else if (isTerminal) {
        cb.title = forcedOut
          ? 'Force-disabled for captures. Check to let the trim strip '
          + 'trailing whitespace from training text.'
          : 'Active for captures. Uncheck to preserve trailing whitespace '
          + 'in stored training text (live /transcribe still trims).';
        cb.addEventListener('change', () => {
          if (opts.onToggle) opts.onToggle(rule.name, cb.checked, globallyEnabled);
          const newForcedOut = !cb.checked;
          row.classList.toggle('excluded', newForcedOut);
          row.dataset.matchesGlobal = newForcedOut ? 'false' : 'true';
          status.textContent = newForcedOut ? 'EXCLUDED' : '';
          if (footer) footer.textContent = _footerText();
        });
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
          // Diff-mode dim follows the new forced state synchronously — no
          // attribute means the dim CSS would freeze on the render-time value.
          row.dataset.matchesGlobal = (!newForcedOut && !newForcedIn) ? 'true' : 'false';
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

    // ----- Labelled-toggle factory (used for `enabled` on line 1 and
    // `user-editable` on line 2). Same shape, same green-when-on cue,
    // same dim-when-off — readable as a parallel pair. ------------------
    function _labelledToggle(text, checked, opts) {
      opts = opts || {};
      const wrap = document.createElement('label');
      wrap.className = 'toggle';
      if (checked) wrap.classList.add('on');
      if (opts.title) wrap.title = opts.title;
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = !!checked;
      if (opts.disabled) cb.disabled = true;
      cb.addEventListener('change', () => {
        wrap.classList.toggle('on', cb.checked);
        if (opts.onChange) opts.onChange(cb.checked);
      });
      wrap.appendChild(cb);
      const lbl = document.createElement('span');
      lbl.textContent = text;
      wrap.appendChild(lbl);
      return wrap;
    }

    // Header is now a flex COLUMN with two children — line 1 (identity)
    // and line 2 (metadata). The previous single-line layout meant the
    // rule-label was the first thing to truncate when 10+ inline
    // controls fought for horizontal space.
    const head = document.createElement('div');
    head.className = 'row-header';
    const headLine1 = document.createElement('div');
    headLine1.className = 'row-header-line1';
    head.appendChild(headLine1);
    const headLine2 = document.createElement('div');
    headLine2.className = 'row-header-line2';

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
    headLine1.appendChild(drag);

    // `enabled` toggle on line 1 — gets a visible label "enabled" (not
    // just a tooltip). Terminal rules disable the toggle but still show
    // it for visual rhythm.
    const enabledToggle = _labelledToggle('enabled', !!rule.enabled, {
      title: 'Enable / disable this rule',
      disabled: rule.type === 'terminal',
      onChange: (val) => {
        rule.enabled = val;
        commitFull();
      },
    });
    headLine1.appendChild(enabledToggle);

    // Line 2 is built later (after we've decided whether this is a
    // terminal row — those skip line 2 entirely). The badge variable
    // is hoisted so _syncUntaggedBadge() can find it before line 2's
    // construction completes.
    let _untaggedBadge = null;
    function _syncUntaggedBadge() {
      if (!_untaggedBadge) return;
      const isUntagged = !rule.tags || !rule.tags.length;
      const visible = !!rule.exposed && isUntagged;
      _untaggedBadge.style.display = visible ? '' : 'none';
    }
    _syncUntaggedBadge();

    const ord = document.createElement('span');
    ord.className = 'ordinal';
    ord.textContent = '#' + (idx + 1);
    headLine1.appendChild(ord);

    const lbl = document.createElement('span');
    lbl.className = 'rule-label';
    lbl.textContent = rule.label || rule.name;
    lbl.title = rule.label || rule.name;   // hover reveals full text when ellipsised
    headLine1.appendChild(lbl);

    const slug = document.createElement('span');
    slug.className = 'rule-slug';
    slug.textContent = '(' + (rule.name || '?') + ')';
    headLine1.appendChild(slug);

    const pill = document.createElement('span');
    pill.className = 'type-pill';
    pill.textContent = _typePill(rule.type);
    headLine1.appendChild(pill);

    // Origin badge — factory / edited / local-only vs config.json. Repainted
    // live by refreshControlsVisibility. Skipped for the terminal rule.
    if (rule.type !== 'terminal') {
      const originBadge = document.createElement('span');
      originBadge.className = 'rule-origin-badge';
      _paintBadge(originBadge, _ruleStatus(rule));
      headLine1.appendChild(originBadge);
    }

    const expandBtn = document.createElement('button');
    expandBtn.type = 'button';
    expandBtn.className = 'expand-btn';
    const _setExpandLabel = () => {
      expandBtn.textContent = row.classList.contains('expanded') ? 'edit ▴' : 'edit ▾';
    };
    _setExpandLabel();
    expandBtn.addEventListener('click', () => {
      row.classList.toggle('expanded');
      if (row.classList.contains('expanded')) {
        expandedNames.add(rule.name);
        if (rule.type !== 'terminal') refresh();
      } else {
        expandedNames.delete(rule.name);
      }
      _setExpandLabel();
    });
    headLine1.appendChild(expandBtn);

    // ⇪ Promote — push this rule into the committed config.json. Shown only
    // for rules that differ from config.json (refreshControlsVisibility
    // toggles visibility); never for the terminal rule.
    if (rule.type !== 'terminal') {
      const promoteBtn = document.createElement('button');
      promoteBtn.type = 'button';
      promoteBtn.className = 'promote-btn';
      promoteBtn.textContent = '⇪ promote';
      promoteBtn.title = 'Promote this rule into the committed config.json';
      promoteBtn.style.display = 'none';
      promoteBtn.addEventListener('click', () => _promoteOne(rule));
      headLine1.appendChild(promoteBtn);
    }

    // ----- Line 2 (metadata): tag picker + user-editable toggle +
    // untagged badge. Terminal rules skip it entirely — they have no
    // tags, no exposed flag, no user-editable concept. ---------------
    if (rule.type !== 'terminal') {
      const tagsLbl = document.createElement('span');
      tagsLbl.className = 'meta-label';
      tagsLbl.textContent = 'tags:';
      headLine2.appendChild(tagsLbl);

      const tagWrap = document.createElement('span');
      tagWrap.className = 'rule-tag-wrap';
      const picker = window._renderTagPicker({
        initial: Array.isArray(rule.tags) ? rule.tags : [],
        available: _allRuleTags(),
        placeholder: '+ tag',
        onChange: (newTags) => {
          rule.tags = newTags;
          _syncUntaggedBadge();
          commitFull();
        },
      });
      tagWrap.appendChild(picker.el);
      headLine2.appendChild(tagWrap);

      // `user-editable` toggle (was the confusing "simple" string).
      // Tooltip explains the where; the visible label is the truthful
      // effect description.
      const exposedToggle = _labelledToggle('user-editable', !!rule.exposed, {
        title: 'Show on /quick-config so non-admin users can edit '
             + 'this rule’s body.',
        onChange: (val) => {
          rule.exposed = val;
          _syncUntaggedBadge();
          commitFull();
        },
      });
      headLine2.appendChild(exposedToggle);

      // Yellow "visible to all users" badge — shown when the rule is
      // exposed AND has no tags (the asymmetric default that lets it
      // reach every non-admin user). Helps admins spot rules they
      // probably wanted to scope but forgot to tag.
      _untaggedBadge = document.createElement('span');
      _untaggedBadge.className = 'rule-untagged-badge';
      _untaggedBadge.textContent = '⚠ untagged · visible to all users';
      _untaggedBadge.title = 'Exposed but untagged — every authenticated user '
        + 'with /quick-config access sees this rule. Add tags to scope it '
        + 'to specific users.';
      headLine2.appendChild(_untaggedBadge);
      _syncUntaggedBadge();

      head.appendChild(headLine2);
    }

    // ----- Body footer destructive actions (reset / delete). Moved
    // out of the head row per the new layout — these are rare +
    // destructive, so they live in the expanded body where mis-clicks
    // are less likely (PatternFly guidance). The variables are built
    // here so the closures capture `rule`/`idx`/`expandedNames`; the
    // button itself is appended to the body footer further down. ------
    const _destructiveBtns = [];
    // A rule present in config.json (factory / edited) can be reset to its
    // committed value; a local-only rule (not in config.json) can be deleted.
    // Membership in config.json — not the `seeded` flag — is authoritative.
    const _canReset = rule.type !== 'terminal' && _factoryHas(rule.name);
    const _canDelete = rule.type !== 'terminal' && !_factoryHas(rule.name);
    if (_canReset) {
      const reset = document.createElement('button');
      reset.type = 'button';
      reset.className = 'reset-link';
      reset.textContent = '↺ reset to default';
      reset.title = 'Discard the local edit — restore this rule to its config.json value';
      reset.style.display = 'none';   // refreshControlsVisibility shows it when 'edited'
      reset.addEventListener('click', () => {
        const baseline = _baselineList().find(b => b.name === rule.name);
        if (!baseline) return;
        rules[idx] = JSON.parse(JSON.stringify(baseline));
        commitFull();
      });
      _destructiveBtns.push(reset);
    } else if (_canDelete) {
      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'delete-btn';
      del.textContent = '× delete';
      del.title = 'Remove this local-only rule';
      del.addEventListener('click', () => {
        expandedNames.delete(rule.name);
        rules.splice(idx, 1);
        commitFull();
      });
      _destructiveBtns.push(del);
    }
    row.appendChild(head);

    // Drop logic lives at list level (see top of makeRuleListEditor) —
    // the shared placeholder follows the cursor between rows there.

    // Body (collapsed by default).
    const body = document.createElement('div');
    body.className = 'row-body';

    // Label editor — skipped for terminal (hardcoded label). For seeded
    // rules the slug stays fixed (it's the contract with the in-repo
    // baseline). For custom rules the slug regenerates from the label so
    // the kebab-case slug stays in sync without a separate field. Live
    // updates the header lbl/slug spans without a full repaint to
    // preserve focus/caret position during typing.
    if (rule.type !== 'terminal') {
      const labelWrap = document.createElement('div');
      labelWrap.className = 'rule-label-edit';
      const labelLbl = document.createElement('div');
      labelLbl.className = 'help';
      labelLbl.textContent = rule.seeded
        ? 'label: (slug stays fixed for seeded rules)'
        : 'label: (slug regenerates from label)';
      labelWrap.appendChild(labelLbl);
      const labelInp = document.createElement('input');
      labelInp.type = 'text';
      labelInp.value = rule.label || '';
      labelInp.spellcheck = false;
      labelInp.style.width = '100%';
      labelInp.addEventListener('input', () => {
        rule.label = labelInp.value;
        lbl.textContent = labelInp.value || rule.name;
        if (!rule.seeded) {
          const others = new Set(rules.filter(r => r !== rule).map(r => r.name));
          const newSlug = _ensureUniqueSlug(_slugify(labelInp.value), others);
          if (newSlug !== rule.name) {
            if (expandedNames.has(rule.name)) {
              expandedNames.delete(rule.name);
              expandedNames.add(newSlug);
            }
            rule.name = newSlug;
            slug.textContent = '(' + newSlug + ')';
            row.dataset.slug = newSlug;
          }
        }
        commitData();
      });
      labelWrap.appendChild(labelInp);
      body.appendChild(labelWrap);
    }

    body.appendChild(renderTypeEditor(rule, commitData, { showNote: true }));

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
      // Only kick off the initial test if this row is rendered expanded;
      // for collapsed rows the body is display:none, so the test output
      // would be invisible. The expand-click handler fires refresh() on
      // first reveal instead.
      if (row.classList.contains('expanded')) requestAnimationFrame(refresh);
    }

    // Destructive actions live in a body-footer (out of the scannable
    // head row). Append AFTER the rule editor + test panel so the
    // visual order reads: edit body → run test → reset/delete.
    if (_destructiveBtns.length) {
      const footer = document.createElement('div');
      footer.className = 'row-body-actions';
      for (const btn of _destructiveBtns) footer.appendChild(btn);
      body.appendChild(footer);
    }

    row.appendChild(body);
    return row;
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
  ctrls.style.marginTop = '0.5rem';
  ctrls.style.display = 'flex';
  ctrls.style.gap = '0.5rem';

  const addBtn = document.createElement('button');
  addBtn.type = 'button';
  addBtn.textContent = '+ Add custom rule';
  addBtn.addEventListener('click', () => _openAddCustomDialog());
  ctrls.appendChild(addBtn);

  // ⇪ Promote all — write every local change (edited + new rules) into the
  // committed config.json. refreshControlsVisibility shows it only when
  // something differs from config.json.
  const promoteAllBtn = document.createElement('button');
  promoteAllBtn.type = 'button';
  promoteAllBtn.className = 'promote-all-btn';
  promoteAllBtn.textContent = '⇪ Promote all changes to config.json';
  promoteAllBtn.title = 'Write every local change into the committed config.json';
  promoteAllBtn.style.display = 'none';
  promoteAllBtn.addEventListener('click', () => _promoteAll());
  ctrls.appendChild(promoteAllBtn);

  const resetOrderBtn = document.createElement('button');
  resetOrderBtn.type = 'button';
  resetOrderBtn.className = 'reset-link';
  resetOrderBtn.textContent = '↺ Reset order';
  resetOrderBtn.title = 'Restore config.json rule order; local-only rules append before terminal';
  resetOrderBtn.addEventListener('click', () => {
    const baseOrder = factoryRules.map(b => b.name);
    const factory = [];
    const customs = [];
    let terminal = null;
    rules.forEach(r => {
      if (r.type === 'terminal') terminal = r;
      else if (_factoryHas(r.name)) factory.push(r);
      else customs.push(r);
    });
    factory.sort((a, b) => baseOrder.indexOf(a.name) - baseOrder.indexOf(b.name));
    rules = [...factory, ...customs];
    if (terminal) rules.push(terminal);
    commitFull();
  });
  ctrls.appendChild(resetOrderBtn);

  const resetAllBtn = document.createElement('button');
  resetAllBtn.type = 'button';
  resetAllBtn.className = 'reset-link';
  resetAllBtn.textContent = '↺ Reset all to config.json';
  resetAllBtn.title = 'Discard every local edit — restore all rules to their config.json values';
  resetAllBtn.addEventListener('click', () => {
    const customs = rules.filter(r => !_factoryHas(r.name) && r.type !== 'terminal');
    const ok = confirm(
      'Discard every local edit and restore all rules to their config.json values.\n' +
      (customs.length ? `Your ${customs.length} local-only rule(s) will be kept at their current positions.\n\n` : '\n') +
      'Continue?'
    );
    if (!ok) return;
    // Replace each config.json-backed rule with its committed copy.
    rules = rules.map(r => _factoryHas(r.name)
      ? JSON.parse(JSON.stringify(_factoryRule(r.name)))
      : r);
    commitFull();
  });
  ctrls.appendChild(resetAllBtn);

  wrap.appendChild(ctrls);

  function _openAddCustomDialog() {
    // Lightweight inline form, appended at the bottom of the rules list.
    const form = document.createElement('div');
    form.className = 'rule-row';
    form.style.borderColor = 'var(--green)';
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
    labelInp.style.width = '100%'; labelInp.style.marginTop = '0.25rem';

    const ok = document.createElement('button');
    ok.type = 'button'; ok.textContent = 'Add'; ok.style.marginTop = '0.375rem';
    const cancel = document.createElement('button');
    cancel.type = 'button'; cancel.textContent = 'Cancel'; cancel.style.marginLeft = '0.375rem';

    body.appendChild(_labeledRow('Type', typeSel));
    body.appendChild(_labeledRow('Label', labelInp));
    const hint = document.createElement('div');
    hint.className = 'help';
    hint.style.marginTop = '0.375rem';
    hint.textContent = 'Type-specific fields (pattern, replacement, wordlist, map) '
      + 'open in the rule body after Add.';
    body.appendChild(hint);
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
      // A newly-added rule is local-only (seeded:false) until promoted to
      // config.json — save_factory_rules normalises seeded:true on promote.
      const newRule = {
        name: slug, label: lbl, type: t,
        enabled: true, locked: false, seeded: false,
      };
      // Seed empty type-specific fields; the user fills them in via the
      // expanded body editor (auto-opened below). Empty patterns are
      // skipped at runtime so the new rule is harmless until edited.
      if (t === 'regex') {
        newRule.pattern = '';
        newRule.replacement = '';
      } else if (t === 'callback:map') {
        newRule.map = {};
      } else if (t === 'callback:lowercase-wordlist') {
        newRule.pattern = '';
        newRule.wordlist = [];
      } else {
        newRule.pattern = '';
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
    wr.style.marginTop = '0.25rem';
    const l = document.createElement('div');
    l.className = 'help'; l.textContent = label + ':';
    wr.appendChild(l); wr.appendChild(el);
    return wr;
  }

  // --- promote: write local rules up into the committed config.json ----
  function _pipelineDirty() {
    try { return Object.prototype.hasOwnProperty.call(dirty, name); }
    catch (_) { return false; }
  }
  function _toast(msg) {
    const t = document.createElement('div');
    t.className = 'rule-toast';
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => { if (t.parentNode) t.parentNode.removeChild(t); }, 4500);
  }
  // Lightweight modal: {title, bodyEl, buttons:[{label, primary, onClick(close)}]}.
  function _modal(o) {
    const backdrop = document.createElement('div');
    backdrop.className = 'rule-modal-backdrop';
    const panel = document.createElement('div');
    panel.className = 'rule-modal';
    const h = document.createElement('div');
    h.className = 'rule-modal-title';
    h.textContent = o.title || '';
    panel.appendChild(h);
    if (o.bodyEl) panel.appendChild(o.bodyEl);
    const btnRow = document.createElement('div');
    btnRow.className = 'rule-modal-buttons';
    function close() {
      if (backdrop.parentNode) backdrop.parentNode.removeChild(backdrop);
    }
    (o.buttons || []).forEach(b => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = b.label;
      if (b.primary) btn.className = 'primary';
      btn.addEventListener('click', () => {
        if (b.onClick) b.onClick(close); else close();
      });
      btnRow.appendChild(btn);
    });
    panel.appendChild(btnRow);
    backdrop.appendChild(panel);
    backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });
    document.body.appendChild(backdrop);
    return close;
  }
  // Field-by-field diff between the rule being promoted and its config.json
  // copy (or a "NEW" note when the rule is not in config.json yet).
  function _ruleDiffEl(effRule, baseRule) {
    const box = document.createElement('div');
    box.className = 'promote-diff';
    if (!baseRule) {
      const p = document.createElement('div');
      p.className = 'help';
      p.textContent = 'NEW — this rule is not in config.json yet; it will be added.';
      box.appendChild(p);
      return box;
    }
    const keys = Array.from(new Set(
      Object.keys(effRule).concat(Object.keys(baseRule)))).sort();
    let any = false;
    keys.forEach(k => {
      if (k === 'seeded') return;
      const ov = JSON.stringify(baseRule[k]);
      const nv = JSON.stringify(effRule[k]);
      if (ov === nv) return;
      any = true;
      const rowEl = document.createElement('div');
      rowEl.className = 'promote-diff-row';
      const kEl = document.createElement('div');
      kEl.className = 'promote-diff-key'; kEl.textContent = k;
      const oEl = document.createElement('div');
      oEl.className = 'promote-diff-old'; oEl.textContent = ov;
      const nEl = document.createElement('div');
      nEl.className = 'promote-diff-new'; nEl.textContent = nv;
      rowEl.appendChild(kEl); rowEl.appendChild(oEl); rowEl.appendChild(nEl);
      box.appendChild(rowEl);
    });
    if (!any) {
      const p = document.createElement('div');
      p.className = 'help';
      p.textContent = 'No differences — already identical to config.json.';
      box.appendChild(p);
    }
    return box;
  }
  function _changeListEl(label, names) {
    const d = document.createElement('div');
    d.className = 'promote-change-line';
    d.textContent = names.length
      ? names.length + ' ' + label + ': ' + names.join(', ')
      : '0 ' + label;
    return d;
  }
  function _unsavedNoteEl() {
    if (!_pipelineDirty()) return null;
    const w = document.createElement('div');
    w.className = 'promote-unsaved-note';
    w.textContent = '⚠ You have unsaved local edits. Promote captures them into '
      + 'config.json now; also use the page Save to apply them on THIS deployment '
      + '(or clear the local override after a "promote all").';
    return w;
  }
  // POST the full intended config.json array; refresh factoryRules + repaint.
  async function _postFactory(arr) {
    let resp;
    try {
      resp = await api('POST', '/settings/factory-rules', { PIPELINE_RULES: arr });
    } catch (e) {
      alert('Could not reach the server.');
      return null;
    }
    if (resp.status === 422) {
      let msg = 'validation failed';
      try {
        const j = await resp.json();
        msg = (j.errors || []).map(x => x.msg || x.message || x).join('; ') || msg;
      } catch (_) {}
      alert('config.json was not saved:\n' + msg);
      return null;
    }
    if (!resp.ok) {
      alert('config.json save failed (HTTP ' + resp.status + ').');
      return null;
    }
    const out = await resp.json();
    if (Array.isArray(out.rules)) factoryRules = out.rules;
    paintAll();
    return out;
  }
  async function _fetchFactory() {
    const r = await api('GET', '/settings/factory-rules');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return (await r.json()).PIPELINE_RULES || [];
  }
  async function _promoteOne(rule) {
    if (!rule || rule.type === 'terminal') return;
    let fresh;
    try { fresh = await _fetchFactory(); }
    catch (e) { alert('Could not load config.json.'); return; }
    factoryRules = fresh;
    refreshControlsVisibility();
    const base = fresh.find(b => b.name === rule.name) || null;
    const body = document.createElement('div');
    const intro = document.createElement('div');
    intro.className = 'help';
    intro.textContent = base
      ? 'This writes the rule into the committed config.json (git-tracked).'
      : 'This adds the rule to the committed config.json (git-tracked).';
    body.appendChild(intro);
    body.appendChild(_ruleDiffEl(rule, base));
    const note = _unsavedNoteEl();
    if (note) body.appendChild(note);
    _modal({
      title: 'Promote “' + (rule.label || rule.name) + '” to config.json?',
      bodyEl: body,
      buttons: [
        { label: 'Cancel' },
        { label: 'Promote to config.json', primary: true, onClick: async (close) => {
            const next = fresh.slice();
            const i = next.findIndex(b => b.name === rule.name);
            const promoted = JSON.parse(JSON.stringify(rule));
            if (i >= 0) {
              next[i] = promoted;
            } else {
              const tIdx = next.findIndex(b => b.type === 'terminal');
              if (tIdx >= 0) next.splice(tIdx, 0, promoted);
              else next.push(promoted);
            }
            close();
            const out = await _postFactory(next);
            if (out) _toast('Promoted to config.json — commit & push to ship it.');
          } },
      ],
    });
  }
  async function _promoteAll() {
    let fresh;
    try { fresh = await _fetchFactory(); }
    catch (e) { alert('Could not load config.json.'); return; }
    factoryRules = fresh;
    refreshControlsVisibility();
    const factByName = new Map(fresh.map(b => [b.name, b]));
    const effByName = new Map(rules.map(r => [r.name, r]));
    const edited = [], added = [], removed = [];
    rules.forEach(r => {
      if (r.type === 'terminal') return;
      const b = factByName.get(r.name);
      if (!b) added.push(r.name);
      else if (!_ruleContentEqual(r, b)) edited.push(r.name);
    });
    fresh.forEach(b => {
      if (b.type !== 'terminal' && !effByName.has(b.name)) removed.push(b.name);
    });
    const body = document.createElement('div');
    const summary = document.createElement('div');
    summary.className = 'help';
    summary.textContent = 'This overwrites config.json with your current rule list:';
    body.appendChild(summary);
    body.appendChild(_changeListEl('edited', edited));
    body.appendChild(_changeListEl('added', added));
    body.appendChild(_changeListEl('removed from config.json', removed));
    const note = _unsavedNoteEl();
    if (note) body.appendChild(note);
    _modal({
      title: 'Promote all changes to config.json?',
      bodyEl: body,
      buttons: [
        { label: 'Cancel' },
        { label: 'Promote all', primary: true, onClick: async (close) => {
            close();
            const out = await _postFactory(JSON.parse(JSON.stringify(rules)));
            if (out) _afterPromoteAll(out);
          } },
      ],
    });
  }
  function _afterPromoteAll(out) {
    if (!out.shadowed_by_local) {
      _toast('Promoted all rules to config.json — commit & push to ship it.');
      return;
    }
    const body = document.createElement('div');
    const p = document.createElement('div');
    p.className = 'help';
    p.textContent = 'config.json now holds your rules. This deployment still has a '
      + 'local PIPELINE_RULES override that shadows config.json. Clear it so '
      + 'config.json runs directly here? config.json is committable either way.';
    body.appendChild(p);
    _modal({
      title: 'Promoted to config.json',
      bodyEl: body,
      buttons: [
        { label: 'Keep local override' },
        { label: 'Clear local override', primary: true, onClick: async (close) => {
            let r;
            try {
              r = await api('POST', '/settings/factory-rules/clear-local-override');
            } catch (e) { alert('Could not reach the server.'); return; }
            if (!r.ok) { alert('Clearing the local override failed.'); return; }
            close();
            location.reload();
          } },
      ],
    });
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
  presetWrap.innerHTML = '<span class="help" style="margin-right:0.375rem">preset:</span>';
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
  wrap.appendChild(sample);

  sel.addEventListener('change', () => {
    sample.value = TEST_PRESETS[sel.value] || '';
  });

  const runBtn = document.createElement('button');
  runBtn.type = 'button';
  runBtn.textContent = 'Run all enabled rules';
  runBtn.style.marginTop = '0.375rem';
  wrap.appendChild(runBtn);

  const out = document.createElement('div');
  out.className = 'regex-test-out';
  wrap.appendChild(out);

  async function run() {
    out.innerHTML = '<em>running…</em>';
    const r = await api('POST', '/settings/test-pipeline', {
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
  wrap.style.display = 'flex'; wrap.style.gap = '0.375rem'; wrap.style.alignItems = 'center';

  const i = document.createElement('input');
  i.type = 'number'; i.step = 'any';
  const btn = document.createElement('button');
  btn.style.padding = '0.125rem 0.5rem';

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
function render() {
  const main = $('main');
  main.innerHTML = '';
  for (const g of state.groups) {
    const sec = document.createElement('section');
    const h = document.createElement('h2');
    h.textContent = g.title;
    sec.appendChild(h);
    // Each group has subgroups; iterate them. A subgroup with title===null
    // emits no subheader. A subgroup WITH a title is wrapped in
    // <details>/<summary> so the admin can collapse long-tail "Advanced —"
    // sections; open/closed state persists in localStorage keyed by
    // `adv.global.{group}.{sub}`.
    for (const sub of g.subgroups) {
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
}

async function save() {
  if (Object.keys(dirty).length === 0) return;
  // Disable BEFORE the await so rapid double-clicks can't fire two
  // POSTs with the same dirty payload. setDirty() re-enables on edit.
  $('save-btn').disabled = true;
  const r = await api('POST', '/settings/state', dirty);
  if (r.status === 422) {
    const j = await r.json();
    const msg = (j.errors || [])
      .map(e => e.loc + ': ' + e.msg).join('  /  ');
    toast('validation: ' + msg, true);
    $('save-btn').disabled = false;
    return;
  }
  if (!r.ok) {
    toast('save failed: ' + r.status, true);
    $('save-btn').disabled = false;
    return;
  }
  const result = await r.json();
  dirty = {};
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

  // Capture the current process identity BEFORE asking it to restart, so
  // the polling loop can detect "different process now serving" even if
  // its polling interval missed the brief "service down" window. Falls
  // back to the old sawDown heuristic if the server lacks boot_id.
  let preBootId = null;
  try {
    const m0 = await fetch('/v1/models', { cache: 'no-store' });
    if (m0.ok) preBootId = (await m0.json()).boot_id || null;
  } catch {}

  const r = await api('POST', '/settings/restart', {});
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

  // Poll /v1/models. Two success criteria: (1) boot_id changed
  // (definitive — new process is up), or (2) we saw the service go down
  // and then come back. Deadline is generous because PRELOAD_MODELS
  // blocks uvicorn's lifespan startup.
  const RESTART_TIMEOUT_MS = 5 * 60 * 1000;
  const deadline = Date.now() + RESTART_TIMEOUT_MS;
  let sawDown = false;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 500));
    try {
      const m = await fetch('/v1/models', { cache: 'no-store' });
      if (m.ok) {
        let postBootId = null;
        try { postBootId = (await m.json()).boot_id || null; } catch {}
        const bootChanged = preBootId && postBootId && postBootId !== preBootId;
        if (bootChanged || sawDown) {
          $('restart-progress').classList.remove('show');
          toast('service is back; reloading');
          setTimeout(() => location.reload(), 600);
          return;
        }
      } else {
        sawDown = true;
      }
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
  // Single delegated 'admin:dirty' listener — refreshes one field's reset
  // visibility per dispatch. Replaces the per-fieldRow subscription that
  // leaked a listener on every render/reset cycle.
  document.addEventListener('admin:dirty', (e) => {
    if (e && e.detail && e.detail.name) refreshFieldReset(e.detail.name);
  });
  $('login-btn').addEventListener('click', async () => {
    const t = $('login-token').value.trim();
    if (!t) return;
    let r;
    try {
      r = await fetch('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: t }),
      });
    } catch (_) { showLogin('network error — try again'); return; }
    if (!r.ok) { showLogin('that key was rejected'); return; }
    $('login-token').value = '';
    window.dispatchEvent(new Event('whisper:auth-changed'));
    showApp();
    await loadState();
  });
  $('login-token').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') $('login-btn').click();
  });
  // #logout-btn + #reload-btn are wired globally in OPEN_MODE_BANNER_JS;
  // expose this page's soft refresh (re-fetch config) as the reload hook.
  window._pageReload = loadState;
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

  // Probe the state endpoint to figure out whether a sign-in is required.
  const probe = await fetch('/settings/state', { cache: 'no-store' });
  if (probe.status === 401) {
    showLogin('Sign in to continue.');
    return;
  }
  if (probe.status === 403) {
    // 403 with a presented bearer = signed-in but not admin. Render the
    // shared no-access landing. CRITICAL: the landing replaces <main>'s
    // innerHTML, but <main> lives inside `#app-wrap` which is hidden by
    // default — so we must call showApp() first or the landing renders
    // into a display:none subtree and the user sees a blank page.
    try {
      const wr = await fetch('/auth/whoami');
      if (wr.ok) {
        const wj = await wr.json();
        try { window.__whoami = wj; } catch (_) {}
        if (wj && wj.is_admin === false) {
          showApp();
          _renderNoAccessLanding({ page: 'config' });
          return;
        }
      }
    } catch (_) {}
  }
  if (!probe.ok) {
    document.body.innerHTML = '<main style="padding:1.25rem;color:#ff7b72">'
      + 'Could not load /settings/state (' + probe.status + '). '
      + 'Check service logs.</main>';
    return;
  }
  showApp();
  await loadState();
});

{{NOT_ADMIN_LANDING_JS}}

})();
</script>
{{SCALE_PICKER_JS}}
{{SEV_POLLER_JS}}
{{TAG_PICKER_JS}}
</body></html>"""
