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
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
        # `locks` and `requestable` are profile-level metadata, not per-field
        # overrides — rendered by dedicated controls, never in the field grid.
        if name in ("locks", "requestable"):
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
        # Read-only echo of the two global request gates so the page can show
        # whether requesting is enabled at all (they're edited on /settings).
        "globals": {
            "ALLOW_REQUEST_OVERRIDE_PROFILE":
                bool(getattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", True)),
            "ALLOW_REQUEST_DECODE_OVERRIDES":
                bool(getattr(cfg, "ALLOW_REQUEST_DECODE_OVERRIDES", True)),
        },
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


class _RenameProfileIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    old: str = Field(min_length=1)
    new: str = Field(min_length=1)


@router.post("/profiles/rename",
             dependencies=[Depends(require_admin_webui_host), Depends(require_admin)])
async def rename_profile(payload: _RenameProfileIn, request: Request) -> JSONResponse:
    """Rename an override profile (its dict key in OVERRIDE_PROFILES) and cascade
    the new name through every per-user / per-key binding that references it —
    the binding `profiles` list and `allowed_override_profiles` allowlist. The
    profile's overrides are preserved untouched.

    Unlike delete (which refuses an in-use profile and asks the admin to unbind
    first), rename FOLLOWS the references so in-use bindings keep resolving — the
    whole point of renaming is usually to retitle a profile that is already in
    use. The library's dict order is irrelevant (the page sorts for display), so
    the rename keeps the surviving overrides exactly and only swaps the key."""
    profiles = dict(getattr(cfg, "OVERRIDE_PROFILES", None) or {})
    old = payload.old.strip()
    new = payload.new.strip().lower()
    if old not in profiles:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown profile {old!r}")
    if new == old:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "new name is the same as the current name")
    if not config_store.TAG_RE.match(new):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "invalid profile name (a-z 0-9 -, max 32, must start "
                            "with a letter or digit)")
    if new in profiles:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"a profile named {new!r} already exists")

    # 1) Rename the key in OVERRIDE_PROFILES (preserve insertion order), then
    #    persist + hot-apply through the same path /state uses.
    renamed = {(new if k == old else k): v for k, v in profiles.items()}
    try:
        written = config_store.save_overrides({"OVERRIDE_PROFILES": renamed})
    except ValidationError as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"errors": config_store.format_validation_errors(e)},
        )
    except OSError as e:
        logger.error("[overrides] rename save failed: %s", e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"could not write config.local.json: {e}")

    # 2) Cascade the new name through all bindings (now that OVERRIDE_PROFILES
    #    holds the new key, the binding set stays referentially consistent).
    affected = api_keys_store.rename_profile_refs(old, new)

    import admin_routes
    applied = await admin_routes._apply_hot_changes(written)
    client_host = request.client.host if request.client else "?"
    logger.info("[overrides] profile renamed %r->%r from=%s bindings=%d",
                old, new, client_host, affected)
    return JSONResponse({
        "ok": True, "old": old, "new": new, "bindings_updated": affected,
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
            # Reflect the SAME gate the live decode path applies (main._apply_
            # decode_overrides / batch+streaming overrides_ignored key off
            # locked_client_keys): a field-level lock OR the per-identity decode
            # master gate being off (which locks every client key) → ignored.
            # Checking only r.locked would report "applied" for a key the server
            # actually drops whenever the master gate is off.
            client_sim = {
                "value": sim_dict[ck],
                "outcome": ("ignored_locked"
                            if (fname in r.locked or ck in r.locked_client_keys)
                            else "applied"),
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
    --red: #ff7b72; --bold: #f0f6fc; --border: #30363d; --border2: #3d444d;
    --input-bg: #0d1117; --help: #8b949e;
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
  /* shared main-area button system (mirrors /settings/api-keys) */
  main button { font: inherit; font-size: var(--fs-sm); cursor: pointer;
    line-height: 1.3; white-space: nowrap; border-radius: 6px; padding: 0.3rem 0.65rem;
    display: inline-flex; align-items: center; gap: 0.35rem; background: #21262d;
    color: var(--fg); border: 1px solid var(--border);
    transition: background 120ms ease, border-color 120ms ease, color 120ms ease; }
  main button:hover:not(:disabled) { background: #30363d; border-color: #484f58; }
  main button:disabled { opacity: 0.45; cursor: not-allowed; }
  main button.primary { background: #238636; border-color: #2ea043; color: var(--bold); }
  main button.primary:hover:not(:disabled) { background: #2ea043; }
  main button.ghost { background: transparent; color: var(--dim); border-color: transparent; }
  main button.ghost:hover:not(:disabled) { background: #1f2630; border-color: var(--border2); color: var(--fg); }
  main button.danger { background: transparent; color: var(--red); border-color: transparent; }
  main button.danger:hover:not(:disabled) { background: #3a0d0d; border-color: #5a2424; }
  main { padding: 1rem; max-width: 78rem; margin: 0 auto; }
  .hint { color: var(--help); font-size: var(--fs-sm); margin: 0.3rem 0 0; }
  .status { font-size: var(--fs-sm); color: var(--dim); margin-left: 0.5rem; }
  .status.ok { color: var(--green); } .status.err { color: var(--red); }
  /* contained intro / help banner (was bare floating text) */
  .ov-intro { border: 1px solid var(--border); border-radius: 9px;
    background: linear-gradient(180deg, #12171f, var(--panel));
    padding: 0.6rem 0.85rem; margin-bottom: 1rem; color: var(--help);
    font-size: var(--fs-sm); line-height: 1.55; }
  .ov-intro p { margin: 0; }
  .ov-intro #ov-globals { margin-top: 0.3rem; }
  .ov-intro .lk { width: 0.9em; height: 0.9em; vertical-align: -0.12em; color: var(--yellow); }
  /* subbar tabs */
  .ov-tab { background: transparent; border: 1px solid var(--border);
    color: var(--dim); padding: 0.2rem 0.7rem; border-radius: 999px;
    cursor: pointer; font: inherit; font-size: var(--fs-sm); }
  .ov-tab.active { color: var(--bold); border-color: var(--cyan); }
  /* master-detail */
  .ov-wrap { display: grid; grid-template-columns: 17rem 1fr; gap: 1rem; align-items: start; }
  .ov-side { display: flex; flex-direction: column; gap: 0.15rem;
    border: 1px solid var(--border); border-radius: 9px; padding: 0.5rem;
    background: var(--panel); align-self: start; }
  .ov-side-h { display: flex; align-items: center; justify-content: space-between;
    padding: 0.1rem 0.35rem 0.4rem; color: var(--dim); font-size: var(--fs-xs);
    text-transform: uppercase; letter-spacing: 0.04em; }
  .ov-item { display: flex; align-items: center; gap: 0.5rem; cursor: pointer;
    padding: 0.4rem 0.5rem; border-radius: 7px; border: 1px solid transparent; }
  .ov-item:hover { background: #1c2230; }
  .ov-item.active { border-color: var(--border2); background: #1c2230; }
  .ov-item .nm { flex: 1; font-family: var(--font-mono);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .ov-item.active .nm { color: var(--bold); }
  .ov-dot { width: 0.6rem; flex: 0 0 auto; }
  .ov-dot.on { color: var(--green); } .ov-dot.off { color: var(--dim); }
  .ov-count { font-size: var(--fs-xs); color: var(--dim);
    font-family: var(--font-mono); }
  .ov-usage { font-size: var(--fs-xs); color: var(--dim);
    padding: 0.25rem 0.55rem 0.15rem; margin: 0.1rem 0 0.3rem;
    border-left: 2px solid var(--border2); }
  .ov-side-actions { display: flex; gap: 0.2rem; flex-wrap: wrap;
    padding-top: 0.5rem; margin-top: 0.3rem; border-top: 1px solid var(--border); }
  .ov-side-actions button { padding: 0.25rem 0.5rem; font-size: var(--fs-xs); }
  .ov-confirm { display: inline-flex; align-items: center; gap: 0.3rem;
    font-size: var(--fs-xs); color: var(--red); }
  /* inline add / rename (replaces prompt) */
  .ov-inline { display: flex; align-items: center; padding: 0.3rem 0.1rem 0; }
  .ov-inline-edit { display: flex; align-items: center; gap: 0.25rem; flex: 1; min-width: 0; }
  .ov-edit-input { flex: 1; min-width: 4rem; box-sizing: border-box;
    background: var(--input-bg); color: var(--bold); border: 1px solid var(--cyan);
    border-radius: 6px; padding: 0.25rem 0.45rem; font: inherit;
    font-size: var(--fs-sm); font-family: var(--font-mono); }
  .icon-btn { background: none; border: 1px solid transparent; border-radius: 6px;
    padding: 0.2rem 0.3rem; color: var(--dim); display: inline-flex; align-items: center;
    cursor: pointer; font-size: var(--fs-sm); line-height: 1; }
  .icon-btn:hover { background: #1f2630; color: var(--fg); }
  .icon-btn.ok { color: var(--green); }
  .ov-main { padding: 0; min-height: 8rem; }
  .ov-empty { color: var(--dim); padding: 2rem 1rem; text-align: center;
    border: 1px solid var(--border); border-radius: 9px; background: var(--panel); }
  .ov-pane-head { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.85rem; }
  .ov-pane-head .pname { font-family: var(--font-mono); color: var(--bold);
    font-size: var(--fs-lg); }
  /* section cards */
  .ov-sec { border: 1px solid var(--border); border-radius: 9px;
    background: var(--panel); overflow: hidden; margin-bottom: 0.9rem; }
  .ov-sec:last-child { margin-bottom: 0; }
  .ov-sec > h4 { margin: 0; padding: 0.5rem 0.8rem; background: #12171f;
    border-bottom: 1px solid var(--border); color: var(--bold);
    font-size: var(--fs-md); font-weight: 600; display: flex;
    align-items: baseline; gap: 0.5rem; }
  .ov-sec-sub { color: var(--dim); font-size: var(--fs-xs); font-weight: 400; }
  .ov-sec-b { padding: 0.2rem 0.8rem 0.45rem; }
  .ov-help { color: var(--help); font-size: var(--fs-xs); line-height: 1.5;
    margin-top: 0.15rem; }
  .ov-access-row { display: flex; align-items: center; gap: 0.6rem; padding: 0.35rem 0 0.2rem; }
  .ov-access-row label { font-size: var(--fs-sm); color: var(--fg); cursor: pointer; }
  /* field row: dot | name | value | lock | ctrl — hairline dividers + hover */
  .ov-row { display: grid;
    grid-template-columns: 0.8rem minmax(11rem, 1fr) minmax(8rem, 1.4fr) 2rem 5.2rem;
    align-items: center; gap: 0.4rem; padding: 0.32rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.055); }
  .ov-row:hover { background: rgba(255,255,255,0.025); }
  .ov-sec-b > .ov-row:last-child, .ov-sec-b > .ov-rule:last-child { border-bottom: 0; }
  .ov-row .ov-name { font-family: var(--font-mono); font-size: var(--fs-sm);
    color: var(--dim); overflow: hidden; text-overflow: ellipsis; }
  .ov-row.is-set .ov-name { color: var(--bold); }
  .ov-val input[type=text], .ov-val input[type=number], .ov-val select {
    width: 100%; box-sizing: border-box; background: var(--input-bg);
    color: var(--fg); border: 1px solid var(--border); border-radius: 6px;
    padding: 0.2rem 0.4rem; font-family: var(--font-mono);
    font-size: var(--fs-sm); }
  .ov-inherits { color: var(--dim); font-size: var(--fs-xs);
    font-family: var(--font-mono); font-style: italic; }
  .ov-lock { background: none; border: 1px solid transparent; border-radius: 6px;
    cursor: pointer; color: var(--dim); padding: 0.15rem; font-size: 1rem;
    display: inline-flex; align-items: center; justify-content: center; }
  .ov-lock svg { width: 1.1em; height: 1.1em; display: block; }
  .ov-lock:hover:not(:disabled) { background: #1f2630; }
  .ov-lock.locked { color: var(--yellow); }
  .ov-lock:disabled { opacity: 0.3; cursor: default; }
  .ov-ctrl { justify-self: end; }
  .ov-ctrl button { background: none; border: 0; cursor: pointer;
    font-size: var(--fs-xs); color: var(--cyan); padding: 0.15rem 0.2rem; border-radius: 0; }
  .ov-ctrl .reset { color: var(--dim); }
  /* pipeline tri-state — label left, shared segmented control (inherit/on/off)
     right; .status-btn-group styling comes from web_common.NAV_CSS. */
  .ov-rule { display: grid; grid-template-columns: 1fr auto; align-items: center;
    gap: 0.4rem; padding: 0.32rem 0; border-bottom: 1px solid rgba(255,255,255,0.055); }
  .ov-rule:hover { background: rgba(255,255,255,0.025); }
  .ov-rule .rl { font-size: var(--fs-sm); }
  .ov-rule .rl .slug { font-family: var(--font-mono); color: var(--dim);
    font-size: var(--fs-xs); }
  /* explorer — two labelled zones: Identity (who) + What-if (simulate) */
  .ex-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
  .ex-card { border: 1px solid var(--border); border-radius: 9px; background: var(--panel); }
  .ex-card.identity { border-left: 3px solid rgba(121,192,255,0.7);
    background: linear-gradient(90deg, rgba(121,192,255,0.08), rgba(121,192,255,0.015) 55%), var(--panel); }
  .ex-card.simulate { border-left: 3px solid rgba(210,153,34,0.75);
    background: linear-gradient(90deg, rgba(210,153,34,0.08), rgba(210,153,34,0.015) 55%), var(--panel); }
  .ex-card .eh { display: flex; align-items: center; gap: 0.5rem;
    padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--border); }
  .ex-card .eh .lab { font-size: var(--fs-xs); text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--dim); }
  .ex-card .eh .lab b { color: var(--bold); }
  .badge-sim { font-size: var(--fs-xs); border: 1px dashed rgba(210,153,34,0.6);
    color: var(--yellow); border-radius: 999px; padding: 0.02rem 0.45rem;
    text-transform: uppercase; letter-spacing: 0.05em; margin-left: auto; }
  .ex-card .eb { padding: 0.6rem 0.8rem 0.7rem; display: flex;
    flex-direction: column; gap: 0.55rem; }
  .ex-field { display: flex; flex-direction: column; gap: 0.2rem; }
  .ex-field label { font-size: var(--fs-xs); color: var(--dim); }
  .ex-field select, .ex-field input { width: 100%; box-sizing: border-box;
    background: var(--input-bg); color: var(--fg); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.28rem 0.45rem; font: inherit; font-size: var(--fs-sm); }
  .ex-field input.ov-sim { font-family: var(--font-mono); }
  .ex-field .fhelp { color: var(--help); font-size: var(--fs-xs); line-height: 1.4; }
  .ex-row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.55rem; }
  .ov-sim.err { border-color: var(--red); }
  @media (max-width: 56rem) { .ex-grid { grid-template-columns: 1fr; } }
  /* waterfall */
  .ov-wf { border: 1px solid var(--border); border-radius: 4px;
    margin: 0.4rem 0; background: var(--bg); }
  .ov-wf-head { display: flex; align-items: baseline; gap: 0.5rem;
    padding: 0.3rem 0.5rem; border-bottom: 1px solid var(--border);
    flex-wrap: wrap; }
  .ov-wf-name { font-family: var(--font-mono); color: var(--bold); }
  .ov-wf-win { font-size: var(--fs-sm); color: var(--dim); }
  .ov-wf-win code { color: var(--green); }
  .ov-wf-lock { color: var(--yellow); font-size: var(--fs-xs);
    display: inline-flex; align-items: center; gap: 0.25rem; }
  .ov-wf-layers { list-style: none; margin: 0; padding: 0.2rem 0.3rem; }
  .ov-wf-layer { display: grid;
    grid-template-columns: 1.1rem 14rem minmax(0, 1fr) auto;
    align-items: baseline; gap: 0.5rem; padding: 0.16rem 0.3rem;
    border-bottom: 1px solid rgba(255,255,255,0.05); }
  .ov-wf-layer:last-child { border-bottom: 0; }
  .ov-wf-layer .lbl { font-size: var(--fs-sm); overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
  .ov-wf-layer .val { font-family: var(--font-mono); font-size: var(--fs-sm);
    text-align: left; word-break: break-word; }
  .ov-wf-layer .flag { font-size: var(--fs-xs); text-align: right; white-space: nowrap; }
  .ov-wf-layer.win { box-shadow: inset 3px 0 0 var(--green); background: #11271a; }
  .ov-wf-layer.win .lbl { color: var(--bold); } .ov-wf-layer.win .val { color: var(--green); }
  .ov-wf-layer.over .val { opacity: 0.5; text-decoration: line-through; }
  .ov-wf-layer.over .lbl { opacity: 0.6; }
  .ov-wf-layer.unset .val { color: var(--dim); font-style: italic; }
  .ov-wf-layer.sim .lbl { color: var(--cyan); }
  .ov-wf-layer.sim.ignored .flag { color: var(--red); }
  .ov-wf-layer .flag.locked { color: var(--yellow); display: inline-flex;
    align-items: center; gap: 0.2rem; }
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
    <div class="ov-intro">
      <p>Reusable config profiles. Assign them (ordered, earlier wins) to users
        &amp; API keys on the <a href="/settings/api-keys">API keys</a> page. A
        <svg class="lk" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor"
          stroke-width="1.6" stroke-linejoin="round" aria-hidden="true"><rect x="5"
          y="11" width="14" height="9.5" rx="2"/><path d="M8 11V7.5a4 4 0 0 1 8 0V11"
          fill="none"/></svg>
        field cannot be overridden per-request by the client.</p>
      <div id="ov-globals"></div>
    </div>
    <div class="ov-wrap">
      <div class="ov-side" id="profile-list"></div>
      <div class="ov-main" id="profile-main"></div>
    </div>
  </section>
  <section id="panel-explorer" hidden>
    <div class="ov-intro">
      <p>What-if: pick who is resolving and (optionally) a hypothetical client
        override to see how every setting resolves &mdash; which layer wins, what
        is overridden, and what is locked.</p>
    </div>
    <div class="ex-grid">
      <div class="ex-card identity">
        <div class="eh"><span class="lab"><b>Identity</b> &mdash; who is resolving</span></div>
        <div class="eb">
          <div class="ex-field"><label for="ex-user">User</label>
            <select id="ex-user"></select></div>
          <div class="ex-row2">
            <div class="ex-field"><label for="ex-key">API key</label>
              <select id="ex-key"></select></div>
            <div class="ex-field"><label for="ex-model">Model</label>
              <select id="ex-model"></select></div>
          </div>
        </div>
      </div>
      <div class="ex-card simulate">
        <div class="eh"><span class="lab"><b>What-if</b> &mdash; simulate a client request</span>
          <span class="badge-sim">simulated</span></div>
        <div class="eb">
          <div class="ex-field"><label for="ex-sim">Client-supplied decode override (JSON)</label>
            <input id="ex-sim" class="ov-sim" placeholder='{"beam_size": 8}'>
            <span class="fhelp">Hypothetical value a client sends with the request.
              <b>What-if only &mdash; nothing is saved.</b> It just changes what this
              Explorer resolves below.</span>
          </div>
        </div>
      </div>
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
  // Solid padlock at currentColor — same iconography as the field-row locks.
  var LOCK_SVG = '<svg viewBox="0 0 24 24" fill="currentColor" stroke="currentColor"'
    + ' stroke-width="1.6" stroke-linejoin="round" aria-hidden="true"'
    + ' style="width:.9em;height:.9em;vertical-align:-.12em">'
    + '<rect x="5" y="11" width="14" height="9.5" rx="2"/>'
    + '<path d="M8 11V7.5a4 4 0 0 1 8 0V11" fill="none"/></svg>';
  return function (name, fr) {
    var el = document.createElement('div');
    el.className = 'ov-wf';
    var locked = fr.locked
      ? ' <span class="ov-wf-lock" title="locked — client cannot override">' + LOCK_SVG + ' locked</span>'
      : '';
    var rows = (fr.layers || []).map(function (h) {
      var cls = 'ov-wf-layer', label = '';
      if (h.is_winner) { cls += ' win'; label = 'winner'; }
      else if (h.is_set) { cls += ' over'; label = 'overridden'; }
      else cls += ' unset';
      var flag = h.locked
        ? '<span class="flag locked">' + LOCK_SVG + ' locked</span>'
        : '<span class="flag">' + label + '</span>';
      return '<li class="' + cls + '">'
        + '<span class="ov-wf-tick">' + (h.is_winner ? '✓' : '') + '</span>'
        + '<span class="lbl">' + esc(h.label) + '</span>'
        + '<span class="val">' + esc(h.is_set ? fmt(h.value) : 'not set') + '</span>'
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
  var NAME_RE = /^[a-z0-9][a-z0-9-]{0,31}$/;
  // inline-UI flags (replace the old prompt/confirm/alert dialogs)
  var uiAdding = false;         // sidebar "new profile" input showing
  var uiRenaming = false;       // selected profile in inline-rename mode
  var uiConfirmDel = false;     // delete confirmation showing

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
      if (k === 'locks' || k === 'requestable') return;
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
  // Inline editor (input + ✓ / ✕) for both create and rename, replacing the old
  // browser prompt(). Enter commits, Esc cancels; the field auto-focuses.
  function inlineInput(value, onCommit, onCancel) {
    var wrap = document.createElement('span'); wrap.className = 'ov-inline-edit';
    var inp = document.createElement('input'); inp.type = 'text';
    inp.className = 'ov-edit-input'; inp.value = value || ''; inp.maxLength = 32;
    inp.setAttribute('aria-label', 'profile name');
    var ok = document.createElement('button'); ok.type = 'button';
    ok.className = 'icon-btn ok'; ok.innerHTML = '✓'; ok.title = 'confirm (Enter)';
    var no = document.createElement('button'); no.type = 'button';
    no.className = 'icon-btn'; no.innerHTML = '✕'; no.title = 'cancel (Esc)';
    function commit() { onCommit((inp.value || '').trim().toLowerCase()); }
    ok.onclick = commit;
    no.onclick = function () { onCancel(); };
    inp.onkeydown = function (e) {
      if (e.key === 'Enter') { e.preventDefault(); commit(); }
      else if (e.key === 'Escape') { e.preventDefault(); onCancel(); }
    };
    wrap.appendChild(inp); wrap.appendChild(ok); wrap.appendChild(no);
    setTimeout(function () { inp.focus(); inp.select(); }, 0);
    return wrap;
  }

  function renderSide() {
    listEl.innerHTML = '';
    var names = profileNames();
    var hdr = document.createElement('div'); hdr.className = 'ov-side-h';
    hdr.innerHTML = '<span>Profiles</span><span class="ov-count">' + names.length + '</span>';
    listEl.appendChild(hdr);
    names.forEach(function (name) {
      var p = profiles[name];
      var isSel = name === sel;
      var cnt = overrideCount(p);
      var item = document.createElement('div');
      item.className = 'ov-item' + (isSel ? ' active' : '');
      if (isSel && uiRenaming) {
        item.innerHTML = '<span class="ov-dot ' + (cnt ? 'on' : 'off') + '">'
          + (cnt ? '●' : '○') + '</span>';
        item.appendChild(inlineInput(name,
          function (nn) { commitRename(name, nn); },
          function () { uiRenaming = false; setStatus(''); render(); }));
        listEl.appendChild(item);
        return;
      }
      item.innerHTML = '<span class="ov-dot ' + (cnt ? 'on' : 'off') + '">'
        + (cnt ? '●' : '○') + '</span>'
        + '<span class="nm">' + esc(name) + '</span>'
        + '<span class="ov-count">' + cnt + '</span>';
      item.onclick = function () {
        sel = name; uiRenaming = false; uiConfirmDel = false; setStatus(''); render();
      };
      listEl.appendChild(item);
      if (isSel) {
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
    if (uiAdding) {
      var addWrap = document.createElement('div'); addWrap.className = 'ov-inline';
      addWrap.appendChild(inlineInput('', commitNew,
        function () { uiAdding = false; setStatus(''); render(); }));
      listEl.appendChild(addWrap);
    }
    var acts = document.createElement('div');
    acts.className = 'ov-side-actions';
    var nb = document.createElement('button'); nb.className = 'ghost'; nb.textContent = '+ New';
    nb.onclick = function () {
      uiAdding = true; uiRenaming = false; uiConfirmDel = false; setStatus(''); render();
    };
    acts.appendChild(nb);
    if (sel) {
      var rb = document.createElement('button'); rb.className = 'ghost'; rb.textContent = '✎ Rename';
      rb.onclick = startRename; acts.appendChild(rb);
      var db = document.createElement('button'); db.className = 'ghost'; db.textContent = '⧉ Duplicate';
      db.onclick = duplicateProfile; acts.appendChild(db);
      if (uiConfirmDel) {
        var cf = document.createElement('span'); cf.className = 'ov-confirm';
        cf.appendChild(document.createTextNode('Delete?'));
        var yes = document.createElement('button'); yes.className = 'danger'; yes.textContent = '✓ Yes';
        yes.onclick = confirmDelete; cf.appendChild(yes);
        var dno = document.createElement('button'); dno.className = 'ghost'; dno.textContent = '✕';
        dno.onclick = function () { uiConfirmDel = false; render(); };
        cf.appendChild(dno); acts.appendChild(cf);
      } else {
        var xb = document.createElement('button'); xb.className = 'danger'; xb.textContent = '✕ Delete';
        xb.onclick = startDelete; acts.appendChild(xb);
      }
    }
    listEl.appendChild(acts);
  }

  function commitNew(name) {
    if (!name) { uiAdding = false; render(); return; }
    if (!NAME_RE.test(name)) { setStatus('invalid name — use a-z 0-9 - (max 32)', 'err'); return; }
    if (profiles[name]) { setStatus('profile "' + name + '" already exists', 'err'); return; }
    uiAdding = false; profiles[name] = {}; sel = name; setStatus(''); render(); refreshButtons();
  }
  function duplicateProfile() {
    if (!sel) return;
    var base = sel + '-copy'; var name = base; var i = 2;
    while (profiles[name]) { name = base + i; i++; }
    profiles[name] = JSON.parse(JSON.stringify(profiles[sel]));
    sel = name; uiRenaming = false; uiConfirmDel = false; render(); refreshButtons();
  }
  function startRename() {
    if (!sel) return;
    // A saved profile renames on the server (which reloads), so a dirty working
    // copy must be saved/discarded first or those edits would be dropped.
    if ((S.profiles && S.profiles[sel]) && dirty()) {
      setStatus('Save or discard your changes before renaming "' + sel + '".', 'err');
      return;
    }
    uiRenaming = true; uiAdding = false; uiConfirmDel = false; setStatus(''); render();
  }
  function commitRename(cur, nn) {
    if (!nn || nn === cur) { uiRenaming = false; setStatus(''); render(); return; }
    if (!NAME_RE.test(nn)) { setStatus('invalid name — use a-z 0-9 - (max 32)', 'err'); return; }
    if (profiles[nn]) { setStatus('a profile named "' + nn + '" already exists', 'err'); return; }
    uiRenaming = false;
    // Never-saved profile → no binding references it; move the key and let the
    // rename ride along on the next Save like any other edit.
    if (!(S.profiles && S.profiles[cur])) {
      profiles[nn] = profiles[cur]; delete profiles[cur];
      sel = nn; setStatus(''); render(); refreshButtons();
      return;
    }
    doRename(cur, nn);   // saved → server cascade (dirty already guarded in startRename)
  }
  async function doRename(oldName, newName) {
    setStatus('renaming…');
    var r = await api('POST', '/settings/overrides/profiles/rename',
                      { old: oldName, new: newName });
    if (await guard403(r)) return;
    if (r.status === 409) {
      setStatus('a profile named "' + newName + '" already exists', 'err'); return;
    }
    if (!r.ok) { setStatus('rename failed (' + r.status + ')', 'err'); return; }
    var j = await r.json();
    await loadState(true);                  // canonical reload (profiles + usage)
    sel = profiles[newName] ? newName : (profileNames()[0] || null);
    render(); refreshButtons();
    var n = (j && j.bindings_updated) || 0;
    setStatus('renamed' + (n ? ' · ' + n + ' binding' + (n === 1 ? '' : 's')
                                  + ' updated' : ''), 'ok');
  }
  function startDelete() {
    if (!sel) return;
    var u = (S.usage || {})[sel] || { users: [], keys: [] };
    if (u.users.length || u.keys.length) {
      setStatus('"' + sel + '" is in use by ' + u.users.length + ' user(s) / '
        + u.keys.length + ' key(s) — unbind it on the API keys page first.', 'err');
      return;
    }
    uiConfirmDel = true; uiRenaming = false; uiAdding = false; setStatus(''); render();
  }
  function confirmDelete() {
    if (!sel) return;
    delete profiles[sel]; sel = null; uiConfirmDel = false; render(); refreshButtons();
  }

  // ---- global gates (read-only echo; edited on /settings) ----
  function renderGlobals() {
    var el = document.getElementById('ov-globals');
    if (!el) return;
    var g = (S && S.globals) || {};
    function badge(on) {
      return '<b style="color:' + (on ? 'var(--green)' : 'var(--red)') + '">'
        + (on ? 'on' : 'off') + '</b>';
    }
    el.innerHTML =
      'Request gates (edited on <a href="/settings">Settings</a>): '
      + 'override-profile ' + badge(g.ALLOW_REQUEST_OVERRIDE_PROFILE !== false)
      + ' · decode-overrides ' + badge(g.ALLOW_REQUEST_DECODE_OVERRIDES !== false)
      + '. Per-user / per-key gates &amp; allowlists are set in the '
      + '&#9881; overrides drawer on the <a href="/settings/api-keys">API keys</a> page.';
  }

  // ---- main pane ----
  function render() { renderGlobals(); renderSide(); renderMain(); }

  // A titled section card (header strip + padded body). Returns { sec, body }.
  function makeSection(title, sub) {
    var sec = document.createElement('div'); sec.className = 'ov-sec';
    sec.innerHTML = '<h4>' + esc(title)
      + (sub ? ' <span class="ov-sec-sub">' + esc(sub) + '</span>' : '') + '</h4>';
    var body = document.createElement('div'); body.className = 'ov-sec-b';
    sec.appendChild(body);
    return { sec: sec, body: body };
  }

  function renderMain() {
    mainEl.innerHTML = '';
    if (!sel) {
      mainEl.innerHTML = '<div class="ov-empty">Select a profile, or create one with <b>+ New</b>.</div>';
      return;
    }
    var p = profiles[sel];
    var cnt = overrideCount(p);
    var head = document.createElement('div'); head.className = 'ov-pane-head';
    head.innerHTML = '<span class="ov-dot ' + (cnt ? 'on' : 'off') + '">'
      + (cnt ? '●' : '○') + '</span><span class="pname">' + esc(sel) + '</span>';
    mainEl.appendChild(head);
    mainEl.appendChild(accessSection(p));
    (S.groups || []).forEach(function (g) {
      var fields = [];
      (g.subgroups || []).forEach(function (sg) {
        (sg.fields || []).forEach(function (f) { fields.push(f); });
      });
      if (!fields.length) return;
      var s = makeSection(g.title);
      fields.forEach(function (f) { s.body.appendChild(fieldRow(p, f)); });
      mainEl.appendChild(s.sec);
    });
    mainEl.appendChild(pipelineSection(p));
  }

  // Profile-level "Access" — whether clients may NAME this profile per request.
  // Default = requestable; storing requestable:false opts it out globally.
  function accessSection(p) {
    var s = makeSection('Access', 'who may apply this profile');
    var row = document.createElement('div'); row.className = 'ov-access-row';
    var cb = document.createElement('input');
    cb.type = 'checkbox'; cb.className = 'switch'; cb.setAttribute('role', 'switch');
    cb.id = 'ov-requestable';
    cb.checked = p.requestable !== false;       // absent / true = requestable
    cb.onchange = function () {
      if (cb.checked) delete p.requestable; else p.requestable = false;
      refreshButtons(); renderSide();
    };
    var lbl = document.createElement('label');
    lbl.setAttribute('for', 'ov-requestable');
    lbl.textContent = 'Requestable by clients';
    row.appendChild(cb); row.appendChild(lbl);
    s.body.appendChild(row);
    var help = document.createElement('div'); help.className = 'ov-help';
    help.innerHTML = 'Clients may name this profile in a per-request '
      + '<code>override_profile</code> (subject to per-user / per-key allowlists). '
      + 'Off = admin-applied only.';
    s.body.appendChild(help);
    return s.sec;
  }

  function isSet(p, f) { return p[f] !== null && p[f] !== undefined; }

  // Line-art padlock at currentColor (matches the key icon on /api-keys).
  // Locked = solid body + closed shackle; unlocked = outline body + open shackle
  // — distinct on shape AND fill AND colour, not colour alone (WCAG 1.4.1).
  function lockSvg(locked) {
    return locked
      ? '<svg viewBox="0 0 24 24" fill="currentColor" stroke="currentColor"'
        + ' stroke-width="1.6" stroke-linejoin="round" aria-hidden="true">'
        + '<rect x="5" y="11" width="14" height="9.5" rx="2"/>'
        + '<path d="M8 11V7.5a4 4 0 0 1 8 0V11" fill="none"/></svg>'
      : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"'
        + ' stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" aria-hidden="true">'
        + '<rect x="5" y="11" width="14" height="9.5" rx="2"/>'
        + '<path d="M8 11V7.5a4 4 0 0 1 7-2.4"/></svg>';
  }

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
    lockCell.innerHTML = lockSvg(locked);
    if (locked) lockCell.classList.add('locked');
    lockCell.setAttribute('role', 'switch');
    lockCell.setAttribute('aria-checked', locked ? 'true' : 'false');
    // Available even when the field is unset: a value-less lock pins the
    // inherited (per-model/global) value and still blocks client overrides.
    var lockWhat = isSet(p, name) ? 'this field' : 'the inherited value';
    var lockTitle = locked
      ? 'Locked — clients cannot override ' + lockWhat + ' per request (click to unlock)'
      : 'Unlocked — clients may override ' + lockWhat + ' per request (click to lock)';
    lockCell.title = lockTitle;
    lockCell.setAttribute('aria-label', lockTitle);
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
    var s = makeSection('Pipeline rules', 'enable / disable text rules for this profile');
    var exc = p.PIPELINE_RULES_EXCLUDE || [];
    var inc = p.PIPELINE_RULES_INCLUDE || [];
    (S.rules || []).forEach(function (r) {
      var row = document.createElement('div'); row.className = 'ov-rule';
      var state = exc.indexOf(r.name) >= 0 ? 'off' : (inc.indexOf(r.name) >= 0 ? 'on' : 'inherit');
      row.innerHTML = '<span class="rl">' + esc(r.label)
        + ' <span class="slug">' + esc(r.name) + (r.enabled ? '' : ' (off)') + '</span></span>';
      var grp = document.createElement('span'); grp.className = 'status-btn-group';
      grp.setAttribute('role', 'radiogroup');
      // inherit label carries the resolved global default, as the old select did
      [['inherit', 'Inherit (' + (r.enabled ? 'on' : 'off') + ')', 'inherit'],
       ['on', 'On', 'allow'], ['off', 'Off', 'deny']].forEach(function (o) {
        var btn = document.createElement('button'); btn.type = 'button';
        btn.className = 'status-btn' + (o[0] === state ? ' active' : '');
        btn.dataset.val = o[0]; btn.dataset.tone = o[2]; btn.textContent = o[1];
        btn.setAttribute('role', 'radio');
        btn.setAttribute('aria-checked', o[0] === state ? 'true' : 'false');
        btn.onclick = function () {
          if (btn.classList.contains('active')) return;
          grp.querySelectorAll('.status-btn').forEach(function (x) {
            x.classList.remove('active'); x.setAttribute('aria-checked', 'false');
          });
          btn.classList.add('active'); btn.setAttribute('aria-checked', 'true');
          setRule(p, r.name, o[0]);
        };
        grp.appendChild(btn);
      });
      row.appendChild(grp); s.body.appendChild(row);
    });
    if (!(S.rules || []).length) {
      var none = document.createElement('p'); none.className = 'ov-help';
      none.textContent = 'No pipeline rules configured.';
      s.body.appendChild(none);
    }
    return s.sec;
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
