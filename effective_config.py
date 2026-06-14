"""
Layered per-identity config resolution.

Resolves the *effective* value of every overridable decode / streaming /
pipeline setting for a given caller, walking the precedence stack

    per-key direct → per-key profiles → per-user direct → per-user profiles
    → per-model → global → (faster-whisper builtin)

The FIRST identity layer that sets a scalar field owns BOTH its value and its
lock state. A locked field cannot be replaced by the client's per-request
``decode_overrides``. Pipeline-rule enable/disable resolves analogously (first
layer mentioning a slug in include/exclude decides; exclude wins within a
layer), folding the per-model layer in too — locking does not apply to rules.

This is a LEAF module: it imports only ``config``, ``config_store`` and
``api_keys_store`` and is itself imported by ``main`` / ``streaming_routes`` /
``captures_*``. It never imports ``main`` (no cycles). The runtime reader
``main.cfg_for(model_id, field, ident)`` consults a resolved object's
``values`` first, so threading ``ident=None`` anywhere is byte-identical to the
pre-feature behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import config as cfg
import config_store

# Sentinel: "this layer does not set this field" (distinct from a legitimate
# None override such as SUPPRESS_TOKENS=None meaning "clear the suppression").
UNSET = object()

# Overridable scalar fields = every call-time + streaming field EXCEPT the
# pipeline include/exclude lists (those resolve via the rule path, not here).
# Single-sourced from the schema so it cannot drift.
SCALAR_OVERRIDE_FIELDS: frozenset[str] = config_store.LOCKABLE_FIELDS

# Map an UPPER_CASE config field → the lowercase client decode_override key it
# governs. Mirrors the allow-list enforced by main._apply_decode_overrides, so
# a lock on the config field blocks the matching client key. Fields absent here
# are not client-overridable, so a lock on them is a no-op for the client gate.
_CONFIG_TO_CLIENT_KEY: dict[str, str] = {
    "BEAM_SIZE": "beam_size",
    "BEST_OF": "best_of",
    "NO_REPEAT_NGRAM_SIZE": "no_repeat_ngram_size",
    "TEMPERATURE": "temperature",
    "NO_SPEECH_THRESHOLD": "no_speech_threshold",
    "LOG_PROB_THRESHOLD": "log_prob_threshold",
    "COMPRESSION_RATIO_THRESHOLD": "compression_ratio_threshold",
    "PATIENCE": "patience",
    "LENGTH_PENALTY": "length_penalty",
    "REPETITION_PENALTY": "repetition_penalty",
    "DEFAULT_HOTWORDS": "hotwords",
    "PREPEND_PUNCTUATIONS": "prepend_punctuations",
    "APPEND_PUNCTUATIONS": "append_punctuations",
    "SUPPRESS_TOKENS": "suppress_tokens",
    "CONDITION_ON_PREVIOUS_TEXT": "condition_on_previous_text",
    "VAD_FILTER": "vad_filter",
    "VAD_MIN_SILENCE_MS": "vad_min_silence_duration_ms",
    "VAD_SPEECH_PAD_MS": "vad_speech_pad_ms",
    "VAD_THRESHOLD": "vad_threshold",
}

# Identity carriers that have no per-identity config: the synthetic open-mode
# admin and the cookie-session pseudo key id. Both resolve to "no layer."
_SENTINEL_IDS: frozenset[str] = frozenset({"(open-mode)", "(session)"})


@dataclass
class Resolved:
    """The effective per-identity config for one request / handshake / row.

    Threaded into ``main.cfg_for`` / ``assemble_transcribe_kwargs`` /
    ``_apply_decode_overrides`` / ``_postprocess_text`` and the streaming
    builders. Computed ONCE per request and reused across all partial + final
    decodes of a stream.
    """
    # Scalar fields an identity layer set (config-field-name → value). cfg_for
    # consults this first; absent → falls through to per-model → global.
    values: dict[str, Any] = field(default_factory=dict)
    # Fields locked by their owning identity layer (config-field names).
    locked: set[str] = field(default_factory=set)
    # The lowercase client decode_override keys blocked by `locked`.
    locked_client_keys: frozenset[str] = frozenset()
    # Pipeline rules force-enabled / force-disabled (per-model already folded).
    pipeline_include: set[str] = field(default_factory=set)
    pipeline_exclude: set[str] = field(default_factory=set)
    # Client override keys that WERE present in this request but dropped because
    # the field is locked — surfaced to the caller (never silently swallowed).
    dropped: list[str] = field(default_factory=list)
    # Ordered profile names applied (most → least specific), for observability.
    profiles_applied: list[str] = field(default_factory=list)
    # The request-named override profile actually applied (gate on + valid +
    # exists + identity-allowed), else None. Echoed to the client as
    # `profile_applied`.
    request_profile_applied: str | None = None
    # Effective per-identity gate: may this caller send per-request
    # decode_overrides at all? When False, every client override is treated as
    # locked (locked_client_keys covers all client keys) and reported ignored.
    allow_request_decode_overrides: bool = True
    # Ordered identity layer ids that contributed (most → least specific).
    layers: list[str] = field(default_factory=list)
    # Per-field provenance (verbose path only) — drives the /resolve waterfall.
    provenance: dict[str, list[dict[str, Any]]] | None = None
    rule_provenance: dict[str, dict[str, Any]] | None = None

    def has_identity(self) -> bool:
        """True if any identity layer (key/user direct or profile) contributed.
        per-model-only folding does not count as identity."""
        return bool(self.layers)


# ---------------------------------------------------------------------------
# Layer extraction
# ---------------------------------------------------------------------------

def _blob_to_layer(layer_id: str, label: str, profile_name: str | None,
                   blob: Any) -> dict[str, Any] | None:
    """Normalise an OverrideProfile-shaped dict (a direct blob or a profile)
    into a layer. Returns None for an empty / non-dict blob so it contributes
    nothing."""
    if not isinstance(blob, dict) or not blob:
        return None
    fields = {
        k: v for k, v in blob.items()
        if k in SCALAR_OVERRIDE_FIELDS and v is not None
    }
    locks = {f for f in (blob.get("locks") or []) if f in SCALAR_OVERRIDE_FIELDS}
    exclude = {s for s in (blob.get("PIPELINE_RULES_EXCLUDE") or [])}
    include = {s for s in (blob.get("PIPELINE_RULES_INCLUDE") or [])}
    if not (fields or locks or exclude or include):
        return None
    return {
        "id": layer_id, "label": label, "profile": profile_name,
        "fields": fields, "locks": locks,
        "exclude": exclude, "include": include,
    }


def _safe_binding(getter: Any, ident_id: str | None) -> dict[str, Any]:
    """Fetch a per-identity binding, tolerating sentinel ids and any store
    error — a bad binding must never crash a decode."""
    if not ident_id or ident_id in _SENTINEL_IDS:
        return {}
    try:
        return getter(ident_id) or {}
    except Exception:
        return {}


def _binding_flag(key_binding: dict[str, Any], user_binding: dict[str, Any],
                  field_name: str) -> bool | None:
    """Most-specific per-identity opinion on a boolean gate: key wins over user;
    None = no opinion (inherit the global floor)."""
    for b in (key_binding, user_binding):
        v = b.get(field_name) if isinstance(b, dict) else None
        if isinstance(v, bool):
            return v
    return None


def _effective_flag(key_binding: dict[str, Any], user_binding: dict[str, Any],
                    field_name: str, global_value: Any) -> bool:
    """Effective gate = the global floor AND the most-specific identity opinion
    (default allow when no identity sets it). A per-identity binding can only
    NARROW the global gate, never widen it: if global is off, no binding can
    re-enable the feature."""
    identity = _binding_flag(key_binding, user_binding, field_name)
    if identity is None:
        return bool(global_value)
    return bool(global_value) and identity


def _effective_allowlist(key_binding: dict[str, Any],
                         user_binding: dict[str, Any]) -> list[str] | None:
    """The most-specific override-profile allowlist (key over user), or None when
    neither sets one (= no restriction, all requestable profiles allowed)."""
    for b in (key_binding, user_binding):
        v = b.get("allowed_override_profiles") if isinstance(b, dict) else None
        if v is not None:
            return v
    return None


def _allowlist_permits(allowlist: list[str] | None, name: str) -> bool:
    """True if `name` is permitted by an allowlist. None = no restriction; the
    wildcard "*" = all; an explicit (possibly empty) list must contain the name."""
    if allowlist is None:
        return True
    if config_store.ALLOWED_PROFILES_WILDCARD in allowlist:
        return True
    return name in allowlist


def _gather_identity_layers(key_binding: dict[str, Any],
                            user_binding: dict[str, Any],
                            request_profile: str | None = None,
                            *,
                            request_allowed: bool = True,
                            allowlist: list[str] | None = None) -> list[dict[str, Any]]:
    """Build the ordered (most → least specific) identity layer list from the
    already-fetched bindings: key.direct, key.profiles…, user.direct,
    user.profiles…, then the request-named profile (least specific) if any and
    permitted."""
    profiles = getattr(cfg, "OVERRIDE_PROFILES", None) or {}
    layers: list[dict[str, Any]] = []

    def _append_binding(scope: str, binding: Any) -> None:
        if not isinstance(binding, dict):
            return
        direct = _blob_to_layer(f"{scope}.direct", f"{scope} · direct",
                                None, binding.get("direct"))
        if direct is not None:
            layers.append(direct)
        for pname in (binding.get("profiles") or []):
            prof = profiles.get(pname)
            lyr = _blob_to_layer(f"{scope}.profile:{pname}",
                                 f"{scope} · profile {pname}", pname, prof)
            if lyr is not None:
                layers.append(lyr)
            # A referenced-but-missing/empty profile contributes nothing — the
            # resolver tolerates it (delete is blocked while in use, but a race
            # or hand-edited file must never crash a decode).

    # Key scope is more specific than user scope.
    _append_binding("key", key_binding)
    _append_binding("user", user_binding)
    # The request-named profile is the LEAST-specific identity layer: appended
    # last, so it fills only fields no key/user layer set and can never override
    # or unlock an admin-pinned value (first layer with an opinion wins).
    req_layer = _request_profile_layer(request_profile, allowed=request_allowed,
                                       allowlist=allowlist)
    if req_layer is not None:
        layers.append(req_layer)
    return layers


def _request_profile_layer(request_profile: str | None, *,
                           allowed: bool = True,
                           allowlist: list[str] | None = None) -> dict[str, Any] | None:
    """The request-supplied profile name as a (least-specific) identity layer, or
    None when refused. Refused if: the global gate is off; the per-identity gate
    (`allowed`) is off; the name is malformed / unknown; the profile is flagged
    `requestable: false`; or the per-identity allowlist excludes it. Tolerant —
    an unusable name contributes nothing and never raises."""
    if not request_profile:
        return None
    if not getattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", True):
        return None
    if not allowed:
        return None
    if not config_store.TAG_RE.match(request_profile):
        return None
    if not _allowlist_permits(allowlist, request_profile):
        return None
    profiles = getattr(cfg, "OVERRIDE_PROFILES", None) or {}
    blob = profiles.get(request_profile)
    # Per-profile opt-out: a profile flagged not-requestable is never selectable
    # by a client, even if the identity's allowlist would otherwise permit it.
    if isinstance(blob, dict) and blob.get("requestable") is False:
        return None
    return _blob_to_layer(
        f"request.profile:{request_profile}",
        f"request · profile {request_profile}",
        request_profile,
        blob,
    )


def _per_model_value(model_id: str | None, field_name: str) -> Any:
    """Per-model override value for a scalar field, or UNSET."""
    mo = getattr(cfg, "MODEL_OVERRIDES", None) or {}
    if model_id and isinstance(mo, dict):
        m = mo.get(model_id)
        if isinstance(m, dict):
            v = m.get(field_name)
            if v is not None:
                return v
    return UNSET


def _per_model_rule_layer(model_id: str | None) -> dict[str, Any] | None:
    """The per-model pipeline include/exclude as a (lowest-priority) rule
    layer, or None when the model has no per-model scoping."""
    mo = getattr(cfg, "MODEL_OVERRIDES", None) or {}
    m = mo.get(model_id) if (model_id and isinstance(mo, dict)) else None
    if not isinstance(m, dict):
        return None
    exclude = {s for s in (m.get("PIPELINE_RULES_EXCLUDE") or [])}
    include = {s for s in (m.get("PIPELINE_RULES_INCLUDE") or [])}
    if not (exclude or include):
        return None
    return {
        "id": "per-model", "label": f"per-model · {model_id}", "profile": None,
        "fields": {}, "locks": set(), "exclude": exclude, "include": include,
    }


# ---------------------------------------------------------------------------
# Core resolution
# ---------------------------------------------------------------------------

def _resolve_from_layers(model_id: str | None, layers: list[dict[str, Any]],
                         request_overrides: dict[str, Any],
                         with_provenance: bool) -> Resolved:
    """Pure core: resolve a Resolved from already-extracted identity layers
    (ordered most → least specific). Exposed for unit testing without any DB."""
    values: dict[str, Any] = {}
    locked: set[str] = set()
    provenance: dict[str, list[dict[str, Any]]] | None = {} if with_provenance else None

    for fname in SCALAR_OVERRIDE_FIELDS:
        # Most-specific-wins: the winner is the first layer with ANY opinion on
        # the field — it either SETS a value or LOCKS it (a value-less
        # lock-to-inherited counts as an opinion). The winner's value (if it
        # sets one) and lock (if it declares one) take effect; every lower layer
        # is shadowed. So a more-specific value beats a less-specific lock, AND
        # — symmetrically — a more-specific value-less lock beats a less-specific
        # value: it pins the inherited (per-model/global) value, drops the lower
        # override, and forbids the client from replacing it via decode_overrides.
        winner = None
        for layer in layers:
            if fname in layer["fields"] or fname in layer["locks"]:
                winner = layer
                break
        if winner is not None:
            if fname in winner["fields"]:
                values[fname] = winner["fields"][fname]
            if fname in winner["locks"]:
                locked.add(fname)
        if with_provenance:
            provenance[fname] = _scalar_provenance(fname, layers, winner, model_id)

    # Pipeline rules: identity layers first, then per-model. First mention wins.
    rule_layers = list(layers)
    pm_rule = _per_model_rule_layer(model_id)
    if pm_rule is not None:
        rule_layers.append(pm_rule)
    include: set[str] = set()
    exclude: set[str] = set()
    rule_provenance: dict[str, dict[str, Any]] | None = {} if with_provenance else None
    all_slugs: set[str] = set()
    for layer in rule_layers:
        all_slugs |= layer["exclude"] | layer["include"]
    for slug in all_slugs:
        decided_layer = None
        decided_state = None
        for layer in rule_layers:
            if slug in layer["exclude"]:
                exclude.add(slug)
                decided_layer, decided_state = layer, False
                break
            if slug in layer["include"]:
                include.add(slug)
                decided_layer, decided_state = layer, True
                break
        if with_provenance:
            rule_provenance[slug] = {
                "decided_by": decided_layer["id"] if decided_layer else None,
                "enabled": decided_state,
            }

    locked_client_keys = frozenset(
        _CONFIG_TO_CLIENT_KEY[f] for f in locked if f in _CONFIG_TO_CLIENT_KEY
    )
    dropped = sorted(
        k for k in (request_overrides or {}) if k in locked_client_keys
    )

    return Resolved(
        values=values,
        locked=locked,
        locked_client_keys=locked_client_keys,
        pipeline_include=include,
        pipeline_exclude=exclude,
        dropped=dropped,
        profiles_applied=[l["profile"] for l in layers if l["profile"]],
        layers=[l["id"] for l in layers],
        provenance=provenance,
        rule_provenance=rule_provenance,
    )


def _scalar_provenance(fname: str, layers: list[dict[str, Any]],
                       identity_winner: dict[str, Any] | None,
                       model_id: str | None) -> list[dict[str, Any]]:
    """Ordered layer stack for one scalar field, for the /resolve waterfall:
    identity layers, then per-model, then global. The winner is the most-
    specific layer with an opinion on the field; it owns the lock (if it
    declares one) and the value (if it sets one). A value-less winner locks the
    field but leaves the value to per-model/global, shadowing any lower layer's
    value — so the lock badge and the value-winner row can be different rows."""
    # A value-less lock-to-inherited winner owns the lock but supplies no value,
    # so the value still falls through to per-model/global.
    winner_sets = (identity_winner is not None
                   and fname in identity_winner["fields"])
    hits: list[dict[str, Any]] = []
    for layer in layers:
        has = fname in layer["fields"]
        hits.append({
            "layer_id": layer["id"],
            "label": layer["label"],
            "value": layer["fields"].get(fname) if has else None,
            "is_set": has,
            "is_winner": layer is identity_winner and winner_sets,
            "locked": layer is identity_winner and fname in layer["locks"],
        })
    pm = _per_model_value(model_id, fname)
    pm_set = pm is not UNSET
    pm_winner = not winner_sets and pm_set
    hits.append({
        "layer_id": "per-model",
        "label": f"per-model · {model_id}" if model_id else "per-model",
        "value": pm if pm_set else None,
        "is_set": pm_set, "is_winner": pm_winner, "locked": False,
    })
    gv = getattr(cfg, fname, None)
    g_set = gv is not None
    g_winner = not winner_sets and not pm_set and g_set
    hits.append({
        "layer_id": "global", "label": "global default",
        "value": gv, "is_set": g_set, "is_winner": g_winner, "locked": False,
    })
    return hits


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def resolve(model_id: str | None, *, user_id: str | None = None,
            key_id: str | None = None,
            request_overrides: dict[str, Any] | None = None,
            request_profile: str | None = None,
            with_provenance: bool = False) -> Resolved:
    """Resolve the effective per-identity config.

    ``request_overrides`` is the client's per-request decode_overrides dict
    (lowercase faster-whisper kwarg keys), used only to compute ``dropped``
    (which locked keys this request tried to set). Pass ``with_provenance=True``
    for the verbose /resolve waterfall; the live decode path leaves it False.

    A caller with no per-identity config (or open mode) yields a Resolved whose
    identity layers are empty but whose pipeline_include/exclude still fold the
    per-model rules — so threading it through behaves exactly like today.

    ``request_profile`` (optional) names an OVERRIDE_PROFILES entry to apply as
    the least-specific identity layer; gated by ALLOW_REQUEST_OVERRIDE_PROFILE
    (global) AND the per-identity gate + allowlist, and ignored if
    malformed/unknown/not-requestable. ``request_profile_applied`` echoes what
    actually took effect. The per-identity ALLOW_REQUEST_DECODE_OVERRIDES gate is
    resolved the same way; when off, every client decode override is treated as
    locked (``locked_client_keys`` covers all client keys) and reported ignored.
    """
    import api_keys_store  # local import keeps module import order flexible
    key_binding = _safe_binding(api_keys_store.get_key_config, key_id)
    user_binding = _safe_binding(api_keys_store.get_user_config, user_id)

    op_allowed = _effective_flag(key_binding, user_binding,
                                 "allow_request_override_profile",
                                 getattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", True))
    do_allowed = _effective_flag(key_binding, user_binding,
                                 "allow_request_decode_overrides",
                                 getattr(cfg, "ALLOW_REQUEST_DECODE_OVERRIDES", True))
    allowlist = _effective_allowlist(key_binding, user_binding)

    layers = _gather_identity_layers(key_binding, user_binding, request_profile,
                                     request_allowed=op_allowed, allowlist=allowlist)
    req_overrides = request_overrides or {}
    resolved = _resolve_from_layers(model_id, layers, req_overrides, with_provenance)

    # The request profile "applied" iff its layer actually contributed.
    req_id = f"request.profile:{request_profile}" if request_profile else None
    resolved.request_profile_applied = (
        request_profile if (req_id is not None and req_id in resolved.layers) else None
    )

    # Per-identity decode-override master gate. When off, lock EVERY client key so
    # _apply_decode_overrides drops them all and every site that reports
    # `overrides_ignored` (batch + streaming) reports them — one enforcement point.
    resolved.allow_request_decode_overrides = do_allowed
    if not do_allowed:
        all_client_keys = frozenset(_CONFIG_TO_CLIENT_KEY.values())
        resolved.locked_client_keys = all_client_keys
        resolved.dropped = sorted(k for k in req_overrides if k in all_client_keys)
    return resolved


# ---------------------------------------------------------------------------
# Capabilities — what a caller may do (drives the client UI; the server still
# enforces everything in resolve(), the client UI is convenience only).
# ---------------------------------------------------------------------------

def _caller_gates(user_id: str | None,
                  key_id: str | None) -> tuple[bool, bool, list[str] | None]:
    """The caller's effective (override-profile gate, decode-override gate,
    override-profile allowlist). Shared by the capabilities + names endpoints."""
    import api_keys_store
    kb = _safe_binding(api_keys_store.get_key_config, key_id)
    ub = _safe_binding(api_keys_store.get_user_config, user_id)
    op = _effective_flag(kb, ub, "allow_request_override_profile",
                         getattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", True))
    do = _effective_flag(kb, ub, "allow_request_decode_overrides",
                         getattr(cfg, "ALLOW_REQUEST_DECODE_OVERRIDES", True))
    allowlist = _effective_allowlist(kb, ub)
    return op, do, allowlist


def _filter_profile_names(gate_on: bool, allowlist: list[str] | None) -> list[str]:
    """The concrete OVERRIDE_PROFILES names a caller with this gate + allowlist may
    request: requestable profiles intersected with the allowlist. [] when off."""
    if not gate_on:
        return []
    profiles = getattr(cfg, "OVERRIDE_PROFILES", None) or {}
    return sorted(
        name for name, blob in profiles.items()
        if not (isinstance(blob, dict) and blob.get("requestable") is False)
        and _allowlist_permits(allowlist, name)
    )


def allowed_profile_names(user_id: str | None = None,
                          key_id: str | None = None) -> list[str]:
    """The concrete override-profile names this caller may NAME per request."""
    op, _do, allowlist = _caller_gates(user_id, key_id)
    return _filter_profile_names(op, allowlist)


def resolve_capabilities(user_id: str | None = None,
                         key_id: str | None = None) -> dict[str, Any]:
    """The caller's request-override capabilities, for GET /v1/me. `allowed_
    override_profiles` is ["*"] when unrestricted (free choice from the names
    endpoint), an explicit concrete list when the admin restricted it, or []
    when the gate is off."""
    op, do, allowlist = _caller_gates(user_id, key_id)
    if not op:
        allowed: list[str] = []
    elif allowlist is None or config_store.ALLOWED_PROFILES_WILDCARD in allowlist:
        allowed = [config_store.ALLOWED_PROFILES_WILDCARD]
    else:
        allowed = _filter_profile_names(op, allowlist)
    return {
        "can_request_override_profile": op,
        "can_request_decode_overrides": do,
        "allowed_override_profiles": allowed,
    }


def project_profile_to_client(blob: Any) -> tuple[dict[str, Any], list[str]]:
    """Project an OVERRIDE_PROFILES blob to the lowercase client decode_override
    keys it sets, plus the client keys it locks. Server-managed-only fields
    (streaming, output wrappers, language detection) have no client key and are
    omitted — they are reachable only by naming the profile, never per field."""
    if not isinstance(blob, dict):
        return {}, []
    values: dict[str, Any] = {}
    for cfg_field, client_key in _CONFIG_TO_CLIENT_KEY.items():
        v = blob.get(cfg_field)
        if v is not None:
            values[client_key] = v
    locked = sorted(
        _CONFIG_TO_CLIENT_KEY[f] for f in (blob.get("locks") or [])
        if f in _CONFIG_TO_CLIENT_KEY
    )
    return values, locked
