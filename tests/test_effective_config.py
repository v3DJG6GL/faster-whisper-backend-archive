"""Pure-Python tests for the layered per-identity config resolver.

These exercise effective_config._resolve_from_layers (the pure core) and the
config_store schema. No faster-whisper / DB needed — runnable on the web-only
box with `pytest -o addopts="" tests/test_effective_config.py`.
"""

import pytest

import config as cfg
import config_store as cs
import effective_config as ec


# --- layer builders -------------------------------------------------------

def _layer(scope, **blob):
    return ec._blob_to_layer(f"{scope}.direct", f"{scope} · direct", None, blob)


def _prof(scope, name, **blob):
    return ec._blob_to_layer(f"{scope}.profile:{name}",
                             f"{scope} · profile {name}", name, blob)


def _resolve(layers, model_id=None, req=None, prov=False):
    layers = [l for l in layers if l is not None]
    return ec._resolve_from_layers(model_id, layers, req or {}, prov)


# --- schema ---------------------------------------------------------------

def test_lockable_excludes_pipeline_lists():
    assert "BEAM_SIZE" in cs.LOCKABLE_FIELDS
    assert "STREAMING_PARTIAL_BEAM" in cs.LOCKABLE_FIELDS
    # the idle timeout is a per-caller policy → per-identity overridable + lockable
    assert "STREAMING_IDLE_TIMEOUT_SEC" in cs.LOCKABLE_FIELDS
    assert "PIPELINE_RULES_EXCLUDE" not in cs.LOCKABLE_FIELDS
    assert "PIPELINE_RULES_INCLUDE" not in cs.LOCKABLE_FIELDS
    # load-time model fields are never per-identity overridable
    assert "MODEL_DEVICE" not in cs.LOCKABLE_FIELDS
    # hard server-capacity caps are server-wide, never per-identity
    assert "STREAMING_MAX_SESSIONS" not in cs.LOCKABLE_FIELDS


def test_model_override_keeps_loadtime_and_calltime():
    f = set(cs.ModelOverride.model_fields)
    assert {"MODEL_DEVICE", "NUM_WORKERS", "REVISION"} <= f      # load-time
    assert {"BEAM_SIZE", "PIPELINE_RULES_EXCLUDE"} <= f          # call-time mixin


def test_profile_rejects_loadtime_and_bad_lock():
    with pytest.raises(Exception):
        cs.OverrideProfile.model_validate({"MODEL_DEVICE": "cpu"})
    with pytest.raises(Exception):
        cs.OverrideProfile.model_validate({"locks": ["NOPE"]})


def test_profile_roundtrip_through_adminconfig():
    ac = cs.AdminConfig.model_validate({"OVERRIDE_PROFILES": {
        "clinic-de": {"DEFAULT_LANGUAGE": "de", "BEAM_SIZE": 8,
                      "STREAMING_PARTIAL_BEAM": 3, "locks": ["DEFAULT_LANGUAGE"]}}})
    out = ac.model_dump(exclude_none=True)["OVERRIDE_PROFILES"]["clinic-de"]
    assert out["DEFAULT_LANGUAGE"] == "de" and out["BEAM_SIZE"] == 8
    assert out["locks"] == ["DEFAULT_LANGUAGE"]


# --- scalar precedence ----------------------------------------------------

def test_empty_is_noop():
    r = _resolve([])
    assert r.values == {} and not r.locked
    assert not r.pipeline_include and not r.pipeline_exclude
    assert not r.has_identity()


def test_single_layer_sets_value_unlocked():
    r = _resolve([_layer("user", BEAM_SIZE=8)])
    assert r.values["BEAM_SIZE"] == 8
    assert "BEAM_SIZE" not in r.locked
    assert r.has_identity()


def test_key_beats_user():
    r = _resolve([_layer("key", BEAM_SIZE=10), _layer("user", BEAM_SIZE=8)])
    assert r.values["BEAM_SIZE"] == 10


def test_first_layer_owns_lock_surprise():
    # key wins the value (unlocked); user's lock is shadowed because key is the
    # owning (most-specific) layer. Documented, intentional semantics.
    r = _resolve([_layer("key", BEAM_SIZE=10),
                  _layer("user", BEAM_SIZE=8, locks=["BEAM_SIZE"])])
    assert r.values["BEAM_SIZE"] == 10
    assert "BEAM_SIZE" not in r.locked


def test_profiles_earlier_wins():
    # user.profiles [a, b]: a is earlier → wins on a conflicting field.
    r = _resolve([_prof("user", "a", DEFAULT_LANGUAGE="de"),
                  _prof("user", "b", DEFAULT_LANGUAGE="fr")])
    assert r.values["DEFAULT_LANGUAGE"] == "de"


def test_lock_sets_client_keys_and_dropped():
    r = _resolve([_layer("user", TEMPERATURE="0.0", locks=["TEMPERATURE"])],
                 req={"temperature": 0.5, "beam_size": 12})
    assert "TEMPERATURE" in r.locked
    assert "temperature" in r.locked_client_keys
    assert r.dropped == ["temperature"]            # beam_size not locked → kept


def test_value_less_lock_pins_inherited():
    # A layer that LOCKS a field WITHOUT overriding it: the value stays inherited
    # (no entry in .values → cfg_for falls through to per-model/global), but the
    # field is locked so a client decode_override for it is dropped.
    r = _resolve([_layer("user", locks=["TEMPERATURE"])],
                 req={"temperature": 0.9})
    assert "TEMPERATURE" not in r.values
    assert "TEMPERATURE" in r.locked
    assert "temperature" in r.locked_client_keys
    assert r.dropped == ["temperature"]


def test_value_setting_winner_shadows_lower_value_less_lock():
    # first-layer-owns-lock still holds: a higher layer that SETS the value
    # (unlocked) shadows a lower layer's value-less lock on the same field.
    r = _resolve([_layer("key", TEMPERATURE="0.0"),
                  _layer("user", locks=["TEMPERATURE"])],
                 req={"temperature": 0.9})
    assert r.values["TEMPERATURE"] == "0.0"
    assert "TEMPERATURE" not in r.locked
    assert "temperature" not in r.locked_client_keys


def test_more_specific_value_less_lock_shadows_lower_value(monkeypatch):
    # Mirror of the above (most-specific-wins both ways): a MORE-specific
    # value-less lock beats a LESS-specific value. The lock wins, so the value
    # is pinned to inherited (here the per-model default), the lower layer's
    # value is dropped, and the field is locked against client overrides.
    monkeypatch.setattr(cfg, "MODEL_OVERRIDES",
                        {"m1": {"TEMPERATURE": "0.0"}}, raising=False)
    r = _resolve([_layer("key", locks=["TEMPERATURE"]),
                  _layer("user", TEMPERATURE="0.7")],
                 model_id="m1", req={"temperature": 0.9})
    assert "TEMPERATURE" not in r.values          # user's 0.7 shadowed → inherits
    assert "TEMPERATURE" in r.locked
    assert "temperature" in r.locked_client_keys
    assert r.dropped == ["temperature"]


def test_vad_lock_maps_to_client_subparam_key():
    r = _resolve([_layer("user", VAD_MIN_SILENCE_MS=500,
                         locks=["VAD_MIN_SILENCE_MS"])],
                 req={"vad_min_silence_duration_ms": 100})
    assert "vad_min_silence_duration_ms" in r.dropped


# --- pipeline rules -------------------------------------------------------

def test_rule_first_mention_wins():
    # key.include r2 is more specific than user.exclude r2 → enabled.
    r = _resolve([_layer("key", PIPELINE_RULES_INCLUDE=["r2"]),
                  _layer("user", PIPELINE_RULES_EXCLUDE=["r2"])])
    assert "r2" in r.pipeline_include and "r2" not in r.pipeline_exclude


def test_rule_exclude_wins_within_layer():
    r = _resolve([_layer("user", PIPELINE_RULES_EXCLUDE=["r1"],
                         PIPELINE_RULES_INCLUDE=["r9"])])
    assert "r1" in r.pipeline_exclude and "r9" in r.pipeline_include


def test_per_model_rules_folded_without_identity(monkeypatch):
    monkeypatch.setattr(cfg, "MODEL_OVERRIDES",
                        {"m1": {"PIPELINE_RULES_EXCLUDE": ["r3"]}}, raising=False)
    r = _resolve([], model_id="m1")
    assert "r3" in r.pipeline_exclude
    assert not r.has_identity()          # per-model folding is not identity


def test_identity_rule_overrides_per_model(monkeypatch):
    monkeypatch.setattr(cfg, "MODEL_OVERRIDES",
                        {"m1": {"PIPELINE_RULES_EXCLUDE": ["r3"]}}, raising=False)
    # user force-includes r3 → identity (more specific) wins over per-model.
    r = _resolve([_layer("user", PIPELINE_RULES_INCLUDE=["r3"])], model_id="m1")
    assert "r3" in r.pipeline_include and "r3" not in r.pipeline_exclude


# --- provenance (verbose path) -------------------------------------------

def test_provenance_marks_single_winner(monkeypatch):
    monkeypatch.setattr(cfg, "MODEL_OVERRIDES",
                        {"m1": {"BEAM_SIZE": 6}}, raising=False)
    r = _resolve([_layer("user", BEAM_SIZE=8)], model_id="m1", prov=True)
    stack = r.provenance["BEAM_SIZE"]
    winners = [h for h in stack if h["is_winner"]]
    assert len(winners) == 1
    assert winners[0]["layer_id"] == "user.direct" and winners[0]["value"] == 8
    # per-model + global rows present and overridden (not winners)
    pm = [h for h in stack if h["layer_id"] == "per-model"][0]
    assert pm["is_set"] and pm["value"] == 6 and not pm["is_winner"]


def test_provenance_falls_to_per_model_then_global(monkeypatch):
    monkeypatch.setattr(cfg, "MODEL_OVERRIDES",
                        {"m1": {"BEAM_SIZE": 6}}, raising=False)
    r = _resolve([], model_id="m1", prov=True)
    stack = r.provenance["BEAM_SIZE"]
    winners = [h for h in stack if h["is_winner"]]
    assert len(winners) == 1 and winners[0]["layer_id"] == "per-model"


def test_provenance_value_less_lock_owned_by_first_layer():
    # Two layers value-less-lock the same field and nobody sets a value. The
    # resolver owns the lock at the first (most-specific) layer only, so the
    # waterfall must show the lock badge there alone — not on every layer that
    # declares the lock.
    r = _resolve([_layer("key", locks=["TEMPERATURE"]),
                  _layer("user", locks=["TEMPERATURE"])], prov=True)
    assert "TEMPERATURE" in r.locked
    stack = r.provenance["TEMPERATURE"]
    assert [h["layer_id"] for h in stack if h["locked"]] == ["key.direct"]


def test_provenance_value_less_lock_winner_inherits_value(monkeypatch):
    # When a value-less lock is the most-specific opinion it owns the lock, but
    # the VALUE still comes from per-model/global. The waterfall marks the lock
    # layer's row as locked (not value-winner), the per-model row as the value
    # winner, and the shadowed lower value as set-but-not-winner.
    monkeypatch.setattr(cfg, "MODEL_OVERRIDES",
                        {"m1": {"TEMPERATURE": "0.0"}}, raising=False)
    r = _resolve([_layer("key", locks=["TEMPERATURE"]),
                  _layer("user", TEMPERATURE="0.7")],
                 model_id="m1", prov=True)
    stack = r.provenance["TEMPERATURE"]
    assert [h["layer_id"] for h in stack if h["locked"]] == ["key.direct"]
    assert [h["layer_id"] for h in stack if h["is_winner"]] == ["per-model"]
    user = [h for h in stack if h["layer_id"] == "user.direct"][0]
    assert user["is_set"] and not user["is_winner"]


# --- public resolve() open-mode shortcut ----------------------------------

def test_resolve_open_mode_no_identity():
    r = ec.resolve("whisper-1", user_id="(open-mode)", key_id="(open-mode)")
    assert not r.has_identity()
    assert r.values == {}


def test_blob_to_layer_empty_is_none():
    assert ec._blob_to_layer("x", "x", None, {}) is None
    assert ec._blob_to_layer("x", "x", None, {"locks": []}) is None


# --- request-named override profile (P10) ---------------------------------

def _set_profiles(monkeypatch, profiles, allow=True):
    monkeypatch.setattr(cfg, "OVERRIDE_PROFILES", profiles, raising=False)
    monkeypatch.setattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", allow, raising=False)


def test_request_profile_layer_valid(monkeypatch):
    _set_profiles(monkeypatch, {"fast": {"BEAM_SIZE": 3}})
    lyr = ec._request_profile_layer("fast")
    assert lyr is not None and lyr["fields"]["BEAM_SIZE"] == 3


def test_request_profile_layer_unknown_bad_or_gated(monkeypatch):
    _set_profiles(monkeypatch, {"fast": {"BEAM_SIZE": 3}})
    assert ec._request_profile_layer("nope") is None        # unknown name
    assert ec._request_profile_layer("Bad Name") is None    # fails TAG_RE
    assert ec._request_profile_layer("") is None            # empty
    assert ec._request_profile_layer(None) is None
    _set_profiles(monkeypatch, {"fast": {"BEAM_SIZE": 3}}, allow=False)
    assert ec._request_profile_layer("fast") is None        # gated off


def test_resolve_applies_request_profile(monkeypatch):
    # No user_id/key_id → no DB hit; only the request profile contributes.
    _set_profiles(monkeypatch, {"fast": {"BEAM_SIZE": 3}})
    r = ec.resolve("m1", request_profile="fast")
    assert r.values["BEAM_SIZE"] == 3
    assert r.request_profile_applied == "fast"


def test_resolve_request_profile_gated_off(monkeypatch):
    _set_profiles(monkeypatch, {"fast": {"BEAM_SIZE": 3}}, allow=False)
    r = ec.resolve("m1", request_profile="fast")
    assert "BEAM_SIZE" not in r.values
    assert r.request_profile_applied is None


def test_resolve_request_profile_unknown_not_applied(monkeypatch):
    _set_profiles(monkeypatch, {"fast": {"BEAM_SIZE": 3}})
    r = ec.resolve("m1", request_profile="nope")
    assert "BEAM_SIZE" not in r.values
    assert r.request_profile_applied is None


def test_request_profile_is_least_specific(monkeypatch):
    # SECURITY: the request profile is appended last, so it fills only fields no
    # key/user layer set, and can never override a value or escape a value-less
    # lock. Built with the pure core (request layer last) — no DB needed.
    _set_profiles(monkeypatch, {"fast": {"BEAM_SIZE": 3, "TEMPERATURE": "0.0",
                                         "DEFAULT_LANGUAGE": "fr"}})
    req_layer = ec._request_profile_layer("fast")
    layers = [_layer("key", BEAM_SIZE=10),         # key sets beam → profile can't move it
              _layer("user", locks=["TEMPERATURE"]),  # value-less lock → profile can't unlock
              req_layer]
    r = _resolve(layers, req={"temperature": 0.9})
    assert r.values["BEAM_SIZE"] == 10             # key wins over the profile
    assert "TEMPERATURE" not in r.values           # lock pins inherited; profile value shadowed
    assert "TEMPERATURE" in r.locked               # still locked despite the profile
    assert "temperature" in r.locked_client_keys
    assert r.values["DEFAULT_LANGUAGE"] == "fr"    # profile fills a field nobody set


# --- per-identity request gates + allowlist (P11) -------------------------

def _bindings(monkeypatch, *, key=None, user=None):
    """Monkeypatch the per-identity binding fetch — no DB. Pass partial binding
    dicts (e.g. {"allow_request_override_profile": False})."""
    import api_keys_store
    monkeypatch.setattr(api_keys_store, "get_key_config",
                        lambda kid: key or {"direct": {}, "profiles": []}, raising=False)
    monkeypatch.setattr(api_keys_store, "get_user_config",
                        lambda uid: user or {"direct": {}, "profiles": []}, raising=False)


def test_effective_flag_inherits_and_narrows():
    # unset → inherit the global floor
    assert ec._effective_flag({}, {}, "allow_request_override_profile", True) is True
    assert ec._effective_flag({}, {}, "allow_request_override_profile", False) is False
    # key beats user; either can NARROW the global on
    assert ec._effective_flag({"allow_request_override_profile": False},
                              {"allow_request_override_profile": True},
                              "allow_request_override_profile", True) is False
    # NEVER widen: global off can't be re-enabled per-identity
    assert ec._effective_flag({"allow_request_override_profile": True}, {},
                              "allow_request_override_profile", False) is False


def test_effective_allowlist_key_over_user():
    assert ec._effective_allowlist({}, {}) is None                       # inherit / all
    assert ec._effective_allowlist({"allowed_override_profiles": ["a"]},
                                   {"allowed_override_profiles": ["b"]}) == ["a"]
    assert ec._effective_allowlist({}, {"allowed_override_profiles": []}) == []


def test_request_profile_per_identity_gate(monkeypatch):
    _set_profiles(monkeypatch, {"fast": {"BEAM_SIZE": 3}})
    # key gate off → refused even though global is on
    _bindings(monkeypatch, key={"direct": {}, "profiles": [],
                                "allow_request_override_profile": False})
    r = ec.resolve("m", key_id="k", request_profile="fast")
    assert r.request_profile_applied is None and "BEAM_SIZE" not in r.values
    # no per-identity opinion → inherits global on
    _bindings(monkeypatch)
    r = ec.resolve("m", key_id="k", request_profile="fast")
    assert r.request_profile_applied == "fast" and r.values["BEAM_SIZE"] == 3


def test_request_profile_allowlist(monkeypatch):
    _set_profiles(monkeypatch, {"fast": {"BEAM_SIZE": 3}})
    _bindings(monkeypatch, key={"direct": {}, "profiles": [],
                                "allowed_override_profiles": ["other"]})
    assert ec.resolve("m", key_id="k", request_profile="fast").request_profile_applied is None
    _bindings(monkeypatch, key={"direct": {}, "profiles": [],
                                "allowed_override_profiles": ["fast"]})
    assert ec.resolve("m", key_id="k", request_profile="fast").request_profile_applied == "fast"
    _bindings(monkeypatch, key={"direct": {}, "profiles": [],
                                "allowed_override_profiles": ["*"]})
    assert ec.resolve("m", key_id="k", request_profile="fast").request_profile_applied == "fast"
    _bindings(monkeypatch, key={"direct": {}, "profiles": [],
                                "allowed_override_profiles": []})
    assert ec.resolve("m", key_id="k", request_profile="fast").request_profile_applied is None


def test_request_profile_requestable_false_refused(monkeypatch):
    _set_profiles(monkeypatch, {"internal": {"BEAM_SIZE": 1, "requestable": False}})
    _bindings(monkeypatch)  # gate on, no allowlist → all permitted
    r = ec.resolve("m", key_id="k", request_profile="internal")
    assert r.request_profile_applied is None and "BEAM_SIZE" not in r.values
    assert ec._request_profile_layer("internal", allowed=True, allowlist=["*"]) is None


def test_decode_master_gate_off_drops_all(monkeypatch):
    _set_profiles(monkeypatch, {})
    _bindings(monkeypatch, key={"direct": {}, "profiles": [],
                                "allow_request_decode_overrides": False})
    r = ec.resolve("m", key_id="k",
                   request_overrides={"beam_size": 9, "temperature": 0.5})
    assert r.allow_request_decode_overrides is False
    assert set(r.dropped) == {"beam_size", "temperature"}
    assert {"beam_size", "temperature"} <= set(r.locked_client_keys)
    # gate on (default) leaves request overrides alone
    _bindings(monkeypatch)
    r2 = ec.resolve("m", key_id="k", request_overrides={"beam_size": 9})
    assert r2.allow_request_decode_overrides is True and r2.dropped == []


def test_request_profile_cant_escape_lock_via_resolve(monkeypatch):
    # SECURITY through the full resolve() path: an allowlisted request profile
    # still cannot move a key-locked value.
    _set_profiles(monkeypatch, {"fast": {"BEAM_SIZE": 3}})
    _bindings(monkeypatch, key={"direct": {"BEAM_SIZE": 10, "locks": ["BEAM_SIZE"]},
                                "profiles": [], "allowed_override_profiles": ["fast"]})
    r = ec.resolve("m", key_id="k", request_profile="fast",
                   request_overrides={"beam_size": 7})
    assert r.values["BEAM_SIZE"] == 10            # key value wins
    assert "BEAM_SIZE" in r.locked                # still locked
    assert "beam_size" in r.dropped               # client override refused
    assert r.request_profile_applied == "fast"    # applied, just shadowed


# --- "no profile" suppression sentinel (P27) ------------------------------

def test_none_sentinel_suppresses_bound_profile(monkeypatch):
    # A profile bound to the identity normally applies; the reserved "__none__"
    # request name suppresses every bound profile → plain defaults.
    _set_profiles(monkeypatch, {"clinic": {"BEAM_SIZE": 7}})
    _bindings(monkeypatch, key={"direct": {}, "profiles": ["clinic"]})
    assert ec.resolve("m", key_id="k").values["BEAM_SIZE"] == 7   # bound profile applies
    r = ec.resolve("m", key_id="k", request_profile=cs.NO_PROFILE_SENTINEL)
    assert "BEAM_SIZE" not in r.values            # bound profile suppressed
    assert r.request_profile_applied is None       # the sentinel adds no layer


def test_none_sentinel_keeps_direct_identity_config(monkeypatch):
    # Suppression drops bound PROFILE layers but keeps the identity's direct
    # config (direct isn't a "profile").
    _set_profiles(monkeypatch, {"clinic": {"DEFAULT_LANGUAGE": "de"}})
    _bindings(monkeypatch, key={"direct": {"BEAM_SIZE": 5}, "profiles": ["clinic"]})
    r = ec.resolve("m", key_id="k", request_profile=cs.NO_PROFILE_SENTINEL)
    assert r.values["BEAM_SIZE"] == 5             # direct config retained
    assert "DEFAULT_LANGUAGE" not in r.values     # bound profile suppressed


def test_none_sentinel_refused_when_gated_off(monkeypatch):
    # Global gate off → suppression refused; the bound profile still applies (an
    # admin who forces profiles by turning the gate off is not bypassed).
    _set_profiles(monkeypatch, {"clinic": {"BEAM_SIZE": 7}}, allow=False)
    _bindings(monkeypatch, key={"direct": {}, "profiles": ["clinic"]})
    assert ec.resolve("m", key_id="k",
                      request_profile=cs.NO_PROFILE_SENTINEL).values["BEAM_SIZE"] == 7
    # Per-identity gate off (global on) → also refused.
    _set_profiles(monkeypatch, {"clinic": {"BEAM_SIZE": 7}}, allow=True)
    _bindings(monkeypatch, key={"direct": {}, "profiles": ["clinic"],
                                "allow_request_override_profile": False})
    assert ec.resolve("m", key_id="k",
                      request_profile=cs.NO_PROFILE_SENTINEL).values["BEAM_SIZE"] == 7


# --- admin per-key apply_no_profiles force --------------------------------

def test_apply_no_profiles_suppresses_user_bound_profile(monkeypatch):
    # The per-key admin force suppresses profiles bound at the USER scope too.
    _set_profiles(monkeypatch, {"clinic": {"BEAM_SIZE": 7}})
    _bindings(monkeypatch, user={"direct": {}, "profiles": ["clinic"]})
    assert ec.resolve("m", key_id="k", user_id="u").values["BEAM_SIZE"] == 7  # applies
    _bindings(monkeypatch, key={"direct": {}, "profiles": [], "apply_no_profiles": True},
              user={"direct": {}, "profiles": ["clinic"]})
    r = ec.resolve("m", key_id="k", user_id="u")
    assert "BEAM_SIZE" not in r.values           # user-bound profile suppressed
    assert r.profiles_applied == []


def test_apply_no_profiles_suppresses_key_bound_profile_keeps_direct(monkeypatch):
    # Suppression drops the key's own bound profile but keeps its direct config.
    _set_profiles(monkeypatch, {"clinic": {"DEFAULT_LANGUAGE": "de"}})
    _bindings(monkeypatch, key={"direct": {"BEAM_SIZE": 5}, "profiles": ["clinic"],
                                "apply_no_profiles": True})
    r = ec.resolve("m", key_id="k")
    assert r.values["BEAM_SIZE"] == 5            # direct config retained
    assert "DEFAULT_LANGUAGE" not in r.values    # bound profile suppressed
    assert r.profiles_applied == []


def test_apply_no_profiles_not_gated_by_global(monkeypatch):
    # Unlike the client "__none__" opt-out, the admin force is NOT bound by
    # ALLOW_REQUEST_OVERRIDE_PROFILE: it suppresses even when the gate is off.
    _set_profiles(monkeypatch, {"clinic": {"BEAM_SIZE": 7}}, allow=False)
    _bindings(monkeypatch, key={"direct": {}, "profiles": ["clinic"],
                                "apply_no_profiles": True})
    assert "BEAM_SIZE" not in ec.resolve("m", key_id="k").values


def test_apply_no_profiles_suppresses_requested_profile(monkeypatch):
    # With the force on, even a permitted per-request profile is suppressed.
    _set_profiles(monkeypatch, {"fast": {"BEAM_SIZE": 3}})
    _bindings(monkeypatch, key={"direct": {}, "profiles": [], "apply_no_profiles": True})
    r = ec.resolve("m", key_id="k", request_profile="fast")
    assert "BEAM_SIZE" not in r.values
    assert r.request_profile_applied is None


def test_apply_no_profiles_ignored_on_user_binding(monkeypatch):
    # The force is read off the KEY binding only; set on a user binding it does
    # nothing (the user-bound profile still applies).
    _set_profiles(monkeypatch, {"clinic": {"BEAM_SIZE": 7}})
    _bindings(monkeypatch, user={"direct": {}, "profiles": ["clinic"],
                                 "apply_no_profiles": True})
    assert ec.resolve("m", key_id="k", user_id="u").values["BEAM_SIZE"] == 7
