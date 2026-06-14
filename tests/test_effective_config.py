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
    assert "PIPELINE_RULES_EXCLUDE" not in cs.LOCKABLE_FIELDS
    assert "PIPELINE_RULES_INCLUDE" not in cs.LOCKABLE_FIELDS
    # load-time model fields are never per-identity overridable
    assert "MODEL_DEVICE" not in cs.LOCKABLE_FIELDS


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


# --- public resolve() open-mode shortcut ----------------------------------

def test_resolve_open_mode_no_identity():
    r = ec.resolve("whisper-1", user_id="(open-mode)", key_id="(open-mode)")
    assert not r.has_identity()
    assert r.values == {}


def test_blob_to_layer_empty_is_none():
    assert ec._blob_to_layer("x", "x", None, {}) is None
    assert ec._blob_to_layer("x", "x", None, {"locks": []}) is None
