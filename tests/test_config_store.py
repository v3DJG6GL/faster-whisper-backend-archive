"""Exhaustive tests for config_store: AdminConfig field bounds & validators,
ModelOverride, normalize_tags, the overrides load/save layer, the atomic
writer, and the small helper functions.

Factory-rule round-trips (load/save_factory_rules, terminal/dup/bad-regex)
are already covered by test_factory_rules.py at the repo root; here we add the
override layer, the scalar/model validators, and the helpers it does not touch.
"""

import json
import os
import time

import pytest
from pydantic import ValidationError

import config_store as cs


def _ok(**fields):
    """Validate a partial AdminConfig payload; return the model."""
    return cs.AdminConfig.model_validate(fields)


def _bad(**fields):
    with pytest.raises(ValidationError):
        cs.AdminConfig.model_validate(fields)


# ---------------------------------------------------------------------------
# extra=forbid
# ---------------------------------------------------------------------------

def test_unknown_key_rejected():
    _bad(NOT_A_REAL_FIELD=1)


def test_empty_payload_ok():
    m = _ok()
    assert m.BEAM_SIZE is None


# ---------------------------------------------------------------------------
# Numeric bounds (reject below / accept at / accept at / reject above)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,lo,hi", [
    ("MAX_LOADED_MODELS", 1, 8),
    ("MODEL_IDLE_TIMEOUT_S", 0, 86400),
    ("BEAM_SIZE", 1, 20),
    ("BEST_OF", 1, 20),
    ("VAD_MIN_SILENCE_MS", 0, 10000),
    ("VAD_SPEECH_PAD_MS", 0, 2000),
    ("NO_REPEAT_NGRAM_SIZE", 0, 10),
    ("LANGUAGE_DETECTION_SEGMENTS", 1, 10),
    ("CPU_THREADS", 0, 128),
    ("NUM_WORKERS", 1, 8),
    ("DEVICE_INDEX", 0, 15),
    ("LOG_BACKUP_COUNT", 1, 100),
    ("LOG_VIEWER_INITIAL_LINES", 10, 100_000),
    ("LOG_VIEWER_DOM_MAX", 0, 1_000_000),
    ("SERVER_PORT", 1, 65535),
    ("SERVER_WORKERS", 1, 8),
    ("REPORTS_MAX", 10, 100_000),
    ("REPORTS_RETENTION_DAYS", 0, 3650),
    ("RECENT_TRANSCRIPTIONS_MAX", 0, 100_000),
    ("STATS_RECENT_TRANSCRIPTIONS_COUNT", 1, 100),
    ("CAPTURES_MAX", 10, 1_000_000),
    ("CAPTURES_MAX_MB", 1, 10_000_000),
    ("LOG_MAX_BYTES", 1024 * 1024, 1024 * 1024 * 1024),
])
def test_int_bounds(field, lo, hi):
    _ok(**{field: lo})
    _ok(**{field: hi})
    _bad(**{field: lo - 1})
    _bad(**{field: hi + 1})


@pytest.mark.parametrize("field,lo,hi", [
    ("VAD_THRESHOLD", 0.0, 1.0),
    ("NO_SPEECH_THRESHOLD", 0.0, 1.0),
    ("LOG_PROB_THRESHOLD", -10.0, 0.0),
    ("COMPRESSION_RATIO_THRESHOLD", 0.0, 10.0),
    ("PATIENCE", 0.5, 5.0),
    ("LENGTH_PENALTY", 0.1, 5.0),
    ("REPETITION_PENALTY", 0.5, 5.0),
    ("PROMPT_RESET_ON_TEMPERATURE", 0.0, 1.0),
    ("LANGUAGE_DETECTION_THRESHOLD", 0.0, 1.0),
    ("HALLUCINATION_SILENCE_THRESHOLD", 0.0, 60.0),
    ("CAPTURE_RECORDINGS_SAMPLE_RATE", 0.0, 1.0),
    ("CAPTURE_RECORDINGS_MIN_DURATION_SEC", 0.0, 3600.0),
])
def test_float_bounds(field, lo, hi):
    _ok(**{field: lo})
    _ok(**{field: hi})
    _bad(**{field: lo - 0.1})
    _bad(**{field: hi + 0.1})


def test_capture_max_duration_min_is_0_1():
    # Asymmetric: MIN allows 0.0 but MAX requires ge=0.1.
    _ok(CAPTURE_RECORDINGS_MAX_DURATION_SEC=0.1)
    _bad(CAPTURE_RECORDINGS_MAX_DURATION_SEC=0.0)


# ---------------------------------------------------------------------------
# Patterns / literals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("val", ["large-v2", "org/name", "a", "A1_.-"])
def test_model_id_valid(val):
    assert _ok(DEFAULT_MODEL=val).DEFAULT_MODEL == val


@pytest.mark.parametrize("val", ["/leading", "trailing/", "has space", "", "a/b/c", "x" * 97])
def test_model_id_invalid(val):
    _bad(DEFAULT_MODEL=val)


@pytest.mark.parametrize("val", ["", "de", "en"])
def test_default_language_valid(val):
    assert _ok(DEFAULT_LANGUAGE=val).DEFAULT_LANGUAGE == val


@pytest.mark.parametrize("val", ["deu", "DE", "d", "d1"])
def test_default_language_invalid(val):
    _bad(DEFAULT_LANGUAGE=val)


def test_device_and_compute_literals():
    _ok(MODEL_DEVICE="cuda", MODEL_COMPUTE_TYPE="float16")
    _ok(MODEL_DEVICE="cpu", MODEL_COMPUTE_TYPE="int8")
    _bad(MODEL_DEVICE="rocm")
    _bad(MODEL_COMPUTE_TYPE="int4")


def test_compute_literal_excludes_int16():
    # int16 is valid for CONVERT_QUANTIZATION but NOT a ComputeLit.
    _bad(MODEL_COMPUTE_TYPE="int16")
    assert _ok(CONVERT_QUANTIZATION="int16").CONVERT_QUANTIZATION == "int16"


def test_server_log_level_literal():
    _ok(SERVER_LOG_LEVEL="debug")
    _bad(SERVER_LOG_LEVEL="verbose")


# ---------------------------------------------------------------------------
# CONVERT_QUANTIZATION / TEMPERATURE / SUPPRESS_TOKENS validators
# ---------------------------------------------------------------------------

def test_convert_quantisation():
    for v in ["float32", "int8_float16", "bfloat16"]:
        _ok(CONVERT_QUANTIZATION=v)
    assert _ok(CONVERT_QUANTIZATION="").CONVERT_QUANTIZATION == ""
    _bad(CONVERT_QUANTIZATION="int4")


def test_temperature():
    _ok(TEMPERATURE="")
    _ok(TEMPERATURE="0,0.2,0.4,0.6,0.8,1.0")
    _ok(TEMPERATURE="0.8,0.2")          # descending allowed (order not enforced)
    _ok(TEMPERATURE="0.5 , 0.5")        # whitespace tolerated
    _bad(TEMPERATURE="1.1")             # out of range
    _bad(TEMPERATURE="abc")             # not a float


def test_suppress_tokens():
    _ok(SUPPRESS_TOKENS="-1")
    _ok(SUPPRESS_TOKENS="1, 2 ,3")
    _ok(SUPPRESS_TOKENS="")
    _bad(SUPPRESS_TOKENS="1.5")
    _bad(SUPPRESS_TOKENS="x")


# ---------------------------------------------------------------------------
# Host validators (two different ones!)
# ---------------------------------------------------------------------------

def test_allowed_hosts_ip_cidr():
    _ok(ADMIN_WEBUI_ALLOWED_HOSTS=["127.0.0.1", "::1", "192.168.1.0/24"])
    _ok(USER_WEBUI_ALLOWED_HOSTS=["10.0.0.0/8", "0.0.0.0/0", "::/0"])
    _bad(ADMIN_WEBUI_ALLOWED_HOSTS=["not-an-ip"])
    _bad(USER_WEBUI_ALLOWED_HOSTS=["example.com"])  # hostname is not an IP/CIDR


def test_server_host_loose_charset():
    _ok(SERVER_HOST="0.0.0.0")
    _ok(SERVER_HOST="::")
    _ok(SERVER_HOST="my-host.local")
    _bad(SERVER_HOST="bad host")       # space rejected
    _bad(SERVER_HOST="has/slash")


# ---------------------------------------------------------------------------
# LOG_FILE path safety
# ---------------------------------------------------------------------------

def test_log_file_rejects_unc_and_traversal():
    _ok(LOG_FILE="logs/whisper.log")
    _bad(LOG_FILE="\\\\server\\share\\x.log")  # UNC
    _bad(LOG_FILE="//server/share/x.log")       # posix UNC-ish
    _bad(LOG_FILE="../etc/passwd")              # .. segment
    _bad(LOG_FILE="logs/../../x")               # windows-style .. caught too


# ---------------------------------------------------------------------------
# _cap_list (ALLOWED_MODELS / PRELOAD_MODELS)
# ---------------------------------------------------------------------------

def test_cap_list_over_1000():
    _bad(ALLOWED_MODELS=[f"m{i}" for i in range(1001)])
    _ok(ALLOWED_MODELS=[f"m{i}" for i in range(1000)])


# ---------------------------------------------------------------------------
# normalize_tags
# ---------------------------------------------------------------------------

def test_normalize_tags_basic():
    assert cs.normalize_tags(None) == []
    assert cs.normalize_tags([]) == []
    assert cs.normalize_tags(["B", "a", "a", " c "]) == ["a", "b", "c"]


def test_normalize_tags_drops_empty():
    assert cs.normalize_tags(["", "   ", "ok"]) == ["ok"]


def test_normalize_tags_rejects_bad():
    with pytest.raises(ValueError):
        cs.normalize_tags("notalist")
    with pytest.raises(ValueError):
        cs.normalize_tags([123])
    with pytest.raises(ValueError):
        cs.normalize_tags(["-leadinghyphen"])
    with pytest.raises(ValueError):
        cs.normalize_tags(["x" * 33])


# ---------------------------------------------------------------------------
# Pipeline rule validators (the parts not covered via save_factory_rules)
# ---------------------------------------------------------------------------

def _regex(name, pattern="x", replacement="y"):
    # A one-entry regex-list == a former single `regex` rule.
    return {"name": name, "label": name, "type": "regex-list",
            "entries": [{"pattern": pattern, "replacement": replacement}]}


def _terminal():
    return {"name": "trim-edges", "label": "Trim", "type": "terminal"}


def test_pipeline_callback_map_skips_pattern_validation():
    rule = {"name": "m", "label": "m", "type": "callback:map",
            "map": {"Komma": ","}}
    m = _ok(PIPELINE_RULES=[rule, _terminal()])
    assert m.PIPELINE_RULES[0].type == "callback:map"


def test_pipeline_bad_backref_reported():
    # Replacement \3 with one group -> re.sub raises -> "regex test failed".
    with pytest.raises(ValidationError) as ei:
        _ok(PIPELINE_RULES=[_regex("b", pattern="(a)", replacement=r"\3"), _terminal()])
    assert "regex test failed" in str(ei.value)


def test_pipeline_duplicate_slug():
    _bad(PIPELINE_RULES=[_regex("dup"), _regex("dup"), _terminal()])


def test_pipeline_terminal_must_be_last():
    _bad(PIPELINE_RULES=[_terminal(), _regex("after")])


def test_regex_list_validates_and_keeps_order():
    rule = {"name": "rl", "label": "RL", "type": "regex-list",
            "entries": [{"pattern": "a", "replacement": "b"},
                        {"pattern": "b", "replacement": "c", "label": "x", "note": "n"}]}
    m = _ok(PIPELINE_RULES=[rule, _terminal()])
    assert m.PIPELINE_RULES[0].type == "regex-list"
    assert [e.pattern for e in m.PIPELINE_RULES[0].entries] == ["a", "b"]


def test_regex_list_requires_pattern_per_entry():
    # `pattern` is required on every entry.
    _bad(PIPELINE_RULES=[{"name": "rl", "label": "RL", "type": "regex-list",
                          "entries": [{"replacement": "b"}]}, _terminal()])


def test_regex_list_entry_extra_forbid():
    # Unknown per-entry key rejected (RegexListEntry has extra="forbid").
    _bad(PIPELINE_RULES=[{"name": "rl", "label": "RL", "type": "regex-list",
                          "entries": [{"pattern": "a", "bogus": 1}]}, _terminal()])


def test_regex_list_optional_fields_default_and_survive_exclude_none():
    m = _ok(PIPELINE_RULES=[{"name": "rl", "label": "RL", "type": "regex-list",
                             "entries": [{"pattern": "a"}]}, _terminal()])
    e = m.PIPELINE_RULES[0].entries[0]
    assert (e.replacement, e.label, e.note) == ("", "", "")
    # exclude_none must KEEP the "" defaults (they are "" not None).
    dumped = m.model_dump(exclude_none=True, mode="json")["PIPELINE_RULES"][0]["entries"][0]
    assert dumped == {"pattern": "a", "replacement": "", "label": "", "note": ""}


def test_regex_list_entry_bad_regex_reports_index():
    with pytest.raises(ValidationError) as ei:
        _ok(PIPELINE_RULES=[{"name": "rl", "label": "RL", "type": "regex-list",
                             "entries": [{"pattern": "("}]}, _terminal()])
    assert "entry 0" in str(ei.value)


def test_map_meta_pruned_to_map_keys():
    rule = {"name": "m", "label": "m", "type": "callback:map",
            "map": {"Komma": ","}, "map_meta": {"Komma": 5, "ghost": 9}}
    m = _ok(PIPELINE_RULES=[rule, _terminal()])
    assert m.PIPELINE_RULES[0].map_meta == {"Komma": 5}


# NOTE: the validator's "took > 2 s" catastrophic-backtracking branch is
# deliberately NOT tested here. Triggering it requires a pattern that never
# terminates (e.g. (.+)+# against the validator's fixed ~1 KB fixture); the
# validator abandons the work via a daemon thread join(timeout=2.0), but that
# daemon thread then runs the runaway regex forever, pinning a CPU core and
# contending the GIL for the rest of the pytest session. The error branch is
# covered by test_pipeline_bad_backref_reported above.


# ---------------------------------------------------------------------------
# ModelOverride validators
# ---------------------------------------------------------------------------

def test_model_override_include_exclude_overlap():
    with pytest.raises(ValidationError):
        cs.ModelOverride.model_validate({
            "PIPELINE_RULES_EXCLUDE": ["a"],
            "PIPELINE_RULES_INCLUDE": ["a"],
        })


def test_model_override_bounds_inherit_global():
    cs.ModelOverride.model_validate({"BEAM_SIZE": 20})
    with pytest.raises(ValidationError):
        cs.ModelOverride.model_validate({"BEAM_SIZE": 21})


def test_admin_extra_forbid_on_override():
    with pytest.raises(ValidationError):
        cs.ModelOverride.model_validate({"NONSENSE": 1})


# ---------------------------------------------------------------------------
# Model-level cross-field validators (only fire when both keys present)
# ---------------------------------------------------------------------------

def test_no_orphan_overrides_fires_only_with_both():
    # Both present + non-empty allowlist + orphan -> reject.
    _bad(ALLOWED_MODELS=["a"], MODEL_OVERRIDES={"b": {"BEAM_SIZE": 5}})
    # Empty allowlist = anything goes -> skip check.
    _ok(ALLOWED_MODELS=[], MODEL_OVERRIDES={"b": {"BEAM_SIZE": 5}})
    # Only overrides present -> cross-check skipped.
    _ok(MODEL_OVERRIDES={"b": {"BEAM_SIZE": 5}})
    # Override model in allowlist -> ok.
    _ok(ALLOWED_MODELS=["a", "b"], MODEL_OVERRIDES={"b": {"BEAM_SIZE": 5}})


def test_pipeline_rule_slugs_cross_check():
    rules = [_regex("known"), _terminal()]
    # Unknown slug in a per-model EXCLUDE -> reject (both keys present).
    _bad(PIPELINE_RULES=rules,
         MODEL_OVERRIDES={"m": {"PIPELINE_RULES_EXCLUDE": ["bogus"]}})
    # Known slug -> ok.
    _ok(PIPELINE_RULES=rules,
        MODEL_OVERRIDES={"m": {"PIPELINE_RULES_EXCLUDE": ["known"]}})
    # Only MODEL_OVERRIDES present -> skipped.
    _ok(MODEL_OVERRIDES={"m": {"PIPELINE_RULES_EXCLUDE": ["bogus"]}})


# ---------------------------------------------------------------------------
# load_overrides / save_overrides
# ---------------------------------------------------------------------------

def test_load_overrides_missing_returns_empty(tmp_path):
    assert cs.load_overrides(str(tmp_path / "nope.json")) == {}


def test_load_overrides_corrupt_returns_empty(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{ not json", encoding="utf-8")
    assert cs.load_overrides(str(p)) == {}


def test_load_overrides_non_object_returns_empty(tmp_path):
    p = tmp_path / "arr.json"
    p.write_text("[1,2,3]", encoding="utf-8")
    assert cs.load_overrides(str(p)) == {}


def test_load_overrides_unknown_key_ignored_whole_file(tmp_path):
    p = tmp_path / "u.json"
    p.write_text(json.dumps({"BEAM_SIZE": 5, "BOGUS": 1}), encoding="utf-8")
    # Whole file is rejected on validation failure -> {}.
    assert cs.load_overrides(str(p)) == {}


def test_load_overrides_coerces_allowed_models_to_set(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"ALLOWED_MODELS": ["a", "b"]}), encoding="utf-8")
    out = cs.load_overrides(str(p))
    assert isinstance(out["ALLOWED_MODELS"], set)
    assert out["ALLOWED_MODELS"] == {"a", "b"}


def test_save_overrides_roundtrip_and_merge(tmp_path):
    p = str(tmp_path / "config.local.json")
    changed = cs.save_overrides({"BEAM_SIZE": 5}, p)
    assert changed == {"BEAM_SIZE": 5}
    # Merge: a second partial save keeps the first field.
    cs.save_overrides({"BEST_OF": 3}, p)
    on_disk = json.loads(open(p, encoding="utf-8").read())
    assert on_disk["BEAM_SIZE"] == 5 and on_disk["BEST_OF"] == 3


def test_save_overrides_none_removes(tmp_path):
    p = str(tmp_path / "config.local.json")
    cs.save_overrides({"BEAM_SIZE": 5, "BEST_OF": 3}, p)
    changed = cs.save_overrides({"BEAM_SIZE": None}, p)
    assert "BEAM_SIZE" in changed and changed["BEAM_SIZE"] is None
    on_disk = json.loads(open(p, encoding="utf-8").read())
    assert "BEAM_SIZE" not in on_disk and on_disk["BEST_OF"] == 3


def test_save_overrides_changed_excludes_unchanged(tmp_path):
    p = str(tmp_path / "config.local.json")
    cs.save_overrides({"BEAM_SIZE": 5}, p)
    # Re-saving the same value reports no change for it.
    changed = cs.save_overrides({"BEAM_SIZE": 5}, p)
    assert changed == {}


def test_save_factory_rules_preserves_sibling_defaults(tmp_path):
    # config.json now holds ALL factory defaults, not just PIPELINE_RULES, so a
    # rules "Promote to factory" must read-modify-write — not clobber the sibling
    # scalar defaults (the old whole-file replace would wipe every other value).
    p = str(tmp_path / "config.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"schema_version": 1, "DEFAULT_MODEL": "keep-me", "BEST_OF": 7,
                   "PIPELINE_RULES": [{"name": "t", "label": "T", "type": "terminal"}]}, f)
    cs.save_factory_rules([{"name": "trim", "label": "Trim", "type": "terminal"}], p)
    on_disk = json.loads(open(p, encoding="utf-8").read())
    assert on_disk["DEFAULT_MODEL"] == "keep-me"     # sibling default preserved
    assert on_disk["BEST_OF"] == 7                    # sibling default preserved
    assert [r["name"] for r in on_disk["PIPELINE_RULES"]] == ["trim"]   # rules updated


def test_sample_sizing_absent_field_uses_baseline_not_live_override(monkeypatch):
    # Regression: _validate_sample_sizing must fall back to config._BASELINE
    # (the immutable in-repo default) for an absent field, NOT the live config
    # attribute. The live attribute already carries any applied override, so at
    # save time (server running) it would be the OLD override while at load
    # time (config import) it is the bare default — that asymmetry let a save
    # pass validation, then the next restart's load fail it and silently drop
    # EVERY override on disk.
    import config as _cfg

    # Simulate a server running with a previously-applied TARGET override of 5.
    monkeypatch.setattr(_cfg, "CAPTURES_PROPOSER_TARGET_S", 5.0, raising=False)
    # _BASELINE keeps the real in-repo default (26.0), which exceeds MAX=6.
    assert _cfg._BASELINE["CAPTURES_PROPOSER_TARGET_S"] > 6.0

    # Removing TARGET reverts it to the 26.0 baseline → 1 ≤ 26 ≤ 6 is false.
    # Must reject regardless of the stale live value of 5.0.
    _bad(CAPTURES_SAMPLE_MIN_DURATION_S=1.0, CAPTURES_SAMPLE_MAX_DURATION_S=6.0)


def test_save_overrides_corrupt_existing_rewrites(tmp_path):
    p = str(tmp_path / "config.local.json")
    open(p, "w", encoding="utf-8").write("{ corrupt")
    cs.save_overrides({"BEAM_SIZE": 7}, p)
    assert json.loads(open(p, encoding="utf-8").read())["BEAM_SIZE"] == 7


def test_save_overrides_invalid_raises(tmp_path):
    p = str(tmp_path / "config.local.json")
    with pytest.raises(ValidationError):
        cs.save_overrides({"BEAM_SIZE": 999}, p)
    assert not os.path.exists(p)  # nothing written


# ---------------------------------------------------------------------------
# _atomic_write_json
# ---------------------------------------------------------------------------

def test_atomic_write_unicode(tmp_path):
    p = str(tmp_path / "u.json")
    cs._atomic_write_json({"k": "Müller"}, p, sort_keys=True, tmp_prefix=".t")
    assert json.loads(open(p, encoding="utf-8").read())["k"] == "Müller"
    # ensure_ascii=False keeps the literal char on disk.
    assert "Müller" in open(p, encoding="utf-8").read()


def test_atomic_write_retries_then_succeeds(tmp_path, monkeypatch):
    p = str(tmp_path / "r.json")
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("AV lock")
        return real_replace(src, dst)

    monkeypatch.setattr(cs.os, "replace", flaky)
    monkeypatch.setattr(cs.time, "sleep", lambda *_: None)
    cs._atomic_write_json({"ok": 1}, p, sort_keys=True, tmp_prefix=".t")
    assert calls["n"] == 3
    assert json.loads(open(p, encoding="utf-8").read()) == {"ok": 1}


def test_atomic_write_gives_up_after_retries(tmp_path, monkeypatch):
    p = str(tmp_path / "x.json")

    def always_fail(src, dst):
        raise PermissionError("locked")

    monkeypatch.setattr(cs.os, "replace", always_fail)
    monkeypatch.setattr(cs.time, "sleep", lambda *_: None)
    with pytest.raises(PermissionError):
        cs._atomic_write_json({"ok": 1}, p, sort_keys=True, tmp_prefix=".t")
    # The temp file is cleaned up in finally; only the (untouched) dir remains.
    leftovers = [f for f in os.listdir(tmp_path) if f != "x.json"]
    assert leftovers == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_pipeline_rule_tags_union():
    rules = [
        {"tags": ["b", "a"]},
        {"tags": ["a", "c"]},
        {"tags": []},
        "not-a-dict",
    ]
    assert cs.pipeline_rule_tags(rules) == ["a", "b", "c"]


def test_env_pinned_fields(monkeypatch):
    monkeypatch.delenv("WHISPER_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_BEAM_SIZE", raising=False)
    # A mapped field whose env var is set IS reported as pinned...
    monkeypatch.setenv("WHISPER_DEFAULT_MODEL", "large-v3")
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "5")
    # ...but a WHISPER_* var with no AdminConfig field (not in the mapping) is not.
    monkeypatch.setenv("WHISPER_USAGE_DB", "/tmp/u.sqlite3")
    pinned = cs.env_pinned_fields()
    assert pinned.get("DEFAULT_MODEL") == "WHISPER_DEFAULT_MODEL"
    assert pinned.get("BEAM_SIZE") == "WHISPER_BEAM_SIZE"
    assert "USAGE_DB" not in pinned  # USAGE_DB is not an editable AdminConfig field


def test_format_validation_errors_shape():
    try:
        cs.AdminConfig.model_validate({"BEAM_SIZE": 999})
    except ValidationError as e:
        out = cs.format_validation_errors(e)
        assert isinstance(out, list) and out
        assert set(out[0]) == {"loc", "msg"}
        assert "BEAM_SIZE" in out[0]["loc"]
