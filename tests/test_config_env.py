"""Tests for config.py's environment-variable layer: the coercion helpers,
the per-model override decoder, and the stdlib factory-rules loader.

The helper functions read os.environ at call time, so they're tested directly
with monkeypatched env vars. The module-level per-model scanner is exercised
once via a guarded importlib.reload (restored in a finally).
"""

import importlib
import json
import os

import pytest

import config


# ---------------------------------------------------------------------------
# Scalar coercion helpers
# ---------------------------------------------------------------------------

def test_truthy():
    for v in ["1", "true", "TRUE", "Yes", "on", "  on  "]:
        assert config._truthy(v) is True
    for v in ["0", "false", "no", "off", "", "maybe"]:
        assert config._truthy(v) is False


def test_env_int(monkeypatch):
    monkeypatch.setenv("X_INT", "42")
    assert config._env_int("X_INT", 7) == 42
    monkeypatch.setenv("X_INT", "  ")
    assert config._env_int("X_INT", 7) == 7          # blank -> current
    monkeypatch.setenv("X_INT", "notanint")
    assert config._env_int("X_INT", 7) == 7          # invalid -> current
    monkeypatch.delenv("X_INT")
    assert config._env_int("X_INT", 7) == 7          # unset -> current


def test_env_float(monkeypatch):
    monkeypatch.setenv("X_F", "1.5")
    assert config._env_float("X_F", 0.0) == 1.5
    monkeypatch.setenv("X_F", "bad")
    assert config._env_float("X_F", 0.25) == 0.25


def test_env_bool(monkeypatch):
    monkeypatch.setenv("X_B", "yes")
    assert config._env_bool("X_B", False) is True
    monkeypatch.setenv("X_B", "")
    assert config._env_bool("X_B", True) is True     # blank -> current
    monkeypatch.delenv("X_B")
    assert config._env_bool("X_B", True) is True


def test_env_str(monkeypatch):
    monkeypatch.setenv("X_S", "  hi ")
    assert config._env_str("X_S", "cur") == "hi"
    monkeypatch.setenv("X_S", "   ")
    assert config._env_str("X_S", "cur") == "cur"    # blank -> current


def test_env_str_or_none(monkeypatch):
    # explicit empty string -> None (disable)
    monkeypatch.setenv("X_SON", "")
    assert config._env_str_or_none("X_SON", "cur") is None
    monkeypatch.setenv("X_SON", "val")
    assert config._env_str_or_none("X_SON", "cur") == "val"
    monkeypatch.delenv("X_SON")
    assert config._env_str_or_none("X_SON", "cur") == "cur"


def test_env_str_passthrough(monkeypatch):
    # empty string is preserved as a real value (NOT None / current)
    monkeypatch.setenv("X_SP", "")
    assert config._env_str_passthrough("X_SP", "cur") == ""
    monkeypatch.delenv("X_SP")
    assert config._env_str_passthrough("X_SP", "cur") == "cur"


def test_env_csv_list(monkeypatch):
    monkeypatch.setenv("X_L", "a, b ,,c")
    assert config._env_csv_list("X_L", ["z"]) == ["a", "b", "c"]
    monkeypatch.setenv("X_L", "")          # explicit empty -> empty list
    assert config._env_csv_list("X_L", ["z"]) == []
    monkeypatch.delenv("X_L")
    assert config._env_csv_list("X_L", ["z"]) == ["z"]   # unset -> current


# ---------------------------------------------------------------------------
# Per-model override decode helpers
# ---------------------------------------------------------------------------

def test_decode_model_id():
    assert config._decode_model_id("org__SLASH__name__DOT__ct2") == "org/name.ct2"
    assert config._decode_model_id("plain") == "plain"


def test_coerce_override_value_types():
    assert config._coerce_override_value("VAD_FILTER", "true") is True
    assert config._coerce_override_value("BEAM_SIZE", "7") == 7
    assert config._coerce_override_value("BEAM_SIZE", "x") == "x"   # invalid -> raw
    assert config._coerce_override_value("VAD_THRESHOLD", "0.5") == 0.5
    assert config._coerce_override_value("PIPELINE_RULES_EXCLUDE", "a, b ,c") == ["a", "b", "c"]
    # TEMPERATURE is unclassified -> raw string passthrough
    assert config._coerce_override_value("TEMPERATURE", "0,0.2") == "0,0.2"


def test_per_model_env_scanner_end_to_end(monkeypatch):
    # WHISPER_MODEL_OVERRIDE__<encoded id>__<FIELD> populates MODEL_OVERRIDES.
    # NOTE: the encoded id is UPPERCASE on purpose. Windows normalises
    # os.environ keys to uppercase, so a lowercase id in the var NAME would not
    # round-trip there; an uppercase id is case-stable on every platform and
    # still exercises the right-to-left "__" boundary scanner + _decode_model_id.
    monkeypatch.setenv(
        "WHISPER_MODEL_OVERRIDE__ORG__SLASH__NAME__DOT__CT2__BEAM_SIZE", "7"
    )
    try:
        importlib.reload(config)
        assert config.MODEL_OVERRIDES.get("ORG/NAME.CT2", {}).get("BEAM_SIZE") == 7
    finally:
        monkeypatch.undo()
        importlib.reload(config)  # restore from the clean environment


# ---------------------------------------------------------------------------
# _load_defaults (stdlib loader: config.json is the single source of factory
# defaults; config.py reads every value from it at import)
# ---------------------------------------------------------------------------

def _write_cfg(tmp_path, **extra):
    """Write a minimal-but-valid config.json into tmp_path and return its path."""
    data = {"schema_version": 1,
            "PIPELINE_RULES": [{"name": "trim", "label": "Trim", "type": "terminal"}],
            **extra}
    (tmp_path / "config.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


def test_load_defaults_reads_committed_config():
    d = config._load_defaults()
    # Returns ALL defaults, not just the rules: scalars + the rules list.
    assert isinstance(d, dict)
    assert isinstance(d["PIPELINE_RULES"], list) and len(d["PIPELINE_RULES"]) >= 2
    assert d["DEFAULT_MODEL"] == "large-v2"
    assert "schema_version" not in d          # stripped — it's metadata, not a setting


def test_load_defaults_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        config._load_defaults()


def test_load_defaults_corrupt_raises(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        config._load_defaults()


def test_load_defaults_missing_rules_raises(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"schema_version": 1}),
                                          encoding="utf-8")
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        config._load_defaults()


def test_load_defaults_is_the_source(monkeypatch, tmp_path):
    # A value placed in config.json is what _load_defaults returns — proving
    # config.json (not config.py) is the source of truth.
    _write_cfg(tmp_path, BEST_OF=9, DEFAULT_MODEL="my-model")
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    d = config._load_defaults()
    assert d["BEST_OF"] == 9
    assert d["DEFAULT_MODEL"] == "my-model"


def test_load_defaults_resolves_repo_dir_placeholder(monkeypatch, tmp_path):
    _write_cfg(tmp_path, LOG_FILE="{REPO_DIR}/logs/whisper.log")
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    d = config._load_defaults()
    assert d["LOG_FILE"] == os.path.normpath(os.path.join(str(tmp_path), "logs/whisper.log"))
    assert "{REPO_DIR}" not in d["LOG_FILE"]


def test_load_defaults_coerces_set_fields(monkeypatch, tmp_path):
    _write_cfg(tmp_path,
               ALLOWED_MODELS=["large-v2", "large-v3"],
               CAPTURES_PIPELINE_RULES_EXCLUDE=["dictation-map"])
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    d = config._load_defaults()
    assert d["ALLOWED_MODELS"] == {"large-v2", "large-v3"}
    assert isinstance(d["ALLOWED_MODELS"], set)
    assert isinstance(d["CAPTURES_PIPELINE_RULES_EXCLUDE"], set)


def test_baseline_comes_from_config_json():
    # _BASELINE (what "↺ Reset to default" reverts to) must equal the values in
    # config.json, with the same set-coercion + {REPO_DIR} resolution applied.
    # Locks "config.json is the single source of truth for factory defaults".
    expected = config._load_defaults()
    for k, v in expected.items():
        assert config._BASELINE[k] == v, k
    assert set(config._BASELINE) == set(expected)


def test_env_float_or_none(monkeypatch):
    # explicit empty string -> None (disable the check)
    monkeypatch.setenv("X_FON", "")
    assert config._env_float_or_none("X_FON", 0.6) is None
    monkeypatch.setenv("X_FON", "0.3")
    assert config._env_float_or_none("X_FON", 0.6) == 0.3
    monkeypatch.setenv("X_FON", "bad")
    assert config._env_float_or_none("X_FON", 0.6) == 0.6   # invalid -> current
    monkeypatch.delenv("X_FON")
    assert config._env_float_or_none("X_FON", 0.6) == 0.6   # unset -> current


# ---------------------------------------------------------------------------
# Single-source-of-truth invariant: every AdminConfig field is env-configurable
# ---------------------------------------------------------------------------

def test_every_admin_field_is_env_mapped():
    # ENV_VAR_MAPPING is the source of truth driving config.py's schema loop,
    # the WebUI "env-pinned" badge, and env > GUI precedence. Every editable
    # AdminConfig field MUST be present (and vice-versa) or it silently loses
    # env-configurability / badging. This guards against future drift.
    import config_store as cs
    fields = set(cs.AdminConfig.model_fields)
    mapped = set(cs.ENV_VAR_MAPPING)
    assert fields == mapped, (
        f"missing from ENV_VAR_MAPPING: {sorted(fields - mapped)}; "
        f"mapping entries not in schema: {sorted(mapped - fields)}")


def test_env_var_names_are_unique():
    import config_store as cs
    names = list(cs.ENV_VAR_MAPPING.values())
    assert len(names) == len(set(names)), "duplicate WHISPER_* env var names"


# ---------------------------------------------------------------------------
# End-to-end schema-driven env application (importlib.reload)
# ---------------------------------------------------------------------------

def _reload_with_env(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    importlib.reload(config)


def test_scalar_env_overrides_apply(monkeypatch):
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_BEAM_SIZE="3",
            WHISPER_SERVER_PORT="8123",
            WHISPER_VAD_FILTER="0",
            WHISPER_MODEL_DEVICE="cpu",
        )
        assert config.BEAM_SIZE == 3
        assert config.SERVER_PORT == 8123
        assert config.VAD_FILTER is False
        assert config.MODEL_DEVICE == "cpu"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_optional_threshold_empty_disables(monkeypatch):
    try:
        _reload_with_env(monkeypatch, WHISPER_NO_SPEECH_THRESHOLD="")
        assert config.NO_SPEECH_THRESHOLD is None
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_default_language_empty_means_autodetect(monkeypatch):
    # DEFAULT_LANGUAGE="" is a meaningful literal (auto-detect), not None.
    try:
        _reload_with_env(monkeypatch, WHISPER_DEFAULT_LANGUAGE="")
        assert config.DEFAULT_LANGUAGE == ""
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_set_typed_special_cases_stay_sets(monkeypatch):
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_ALLOWED_MODELS="a,b",
            WHISPER_CAPTURES_PIPELINE_RULES_EXCLUDE="x,y",
        )
        assert config.ALLOWED_MODELS == {"a", "b"}
        assert config.CAPTURES_PIPELINE_RULES_EXCLUDE == {"x", "y"}
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_json_model_overrides_env(monkeypatch):
    # JSON blob validates + normalises to plain dicts; per-model var merges atop.
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_MODEL_OVERRIDES='{"large-v2": {"BEAM_SIZE": 4}}',
            # Uppercase id on purpose: Windows normalises os.environ keys to
            # uppercase, so a lowercase id in the var NAME wouldn't round-trip
            # there (same reason as test_per_model_env_scanner_end_to_end).
            WHISPER_MODEL_OVERRIDE__LARGE__DOT__V3__BEAM_SIZE="7",
        )
        assert config.MODEL_OVERRIDES["large-v2"] == {"BEAM_SIZE": 4}
        assert isinstance(config.MODEL_OVERRIDES["large-v2"], dict)
        assert config.MODEL_OVERRIDES["LARGE.V3"]["BEAM_SIZE"] == 7
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_json_pipeline_rules_env(monkeypatch):
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_PIPELINE_RULES='[{"name": "x", "label": "X", "type": "terminal"}]',
        )
        assert isinstance(config.PIPELINE_RULES, list)
        assert config.PIPELINE_RULES[0]["name"] == "x"
        assert isinstance(config.PIPELINE_RULES[0], dict)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_invalid_json_keeps_default_and_warns(monkeypatch):
    try:
        _reload_with_env(monkeypatch, WHISPER_PIPELINE_RULES="not json")
        # factory rules remain in place
        assert len(config.PIPELINE_RULES) >= 2
        assert any("WHISPER_PIPELINE_RULES" in m for m in config._ENV_WARNINGS)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_bad_scalar_records_warning(monkeypatch):
    try:
        _reload_with_env(monkeypatch, WHISPER_BEAM_SIZE="ten")
        assert config.BEAM_SIZE == 10   # default kept
        assert any("WHISPER_BEAM_SIZE" in m for m in config._ENV_WARNINGS)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_secret_file_indirection(monkeypatch, tmp_path):
    secret = tmp_path / "key"
    secret.write_text("  sk-from-file  \n", encoding="utf-8")
    try:
        _reload_with_env(
            monkeypatch,
            WHISPER_BOOTSTRAP_ADMIN_KEY_FILE=str(secret),
        )
        assert config.BOOTSTRAP_ADMIN_KEY == "sk-from-file"
    finally:
        monkeypatch.undo()
        # The *_FILE prepass writes the resolved secret straight into os.environ
        # (so both the explicit reader and the schema loop see it); monkeypatch
        # can't undo that, so clear it before the restoring reload.
        os.environ.pop("WHISPER_BOOTSTRAP_ADMIN_KEY", None)
        importlib.reload(config)
