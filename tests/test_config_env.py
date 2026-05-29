"""Tests for config.py's environment-variable layer: the coercion helpers,
the per-model override decoder, and the stdlib factory-rules loader.

The helper functions read os.environ at call time, so they're tested directly
with monkeypatched env vars. The module-level per-model scanner is exercised
once via a guarded importlib.reload (restored in a finally).
"""

import importlib
import json

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
# _load_factory_pipeline_rules (stdlib loader, distinct from config_store's)
# ---------------------------------------------------------------------------

def test_factory_loader_reads_committed_config():
    rules = config._load_factory_pipeline_rules()
    assert isinstance(rules, list) and len(rules) >= 2


def test_factory_loader_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        config._load_factory_pipeline_rules()


def test_factory_loader_corrupt_raises(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        config._load_factory_pipeline_rules()


def test_factory_loader_missing_key_raises(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"schema_version": 1}),
                                          encoding="utf-8")
    monkeypatch.setattr(config, "_REPO_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        config._load_factory_pipeline_rules()


# ---------------------------------------------------------------------------
# Precedence / asymmetry documentation tests
# ---------------------------------------------------------------------------

def test_some_admin_fields_have_no_env_reader():
    # These AdminConfig fields are editable via local.json but have NO global
    # WHISPER_* reader in config.py (documented asymmetry). Setting the
    # name-matching env var must not change them after reload.
    import config_store as cs
    no_env = {"BEAM_SIZE", "BEST_OF", "VAD_FILTER", "DEFAULT_LANGUAGE",
              "SERVER_PORT", "LOG_MAX_BYTES"}
    for f in no_env:
        assert f in cs.AdminConfig.model_fields
        assert f not in cs.ENV_VAR_MAPPING
