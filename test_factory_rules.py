"""Tests for the committed factory pipeline-rules layer (config.json).

Covers config_store.load_factory_rules / save_factory_rules: validation,
round-trip, the `note` field, the terminal-rule invariant, and the
fail-fast behaviour on a missing/corrupt file.

Runnable two ways:
    pytest test_factory_rules.py
    python  test_factory_rules.py        (no pytest needed)

Only depends on pydantic (same as config_store) — not the full app stack.
"""

import json
import os
import tempfile

from pydantic import ValidationError

import config_store as cs


def _regex_rule(name, pattern="x", replacement="y", **kw):
    r = {"name": name, "label": name, "type": "regex",
         "pattern": pattern, "replacement": replacement}
    r.update(kw)
    return r


def _terminal():
    return {"name": "trim-edges", "label": "Trim edges", "type": "terminal"}


def _tmp_path():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)          # save_factory_rules creates it
    return path


def test_load_real_config_json():
    """The committed config.json loads, validates, and ends with a terminal."""
    rules = cs.load_factory_rules()
    assert len(rules) >= 2
    assert rules[-1]["type"] == "terminal"
    assert sum(1 for r in rules if r["type"] == "terminal") == 1
    assert all(r.get("note") for r in rules), "every committed rule has a note"


def test_save_load_roundtrip():
    """save_factory_rules → load_factory_rules is a stable round-trip."""
    path = _tmp_path()
    try:
        rules = [_regex_rule("alpha"), _regex_rule("beta"), _terminal()]
        saved = cs.save_factory_rules(rules, path=path)
        loaded = cs.load_factory_rules(path=path)
        assert loaded == saved
        assert [r["name"] for r in loaded] == ["alpha", "beta", "trim-edges"]
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_wrapped_object_shape():
    """The on-disk file is {schema_version, PIPELINE_RULES}, not a bare array."""
    path = _tmp_path()
    try:
        cs.save_factory_rules([_regex_rule("only"), _terminal()], path=path)
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        assert raw["schema_version"] == 1
        assert isinstance(raw["PIPELINE_RULES"], list)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_note_round_trips():
    """The `note` field survives a save/load cycle."""
    path = _tmp_path()
    try:
        why = "explains why this rule exists"
        cs.save_factory_rules([_regex_rule("noted", note=why), _terminal()], path=path)
        loaded = cs.load_factory_rules(path=path)
        assert loaded[0]["note"] == why
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_save_normalizes_seeded():
    """save_factory_rules forces seeded=True on every written rule — a rule in
    the committed factory file is a factory default by definition."""
    path = _tmp_path()
    try:
        rules = [_regex_rule("a", seeded=False), _regex_rule("b"), _terminal()]
        saved = cs.save_factory_rules(rules, path=path)
        assert all(r["seeded"] is True for r in saved), saved
        loaded = cs.load_factory_rules(path=path)
        assert all(r["seeded"] is True for r in loaded), loaded
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_bad_regex_rejected():
    """An uncompilable pattern is rejected before anything is written."""
    path = _tmp_path()
    try:
        bad = [_regex_rule("broken", pattern="("), _terminal()]
        try:
            cs.save_factory_rules(bad, path=path)
            assert False, "expected ValidationError for an invalid regex"
        except ValidationError:
            pass
        assert not os.path.exists(path), "nothing should be written on a bad save"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_terminal_must_be_last():
    """The terminal rule must be the final entry."""
    path = _tmp_path()
    try:
        try:
            cs.save_factory_rules([_terminal(), _regex_rule("after")], path=path)
            assert False, "expected ValidationError for a non-last terminal"
        except ValidationError:
            pass
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_duplicate_slug_rejected():
    """Two rules with the same name are rejected."""
    path = _tmp_path()
    try:
        try:
            cs.save_factory_rules(
                [_regex_rule("dup"), _regex_rule("dup"), _terminal()], path=path)
            assert False, "expected ValidationError for a duplicate slug"
        except ValidationError:
            pass
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_missing_file_raises():
    """load_factory_rules fails fast on a missing file (config.json is required)."""
    try:
        cs.load_factory_rules(path="/nonexistent/does-not-exist.json")
        assert False, "expected RuntimeError for a missing file"
    except RuntimeError as e:
        assert "git checkout config.json" in str(e)


def test_corrupt_json_raises():
    """load_factory_rules fails fast on malformed JSON."""
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        os.write(fd, b"{ not valid json")
        os.close(fd)
        try:
            cs.load_factory_rules(path=path)
            assert False, "expected RuntimeError for malformed JSON"
        except RuntimeError:
            pass
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_missing_pipeline_rules_key_raises():
    """A JSON object without a PIPELINE_RULES key is rejected."""
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        os.write(fd, b'{"schema_version": 1}')
        os.close(fd)
        try:
            cs.load_factory_rules(path=path)
            assert False, "expected RuntimeError for a missing PIPELINE_RULES key"
        except RuntimeError:
            pass
    finally:
        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    tests = sorted(
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
