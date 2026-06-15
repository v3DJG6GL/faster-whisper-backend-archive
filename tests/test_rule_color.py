"""The `color` card-tint token on pipeline rules (config_store._RuleBase).

Accepts the curated palette + "", and normalises unknown/junk to "" rather than
raising — forgiving by design: a cosmetic value must never trip load_overrides,
which drops ALL overrides on any validation error (the lockout failure mode)."""

import pytest

import config_store as cs


def _rule(**kw):
    base = {"name": "r", "label": "R", "type": "regex-list", "entries": []}
    base.update(kw)
    return cs.RegexListRule.model_validate(base)


@pytest.mark.parametrize("tok", list(cs.RULE_CARD_COLORS) + [""])
def test_color_accepts_palette_and_empty(tok):
    assert _rule(color=tok).color == tok


def test_color_default_is_empty():
    assert _rule().color == ""


@pytest.mark.parametrize("junk", ["NEONpuke", "#ff0000", "rgb(1,2,3)", "12", "   "])
def test_color_junk_normalises_to_empty(junk):
    # MUST NOT raise — a bad cosmetic token would otherwise nuke every override.
    assert _rule(color=junk).color == ""


def test_color_case_and_space_insensitive():
    assert _rule(color="  TEAL ").color == "teal"


def test_color_survives_round_trip():
    dumped = _rule(color="blue").model_dump()
    assert dumped["color"] == "blue"
    assert cs.RegexListRule.model_validate(dumped).color == "blue"


def test_color_empty_survives_exclude_none():
    # The export/env path uses model_dump(exclude_none=True); color defaults to
    # "" (never None) so it survives that round-trip.
    dumped = _rule(color="").model_dump(exclude_none=True)
    assert dumped.get("color") == ""


def test_color_on_every_rule_type():
    # color lives on _RuleBase, so every discriminated-union member carries it.
    from pydantic import TypeAdapter

    ta = TypeAdapter(cs.PipelineRule)
    for payload in (
        {"name": "w", "label": "W", "type": "callback:lowercase-wordlist",
         "pattern": "x", "wordlist": ["foo"]},
        {"name": "m", "label": "M", "type": "callback:map", "map": {"a": "b"}},
        {"name": "d", "label": "D", "type": "callback:dedup", "pattern": "x"},
        {"name": "t", "label": "T", "type": "terminal"},
    ):
        assert ta.validate_python({**payload, "color": "amber"}).color == "amber"
