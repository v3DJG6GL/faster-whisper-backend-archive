"""Trace step numbering: the `/logs` + `/quick-config` pipeline step number must
match the /settings/pipeline card position — `#P` for a single-compile rule and
`#P.S` for a regex-list entry (S = the entry's row within the card). Regression
guard for the flat-index bug where each expanded regex-list entry was counted as
its own top-level step. Exercises the real `main` pipeline via the app_module
fixture (which reloads main per test, so cfg/_COMPILED_RULES/_TERMINAL_CARD_NO
mutations are isolated)."""


def _term():
    return {"name": "trim-edges", "label": "Trim edges (always-last)", "type": "terminal"}


def _regex_list(name, label, entries, enabled=True):
    return {"name": name, "label": label, "type": "regex-list",
            "enabled": enabled, "entries": entries}


def _set(app_module, rules):
    app_module.cfg.PIPELINE_RULES = rules
    app_module.rebuild_caches()


def _labels(app_module, text, **kw):
    """Run the pipeline with tracing and return the per-step label strings."""
    trace: list = []
    app_module._postprocess_text(text, model_name="", trace=trace, **kw)
    return [s[0] for s in trace]


def test_rule_ordinal_helper(app_module):
    assert app_module._rule_ordinal(4) == "#4"
    assert app_module._rule_ordinal(4, None) == "#4"
    assert app_module._rule_ordinal(1, 3) == "#1.3"
    assert app_module._rule_ordinal(17) == "#17"


def test_regex_list_entries_numbered_card_dot_sub(app_module):
    # Card #1 with three entries, each changing the text -> #1.1 / #1.2 / #1.3.
    _set(app_module, [
        _regex_list("rl", "RL", [
            {"pattern": "a", "replacement": "b"},
            {"pattern": "b", "replacement": "c"},
            {"pattern": "c", "replacement": "d"},
        ]),
        _term(),
    ])
    labels = _labels(app_module, "a")  # a -> b -> c -> d
    assert labels[0].startswith("#1.1 ")
    assert labels[1].startswith("#1.2 ")
    assert labels[2].startswith("#1.3 ")


def test_single_compile_rule_numbered_card_only(app_module):
    # Three leading no-op cards push the map rule to card #4 -> "#4", no sub-index.
    def noop(n):
        return _regex_list(n, n, [{"pattern": "zzz", "replacement": "q"}])
    _set(app_module, [
        noop("r1"), noop("r2"), noop("r3"),
        {"name": "m", "label": "Map", "type": "callback:map", "map": {"foo": "bar"}},
        _term(),
    ])
    labels = _labels(app_module, "foo")  # leading no-ops don't fire; map -> #4
    assert labels == ["#4 Map"]


def test_terminal_trim_uses_card_position(app_module):
    # Terminal is the last card (#2) -> the trim step number is its card position.
    _set(app_module, [
        _regex_list("rl", "RL", [{"pattern": "a", "replacement": "b"}]),
        _term(),
    ])
    labels = _labels(app_module, "a   ")  # trailing spaces so the trim fires
    assert labels[0].startswith("#1.1 ")
    assert labels[-1].startswith("#2 ")


def test_skipped_entry_still_consumes_sub_number(app_module):
    # A middle entry with an empty pattern is skipped at compile time, but the
    # following real entry keeps its true row number (#1.3, not #1.2) so the
    # number lines up with the entry's row in the card editor.
    _set(app_module, [
        _regex_list("rl", "RL", [
            {"pattern": "a", "replacement": "b"},   # row 1
            {"pattern": "", "replacement": "z"},    # row 2 (skipped)
            {"pattern": "b", "replacement": "c"},   # row 3
        ]),
        _term(),
    ])
    labels = _labels(app_module, "a")  # a -> b (#1.1), then b -> c (#1.3)
    assert labels[0].startswith("#1.1 ")
    assert labels[1].startswith("#1.3 ")
