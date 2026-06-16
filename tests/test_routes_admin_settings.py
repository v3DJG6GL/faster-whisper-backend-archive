"""Integration tests for /settings admin routes (admin UI enabled by default)."""


def test_settings_page_loopback(client):
    r = client.get("/settings")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_get_state_open_mode(client):
    r = client.get("/settings/state")
    assert r.status_code == 200
    body = r.json()
    assert "fields" in body
    assert "BEAM_SIZE" in body["fields"]


def test_post_state_valid(client):
    # Use a value that DIFFERS from the baseline (BEAM_SIZE default is 10) so
    # save_overrides actually records it as a changed field.
    r = client.post("/settings/state", json={"BEAM_SIZE": 3})
    assert r.status_code == 200
    body = r.json()
    assert "BEAM_SIZE" in body["saved"]
    assert "hot_applied" in body
    assert "requires_restart" in body


def test_post_state_invalid_value_422(client):
    # BEAM_SIZE is Annotated[int, Field(ge=1, le=20)] -> 999 fails validation.
    r = client.post("/settings/state", json={"BEAM_SIZE": 999})
    assert r.status_code == 422
    assert "errors" in r.json()


def test_reset_to_default_clears_local_override(client):
    """Resetting a setting to its in-repo default must DELETE the override key
    from config.local.json — clearing the 'local.json' badge AND reverting the
    running value. Regression: the WebUI '↺ Reset to default' button submits the
    default *value* (not a removal), which previously rewrote the key and left
    the badge stuck on 'local.json' (and the running cfg on the stale value)."""
    import config_store

    default_val = client.get("/settings/state").json()["fields"]["BEST_OF"]["default_value"]
    assert default_val is not None
    override_val = default_val + 1   # BEST_OF is ge=1, le=20 -> still valid

    # Override it: key present on disk, badge = local.json, value applied live.
    client.post("/settings/state", json={"BEST_OF": override_val})
    field = client.get("/settings/state").json()["fields"]["BEST_OF"]
    assert field["provenance"] == "local.json"
    assert field["value"] == override_val
    assert "BEST_OF" in config_store.load_overrides()

    # Reset = submit the default value back (exactly what the ↺ button sends).
    saved = client.post("/settings/state", json={"BEST_OF": default_val}).json()["saved"]
    assert "BEST_OF" in saved

    # Key gone from disk, badge cleared, running value reverted to the baseline.
    field = client.get("/settings/state").json()["fields"]["BEST_OF"]
    assert "BEST_OF" not in config_store.load_overrides()
    assert field["provenance"] == "default"
    assert field["value"] == default_val


def test_post_default_value_creates_no_override(client):
    """Submitting a value equal to the baseline when nothing was overridden is a
    no-op: prune-on-default keeps the key out of config.local.json entirely."""
    import config_store

    default_val = client.get("/settings/state").json()["fields"]["BEST_OF"]["default_value"]
    body = client.post("/settings/state", json={"BEST_OF": default_val}).json()
    assert body["saved"] == []
    assert "BEST_OF" not in config_store.load_overrides()


def test_reset_float_field_default_sent_as_int_clears_override(client):
    """REPETITION_PENALTY's default is a float (1.0), but the JS client submits a
    whole-number float without its decimal (JSON.stringify(1.0) -> '1'), so the
    server receives int 1. Prune-on-default must treat int 1 == float 1.0 as the
    default and drop the override — a json.dumps comparison ('1' != '1.0') would
    miss it and leave the 'local.json' badge stuck."""
    import config_store

    default_val = client.get("/settings/state").json()["fields"]["REPETITION_PENALTY"]["default_value"]
    assert default_val == 1.0

    # Override it, then "reset" by POSTing the default as a bare int — exactly
    # what the WebUI sends for a whole-number float.
    client.post("/settings/state", json={"REPETITION_PENALTY": 2.0})
    assert "REPETITION_PENALTY" in config_store.load_overrides()

    client.post("/settings/state", json={"REPETITION_PENALTY": 1})   # int, not 1.0
    field = client.get("/settings/state").json()["fields"]["REPETITION_PENALTY"]
    assert "REPETITION_PENALTY" not in config_store.load_overrides()
    assert field["provenance"] == "default"
    assert field["value"] == default_val


def _as_js_sends(v):
    """Mimic how the browser serializes a value on reset: JSON.stringify has no
    int/float distinction, so a whole-number float (1.0, 0.0) goes out without
    its decimal and arrives server-side as an int. Applied recursively so this
    reproduces the real wire payload for list/tuple/dict-valued fields too."""
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, list):
        return [_as_js_sends(x) for x in v]
    if isinstance(v, dict):
        return {k: _as_js_sends(x) for k, x in v.items()}
    return v


def test_every_field_reset_to_default_is_pruned(client):
    """Sweep EVERY admin setting: posting its in-repo default (coerced the way
    the browser sends it) must be recognized as 'not an override' and leave
    config.local.json untouched. Guards against type-specific prune gaps like
    the int/float REPETITION_PENALTY bug for any current or future field."""
    import admin_routes
    import config_store

    fields = client.get("/settings/state").json()["fields"]
    checked = 0
    not_pruned = []
    post_failed = []
    for name, meta in fields.items():
        if meta["provenance"] == "env":
            continue                      # env-pinned saves are ignored anyway
        if name in admin_routes._PRUNE_EXEMPT:
            continue                      # bespoke override mgmt (see below)
        dv = meta["default_value"]
        if dv is None:
            continue                      # null default -> reset sends null (handled)
        checked += 1
        r = client.post("/settings/state", json={name: _as_js_sends(dv)})
        if r.status_code != 200:
            post_failed.append((name, r.status_code, r.text[:160]))
            continue
        if name in r.json()["saved"]:
            not_pruned.append((name, dv, _as_js_sends(dv)))

    assert not not_pruned, (
        "default value NOT recognized as default (override persists, badge sticks):\n"
        + "\n".join(f"  {n}: baseline={d!r} sent={s!r}" for n, d, s in not_pruned)
    )
    assert not post_failed, f"posting the default value failed: {post_failed}"
    assert config_store.load_overrides() == {}   # nothing leaked onto disk
    assert checked >= 10                          # sanity: the sweep ran broadly


def test_pipeline_rules_not_auto_pruned(client):
    """PIPELINE_RULES is intentionally exempt from prune-on-default: a local copy
    equal to the factory rules SHADOWS config.json (managed by the pipeline
    page's dedicated 'clear local override' action), so saving rules equal to the
    factory default must KEEP the override, not silently drop it. Locks in the
    _PRUNE_EXEMPT carve-out so a future change can't start auto-pruning it."""
    import config_store

    rules = client.get("/settings/state").json()["fields"]["PIPELINE_RULES"]["default_value"]
    body = client.post("/settings/state", json={"PIPELINE_RULES": rules}).json()
    assert "PIPELINE_RULES" in body["saved"]
    assert "PIPELINE_RULES" in config_store.load_overrides()


def test_enum_choices_match_schema(client):
    """Dropdown options are derived from the AdminConfig Literal (single source).
    GET /settings/state must surface `choices` == the field's Literal values for
    every enum field, and None for non-enum fields — so the UI <select> and the
    server-side validation can never disagree. Guards the _field_choices dedup."""
    import typing
    import config_store

    def literal_args(field):
        ann = config_store.AdminConfig.model_fields[field].annotation
        for c in (ann, *typing.get_args(ann)):
            if typing.get_origin(c) is typing.Literal:
                return list(typing.get_args(c))
        return None

    fields = client.get("/settings/state").json()["fields"]
    enum_count = 0
    for name in fields:
        if name not in config_store.AdminConfig.model_fields:
            continue
        expected = literal_args(name)
        assert fields[name].get("choices") == expected, name
        if expected is not None:
            enum_count += 1

    # Concrete spot-checks: API choices ARE the Literal (catch a broken extractor).
    assert fields["MODEL_COMPUTE_TYPE"]["choices"] == list(config_store.ComputeLit.__args__)
    assert fields["CONVERT_QUANTIZATION"]["choices"] == list(config_store.ConvertQuantLit.__args__)
    assert fields["MODEL_DEVICE"]["choices"] == ["cuda", "cpu"]
    assert fields["BEST_OF"]["choices"] is None        # non-enum -> free input, no dropdown
    assert enum_count >= 6                              # sanity: the sweep found the enums


def test_settings_page_has_no_hardcoded_enum_opts(client):
    """Both editors (main form + per-model pane) build dropdowns from the server
    `choices`, not hardcoded JS arrays — so options can't drift from the schema."""
    page = client.get("/settings").text
    assert "_ed.choices" in page                        # main form derives from choices
    assert "fieldDef(field).choices" in page            # per-model pane derives too
    # The old hardcoded compute-type option arrays must be gone.
    assert "'int8_float16','int8','float32'" not in page


def test_get_factory_rules(client):
    r = client.get("/settings/factory-rules")
    assert r.status_code == 200
    assert "PIPELINE_RULES" in r.json()
    assert isinstance(r.json()["PIPELINE_RULES"], list)


def test_post_factory_rules_non_list_400(client):
    r = client.post("/settings/factory-rules", json={"PIPELINE_RULES": "not-a-list"})
    assert r.status_code == 400


def _seed_factory(monkeypatch, tmp_path, rules):
    """Repoint config_store's factory file at a per-test temp config.json and
    seed it with `rules`, so persisting factory-rules tests never clobber the
    committed repo config.json. Mirrors conftest's OVERRIDES_PATH repoint:
    FACTORY_PATH is a default ARG bound at def time, so each function's
    __defaults__ is rewritten in addition to the module constant."""
    import json
    import config_store

    tmp_factory = str(tmp_path / "factory_config.json")
    with open(tmp_factory, "w", encoding="utf-8") as f:
        json.dump({"schema_version": 1, "PIPELINE_RULES": rules}, f)
    monkeypatch.setattr(config_store, "FACTORY_PATH", tmp_factory, raising=False)
    for fn in (config_store.load_factory_rules, config_store.save_factory_rules):
        defaults = list(fn.__defaults__ or ())
        if defaults:
            defaults[-1] = tmp_factory
            monkeypatch.setattr(fn, "__defaults__", tuple(defaults), raising=False)
    return tmp_factory


def _rules(*names):
    """A valid rule list: one one-entry regex-list rule per name, terminal last."""
    out = [{"name": n, "label": n.title(), "type": "regex-list",
            "entries": [{"pattern": n[0], "replacement": n[0].upper()}]} for n in names]
    out.append({"name": "trim-edges", "label": "Trim edges", "type": "terminal"})
    return out


def test_post_factory_rules_preserves_order(client, tmp_path, monkeypatch):
    """An order-only promote persists the posted order to config.json verbatim —
    response, GET, and on-disk file all agree. (Backs the JS "Promote order".)"""
    import json

    tmp_factory = _seed_factory(monkeypatch, tmp_path, _rules("alpha", "beta", "gamma"))
    reordered = _rules("gamma", "alpha", "beta")   # terminal stays last
    r = client.post("/settings/factory-rules", json={"PIPELINE_RULES": reordered})
    assert r.status_code == 200, r.text
    expected = ["gamma", "alpha", "beta", "trim-edges"]
    assert [x["name"] for x in r.json()["rules"]] == expected
    g = client.get("/settings/factory-rules")
    assert [x["name"] for x in g.json()["PIPELINE_RULES"]] == expected
    with open(tmp_factory, encoding="utf-8") as f:
        raw = json.load(f)
    assert [x["name"] for x in raw["PIPELINE_RULES"]] == expected


def test_post_factory_rules_reports_shadowed_by_local(client, tmp_path, monkeypatch):
    """shadowed_by_local is False with no local PIPELINE_RULES override and True
    once one exists — the flag the post-promote "clear local override" UX keys on."""
    import config_store

    _seed_factory(monkeypatch, tmp_path, _rules("alpha", "beta"))
    r = client.post("/settings/factory-rules", json={"PIPELINE_RULES": _rules("beta", "alpha")})
    assert r.status_code == 200, r.text
    assert r.json()["shadowed_by_local"] is False

    config_store.save_overrides({"PIPELINE_RULES": _rules("alpha", "beta")})
    r2 = client.post("/settings/factory-rules", json={"PIPELINE_RULES": _rules("beta", "alpha")})
    assert r2.status_code == 200, r2.text
    assert r2.json()["shadowed_by_local"] is True


def test_test_pipeline_dry_run(client):
    r = client.post(
        "/settings/test-pipeline",
        json={
            "sample": "  hallo welt  ",
            "rules": [
                {"name": "noop", "type": "regex-list", "enabled": True,
                 "entries": [{"pattern": "welt", "replacement": "world"}]},
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "steps" in body and "final" in body
    # The regex-list entry runs, then the implicit terminal trim strips edges.
    assert body["final"] == "hallo world"


def test_test_pipeline_regex_list_skips_bad_entry(client):
    # A regex-list with one uncompilable entry must NOT blank the whole card:
    # the engine (main.rebuild_caches) skips the bad entry per-entry and still
    # applies the valid ones. The dry-run mirrors that and reports the bad
    # pattern as an advisory rather than discarding every entry's effect.
    r = client.post(
        "/settings/test-pipeline",
        json={
            "sample": "foo bar",
            "rules": [
                {"name": "rl", "type": "regex-list", "enabled": True, "entries": [
                    {"pattern": "foo", "replacement": "X"},
                    {"pattern": "(", "replacement": "Y"},   # uncompilable
                    {"pattern": "bar", "replacement": "Z"},
                ]},
            ],
        },
    )
    assert r.status_code == 200
    step = r.json()["steps"][0]
    assert step["after"] == "X Z"        # valid entries applied despite the bad one
    assert step["matches"] == 2
    assert step["error"]                 # bad pattern surfaced as an advisory
    assert r.json()["final"] == "X Z"


def test_test_pipeline_rules_not_list_400(client):
    r = client.post("/settings/test-pipeline", json={"sample": "x", "rules": "nope"})
    assert r.status_code == 400


def test_post_state_requires_admin_when_locked(client, make_user_key):
    from conftest import bearer

    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", is_admin=False)
    r = client.post("/settings/state", json={"BEAM_SIZE": 5}, headers=bearer(raw))
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Env-pinned provenance + read-only WebUI behaviour
# ---------------------------------------------------------------------------
# env_pinned_fields() reads os.environ live, and get_state()/post_state() use it,
# so setting the env var is enough to exercise the precedence + badge path
# without reloading config.

def test_get_state_marks_env_pinned(client, monkeypatch):
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "3")
    field = client.get("/settings/state").json()["fields"]["BEAM_SIZE"]
    assert field["provenance"] == "env"
    assert field["env_var"] == "WHISPER_BEAM_SIZE"


def test_post_state_env_pinned_is_ignored_at_runtime(client, monkeypatch):
    # An env-pinned field saves to config.local.json but must NOT change the
    # running cfg — the env var wins. The response flags it as ignored.
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "3")
    body = client.post("/settings/state", json={"BEAM_SIZE": 7}).json()
    assert "BEAM_SIZE" in body["env_pinned_ignored"]
    assert "BEAM_SIZE" not in body["hot_applied"]


def test_settings_page_greys_out_env_pinned_inputs(client):
    # The rendered admin page ships the JS/CSS that disables + greys env-pinned
    # editors (the runtime DOM disabling is driven by provenance=="env").
    text = client.get("/settings").text
    assert "function disableEnvPinnedEditor" in text
    assert "if (isEnvPinned(name)) return;" in text   # setDirty guard
    assert ".field.env-pinned" in text                # greyed styling


def test_field_groups_cover_every_setting():
    """The WebUI layout (_FIELD_GROUPS) must list exactly the AdminConfig schema
    fields: no setting silently missing from the form, no stale/typo'd entry.
    Regression guard — adding a config field without wiring its WebUI group (as
    happened with STREAMING_HARD_BREAK_*) should fail here, not ship invisible."""
    import admin_routes
    import config_store

    displayed = admin_routes._all_fields()
    assert len(displayed) == len(set(displayed)), "duplicate field in _FIELD_GROUPS"
    schema = set(config_store.AdminConfig.model_fields)
    # Fields intentionally edited on a DEDICATED page, not the /settings form.
    # OVERRIDE_PROFILES has its own master-detail editor on /settings/overrides
    # (served by /settings/overrides/state), so it is not in _FIELD_GROUPS.
    managed_elsewhere = {"OVERRIDE_PROFILES"}
    missing = schema - set(displayed) - managed_elsewhere
    stale = set(displayed) - schema
    assert not missing, f"settings missing from the WebUI layout: {sorted(missing)}"
    assert not stale, f"_FIELD_GROUPS entries that are not config fields: {sorted(stale)}"
