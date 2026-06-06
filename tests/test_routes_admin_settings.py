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
    """A valid rule list: one regex rule per name, terminal rule last."""
    out = [{"name": n, "label": n.title(), "type": "regex",
            "pattern": n[0], "replacement": n[0].upper()} for n in names]
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
                {"name": "noop", "type": "regex", "pattern": "x", "replacement": "y",
                 "enabled": True},
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "steps" in body and "final" in body
    # The implicit terminal trim strips the edge whitespace.
    assert body["final"] == "hallo welt"


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
