"""Integration tests for /quick-config routes."""

import copy


def _expose_first_regex_rule(app_module):
    """Mark the first regex rule exposed so /quick-config can see + patch it.
    Returns its slug. Mutates a deep copy assigned back onto cfg so the test's
    monkeypatched view is isolated; the per-test config reload restores it."""
    rules = copy.deepcopy(list(app_module.cfg.PIPELINE_RULES))
    slug = None
    for r in rules:
        if isinstance(r, dict) and r.get("type") == "regex":
            r["exposed"] = True
            slug = r["name"]
            break
    app_module.cfg.PIPELINE_RULES = rules
    return slug


def test_quick_config_page(client):
    r = client.get("/quick-config")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_state_open_mode(client):
    r = client.get("/quick-config/state")
    assert r.status_code == 200
    body = r.json()
    assert "rules" in body
    assert body["role"] == "admin"  # open mode = synthetic admin


def test_usage_open_mode(client):
    r = client.get("/quick-config/usage")
    assert r.status_code == 200
    body = r.json()
    assert "today" in body and "total" in body


def test_post_patch_empty_is_noop(client):
    r = client.post("/quick-config/state", json={"rules_patch": {}})
    assert r.status_code == 200
    assert r.json()["saved"] == []


def test_post_patch_unknown_slug_400(client):
    r = client.post(
        "/quick-config/state",
        json={"rules_patch": {"no-such-rule": {"enabled": False}}},
    )
    assert r.status_code == 400


def test_post_patch_unknown_field_400(client, app_module):
    slug = _expose_first_regex_rule(app_module)
    assert slug is not None
    # `label` is admin-only / not in the per-type allow-list -> 400.
    r = client.post(
        "/quick-config/state",
        json={"rules_patch": {slug: {"label": "hacked"}}},
    )
    assert r.status_code == 400


def test_post_patch_unknown_top_level_field_422(client):
    # QuickPatchPayload has extra="forbid".
    r = client.post(
        "/quick-config/state",
        json={"rules_patch": {}, "bogus": 1},
    )
    assert r.status_code == 422


def test_post_patch_valid_field_saves(client, app_module):
    slug = _expose_first_regex_rule(app_module)
    r = client.post(
        "/quick-config/state",
        json={"rules_patch": {slug: {"enabled": False}}},
    )
    assert r.status_code == 200
    assert slug in r.json()["saved"]


def test_recent_open_mode(client):
    r = client.get("/quick-config/recent")
    assert r.status_code == 200
    body = r.json()
    assert "recent" in body


def test_reapply_rules_status(client):
    r = client.get("/quick-config/reapply-rules/status")
    assert r.status_code == 200
    # captures_reapply.status() returns the worker state dict.
    assert isinstance(r.json(), dict)


def test_reapply_rules_start_captures_disabled(client, app_module):
    # CAPTURE_RECORDINGS_ENABLED defaults False -> idle, no-op note.
    app_module.cfg.CAPTURE_RECORDINGS_ENABLED = False
    r = client.post("/quick-config/reapply-rules")
    assert r.status_code == 200
    assert r.json().get("status") == "idle"
