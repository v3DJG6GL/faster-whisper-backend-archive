"""Integration tests for the /v1/pipeline-rules client API.

This is the endpoint the desktop "Dictionary" editor uses. It mirrors the
/quick-config view/edit behaviour exactly (shared build_visible_rules /
apply_rules_patch in quick_config_routes) — same `exposed` + tag gating, same
per-type field allow-list, same optimistic-concurrency + validation — but in
the /v1 namespace with NO host allowlist (so a remote client isn't 403'd by
USER_WEBUI_ALLOWED_HOSTS, unlike the browser /quick-config page).
"""

import copy

from conftest import bearer


# --------------------------------------------------------------------------
# Helpers (mirror tests/test_routes_quick_config.py)
# --------------------------------------------------------------------------

def _expose_first_regex_list_rule(app_module):
    """Mark the first regex-list rule exposed (no tags = visible to all).
    Returns its slug. Mutates a deep copy assigned back onto cfg so the per-test
    config reload restores it."""
    rules = copy.deepcopy(list(app_module.cfg.PIPELINE_RULES))
    slug = None
    for r in rules:
        if isinstance(r, dict) and r.get("type") == "regex-list":
            r["exposed"] = True
            slug = r["name"]
            break
    app_module.cfg.PIPELINE_RULES = rules
    return slug


def _expose_tagged(app_module):
    """Expose the first three regex-list rules with tags alpha / beta / (none).
    Returns (alpha_slug, beta_slug, untagged_slug)."""
    rules = copy.deepcopy(list(app_module.cfg.PIPELINE_RULES))
    rl = [r for r in rules if isinstance(r, dict) and r.get("type") == "regex-list"]
    assert len(rl) >= 3, "factory should ship >=3 regex-list rules for this test"
    rl[0]["exposed"] = True
    rl[0]["tags"] = ["alpha"]
    rl[1]["exposed"] = True
    rl[1]["tags"] = ["beta"]
    rl[2]["exposed"] = True
    rl[2]["tags"] = []
    app_module.cfg.PIPELINE_RULES = rules
    return rl[0]["name"], rl[1]["name"], rl[2]["name"]


# --------------------------------------------------------------------------
# GET /v1/pipeline-rules
# --------------------------------------------------------------------------

def test_v1_get_open_mode_shape(client):
    r = client.get("/v1/pipeline-rules")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"rules", "role", "editable_fields", "map_collapse_after"}
    assert isinstance(body["rules"], list)
    assert body["role"] == "admin"  # open mode = synthetic admin
    # editable_fields advertises the per-type allow-list so the client need not
    # hardcode it.
    assert body["editable_fields"]["regex-list"] == ["enabled", "entries"]
    assert "callback:map" in body["editable_fields"]
    # The backend-configured "show newest N cb:map entries" threshold (default 15),
    # served so the desktop client + web page agree.
    assert body["map_collapse_after"] == 15


def test_v1_get_empty_when_nothing_exposed(client):
    # Every factory rule ships exposed=False; even the (open-mode) admin only
    # sees exposed rules on this curated endpoint.
    body = client.get("/v1/pipeline-rules").json()
    assert body["rules"] == []


def test_v1_get_includes_exposed_rule_with_fingerprint(client, app_module):
    slug = _expose_first_regex_list_rule(app_module)
    assert slug is not None
    body = client.get("/v1/pipeline-rules").json()
    names = [r["name"] for r in body["rules"]]
    assert slug in names
    rule = next(r for r in body["rules"] if r["name"] == slug)
    assert rule["_fp"] and isinstance(rule["_fp"], str)
    # terminal sentinel is never exposed to the client
    assert all(r.get("type") != "terminal" for r in body["rules"])


# --------------------------------------------------------------------------
# PATCH /v1/pipeline-rules — allow-list, validation, concurrency
# --------------------------------------------------------------------------

def test_v1_patch_empty_is_noop(client):
    r = client.patch("/v1/pipeline-rules", json={"rules_patch": {}})
    assert r.status_code == 200
    assert r.json()["saved"] == []


def test_v1_patch_valid_field_saves(client, app_module):
    slug = _expose_first_regex_list_rule(app_module)
    r = client.patch(
        "/v1/pipeline-rules",
        json={"rules_patch": {slug: {"enabled": False}}},
    )
    assert r.status_code == 200
    assert slug in r.json()["saved"]


def test_v1_patch_unknown_slug_400(client):
    r = client.patch(
        "/v1/pipeline-rules",
        json={"rules_patch": {"no-such-rule": {"enabled": False}}},
    )
    assert r.status_code == 400


def test_v1_patch_disallowed_field_400(client, app_module):
    slug = _expose_first_regex_list_rule(app_module)
    # `label` is admin-only / not in the per-type allow-list -> 400.
    r = client.patch(
        "/v1/pipeline-rules",
        json={"rules_patch": {slug: {"label": "hacked"}}},
    )
    assert r.status_code == 400


def test_v1_patch_extra_top_level_field_422(client):
    # QuickPatchPayload has extra="forbid".
    r = client.patch(
        "/v1/pipeline-rules",
        json={"rules_patch": {}, "bogus": 1},
    )
    assert r.status_code == 422


def test_v1_patch_bad_regex_returns_422_errors(client, app_module):
    slug = _expose_first_regex_list_rule(app_module)
    # An uncompilable pattern fails the server-side save validation.
    r = client.patch(
        "/v1/pipeline-rules",
        json={"rules_patch": {slug: {"entries": [{"pattern": "(", "replacement": ""}]}}},
    )
    assert r.status_code == 422
    assert "errors" in r.json()


def test_v1_patch_fingerprint_conflict_then_match(client, app_module):
    slug = _expose_first_regex_list_rule(app_module)
    rule = next(r for r in client.get("/v1/pipeline-rules").json()["rules"]
                if r["name"] == slug)
    fp = rule["_fp"]
    # Stale fingerprint -> reported as a conflict, nothing written.
    bad = client.patch("/v1/pipeline-rules", json={
        "rules_patch": {slug: {"enabled": False}},
        "fingerprints": {slug: "deadbeef0000"},
    })
    assert bad.status_code == 200
    assert bad.json()["saved"] == []
    assert any(c["slug"] == slug for c in bad.json()["conflicts"])
    # Matching fingerprint (unchanged since GET) -> saved.
    ok = client.patch("/v1/pipeline-rules", json={
        "rules_patch": {slug: {"enabled": False}},
        "fingerprints": {slug: fp},
    })
    assert ok.status_code == 200
    assert slug in ok.json()["saved"]


# --------------------------------------------------------------------------
# Tag / permission gating (locked-down mode, non-admin keys)
# --------------------------------------------------------------------------

def test_v1_tag_filtering_for_nonadmin(client, app_module, make_user_key):
    import api_keys_store
    alpha, beta, untagged = _expose_tagged(app_module)
    make_user_key("root", is_admin=True)  # flips lockdown
    uid = api_keys_store.create_user("alice", is_admin=False)
    api_keys_store.set_user_permissions(
        uid, {"pages": {"quick_config": "own"}, "quick_config_tags": ["alpha"]})
    raw, _rec = api_keys_store.create_key(uid)

    body = client.get("/v1/pipeline-rules", headers=bearer(raw)).json()
    names = {r["name"] for r in body["rules"]}
    assert body["role"] == "user"
    assert alpha in names           # rule.tags ∩ caller.tags
    assert untagged in names        # empty rule.tags = visible to all
    assert beta not in names        # caller lacks the 'beta' tag


def test_v1_patch_rule_not_visible_403(client, app_module, make_user_key):
    import api_keys_store
    _alpha, beta, _untagged = _expose_tagged(app_module)
    make_user_key("root", is_admin=True)
    uid = api_keys_store.create_user("alice", is_admin=False)
    api_keys_store.set_user_permissions(
        uid, {"pages": {"quick_config": "own"}, "quick_config_tags": ["alpha"]})
    raw, _rec = api_keys_store.create_key(uid)
    # alice can't see the 'beta' rule, so she can't patch it either.
    r = client.patch(
        "/v1/pipeline-rules",
        json={"rules_patch": {beta: {"enabled": False}}},
        headers=bearer(raw),
    )
    assert r.status_code == 403


def test_v1_requires_quick_config_page(client, make_user_key):
    make_user_key("root", is_admin=True)  # lockdown
    _uid, raw = make_user_key("bob", pages={"quick_config": "none"})
    r = client.get("/v1/pipeline-rules", headers=bearer(raw))
    assert r.status_code == 403


def test_v1_requires_auth_when_locked_down(client, make_user_key):
    make_user_key("root", is_admin=True)  # lockdown
    r = client.get("/v1/pipeline-rules")  # no bearer
    assert r.status_code == 401


# --------------------------------------------------------------------------
# The whole point: /v1/pipeline-rules is NOT host-gated (unlike /quick-config)
# --------------------------------------------------------------------------

def test_v1_not_host_gated_unlike_quick_config(app_module):
    from starlette.testclient import TestClient
    # Narrow the user-WebUI allowlist to loopback only, then call from a
    # non-loopback client: the browser /quick-config page is host-gated (403),
    # but the /v1 client API is reachable (200).
    app_module.cfg.USER_WEBUI_ALLOWED_HOSTS = ["127.0.0.1/32"]
    with TestClient(app_module.app, client=("203.0.113.9", 9999)) as c:
        assert c.get("/quick-config/state").status_code == 403
        assert c.get("/v1/pipeline-rules").status_code == 200
