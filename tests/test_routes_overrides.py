"""Route tests for /settings/overrides (state / resolve) + the per-user and
per-key binding endpoints on /settings/api-keys. Driven through the real app
via the conftest TestClient; no faster-whisper needed."""

import json

from tests.conftest import bearer

PERMS = "/settings/api-keys/api/users"
OV = "/settings/overrides"


def _admin(make_user_key):
    uid, raw = make_user_key("admin", is_admin=True)
    return uid, raw, bearer(raw)


def _make_profile(client, h, name="clinic-de", **fields):
    body = {"OVERRIDE_PROFILES": {name: fields}}
    r = client.post(f"{OV}/state", headers=h, json=body)
    assert r.status_code == 200, r.text
    return r


def test_overrides_page_renders(client):
    import re
    r = client.get(OV)
    assert r.status_code == 200
    body = r.text
    # render_page substituted every placeholder (none leak through)
    assert not re.findall(r"\{\{[A-Z_]+\}\}", body)
    for marker in ("panel-profiles", "panel-explorer", "_renderWaterfall",
                   "ov-wrap", "tab-explorer", "/settings/overrides"):
        assert marker in body, marker


def test_state_shape_and_field_meta(client, make_user_key):
    _, _, h = _admin(make_user_key)
    j = client.get(f"{OV}/state", headers=h).json()
    assert set(j) >= {"profiles", "field_meta", "defaults", "groups", "rules", "usage"}
    assert j["field_meta"]["BEAM_SIZE"] == {"kind": "int", "min": 1, "max": 20}
    assert j["field_meta"]["STREAMING_VAD_BACKEND"]["kind"] == "enum"
    assert "auto" in j["field_meta"]["STREAMING_VAD_BACKEND"]["opts"]
    # load-time model fields are NOT overridable per-identity → absent
    assert "MODEL_DEVICE" not in j["field_meta"]


def test_state_includes_inherited_defaults(client, make_user_key, app_module):
    """The /state payload ships the live global value for every overridable field
    so the editor can render `inherits <value>` (and seed `+ override` from it).
    The source must match the /settings per-model page byte-for-byte."""
    import admin_routes
    _, _, h = _admin(make_user_key)
    j = client.get(f"{OV}/state", headers=h).json()

    defaults = j["defaults"]
    # Built by iterating field_meta → identical key set.
    assert set(defaults) == set(j["field_meta"])
    # Values come from the same serializer the per-model page uses.
    for name in ("BEAM_SIZE", "DEFAULT_LANGUAGE", "VAD_FILTER"):
        assert defaults[name] == admin_routes._resolved_value(name)
    # Every scalar field shown in the editor grid has a default, so no real row
    # can fall back to the ∅ "missing" glyph.
    grouped = {f for g in j["groups"] for sg in g["subgroups"] for f in sg["fields"]}
    assert grouped, "expected at least one editor field group"
    assert grouped <= set(defaults)


def test_state_requires_admin(client, make_user_key):
    # create the admin first (flips lockdown), then a non-admin caller
    _admin(make_user_key)
    _, raw = make_user_key("bob", is_admin=False)
    r = client.get(f"{OV}/state", headers=bearer(raw))
    assert r.status_code == 403


def test_create_profile_roundtrip_and_usage(client, make_user_key):
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de", BEAM_SIZE=8,
                  locks=["DEFAULT_LANGUAGE"])
    j = client.get(f"{OV}/state", headers=h).json()
    assert j["profiles"]["clinic-de"]["BEAM_SIZE"] == 8
    assert j["profiles"]["clinic-de"]["locks"] == ["DEFAULT_LANGUAGE"]
    assert "clinic-de" in j["usage"]


def test_delete_guard_counts_allowlist_only_reference(client, make_user_key):
    # A profile referenced ONLY in a key's requestable allowlist (not forced in
    # `profiles`) must still appear in usage, so the WebUI delete guard refuses
    # it instead of allowing a silent delete that strands a dangling allowlist
    # name — which that binding's next save would then reject.
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de")
    uid, _ = make_user_key("alice", is_admin=False)
    kid = client.get(f"{PERMS}/{uid}/keys", headers=h).json()["keys"][0]["id"]
    # bind the profile ONLY via the allowlist — NOT forced in `profiles`
    r = client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=h, json={
        "overrides": {}, "profiles": [], "locks": [],
        "allowed_override_profiles": ["clinic-de"]})
    assert r.status_code == 200, r.text

    usage = client.get(f"{OV}/state", headers=h).json()["usage"]["clinic-de"]
    assert kid in usage["keys"]          # allowlist-only ref now counted
    assert usage["users"] == []          # not forced/allowed on any user


def test_bad_profile_value_422(client, make_user_key):
    _, _, h = _admin(make_user_key)
    r = client.post(f"{OV}/state", headers=h,
                    json={"OVERRIDE_PROFILES": {"x": {"BEAM_SIZE": 999}}})
    assert r.status_code == 422


def test_post_rejects_foreign_field(client, make_user_key):
    _, _, h = _admin(make_user_key)
    r = client.post(f"{OV}/state", headers=h, json={"BEAM_SIZE": 5})
    assert r.status_code == 400


def test_bind_user_and_resolve(client, make_user_key):
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de", BEAM_SIZE=8,
                  locks=["DEFAULT_LANGUAGE"], TEMPERATURE="0.0")
    uid, _ = make_user_key("alice", is_admin=False)
    # bind alice to the profile + a direct BEST_OF override + lock TEMPERATURE
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {"BEST_OF": 7, "TEMPERATURE": "0.0"},
                                "profiles": ["clinic-de"], "locks": ["TEMPERATURE"]}})
    assert r.status_code == 200, r.text

    rj = client.get(f"{OV}/resolve", headers=h, params={
        "user_id": uid, "model": "whisper-1",
        "sim": json.dumps({"beam_size": 12, "temperature": 0.5})}).json()
    f = rj["fields"]
    assert f["DEFAULT_LANGUAGE"]["winner_value"] == "de"
    assert f["DEFAULT_LANGUAGE"]["winner_layer"] == "user.profile:clinic-de"
    assert f["DEFAULT_LANGUAGE"]["locked"] is True
    assert f["BEAM_SIZE"]["winner_value"] == 8                      # from profile
    assert f["BEST_OF"]["winner_value"] == 7                        # user.direct
    # TEMPERATURE locked by user.direct → simulated client temp is ignored
    assert f["TEMPERATURE"]["client_sim"]["outcome"] == "ignored_locked"
    assert "clinic-de" in rj["profiles_applied"]


def test_resolve_decode_master_gate_off_reports_ignored(client, make_user_key):
    # When an identity's decode-override master gate is OFF, the live path drops
    # EVERY client decode override (resolve sets locked_client_keys = all client
    # keys). The /resolve diagnostic must agree and report a sim'd override as
    # ignored — even though the field itself carries no field-level lock.
    _, _, h = _admin(make_user_key)
    bob_uid, _ = make_user_key("bob", is_admin=False)
    kid = client.get(f"{PERMS}/{bob_uid}/keys", headers=h).json()["keys"][0]["id"]
    r = client.patch(f"{PERMS}/{bob_uid}/keys/{kid}/config", headers=h, json={
        "overrides": {}, "profiles": [], "locks": [],
        "allow_request_decode_overrides": False})
    assert r.status_code == 200, r.text

    rj = client.get(f"{OV}/resolve", headers=h, params={
        "user_id": bob_uid, "key_id": kid, "model": "whisper-1",
        "sim": json.dumps({"beam_size": 12})}).json()
    bs = rj["fields"]["BEAM_SIZE"]
    assert bs["locked"] is False                       # no field-level lock
    assert bs["client_sim"]["value"] == 12
    assert bs["client_sim"]["outcome"] == "ignored_locked"  # gate, not field-lock


def test_per_key_config_overrides_user(client, make_user_key):
    _, _, h = _admin(make_user_key)
    uid, _ = make_user_key("alice", is_admin=False)
    # alice (user) gets BEAM_SIZE 8; her laptop key forces BEAM_SIZE 4
    client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {"BEAM_SIZE": 8}, "profiles": [], "locks": []}})
    keys = client.get(f"{PERMS}/{uid}/keys", headers=h).json()["keys"]
    kid = keys[0]["id"]
    r = client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=h,
                     json={"overrides": {"BEAM_SIZE": 4}, "profiles": [], "locks": []})
    assert r.status_code == 200, r.text
    assert r.json()["config"]["direct"]["BEAM_SIZE"] == 4

    rj = client.get(f"{OV}/resolve", headers=h, params={
        "user_id": uid, "key_id": kid, "model": "whisper-1"}).json()
    bs = rj["fields"]["BEAM_SIZE"]
    assert bs["winner_value"] == 4 and bs["winner_layer"] == "key.direct"


def test_per_key_config_unknown_profile_400(client, make_user_key):
    _, _, h = _admin(make_user_key)
    uid, _ = make_user_key("alice", is_admin=False)
    kid = client.get(f"{PERMS}/{uid}/keys", headers=h).json()["keys"][0]["id"]
    r = client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=h,
                     json={"overrides": {}, "profiles": ["ghost"], "locks": []})
    assert r.status_code == 400


# --- profile rename (key migration + reference cascade) --------------------

def test_rename_profile_cascades_to_user_and_key(client, make_user_key):
    import api_keys_store
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de", BEAM_SIZE=8)
    uid, _ = make_user_key("alice", is_admin=False)
    # alice references the profile in BOTH her ordered list and her allowlist
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {}, "profiles": ["clinic-de"],
                                "locks": [],
                                "allowed_override_profiles": ["clinic-de"]}})
    assert r.status_code == 200, r.text
    kid = client.get(f"{PERMS}/{uid}/keys", headers=h).json()["keys"][0]["id"]
    r = client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=h,
                     json={"overrides": {}, "profiles": ["clinic-de"], "locks": []})
    assert r.status_code == 200, r.text

    r = client.post(f"{OV}/profiles/rename", headers=h,
                    json={"old": "clinic-de", "new": "clinic-deutsch"})
    assert r.status_code == 200, r.text
    assert r.json()["bindings_updated"] == 2          # user row + key row

    j = client.get(f"{OV}/state", headers=h).json()
    assert "clinic-de" not in j["profiles"]
    assert j["profiles"]["clinic-deutsch"]["BEAM_SIZE"] == 8   # overrides preserved

    uc = api_keys_store.get_user_config(uid)
    assert uc["profiles"] == ["clinic-deutsch"]
    assert uc["allowed_override_profiles"] == ["clinic-deutsch"]
    assert api_keys_store.get_key_config(kid)["profiles"] == ["clinic-deutsch"]

    # The renamed profile still resolves end-to-end under its new name.
    rj = client.get(f"{OV}/resolve", headers=h,
                    params={"user_id": uid, "model": "whisper-1"}).json()
    assert rj["fields"]["DEFAULT_LANGUAGE"]["winner_layer"] \
        == "user.profile:clinic-deutsch"


def test_rename_profile_unknown_404(client, make_user_key):
    _, _, h = _admin(make_user_key)
    r = client.post(f"{OV}/profiles/rename", headers=h,
                    json={"old": "ghost", "new": "phantom"})
    assert r.status_code == 404


def test_rename_profile_collision_409(client, make_user_key):
    _, _, h = _admin(make_user_key)
    # Both profiles must coexist — the page always saves the full dict.
    client.post(f"{OV}/state", headers=h, json={"OVERRIDE_PROFILES": {
        "a-prof": {"BEAM_SIZE": 5}, "b-prof": {"BEAM_SIZE": 6}}})
    r = client.post(f"{OV}/profiles/rename", headers=h,
                    json={"old": "a-prof", "new": "b-prof"})
    assert r.status_code == 409


def test_rename_profile_bad_name_and_noop_400(client, make_user_key):
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "good", BEAM_SIZE=5)
    # invalid characters / shape
    assert client.post(f"{OV}/profiles/rename", headers=h,
                       json={"old": "good", "new": "Bad Name!"}).status_code == 400
    # renaming to the current name is a no-op error
    assert client.post(f"{OV}/profiles/rename", headers=h,
                       json={"old": "good", "new": "good"}).status_code == 400


def test_rename_profile_old_name_case_insensitive(client, make_user_key):
    """Profile keys are always stored lowercase, so the endpoint lowercases `old`
    before both the lookup and the binding cascade. A direct API caller passing
    `old` in mixed case must still match (200, not a spurious 404) and the same
    lowercased name must flow to the reference cascade."""
    import api_keys_store
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de")
    uid, _ = make_user_key("alice", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {}, "profiles": ["clinic-de"],
                                "locks": [], "allowed_override_profiles": []}})
    assert r.status_code == 200, r.text
    # `old` sent upper-case; stored key is lowercase "clinic-de".
    r = client.post(f"{OV}/profiles/rename", headers=h,
                    json={"old": "CLINIC-DE", "new": "clinic-deutsch"})
    assert r.status_code == 200, r.text
    # The lowercased `old` also reached the cascade (the binding ref matched).
    assert r.json()["bindings_updated"] == 1
    j = client.get(f"{OV}/state", headers=h).json()
    assert "clinic-de" not in j["profiles"]
    assert "clinic-deutsch" in j["profiles"]
    assert api_keys_store.get_user_config(uid)["profiles"] == ["clinic-deutsch"]
