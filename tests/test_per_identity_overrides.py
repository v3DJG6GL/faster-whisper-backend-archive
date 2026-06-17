"""P11: per-identity request gates + allowlist, the GET /v1/me capability
contract, the caller-filtered GET /v1/override-profiles, and the per-profile
GET /v1/override-profiles/{name} values endpoint. Driven through the real app.

Resolver-level precedence/gate semantics are covered purely (no DB) in
test_effective_config.py; here we assert the HTTP wiring and the admin binding
round-trip (the new request-gate keys survive validate_binding/_parse_binding).
"""

from tests.conftest import bearer

OV = "/settings/overrides"
PERMS = "/settings/api-keys/api/users"


def _profiles(client, h, profiles):
    r = client.post(f"{OV}/state", headers=h, json={"OVERRIDE_PROFILES": profiles})
    assert r.status_code == 200, r.text


def _key_id(client, h, uid):
    r = client.get(f"{PERMS}/{uid}/keys", headers=h)
    assert r.status_code == 200, r.text
    return r.json()["keys"][0]["id"]


def _set_key_binding(client, h, uid, kid, **binding):
    body = {"overrides": {}, "profiles": [], "locks": [], **binding}
    r = client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=h, json=body)
    assert r.status_code == 200, r.text
    return r.json()["config"]


# --- GET /v1/me -----------------------------------------------------------

def test_me_open_mode_all_allowed(client):
    j = client.get("/v1/me").json()
    assert j["can_request_override_profile"] is True
    assert j["can_request_decode_overrides"] is True
    assert j["allowed_override_profiles"] == ["*"]


def test_me_reflects_per_key_gate(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 3}})
    uid, raw_alice = make_user_key("alice")
    kid = _key_id(client, h, uid)
    _set_key_binding(client, h, uid, kid,
                     allow_request_override_profile=False,
                     allow_request_decode_overrides=False,
                     allowed_override_profiles=["fast"])
    j = client.get("/v1/me", headers=bearer(raw_alice)).json()
    assert j["can_request_override_profile"] is False
    assert j["can_request_decode_overrides"] is False
    # gate off ⇒ no names, even though the allowlist named one
    assert j["allowed_override_profiles"] == []


def test_me_explicit_allowlist(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 3}, "slow": {"BEAM_SIZE": 12}})
    uid, raw_alice = make_user_key("alice")
    kid = _key_id(client, h, uid)
    _set_key_binding(client, h, uid, kid, allowed_override_profiles=["fast"])
    j = client.get("/v1/me", headers=bearer(raw_alice)).json()
    assert j["can_request_override_profile"] is True
    assert j["allowed_override_profiles"] == ["fast"]


# --- GET /v1/override-profiles (caller-filtered) --------------------------

def test_override_profiles_filtered_by_allowlist(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 3}, "slow": {"BEAM_SIZE": 12}})
    uid, raw_alice = make_user_key("alice")
    kid = _key_id(client, h, uid)
    _set_key_binding(client, h, uid, kid, allowed_override_profiles=["fast"])
    r = client.get("/v1/override-profiles", headers=bearer(raw_alice))
    assert r.json() == {"profiles": ["fast"]}


def test_override_profiles_excludes_non_requestable(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 3},
                          "internal": {"BEAM_SIZE": 1, "requestable": False}})
    _, raw_alice = make_user_key("alice")
    r = client.get("/v1/override-profiles", headers=bearer(raw_alice))
    assert r.json() == {"profiles": ["fast"]}     # internal hidden from clients


# --- GET /v1/override-profiles/{name} (values) ----------------------------

def test_override_profile_detail_values_and_locks(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 3, "VAD_FILTER": True,
                                   "locks": ["BEAM_SIZE"]}})
    _, raw_alice = make_user_key("alice")
    r = client.get("/v1/override-profiles/fast", headers=bearer(raw_alice))
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["name"] == "fast"
    assert j["values"] == {"beam_size": 3, "vad_filter": True}   # projected to client keys
    assert j["locked"] == ["beam_size"]


def test_override_profile_detail_exposes_prompt(client, make_user_key):
    """B5: the detail endpoint exposes the profile's DEFAULT_PROMPT SEPARATELY (not
    in `values`, which is the 19 client decode keys) so the editor can ghost it as
    the inherited 'Vocabulary / prompt'. Its lock state rides along; a prompt-less
    profile reports null/false."""
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {
        "withp": {"BEAM_SIZE": 3, "DEFAULT_PROMPT": "Medizin: Anamnese",
                  "locks": ["DEFAULT_PROMPT"]},
        "nop": {"BEAM_SIZE": 5},
    })
    _, raw_alice = make_user_key("alice")
    ah = bearer(raw_alice)
    j = client.get("/v1/override-profiles/withp", headers=ah).json()
    assert j["prompt"] == "Medizin: Anamnese"
    assert j["prompt_locked"] is True
    assert "default_prompt" not in j["values"]      # prompt is NOT a client decode key
    j2 = client.get("/v1/override-profiles/nop", headers=ah).json()
    assert j2["prompt"] is None
    assert j2["prompt_locked"] is False


def test_override_profile_detail_404_when_not_allowed(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 3},
                          "internal": {"BEAM_SIZE": 1, "requestable": False}})
    uid, raw_alice = make_user_key("alice")
    kid = _key_id(client, h, uid)
    _set_key_binding(client, h, uid, kid, allowed_override_profiles=["fast"])
    ah = bearer(raw_alice)
    assert client.get("/v1/override-profiles/internal", headers=ah).status_code == 404
    assert client.get("/v1/override-profiles/slow", headers=ah).status_code == 404  # unknown
    assert client.get("/v1/override-profiles/fast", headers=ah).status_code == 200


# --- admin binding round-trip (new keys survive validate/parse) -----------

def test_binding_roundtrip_preserves_request_gates(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 3}})
    uid, _ = make_user_key("alice")
    kid = _key_id(client, h, uid)
    stored = _set_key_binding(client, h, uid, kid,
                              allow_request_override_profile=False,
                              allow_request_decode_overrides=True,
                              allowed_override_profiles=["fast"])
    assert stored["allow_request_override_profile"] is False
    assert stored["allow_request_decode_overrides"] is True
    assert stored["allowed_override_profiles"] == ["fast"]
    # re-read via the keys listing → the stored config carries the gates
    r = client.get(f"{PERMS}/{uid}/keys", headers=h)
    cfg = r.json()["keys"][0]["config"]
    assert cfg["allow_request_override_profile"] is False
    assert cfg["allowed_override_profiles"] == ["fast"]


# --- admin per-key "apply no profiles" force ------------------------------

def test_apply_no_profiles_roundtrip_and_suppresses_user_profile(client, make_user_key):
    # End-to-end: a profile bound at the USER scope applies, until the per-KEY
    # apply_no_profiles force flips the key to plain defaults — through the real
    # PATCH → store → resolve path.
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"clinic": {"BEAM_SIZE": 7}})
    uid, _ = make_user_key("alice")
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {}, "profiles": ["clinic"], "locks": []}})
    assert r.status_code == 200, r.text
    kid = _key_id(client, h, uid)

    # baseline: the user-bound profile applies for this key
    rj = client.get(f"{OV}/resolve", headers=h, params={
        "user_id": uid, "key_id": kid, "model": "whisper-1"}).json()
    assert rj["fields"]["BEAM_SIZE"]["winner_value"] == 7
    assert "clinic" in rj["profiles_applied"]

    # set the per-key force → stored + re-read carry it
    stored = _set_key_binding(client, h, uid, kid, apply_no_profiles=True)
    assert stored["apply_no_profiles"] is True
    cfg = client.get(f"{PERMS}/{uid}/keys", headers=h).json()["keys"][0]["config"]
    assert cfg["apply_no_profiles"] is True

    # resolve now ignores the user-bound profile → plain defaults
    rj = client.get(f"{OV}/resolve", headers=h, params={
        "user_id": uid, "key_id": kid, "model": "whisper-1"}).json()
    assert rj["profiles_applied"] == []
    assert rj["fields"]["BEAM_SIZE"]["winner_layer"] != "user.profile:clinic"
