"""P10: per-request `override_profile` selection on the batch endpoint + the
`/v1/override-profiles` names endpoint. Driven through the real app; no
faster-whisper needed (FakeModel records the decode kwargs).

The resolver-level precedence/gate semantics are covered purely (no DB) in
test_effective_config.py; here we assert the HTTP wiring: the Form field reaches
the resolver, `profile_applied` is echoed, an admin binding/lock still wins over
the request profile, the gate disables it, and the names endpoint behaves.
"""

import json

import config as cfg
from tests.conftest import bearer

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}
OV = "/settings/overrides"
PERMS = "/settings/api-keys/api/users"


def test_request_profile_applies_to_decode_kwargs(client, fake_model, monkeypatch):
    # Open mode (no keys) → synthetic admin; the request profile is the only layer.
    monkeypatch.setattr(cfg, "OVERRIDE_PROFILES", {"fast": {"BEAM_SIZE": 3}}, raising=False)
    monkeypatch.setattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", True, raising=False)
    r = client.post(
        "/v1/audio/transcriptions", files=_FILE,
        data={"model": "whisper-1", "response_format": "verbose_json",
              "override_profile": "fast"},
    )
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["beam_size"] == 3
    assert r.json()["profile_applied"] == "fast"


def test_request_profile_gated_off_is_ignored(client, fake_model, monkeypatch):
    monkeypatch.setattr(cfg, "OVERRIDE_PROFILES", {"fast": {"BEAM_SIZE": 3}}, raising=False)
    monkeypatch.setattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", False, raising=False)
    r = client.post(
        "/v1/audio/transcriptions", files=_FILE,
        data={"model": "whisper-1", "response_format": "verbose_json",
              "override_profile": "fast"},
    )
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["beam_size"] != 3      # profile ignored
    assert r.json()["profile_applied"] is None


def test_unknown_request_profile_not_applied(client, fake_model, monkeypatch):
    monkeypatch.setattr(cfg, "OVERRIDE_PROFILES", {"fast": {"BEAM_SIZE": 3}}, raising=False)
    monkeypatch.setattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", True, raising=False)
    r = client.post(
        "/v1/audio/transcriptions", files=_FILE,
        data={"model": "whisper-1", "response_format": "verbose_json",
              "override_profile": "nope"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["profile_applied"] is None


def test_binding_lock_beats_request_profile(client, make_user_key, fake_model):
    # An admin-bound profile that LOCKS beam_size must win over a request profile
    # that sets a different beam_size — the request profile is least-specific.
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    r = client.post(f"{OV}/state", headers=admin_h, json={"OVERRIDE_PROFILES": {
        "locked": {"BEAM_SIZE": 8, "locks": ["BEAM_SIZE"]},
        "fast": {"BEAM_SIZE": 3},
    }})
    assert r.status_code == 200, r.text
    uid, raw_alice = make_user_key("alice", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=admin_h,
                     json={"pages": {}, "config": {"overrides": {}, "profiles": ["locked"], "locks": []}})
    assert r.status_code == 200, r.text

    r = client.post(
        "/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
        data={"model": "whisper-1", "response_format": "verbose_json",
              "override_profile": "fast"},
    )
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["beam_size"] == 8      # binding wins; request profile shadowed
    assert r.json()["profile_applied"] == "fast"         # it was applied (just shadowed on beam_size)


# --- /v1/override-profiles names endpoint ---------------------------------

def test_override_profiles_endpoint_lists_names(client, monkeypatch):
    monkeypatch.setattr(cfg, "OVERRIDE_PROFILES",
                        {"fast": {"BEAM_SIZE": 3}, "clinic-de": {"DEFAULT_LANGUAGE": "de"}},
                        raising=False)
    monkeypatch.setattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", True, raising=False)
    r = client.get("/v1/override-profiles")
    assert r.status_code == 200, r.text
    assert r.json() == {"profiles": ["clinic-de", "fast"]}   # sorted, names only


def test_override_profiles_endpoint_empty_when_gated(client, monkeypatch):
    monkeypatch.setattr(cfg, "OVERRIDE_PROFILES", {"fast": {"BEAM_SIZE": 3}}, raising=False)
    monkeypatch.setattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", False, raising=False)
    r = client.get("/v1/override-profiles")
    assert r.status_code == 200, r.text
    assert r.json() == {"profiles": []}


def test_override_profiles_endpoint_requires_auth_when_locked(client, make_user_key, monkeypatch):
    monkeypatch.setattr(cfg, "OVERRIDE_PROFILES", {"fast": {"BEAM_SIZE": 3}}, raising=False)
    monkeypatch.setattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", True, raising=False)
    # Creating the first admin key flips the app to locked-down mode.
    _, raw_admin = make_user_key("admin", is_admin=True)
    assert client.get("/v1/override-profiles").status_code == 401   # no bearer
    _, raw_alice = make_user_key("alice", is_admin=False)            # non-admin is fine
    r = client.get("/v1/override-profiles", headers=bearer(raw_alice))
    assert r.status_code == 200, r.text
    assert r.json()["profiles"] == ["fast"]


# --- "__none__" suppression sentinel (P27) --------------------------------

def test_none_sentinel_suppresses_bound_profile(client, make_user_key, fake_model):
    # A profile bound to a user normally applies; sending the reserved "__none__"
    # request name suppresses it, falling back to plain server defaults.
    from config_store import NO_PROFILE_SENTINEL
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    r = client.post(f"{OV}/state", headers=admin_h,
                    json={"OVERRIDE_PROFILES": {"clinic": {"BEAM_SIZE": 7}}})
    assert r.status_code == 200, r.text
    uid, raw_alice = make_user_key("alice", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=admin_h,
                     json={"pages": {}, "config": {"overrides": {}, "profiles": ["clinic"], "locks": []}})
    assert r.status_code == 200, r.text

    base = {"model": "whisper-1", "response_format": "verbose_json"}
    # Baseline: admin has no bound profile → the plain default beam_size.
    r = client.post("/v1/audio/transcriptions", files=_FILE, headers=admin_h, data=base)
    assert r.status_code == 200, r.text
    default_beam = fake_model.last_kwargs["beam_size"]
    assert default_beam != 7

    # Alice with the bound profile, no override → the profile applies.
    r = client.post("/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice), data=base)
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["beam_size"] == 7

    # Alice sends "__none__" → bound profile suppressed → back to the plain default.
    r = client.post("/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
                    data={**base, "override_profile": NO_PROFILE_SENTINEL})
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["beam_size"] == default_beam   # suppressed → default
    assert r.json()["profile_applied"] is None


def test_none_sentinel_never_listed(client, monkeypatch):
    from config_store import NO_PROFILE_SENTINEL
    monkeypatch.setattr(cfg, "OVERRIDE_PROFILES", {"fast": {"BEAM_SIZE": 3}}, raising=False)
    monkeypatch.setattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", True, raising=False)
    r = client.get("/v1/override-profiles")
    assert r.status_code == 200, r.text
    assert NO_PROFILE_SENTINEL not in r.json()["profiles"]
