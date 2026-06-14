"""Phase 3: the batch /v1/audio/transcriptions path honours per-identity
config. Asserts the kwargs the FakeModel receives + the overrides_ignored
feedback. Driven through the real app; no faster-whisper needed."""

import json

from tests.conftest import bearer

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}
OV = "/settings/overrides"
PERMS = "/settings/api-keys/api/users"


def _setup_profile(client, admin_h, name, **fields):
    r = client.post(f"{OV}/state", headers=admin_h,
                    json={"OVERRIDE_PROFILES": {name: fields}})
    assert r.status_code == 200, r.text


def test_identity_profile_applies_to_decode_kwargs(client, make_user_key, fake_model):
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    # profile: beam=8 (LOCKED), best_of=5 (unlocked)
    _setup_profile(client, admin_h, "p", BEAM_SIZE=8, BEST_OF=5, locks=["BEAM_SIZE"])
    uid, raw_alice = make_user_key("alice", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=admin_h,
                     json={"pages": {}, "config": {"overrides": {}, "profiles": ["p"], "locks": []}})
    assert r.status_code == 200, r.text

    # alice transcribes, trying to override BOTH beam_size (locked) and best_of.
    r = client.post(
        "/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
        data={"model": "whisper-1", "response_format": "verbose_json",
              "decode_overrides": json.dumps({"beam_size": 20, "best_of": 3})},
    )
    assert r.status_code == 200, r.text
    kw = fake_model.last_kwargs
    assert kw["beam_size"] == 8        # profile value; client 20 dropped (locked)
    assert kw["best_of"] == 3          # unlocked → client override wins
    assert r.json()["overrides_ignored"] == ["beam_size"]


def test_locked_language_ignores_client_param(client, make_user_key, fake_model):
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    _setup_profile(client, admin_h, "de", DEFAULT_LANGUAGE="de", locks=["DEFAULT_LANGUAGE"])
    uid, raw_alice = make_user_key("alice", is_admin=False)
    client.patch(f"{PERMS}/{uid}/permissions", headers=admin_h,
                 json={"pages": {}, "config": {"overrides": {}, "profiles": ["de"], "locks": []}})

    r = client.post(
        "/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
        data={"model": "whisper-1", "response_format": "verbose_json", "language": "fr"},
    )
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["language"] == "de"      # locked → client 'fr' ignored
    assert "language" in r.json()["overrides_ignored"]


def test_no_identity_config_is_unchanged(client, make_user_key, fake_model):
    # A user with no binding decodes with the global defaults + the client
    # override applied (no lock) and no overrides_ignored field.
    _, raw_admin = make_user_key("admin", is_admin=True)
    uid, raw_alice = make_user_key("alice", is_admin=False)
    r = client.post(
        "/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
        data={"model": "whisper-1", "response_format": "verbose_json",
              "decode_overrides": json.dumps({"beam_size": 7})},
    )
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["beam_size"] == 7        # client override applies
    assert "overrides_ignored" not in r.json()


def test_decode_overrides_drop_non_finite_floats():
    """JSON permits NaN/Infinity literals; a non-finite float override is dropped
    (ignored) rather than clamped to the field's bound, matching the integer path.
    Shared by the batch route and the streaming FINAL decode via
    _apply_decode_overrides."""
    import main
    # a valid float still applies
    assert main._apply_decode_overrides({}, "whisper-1", {"temperature": 0.7})["temperature"] == 0.7
    # NaN / +inf / -inf each dropped, never clamped to the field's max/min
    for literal in ("NaN", "Infinity", "-Infinity"):
        ov = json.loads('{"temperature": %s, "no_speech_threshold": %s}' % (literal, literal))
        kw = main._apply_decode_overrides({}, "whisper-1", ov)
        assert "temperature" not in kw and "no_speech_threshold" not in kw, (literal, kw)


def test_request_block_identity_section():
    from types import SimpleNamespace
    import main
    ident = SimpleNamespace(layers=["user.profile:clinic-de"], locked={"BEAM_SIZE"},
                            profiles_applied=["clinic-de"])
    info = SimpleNamespace(language="de", language_probability=0.99,
                           duration=1.0, duration_after_vad=1.0)
    seg = [{"id": 0, "start": 0.0, "end": 1.0, "alp": -0.1, "nsp": 0.01,
            "cr": 1.2, "temp": 0.0, "text": "hi"}]
    block = main._format_request_block(
        file_label="x", model_name="whisper-1", info=info,
        kwargs={"beam_size": 8}, seg_diag=seg, raw="hi", final="hi",
        ident=ident, overrides_ignored=["beam_size"])
    assert "Identity" in block
    assert "clinic-de" in block and "BEAM_SIZE" in block
    assert "overrides_ignored" in block and "beam_size" in block
    # no identity + nothing ignored → no Identity section (logs stay terse)
    empty = SimpleNamespace(layers=[], locked=set(), profiles_applied=[])
    block2 = main._format_request_block(
        file_label="x", model_name="whisper-1", info=info, kwargs={},
        seg_diag=seg, raw="hi", final="hi", ident=empty)
    assert "Identity" not in block2


def test_request_block_identity_always_shows_user_even_without_layers():
    """An authenticated caller with NO per-identity binding still gets an
    Identity block naming them + an explicit 'inherits' note — so a missing
    binding (the classic 'my override didn't apply') is visible in the log."""
    from types import SimpleNamespace
    import main
    empty = SimpleNamespace(layers=[], locked=set(), profiles_applied=[])
    info = SimpleNamespace(language="de", language_probability=0.99,
                           duration=1.0, duration_after_vad=1.0)
    seg = [{"id": 0, "start": 0.0, "end": 1.0, "alp": -0.1, "nsp": 0.01,
            "cr": 1.2, "temp": 0.0, "text": "hi"}]
    block = main._format_request_block(
        file_label="x", model_name="whisper-1", info=info, kwargs={},
        seg_diag=seg, raw="hi", final="hi", ident=empty,
        user_id="abcd1234ef", key_id="ffee0011bb", username="Admin")
    assert "Identity" in block
    assert "Admin" in block and "abcd1234" in block      # username + short user id
    assert "ffee0011" in block                            # short key id
    assert "inherits" in block                            # the "(none — inherits…)" note


def test_config_version_bumps_on_binding_and_profile_changes(client, make_user_key):
    """Saving a profile or a per-user / per-key binding bumps config_store's
    version counter — the signal a live streaming connection polls to know it
    must re-resolve its ident (so edits apply without a reconnect)."""
    import config_store
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    v0 = config_store.config_version()
    _setup_profile(client, admin_h, "pv", BEAM_SIZE=6)
    v1 = config_store.config_version()
    assert v1 > v0                                         # profile save bumps
    uid, _ = make_user_key("bob", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=admin_h,
                     json={"pages": {}, "config": {"overrides": {}, "profiles": ["pv"], "locks": []}})
    assert r.status_code == 200, r.text
    v2 = config_store.config_version()
    assert v2 > v1                                         # per-user binding bumps
    kid = client.get(f"{PERMS}/{uid}/keys", headers=admin_h).json()["keys"][0]["id"]
    r = client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=admin_h,
                     json={"overrides": {"BEAM_SIZE": 4}, "profiles": [], "locks": []})
    assert r.status_code == 200, r.text
    v3 = config_store.config_version()
    assert v3 > v2                                         # per-key binding bumps


def test_per_key_override_beats_user(client, make_user_key, fake_model):
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    uid, raw_alice = make_user_key("alice", is_admin=False)
    # user-level beam 8; alice's key forces beam 4
    client.patch(f"{PERMS}/{uid}/permissions", headers=admin_h,
                 json={"pages": {}, "config": {"overrides": {"BEAM_SIZE": 8}, "profiles": [], "locks": []}})
    kid = client.get(f"{PERMS}/{uid}/keys", headers=admin_h).json()["keys"][0]["id"]
    client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=admin_h,
                 json={"overrides": {"BEAM_SIZE": 4}, "profiles": [], "locks": []})

    r = client.post(
        "/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
        data={"model": "whisper-1"},
    )
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["beam_size"] == 4       # key.direct wins over user.direct
