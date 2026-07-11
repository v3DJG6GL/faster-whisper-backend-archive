"""Integration tests for /v1/client-settings (desktop settings sync)."""

from conftest import bearer

_URL = "/v1/client-settings"


def _put(client, blob, base_version, device=None, headers=None):
    body = {"blob": blob, "base_version": base_version}
    if device is not None:
        body["device"] = device
    return client.put(_URL, json=body, headers=headers or {})


def test_get_empty_zero_state(client):
    r = client.get(_URL)
    assert r.status_code == 200
    assert r.json() == {"version": 0, "blob": None, "updated_at": None, "device": None}


def test_put_create_then_get_echo(client):
    r = _put(client, {"theme": "dark", "backends": []}, 0, device="laptop")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 1
    assert body["blob"] == {"theme": "dark", "backends": []}
    assert body["device"] == "laptop"

    r = client.get(_URL)
    assert r.status_code == 200
    got = r.json()
    assert got["version"] == 1
    assert got["blob"] == {"theme": "dark", "backends": []}
    assert got["updated_at"] is not None


def test_stale_put_409_carries_current_then_force_put(client):
    _put(client, {"n": 1}, 0)
    _put(client, {"n": 2}, 1)  # server now at version 2

    r = _put(client, {"n": 99}, 1)  # stale base_version
    assert r.status_code == 409
    body = r.json()
    assert body["detail"] == "version conflict"
    assert body["version"] == 2
    assert body["blob"] == {"n": 2}

    # Force-push = echo the version the 409 just reported.
    r = _put(client, {"n": 99}, body["version"])
    assert r.status_code == 200
    assert r.json()["version"] == 3


def test_oversize_blob_413(client):
    import client_settings_store as store
    big = {"x": "a" * (store._CAP_BLOB + 100)}
    r = _put(client, big, 0)
    assert r.status_code == 413


def test_malformed_422(client):
    # Missing base_version.
    r = client.put(_URL, json={"blob": {}})
    assert r.status_code == 422
    # Unknown extra field (extra="forbid").
    r = client.put(_URL, json={"blob": {}, "base_version": 0, "bogus": 1})
    assert r.status_code == 422
    # Non-object blob (scalars must never be storable — GET's blob:null
    # zero-state has to stay unambiguous).
    r = client.put(_URL, json={"blob": "just a string", "base_version": 0})
    assert r.status_code == 422


def test_delete_then_zero_state(client):
    _put(client, {"n": 1}, 0)
    r = client.delete(_URL)
    assert r.status_code == 200
    assert r.json() == {"ok": True, "deleted": True}
    r = client.get(_URL)
    assert r.json()["version"] == 0 and r.json()["blob"] is None
    r = client.delete(_URL)
    assert r.json() == {"ok": True, "deleted": False}


def test_open_mode_shares_one_row(client):
    """With no admin key the server is in OPEN mode: every caller resolves
    to the "(open-mode)" account, so two 'devices' see one blob."""
    _put(client, {"from": "device-a"}, 0, device="a")
    r = client.get(_URL)  # a second unauthenticated caller
    assert r.json()["blob"] == {"from": "device-a"}
    assert r.json()["device"] == "a"


def test_locked_down_requires_bearer_and_isolates_users(client, make_user_key):
    _uid_admin, admin_key = make_user_key("admin", is_admin=True)  # flips to locked-down
    _uid_a, key_a = make_user_key("alice")
    _uid_b, key_b = make_user_key("bob")

    # No bearer once locked down -> 401.
    assert client.get(_URL).status_code == 401
    assert _put(client, {"n": 1}, 0).status_code == 401

    # Each account gets its own isolated blob.
    r = _put(client, {"who": "alice"}, 0, headers=bearer(key_a))
    assert r.status_code == 200
    r = _put(client, {"who": "bob"}, 0, headers=bearer(key_b))
    assert r.status_code == 200
    assert client.get(_URL, headers=bearer(key_a)).json()["blob"] == {"who": "alice"}
    assert client.get(_URL, headers=bearer(key_b)).json()["blob"] == {"who": "bob"}


def test_two_keys_same_user_share_one_blob(client, make_user_key):
    """The sync-scope decision: the store keys on user_id, so a second key
    belonging to the SAME account reads/writes the same settings set."""
    import api_keys_store
    uid, key1 = make_user_key("carol", is_admin=True)
    key2, _rec = api_keys_store.create_key(uid, label="second machine")

    r = _put(client, {"synced": True}, 0, device="machine-1", headers=bearer(key1))
    assert r.status_code == 200
    got = client.get(_URL, headers=bearer(key2)).json()
    assert got["blob"] == {"synced": True}
    assert got["device"] == "machine-1"


def test_store_unavailable_maps_to_503_not_500(client, monkeypatch):
    """init_db failing at startup (e.g. an unwritable CLIENT_SETTINGS_DB in a
    containerized deployment) must surface as an actionable 503 pointing at
    the config knob — not the bare 500 it used to be."""
    import client_settings_store

    monkeypatch.setattr(client_settings_store, "_conn", None)
    r = client.get(_URL)
    assert r.status_code == 503
    assert "CLIENT_SETTINGS_DB" in r.json()["detail"]
    assert _put(client, {"n": 1}, 0).status_code == 503
    assert client.delete(_URL).status_code == 503
