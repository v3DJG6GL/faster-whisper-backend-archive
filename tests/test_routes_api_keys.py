"""Integration tests for /config/api-keys routes."""

from conftest import bearer


def test_api_keys_page(client):
    r = client.get("/config/api-keys")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_list_users_open_mode(client):
    r = client.get("/config/api-keys/api/users")
    assert r.status_code == 200
    body = r.json()
    assert "users" in body
    assert body["open_mode"] is True
    assert "pages" in body


def test_create_user_and_key_flow(client):
    # Create a user.
    r = client.post(
        "/config/api-keys/api/users",
        json={"username": "carol", "is_admin": False},
    )
    assert r.status_code == 200
    uid = r.json()["user_id"]

    # Create a key — raw value shown once.
    r = client.post(f"/config/api-keys/api/users/{uid}/keys", json={"label": "k1"})
    assert r.status_code == 200
    body = r.json()
    assert body["key"].startswith("wk_")
    kid = body["record"]["id"]

    # List the user's keys — raw value NOT returned.
    r = client.get(f"/config/api-keys/api/users/{uid}/keys")
    assert r.status_code == 200
    keys = r.json()["keys"]
    assert any(k["id"] == kid for k in keys)
    assert all("key" not in k for k in keys)  # no raw value leaks

    # Delete the key.
    r = client.delete(f"/config/api-keys/api/users/{uid}/keys/{kid}")
    assert r.status_code == 200
    assert r.json()["ok"] is True

    # Delete the user.
    r = client.delete(f"/config/api-keys/api/users/{uid}")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_create_user_unknown_field_422(client):
    r = client.post(
        "/config/api-keys/api/users",
        json={"username": "x", "bogus": 1},
    )
    assert r.status_code == 422  # extra="forbid"


def test_create_user_blank_username_400(client):
    # Pydantic min_length=1 rejects "" at the schema layer -> 422.
    r = client.post("/config/api-keys/api/users", json={"username": ""})
    assert r.status_code == 422


def test_delete_unknown_user_404(client):
    r = client.delete("/config/api-keys/api/users/does-not-exist")
    assert r.status_code == 404


def test_delete_last_admin_409(client, make_user_key):
    # Lock down with a single admin, then try to revoke that admin -> 409.
    uid, raw = make_user_key("root", is_admin=True)
    r = client.delete(
        f"/config/api-keys/api/users/{uid}", headers=bearer(raw)
    )
    assert r.status_code == 409


def test_patch_permissions_invalid_scope_400(client):
    r = client.post(
        "/config/api-keys/api/users",
        json={"username": "dan", "is_admin": False},
    )
    uid = r.json()["user_id"]
    # "bogus" is not a valid scope -> set_user_permissions raises ValueError -> 400.
    r = client.patch(
        f"/config/api-keys/api/users/{uid}/permissions",
        json={"pages": {"reports": "bogus"}},
    )
    assert r.status_code == 400


def test_patch_permissions_unknown_field_422(client):
    r = client.post(
        "/config/api-keys/api/users",
        json={"username": "eve", "is_admin": False},
    )
    uid = r.json()["user_id"]
    r = client.patch(
        f"/config/api-keys/api/users/{uid}/permissions",
        json={"pages": {}, "nope": True},
    )
    assert r.status_code == 422


def test_patch_permissions_unknown_user_404(client):
    r = client.patch(
        "/config/api-keys/api/users/missing/permissions",
        json={"pages": {"reports": "own"}},
    )
    assert r.status_code == 404


def test_usage_api_ok(client):
    r = client.get("/config/api-keys/api/usage")
    assert r.status_code == 200
    body = r.json()
    assert "by_user" in body and "by_key" in body
