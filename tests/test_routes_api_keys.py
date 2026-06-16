"""Integration tests for /settings/api-keys routes."""

from conftest import bearer


def test_api_keys_page(client):
    r = client.get("/settings/api-keys")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_list_users_open_mode(client):
    r = client.get("/settings/api-keys/api/users")
    assert r.status_code == 200
    body = r.json()
    assert "users" in body
    assert body["open_mode"] is True
    assert "pages" in body


def test_list_users_surfaces_activity_fields(client):
    # The user-card header reads both batched fields straight off /api/users:
    # active_key_count and (server-computed) last_used_ts. An unused key yields
    # a null last_used_ts (the header renders "—" / a grey dot).
    r = client.post(
        "/settings/api-keys/api/users",
        json={"username": "dave", "is_admin": False},
    )
    uid = r.json()["user_id"]
    client.post(f"/settings/api-keys/api/users/{uid}/keys", json={"label": "k1"})
    users = client.get("/settings/api-keys/api/users").json()["users"]
    u = next(x for x in users if x["id"] == uid)
    assert u["active_key_count"] == 1
    assert "last_used_ts" in u
    assert u["last_used_ts"] is None


def test_create_user_and_key_flow(client):
    # Create a user.
    r = client.post(
        "/settings/api-keys/api/users",
        json={"username": "carol", "is_admin": False},
    )
    assert r.status_code == 200
    uid = r.json()["user_id"]

    # Create a key — raw value shown once.
    r = client.post(f"/settings/api-keys/api/users/{uid}/keys", json={"label": "k1"})
    assert r.status_code == 200
    body = r.json()
    assert body["key"].startswith("wk_")
    kid = body["record"]["id"]

    # List the user's keys — raw value NOT returned.
    r = client.get(f"/settings/api-keys/api/users/{uid}/keys")
    assert r.status_code == 200
    keys = r.json()["keys"]
    assert any(k["id"] == kid for k in keys)
    assert all("key" not in k for k in keys)  # no raw value leaks

    # Delete the key.
    r = client.delete(f"/settings/api-keys/api/users/{uid}/keys/{kid}")
    assert r.status_code == 200
    assert r.json()["ok"] is True

    # Delete the user.
    r = client.delete(f"/settings/api-keys/api/users/{uid}")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_create_user_unknown_field_422(client):
    r = client.post(
        "/settings/api-keys/api/users",
        json={"username": "x", "bogus": 1},
    )
    assert r.status_code == 422  # extra="forbid"


def test_create_user_blank_username_400(client):
    # Pydantic min_length=1 rejects "" at the schema layer -> 422.
    r = client.post("/settings/api-keys/api/users", json={"username": ""})
    assert r.status_code == 422


def test_delete_unknown_user_404(client):
    r = client.delete("/settings/api-keys/api/users/does-not-exist")
    assert r.status_code == 404


def test_delete_last_admin_409(client, make_user_key):
    # Lock down with a single admin, then try to revoke that admin -> 409.
    uid, raw = make_user_key("root", is_admin=True)
    r = client.delete(
        f"/settings/api-keys/api/users/{uid}", headers=bearer(raw)
    )
    assert r.status_code == 409


def test_patch_permissions_invalid_scope_400(client):
    r = client.post(
        "/settings/api-keys/api/users",
        json={"username": "dan", "is_admin": False},
    )
    uid = r.json()["user_id"]
    # "bogus" is not a valid scope -> set_user_permissions raises ValueError -> 400.
    r = client.patch(
        f"/settings/api-keys/api/users/{uid}/permissions",
        json={"pages": {"reports": "bogus"}},
    )
    assert r.status_code == 400


def test_patch_permissions_unknown_field_422(client):
    r = client.post(
        "/settings/api-keys/api/users",
        json={"username": "eve", "is_admin": False},
    )
    uid = r.json()["user_id"]
    r = client.patch(
        f"/settings/api-keys/api/users/{uid}/permissions",
        json={"pages": {}, "nope": True},
    )
    assert r.status_code == 422


def test_patch_permissions_unknown_user_404(client):
    r = client.patch(
        "/settings/api-keys/api/users/missing/permissions",
        json={"pages": {"reports": "own"}},
    )
    assert r.status_code == 404


def test_usage_api_ok(client):
    r = client.get("/settings/api-keys/api/usage")
    assert r.status_code == 200
    body = r.json()
    assert "by_user" in body and "by_key" in body


# --- mandatory label on create ----------------------------------------

def _new_user(client, username="zoe", is_admin=False):
    r = client.post(
        "/settings/api-keys/api/users",
        json={"username": username, "is_admin": is_admin},
    )
    assert r.status_code == 200
    return r.json()["user_id"]


def _new_key(client, uid, label="k1"):
    r = client.post(f"/settings/api-keys/api/users/{uid}/keys", json={"label": label})
    assert r.status_code == 200
    return r.json()["record"]["id"]


def test_create_key_blank_label_400(client):
    uid = _new_user(client, "blanky")
    # Explicit empty, whitespace-only, and the defaulted/missing field all 400.
    for payload in ({"label": ""}, {"label": "   "}, {}):
        r = client.post(f"/settings/api-keys/api/users/{uid}/keys", json=payload)
        assert r.status_code == 400, payload


def test_rename_key_flow(client):
    uid = _new_user(client, "renamer")
    kid = _new_key(client, uid, label="old")
    r = client.patch(
        f"/settings/api-keys/api/users/{uid}/keys/{kid}/label",
        json={"label": "  new name  "},
    )
    assert r.status_code == 200
    assert r.json()["record"]["label"] == "new name"  # trimmed
    # List reflects the new label and never leaks the raw key.
    keys = client.get(f"/settings/api-keys/api/users/{uid}/keys").json()["keys"]
    row = next(k for k in keys if k["id"] == kid)
    assert row["label"] == "new name"


def test_rename_key_blank_400(client):
    uid = _new_user(client, "badlabel")
    kid = _new_key(client, uid, label="ok")
    # Blank / whitespace-only are caught by the store -> 400.
    for label in ("", "   "):
        r = client.patch(
            f"/settings/api-keys/api/users/{uid}/keys/{kid}/label",
            json={"label": label},
        )
        assert r.status_code == 400, label
    # Over-length is caught by the RenameKeyIn max_length validator -> 422.
    r = client.patch(
        f"/settings/api-keys/api/users/{uid}/keys/{kid}/label",
        json={"label": "x" * 129},
    )
    assert r.status_code == 422


def test_rename_missing_field_422(client):
    uid = _new_user(client, "nofield")
    kid = _new_key(client, uid, label="ok")
    r = client.patch(
        f"/settings/api-keys/api/users/{uid}/keys/{kid}/label", json={}
    )
    assert r.status_code == 422  # label is a required field on RenameKeyIn


def test_rename_unknown_key_404(client):
    uid = _new_user(client, "ghost")
    r = client.patch(
        f"/settings/api-keys/api/users/{uid}/keys/does-not-exist/label",
        json={"label": "x"},
    )
    assert r.status_code == 404


def test_rename_key_wrong_user_404(client):
    uid_a = _new_user(client, "owner-a")
    kid = _new_key(client, uid_a, label="a-key")
    uid_b = _new_user(client, "owner-b")
    # B cannot rename A's key.
    r = client.patch(
        f"/settings/api-keys/api/users/{uid_b}/keys/{kid}/label",
        json={"label": "stolen"},
    )
    assert r.status_code == 404


def test_rename_revoked_key_409(client):
    uid = _new_user(client, "revoked-rename")
    kid = _new_key(client, uid, label="doomed")
    assert client.delete(
        f"/settings/api-keys/api/users/{uid}/keys/{kid}"
    ).status_code == 200
    r = client.patch(
        f"/settings/api-keys/api/users/{uid}/keys/{kid}/label",
        json={"label": "too late"},
    )
    assert r.status_code == 409
