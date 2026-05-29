"""Smoke test validating the app_client harness drives the real app."""


def test_v1_models_open_mode(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert "data" in body


def test_whoami_open_mode_is_admin(client):
    r = client.get("/auth/whoami")
    assert r.status_code == 200
    assert r.json()["open_mode"] is True
    assert r.json()["is_admin"] is True


def test_sev_no_auth(client):
    r = client.get("/sev")
    assert r.status_code == 200
    assert set(r.json()) >= {"warn", "err", "crit"}
