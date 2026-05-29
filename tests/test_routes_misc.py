"""Misc app-level routes: /v1/models, /logs, /sev, /auth/whoami."""


def test_v1_models_shape(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert "boot_id" in body and isinstance(body["boot_id"], str)
    assert isinstance(body["data"], list)
    for entry in body["data"]:
        assert entry["object"] == "model"
        assert "id" in entry
        assert "loaded" in entry  # bool flag
        assert isinstance(entry["loaded"], bool)
    # No model is loaded in the harness (preload neutralised), so every
    # listed model reports loaded=False.
    assert all(e["loaded"] is False for e in body["data"])


def test_logs_page_open_no_auth(client):
    r = client.get("/logs")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_sev_shape(client):
    r = client.get("/sev")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"warn", "err", "crit"}
    assert all(isinstance(v, int) for v in body.values())


def test_whoami_open_mode_admin(client):
    r = client.get("/auth/whoami")
    assert r.status_code == 200
    body = r.json()
    assert body["open_mode"] is True
    assert body["is_admin"] is True
    assert "permissions" in body and "pages" in body["permissions"]


def test_logs_older_open(client):
    # /logs/older needs the 'logs' scope='all'. In open mode the synthetic
    # admin bypasses the page gate, so it returns the pagination envelope.
    r = client.get("/logs/older")
    assert r.status_code == 200
    body = r.json()
    assert "lines" in body and "next_skip" in body
