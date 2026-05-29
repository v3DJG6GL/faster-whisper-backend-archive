"""Integration tests for /config admin routes (admin UI enabled by default)."""


def test_config_page_loopback(client):
    r = client.get("/config")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_get_state_open_mode(client):
    r = client.get("/config/state")
    assert r.status_code == 200
    body = r.json()
    assert "fields" in body
    assert "BEAM_SIZE" in body["fields"]


def test_post_state_valid(client):
    # Use a value that DIFFERS from the baseline (BEAM_SIZE default is 10) so
    # save_overrides actually records it as a changed field.
    r = client.post("/config/state", json={"BEAM_SIZE": 3})
    assert r.status_code == 200
    body = r.json()
    assert "BEAM_SIZE" in body["saved"]
    assert "hot_applied" in body
    assert "requires_restart" in body


def test_post_state_invalid_value_422(client):
    # BEAM_SIZE is Annotated[int, Field(ge=1, le=20)] -> 999 fails validation.
    r = client.post("/config/state", json={"BEAM_SIZE": 999})
    assert r.status_code == 422
    assert "errors" in r.json()


def test_get_factory_rules(client):
    r = client.get("/config/factory-rules")
    assert r.status_code == 200
    assert "PIPELINE_RULES" in r.json()
    assert isinstance(r.json()["PIPELINE_RULES"], list)


def test_post_factory_rules_non_list_400(client):
    r = client.post("/config/factory-rules", json={"PIPELINE_RULES": "not-a-list"})
    assert r.status_code == 400


def test_test_pipeline_dry_run(client):
    r = client.post(
        "/config/test-pipeline",
        json={
            "sample": "  hallo welt  ",
            "rules": [
                {"name": "noop", "type": "regex", "pattern": "x", "replacement": "y",
                 "enabled": True},
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "steps" in body and "final" in body
    # The implicit terminal trim strips the edge whitespace.
    assert body["final"] == "hallo welt"


def test_test_pipeline_rules_not_list_400(client):
    r = client.post("/config/test-pipeline", json={"sample": "x", "rules": "nope"})
    assert r.status_code == 400


def test_post_state_requires_admin_when_locked(client, make_user_key):
    from conftest import bearer

    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", is_admin=False)
    r = client.post("/config/state", json={"BEAM_SIZE": 5}, headers=bearer(raw))
    assert r.status_code == 403
