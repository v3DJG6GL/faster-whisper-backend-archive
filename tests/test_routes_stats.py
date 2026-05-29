"""Integration tests for the /stats router (host-gated dashboard)."""

from starlette.testclient import TestClient


def test_stats_page_loopback_ok(client):
    r = client.get("/stats")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_stats_snapshot_open_mode_ok(client):
    r = client.get("/stats/snapshot")
    assert r.status_code == 200
    # The snapshot is a JSON object payload built by _build_payload.
    assert isinstance(r.json(), dict)


def test_stats_usage_ok(client):
    r = client.get("/stats/usage")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"days", "metric", "by", "bucket", "lines", "leaderboard"}


def test_stats_snapshot_host_gate_rejects_non_loopback(app_module):
    with TestClient(app_module.app, client=("8.8.8.8", 1)) as c:
        r = c.get("/stats/snapshot")
        assert r.status_code == 403


def test_stats_page_host_gate_rejects_non_loopback(app_module):
    with TestClient(app_module.app, client=("8.8.8.8", 1)) as c:
        assert c.get("/stats").status_code == 403
