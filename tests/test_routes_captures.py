"""Integration tests for /captures routes.

Captures rows require real audio transcode (ffmpeg) to create, so these
tests focus on the read/list/route-ordering/auth surface that works without
fabricating audio blobs.
"""

from starlette.testclient import TestClient

from conftest import bearer


def test_captures_page(client):
    r = client.get("/captures")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_captures_list_open_mode(client):
    r = client.get("/captures/api/list")
    assert r.status_code == 200
    body = r.json()
    assert "captures" in body and "counts" in body
    assert "is_admin" in body


def test_reprocess_vad_job_lifecycle(client):
    # Status endpoint is registered and reports a known state.
    s0 = client.get("/captures/api/reprocess-vad/status")
    assert s0.status_code == 200
    assert s0.json()["status"] in ("idle", "running", "done", "error")
    # Start the bulk VAD re-merge on an empty store → runs and finishes clean.
    assert client.post("/captures/api/reprocess-vad").status_code == 200
    import time
    s = s0.json()
    for _ in range(30):
        s = client.get("/captures/api/reprocess-vad/status").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert s["status"] == "done"
    assert s["total"] == 0 and s["rebuilt"] == 0


def test_samples_route_not_swallowed_by_cid(client):
    # Regression: /captures/api/samples must resolve to the sample-list handler,
    # NOT the parameterized /captures/api/{cid} handler (which would 404 with
    # cid="samples"). A 200 with a "samples" key proves correct route ordering.
    r = client.get("/captures/api/samples")
    assert r.status_code == 200
    assert "samples" in r.json()


def test_export_route_not_swallowed_by_cid(client):
    # /captures/api/export is also a literal route declared before /{cid}.
    r = client.get("/captures/api/export")
    assert r.status_code == 200
    assert "application/gzip" in r.headers.get("content-type", "")


def test_unknown_cid_404(client):
    r = client.get("/captures/api/does-not-exist")
    assert r.status_code == 404


def test_propose_merges_ok(client):
    r = client.get("/captures/api/propose-merges")
    assert r.status_code == 200
    assert "proposals" in r.json()


def test_by_request_id_ok(client):
    r = client.get("/captures/api/by-request/unknown-req")
    assert r.status_code == 200
    assert r.json()["captures"] == []  # no captures for an unknown request id


def test_host_gate_rejects_non_loopback(app_module):
    # The whole /captures router carries require_admin_host.
    with TestClient(app_module.app, client=("8.8.8.8", 1)) as c:
        assert c.get("/captures/api/list").status_code == 403


def test_list_requires_page_when_locked(client, make_user_key):
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"captures": "none"})
    r = client.get("/captures/api/list", headers=bearer(raw))
    assert r.status_code == 403


def test_clear_requires_admin_when_locked(client, make_user_key):
    # POST /captures/api/clear additionally Depends(require_admin).
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"captures": "own"})
    r = client.post("/captures/api/clear", headers=bearer(raw))
    assert r.status_code == 403
