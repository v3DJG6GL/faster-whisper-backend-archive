"""Integration tests for /reports + the /quick-config submission endpoint."""

from conftest import bearer

_SUBMIT = "/quick-config/reports/api/submit"


def _payload(request_id="req-1", comment="please fix"):
    return {"request_id": request_id, "user_comment": comment}


def test_submit_returns_ok_id(client):
    r = client.post(_SUBMIT, json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "id" in body
    assert body["was_updated"] is False


def test_submit_nothing_to_submit_400(client):
    # No corrections, no intended_text, no comment -> 400.
    r = client.post(_SUBMIT, json={"request_id": "req-x"})
    assert r.status_code == 400


def test_submit_unknown_field_422(client):
    r = client.post(_SUBMIT, json={"user_comment": "x", "bogus": 1})
    assert r.status_code == 422


def test_submit_rate_limit_429(client):
    # 20/600s per user (open-mode user_id is a single sentinel key). The 21st
    # submit in the window trips the limiter.
    last = None
    for i in range(21):
        last = client.post(_SUBMIT, json=_payload(request_id=f"req-{i}"))
    assert last.status_code == 429


def test_reports_page(client):
    r = client.get("/reports")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_reports_list(client):
    client.post(_SUBMIT, json=_payload(request_id="list-1"))
    r = client.get("/reports/api/list")
    assert r.status_code == 200
    body = r.json()
    assert "reports" in body and "counts" in body


def test_patch_report_invalid_status_422(client):
    sub = client.post(_SUBMIT, json=_payload(request_id="patch-1"))
    rid = sub.json()["id"]
    # status is a Literal -> "bogus" fails pydantic validation -> 422.
    r = client.patch(f"/reports/api/{rid}", json={"status": "bogus"})
    assert r.status_code == 422


def test_patch_report_valid_status(client):
    sub = client.post(_SUBMIT, json=_payload(request_id="patch-2"))
    rid = sub.json()["id"]
    r = client.patch(f"/reports/api/{rid}", json={"status": "resolved"})
    assert r.status_code == 200
    assert r.json()["report"]["status"] == "resolved"


def test_patch_unknown_report_404(client):
    r = client.patch("/reports/api/missing", json={"status": "resolved"})
    assert r.status_code == 404


def test_delete_report(client):
    sub = client.post(_SUBMIT, json=_payload(request_id="del-1"))
    rid = sub.json()["id"]
    r = client.delete(f"/reports/api/{rid}")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_delete_unknown_report_404(client):
    r = client.delete("/reports/api/missing")
    assert r.status_code == 404


def test_clear_reports_admin(client):
    client.post(_SUBMIT, json=_payload(request_id="clear-1"))
    r = client.post("/reports/api/clear")
    assert r.status_code == 200
    assert "deleted" in r.json()


def test_export_reports_admin(client):
    r = client.get("/reports/api/export")
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")
    assert "reports" in r.json()


def test_submit_disabled_for_nonadmin_403(client, app_module, make_user_key):
    app_module.cfg.REPORTS_ALLOW_USER_SUBMIT = False
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    r = client.post(_SUBMIT, json=_payload(request_id="nope"), headers=bearer(raw))
    assert r.status_code == 403
