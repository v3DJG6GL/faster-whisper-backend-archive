"""Auth-model integration tests: open mode vs locked-down, admin vs
non-admin keys, host gate, cross-user 404, and the SSE ?key= fallback."""

from starlette.testclient import TestClient

from conftest import bearer


def test_open_mode_admin_everywhere(client):
    # No admin key => open mode => synthetic admin can reach admin routes.
    assert client.get("/auth/whoami").json()["is_admin"] is True
    assert client.get("/settings/state").status_code == 200


def test_locked_down_no_bearer_is_401(client, make_user_key):
    # Creating the first admin key flips the server to locked-down.
    make_user_key("root", is_admin=True)
    r = client.get("/settings/state")
    assert r.status_code == 401


def test_locked_down_admin_bearer_ok(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    r = client.get("/settings/state", headers=bearer(raw))
    assert r.status_code == 200


def test_locked_down_bad_bearer_is_401(client, make_user_key):
    make_user_key("root", is_admin=True)
    r = client.get("/settings/state", headers=bearer("wk_not_a_real_key"))
    assert r.status_code == 401


def test_non_admin_403_on_admin_route(client, make_user_key):
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", is_admin=False)
    # /settings/state requires admin -> 403 for a non-admin key.
    r = client.get("/settings/state", headers=bearer(raw))
    assert r.status_code == 403


def test_non_admin_200_on_permitted_page(client, make_user_key):
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key(
        "alice", is_admin=False, pages={"quick_config": "own"}
    )
    # quick_config page granted -> the /state JSON API is reachable.
    r = client.get("/quick-config/state", headers=bearer(raw))
    assert r.status_code == 200


def test_non_admin_403_on_unpermitted_page(client, make_user_key):
    # New non-admins default to quick_config/reports/captures="own"; explicitly
    # revoke quick_config to "none" to exercise the require_page 403 path.
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key(
        "alice", is_admin=False, pages={"quick_config": "none"}
    )
    r = client.get("/quick-config/state", headers=bearer(raw))
    assert r.status_code == 403


def test_host_gate_rejects_non_loopback(app_module):
    # Build a SEPARATE client from a non-loopback source IP. The host gate
    # 403s before auth on host-gated routes.
    with TestClient(app_module.app, client=("8.8.8.8", 1)) as c:
        r = c.get("/settings/state")
        assert r.status_code == 403


def test_host_gate_allows_loopback(client):
    # The default loopback client is admitted by the host gate.
    assert client.get("/settings/state").status_code == 200


def test_cross_user_report_read_is_404(client, make_user_key):
    # Two non-admin users with reports scope=own; one cannot read the other's
    # report by id -> 404 (anti-IDOR), not 403.
    import reports_store

    make_user_key("root", is_admin=True)
    uid_a, _raw_a = make_user_key("alice", pages={"reports": "own"})
    uid_b, raw_b = make_user_key("bob", pages={"reports": "own"})

    rid_a, _ = reports_store.upsert_report(
        user_id=uid_a, request_id="req-a", trace_ts=0.0, model="whisper-1",
        raw="r", final="f", steps=[], corrections=[],
        intended_text="x", user_comment="c", reporter_role="user",
        reporter_host="127.0.0.1",
    )
    # Bob (scope=own) tries to PATCH Alice's report -> 404.
    r = client.patch(
        f"/reports/api/{rid_a}",
        headers=bearer(raw_b),
        json={"status": "resolved"},
    )
    assert r.status_code == 404


# The /quick-config/stream SSE body is an infinite generator (keepalive loop),
# so driving it over HTTP with TestClient hangs on close. The auth/?key=
# fallback logic lives entirely in require_user_or_admin_sse and runs BEFORE
# any streaming — so we exercise that dependency directly with a constructed
# ASGI scope. This is the documented "assert without consuming the stream"
# approach.

def _fake_request(headers=None, query=b""):
    from starlette.requests import Request

    raw_headers = [
        (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/quick-config/stream",
        "headers": raw_headers,
        "query_string": query,
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


def test_sse_key_query_fallback_admits(client, make_user_key):
    # SSE endpoints accept ?key=<raw> since EventSource cannot set headers.
    import quick_config_routes

    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    req = _fake_request(query=f"key={raw}".encode())
    rec = quick_config_routes.require_user_or_admin_sse(req)
    assert rec["username"] == "alice"


def test_sse_missing_key_rejected_when_locked(client, make_user_key):
    import quick_config_routes
    from fastapi import HTTPException

    make_user_key("root", is_admin=True)
    req = _fake_request()  # no bearer, no ?key=
    try:
        quick_config_routes.require_user_or_admin_sse(req)
    except HTTPException as e:
        assert e.status_code == 401
    else:
        raise AssertionError("expected 401 HTTPException")
