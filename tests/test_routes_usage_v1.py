"""Integration tests for GET /v1/usage — the desktop app's self-scoped usage:
today + lifetime totals + a daily/weekly trend SERIES (for the Home stats tiles,
trend chart, and the optional chip readout).

Like /v1/recent-words and /v1/pipeline-rules it lives in the /v1 namespace with
NO host allowlist (so a remote client isn't 403'd by USER_WEBUI_ALLOWED_HOSTS),
and it is STRICTLY self-scoped: a caller only ever sees their own user_id's
numbers — even an admin (the global view is the host-gated /stats page).

Day/week bucketing is server-local, so the series tests pin TZ via set_tz.
"""

from conftest import bearer


def _seed(uid, *, hour, words=0, audio_s=0.0, status="ok", key_id=None):
    """Insert one hourly rollup row for `uid` directly (the app lifespan has
    already init'd usage_store onto the temp DB). key_id defaults to a per-uid
    value because usage_hourly's PRIMARY KEY is (hour, key_id) — two users
    sharing a key at the same hour would collide into one row."""
    import usage_store
    usage_store.record_usage(
        key_id=key_id or f"k-{uid}", user_id=uid, audio_s=audio_s, words=words,
        status=status, hour=hour,
    )


# --------------------------------------------------------------------------
# Shape (open mode)
# --------------------------------------------------------------------------

def test_v1_usage_shape(client):
    body = client.get("/v1/usage").json()
    assert set(body) == {"username", "today", "total", "range", "series"}
    for k in ("today", "total"):
        assert set(body[k]) == {"requests", "errors", "words", "audio_s"}
    assert set(body["range"]) == {"days", "bucket"}
    assert body["range"]["bucket"] == "day" and body["range"]["days"] == 30
    assert isinstance(body["series"], list)


# --------------------------------------------------------------------------
# Totals + series correctness
# --------------------------------------------------------------------------

def test_v1_usage_totals_and_series(client, make_user_key, set_tz):
    set_tz("UTC")
    import usage_store
    make_user_key("root", is_admin=True)  # flip lockdown
    uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    today_h = usage_store.local_day_start_hour(0)
    _seed(uid, hour=today_h, words=100, audio_s=60.0)
    _seed(uid, hour=today_h + 1, words=40, audio_s=30.0)                  # also today
    _seed(uid, hour=usage_store.local_day_start_hour(2), words=10, audio_s=5.0)  # 2 days ago

    body = client.get("/v1/usage?days=7", headers=bearer(raw)).json()
    assert body["username"] == "alice"
    assert body["today"]["words"] == 140 and body["today"]["requests"] == 2
    assert body["today"]["audio_s"] == 90.0
    assert body["total"]["words"] == 150 and body["total"]["requests"] == 3
    assert body["range"] == {"days": 7, "bucket": "day"}
    # Series spans >=2 distinct days and sums to the lifetime total here.
    assert len(body["series"]) >= 2
    assert sum(c["words"] for c in body["series"]) == 150
    assert all(set(c) == {"day", "requests", "errors", "words", "audio_s"} for c in body["series"])


def test_v1_usage_tz_midnight_shifts_today(client, make_user_key, set_tz):
    set_tz("UTC")
    import usage_store
    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    today_h = usage_store.local_day_start_hour(0)
    _seed(uid, hour=today_h, words=50, audio_s=10.0)        # after today's midnight
    _seed(uid, hour=today_h - 5, words=7, audio_s=2.0)      # before it (yesterday)

    tz_mid = today_h * 3600  # the client's local midnight, in epoch seconds
    body = client.get(f"/v1/usage?tz_midnight={tz_mid}", headers=bearer(raw)).json()
    assert body["today"]["words"] == 50     # only the post-midnight row
    assert body["total"]["words"] == 57     # both rows


# --------------------------------------------------------------------------
# Self-scoping (the security property)
# --------------------------------------------------------------------------

def test_v1_usage_self_scoped(client, make_user_key, set_tz):
    set_tz("UTC")
    import usage_store
    make_user_key("root", is_admin=True)
    uid_a, raw_a = make_user_key("alice", pages={"quick_config": "own"})
    uid_b, raw_b = make_user_key("bob", pages={"quick_config": "own"})
    h = usage_store.local_day_start_hour(0)
    _seed(uid_a, hour=h, words=11, audio_s=1.0)
    _seed(uid_b, hour=h, words=99, audio_s=9.0)

    assert client.get("/v1/usage", headers=bearer(raw_a)).json()["total"]["words"] == 11
    assert client.get("/v1/usage", headers=bearer(raw_b)).json()["total"]["words"] == 99


def test_v1_usage_admin_is_self_scoped(client, make_user_key, set_tz):
    set_tz("UTC")
    import api_keys_store, usage_store
    uid_root, raw_root = make_user_key("root", is_admin=True)
    uid_other = api_keys_store.create_user("alice", is_admin=False)
    h = usage_store.local_day_start_hour(0)
    _seed(uid_root, hour=h, words=3, audio_s=1.0)
    _seed(uid_other, hour=h, words=500, audio_s=50.0)
    # The admin sees ONLY their own usage here — not the global total.
    body = client.get("/v1/usage", headers=bearer(raw_root)).json()
    assert body["total"]["words"] == 3


# --------------------------------------------------------------------------
# Window: lifetime (days<=0) + week bucket
# --------------------------------------------------------------------------

def test_v1_usage_lifetime_and_week_bucket(client, make_user_key, set_tz):
    set_tz("UTC")
    import usage_store
    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    old_h = usage_store.now_hour() - 24 * 100   # ~100 days ago
    _seed(uid, hour=old_h, words=5, audio_s=1.0)
    _seed(uid, hour=usage_store.local_day_start_hour(0), words=8, audio_s=2.0)

    # days=7: total still counts the old row, but the series window excludes it.
    wk = client.get("/v1/usage?days=7", headers=bearer(raw)).json()
    assert wk["total"]["words"] == 13
    assert sum(c["words"] for c in wk["series"]) == 8

    # days<=0: lifetime series includes the old row too.
    life = client.get("/v1/usage?days=0&bucket=week", headers=bearer(raw)).json()
    assert life["range"] == {"days": 0, "bucket": "week"}
    assert sum(c["words"] for c in life["series"]) == 13


def test_v1_usage_days_clamped(client, make_user_key):
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    assert client.get("/v1/usage?days=99999", headers=bearer(raw)).json()["range"]["days"] == 366


# --------------------------------------------------------------------------
# Auth / page-permission gating
# --------------------------------------------------------------------------

def test_v1_usage_requires_quick_config_page(client, make_user_key):
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("bob", pages={"quick_config": "none"})
    assert client.get("/v1/usage", headers=bearer(raw)).status_code == 403


def test_v1_usage_requires_auth_when_locked_down(client, make_user_key):
    make_user_key("root", is_admin=True)
    assert client.get("/v1/usage").status_code == 401  # no bearer


# --------------------------------------------------------------------------
# The whole point: /v1/usage is NOT host-gated (unlike /quick-config/usage)
# --------------------------------------------------------------------------

def test_v1_usage_not_host_gated(app_module):
    from starlette.testclient import TestClient
    app_module.cfg.USER_WEBUI_ALLOWED_HOSTS = ["127.0.0.1/32"]
    with TestClient(app_module.app, client=("203.0.113.9", 9999)) as c:
        assert c.get("/quick-config/usage").status_code == 403
        assert c.get("/v1/usage").status_code == 200
