"""Integration tests for the GET /v1/recent-words client API.

Recent-word suggestions for the desktop "Dictionary" spoken-symbol (callback:map)
key field — the /v1 analogue of the /quick-config autocomplete datalist. It
aggregates the caller's recent-transcription tokens ∪ bigrams (newest-first,
case-insensitive dedup, capped at QUICK_CONFIG_WORD_SUGGESTIONS_MAX) via the
shared build_word_suggestions() helper, scoped exactly like /quick-config/recent
(own rows unless quick_config scope == "all"). Like /v1/pipeline-rules it lives
in the /v1 namespace with NO host allowlist, so a remote client isn't 403'd by
USER_WEBUI_ALLOWED_HOSTS (unlike the browser /quick-config page).
"""

from conftest import bearer


def _seed(uid, tokens, *, request_id, bigrams=None):
    """Insert a recent-transcription row with verbatim tokens/bigrams for `uid`.
    record_trace stores them as-is (the tokenizer runs upstream), so the test
    controls the suggestion pool directly."""
    import transcriptions_store
    transcriptions_store.record_trace(
        request_id=request_id, model="m", raw="raw", final="final",
        tokens=tokens, bigrams=bigrams or [], user_id=uid,
    )


# --------------------------------------------------------------------------
# Shape / open mode (synthetic admin sees all)
# --------------------------------------------------------------------------

def test_v1_recent_words_open_mode_shape(client):
    body = client.get("/v1/recent-words").json()
    assert set(body) == {"words", "max"}
    assert isinstance(body["words"], list)
    assert body["max"] == 200  # default cap


def test_v1_recent_words_empty_when_no_traces(client):
    assert client.get("/v1/recent-words").json()["words"] == []


def test_v1_recent_words_aggregates_tokens_and_bigrams(client):
    _seed("u1", ["Komma", "Punkt"], bigrams=["Hans Peter"], request_id="r1")
    words = client.get("/v1/recent-words").json()["words"]
    assert "Komma" in words and "Punkt" in words and "Hans Peter" in words


def test_v1_recent_words_case_insensitive_dedup(client):
    _seed("u1", ["Komma"], request_id="r1")
    _seed("u1", ["komma"], request_id="r2")  # newer; same lowercased key
    words = client.get("/v1/recent-words").json()["words"]
    assert sum(1 for w in words if w.lower() == "komma") == 1


# --------------------------------------------------------------------------
# Cap / limit / disabled
# --------------------------------------------------------------------------

def test_v1_recent_words_cap_honored(client, app_module):
    app_module.cfg.QUICK_CONFIG_WORD_SUGGESTIONS_MAX = 2
    _seed("u1", ["a-word", "b-word", "c-word", "d-word"], request_id="r1")
    body = client.get("/v1/recent-words").json()
    assert body["max"] == 2
    assert len(body["words"]) == 2


def test_v1_recent_words_zero_disables(client, app_module):
    app_module.cfg.QUICK_CONFIG_WORD_SUGGESTIONS_MAX = 0
    _seed("u1", ["Komma"], request_id="r1")
    body = client.get("/v1/recent-words").json()
    assert body["max"] == 0
    assert body["words"] == []


def test_v1_recent_words_limit_param_clamps(client, app_module):
    app_module.cfg.QUICK_CONFIG_WORD_SUGGESTIONS_MAX = 50
    _seed("u1", ["a-word", "b-word", "c-word"], request_id="r1")
    # The client may request fewer than the cap...
    assert len(client.get("/v1/recent-words?limit=1").json()["words"]) == 1
    # ...but `max` always echoes the server cap, and limit can't exceed it.
    assert client.get("/v1/recent-words?limit=999").json()["max"] == 50


# --------------------------------------------------------------------------
# Per-user scoping (the security property): user A never sees user B's words
# --------------------------------------------------------------------------

def test_v1_recent_words_scoped_per_user(client, make_user_key):
    import api_keys_store
    make_user_key("root", is_admin=True)  # flips lockdown
    uid_a = api_keys_store.create_user("alice", is_admin=False)
    api_keys_store.set_user_permissions(uid_a, {"pages": {"quick_config": "own"}})
    raw_a, _ = api_keys_store.create_key(uid_a)
    uid_b = api_keys_store.create_user("bob", is_admin=False)
    api_keys_store.set_user_permissions(uid_b, {"pages": {"quick_config": "own"}})
    raw_b, _ = api_keys_store.create_key(uid_b)

    _seed(uid_a, ["alpha-word"], request_id="ra")
    _seed(uid_b, ["bravo-word"], request_id="rb")

    a_words = client.get("/v1/recent-words", headers=bearer(raw_a)).json()["words"]
    assert "alpha-word" in a_words and "bravo-word" not in a_words

    b_words = client.get("/v1/recent-words", headers=bearer(raw_b)).json()["words"]
    assert "bravo-word" in b_words and "alpha-word" not in b_words


def test_v1_recent_words_admin_sees_all(client, make_user_key):
    import api_keys_store
    _uid_root, raw_root = make_user_key("root", is_admin=True)
    uid_a = api_keys_store.create_user("alice", is_admin=False)
    _seed(uid_a, ["alpha-word"], request_id="ra")
    _seed("someone-else", ["bravo-word"], request_id="rb")
    words = client.get("/v1/recent-words", headers=bearer(raw_root)).json()["words"]
    assert "alpha-word" in words and "bravo-word" in words


# --------------------------------------------------------------------------
# Page-permission + lockdown gating
# --------------------------------------------------------------------------

def test_v1_recent_words_requires_quick_config_page(client, make_user_key):
    make_user_key("root", is_admin=True)  # lockdown
    _uid, raw = make_user_key("bob", pages={"quick_config": "none"})
    assert client.get("/v1/recent-words", headers=bearer(raw)).status_code == 403


def test_v1_recent_words_requires_auth_when_locked_down(client, make_user_key):
    make_user_key("root", is_admin=True)  # lockdown
    assert client.get("/v1/recent-words").status_code == 401  # no bearer


# --------------------------------------------------------------------------
# The whole point: /v1/recent-words is NOT host-gated (unlike /quick-config)
# --------------------------------------------------------------------------

def test_v1_recent_words_not_host_gated(app_module):
    from starlette.testclient import TestClient
    # Narrow the user-WebUI allowlist to loopback, then call from a non-loopback
    # client: the browser /quick-config/recent is host-gated (403), but the /v1
    # client API is reachable (200).
    app_module.cfg.USER_WEBUI_ALLOWED_HOSTS = ["127.0.0.1/32"]
    with TestClient(app_module.app, client=("203.0.113.9", 9999)) as c:
        assert c.get("/quick-config/recent").status_code == 403
        assert c.get("/v1/recent-words").status_code == 200
