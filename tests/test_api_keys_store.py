"""Tests for api_keys_store — the security-sensitive identity layer.

Covers key hashing/generation, the O(1) lookup + last-used debounce, user/key
CRUD, the atomic last-admin guard (including a threaded concurrency test),
lockdown state transitions, and the permission model.
"""

import threading

import pytest


# ---------------------------------------------------------------------------
# Hashing / generation (pure)
# ---------------------------------------------------------------------------

def test_hash_key_deterministic_hex():
    import api_keys_store as ak
    h1 = ak.hash_key("wk_abc")
    h2 = ak.hash_key("wk_abc")
    assert h1 == h2 and len(h1) == 64
    int(h1, 16)  # valid hex
    assert ak.hash_key("wk_abc") != ak.hash_key("wk_abd")


def test_hash_key_unicode_safe():
    import api_keys_store as ak
    assert len(ak.hash_key("schlüssel-Ω")) == 64


def test_generate_raw_key_shape_and_uniqueness():
    import api_keys_store as ak
    keys = {ak.generate_raw_key() for _ in range(200)}
    assert len(keys) == 200  # all unique
    for k in list(keys)[:5]:
        assert k.startswith(ak.KEY_PREFIX)
        assert len(k) == len(ak.KEY_PREFIX) + 43


def test_split_display_parts():
    import api_keys_store as ak
    prefix, last4 = ak._split_display_parts("wk_abcdef1234567890wxyz")
    assert prefix == "wk_abcde" and last4 == "wxyz"


# ---------------------------------------------------------------------------
# create_user
# ---------------------------------------------------------------------------

def test_create_user_and_duplicate(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("alice", is_admin=False)
    assert ak.get_user(uid)["username"] == "alice"
    with pytest.raises(ValueError):
        ak.create_user("alice", is_admin=False)  # duplicate


def test_create_user_blank_and_too_long(api_keys_db):
    ak = api_keys_db
    with pytest.raises(ValueError):
        ak.create_user("   ", is_admin=False)
    with pytest.raises(ValueError):
        ak.create_user("x" * 129, is_admin=False)


def test_create_user_default_perms(api_keys_db):
    ak = api_keys_db
    nonadmin = ak.create_user("bob", is_admin=False)
    admin = ak.create_user("root", is_admin=True)
    assert ak.get_user_permissions(nonadmin)["pages"]["quick_config"] == "own"
    assert ak.get_user_permissions(admin) == {}  # admins bypass policy


# ---------------------------------------------------------------------------
# Lockdown transitions
# ---------------------------------------------------------------------------

def test_lockdown_transitions(api_keys_db):
    ak = api_keys_db
    assert ak.is_locked_down() is False                 # fresh DB -> open
    uid = ak.create_user("root", is_admin=True)
    assert ak.is_locked_down() is False                 # admin user, no key yet
    _, rec = ak.create_key(uid)
    assert ak.is_locked_down() is True                  # active admin key -> locked
    # Add a second admin key, then revoke one -> still locked.
    _, rec2 = ak.create_key(uid)
    ak.revoke_key(rec["id"])
    assert ak.is_locked_down() is True
    # Revoking the last admin key is blocked by the guard, so stays locked.
    with pytest.raises(ak.LastAdminError):
        ak.revoke_key(rec2["id"])


# ---------------------------------------------------------------------------
# lookup_by_raw_key + debounce
# ---------------------------------------------------------------------------

def test_lookup_hit_miss_and_falsy(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    raw, rec = ak.create_key(uid)
    got = ak.lookup_by_raw_key(raw)
    assert got["user_id"] == uid and got["key_id"] == rec["id"]
    assert got["is_admin"] is False
    assert ak.lookup_by_raw_key("wk_nope") is None
    assert ak.lookup_by_raw_key("") is None


def test_last_used_debounce(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    raw, rec = ak.create_key(uid)
    ak.lookup_by_raw_key(raw)
    t1 = ak.get_key(rec["id"])["last_used_ts"]
    assert t1 is not None
    # Second lookup within the 60s window must NOT write again.
    ak.lookup_by_raw_key(raw)
    assert ak.get_key(rec["id"])["last_used_ts"] == t1


# ---------------------------------------------------------------------------
# create_key validation
# ---------------------------------------------------------------------------

def test_create_key_user_missing(api_keys_db):
    ak = api_keys_db
    with pytest.raises(ValueError):
        ak.create_key("nonexistent")


def test_create_key_label_too_long(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    with pytest.raises(ValueError):
        ak.create_key(uid, label="x" * 129)


def test_create_key_returns_prefixed_raw(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    raw, rec = ak.create_key(uid, label="laptop")
    assert raw.startswith("wk_")
    assert rec["label"] == "laptop" and rec["revoked_ts"] is None


# ---------------------------------------------------------------------------
# Last-admin guard
# ---------------------------------------------------------------------------

def test_revoke_last_admin_key_blocked(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("root", is_admin=True)
    _, rec = ak.create_key(uid)
    with pytest.raises(ak.LastAdminError):
        ak.revoke_key(rec["id"])


def test_revoke_admin_key_ok_when_second_exists(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("root", is_admin=True)
    _, r1 = ak.create_key(uid)
    _, r2 = ak.create_key(uid)
    ak.revoke_key(r1["id"])  # second admin key remains -> allowed
    assert ak.get_key(r1["id"])["revoked_ts"] is not None


def test_revoke_last_admin_user_blocked(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("root", is_admin=True)
    ak.create_key(uid)
    with pytest.raises(ak.LastAdminError):
        ak.revoke_user(uid)


def test_revoke_nonadmin_unguarded(api_keys_db):
    ak = api_keys_db
    admin = ak.create_user("root", is_admin=True)
    ak.create_key(admin)
    u = ak.create_user("bob", is_admin=False)
    ak.create_key(u)
    ak.revoke_user(u)  # no guard for non-admins
    assert ak.get_user(u)["revoked_ts"] is not None


def test_concurrent_revoke_of_two_admin_keys_keeps_one(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("root", is_admin=True)
    _, r1 = ak.create_key(uid)
    _, r2 = ak.create_key(uid)
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def worker(kid):
        barrier.wait()
        try:
            ak.revoke_key(kid)
        except ak.LastAdminError as e:
            errors.append(e)

    t1 = threading.Thread(target=worker, args=(r1["id"],))
    t2 = threading.Thread(target=worker, args=(r2["id"],))
    t1.start(); t2.start(); t1.join(); t2.join()
    # Exactly one revoke is refused by the atomic guard -> one admin key left.
    assert len(errors) == 1
    assert ak.active_key_counts().get(uid, 0) == 1
    assert ak.is_locked_down() is True


# ---------------------------------------------------------------------------
# set_user_permissions
# ---------------------------------------------------------------------------

def test_set_permissions_validates_page_and_scope(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    with pytest.raises(ValueError):
        ak.set_user_permissions(uid, {"pages": {"nosuchpage": "all"}})
    with pytest.raises(ValueError):
        ak.set_user_permissions(uid, {"pages": {"captures": "sideways"}})


def test_set_permissions_access_only_rejects_own(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    # stats is access-only: none|all, not own.
    with pytest.raises(ValueError):
        ak.set_user_permissions(uid, {"pages": {"stats": "own"}})
    ak.set_user_permissions(uid, {"pages": {"stats": "all"}})


def test_set_permissions_merge_preserves_untouched(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    ak.set_user_permissions(uid, {"pages": {"captures": "all"}})
    # Patch only reports; captures must survive the merge.
    clean = ak.set_user_permissions(uid, {"pages": {"reports": "all"}})
    assert clean["pages"]["captures"] == "all"
    assert clean["pages"]["reports"] == "all"


def test_set_permissions_normalises_tags(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    clean = ak.set_user_permissions(uid, {"pages": {}, "quick_config_tags": ["B", "a", "a"]})
    assert clean["quick_config_tags"] == ["a", "b"]


def test_set_permissions_revoked_user_raises(api_keys_db):
    ak = api_keys_db
    admin = ak.create_user("root", is_admin=True)
    ak.create_key(admin)
    u = ak.create_user("bob", is_admin=False)
    ak.create_key(u)
    ak.revoke_user(u)
    with pytest.raises(ValueError):
        ak.set_user_permissions(u, {"pages": {"captures": "all"}})


# ---------------------------------------------------------------------------
# username / sentinel helpers
# ---------------------------------------------------------------------------

def test_username_helpers(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("alice", is_admin=False)
    assert ak.get_username(uid) == "alice"
    assert ak.get_username(None) is None
    assert ak.get_username("(open-mode)") is None
    assert ak.get_username("missing") is None
    batch = ak.get_usernames([uid, "(open-mode)", None, "missing"])
    assert batch[uid] == "alice" and batch["missing"] is None
    assert "(open-mode)" not in batch


def test_get_user_permissions_sentinel(api_keys_db):
    ak = api_keys_db
    assert ak.get_user_permissions("(open-mode)") == {}
    assert ak.get_user_permissions("missing") == {}


def test_open_mode_user_is_admin():
    import api_keys_store as ak
    assert ak.OPEN_MODE_USER["is_admin"] is True
