"""Tests for reports_store: upsert insert-vs-merge, lookup, list/get/recent,
update validation, cap eviction order, retention sweep, and helpers."""

import pytest


def _submit(rs, **over):
    kw = dict(
        user_id="u1", request_id="req1", trace_ts=1000.0, model="m",
        raw="raw text", final="final text", steps=[], corrections=[],
        intended_text="", user_comment="", reporter_role="user",
        reporter_host="127.0.0.1",
    )
    kw.update(over)
    return rs.upsert_report(**kw)


# ---------------------------------------------------------------------------
# upsert insert vs update
# ---------------------------------------------------------------------------

def test_insert_returns_not_updated(reports_store_db):
    rid, was_updated = _submit(reports_store_db)
    assert was_updated is False
    row = reports_store_db.get_report(rid)
    assert row["raw"] == "raw text" and row["status"] == "open"
    assert row["reporter_role"] == "user"


def test_reporter_role_normalised(reports_store_db):
    rid, _ = _submit(reports_store_db, reporter_role="superuser")
    assert reports_store_db.get_report(rid)["reporter_role"] == "user"
    rid2, _ = _submit(reports_store_db, request_id="req2", reporter_role="admin")
    assert reports_store_db.get_report(rid2)["reporter_role"] == "admin"


def test_resubmit_updates_same_row_and_merges(reports_store_db):
    rs = reports_store_db
    rid1, _ = _submit(rs, corrections=[{"wrong": "a", "correct": "b", "idx": 0}])
    # Resolve it, then resubmit -> reopens + merges corrections.
    rs.update_report(rid1, {"status": "resolved"})
    rid2, was_updated = _submit(
        rs, corrections=[{"wrong": "c", "correct": "d", "idx": 1}])
    assert was_updated is True and rid2 == rid1
    row = rs.get_report(rid1)
    assert row["status"] == "open"  # reset on resubmit
    idxs = {c.get("idx") for c in row["corrections"]}
    assert idxs == {0, 1}


def test_missing_request_id_always_inserts(reports_store_db):
    rs = reports_store_db
    rid1, u1 = _submit(rs, request_id=None)
    rid2, u2 = _submit(rs, request_id=None)
    # find_by_request_user returns None when request_id falsy -> two rows.
    assert u1 is False and u2 is False and rid1 != rid2


def test_missing_user_id_always_inserts(reports_store_db):
    rs = reports_store_db
    rid1, u1 = _submit(rs, user_id=None)
    rid2, u2 = _submit(rs, user_id=None)
    assert u1 is False and u2 is False and rid1 != rid2


# ---------------------------------------------------------------------------
# find_by_request_user
# ---------------------------------------------------------------------------

def test_find_by_request_user_requires_both(reports_store_db):
    rs = reports_store_db
    _submit(rs, user_id="u1", request_id="req1")
    assert rs.find_by_request_user("req1", "u1") is not None
    assert rs.find_by_request_user("req1", None) is None
    assert rs.find_by_request_user(None, "u1") is None
    assert rs.find_by_request_user("nope", "u1") is None


# ---------------------------------------------------------------------------
# list / get / recent
# ---------------------------------------------------------------------------

def test_list_reports_filter(reports_store_db):
    rs = reports_store_db
    _submit(rs, user_id="a", request_id="r1")
    _submit(rs, user_id="b", request_id="r2")
    assert len(rs.list_reports()) == 2          # None = all
    assert len(rs.list_reports(user_id="a")) == 1


def test_get_report_missing(reports_store_db):
    assert reports_store_db.get_report("does-not-exist") is None


def test_recent_reports_for_user(reports_store_db):
    rs = reports_store_db
    rid_open, _ = _submit(rs, user_id="u", request_id="r1")
    rid_res, _ = _submit(rs, user_id="u", request_id="r2")
    rs.update_report(rid_res, {"status": "resolved"})
    # null request_id row should also be excluded
    _submit(rs, user_id="u", request_id=None)
    recent = rs.recent_reports_for_user("u")
    ids = {r["id"] for r in recent}
    assert ids == {rid_open}                    # only open + request_id present
    assert rs.recent_reports_for_user("") == []


def test_counts_by_status(reports_store_db):
    rs = reports_store_db
    a, _ = _submit(rs, request_id="r1")
    b, _ = _submit(rs, request_id="r2")
    rs.update_report(b, {"status": "dismissed"})
    counts = rs.counts_by_status()
    assert counts == {"open": 1, "resolved": 0, "dismissed": 1}


# ---------------------------------------------------------------------------
# update_report
# ---------------------------------------------------------------------------

def test_update_status_toggles_resolved_ts(reports_store_db):
    rs = reports_store_db
    rid, _ = _submit(rs)
    row = rs.update_report(rid, {"status": "resolved"})
    assert row["status"] == "resolved" and row["resolved_ts"] is not None
    row = rs.update_report(rid, {"status": "open"})
    assert row["resolved_ts"] is None


def test_update_invalid_status_raises(reports_store_db):
    rs = reports_store_db
    rid, _ = _submit(rs)
    with pytest.raises(ValueError):
        rs.update_report(rid, {"status": "bogus"})


def test_update_admin_notes_capped(reports_store_db):
    rs = reports_store_db
    rid, _ = _submit(rs)
    row = rs.update_report(rid, {"admin_notes": "x" * 9000})
    assert len(row["admin_notes"]) == rs._CAP_ADMIN_NOTES


def test_update_missing_row_returns_none(reports_store_db):
    assert reports_store_db.update_report("nope", {"status": "open"}) is None


def test_update_empty_patch_returns_current(reports_store_db):
    rs = reports_store_db
    rid, _ = _submit(rs)
    assert rs.update_report(rid, {})["id"] == rid


# ---------------------------------------------------------------------------
# delete / clear_all
# ---------------------------------------------------------------------------

def test_delete_report(reports_store_db):
    rs = reports_store_db
    rid, _ = _submit(rs)
    assert rs.delete_report(rid) is True
    assert rs.delete_report(rid) is False


def test_clear_all(reports_store_db):
    rs = reports_store_db
    _submit(rs, request_id="r1")
    _submit(rs, request_id="r2")
    assert rs.clear_all() == 2
    assert rs.list_reports() == []


# ---------------------------------------------------------------------------
# _evict_to_cap (closed before open)
# ---------------------------------------------------------------------------

def test_evict_closed_before_open(reports_store_db, monkeypatch):
    import config
    monkeypatch.setattr(config, "REPORTS_MAX", 2, raising=False)
    rs = reports_store_db
    r1, _ = _submit(rs, request_id="r1")
    r2, _ = _submit(rs, request_id="r2")
    rs.update_report(r1, {"status": "resolved"})   # r1 is the only closed one
    # Inserting r3 pushes total to 3 > cap 2 -> evict 1 closed first (r1).
    r3, _ = _submit(rs, request_id="r3")
    assert rs.get_report(r1) is None
    assert rs.get_report(r2) is not None and rs.get_report(r3) is not None


def test_evict_cap_disabled(reports_store_db, monkeypatch):
    import config
    monkeypatch.setattr(config, "REPORTS_MAX", 0, raising=False)
    rs = reports_store_db
    for i in range(5):
        _submit(rs, request_id=f"r{i}")
    assert len(rs.list_reports()) == 5  # cap<1 disables eviction


# ---------------------------------------------------------------------------
# retention sweep
# ---------------------------------------------------------------------------

def test_sweep_retention_disabled(reports_store_db, monkeypatch):
    import config
    monkeypatch.setattr(config, "REPORTS_RETENTION_DAYS", 0, raising=False)
    _submit(reports_store_db)
    assert reports_store_db.sweep_retention() == 0


def test_sweep_retention_deletes_old(reports_store_db, monkeypatch):
    import config, time
    monkeypatch.setattr(config, "REPORTS_RETENTION_DAYS", 30, raising=False)
    rs = reports_store_db
    rid, _ = _submit(rs, request_id="old", trace_ts=1.0)
    # Backdate created_ts directly.
    rs._require_conn().execute(
        "UPDATE reports SET created_ts = ? WHERE id = ?",
        (time.time() - 40 * 86400, rid),
    )
    _submit(rs, request_id="new")
    assert rs.sweep_retention() == 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_truncate_steps_front_trim(reports_store_db, monkeypatch):
    import reports_store as rs
    monkeypatch.setattr(rs, "_CAP_STEPS_JSON", 80)
    steps = [(f"l{i}", "x" * 20, "y" * 20) for i in range(10)] + [("bad",), 5]
    out = rs._truncate_steps(steps)
    assert all(len(s) == 3 for s in out)
    assert out[-1][0] == "l9"  # newest kept


def test_row_to_dict_decodes_json(reports_store_db):
    rs = reports_store_db
    rid, _ = _submit(rs, corrections=[{"wrong": "a", "correct": "b"}])
    row = rs.get_report(rid)
    assert isinstance(row["corrections"], list) and isinstance(row["steps"], list)
