"""Tests for capture_samples_store: _relpath_for, abs_path_for traversal
defense, _row_to_dict decoding, get_sample, list_samples (status filter only
when valid), get_members (corrections decode + bad-JSON fallback),
update_sample (whitelist + status validation + admin_notes cap), dissolve_sample
(NULLs member FKs + unlinks WAV), clear_all_samples, reconcile_on_startup
(missing WAV count, orphan file unlink, orphan-FK sweep), and the language
backfill in init()."""

import json
import os

import pytest


# ---------------------------------------------------------------------------
# Helpers — insert group rows + member captures directly via the shared conn.
# ---------------------------------------------------------------------------

def _insert_group(gs, sid, *, user_id="u1", created_ts=1.0, status="new",
                  language="de", transcript="hello", member_hashes_json="{}",
                  admin_notes="", member_trims_json="{}"):
    relpath = gs._relpath_for(sid)
    gs._require_conn().execute(
        "INSERT INTO capture_samples (id, user_id, created_ts,"
        " merged_wav_relpath, merged_duration_ms, transcript,"
        " transcript_join_strategy, member_hashes_json,"
        " inter_segment_silence_ms, is_stale, is_locked, status,"
        " admin_notes, language, member_trims_json)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (sid, user_id, created_ts, relpath, 5000, transcript, "space",
         member_hashes_json, 300, 0, 0, status, admin_notes, language,
         member_trims_json),
    )
    return relpath


def _insert_capture(cs, cid, *, sample_id=None, sample_order=None, language="de",
                    user_id="u1", created_ts=1.0, corrections_json="[]"):
    rel = os.path.join(cid[0:2], cid[2:4], f"{cid}.wav")
    cs._require_conn().execute(
        "INSERT INTO captures (id, created_ts, request_id, model, language,"
        " duration_seconds, audio_relpath, audio_format, raw, final,"
        " words_json, segments_json, corrections_json, status, user_id,"
        " sample_id, sample_order)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (cid, created_ts, None, "m", language, 2.0, rel, "wav", "raw",
         "final", "[]", "[]", corrections_json, "new", user_id,
         sample_id, sample_order),
    )


# ---------------------------------------------------------------------------
# _relpath_for / abs_path_for
# ---------------------------------------------------------------------------

def test_relpath_for_fanout(groups_store_db):
    rel = groups_store_db._relpath_for("abcdef0123")
    assert rel == os.path.join("groups", "ab", "cd", "abcdef0123.wav")


def test_abs_path_for_ok(groups_store_db, captures_store_db):
    abs_p = groups_store_db.abs_path_for(os.path.join("groups", "ab", "cd", "x.wav"))
    assert os.path.abspath(captures_store_db._require_audio_dir()) in abs_p


@pytest.mark.parametrize("bad", [
    os.path.join("..", "..", "escape.wav"),
    os.path.join("groups", "..", "..", "x.wav"),
    "/etc/passwd",
])
def test_abs_path_for_rejects_escape(groups_store_db, captures_store_db, bad):
    with pytest.raises(ValueError):
        groups_store_db.abs_path_for(bad)


# ---------------------------------------------------------------------------
# _row_to_dict / get_sample
# ---------------------------------------------------------------------------

def test_get_group_missing(groups_store_db):
    assert groups_store_db.get_sample("nope") is None


def test_row_to_dict_decodes_and_coerces(groups_store_db):
    gs = groups_store_db
    _insert_group(gs, "g00000000000000a", member_hashes_json='{"m1": "h1"}',
                  member_trims_json='{"m1": {"lead_ms": 5}}')
    g = gs.get_sample("g00000000000000a")
    assert g["member_hashes"] == {"m1": "h1"}
    assert g["member_trims"] == {"m1": {"lead_ms": 5}}
    assert g["is_stale"] is False and g["is_locked"] is False
    assert g["transcript_join_strategy"] == "space"
    assert g["language"] == "de"
    assert isinstance(g["created_ts"], float)


# ---------------------------------------------------------------------------
# list_samples
# ---------------------------------------------------------------------------

def test_list_groups_user_filter(groups_store_db):
    gs = groups_store_db
    _insert_group(gs, "g0000000000000u1", user_id="u1")
    _insert_group(gs, "g0000000000000u2", user_id="u2")
    assert len(gs.list_samples(user_id="u1")) == 1
    assert len(gs.list_samples()) == 2


def test_list_groups_valid_status_filter(groups_store_db):
    gs = groups_store_db
    _insert_group(gs, "g000000000000new", status="new")
    _insert_group(gs, "g00000000000rdy0", status="ready")
    assert len(gs.list_samples(status="ready")) == 1


def test_list_groups_invalid_status_ignored(groups_store_db):
    gs = groups_store_db
    _insert_group(gs, "g000000000000aaa", status="new")
    _insert_group(gs, "g000000000000bbb", status="ready")
    # An invalid status is silently ignored → no filter applied.
    assert len(gs.list_samples(status="garbage")) == 2


def test_list_groups_newest_first(groups_store_db):
    gs = groups_store_db
    _insert_group(gs, "g0000000000old00", created_ts=100.0)
    _insert_group(gs, "g0000000000new00", created_ts=200.0)
    ids = [g["id"] for g in gs.list_samples()]
    assert ids == ["g0000000000new00", "g0000000000old00"]


# ---------------------------------------------------------------------------
# get_members
# ---------------------------------------------------------------------------

def test_get_members_ordered_and_decodes_corrections(captures_store_db, groups_store_db):
    cs = captures_store_db
    gs = groups_store_db
    sid = "gmembers0000000a"
    _insert_group(gs, sid)
    _insert_capture(cs, "cap0000000000001", sample_id=sid, sample_order=1,
                    corrections_json='[{"wrong": "a", "correct": "b"}]')
    _insert_capture(cs, "cap0000000000000", sample_id=sid, sample_order=0)
    members = gs.get_members(sid)
    assert [m["sample_order"] for m in members] == [0, 1]  # ordered ASC
    assert members[1]["corrections"] == [{"wrong": "a", "correct": "b"}]
    assert members[0]["corrections"] == []


def test_get_members_bad_corrections_json_falls_back(captures_store_db, groups_store_db):
    cs = captures_store_db
    gs = groups_store_db
    sid = "gmembersbad00000"
    _insert_group(gs, sid)
    _insert_capture(cs, "capbad0000000001", sample_id=sid, sample_order=0,
                    corrections_json="{not json")
    members = gs.get_members(sid)
    assert members[0]["corrections"] == []


def test_get_members_non_list_corrections_falls_back(captures_store_db, groups_store_db):
    cs = captures_store_db
    gs = groups_store_db
    sid = "gmembersobj00000"
    _insert_group(gs, sid)
    # Valid JSON but not a list → coerced to [].
    _insert_capture(cs, "capobj0000000001", sample_id=sid, sample_order=0,
                    corrections_json='{"a": 1}')
    members = gs.get_members(sid)
    assert members[0]["corrections"] == []


def test_get_members_empty(groups_store_db):
    _insert_group(groups_store_db, "gempty0000000000")
    assert groups_store_db.get_members("gempty0000000000") == []


# ---------------------------------------------------------------------------
# update_sample
# ---------------------------------------------------------------------------

def test_update_group_whitelist_fields(groups_store_db):
    gs = groups_store_db
    sid = "gupdate000000001"
    _insert_group(gs, sid)
    g = gs.update_sample(sid, {
        "transcript": "new transcript",
        "inter_segment_silence_ms": 500,
        "is_locked": 1,
        "merged_duration_ms": 9999,
        "language": "en",
    })
    assert g["transcript"] == "new transcript"
    assert g["inter_segment_silence_ms"] == 500
    assert g["is_locked"] is True
    assert g["merged_duration_ms"] == 9999
    assert g["language"] == "en"


def test_update_group_unknown_field_raises(groups_store_db):
    gs = groups_store_db
    sid = "gupdate000000002"
    _insert_group(gs, sid)
    with pytest.raises(ValueError):
        gs.update_sample(sid, {"not_a_field": 1})


def test_update_group_invalid_status_raises(groups_store_db):
    gs = groups_store_db
    sid = "gupdate000000003"
    _insert_group(gs, sid)
    with pytest.raises(ValueError):
        gs.update_sample(sid, {"status": "bogus"})


def test_update_group_status_valid(groups_store_db):
    gs = groups_store_db
    sid = "gupdate000000004"
    _insert_group(gs, sid)
    assert gs.update_sample(sid, {"status": "ready"})["status"] == "ready"


def test_update_group_admin_notes_capped(groups_store_db):
    gs = groups_store_db
    sid = "gupdate000000005"
    _insert_group(gs, sid)
    g = gs.update_sample(sid, {"admin_notes": "x" * (gs._CAP_ADMIN_NOTES + 100)})
    assert len(g["admin_notes"]) == gs._CAP_ADMIN_NOTES


def test_update_group_empty_patch_returns_current(groups_store_db):
    gs = groups_store_db
    sid = "gupdate000000006"
    _insert_group(gs, sid)
    assert gs.update_sample(sid, {})["id"] == sid


# ---------------------------------------------------------------------------
# dissolve_sample
# ---------------------------------------------------------------------------

def test_dissolve_group_nulls_members_and_unlinks(captures_store_db, groups_store_db):
    cs = captures_store_db
    gs = groups_store_db
    sid = "gdissolve000000a"
    relpath = _insert_group(gs, sid)
    _insert_capture(cs, "capdissolve00001", sample_id=sid, sample_order=0)
    # Put a merged WAV on disk so dissolve has something to unlink.
    abs_p = gs.abs_path_for(relpath)
    os.makedirs(os.path.dirname(abs_p), exist_ok=True)
    with open(abs_p, "wb") as f:
        f.write(b"x")

    gs.dissolve_sample(sid)
    assert gs.get_sample(sid) is None
    assert not os.path.isfile(abs_p)
    # Member returned to flat list (sample_id/order NULLed).
    row = cs.get_capture("capdissolve00001")
    assert row["sample_id"] is None and row["sample_order"] is None


def test_dissolve_missing_group_is_noop(groups_store_db):
    # No exception, no-op.
    groups_store_db.dissolve_sample("does-not-exist")


# ---------------------------------------------------------------------------
# clear_all_samples
# ---------------------------------------------------------------------------

def test_clear_all_groups(groups_store_db):
    gs = groups_store_db
    _insert_group(gs, "gclear000000001a")
    _insert_group(gs, "gclear000000001b")
    assert gs.clear_all_samples() == 2
    assert gs.list_samples() == []


# ---------------------------------------------------------------------------
# reconcile_on_startup
# ---------------------------------------------------------------------------

def test_reconcile_counts_missing_wav(captures_store_db, groups_store_db):
    gs = groups_store_db
    _insert_group(gs, "greconmiss00000a")  # no WAV on disk
    missing, unlinked, orphan_fks = gs.reconcile_on_startup()
    assert missing == 1


def test_reconcile_unlinks_orphan_files(captures_store_db, groups_store_db):
    gs = groups_store_db
    groups_dir = gs._require_audio_root()
    sub = os.path.join(groups_dir, "ab", "cd")
    os.makedirs(sub, exist_ok=True)
    orphan = os.path.join(sub, "orphan.wav")
    with open(orphan, "wb") as f:
        f.write(b"x")
    tmpf = os.path.join(sub, "stale.wav.tmp")
    with open(tmpf, "wb") as f:
        f.write(b"x")
    missing, unlinked, orphan_fks = gs.reconcile_on_startup()
    assert not os.path.isfile(orphan)
    assert not os.path.isfile(tmpf)
    assert unlinked >= 2


def test_reconcile_keeps_known_wav(captures_store_db, groups_store_db):
    gs = groups_store_db
    sid = "greconkeep00000a"
    relpath = _insert_group(gs, sid)
    abs_p = gs.abs_path_for(relpath)
    os.makedirs(os.path.dirname(abs_p), exist_ok=True)
    with open(abs_p, "wb") as f:
        f.write(b"x")
    missing, unlinked, orphan_fks = gs.reconcile_on_startup()
    assert missing == 0
    assert os.path.isfile(abs_p)  # known WAV is not unlinked


def test_reconcile_clears_orphan_fks(captures_store_db, groups_store_db):
    cs = captures_store_db
    gs = groups_store_db
    # Capture pointing at a group id that doesn't exist.
    _insert_capture(cs, "caporphanfk00001", sample_id="ghost00000000000",
                    sample_order=0)
    missing, unlinked, orphan_fks = gs.reconcile_on_startup()
    assert orphan_fks == 1
    row = cs.get_capture("caporphanfk00001")
    assert row["sample_id"] is None and row["sample_order"] is None


def test_reconcile_keeps_valid_fk(captures_store_db, groups_store_db):
    cs = captures_store_db
    gs = groups_store_db
    sid = "gvalidfk0000000a"
    _insert_group(gs, sid)
    _insert_capture(cs, "capvalidfk000001", sample_id=sid, sample_order=0)
    _, _, orphan_fks = gs.reconcile_on_startup()
    assert orphan_fks == 0
    assert cs.get_capture("capvalidfk000001")["sample_id"] == sid


# ---------------------------------------------------------------------------
# init() language backfill
# ---------------------------------------------------------------------------

def test_init_language_backfill(captures_store_db, groups_store_db):
    """A group with no language, whose first member has one, gets backfilled
    on the next init()."""
    cs = captures_store_db
    gs = groups_store_db
    sid = "gbackfill000000a"
    # Group with empty language.
    _insert_group(gs, sid, language="")
    _insert_capture(cs, "capbackfill00001", sample_id=sid, sample_order=0,
                    language="fr")
    # Re-run init (shares the same conn) → correlated UPDATE backfills.
    import capture_samples_store
    capture_samples_store.init(cs._require_conn(), cs._require_audio_dir())
    assert gs.get_sample(sid)["language"] == "fr"


def test_init_language_backfill_skips_when_member_blank(captures_store_db, groups_store_db):
    cs = captures_store_db
    gs = groups_store_db
    sid = "gbackfillblank00"
    _insert_group(gs, sid, language="")
    # Member has no language → nothing to backfill.
    _insert_capture(cs, "capbackfillblk01", sample_id=sid, sample_order=0,
                    language="")
    import capture_samples_store
    capture_samples_store.init(cs._require_conn(), cs._require_audio_dir())
    assert gs.get_sample(sid)["language"] == ""
