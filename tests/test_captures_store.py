"""Tests for captures_store: path helpers + traversal defense, _truncate_json
binary-search tail-trim, _safe_unlink, create_capture (success + cleanup),
list/iter/get/find/counts, update_capture validation + reviewed_ts toggle,
delete_capture (+ group auto-dissolve), clear_all, _evict_to_cap priority +
disabled, reconcile_on_startup, sweep_retention."""

import json
import os
import wave

import pytest

RATE = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_transcode(cs, monkeypatch, nbytes=1234):
    """Patch transcode_to_wav_16k_mono to write a minimal valid WAV at dst and
    return a byte count, so create_capture's insert path runs without ffmpeg."""
    import audio_transcode

    def _fake(src_path, dst_path):
        with wave.open(dst_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(b"\x00\x00" * 100)
        return nbytes

    monkeypatch.setattr(audio_transcode, "transcode_to_wav_16k_mono", _fake)


def _make(cs, monkeypatch, tmp_path, **over):
    """Create a capture row through the real create_capture (transcode faked)."""
    _fake_transcode(cs, monkeypatch)
    src = tmp_path / "src_in.bin"
    src.write_bytes(b"junk")
    kw = dict(
        audio_src_path=str(src),
        request_id="req-1",
        model="small",
        language="de",
        duration_seconds=3.0,
        raw="raw text",
        final="final text",
        words=[{"word": "hi", "start": 0.0, "end": 0.5}],
        segments=[{"text": "hi", "start": 0.0, "end": 0.5}],
        user_id="u1",
    )
    kw.update(over)
    return cs.create_capture(**kw)


# ---------------------------------------------------------------------------
# _relpath_for
# ---------------------------------------------------------------------------

def test_relpath_fanout_and_ext_lowercase(captures_store_db):
    cs = captures_store_db
    rel = cs._relpath_for("abcdef1234", ".WAV")
    assert rel == os.path.join("ab", "cd", "abcdef1234.wav")


def test_relpath_empty_ext_becomes_bin(captures_store_db):
    rel = captures_store_db._relpath_for("zzxxccvv", "")
    assert rel.endswith(".bin")


def test_relpath_ext_capped(captures_store_db):
    cs = captures_store_db
    rel = cs._relpath_for("aabbccdd", "x" * 50)
    # ext is capped at _CAP_AUDIO_FORMAT (16) chars.
    name = os.path.basename(rel)
    assert name == "aabbccdd." + "x" * cs._CAP_AUDIO_FORMAT


# ---------------------------------------------------------------------------
# abs_audio_path — path traversal defense
# ---------------------------------------------------------------------------

def test_abs_audio_path_ok(captures_store_db):
    cs = captures_store_db
    p = cs.abs_audio_path(os.path.join("ab", "cd", "x.wav"))
    assert os.path.abspath(cs._require_audio_dir()) in p


@pytest.mark.parametrize("bad", [
    os.path.join("..", "etc", "passwd"),
    os.path.join("ab", "..", "..", "escape.wav"),
    "/etc/passwd",
])
def test_abs_audio_path_rejects_escape(captures_store_db, bad):
    with pytest.raises(ValueError):
        captures_store_db.abs_audio_path(bad)


# ---------------------------------------------------------------------------
# _truncate_json
# ---------------------------------------------------------------------------

def test_truncate_json_under_cap_intact(captures_store_db):
    items = [{"a": 1}, {"b": 2}]
    out = captures_store_db._truncate_json(items, 10_000)
    assert json.loads(out) == items


def test_truncate_json_tail_trim(captures_store_db):
    cs = captures_store_db
    items = [{"w": f"word{i}", "start": i, "end": i + 1} for i in range(50)]
    full = json.dumps(items, ensure_ascii=False)
    cap = len(full) // 2
    out = cs._truncate_json(items, cap)
    kept = json.loads(out)
    assert len(out) <= cap
    assert 0 < len(kept) < len(items)
    # Front entries are kept (tail trimmed).
    assert kept == items[:len(kept)]


def test_truncate_json_non_serializable_returns_empty(captures_store_db):
    out = captures_store_db._truncate_json([{1, 2, 3}], 10_000)
    assert out == "[]"


# ---------------------------------------------------------------------------
# _safe_unlink
# ---------------------------------------------------------------------------

def test_safe_unlink_existing(captures_store_db, tmp_path):
    f = tmp_path / "gone.bin"
    f.write_bytes(b"x")
    assert captures_store_db._safe_unlink(str(f)) is True
    assert not f.exists()


def test_safe_unlink_missing_is_true(captures_store_db, tmp_path):
    assert captures_store_db._safe_unlink(str(tmp_path / "never.bin")) is True


# ---------------------------------------------------------------------------
# count / create_capture
# ---------------------------------------------------------------------------

def test_count_starts_zero(captures_store_db):
    assert captures_store_db.count() == 0


def test_create_capture_inserts_row_and_writes_audio(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    assert cs.count() == 1
    row = cs.get_capture(cid)
    assert row["status"] == "new"
    assert row["model"] == "small"
    assert row["language"] == "de"
    assert row["user_id"] == "u1"
    assert row["words"] == [{"word": "hi", "start": 0.0, "end": 0.5}]
    # Audio file actually exists at the resolved relpath.
    assert os.path.isfile(cs.abs_audio_path(row["audio_relpath"]))


def test_create_capture_truncates_text(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path, raw="r" * (cs._CAP_RAW + 100),
                final="f" * (cs._CAP_FINAL + 100))
    row = cs.get_capture(cid)
    assert len(row["raw"]) == cs._CAP_RAW
    assert len(row["final"]) == cs._CAP_FINAL


def test_create_capture_transcode_failure_cleans_tmp(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    import audio_transcode

    def _boom(src, dst):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(audio_transcode, "transcode_to_wav_16k_mono", _boom)
    src = tmp_path / "in.bin"
    src.write_bytes(b"x")
    with pytest.raises(RuntimeError):
        cs.create_capture(
            audio_src_path=str(src), request_id=None, model="m",
            language="de", duration_seconds=1.0, raw="r", final="f",
            words=[], segments=[],
        )
    assert cs.count() == 0
    # No .tmp left behind in the audio tree.
    leftovers = []
    for root, _d, files in os.walk(cs._require_audio_dir()):
        leftovers += [f for f in files if f.endswith(".tmp")]
    assert leftovers == []


def test_create_capture_insert_failure_unlinks_audio(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    _fake_transcode(cs, monkeypatch)
    # Force the INSERT to fail after the WAV is in place.
    real_conn = cs._require_conn()

    class FailingConn:
        def execute(self, sql, *a, **k):
            if sql.strip().startswith("INSERT"):
                raise RuntimeError("insert boom")
            return real_conn.execute(sql, *a, **k)

    monkeypatch.setattr(cs, "_require_conn", lambda: FailingConn())
    src = tmp_path / "in.bin"
    src.write_bytes(b"x")
    with pytest.raises(RuntimeError):
        cs.create_capture(
            audio_src_path=str(src), request_id=None, model="m",
            language="de", duration_seconds=1.0, raw="r", final="f",
            words=[], segments=[],
        )
    # The audio blob was unlinked on insert failure (no orphans).
    monkeypatch.setattr(cs, "_require_conn", lambda: real_conn)
    files = []
    for root, _d, fs in os.walk(cs._require_audio_dir()):
        files += [f for f in fs if f.endswith(".wav")]
    assert files == []


# ---------------------------------------------------------------------------
# list_captures
# ---------------------------------------------------------------------------

def test_list_captures_newest_first_drops_words(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path, request_id="a")
    cid2 = _make(cs, monkeypatch, tmp_path, request_id="b")
    rows = cs.list_captures()
    assert rows[0]["id"] == cid2  # newest first
    assert "words" not in rows[0] and "segments" not in rows[0]


def test_list_captures_status_filter(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    cs.update_capture(cid, {"status": "ready"})
    _make(cs, monkeypatch, tmp_path)  # stays new
    assert len(cs.list_captures(status="ready")) == 1
    assert len(cs.list_captures(status="new")) == 1
    # status 'all' and None => no filter.
    assert len(cs.list_captures(status="all")) == 2
    assert len(cs.list_captures(status=None)) == 2


def test_list_captures_user_filter(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path, user_id="u1")
    _make(cs, monkeypatch, tmp_path, user_id="u2")
    assert len(cs.list_captures(user_id="u1")) == 1
    assert len(cs.list_captures(user_id=None)) == 2  # None = no filter


def test_list_captures_before_ts_cursor(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid1 = _make(cs, monkeypatch, tmp_path)
    cid2 = _make(cs, monkeypatch, tmp_path)
    # Pin deterministic timestamps.
    conn = cs._require_conn()
    conn.execute("UPDATE captures SET created_ts=? WHERE id=?", (100.0, cid1))
    conn.execute("UPDATE captures SET created_ts=? WHERE id=?", (200.0, cid2))
    page = cs.list_captures(before_ts=150.0)
    assert [r["id"] for r in page] == [cid1]


# ---------------------------------------------------------------------------
# iter_captures_for_export
# ---------------------------------------------------------------------------

def test_iter_export_includes_words_oldest_first(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid1 = _make(cs, monkeypatch, tmp_path)
    cid2 = _make(cs, monkeypatch, tmp_path)
    conn = cs._require_conn()
    conn.execute("UPDATE captures SET created_ts=? WHERE id=?", (100.0, cid1))
    conn.execute("UPDATE captures SET created_ts=? WHERE id=?", (200.0, cid2))
    rows = list(cs.iter_captures_for_export())
    assert [r["id"] for r in rows] == [cid1, cid2]  # ascending
    assert "words" in rows[0]


def test_iter_export_filters(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path, user_id="u1")
    _make(cs, monkeypatch, tmp_path, user_id="u2")
    assert len(list(cs.iter_captures_for_export(user_id="u1"))) == 1
    assert len(list(cs.iter_captures_for_export(status="all"))) == 2


# ---------------------------------------------------------------------------
# get_capture / find_by_request_id / counts_by_status
# ---------------------------------------------------------------------------

def test_get_capture_missing(captures_store_db):
    assert captures_store_db.get_capture("nope") is None


def test_find_by_request_id(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path, request_id="shared")
    _make(cs, monkeypatch, tmp_path, request_id="shared")
    _make(cs, monkeypatch, tmp_path, request_id="other")
    assert len(cs.find_by_request_id("shared")) == 2
    assert cs.find_by_request_id("missing") == []


def test_counts_by_status(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    c1 = _make(cs, monkeypatch, tmp_path)
    _make(cs, monkeypatch, tmp_path)
    cs.update_capture(c1, {"status": "ready"})
    counts = cs.counts_by_status()
    assert counts["new"] == 1 and counts["ready"] == 1
    # All valid statuses are present as keys.
    assert set(counts) == set(cs._VALID_STATUS)


# ---------------------------------------------------------------------------
# update_capture
# ---------------------------------------------------------------------------

def test_update_invalid_status_raises(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        cs.update_capture(cid, {"status": "bogus"})


def test_update_status_toggles_reviewed_ts(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    row = cs.update_capture(cid, {"status": "reviewed"})
    assert row["status"] == "reviewed" and row["reviewed_ts"] is not None
    row = cs.update_capture(cid, {"status": "new"})
    assert row["reviewed_ts"] is None  # back to new clears it


def test_update_corrections_cleaned(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    # One valid (has correct), one dropped (no correct field).
    row = cs.update_capture(cid, {"corrections": [
        {"wrong": "a", "correct": "b"},
        {"wrong": "x", "correct": ""},
    ]})
    assert len(row["corrections"]) == 1
    assert row["corrections"][0]["correct"] == "b"


def test_update_caps_and_fields(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    row = cs.update_capture(cid, {
        "corrected_text": "c" * (cs._CAP_CORRECTED + 50),
        "admin_notes": "n" * (cs._CAP_ADMIN_NOTES + 50),
        "final": "newfinal",
        "text_for_training": "tft",
        "audio_trim_lead_ms": 120,
        "audio_trim_trail_ms": 80,
    })
    assert len(row["corrected_text"]) == cs._CAP_CORRECTED
    assert len(row["admin_notes"]) == cs._CAP_ADMIN_NOTES
    assert row["final"] == "newfinal"
    assert row["text_for_training"] == "tft"
    assert row["audio_trim_lead_ms"] == 120
    assert row["audio_trim_trail_ms"] == 80


def test_update_empty_patch_returns_current(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    assert cs.update_capture(cid, {})["id"] == cid


def test_update_missing_row_returns_none(captures_store_db):
    assert captures_store_db.update_capture("nope", {"status": "ready"}) is None


# ---------------------------------------------------------------------------
# delete_capture
# ---------------------------------------------------------------------------

def test_delete_capture_returns_bool_and_removes_audio(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    abs_p = cs.abs_audio_path(cs.get_capture(cid)["audio_relpath"])
    assert cs.delete_capture(cid) is True
    assert cs.get_capture(cid) is None
    assert not os.path.isfile(abs_p)
    assert cs.delete_capture(cid) is False  # already gone


def test_delete_member_auto_dissolves_group(captures_store_db, groups_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    gs = groups_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    # Manually create a group row + attach the capture as a member.
    conn = cs._require_conn()
    sid = "g0123456789abcdef"
    conn.execute(
        "INSERT INTO capture_samples (id, user_id, created_ts,"
        " merged_wav_relpath, merged_duration_ms, transcript,"
        " member_hashes_json) VALUES (?,?,?,?,?,?,?)",
        (sid, "u1", 1.0, gs._relpath_for(sid), 1000, "t", "{}"),
    )
    conn.execute(
        "UPDATE captures SET sample_id=?, sample_order=0 WHERE id=?", (sid, cid),
    )
    assert gs.get_sample(sid) is not None
    assert cs.delete_capture(cid) is True
    # Group auto-dissolved on member delete.
    assert gs.get_sample(sid) is None


# ---------------------------------------------------------------------------
# clear_all
# ---------------------------------------------------------------------------

def test_clear_all_wipes_rows_and_files(captures_store_db, groups_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path)
    _make(cs, monkeypatch, tmp_path)
    n = cs.clear_all(reporter_host="127.0.0.1")
    assert n == 2
    assert cs.count() == 0
    # Hex-fanout subdirs removed.
    remaining = [d for d in os.listdir(cs._require_audio_dir())
                 if os.path.isdir(os.path.join(cs._require_audio_dir(), d)) and d != "groups"]
    assert remaining == []


# ---------------------------------------------------------------------------
# _evict_to_cap
# ---------------------------------------------------------------------------

def test_evict_priority_dismissed_first(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    import config
    # Insert 3 rows then cap to 2 — eviction should drop the dismissed one
    # before any new/ready row (priority order).
    monkeypatch.setattr(config, "CAPTURES_MAX", 5000, raising=False)
    monkeypatch.setattr(config, "CAPTURES_MAX_MB", 5000, raising=False)
    c_dismissed = _make(cs, monkeypatch, tmp_path)
    c_new = _make(cs, monkeypatch, tmp_path)
    cs.update_capture(c_dismissed, {"status": "dismissed"})
    # Make the dismissed row the oldest so ASC ordering picks it anyway.
    conn = cs._require_conn()
    conn.execute("UPDATE captures SET created_ts=1 WHERE id=?", (c_dismissed,))
    conn.execute("UPDATE captures SET created_ts=2 WHERE id=?", (c_new,))
    # Now tighten cap to 2 and insert a 3rd, forcing 1 eviction.
    monkeypatch.setattr(config, "CAPTURES_MAX", 2, raising=False)
    _make(cs, monkeypatch, tmp_path)
    assert cs.get_capture(c_dismissed) is None      # dismissed evicted first
    assert cs.get_capture(c_new) is not None


def test_evict_excludes_group_members(captures_store_db, groups_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    gs = groups_store_db
    import config
    monkeypatch.setattr(config, "CAPTURES_MAX", 5000, raising=False)
    monkeypatch.setattr(config, "CAPTURES_MAX_MB", 5000, raising=False)
    member = _make(cs, monkeypatch, tmp_path)
    other = _make(cs, monkeypatch, tmp_path)
    sid = "gmemberexcl00000"
    conn = cs._require_conn()
    conn.execute(
        "INSERT INTO capture_samples (id, user_id, created_ts,"
        " merged_wav_relpath, merged_duration_ms, transcript,"
        " member_hashes_json) VALUES (?,?,?,?,?,?,?)",
        (sid, "u1", 1.0, gs._relpath_for(sid), 1000, "t", "{}"),
    )
    # member is the oldest, but it's in a group → must be protected.
    conn.execute("UPDATE captures SET sample_id=?, sample_order=0, created_ts=1 WHERE id=?", (sid, member))
    conn.execute("UPDATE captures SET created_ts=2 WHERE id=?", (other,))
    monkeypatch.setattr(config, "CAPTURES_MAX", 2, raising=False)
    _make(cs, monkeypatch, tmp_path)  # total now 3 > cap 2
    assert cs.get_capture(member) is not None  # group member never evicted
    assert cs.get_capture(other) is None       # the only evictable old row went


def test_evict_disabled_when_caps_below_one(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    import config
    monkeypatch.setattr(config, "CAPTURES_MAX", 0, raising=False)
    monkeypatch.setattr(config, "CAPTURES_MAX_MB", 0, raising=False)
    for _ in range(4):
        _make(cs, monkeypatch, tmp_path)
    assert cs.count() == 4  # no eviction


# ---------------------------------------------------------------------------
# reconcile_on_startup
# ---------------------------------------------------------------------------

def test_reconcile_marks_missing_and_unlinks_orphans(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    keep = _make(cs, monkeypatch, tmp_path)
    missing = _make(cs, monkeypatch, tmp_path)
    # Delete the audio file of `missing` directly (row stays).
    miss_abs = cs.abs_audio_path(cs.get_capture(missing)["audio_relpath"])
    os.unlink(miss_abs)
    # Drop an orphan file with no row, plus a stray .tmp.
    audio = cs._require_audio_dir()
    orphan_dir = os.path.join(audio, "zz", "zz")
    os.makedirs(orphan_dir, exist_ok=True)
    orphan = os.path.join(orphan_dir, "orphan.wav")
    with open(orphan, "wb") as f:
        f.write(b"x")
    tmpf = os.path.join(orphan_dir, "partial.wav.tmp")
    with open(tmpf, "wb") as f:
        f.write(b"x")

    marked, unlinked = cs.reconcile_on_startup()
    assert marked == 1
    assert cs.get_capture(missing)["status"] == "audio_missing"
    assert cs.get_capture(keep)["status"] == "new"
    assert not os.path.isfile(orphan)
    assert not os.path.isfile(tmpf)
    assert unlinked >= 2


def test_reconcile_skips_groups_subtree(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path)
    # A file under groups/ must NOT be treated as an orphan.
    groups_dir = os.path.join(cs._require_audio_dir(), "groups", "ab", "cd")
    os.makedirs(groups_dir, exist_ok=True)
    merged = os.path.join(groups_dir, "merged.wav")
    with open(merged, "wb") as f:
        f.write(b"x")
    cs.reconcile_on_startup()
    assert os.path.isfile(merged)  # groups/ subtree preserved


# ---------------------------------------------------------------------------
# sweep_retention
# ---------------------------------------------------------------------------

def test_sweep_retention_disabled(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    import config
    monkeypatch.setattr(config, "CAPTURES_RETENTION_DAYS", 0, raising=False)
    _make(cs, monkeypatch, tmp_path)
    assert cs.sweep_retention() == 0


def test_sweep_retention_deletes_old(captures_store_db, monkeypatch, tmp_path):
    import time
    cs = captures_store_db
    import config
    monkeypatch.setattr(config, "CAPTURES_RETENTION_DAYS", 30, raising=False)
    old = _make(cs, monkeypatch, tmp_path)
    _make(cs, monkeypatch, tmp_path)  # fresh
    conn = cs._require_conn()
    conn.execute("UPDATE captures SET created_ts=? WHERE id=?",
                 (time.time() - 40 * 86400, old))
    assert cs.sweep_retention() == 1
    assert cs.get_capture(old) is None


def test_sweep_retention_excludes_group_members(captures_store_db, groups_store_db, monkeypatch, tmp_path):
    import time
    cs = captures_store_db
    gs = groups_store_db
    import config
    monkeypatch.setattr(config, "CAPTURES_RETENTION_DAYS", 30, raising=False)
    member = _make(cs, monkeypatch, tmp_path)
    sid = "gretention000000"
    conn = cs._require_conn()
    conn.execute(
        "INSERT INTO capture_samples (id, user_id, created_ts,"
        " merged_wav_relpath, merged_duration_ms, transcript,"
        " member_hashes_json) VALUES (?,?,?,?,?,?,?)",
        (sid, "u1", 1.0, gs._relpath_for(sid), 1000, "t", "{}"),
    )
    conn.execute(
        "UPDATE captures SET sample_id=?, sample_order=0, created_ts=? WHERE id=?",
        (sid, time.time() - 99 * 86400, member),
    )
    # Member is ancient but grouped → never swept.
    assert cs.sweep_retention() == 0
    assert cs.get_capture(member) is not None
