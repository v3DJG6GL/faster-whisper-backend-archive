"""Tests for captures_merge_proposer parts NOT covered by test_group_trim.py.

Already covered there: trimmed_duration_s (cache hit/miss + trim-disabled) and
_build_proposal trimmed durations. Here we cover: _bcp47_primary,
_normalize_text, _pick_text, _dur, _ratio, _build_sample_score boundaries,
_format_reason, _generate_candidates_for_bucket (dup-skip / <2 reject /
over-cap), _eligible (every rejection branch + boundaries), and propose_merges
(admin vs non-admin cache keys, per-(user,lang) partition, greedy
non-overlapping claim, TTL cache copy + was_cached)."""

import os

import pytest

import captures_merge_proposer as P


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inp,out", [
    (None, ""),
    ("", ""),
    ("de", "de"),
    ("de-CH", "de"),
    ("en_US", "en"),
    ("  DE-de  ", "de"),
])
def test_bcp47_primary(inp, out):
    assert P._bcp47_primary(inp) == out


@pytest.mark.parametrize("inp,out", [
    (None, ""),
    ("  Hello World  ", "hello world"),
    ("ABC", "abc"),
])
def test_normalize_text(inp, out):
    assert P._normalize_text(inp) == out


def test_pick_text_priority():
    assert P._pick_text({"text_for_training": "t", "final": "f", "raw": "r"}) == "t"
    assert P._pick_text({"final": "f", "raw": "r"}) == "f"
    assert P._pick_text({"raw": "r"}) == "r"
    assert P._pick_text({}) == ""
    # Empty strings are skipped in favor of the next populated field.
    assert P._pick_text({"text_for_training": "", "final": "f"}) == "f"


def test_dur_prefers_trim_then_raw():
    assert P._dur({"_trim_dur_s": 1.5, "duration_seconds": 9.0}) == 1.5
    assert P._dur({"duration_seconds": 9.0}) == 9.0
    assert P._dur({}) == 0.0
    # Explicit 0 trim is honored (not None).
    assert P._dur({"_trim_dur_s": 0.0, "duration_seconds": 9.0}) == 0.0


def test_ratio_bounds():
    assert P._ratio("", "x") == 0.0
    assert P._ratio("x", "") == 0.0
    assert P._ratio("abc", "abc") == 1.0
    assert 0.0 < P._ratio("hello", "hallo") < 1.0


# ---------------------------------------------------------------------------
# _build_sample_score
# ---------------------------------------------------------------------------

def _member(ts, dur, status="", trim=None):
    m = {"created_ts": ts, "duration_seconds": dur, "status": status}
    if trim is not None:
        m["_trim_dur_s"] = trim
    return m


def test_score_fill_peaks_at_target():
    # total_dur == target → fill_score 1.0.
    members = [_member(0, 13.0), _member(1, 12.7)]  # 25.7 + 0.3 gap = 26.0
    s = P._build_sample_score(members, 0.3, 26.0)
    assert s["fill_score"] == pytest.approx(1.0)


def test_score_density_neutral_below_three_members():
    members = [_member(0, 5.0), _member(1, 5.0)]
    s = P._build_sample_score(members, 0.3, 26.0)
    assert s["density_score"] == 0.5  # n<3 neutralized


def test_score_density_clamped_for_three_plus():
    members = [_member(0, 5.0), _member(1, 5.0), _member(2, 5.0)]
    s = P._build_sample_score(members, 0.3, 26.0)
    assert 0.0 <= s["density_score"] <= 1.0


def test_score_member_peaks_at_four():
    four = [_member(i, 2.0) for i in range(4)]
    s4 = P._build_sample_score(four, 0.3, 26.0)
    assert s4["member_score"] == pytest.approx(1.0)
    two = [_member(i, 2.0) for i in range(2)]
    s2 = P._build_sample_score(two, 0.3, 26.0)
    # |2-4|/8 = 0.25 → 0.75
    assert s2["member_score"] == pytest.approx(0.75)


def test_score_reviewed_boost():
    members = [_member(0, 5.0, "reviewed"), _member(1, 5.0, "reviewed")]
    s = P._build_sample_score(members, 0.3, 26.0)
    assert s["reviewed_count"] == 2
    # reviewed_boost = 0.1 * (2/2) = 0.1 baked into composite.
    none = [_member(0, 5.0), _member(1, 5.0)]
    s0 = P._build_sample_score(none, 0.3, 26.0)
    assert s["composite"] == pytest.approx(s0["composite"] + 0.1)


def test_score_uses_trimmed_durations():
    members = [_member(0, 10.0, trim=2.0), _member(1, 10.0, trim=2.0)]
    s = P._build_sample_score(members, 0.3, 26.0)
    # total uses trimmed (2+2) + gap.
    assert s["total_dur"] == pytest.approx(4.3)


# ---------------------------------------------------------------------------
# _format_reason
# ---------------------------------------------------------------------------

def test_format_reason_short_session():
    s = {"total_dur": 20.0, "wall_dur": 25.0}
    r = P._format_reason(s, 3)
    assert "3 clips" in r and "25 s session" in r and "packs to 20.0 s" in r


def test_format_reason_long_session_minutes():
    # wall materially larger (>1.5x total) and >= 60s → minutes form.
    s = {"total_dur": 20.0, "wall_dur": 125.0}
    r = P._format_reason(s, 4)
    assert "2 min 5 s session" in r


def test_format_reason_long_session_seconds_only():
    s = {"total_dur": 20.0, "wall_dur": 90.0}
    r = P._format_reason(s, 4)
    assert "1 min 30 s session" in r


# ---------------------------------------------------------------------------
# _generate_candidates_for_bucket
# ---------------------------------------------------------------------------

def _bkt_member(i, dur, text, ts=None):
    return {
        "id": f"c{i}", "created_ts": float(ts if ts is not None else i),
        "duration_seconds": dur, "_trim_dur_s": dur, "status": "",
        "text_for_training": text, "language": "de", "user_id": "u1",
    }


def test_bucket_single_over_cap_dropped():
    # Single-capture samples are allowed now. The 5 s member becomes a valid
    # solo candidate; the 30 s member is over the cap (alone or paired) → dropped.
    bucket = [_bkt_member(0, 5.0, "alpha"), _bkt_member(1, 30.0, "beta")]
    cands = P._generate_candidates_for_bucket(bucket, 0.3, 0.85, 26.0, 28.0)
    assert len(cands) == 1
    assert [m["id"] for m in cands[0][1]] == ["c0"]


def test_bucket_pairs_and_solo_when_fits():
    bucket = [_bkt_member(0, 5.0, "alpha"), _bkt_member(1, 5.0, "beta")]
    cands = P._generate_candidates_for_bucket(bucket, 0.3, 0.85, 26.0, 28.0)
    # i=0 yields the 2-member pack; i=1 yields a 1-member (solo) candidate.
    assert len(cands) == 2
    sizes = sorted(len(m) for _, m in cands)
    assert sizes == [1, 2]


def test_bucket_skips_duplicate_text():
    bucket = [
        _bkt_member(0, 5.0, "hello world"),
        _bkt_member(1, 5.0, "hello world"),  # near-identical → dup-skip
        _bkt_member(2, 5.0, "totally different content here"),
    ]
    cands = P._generate_candidates_for_bucket(bucket, 0.3, 0.85, 26.0, 28.0)
    # The i=0 candidate skips member 1 (dup) and adds member 2.
    top = max(cands, key=lambda c: len(c[1]))
    ids = [m["id"] for m in top[1]]
    assert "c1" not in ids
    assert "c0" in ids and "c2" in ids


def test_bucket_over_cap_skipped_smaller_packed():
    # Big middle member overflows; a later small member still packs (skip-allowed).
    bucket = [
        _bkt_member(0, 10.0, "aaaa"),
        _bkt_member(1, 25.0, "bbbb"),   # 10+25+0.3 > 28 → skipped
        _bkt_member(2, 10.0, "cccc"),   # 10+10+0.3 = 20.3 fits
    ]
    cands = P._generate_candidates_for_bucket(bucket, 0.3, 0.85, 26.0, 28.0)
    top = next(c for c in cands if c[1][0]["id"] == "c0")
    ids = [m["id"] for m in top[1]]
    assert ids == ["c0", "c2"]  # big member skipped, small one packed


# ---------------------------------------------------------------------------
# _eligible
# ---------------------------------------------------------------------------

def _row(**over):
    r = {
        "id": "x", "status": "new", "sample_id": None,
        "duration_seconds": 5.0, "language": "de",
        "text_for_training": "hello", "final": "", "raw": "",
    }
    r.update(over)
    return r


def test_eligible_happy_path():
    assert P._eligible(_row(), 1.0, 28.0) is True


@pytest.mark.parametrize("status", ["dismissed", "audio_missing"])
def test_eligible_rejects_status(status):
    assert P._eligible(_row(status=status), 1.0, 28.0) is False


def test_eligible_rejects_grouped():
    assert P._eligible(_row(sample_id="g1"), 1.0, 28.0) is False


def test_eligible_rejects_too_short():
    assert P._eligible(_row(duration_seconds=0.5), 1.0, 28.0) is False


def test_eligible_rejects_too_long():
    assert P._eligible(_row(duration_seconds=28.0 + 1), 1.0, 28.0) is False


def test_eligible_boundary_min_clip_inclusive():
    # dur == min_clip_s passes (not < min).
    assert P._eligible(_row(duration_seconds=1.0), 1.0, 28.0) is True


def test_eligible_boundary_hard_cap_inclusive():
    assert P._eligible(_row(duration_seconds=28.0), 1.0, 28.0) is True


def test_eligible_rejects_missing_language():
    assert P._eligible(_row(language=""), 1.0, 28.0) is False
    assert P._eligible(_row(language="   "), 1.0, 28.0) is False


def test_eligible_rejects_no_text():
    assert P._eligible(
        _row(text_for_training="", final="", raw=""), 1.0, 28.0) is False


def test_eligible_accepts_raw_only_text():
    assert P._eligible(
        _row(text_for_training="", final="", raw="r"), 1.0, 28.0) is True


# ---------------------------------------------------------------------------
# propose_merges
# ---------------------------------------------------------------------------

def _insert_eligible(cs, cid, *, ts, dur=10.0, text="some words here",
                     language="de", user_id="u1", status="new"):
    rel = os.path.join(cid[0:2], cid[2:4], f"{cid}.wav")
    cs._require_conn().execute(
        "INSERT INTO captures (id, created_ts, request_id, model, language,"
        " duration_seconds, audio_relpath, audio_format, raw, final,"
        " text_for_training, words_json, segments_json, status, user_id)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (cid, ts, None, "m", language, dur, rel, "wav", "raw", "final",
         text, "[]", "[]", status, user_id),
    )


@pytest.fixture
def trim_disabled(monkeypatch):
    # Make _dur fall back to raw duration_seconds so we don't need real audio.
    monkeypatch.setattr(P.cfg, "CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS", False,
                        raising=False)


def test_propose_basic_pair(captures_store_db, monkeypatch, trim_disabled):
    cs = captures_store_db
    _insert_eligible(cs, "cap00000000000a", ts=1000.0, text="alpha text one")
    _insert_eligible(cs, "cap00000000000b", ts=1001.0, text="beta text two")
    proposals, was_cached = P.propose_merges(
        user_id_filter=None, is_admin=True, caller_user_id="admin")
    assert was_cached is False
    assert len(proposals) == 1
    assert set(proposals[0]["member_ids"]) == {"cap00000000000a", "cap00000000000b"}
    assert proposals[0]["language"] == "de"


def test_propose_caches_and_returns_copy(captures_store_db, monkeypatch, trim_disabled):
    cs = captures_store_db
    _insert_eligible(cs, "cap0000000000c1", ts=1000.0, text="alpha")
    _insert_eligible(cs, "cap0000000000c2", ts=1001.0, text="bravo")
    first, c1 = P.propose_merges(
        user_id_filter=None, is_admin=True, caller_user_id="admin")
    second, c2 = P.propose_merges(
        user_id_filter=None, is_admin=True, caller_user_id="admin")
    assert c1 is False and c2 is True
    assert first == second
    # Returned list is a copy — mutating it must not poison the cache.
    second.append("junk")
    third, _ = P.propose_merges(
        user_id_filter=None, is_admin=True, caller_user_id="admin")
    assert "junk" not in third


def test_propose_admin_all_users_cache_key(captures_store_db, monkeypatch, trim_disabled):
    cs = captures_store_db
    _insert_eligible(cs, "cap0000000000d1", ts=1000.0)
    _insert_eligible(cs, "cap0000000000d2", ts=1001.0)
    P.propose_merges(user_id_filter=None, is_admin=True, caller_user_id="admin")
    assert P._ALL_USERS in P._CACHE


def test_propose_admin_specific_user_cache_key(captures_store_db, monkeypatch, trim_disabled):
    cs = captures_store_db
    _insert_eligible(cs, "cap0000000000e1", ts=1000.0, user_id="u1")
    _insert_eligible(cs, "cap0000000000e2", ts=1001.0, user_id="u1")
    P.propose_merges(user_id_filter="u1", is_admin=True, caller_user_id="admin")
    assert "u1" in P._CACHE
    assert P._ALL_USERS not in P._CACHE


def test_propose_non_admin_ignores_filter(captures_store_db, monkeypatch, trim_disabled):
    cs = captures_store_db
    _insert_eligible(cs, "cap0000000000f1", ts=1000.0, user_id="alice")
    _insert_eligible(cs, "cap0000000000f2", ts=1001.0, user_id="alice")
    _insert_eligible(cs, "cap0000000000g1", ts=1000.0, user_id="bob")
    _insert_eligible(cs, "cap0000000000g2", ts=1001.0, user_id="bob")
    # alice asks but passes bob's filter; non-admin → forced to her own id.
    proposals, _ = P.propose_merges(
        user_id_filter="bob", is_admin=False, caller_user_id="alice")
    assert "alice" in P._CACHE
    for p in proposals:
        assert p["user_id"] == "alice"


def test_propose_partitions_by_user_and_language(captures_store_db, monkeypatch, trim_disabled):
    cs = captures_store_db
    # Same session window, but different users + languages must not co-group.
    _insert_eligible(cs, "capu1de00000001", ts=1000.0, user_id="u1", language="de")
    _insert_eligible(cs, "capu1de00000002", ts=1001.0, user_id="u1", language="de")
    _insert_eligible(cs, "capu2en00000001", ts=1002.0, user_id="u2", language="en")
    _insert_eligible(cs, "capu2en00000002", ts=1003.0, user_id="u2", language="en")
    proposals, _ = P.propose_merges(
        user_id_filter=None, is_admin=True, caller_user_id="admin")
    # Every proposal is single-user, single-language.
    for p in proposals:
        ids = p["member_ids"]
        assert p["user_id"] in ("u1", "u2")
        prefix = "capu1de" if p["user_id"] == "u1" else "capu2en"
        assert all(i.startswith(prefix) for i in ids)


def test_propose_greedy_non_overlap_claim(captures_store_db, monkeypatch, trim_disabled):
    cs = captures_store_db
    # Three eligible clips; cap proposals to enforce one claim, ensure no
    # member appears in two proposals.
    for i, cid in enumerate(["capclaim0000001", "capclaim0000002", "capclaim0000003"]):
        _insert_eligible(cs, cid, ts=1000.0 + i, text=f"text variant {i}")
    proposals, _ = P.propose_merges(
        user_id_filter=None, is_admin=True, caller_user_id="admin")
    seen = set()
    for p in proposals:
        for mid in p["member_ids"]:
            assert mid not in seen  # no member claimed twice
            seen.add(mid)


def test_propose_respects_max_proposals(captures_store_db, monkeypatch, trim_disabled):
    cs = captures_store_db
    monkeypatch.setattr(P.cfg, "CAPTURES_PROPOSER_MAX_PROPOSALS", 1, raising=False)
    # Two separate sessions (gap > session gap) each yield a pair → 2 candidates,
    # but max caps output to 1.
    monkeypatch.setattr(P.cfg, "CAPTURES_PROPOSER_SESSION_GAP_S", 100, raising=False)
    _insert_eligible(cs, "capsess1000001a", ts=1000.0, text="aaa one")
    _insert_eligible(cs, "capsess1000001b", ts=1001.0, text="bbb two")
    _insert_eligible(cs, "capsess2000001a", ts=5000.0, text="ccc three")
    _insert_eligible(cs, "capsess2000001b", ts=5001.0, text="ddd four")
    proposals, _ = P.propose_merges(
        user_id_filter=None, is_admin=True, caller_user_id="admin")
    assert len(proposals) == 1


def test_propose_no_eligible_returns_empty(captures_store_db, monkeypatch, trim_disabled):
    cs = captures_store_db
    # A dismissed clip and a too-short clip → nothing eligible.
    _insert_eligible(cs, "capnone00000001", ts=1000.0, status="dismissed")
    _insert_eligible(cs, "capnone00000002", ts=1001.0, dur=0.2)
    proposals, was_cached = P.propose_merges(
        user_id_filter=None, is_admin=True, caller_user_id="admin")
    assert proposals == [] and was_cached is False


def test_propose_session_gap_splits(captures_store_db, monkeypatch, trim_disabled):
    cs = captures_store_db
    monkeypatch.setattr(P.cfg, "CAPTURES_PROPOSER_SESSION_GAP_S", 100, raising=False)
    # Clip 1 and 2 are far apart in time (> gap) → different sessions → can't be
    # grouped TOGETHER. With single-capture samples allowed, each isolated clip
    # is proposed as its own one-member sample (never a cross-session pair).
    _insert_eligible(cs, "capgap000000001", ts=1000.0, text="lonely one")
    _insert_eligible(cs, "capgap000000002", ts=9000.0, text="lonely two")
    proposals, _ = P.propose_merges(
        user_id_filter=None, is_admin=True, caller_user_id="admin")
    assert len(proposals) == 2
    for p in proposals:
        assert p["member_count"] == 1
