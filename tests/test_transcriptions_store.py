"""Tests for transcriptions_store: trace/timing UPSERT merge, pagination,
prune, step truncation, and the row-to-dict derivations."""

import time


# ---------------------------------------------------------------------------
# record_trace / record_timing UPSERT
# ---------------------------------------------------------------------------

def test_trace_then_timing_merges_one_row(tx_store):
    ts = tx_store
    ts.record_trace(request_id="r1", model="m", raw="hallo welt", final="Hallo Welt",
                    tokens=["Hallo", "Welt"], language="de", user_id="u")
    ts.record_timing(request_id="r1", model="m", audio_dur_s=2.0, proc_dur_s=1.0,
                     status="ok", words_count=2, user_id="u")
    assert ts.count() == 1
    row = ts.list_recent(limit=10)[0]
    assert row["raw"] == "hallo welt" and row["final"] == "Hallo Welt"
    assert row["audio_dur"] == 2.0 and row["proc_dur"] == 1.0
    assert row["words"] == 2
    assert row["rtf"] == 2.0  # audio/proc


def test_timing_only_inserts_minimal_row(tx_store):
    ts = tx_store
    ts.record_timing(request_id="err1", model="m", audio_dur_s=None, proc_dur_s=0.5,
                     status="error", words_count=0)
    row = ts.list_recent(limit=10)[0]
    assert row["status"] == "error"
    # NULL text coerced to empty strings.
    assert row["raw"] == "" and row["final"] == ""
    assert row["rtf"] is None  # audio_dur_s None -> no rtf


def test_falsy_request_id_skipped(tx_store):
    ts = tx_store
    ts.record_trace(request_id="", model="m", raw="x", final="y")
    ts.record_timing(request_id="", model="m", audio_dur_s=1.0, proc_dur_s=1.0,
                     status="ok", words_count=1)
    assert ts.count() == 0


def test_trace_coalesce_preserves_user_on_timing(tx_store):
    ts = tx_store
    ts.record_trace(request_id="r", model="m", raw="a", final="b", user_id="alice")
    # timing with user_id None must not wipe the existing user (COALESCE).
    ts.record_timing(request_id="r", model="m", audio_dur_s=1.0, proc_dur_s=1.0,
                     status="ok", words_count=1, user_id=None)
    row = ts.list_recent(limit=1)[0]
    assert row["user_id"] == "alice"


# ---------------------------------------------------------------------------
# list_recent pagination
# ---------------------------------------------------------------------------

def test_list_recent_newest_first_and_limit(tx_store):
    ts = tx_store
    for i in range(5):
        ts.record_trace(request_id=f"r{i}", model="m", raw="x", final="y",
                        created_ts=1000.0 + i)
    rows = ts.list_recent(limit=3)
    assert len(rows) == 3
    assert [r["request_id"] for r in rows] == ["r4", "r3", "r2"]


def test_list_recent_before_ts_cursor(tx_store):
    ts = tx_store
    for i in range(5):
        ts.record_trace(request_id=f"r{i}", model="m", raw="x", final="y",
                        created_ts=1000.0 + i)
    rows = ts.list_recent(before_ts=1002.0, limit=10)
    # strictly older than 1002 -> r0 (1000), r1 (1001)
    assert [r["request_id"] for r in rows] == ["r1", "r0"]


def test_list_recent_before_ts_zero_ignored(tx_store):
    ts = tx_store
    ts.record_trace(request_id="r", model="m", raw="x", final="y", created_ts=10.0)
    assert len(ts.list_recent(before_ts=0, limit=10)) == 1


def test_list_recent_user_filter(tx_store):
    ts = tx_store
    ts.record_trace(request_id="a", model="m", raw="x", final="y", user_id="u1",
                    created_ts=1.0)
    ts.record_trace(request_id="b", model="m", raw="x", final="y", user_id="u2",
                    created_ts=2.0)
    rows = ts.list_recent(user_id_filter="u1", limit=10)
    assert [r["request_id"] for r in rows] == ["a"]


def test_list_recent_query_matches_raw_or_final(tx_store):
    ts = tx_store
    ts.record_trace(request_id="r1", model="m", raw="patient hat Fieber",
                    final="Patient hat Fieber", created_ts=1.0)
    ts.record_trace(request_id="r2", model="m", raw="kein treffer hier",
                    final="Aspirin verordnet", created_ts=2.0)
    ts.record_trace(request_id="r3", model="m", raw="nichts", final="nichts",
                    created_ts=3.0)
    # Matches raw of r1 and final of r2 (case-insensitive, ASCII).
    assert {r["request_id"] for r in ts.list_recent(query="fieber", limit=10)} == {"r1"}
    assert {r["request_id"] for r in ts.list_recent(query="ASPIRIN", limit=10)} == {"r2"}
    assert ts.list_recent(query="zzzznope", limit=10) == []


def test_list_recent_query_composes_with_user_and_cursor(tx_store):
    ts = tx_store
    ts.record_trace(request_id="a", model="m", raw="alpha note", final="x",
                    user_id="u1", created_ts=1.0)
    ts.record_trace(request_id="b", model="m", raw="alpha note", final="x",
                    user_id="u2", created_ts=2.0)
    ts.record_trace(request_id="c", model="m", raw="alpha note", final="x",
                    user_id="u1", created_ts=3.0)
    # query AND user_id_filter AND before_ts all compose.
    rows = ts.list_recent(query="alpha", user_id_filter="u1", before_ts=3.0, limit=10)
    assert [r["request_id"] for r in rows] == ["a"]


def test_list_recent_query_escapes_like_wildcards(tx_store):
    ts = tx_store
    ts.record_trace(request_id="lit", model="m", raw="50% done", final="x",
                    created_ts=1.0)
    ts.record_trace(request_id="other", model="m", raw="50 percent", final="x",
                    created_ts=2.0)
    # A literal "%" must match only the row containing it, not act as a wildcard.
    assert {r["request_id"] for r in ts.list_recent(query="50%", limit=10)} == {"lit"}


def test_list_recent_limit_floored_to_one(tx_store):
    ts = tx_store
    ts.record_trace(request_id="r", model="m", raw="x", final="y")
    assert len(ts.list_recent(limit=0)) == 1


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------

def test_prune_noop_when_both_zero(tx_store):
    ts = tx_store
    ts.record_trace(request_id="r", model="m", raw="x", final="y")
    assert ts.prune(max_rows=0, ttl_days=0) == 0
    assert ts.count() == 1


def test_prune_max_rows_keeps_newest(tx_store):
    ts = tx_store
    # Use non-zero timestamps: created_ts=0.0 is falsy and falls back to now().
    for i in range(10):
        ts.record_trace(request_id=f"r{i}", model="m", raw="x", final="y",
                        created_ts=1000.0 + i)
    deleted = ts.prune(max_rows=3, ttl_days=0)
    assert deleted == 7
    assert ts.count() == 3
    ids = {r["request_id"] for r in ts.list_recent(limit=10)}
    assert ids == {"r7", "r8", "r9"}


def test_prune_ttl_drops_old(tx_store):
    ts = tx_store
    ts.record_trace(request_id="old", model="m", raw="x", final="y",
                    created_ts=time.time() - 40 * 86400)
    ts.record_trace(request_id="new", model="m", raw="x", final="y")
    deleted = ts.prune(max_rows=0, ttl_days=30)
    assert deleted == 1
    assert {r["request_id"] for r in ts.list_recent(limit=10)} == {"new"}


def test_clear_all(tx_store):
    ts = tx_store
    for i in range(3):
        ts.record_trace(request_id=f"r{i}", model="m", raw="x", final="y")
    assert ts.clear_all() == 3
    assert ts.count() == 0


# ---------------------------------------------------------------------------
# _truncate_steps (pure)
# ---------------------------------------------------------------------------

def test_truncate_steps_drops_malformed():
    import transcriptions_store as ts
    steps = [("ok", "a", "b"), "bad", ("short",), [1, 2], ("x", "y", "z")]
    out = ts._truncate_steps(steps)
    assert out == [["ok", "a", "b"], ["x", "y", "z"]]


def test_truncate_steps_front_trim(monkeypatch):
    import transcriptions_store as ts
    monkeypatch.setattr(ts, "_CAP_STEPS_JSON", 80)
    steps = [(f"label{i}", "x" * 20, "y" * 20) for i in range(10)]
    out = ts._truncate_steps(steps)
    # Oldest (front) entries shed until the JSON blob fits the cap.
    assert len(out) < 10
    assert out[-1][0] == "label9"  # newest preserved


# ---------------------------------------------------------------------------
# _row_to_dict rtf guards
# ---------------------------------------------------------------------------

def test_row_to_dict_rtf_guard_proc_zero(tx_store):
    ts = tx_store
    ts.record_timing(request_id="z", model="m", audio_dur_s=5.0, proc_dur_s=0.0,
                     status="ok", words_count=1)
    row = ts.list_recent(limit=1)[0]
    assert row["rtf"] is None  # proc 0 -> guarded
