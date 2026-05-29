"""Tests for usage_store: hourly rollup UPSERT, aggregation, time bucketing,
leaderboard whitelist, retention prune, daily->hourly migration, and backfill.

Day/week bucketing is server-local; those tests pin TZ via the set_tz fixture
(POSIX-only; auto-skips on Windows). Pure UTC-hour helpers are tested without
TZ control.
"""

import datetime
import time


# ---------------------------------------------------------------------------
# Pure UTC-hour helpers
# ---------------------------------------------------------------------------

def test_now_hour_and_hour_for_ts():
    import usage_store as us
    assert us.hour_for_ts(7200) == 2
    assert us.hour_for_ts(7199.9) == 1
    assert abs(us.now_hour() - int(time.time() // 3600)) <= 1


# ---------------------------------------------------------------------------
# record_usage
# ---------------------------------------------------------------------------

def test_record_usage_accumulates_same_hour(usage_store_db):
    us = usage_store_db
    us.record_usage(key_id="k1", user_id="u1", audio_s=2.0, words=5, status="ok", hour=100)
    us.record_usage(key_id="k1", user_id="u1", audio_s=3.0, words=7, status="ok", hour=100)
    rows = us.totals_by_key()
    assert len(rows) == 1
    r = rows[0]
    assert r["requests"] == 2 and r["words"] == 12 and r["audio_s"] == 5.0
    assert r["errors"] == 0


def test_record_usage_error_counting(usage_store_db):
    us = usage_store_db
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok", hour=1)
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="error", hour=1)
    assert us.totals_by_key()[0]["errors"] == 1


def test_record_usage_open_mode_sentinel(usage_store_db):
    us = usage_store_db
    us.record_usage(key_id=None, user_id=None, audio_s=1.0, words=1, status="ok", hour=5)
    r = us.totals_by_key()[0]
    assert r["key_id"] == us.OPEN_MODE_ID
    assert r["user_id"] == us.OPEN_MODE_ID


def test_record_usage_never_raises_uninitialised():
    # No init_db here; _require_conn raises internally but record_usage swallows.
    import usage_store as us
    us._conn = None
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok")


# ---------------------------------------------------------------------------
# totals aggregations
# ---------------------------------------------------------------------------

def test_totals_by_key_ordered_by_audio_desc(usage_store_db):
    us = usage_store_db
    us.record_usage(key_id="small", user_id="u", audio_s=1.0, words=1, status="ok", hour=1)
    us.record_usage(key_id="big", user_id="u", audio_s=9.0, words=1, status="ok", hour=1)
    rows = us.totals_by_key()
    assert [r["key_id"] for r in rows] == ["big", "small"]


def test_totals_by_user_keyed(usage_store_db):
    us = usage_store_db
    us.record_usage(key_id="k1", user_id="a", audio_s=1.0, words=1, status="ok", hour=1)
    us.record_usage(key_id="k2", user_id="a", audio_s=1.0, words=1, status="ok", hour=1)
    us.record_usage(key_id="k3", user_id="b", audio_s=1.0, words=1, status="ok", hour=1)
    by_user = us.totals_by_user()
    assert by_user["a"]["requests"] == 2
    assert by_user["b"]["requests"] == 1


def test_totals_for_user_zeros_on_empty(usage_store_db):
    us = usage_store_db
    z = us.totals_for_user("nobody")
    assert z == {"requests": 0, "errors": 0, "words": 0, "audio_s": 0.0}


def test_totals_window_filtering(usage_store_db):
    us = usage_store_db
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok", hour=10)
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok", hour=20)
    # window [15, 25] should only see the hour=20 row.
    rows = us.totals_by_key(start_hour=15, end_hour=25)
    assert rows[0]["requests"] == 1


# ---------------------------------------------------------------------------
# leaderboard (SQL-injection guard on metric)
# ---------------------------------------------------------------------------

def test_leaderboard_invalid_metric_falls_back(usage_store_db):
    us = usage_store_db
    us.record_usage(key_id="k", user_id="u", audio_s=5.0, words=1, status="ok", hour=1)
    rows = us.leaderboard(metric="words); DROP TABLE usage_hourly;--")
    assert rows and rows[0]["user_id"] == "u"
    # Table survived the injection attempt.
    assert not us.is_empty()


def test_leaderboard_by_key_vs_user(usage_store_db):
    us = usage_store_db
    us.record_usage(key_id="k1", user_id="u", audio_s=1.0, words=1, status="ok", hour=1)
    us.record_usage(key_id="k2", user_id="u", audio_s=1.0, words=1, status="ok", hour=1)
    assert len(us.leaderboard(by="key")) == 2
    assert len(us.leaderboard(by="user")) == 1


# ---------------------------------------------------------------------------
# prune / is_empty
# ---------------------------------------------------------------------------

def test_is_empty(usage_store_db):
    us = usage_store_db
    assert us.is_empty() is True
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok", hour=1)
    assert us.is_empty() is False


def test_prune_noop_when_non_positive(usage_store_db):
    us = usage_store_db
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok", hour=1)
    assert us.prune(retention_days=0) == 0
    assert not us.is_empty()


def test_prune_drops_old(usage_store_db):
    us = usage_store_db
    old_hour = us.now_hour() - 24 * 100   # 100 days ago
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok", hour=old_hour)
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok")  # now
    deleted = us.prune(retention_days=30)
    assert deleted == 1


# ---------------------------------------------------------------------------
# series (server-local day/week bucketing)
# ---------------------------------------------------------------------------

def test_series_day_buckets_utc(usage_store_db, set_tz):
    set_tz("UTC")
    us = usage_store_db
    # Two rows on day 0 (hours 0..23) and one on day 1 (hour 24), in UTC.
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok", hour=0)
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok", hour=5)
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok", hour=24)
    s = us.series(bucket="day")
    assert [c["day"] for c in s] == [0, 1]
    assert s[0]["requests"] == 2 and s[1]["requests"] == 1


def test_series_week_buckets_utc(usage_store_db, set_tz):
    set_tz("UTC")
    us = usage_store_db
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok", hour=0)         # day 0
    us.record_usage(key_id="k", user_id="u", audio_s=1.0, words=1, status="ok", hour=24 * 8)     # day 8
    s = us.series(bucket="week")
    # day0 -> week 0; day8 -> 8 - 8%7 = 7
    assert [c["day"] for c in s] == [0, 7]


def test_epoch_day_and_local_midnight_utc(set_tz):
    set_tz("UTC")
    import usage_store as us
    assert us.epoch_day_for(0) == 0
    assert us.epoch_day_for(86400) == 1
    # local-midnight today, expressed as UTC epoch-hour, is a multiple of 24.
    assert us.local_day_start_hour(0) % 24 == 0


# ---------------------------------------------------------------------------
# _migrate_daily_to_hourly
# ---------------------------------------------------------------------------

def test_migrate_daily_to_hourly(usage_store_db, set_tz):
    set_tz("UTC")
    us = usage_store_db
    conn = us._require_conn()
    conn.executescript(
        "CREATE TABLE usage_daily (day INTEGER, key_id TEXT, user_id TEXT,"
        " requests INTEGER, errors INTEGER, words INTEGER, audio_s REAL);"
    )
    conn.execute(
        "INSERT INTO usage_daily VALUES (0, 'k', 'u', 3, 1, 10, 5.0)"
    )
    us._migrate_daily_to_hourly()
    # legacy table dropped
    has_daily = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='usage_daily'"
    ).fetchone()
    assert has_daily is None
    # lifetime totals preserved in the hourly table
    r = us.totals_by_key()[0]
    assert r["requests"] == 3 and r["words"] == 10 and r["audio_s"] == 5.0


# ---------------------------------------------------------------------------
# backfill_from_transcriptions
# ---------------------------------------------------------------------------

def test_backfill_from_transcriptions(usage_store_db, tx_store):
    us = usage_store_db
    # Seed a couple of recent-transcription rows.
    tx_store.record_timing(
        request_id="r1", model="m", audio_dur_s=2.0, proc_dur_s=1.0,
        status="ok", words_count=4, user_id="u1",
    )
    tx_store.record_timing(
        request_id="r2", model="m", audio_dur_s=3.0, proc_dur_s=1.0,
        status="error", words_count=1, user_id="u1",
    )
    n = us.backfill_from_transcriptions()
    assert n >= 1
    # Backfilled rows are bucketed under the synthetic key id.
    rows = us.totals_by_key()
    assert any(r["key_id"] == us.BACKFILL_ID for r in rows)
    # Second call is a no-op (table no longer empty).
    assert us.backfill_from_transcriptions() == 0
