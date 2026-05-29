"""Tests for metrics ring-buffer math and snapshot shape.

Module globals are reset between tests by the autouse _reset_singletons
fixture in conftest.
"""

import time

import metrics


# ---------------------------------------------------------------------------
# _quantile (pure, nearest-rank)
# ---------------------------------------------------------------------------

def test_quantile_empty_is_zero():
    assert metrics._quantile([], 0.5) == 0.0


def test_quantile_single_element():
    assert metrics._quantile([7.0], 0.99) == 7.0


def test_quantile_p50_p95_p99():
    vals = [float(i) for i in range(1, 101)]  # 1..100 sorted
    assert metrics._quantile(vals, 0.50) == vals[int(round(0.50 * 99))]
    assert metrics._quantile(vals, 0.95) == vals[int(round(0.95 * 99))]
    assert metrics._quantile(vals, 0.99) == vals[int(round(0.99 * 99))]


def test_quantile_clamps_index():
    assert metrics._quantile([1.0, 2.0], 1.0) == 2.0
    assert metrics._quantile([1.0, 2.0], 0.0) == 1.0


# ---------------------------------------------------------------------------
# record_request
# ---------------------------------------------------------------------------

def test_record_request_counts_and_latency():
    metrics.record_request("/v1/models", 200, 12.5)
    assert metrics.req_count["/v1/models"] == 1
    assert metrics.err_count["/v1/models"] == 0
    assert list(metrics._latency) == [12.5]


def test_record_request_5xx_records_error():
    metrics.record_request("/x", 500, 5.0)
    assert metrics.err_count["/x"] == 1
    assert len(metrics._errors_ts) == 1


def test_record_request_4xx_not_error():
    metrics.record_request("/x", 404, 5.0)
    assert metrics.err_count["/x"] == 0
    assert len(metrics._errors_ts) == 0


def test_sse_paths_excluded_from_latency():
    for p in metrics.SSE_PATHS:
        metrics.record_request(p, 200, 999.0)
    assert len(metrics._latency) == 0


def test_error_window_prune():
    now = time.time()
    # Inject an old error timestamp beyond the 15-min window.
    metrics._errors_ts.append(now - (metrics._ERROR_WINDOW_SEC + 100))
    metrics.record_request("/x", 500, 1.0)
    # The stale ts must have been pruned, leaving only the fresh one.
    assert len(metrics._errors_ts) == 1
    assert metrics._errors_ts[0] >= now


def test_latency_ring_bounded():
    for i in range(metrics._LATENCY_MAX + 50):
        metrics.record_request("/x", 200, float(i))
    assert len(metrics._latency) == metrics._LATENCY_MAX


# ---------------------------------------------------------------------------
# _errors_in
# ---------------------------------------------------------------------------

def test_errors_in_window():
    now = time.time()
    # Append-ordered: oldest at the front, newest at the back.
    metrics._errors_ts.extend([now - 800, now - 120, now - 10])
    assert metrics._errors_in(60) == 1     # only the -10
    assert metrics._errors_in(300) == 2    # -10, -120
    assert metrics._errors_in(900) == 3    # all three


# ---------------------------------------------------------------------------
# record_model_load
# ---------------------------------------------------------------------------

def test_model_load_records():
    metrics.record_model_load("m", 3.0)
    assert metrics.model_loads["m"] == [3.0]


def test_model_load_bucket0_survives_trim():
    metrics.record_model_load("m", 1.0)  # canonical first
    for i in range(metrics._MODEL_LOAD_KEEP + 20):
        metrics.record_model_load("m", float(100 + i))
    bucket = metrics.model_loads["m"]
    assert len(bucket) == metrics._MODEL_LOAD_KEEP
    assert bucket[0] == 1.0  # first cold-load preserved forever


# ---------------------------------------------------------------------------
# record_transcription
# ---------------------------------------------------------------------------

def test_record_transcription_falsy_request_id_noop():
    # No request_id -> early return, no import/persist attempted, no raise.
    metrics.record_transcription("m", 1.0, 0.5, "ok", 3, request_id=None)
    metrics.record_transcription("m", 1.0, 0.5, "ok", 3, request_id="")


def test_record_transcription_swallows_store_errors():
    # Stores are not initialised here; the lazy record_timing/record_usage
    # calls raise RuntimeError internally but must be swallowed.
    metrics.record_transcription(
        "m", 1.0, 0.5, "ok", 3, request_id="req-1", user_id="u", key_id="k"
    )


# ---------------------------------------------------------------------------
# metrics_snapshot
# ---------------------------------------------------------------------------

def test_snapshot_shape_without_stores():
    metrics.record_request("/v1/models", 200, 10.0)
    metrics.record_request("/x", 500, 20.0)
    metrics.record_model_load("large-v2", 4.0)
    snap = metrics.metrics_snapshot()
    assert set(snap) >= {
        "uptime_sec", "in_flight_transcriptions", "requests", "errors_total",
        "errors_window", "latency_ms", "recent_transcriptions", "model_loads",
    }
    assert snap["requests"]["/v1/models"] == 1
    assert snap["errors_total"]["/x"] == 1
    assert snap["latency_ms"]["n"] == 2
    assert set(snap["errors_window"]) == {"1m", "5m", "15m"}
    assert snap["model_loads"]["large-v2"]["first"] == 4.0
    assert snap["model_loads"]["large-v2"]["count"] == 1
    # transcriptions_store not init -> list_recent raises -> recent=[]
    assert snap["recent_transcriptions"] == []


def test_snapshot_in_flight_reflected():
    metrics.in_flight_transcriptions = 3
    assert metrics.metrics_snapshot()["in_flight_transcriptions"] == 3
