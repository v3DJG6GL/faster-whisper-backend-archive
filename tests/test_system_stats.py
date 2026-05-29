"""Tests for system_stats: snapshot shape, the loaded-model registry
round-trip, and idempotent shutdown.

system_stats has import-time side effects (psutil priming, an NVML init
attempt that degrades gracefully). On this CI box NVML is absent, so the
snapshot's gpu is None and gpu_error is a non-empty string; assertions accept
either the absent-fallback (the real CI condition) or a present GPU.

The module keeps a process-global registry (_loaded_models); a local autouse
fixture clears it between tests so cases don't observe each other's writes.
"""

import time

import pytest

import system_stats


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset system_stats' module-global loaded-model registry."""
    with system_stats._loaded_models_lock:
        system_stats._loaded_models.clear()
    yield
    with system_stats._loaded_models_lock:
        system_stats._loaded_models.clear()


# ---------------------------------------------------------------------------
# system_snapshot
# ---------------------------------------------------------------------------

def test_snapshot_shape():
    snap = system_stats.system_snapshot()
    assert set(snap.keys()) == {"gpu", "gpu_error", "host", "process", "models"}
    assert isinstance(snap["host"], dict)
    assert isinstance(snap["process"], dict)
    assert isinstance(snap["models"], list)
    # On this box NVML is absent: gpu None + non-empty error. Accept a present
    # GPU too (dict + error None) so the test isn't box-specific.
    if snap["gpu"] is None:
        assert isinstance(snap["gpu_error"], str) and snap["gpu_error"]
    else:
        assert isinstance(snap["gpu"], dict)
        assert snap["gpu_error"] is None


def test_snapshot_host_fields():
    host = system_stats.system_snapshot()["host"]
    for key in ("cpu_pct", "cpu_per_core", "ram_used_mb", "ram_total_mb",
                "ram_pct"):
        assert key in host
    assert "disk_free_gb" in host
    assert isinstance(host["cpu_per_core"], list)
    assert host["ram_total_mb"] > 0


def test_snapshot_process_fields():
    import os
    proc = system_stats.system_snapshot()["process"]
    assert proc["pid"] == os.getpid()
    assert proc["uptime_sec"] >= 0
    for key in ("rss_mb", "cpu_pct", "threads"):
        assert key in proc


def test_snapshot_models_reflects_registry():
    system_stats.register_loaded_model("base", 1024 * 1024, "cpu", "int8")
    models = system_stats.system_snapshot()["models"]
    assert len(models) == 1
    assert models[0]["name"] == "base"


# ---------------------------------------------------------------------------
# Loaded-model registry round-trip
# ---------------------------------------------------------------------------

def test_register_and_snapshot():
    system_stats.register_loaded_model("m1", 512 * 1024 * 1024, "cuda", "float16")
    snap = system_stats.loaded_models_snapshot()
    assert len(snap) == 1
    e = snap[0]
    assert e["name"] == "m1"
    assert e["device"] == "cuda"
    assert e["compute_type"] == "float16"
    assert e["vram_mb"] == 512.0
    assert e["age_sec"] >= 0
    assert e["idle_sec"] >= 0


def test_register_none_vram():
    system_stats.register_loaded_model("m2", None, "cpu", "int8")
    e = system_stats.loaded_models_snapshot()[0]
    assert e["vram_mb"] is None


def test_touch_updates_last_used():
    system_stats.register_loaded_model("m3", None, "cpu", "int8")
    with system_stats._loaded_models_lock:
        # Backdate last_used so the touch is observable.
        system_stats._loaded_models["m3"]["last_used"] = time.time() - 100
    before = system_stats.loaded_models_snapshot()[0]["idle_sec"]
    assert before >= 99
    system_stats.touch_loaded_model("m3")
    after = system_stats.loaded_models_snapshot()[0]["idle_sec"]
    assert after < before


def test_touch_unknown_model_noop():
    # Touching a name that was never registered must not raise or create it.
    system_stats.touch_loaded_model("nope")
    assert system_stats.loaded_models_snapshot() == []


def test_unregister():
    system_stats.register_loaded_model("m4", None, "cpu", "int8")
    assert len(system_stats.loaded_models_snapshot()) == 1
    system_stats.unregister_loaded_model("m4")
    assert system_stats.loaded_models_snapshot() == []
    # Unregistering an absent model is a no-op.
    system_stats.unregister_loaded_model("m4")
    assert system_stats.loaded_models_snapshot() == []


def test_snapshot_ordered_by_insertion():
    system_stats.register_loaded_model("a", None, "cpu", "int8")
    system_stats.register_loaded_model("b", None, "cpu", "int8")
    names = [m["name"] for m in system_stats.loaded_models_snapshot()]
    assert names == ["a", "b"]


# ---------------------------------------------------------------------------
# gpu_mem_used_bytes
# ---------------------------------------------------------------------------

def test_gpu_mem_used_bytes():
    val = system_stats.gpu_mem_used_bytes()
    # NVML absent here -> None; if present, a non-negative int.
    if system_stats.NVML_OK:
        assert val is None or val >= 0
    else:
        assert val is None


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------

def test_shutdown_safe_and_idempotent():
    # Safe whether NVML inited or not; safe to call twice. After shutdown
    # NVML_OK must be False.
    system_stats.shutdown()
    assert system_stats.NVML_OK is False
    system_stats.shutdown()  # no raise on second call
    assert system_stats.NVML_OK is False
