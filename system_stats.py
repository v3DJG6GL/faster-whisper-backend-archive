"""
System / GPU snapshot for the /stats dashboard.

Two libraries: nvidia-ml-py (NVML, optional) and psutil. Both are imported
defensively — on a host without an NVIDIA GPU or without the package, the
GPU panel disappears from the UI and the rest still works.

Per-model VRAM accounting: an NVML delta sample taken under main.py's
existing _model_load_lock around WhisperModel(...) construction. We have to
do it this way because per-PID VRAM via nvmlDeviceGetComputeRunningProcesses
returns NVML_VALUE_NOT_AVAILABLE on Windows WDDM (the default driver mode for
consumer cards with a display attached). Documented at:
  https://forums.developer.nvidia.com/t/nvml-problems-for-windows-not-available-in-wddm-driver-model/77557
This is the same constraint DeepSpeed and vLLM hit; the workaround pattern
(delta around construction, serialize loads under a lock) is theirs too.

The CTranslate2 caching allocator can make subsequent loads of the same
size under-report VRAM (cached freed memory gets reused). We don't try to
fight this — the first load's number is the trustworthy one; later
re-reports get whatever the delta showed.
"""

from __future__ import annotations

import os
import time
from threading import Lock
from typing import Any, Callable

# --- NVML init (optional, defensive) -----------------------------------------
NVML_OK = False
NVML_ERR: str | None = None
_nvml_handle: Any = None
try:
    import pynvml  # type: ignore[import-not-found]
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_OK = True
except Exception as e:                  # ImportError, NVMLError, ...
    NVML_ERR = f"{type(e).__name__}: {e}"

# --- psutil ------------------------------------------------------------------
import psutil

# Prime the non-blocking cpu_percent calls so the first /stats fetch returns
# real numbers (the documented psutil contract: first call returns 0.0).
psutil.cpu_percent(interval=None)
_proc = psutil.Process()
_proc.cpu_percent(interval=None)
_PROC_START_TS = _proc.create_time()


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    """Wrap NVML calls so a transient driver hiccup degrades to `default`
    instead of taking the whole /stats request down with a 500."""
    if not NVML_OK:
        return default
    try:
        return fn()
    except Exception:
        return default


# --- Per-model VRAM tracking -------------------------------------------------
# Populated by main.py around the WhisperModel(...) construction. Removed on
# LRU eviction. NOT a measurement loop — this is a registry of what the
# delta sample said at construction time.
_loaded_models_lock = Lock()
_loaded_models: dict[str, dict[str, Any]] = {}


def gpu_mem_used_bytes() -> int | None:
    """Return current global VRAM used (in bytes), or None if NVML unavailable."""
    if not NVML_OK:
        return None
    try:
        return int(pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle).used)
    except Exception:
        return None


def register_loaded_model(name: str, vram_bytes: int | None,
                          device: str, compute_type: str) -> None:
    """Called from main._get_or_load_model after a successful load. The VRAM
    delta sample comes from the caller — see main.py for the before/after
    dance under _model_load_lock."""
    with _loaded_models_lock:
        _loaded_models[name] = {
            "name": name,
            "device": device,
            "compute_type": compute_type,
            "vram_bytes": vram_bytes,
            "loaded_at": time.time(),
            "last_used": time.time(),
            # Monotonic counterpart for the idle-evictor's safe time math
            # (wall-clock can jump on NTP correction; monotonic cannot).
            "last_used_monotonic": time.monotonic(),
        }


def touch_loaded_model(name: str) -> None:
    """Bump last_used timestamp on cache hit — drives the warm/cold UI badge
    and the idle-evictor's eviction decision."""
    with _loaded_models_lock:
        info = _loaded_models.get(name)
        if info is not None:
            info["last_used"] = time.time()
            info["last_used_monotonic"] = time.monotonic()


def unregister_loaded_model(name: str) -> None:
    """Called from main._get_or_load_model when LRU eviction happens."""
    with _loaded_models_lock:
        _loaded_models.pop(name, None)


def loaded_models_snapshot() -> list[dict[str, Any]]:
    """Returned in /stats/snapshot. Sorted by load order (oldest first)."""
    with _loaded_models_lock:
        out = []
        now = time.time()
        for info in _loaded_models.values():
            mb = (info["vram_bytes"] / (1024 * 1024)) if info["vram_bytes"] else None
            out.append({
                "name": info["name"],
                "device": info["device"],
                "compute_type": info["compute_type"],
                "vram_mb": round(mb, 1) if mb is not None else None,
                "age_sec": round(now - info["loaded_at"], 1),
                "idle_sec": round(now - info["last_used"], 1),
            })
        return out


def _build_gpu() -> dict[str, Any] | None:
    if not NVML_OK:
        return None
    h = _nvml_handle
    name = _safe(lambda: pynvml.nvmlDeviceGetName(h))
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    mem = _safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(h))
    util = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(h))
    cuda_int = _safe(lambda: pynvml.nvmlSystemGetCudaDriverVersion_v2())
    cuda_str = (
        f"{cuda_int // 1000}.{(cuda_int % 1000) // 10}"
        if isinstance(cuda_int, int) else None
    )
    p_state = _safe(lambda: pynvml.nvmlDeviceGetPerformanceState(h))
    return {
        "name": name,
        "driver": _safe(lambda: pynvml.nvmlSystemGetDriverVersion()),
        "cuda": cuda_str,
        "mem_used_mb": round(mem.used / (1024 * 1024), 1) if mem else None,
        "mem_total_mb": round(mem.total / (1024 * 1024), 1) if mem else None,
        "util_pct": util.gpu if util else None,
        "temp_c": _safe(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)),
        "power_w": _safe(lambda: round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)),
        "power_limit_w": _safe(
            lambda: round(pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0, 1)
        ),
        "sm_clock_mhz": _safe(
            lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        ),
        "p_state": f"P{p_state}" if isinstance(p_state, int) else None,
    }


def _build_host() -> dict[str, Any]:
    vmem = psutil.virtual_memory()
    # Disk free on the drive containing the model cache (HF default location).
    cache_dir = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    try:
        disk_free_gb = round(psutil.disk_usage(cache_dir).free / (1024 ** 3), 1)
    except (OSError, FileNotFoundError):
        disk_free_gb = None
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "cpu_per_core": psutil.cpu_percent(interval=None, percpu=True),
        "ram_used_mb": round(vmem.used / (1024 * 1024), 1),
        "ram_total_mb": round(vmem.total / (1024 * 1024), 1),
        "ram_pct": vmem.percent,
        "disk_free_gb": disk_free_gb,
    }


def _build_process() -> dict[str, Any]:
    try:
        rss_mb = round(_proc.memory_info().rss / (1024 * 1024), 1)
        cpu = _proc.cpu_percent(interval=None)
        threads = _proc.num_threads()
    except psutil.Error:
        rss_mb = cpu = threads = None  # type: ignore[assignment]
    return {
        "pid": os.getpid(),
        "rss_mb": rss_mb,
        "cpu_pct": cpu,
        "threads": threads,
        "uptime_sec": round(time.time() - _PROC_START_TS, 1),
    }


def system_snapshot() -> dict[str, Any]:
    """Build a snapshot of GPU + host + process + loaded models.

    No TTL cache: the dominant consumer is the /stats/stream SSE generator
    which rebuilds every second (longer than any sub-second cache could
    survive), so a shared cache only helped the rare multi-tab snapshot
    burst, at the cost of an unsafe RMW on a module global."""
    return {
        "gpu": _build_gpu(),
        "gpu_error": NVML_ERR if not NVML_OK else None,
        "host": _build_host(),
        "process": _build_process(),
        "models": loaded_models_snapshot(),
    }


def shutdown() -> None:
    """Best-effort NVML shutdown — call from FastAPI's lifespan exit handler.

    Safe to call when NVML didn't init; safe to call twice."""
    global NVML_OK
    if NVML_OK:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        NVML_OK = False
