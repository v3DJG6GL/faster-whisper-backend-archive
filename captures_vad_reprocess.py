"""Background job: re-merge every capture SAMPLE's audio with the CURRENT
global silence settings (CAPTURES_VAD_MARGIN_GROUP_* / inter-member silence
and the sample-duration cap).

Use after changing the global VAD/silence settings so existing samples adopt
them — the per-sample "Regenerate" button does one sample; this does all.

Scope:
  - Re-runs _build_merged_wav (per-member VAD trim + uniform-silence layout)
    for each sample, rebuilding the merged WAV + member_trims + duration, and
    stamping the sample's inter_segment_silence_ms with the new global value.
  - SKIPS locked samples (admin-frozen exported training data) — mirrors the
    pipeline-reapply skip.
  - If a sample no longer fits the cap under the new settings, it is marked
    `is_stale=true` (export already skips stale) — never truncated.
  - Member capture rows are NEVER touched (only the derived merged artifacts),
    so ungroup/dissolve still restores raw captures + timestamps.
  - Transcript / chips / status / admin_notes are untouched (audio-only job).

Single-worker model (mirrors captures_reapply): at most one job at a time;
state in process memory; a restart wipes it (just click again).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger("whisper-api")

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "status":       "idle",     # idle | running | done | error
    "started_ts":   None,
    "finished_ts":  None,
    "total":        0,
    "processed":    0,
    "rebuilt":      0,
    "skipped":      0,          # locked / no members
    "stale":        0,          # over-cap or build failure → flagged stale
    "error":        None,
}
_worker: "threading.Thread | None" = None


def status() -> dict[str, Any]:
    with _state_lock:
        return dict(_state)


def start() -> dict[str, Any]:
    """Idempotent: a second call while running returns the current state."""
    global _worker
    with _state_lock:
        if _state["status"] == "running":
            return dict(_state)
        _state.update({
            "status":      "running",
            "started_ts":  time.time(),
            "finished_ts": None,
            "total":       0,
            "processed":   0,
            "rebuilt":     0,
            "skipped":     0,
            "stale":       0,
            "error":       None,
        })
    _worker = threading.Thread(target=_run, daemon=True, name="reprocess-vad")
    _worker.start()
    with _state_lock:
        return dict(_state)


def _run() -> None:
    try:
        import capture_samples_store
        import config as cfg
        from captures_routes import (
            _build_merged_wav, _merged_wav_patch, _global_silence_ms,
            _get_rebuild_lock,
        )

        samples = capture_samples_store.list_samples(user_id=None)
        with _state_lock:
            _state["total"] = len(samples)

        for s in samples:
            sid = s["id"]
            try:
                if s.get("is_locked"):
                    with _state_lock:
                        _state["skipped"] += 1
                    continue
                members = capture_samples_store.get_members(sid)
                if not members:
                    with _state_lock:
                        _state["skipped"] += 1
                    continue
                silence_ms = _global_silence_ms()
                try:
                    with _get_rebuild_lock(sid):
                        duration_ms, hashes, member_trims = _build_merged_wav(
                            sid=sid,
                            member_ids=[m["id"] for m in members],
                            silence_ms=silence_ms,
                        )
                except Exception as e:
                    # Over-cap under the new settings (merge_wavs raises) or a
                    # build failure → flag stale (excluded from export), never
                    # truncate. The old merged WAV stays in place.
                    logger.info(
                        "[reprocess-vad] sample %s → stale (%s)", sid[:8], e,
                    )
                    capture_samples_store.update_sample(sid, {"is_stale": 1})
                    with _state_lock:
                        _state["stale"] += 1
                    continue
                patch = _merged_wav_patch(duration_ms, hashes, member_trims)
                patch["inter_segment_silence_ms"] = silence_ms
                capture_samples_store.update_sample(sid, patch)
                with _state_lock:
                    _state["rebuilt"] += 1
            finally:
                with _state_lock:
                    _state["processed"] += 1

        with _state_lock:
            _state["status"] = "done"
            _state["finished_ts"] = time.time()
        logger.info(
            "[reprocess-vad] done: %d/%d samples, %d rebuilt, %d stale, %d skipped",
            _state["processed"], _state["total"], _state["rebuilt"],
            _state["stale"], _state["skipped"],
        )
    except Exception as e:
        logger.exception("[reprocess-vad] job failed")
        with _state_lock:
            _state["status"] = "error"
            _state["error"] = str(e)
            _state["finished_ts"] = time.time()
