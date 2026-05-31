"""Background job: re-run the current PIPELINE_RULES over every
existing capture's `raw` text and update `final` + `text_for_training`.

Scope:
  - Touches `final` and `text_for_training` (the latter is rebuilt
    with the captures-specific `CAPTURES_PIPELINE_RULES_EXCLUDE` set,
    or copied from `final` when no excludes are configured).
    `corrected_text` (admin free-form ground truth) and
    `corrections_json` (chip corrections, index-based and
    rule-independent) stay untouched.
  - For each affected member that belongs to an unlocked group,
    rebuild the group's snapshot `transcript` from the current member
    text via the existing _build_default_transcript helper. Locked
    groups are skipped — they're exported training samples.
  - No audio re-merge. Pipeline rules only affect text; merged WAV
    bytes are unchanged.

Single-worker model:
  - At most one job runs at a time. Concurrent start() returns the
    running job's current state.
  - Job state lives in process memory. A service restart wipes it
    — acceptable, since the user just clicks the button again.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger("whisper-api")

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "status":           "idle",     # idle | running | done | error
    "started_ts":       None,
    "finished_ts":      None,
    "total":            0,
    "processed":        0,
    "captures_updated": 0,
    "groups_updated":   0,
    "error":            None,
}
_worker: "threading.Thread | None" = None


def status() -> dict[str, Any]:
    with _state_lock:
        return dict(_state)


def start() -> dict[str, Any]:
    """Idempotent: if a job is running, return its current state
    instead of spawning a second worker."""
    global _worker
    with _state_lock:
        if _state["status"] == "running":
            return dict(_state)
        _state.update({
            "status":           "running",
            "started_ts":       time.time(),
            "finished_ts":      None,
            "total":            0,
            "processed":        0,
            "captures_updated": 0,
            "groups_updated":   0,
            "error":            None,
        })
    _worker = threading.Thread(target=_run, daemon=True, name="reapply-rules")
    _worker.start()
    with _state_lock:
        return dict(_state)


def _run() -> None:
    try:
        import main
        import captures_store
        import capture_samples_store
        import config as cfg

        captures_excludes = getattr(cfg, "CAPTURES_PIPELINE_RULES_EXCLUDE", None)

        conn = captures_store._require_conn()
        total_row = conn.execute("SELECT COUNT(*) FROM captures").fetchone()
        with _state_lock:
            _state["total"] = int(total_row[0]) if total_row else 0

        affected_sample_ids: set[str] = set()
        # Materialise the small projection up front — captures_store.update_capture
        # writes back to the same connection inside the loop, and an open
        # cursor on the same connection can skip/revisit rows when the
        # underlying table is mutated mid-walk. Payload is just ids +
        # short text columns (no words_json / segments_json), so memory
        # stays bounded even at tens of thousands of rows.
        rows = conn.execute(
            "SELECT id, raw, final, text_for_training, model, sample_id"
            " FROM captures ORDER BY created_ts DESC"
        ).fetchall()
        for r in rows:
            cid = r["id"]
            raw_text = r["raw"] or ""
            patch: dict[str, str] = {}
            try:
                new_final = main._postprocess_text(
                    raw_text, model_name=r["model"],
                )
            except Exception as e:
                logger.warning(
                    "[reapply] capture %s skipped: %s", cid[:8], e,
                )
                with _state_lock:
                    _state["processed"] += 1
                continue
            if new_final != (r["final"] or ""):
                patch["final"] = new_final
                if r["sample_id"]:
                    affected_sample_ids.add(r["sample_id"])
            # Training-form text reflects PIPELINE_RULES minus the
            # captures-specific excludes. When no excludes are configured
            # the pipeline output is identical — skip the second run.
            if captures_excludes:
                try:
                    new_training = main._postprocess_text(
                        raw_text, model_name=r["model"],
                        extra_excludes=captures_excludes,
                    )
                except Exception as e:
                    logger.warning(
                        "[reapply] capture %s training-form skipped: %s",
                        cid[:8], e,
                    )
                    new_training = None
            else:
                new_training = new_final
            if new_training is not None and new_training != (r["text_for_training"] or ""):
                patch["text_for_training"] = new_training
                # _build_default_transcript reads text_for_training before
                # falling back to final/raw, so a training-form change must
                # also trigger a group rebuild — final may be unchanged when
                # captures_excludes drops a rule from the training pipeline.
                if r["sample_id"]:
                    affected_sample_ids.add(r["sample_id"])
            if patch:
                captures_store.update_capture(cid, patch)
                with _state_lock:
                    _state["captures_updated"] += 1
            with _state_lock:
                _state["processed"] += 1

        if affected_sample_ids:
            from captures_routes import _build_default_transcript
            for sid in affected_sample_ids:
                g = capture_samples_store.get_sample(sid)
                if g is None or g.get("is_locked"):
                    continue
                members = capture_samples_store.get_members(sid)
                new_t = _build_default_transcript(
                    members, g.get("transcript_join_strategy") or "space",
                )
                if new_t != (g.get("transcript") or ""):
                    capture_samples_store.update_sample(
                        sid, {"transcript": new_t},
                    )
                    with _state_lock:
                        _state["groups_updated"] += 1

        with _state_lock:
            _state["status"] = "done"
            _state["finished_ts"] = time.time()
        logger.info(
            "[reapply] done: %d/%d captures, %d updated, %d groups",
            _state["processed"], _state["total"],
            _state["captures_updated"], _state["groups_updated"],
        )
    except Exception as e:
        logger.exception("[reapply] job failed")
        with _state_lock:
            _state["status"] = "error"
            _state["error"] = str(e)
            _state["finished_ts"] = time.time()
