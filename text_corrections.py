"""Shared schema for word-correction chips.

Both /reports (transcription-error reports) and /captures (fine-tuning
training samples) accept the same chip shape:

    {wrong: str, correct: str, idx: int?, idx_end: int?}

This module centralizes the cleaner so the two stores stay in lockstep —
adding a new chip-related field here automatically propagates to both
surfaces and to the future "promote a capture into a report" workflow.

The function lives here (not in either store) to avoid a circular
import: captures_store imports reports_store's helpers today would form
a chain through the route layer.
"""
from __future__ import annotations

from typing import Any

# Caps applied server-side before insert. The route layer already validates
# via Pydantic, but accept-then-trim is what protects against future code
# paths that bypass the route (admin scripts, migrations).
CAP_CORRECTION_FIELD = 200
CAP_CORRECTIONS = 50


def clean_corrections(items: list[Any] | None) -> list[dict[str, Any]]:
    """Filter to entries with a non-empty `correct` field, apply length
    caps, and cap the list at CAP_CORRECTIONS. Anything malformed is
    dropped silently — this is end-user input, we tolerate it.

    Optional `idx_end` lets a chip span multiple adjacent words from
    the original final text. Stored only when it's a valid int with
    `idx <= idx_end < 10_000` and `idx_end != idx`; otherwise the entry
    stays single-word."""
    out: list[dict[str, Any]] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        wrong = str(it.get("wrong", "") or "").strip()[:CAP_CORRECTION_FIELD]
        correct = str(it.get("correct", "") or "").strip()[:CAP_CORRECTION_FIELD]
        if not correct:
            continue
        entry: dict[str, Any] = {"wrong": wrong, "correct": correct}
        idx = it.get("idx")
        if isinstance(idx, int) and 0 <= idx < 10_000:
            entry["idx"] = idx
            idx_end = it.get("idx_end")
            if (isinstance(idx_end, int)
                    and idx <= idx_end < 10_000
                    and idx_end != idx):
                entry["idx_end"] = idx_end
        out.append(entry)
        if len(out) >= CAP_CORRECTIONS:
            break
    return out
