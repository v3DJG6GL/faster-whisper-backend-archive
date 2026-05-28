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


def three_way_merge_corrections(
    baseline: list[dict[str, Any]] | None,
    edited: list[dict[str, Any]] | None,
    current: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge chip lists across a concurrent-edit window.

    Inputs:
      baseline — chips the client loaded with its GET (snapshot at T=0).
      edited   — chips the client wants after the user's edits (T=save).
      current  — chips currently in the DB (post any concurrent writes
                 that landed between T=0 and T=save, e.g. another admin
                 saving the same capture/group from another tab).

    Algorithm:
      Start from `current` (post-concurrent state). Then for each chip
      key in (baseline ∪ edited), apply only the user's delta:

        * key in baseline AND NOT in edited  → user removed it: drop it
          from output (idempotent if also gone from `current`).
        * key in edited AND NOT in baseline  → user added it: insert,
          overwriting any concurrent chip at the same key.
        * key in both AND payload differs    → user edited it: payload
          wins over any concurrent edit.
        * key in both AND payload equal      → untouched by the user:
          keep whatever `current` has at that key (which may itself be
          a concurrent edit).

    Merge key is `(idx, idx_end)` with idx_end defaulting to idx for
    anchored chips. Anchorless chips (no integer `idx`) bypass the
    three-way delta — they have no positional identity, so attempting
    to match baseline-edit pairs collapses every (None, None) entry into
    one and silently drops chips. Instead, the result keeps every
    anchorless chip from `current` plus every anchorless chip from
    `edited` (deduplicated by `(wrong, correct)`)."""
    def key(c: dict[str, Any]) -> "tuple[int, int] | None":
        i = c.get("idx")
        if not isinstance(i, int):
            return None
        e = c.get("idx_end")
        return (i, e if isinstance(e, int) else i)

    def _anchorless_id(c: dict[str, Any]) -> tuple[str, str]:
        return (str(c.get("wrong") or ""), str(c.get("correct") or ""))

    def _split(items):
        anchored: dict[tuple[int, int], dict[str, Any]] = {}
        anchorless: dict[tuple[str, str], dict[str, Any]] = {}
        for c in (items or []):
            if not isinstance(c, dict):
                continue
            k = key(c)
            if k is None:
                anchorless[_anchorless_id(c)] = c
            else:
                anchored[k] = c
        return anchored, anchorless

    base_anc, _ = _split(baseline)
    edit_anc, edit_free = _split(edited)
    cur_anc, cur_free = _split(current)

    out = dict(cur_anc)
    for k in set(base_anc) | set(edit_anc):
        in_b = k in base_anc
        in_e = k in edit_anc
        if in_b and not in_e:
            out.pop(k, None)
        elif in_e and not in_b:
            out[k] = edit_anc[k]
        elif base_anc[k] != edit_anc[k]:
            out[k] = edit_anc[k]
        # else: chip key in both with equal payload — user untouched it;
        # keep whatever `current` has so concurrent edits at that key
        # survive.

    def _sort_key(c: dict[str, Any]) -> tuple[int, int]:
        i = c.get("idx")
        try:
            return (0, int(i))
        except (TypeError, ValueError):
            return (1, 0)

    # Anchorless chips: union current + edited, deduped by (wrong, correct).
    # Edited entries win on collision (latest user submission overwrites).
    merged_free = {**cur_free, **edit_free}
    return sorted(out.values(), key=_sort_key) + sorted(
        merged_free.values(), key=_sort_key,
    )
