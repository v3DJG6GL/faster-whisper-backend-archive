"""Admin /captures page + JSON APIs for Whisper fine-tuning data review.

Layout:

  GET    /captures                       HTML page (admin host + token)
  GET    /captures/api/list              paginated metadata (no audio)
  GET    /captures/api/{cid}             single row incl. word timestamps
  GET    /captures/api/by-request/{rid}  cross-link from /reports
  GET    /captures/api/{cid}/audio       streams the raw audio (Range OK)
  PATCH  /captures/api/{cid}             corrections / corrected_text /
                                         admin_notes / status
  DELETE /captures/api/{cid}             single delete
  POST   /captures/api/clear             typed-confirmation wipe
  GET    /captures/api/export            tar.gz (manifest.jsonl + audio/)

Mutating routes use a HEADER-ONLY admin-token guard (no ?token= fallback).
The audio endpoint is GET and also header-only — browsers consume it via
fetch() + URL.createObjectURL(blob), not <audio src=...>, so no token in
URL is needed.

Word-timestamp + chip schema matches /reports' (text_corrections), so a
future "promote a capture into a report" flow needs no translation.
"""
from __future__ import annotations

import io
import json
import logging
import os
import tarfile
import threading
import time
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse,
)
from pydantic import BaseModel, Field

import api_keys_store
import captures_store
import config as cfg
import text_corrections
import web_common
from admin_routes import require_admin_host
from auth import get_current_user, require_admin

logger = logging.getLogger("whisper-api")

router = APIRouter()


# ---------------------------------------------------------------------
# Rate limit per host on the audio-streaming endpoint
# ---------------------------------------------------------------------
# Bursty admins double-clicking review cards shouldn't be punished, but a
# misbehaving script shouldn't be able to pull GB/s. 60 audio fetches per
# 60s gives ~1 row/sec which is far above any human review cadence.

_AUDIO_RATE_WINDOW_S = 60.0
_AUDIO_RATE_MAX = 60
_audio_rate: dict[str, tuple[int, float]] = {}


def _check_audio_rate(host: str) -> None:
    key = host or "<unknown>"
    now = time.time()
    n, start = _audio_rate.get(key, (0, now))
    if now - start > _AUDIO_RATE_WINDOW_S:
        n, start = 0, now
    n += 1
    _audio_rate[key] = (n, start)
    if n > _AUDIO_RATE_MAX:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many audio requests from this host.",
        )


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------

class CorrectionIn(BaseModel):
    model_config = {"extra": "forbid"}
    wrong: str = ""
    correct: str = ""
    idx: int | None = None
    idx_end: int | None = None


class PatchCaptureIn(BaseModel):
    model_config = {"extra": "forbid"}
    status: Literal["new", "reviewed", "ready", "dismissed"] | None = None
    corrected_text: str | None = None
    corrections: list[CorrectionIn] | None = None
    # Snapshot of `corrections` the client loaded with this capture.
    # When provided alongside `corrections`, the server applies a
    # three-way merge against the current DB state so a concurrent
    # write (report cascade, another admin in another tab) doesn't
    # get clobbered by the user's save. Omitted → legacy replace.
    baseline_corrections: list[CorrectionIn] | None = None
    admin_notes: str | None = None


class ClearIn(BaseModel):
    model_config = {"extra": "forbid"}
    # Typed confirmation — the literal string "CAPTURES" must be sent.
    # Training data is irrecoverable; the modal asks the admin to type it.
    confirm: str = Field(default="", max_length=32)


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

@router.get(
    "/captures",
    dependencies=[Depends(require_admin_host)],
    response_class=HTMLResponse,
)
async def captures_page() -> HTMLResponse:
    if not getattr(cfg, "ADMIN_UI_ENABLED", False):
        return HTMLResponse("Admin UI disabled.", status_code=404)
    return HTMLResponse(
        web_common.render_page(_CAPTURES_HTML, current="captures"),
        media_type="text/html",
    )


# ---------------------------------------------------------------------
# JSON APIs
# ---------------------------------------------------------------------

@router.get(
    "/captures/api/list",
    dependencies=[Depends(require_admin_host)],
)
async def list_captures_api(
    status_filter: str = Query("all", alias="status"),
    limit: int = Query(200, ge=1, le=1000),
    before_ts: float | None = Query(None),
    user_filter: str | None = Query(None, alias="user_id"),
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Admin sees all; non-admin sees only their own captures. Admin can
    additionally narrow by `?user_id=...` for the per-user dropdown."""
    if not user.get("is_admin"):
        effective_user = user.get("user_id")
    else:
        effective_user = user_filter  # None = show all
    rows = captures_store.list_captures(
        status=status_filter, limit=limit, before_ts=before_ts,
        user_id=effective_user,
    )
    # Per-row pipeline self-heal happens in get_capture_api (expand) only.
    # Running it here would be 2 _postprocess_text calls × `limit` rows per
    # list render, which dominates response time on /captures with limit=500.
    usernames = api_keys_store.get_usernames([r.get("user_id") for r in rows])
    for r in rows:
        _apply_trim_to_capture_row(r)
        r["username"] = usernames.get(r.get("user_id"))
    return JSONResponse({
        "captures": rows,
        "counts": captures_store.counts_by_status(),
        "enabled": bool(getattr(cfg, "CAPTURE_RECORDINGS_ENABLED", False)),
        "retention_days": int(getattr(cfg, "CAPTURES_RETENTION_DAYS", 0)),
        "total_count": captures_store.count(),
        "is_admin": bool(user.get("is_admin")),
        "user_id": user.get("user_id"),
    })


@router.get(
    "/captures/api/by-request/{request_id}",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_admin),
    ],
)
async def by_request_id_api(request_id: str) -> JSONResponse:
    rows = captures_store.find_by_request_id(request_id)
    usernames = api_keys_store.get_usernames([r.get("user_id") for r in rows])
    for r in rows:
        _apply_trim_to_capture_row(r)
        r["username"] = usernames.get(r.get("user_id"))
    return JSONResponse({"captures": rows})


# Literal-path GET routes (export, groups) MUST be declared BEFORE the
# parameterized /captures/api/{cid} route — FastAPI/Starlette match in
# declaration order, and the `{cid}` placeholder would otherwise swallow
# any literal-named GET like /captures/api/export with cid="export" or
# /captures/api/groups with cid="groups" (which silently 404s the
# group-list fetch and hides newly created groups from the UI).
@router.get(
    "/captures/api/export",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_admin),
    ],
)
async def export_captures_api(
    only_status: str = Query("ready"),
    include_audio: int = Query(1, ge=0, le=1),
) -> Response:
    """Streaming tar.gz of (manifest.jsonl, audio/<id>.<ext>...). The
    `only_status` filter defaults to 'ready' — admins should mark their
    triaged training samples ready before exporting. Pass 'all' to dump
    everything (typically only useful for one-off backup)."""
    status_filter: str | None = None if only_status == "all" else only_status
    fname = f"whisper-captures-{datetime.now().strftime('%Y%m%d-%H%M%S')}.tar.gz"
    return StreamingResponse(
        _build_export_stream(status_filter, bool(include_audio)),
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get(
    "/captures/api/groups",
    dependencies=[Depends(require_admin_host)],
)
async def list_groups_api(
    user_filter: str | None = Query(None, alias="user_id"),
    status_filter: str | None = Query(None, alias="status"),
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """List packed training-sample groups. Admin sees all groups (or
    narrow to one user via `?user_id=`); non-admin sees only their own.
    Optional `?status=` filter accepts the same enum as PatchGroupIn
    (new/reviewed/ready/dismissed); unknown values fall through to no
    filter, matching list_captures_api's tolerance.

    Declared above `/captures/api/{cid}` because GET with cid="groups"
    would otherwise resolve to the single-capture handler and 404 — the
    UI's `load()` then silently swallows the failure and renders no
    groups, making merged groups invisible after creation."""
    import capture_groups_store
    if not user.get("is_admin"):
        scope = user.get("user_id")
    else:
        scope = user_filter
    groups = capture_groups_store.list_groups(
        user_id=scope, status=status_filter,
    )
    usernames = api_keys_store.get_usernames([g.get("user_id") for g in groups])
    for g in groups:
        # Re-derive transcript + corrections per group so the collapsed
        # card preview reflects chip-applied final text (matches the
        # expanded card + export). Members fetched once per group; no
        # merged_words on the list path — that's expand-only.
        members = capture_groups_store.get_members(g["id"])
        _hydrate_members(members)
        g["transcript"] = _build_default_transcript(
            members, g.get("transcript_join_strategy") or "space",
        )
        g["corrections"] = _project_member_corrections(members)
        g["username"] = usernames.get(g.get("user_id"))
    return JSONResponse({"groups": groups})


@router.get(
    "/captures/api/{cid}",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_admin),
    ],
)
async def get_capture_api(cid: str) -> JSONResponse:
    row = captures_store.get_capture(cid)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    _refresh_final_if_stale(row)
    row["words"] = _align_words_to_final(
        row.get("words") or [],
        row.get("final") or "",
        model_name=row.get("model"),
    )
    # Shift word/segment timestamps onto the trimmed-audio timeline
    # when the capture has been VAD-trimmed; the karaoke band plays the
    # trimmed WAV so time math has to match. Stored words stay in
    # original-audio time in the DB — this is read-time projection
    # only.
    _apply_trim_to_capture_row(row)
    row["username"] = api_keys_store.get_username(row.get("user_id"))
    return JSONResponse({"capture": row})


# Audio sniff signatures — first few bytes -> MIME. Used by the audio
# endpoint to set Content-Type without trusting the on-row audio_format
# (filename-derived extensions lie). Browsers are picky: Safari refuses
# m4a/mp4 unless Content-Type is exactly `audio/mp4`.
_AUDIO_SNIFFS: tuple[tuple[bytes, int, str], ...] = (
    # offset 0
    (b"RIFF", 0, "audio/wav"),
    (b"OggS", 0, "audio/ogg"),
    (b"ID3",  0, "audio/mpeg"),
    (b"\xFF\xFB", 0, "audio/mpeg"),   # MP3 frame
    (b"\xFF\xF3", 0, "audio/mpeg"),
    (b"\xFF\xF2", 0, "audio/mpeg"),
    (b"fLaC", 0, "audio/flac"),
    (b"\x1A\x45\xDF\xA3", 0, "audio/webm"),  # EBML — webm/matroska
    # offset 4
    (b"ftyp", 4, "audio/mp4"),  # m4a / mp4 audio
)


def _sniff_audio_mime(abs_path: str, fallback_ext: str) -> str:
    try:
        with open(abs_path, "rb") as f:
            head = f.read(16)
    except OSError:
        head = b""
    for sig, off, mime in _AUDIO_SNIFFS:
        if head[off:off + len(sig)] == sig:
            return mime
    # Fallback to extension-based guess
    ext = (fallback_ext or "").lower().lstrip(".")
    return {
        "wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg",
        "oga": "audio/ogg", "opus": "audio/ogg", "flac": "audio/flac",
        "m4a": "audio/mp4", "mp4": "audio/mp4", "aac": "audio/aac",
        "webm": "audio/webm",
    }.get(ext, "application/octet-stream")


@router.get(
    "/captures/api/{cid}/audio",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_admin),
    ],
)
async def get_audio_api(cid: str, request: Request) -> FileResponse:
    _check_audio_rate(request.client.host if request.client else "")
    row = captures_store.get_capture(cid)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    # Prefer the trimmed WAV when one exists — that's what the export
    # uses, so reviewers should hear the same thing. Falls back to the
    # original if the trimmed file is missing on disk for any reason.
    trimmed_rel = row.get("audio_trimmed_relpath")
    abs_path: str | None = None
    if trimmed_rel:
        try:
            cand = captures_store.abs_audio_path(trimmed_rel)
            if os.path.isfile(cand):
                abs_path = cand
        except ValueError:
            abs_path = None
    if abs_path is None:
        try:
            abs_path = captures_store.abs_audio_path(row["audio_relpath"])
        except ValueError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "audio path invalid")
        if not os.path.isfile(abs_path):
            raise HTTPException(status.HTTP_410_GONE, "audio file is gone")
    mime = _sniff_audio_mime(abs_path, row.get("audio_format", ""))
    # FileResponse handles Range automatically — seeking in the karaoke
    # player won't re-download the whole file.
    return FileResponse(
        path=abs_path,
        media_type=mime,
        filename=f"{cid}.{row.get('audio_format','bin')}",
    )


@router.patch(
    "/captures/api/{cid}",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_admin),
    ],
)
async def patch_capture_api(cid: str, payload: PatchCaptureIn) -> JSONResponse:
    patch: dict[str, Any] = {}
    if payload.status is not None:
        patch["status"] = payload.status
    if payload.corrected_text is not None:
        patch["corrected_text"] = payload.corrected_text
    if payload.corrections is not None:
        edited = [c.model_dump() for c in payload.corrections]
        if payload.baseline_corrections is not None:
            # Three-way merge: apply the user's deltas to the current
            # DB state, not just replace. Protects against concurrent
            # report cascades and cross-tab admin saves.
            cap_now = captures_store.get_capture(cid) or {}
            current = cap_now.get("corrections") or []
            baseline = [c.model_dump() for c in payload.baseline_corrections]
            edited = text_corrections.three_way_merge_corrections(
                baseline, edited, current,
            )
        patch["corrections"] = edited
    if payload.admin_notes is not None:
        patch["admin_notes"] = payload.admin_notes
    try:
        updated = captures_store.update_capture(cid, patch)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    return JSONResponse({"ok": True, "capture": updated})


@router.delete(
    "/captures/api/{cid}",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_admin),
    ],
)
async def delete_capture_api(cid: str) -> JSONResponse:
    if not captures_store.delete_capture(cid):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    return JSONResponse({"ok": True})


@router.post(
    "/captures/api/clear",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_admin),
    ],
)
async def clear_captures_api(payload: ClearIn, request: Request) -> JSONResponse:
    if payload.confirm != "CAPTURES":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Confirm by sending {\"confirm\": \"CAPTURES\"}.",
        )
    host = request.client.host if request.client else ""
    n = captures_store.clear_all(reporter_host=host)
    return JSONResponse({"ok": True, "deleted": n})


# ---------------------------------------------------------------------
# Per-capture audio trim (manual, opt-in)
# ---------------------------------------------------------------------

@router.post(
    "/captures/api/{cid}/trim",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_admin),
    ],
)
async def trim_capture_audio_api(cid: str) -> JSONResponse:
    """Cut leading/trailing silence from a singleton's WAV via Silero VAD.

    Writes the trimmed audio to a NEW file (`<id>.trimmed.wav`) so the
    original `audio_relpath` is preserved. The audio GET endpoint then
    prefers the trimmed path; export emits the trimmed file as the
    sample's `audio_filepath`.

    Returns:
      {"trimmed": True} on success.
      {"trimmed": False, "reason": "..."} when nothing was trimmed
      (no speech detected / already tight / VAD unavailable).
    """
    import audio_vad_trim
    row = captures_store.get_capture(cid)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    src_rel = row.get("audio_relpath") or ""
    if not src_rel:
        raise HTTPException(status.HTTP_410_GONE, "audio file is gone")
    try:
        src_abs = captures_store.abs_audio_path(src_rel)
    except ValueError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "audio path invalid")
    if not os.path.isfile(src_abs):
        raise HTTPException(status.HTTP_410_GONE, "audio file is gone")

    # Compute the trimmed path as a sibling of the original (same
    # fanout). Suffix `.trimmed` keeps it discoverable without a
    # schema migration on the relpath helper.
    base, ext = os.path.splitext(src_rel)
    dst_rel = f"{base}.trimmed{ext or '.wav'}"
    dst_abs = captures_store.abs_audio_path(dst_rel)

    margin = int(getattr(cfg, "CAPTURES_VAD_TRIM_MARGIN_MS", 300))
    try:
        result = audio_vad_trim.trim_wav(src_abs, dst_abs, margin_ms=margin)
    except Exception as e:
        logger.warning("[trim] cid=%s VAD trim failed: %s", cid[:8], e)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"VAD trim failed: {e}",
        )
    if not (result and result.get("trimmed")):
        # Nothing to do — either no speech was detected or the file was
        # already tight. Don't persist a half-written trimmed_relpath.
        return JSONResponse({
            "trimmed": False,
            "reason": "no_speech_or_already_tight",
        })
    captures_store.update_capture(cid, {
        "audio_trimmed_relpath": dst_rel,
        "audio_trim_lead_ms": int(result.get("lead_ms") or 0),
        "audio_trim_trail_ms": int(result.get("trail_ms") or 0),
    })
    return JSONResponse({
        "trimmed": True,
        "audio_trimmed_relpath": dst_rel,
        "lead_ms": int(result.get("lead_ms") or 0),
        "trail_ms": int(result.get("trail_ms") or 0),
        "new_duration_ms": int(result.get("new_duration_ms") or 0),
    })


@router.post(
    "/captures/api/{cid}/untrim",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_admin),
    ],
)
async def untrim_capture_audio_api(cid: str) -> JSONResponse:
    """Restore a singleton's untrimmed audio: unlink the `<id>.trimmed.wav`
    companion file and clear the offset columns. The audio GET endpoint
    then serves the original `audio_relpath` again, and the karaoke band
    uses un-shifted word times (which are still in original-audio time
    in the DB — the trim was non-destructive at the data layer).

    Idempotent: returns OK even if the capture was never trimmed."""
    row = captures_store.get_capture(cid)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    trimmed_rel = row.get("audio_trimmed_relpath")
    if trimmed_rel:
        try:
            captures_store._safe_unlink(
                captures_store.abs_audio_path(trimmed_rel),
            )
        except ValueError:
            pass
    captures_store.update_capture(cid, {
        "audio_trimmed_relpath": None,
        "audio_trim_lead_ms": None,
        "audio_trim_trail_ms": None,
    })
    return JSONResponse({"untrimmed": True})


# ---------------------------------------------------------------------
# Per-capture pipeline reprocess (re-run rules on `raw`)
# ---------------------------------------------------------------------

@router.post(
    "/captures/api/{cid}/reprocess",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_admin),
    ],
)
async def reprocess_capture_api(cid: str) -> JSONResponse:
    """Re-run the post-processing pipeline on the stored `raw` text and
    update both `final` and `text_for_training` to reflect the current
    PIPELINE_RULES (and the captures-specific exclude set).

    Use case: after editing PIPELINE_RULES (e.g. adding a typo-fix or
    a new dictation-map entry), a reviewer wants this specific capture
    re-derived without waiting for the bulk reapply job. The bulk job
    /quick-config/reapply-rules also handles this row eventually, but
    the per-row trigger gives immediate feedback in the UI.
    """
    row = captures_store.get_capture(cid)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    import main
    raw = row.get("raw") or ""
    captures_excludes = getattr(cfg, "CAPTURES_PIPELINE_RULES_EXCLUDE", None)
    try:
        new_final = main._postprocess_text(raw, model_name=row.get("model"))
    except Exception as e:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"pipeline failed on `final`: {e}",
        )
    # When no captures-specific excludes are configured, the training-text
    # pass would produce byte-identical output to `final` — skip the
    # second full pipeline pass and reuse.
    if captures_excludes:
        try:
            new_training = main._postprocess_text(
                raw,
                model_name=row.get("model"),
                extra_excludes=captures_excludes,
            )
        except Exception as e:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"pipeline failed on `text_for_training`: {e}",
            )
    else:
        new_training = new_final
    patch: dict[str, Any] = {}
    if new_final != (row.get("final") or ""):
        patch["final"] = new_final
    if new_training != (row.get("text_for_training") or ""):
        patch["text_for_training"] = new_training
    updated = captures_store.update_capture(cid, patch) if patch else row
    return JSONResponse({"capture": updated or row, "changed": list(patch.keys())})


@router.post(
    "/captures/api/reprocess-all",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_admin),
    ],
)
async def reprocess_all_captures_api() -> JSONResponse:
    """Trigger the bulk pipeline-reapply job — same worker used by
    /quick-config/reapply-rules. Idempotent: a second call while the
    job is running returns the current state instead of spawning a
    duplicate worker. Use after PIPELINE_RULES edits to bring every
    capture's `final` + `text_for_training` (and downstream group
    `transcript`) in line with the current rules."""
    import captures_reapply
    return JSONResponse(captures_reapply.start())


# ---------------------------------------------------------------------
# Capture groups (≤28 s packed training samples)
# ---------------------------------------------------------------------

class CreateGroupIn(BaseModel):
    model_config = {"extra": "forbid"}
    member_ids: list[str] = Field(min_length=2, max_length=30)
    transcript: str = Field(default="", max_length=20_000)
    join_strategy: Literal["space", "period_space"] = "space"
    silence_ms: int = Field(default=300, ge=0, le=2000)


class PatchGroupIn(BaseModel):
    model_config = {"extra": "forbid"}
    join_strategy: Literal["space", "period_space"] | None = None
    silence_ms: int | None = Field(default=None, ge=0, le=2000)
    is_locked: bool | None = None
    corrections: list[CorrectionIn] | None = Field(default=None, max_length=200)
    # Snapshot of the group-derived chips at GET time. When provided
    # alongside `corrections`, the server applies a three-way merge
    # against the current member-projected chips so concurrent reports
    # / cross-tab admin saves survive. Omitted → legacy replace.
    baseline_corrections: list[CorrectionIn] | None = Field(default=None, max_length=200)
    status: Literal["new", "reviewed", "ready", "dismissed"] | None = None
    admin_notes: str | None = Field(default=None, max_length=8000)


_JOIN_STR = {"space": " ", "period_space": ". "}


def _shift_word_times(
    items: list[dict[str, Any]] | None,
    lead_ms: int,
    eff_duration_s: float | None,
) -> list[dict[str, Any]]:
    """Return a NEW list with each item's start/end shifted by -lead_ms/1000
    and clamped to [0, eff_duration_s] (when given).

    Used after a VAD trim where the served audio is shorter than the
    original: stored words/segments live in original-audio time so the
    DB stays canonical; this helper rebases them onto the trimmed
    audio's timeline so audio.currentTime alignment is correct.

    Items whose interval lies entirely outside [0, eff_duration_s] are
    dropped — they map to audio that was cut away. All other fields
    (word, raw_word, removed, member_idx, …) are preserved verbatim.
    Returns a deep-enough copy (dicts re-built so the originals are
    untouched).

    Returns an empty list when items is empty / None. Returns items
    unchanged (as a fresh list) when both lead_ms and eff_duration_s
    indicate no work (no shift, no clamp)."""
    if not items:
        return []
    shift_s = float(lead_ms or 0) / 1000.0
    if shift_s <= 0 and eff_duration_s is None:
        return list(items)
    out: list[dict[str, Any]] = []
    for it in items:
        try:
            s_old = float(it.get("start") or 0.0)
            e_old = float(it.get("end", s_old) or 0.0)
        except (TypeError, ValueError):
            continue
        s_new = s_old - shift_s
        e_new = e_old - shift_s
        if e_new <= 0:
            continue
        if eff_duration_s is not None and s_new >= eff_duration_s:
            continue
        s_clamped = max(0.0, s_new)
        e_clamped = e_new
        if eff_duration_s is not None:
            e_clamped = min(eff_duration_s, e_clamped)
        new_it = dict(it)
        new_it["start"] = s_clamped
        new_it["end"] = max(s_clamped, e_clamped)
        out.append(new_it)
    return out


def _apply_trim_to_capture_row(row: dict[str, Any]) -> None:
    """In-place: if `row` carries trim offsets, shift its `words` and
    `segments` onto the trimmed-audio timeline and add an
    `effective_duration_seconds` field. No-op when the capture was
    never trimmed (lead/trail = None or 0)."""
    if not row:
        return
    lead = row.get("audio_trim_lead_ms")
    trail = row.get("audio_trim_trail_ms")
    if not lead and not trail:
        # Still expose effective_duration_seconds equal to duration_seconds
        # so consumers can use a single field uniformly.
        row["effective_duration_seconds"] = float(row.get("duration_seconds") or 0.0)
        return
    lead_ms = int(lead or 0)
    trail_ms = int(trail or 0)
    orig_s = float(row.get("duration_seconds") or 0.0)
    eff = max(0.0, orig_s - (lead_ms + trail_ms) / 1000.0)
    row["effective_duration_seconds"] = eff
    if "words" in row:
        row["words"] = _shift_word_times(row.get("words"), lead_ms, eff)
    if "segments" in row:
        row["segments"] = _shift_word_times(row.get("segments"), lead_ms, eff)


def _apply_chips_to_text(text: str, corrections: list[dict[str, Any]]) -> str:
    """Substitute each chip's `wrong` text with its `correct` text in
    `text`. Walk in idx order so multi-word spans replace as a unit.
    If `wrong` isn't found verbatim (whitespace drift / regex specials),
    that chip is left alone. Mirrors the JS applyCorrectionsToGround
    logic so server- and client-derived transcripts agree byte-for-byte."""
    if not corrections:
        return text or ""
    out = text or ""
    def _sk(c):
        i = c.get("idx")
        try:
            return (0, int(i))
        except (TypeError, ValueError):
            return (1, 0)
    ordered = sorted(
        (c for c in corrections if isinstance(c, dict)), key=_sk,
    )
    for c in ordered:
        wrong = c.get("wrong") or ""
        correct = c.get("correct") or ""
        if not wrong or not correct:
            continue
        i = out.find(wrong)
        if i >= 0:
            out = out[:i] + correct + out[i + len(wrong):]
    return out


def _build_default_transcript(members: list[dict[str, Any]], strategy: str) -> str:
    """Concatenate member transcripts with chips applied. Each member's
    training-form text (`text_for_training`) gets its chip corrections
    layered on top before the join, so the merged result reflects what
    the export pipeline will actually produce. Falls back through `final`
    then `raw` for legacy members predating the `text_for_training`
    column."""
    parts: list[str] = []
    for m in members:
        base = m.get("text_for_training") or m.get("final") or m.get("raw") or ""
        t = _apply_chips_to_text(base, m.get("corrections") or []).strip()
        if t:
            parts.append(t)
    return _JOIN_STR.get(strategy, " ").join(parts)


def _build_merged_wav(
    *,
    gid: str,
    member_ids: list[str],
    silence_ms: int,
) -> tuple[int, dict[str, str], int, int]:
    """Resolve member audio paths, run the merge, return (duration_ms,
    member_hash_map, lead_trim_ms, trail_trim_ms). Caller must have
    validated member_ids belong to the same user and total ≤28 s.

    The trim offsets are non-zero only when CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS
    auto-trims the merged WAV — they're needed by _build_merged_words to
    keep per-member karaoke timestamps in sync with the trimmed audio."""
    import audio_merge
    import capture_groups_store

    member_paths: list[str] = []
    hashes: dict[str, str] = {}
    for mid in member_ids:
        cap = captures_store.get_capture(mid)
        if cap is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"capture {mid} not found",
            )
        abs_p = captures_store.abs_audio_path(cap["audio_relpath"])
        if not os.path.exists(abs_p):
            raise HTTPException(
                status.HTTP_410_GONE, f"capture {mid} audio is missing",
            )
        member_paths.append(abs_p)
        hashes[mid] = audio_merge.hash_wav_pcm(abs_p)

    dst_relpath = capture_groups_store._relpath_for(gid)
    dst_abs = capture_groups_store.abs_path_for(dst_relpath)
    try:
        _bytes, n_samples, lead_trim_ms, trail_trim_ms = audio_merge.merge_wavs(
            member_paths, dst_abs, gap_ms=silence_ms,
        )
    except audio_merge.WavFormatError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    duration_ms = int(round(n_samples / 16.0))  # 16 samples/ms at 16 kHz
    return duration_ms, hashes, lead_trim_ms, trail_trim_ms


@router.post(
    "/captures/api/groups",
    dependencies=[Depends(require_admin_host)],
)
async def create_group_api(
    payload: CreateGroupIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Pack 2+ same-user captures into a ≤28 s training sample.

    Server-enforced invariants:
      - all members exist, are not yet in a group, are all owned by the
        same user (and either the caller is that user OR is admin)
      - total audio + gap silence ≤ 28 s
      - members' audio files match (1 ch, 16 bit, 16 kHz)
    """
    import capture_groups_store
    import uuid as _uuid

    member_ids = payload.member_ids
    if len(member_ids) != len(set(member_ids)):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "duplicate capture in member_ids",
        )

    captures: list[dict[str, Any]] = []
    user_ids = set()
    for mid in member_ids:
        cap = captures_store.get_capture(mid)
        if cap is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"capture {mid} not found",
            )
        if cap.get("group_id"):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"capture {mid} is already in a group",
            )
        user_ids.add(cap.get("user_id") or "")
        captures.append(cap)
    if len(user_ids) != 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "members must all belong to the same user",
        )
    owner_user_id = next(iter(user_ids))
    # Authorization: admin can merge anyone's captures; non-admin can
    # only merge their own.
    if not user.get("is_admin") and owner_user_id != user.get("user_id"):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "non-admin users can only group their own captures",
        )

    # Build merged WAV — gid generated upfront so the build path is
    # known before the DB insert (mirrors captures_store).
    gid = _uuid.uuid4().hex
    # Pre-flight duration check (server-side defense; UI also enforces).
    total_audio_ms = sum(
        int(round(float(c.get("duration_seconds") or 0.0) * 1000))
        for c in captures
    )
    total_gap_ms = payload.silence_ms * max(0, len(member_ids) - 1)
    if total_audio_ms + total_gap_ms > 28_000:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"merged duration would exceed 28 s "
            f"({(total_audio_ms + total_gap_ms) / 1000:.2f}s)",
        )

    # Synthesize a transcript if the client didn't send one.
    transcript = payload.transcript.strip() or _build_default_transcript(
        captures, payload.join_strategy,
    )
    duration_ms, hashes, lead_trim_ms, trail_trim_ms = _build_merged_wav(
        gid=gid,
        member_ids=member_ids,
        silence_ms=payload.silence_ms,
    )
    # Derive language from the first member with a populated value —
    # Whisper detects language per-clip; members of the same group should
    # all share it, but if a member somehow has an empty language we
    # tolerate that and fall through to the next one rather than
    # emitting an empty `language` in the export manifest.
    group_language = ""
    for _c in captures:
        _lang = (_c.get("language") or "").strip()
        if _lang:
            group_language = _lang
            break
    try:
        _insert_group_with_gid(
            gid=gid,
            user_id=owner_user_id,
            member_ids=member_ids,
            transcript=transcript,
            join_strategy=payload.join_strategy,
            silence_ms=payload.silence_ms,
            member_hash_map=hashes,
            duration_ms=duration_ms,
            language=group_language,
            lead_trim_ms=lead_trim_ms,
            trail_trim_ms=trail_trim_ms,
        )
    except Exception:
        # Insert failed — roll back the WAV we just wrote so the
        # next merge attempt for the same captures starts clean.
        try:
            os.unlink(capture_groups_store.abs_path_for(
                capture_groups_store._relpath_for(gid)))
        except OSError:
            pass
        raise
    return JSONResponse({"group_id": gid})


def _insert_group_with_gid(
    *,
    gid: str,
    user_id: str,
    member_ids: list[str],
    transcript: str,
    join_strategy: str,
    silence_ms: int,
    member_hash_map: dict[str, str],
    duration_ms: int,
    language: str | None = None,
    lead_trim_ms: int = 0,
    trail_trim_ms: int = 0,
) -> None:
    """Direct insert that honours a pre-allocated gid (needed because the
    audio file is written at the gid path before this call).

    Group chip state lives on the member captures, not on the group
    row — every read re-projects from members — so no chip plumbing
    appears here."""
    import capture_groups_store

    relpath = capture_groups_store._relpath_for(gid)
    now = time.time()
    conn = capture_groups_store._require_conn()
    with capture_groups_store._lock:
        with conn:
            conn.execute(
                "INSERT INTO capture_groups"
                " (id, user_id, created_ts, merged_wav_relpath,"
                "  merged_duration_ms, transcript,"
                "  transcript_join_strategy, member_hashes_json,"
                "  inter_segment_silence_ms, is_stale, is_locked,"
                "  language, merged_lead_trim_ms, merged_trail_trim_ms)"
                " VALUES (?,?,?,?,?,?,?,?,?,0,0,?,?,?)",
                (
                    gid, user_id, now, relpath, int(duration_ms),
                    transcript, join_strategy,
                    json.dumps(member_hash_map, sort_keys=True),
                    int(silence_ms),
                    language or None,
                    int(lead_trim_ms or 0),
                    int(trail_trim_ms or 0),
                ),
            )
            for order, mid in enumerate(member_ids):
                conn.execute(
                    "UPDATE captures SET group_id = ?, group_order = ?"
                    " WHERE id = ? AND group_id IS NULL",
                    (gid, order, mid),
                )
    logger.info(
        "[groups] created gid=%s user=%s n=%d dur=%.1fs",
        gid[:8], (user_id or "?")[:8], len(member_ids), duration_ms / 1000.0,
    )


@router.get(
    "/captures/api/groups/{gid}",
    dependencies=[Depends(require_admin_host)],
)
async def get_group_api(
    gid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    import capture_groups_store
    g = capture_groups_store.get_group(gid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "group not found")
    if not user.get("is_admin") and g["user_id"] != user.get("user_id"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not your group")
    return JSONResponse({"group": _enrich_group(g)})


def _hydrate_members(members: list[dict[str, Any]]) -> None:
    """Populate `words` (decoded) and `model` on each member dict in
    place by fetching the full capture row once. `capture_groups_store.
    get_members` drops words_json/model to keep the projection light;
    the chip/karaoke helpers below need both. Idempotent — skips members
    that already carry the fields."""
    for m in members:
        if "words" in m and "model" in m:
            continue
        cap = captures_store.get_capture(m["id"]) or {}
        m["words"] = cap.get("words") or []
        m["model"] = cap.get("model")


def _project_member_corrections(
    members: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project each member's chip corrections into a group-level chip
    list with global word indices. Member chips reference indices into
    that member's words_json (immutable raw STT). Group chips need
    indices into the flattened merged_words array. The offset for
    member m is Σ_{j<m} len(words_j) — silence gaps contribute no
    words, so they don't shift the index.

    Callers must run `_hydrate_members(members)` first so each member
    carries `words`."""
    out: list[dict[str, Any]] = []
    offset = 0
    for m in members:
        words = m.get("words") or []
        for c in (m.get("corrections") or []):
            try:
                idx = int(c["idx"]) + offset
            except (TypeError, ValueError, KeyError):
                continue
            c2 = dict(c)
            c2["idx"] = min(idx, offset + max(0, len(words) - 1))
            if c.get("idx_end") is not None:
                try:
                    end = int(c["idx_end"]) + offset
                    c2["idx_end"] = min(end, offset + max(0, len(words) - 1))
                except (TypeError, ValueError):
                    c2.pop("idx_end", None)
            out.append(c2)
        offset += len(words)
    return out


def _split_corrections_to_members(
    corrections: list[dict[str, Any]],
    members: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Inverse of `_project_member_corrections`. Given a list of
    group-level chips with global word indices, return a dict mapping
    each member_id → list of chips with member-local indices.

    Every member is represented in the result (with `[]` if it owns no
    chips after the split) so the caller can fan out via
    `update_capture(member_id, {"corrections": chips})` and reliably
    REPLACE each member's chip list. Chips whose `idx` is out of range
    are silently dropped; `idx_end` is clipped to the same member's
    last word."""
    word_counts: list[int] = []
    for m in members:
        word_counts.append(len(m.get("words") or []))
    offsets = [0]
    for n in word_counts[:-1]:
        offsets.append(offsets[-1] + n)
    out: dict[str, list[dict[str, Any]]] = {m["id"]: [] for m in members}
    for c in (corrections or []):
        if not isinstance(c, dict):
            continue
        try:
            gidx = int(c["idx"])
        except (TypeError, ValueError, KeyError):
            continue
        target: int | None = None
        for i in range(len(members)):
            start = offsets[i]
            end = start + word_counts[i]
            if start <= gidx < end:
                target = i
                break
        if target is None:
            continue
        c2 = dict(c)
        c2["idx"] = gidx - offsets[target]
        if c.get("idx_end") is not None:
            try:
                end_local = int(c["idx_end"]) - offsets[target]
                max_local = word_counts[target] - 1
                c2["idx_end"] = min(max(end_local, c2["idx"]), max_local)
            except (TypeError, ValueError):
                c2.pop("idx_end", None)
        out[members[target]["id"]].append(c2)
    return out


def _enrich_group(g: dict[str, Any]) -> dict[str, Any]:
    """Add `members` + `merged_words` to a group dict and re-derive the
    chip-dependent fields (transcript + corrections) from current
    member state.

    Source of truth for chips is each MEMBER's `corrections` list. The
    group's own `corrections_json` column is a leftover cache from the
    earlier "one-time projection" design; it's never read here. With
    every read going through this function, report cascades and direct
    member-chip edits on /captures flow through to the group's
    Corrections section automatically — no separate migration pass or
    in-DB chip storage needed at the group level."""
    import capture_groups_store
    members = capture_groups_store.get_members(g["id"])
    _hydrate_members(members)
    usernames = api_keys_store.get_usernames(
        [m.get("user_id") for m in members] + [g.get("user_id")]
    )
    for m in members:
        _refresh_final_if_stale(m)
        m["username"] = usernames.get(m.get("user_id"))
    g["members"] = members
    g["username"] = usernames.get(g.get("user_id"))
    g["transcript"] = _build_default_transcript(
        members, g.get("transcript_join_strategy") or "space",
    )
    g["corrections"] = _project_member_corrections(members)
    g["merged_words"] = _build_merged_words(
        members, int(g["inter_segment_silence_ms"]),
        merged_lead_trim_ms=int(g.get("merged_lead_trim_ms") or 0),
        merged_duration_ms=int(g.get("merged_duration_ms") or 0),
    )
    # Effective duration mirrors the singleton field: equals
    # merged_duration_ms / 1000 since merged_duration_ms is already the
    # post-trim value. Exposed for a uniform display shape with
    # singletons.
    g["effective_duration_seconds"] = float(g.get("merged_duration_ms") or 0) / 1000.0
    return g


def _refresh_final_if_stale(row: dict[str, Any]) -> None:
    """Recompute `final` AND `text_for_training` from `raw` via the
    current pipeline. If either differs from what's stored, write it
    back and update the row in place.

    This is the per-row self-heal that keeps fetched captures
    rule-current without requiring the user to click "Re-apply rules"
    first. The bulk reapply job is still useful for unfetched captures
    (export, retention sweep).

    `text_for_training` is the canonical /captures display text — it
    must reflect current PIPELINE_RULES (minus the captures-specific
    excludes) so reviewers see what the export will emit and chips
    apply against the same text the trainer will consume."""
    raw = row.get("raw") or ""
    stored_training = row.get("text_for_training") or ""
    if not raw:
        return
    stored_final = row.get("final") or ""
    try:
        import main
        fresh_final = main._postprocess_text(raw, model_name=row.get("model"))
    except Exception:
        return
    patch: dict[str, Any] = {}
    if fresh_final != stored_final:
        patch["final"] = fresh_final
        row["final"] = fresh_final
    # Self-heal text_for_training even when `final` is up-to-date — the
    # captures-excludes set could have changed (admin tweaked
    # CAPTURES_PIPELINE_RULES_EXCLUDE) without an underlying PIPELINE_RULES
    # change, and stale training text would mislead reviewers and
    # the export.
    captures_excludes = getattr(cfg, "CAPTURES_PIPELINE_RULES_EXCLUDE", None)
    if captures_excludes:
        try:
            fresh_training = main._postprocess_text(
                raw,
                model_name=row.get("model"),
                extra_excludes=captures_excludes,
            )
        except Exception:
            fresh_training = None
    else:
        # No captures-specific excludes — training text is identical to
        # final, skip the second pipeline pass.
        fresh_training = fresh_final
    if fresh_training is not None and fresh_training != stored_training:
        patch["text_for_training"] = fresh_training
        row["text_for_training"] = fresh_training
    if patch:
        try:
            captures_store.update_capture(row["id"], patch)
        except Exception as e:
            logger.warning(
                "[captures] self-heal write-back failed for %s: %s",
                str(row.get("id"))[:8], e,
            )


def _align_key(s: str) -> str:
    """Normalise a token for LCS comparison: strip surrounding
    whitespace + casefold. Internal punctuation is preserved so
    'Hello,' and 'Hello' don't cross-match — the rule rewrote the
    word with the comma for a reason and we want it surfaced as a
    `raw_word` diff."""
    return (s or "").strip().casefold()


def _align_words_to_final(
    words: list[dict[str, Any]],
    final: str,
    model_name: "str | None" = None,
) -> list[dict[str, Any]]:
    """Project raw STT words onto post-pipeline `final` via LCS
    alignment. Replaces the all-or-nothing fallback that came before
    — most rule output IS faithfully attributable per word even when
    a cross-word rule (dedup, "Neue Zeile → \\n", etc.) also fires.

    Output items (one per raw word) carry:
      - `word`: the matched final token (display text for the
        karaoke band + the chip's `wrong` reference)
      - `raw_word`: present when display != raw — powers the dotted
        underline + `title="raw: …"` tooltip
      - `removed`: True when LCS found no match for this raw token,
        i.e. a cross-word rule deleted it from `final`. The UI fades
        + strikes-through these slots; chip creation is suppressed.

    Insertions (final tokens with no raw correspondent) are NOT
    materialised — there's no audio timestamp to anchor them to.
    They're still in `final` for export; the band's job is to
    faithfully represent timestamped audio, not to invent positions.
    """
    src = list(words or [])
    if not src:
        return []
    try:
        import main  # for _postprocess_text
    except Exception:
        return [_clone_word(w) for w in src]

    # Memo: many captures repeat the same raw token (filler words, punctuation
    # carriers). Without the cache, _postprocess_text runs O(N) times per
    # caller and dominates karaoke-band assembly when /captures expands a
    # group with hundreds of words.
    post_cache: dict[str, str] = {}
    raw_keys: list[str] = []
    for w in src:
        raw_w = w.get("word") or ""
        post_w = post_cache.get(raw_w)
        if post_w is None:
            try:
                post_w = main._postprocess_text(raw_w, model_name=model_name)
            except Exception:
                post_w = raw_w
            post_cache[raw_w] = post_w
        raw_keys.append(_align_key(post_w) or _align_key(raw_w))

    fin_tokens = (final or "").split()
    fin_keys = [_align_key(t) for t in fin_tokens]

    n, m = len(raw_keys), len(fin_keys)
    matches: list[int] = [-1] * n
    if n and m:
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                if raw_keys[i] and raw_keys[i] == fin_keys[j]:
                    dp[i][j] = dp[i + 1][j + 1] + 1
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
        i = j = 0
        while i < n and j < m:
            if raw_keys[i] and raw_keys[i] == fin_keys[j]:
                matches[i] = j
                i += 1
                j += 1
            elif dp[i + 1][j] >= dp[i][j + 1]:
                i += 1
            else:
                j += 1

    out: list[dict[str, Any]] = []
    for i, w in enumerate(src):
        item = _clone_word(w)
        raw_w = w.get("word") or ""
        if matches[i] >= 0:
            item["word"] = fin_tokens[matches[i]]
            if item["word"].strip() != raw_w.strip():
                item["raw_word"] = raw_w
        else:
            item["word"] = raw_w
            item["raw_word"] = raw_w
            item["removed"] = True
        out.append(item)
    return out


def _clone_word(w: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of a words_json item with only the keys
    we care about preserved (word / start / end). Other fields
    (probability, etc.) are dropped — the UI doesn't use them and
    they bloat the JSON payload."""
    return {
        "word":  w.get("word") or "",
        "start": w.get("start"),
        "end":   w.get("end", w.get("start")),
    }


def _build_merged_words(
    members: list[dict[str, Any]],
    silence_ms: int,
    *,
    merged_lead_trim_ms: int = 0,
    merged_duration_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Project each member's per-word timestamps onto the merged-audio
    timeline. Member i starts at (Σ_{j<i} dur_j) + i × silence_s seconds —
    i.e. cumulative member duration plus one silence gap PER preceding
    member. start/end are returned in seconds (matches audio.currentTime
    and the single-capture karaoke band's expectation).

    When the merged WAV was VAD-trimmed at merge time, `merged_lead_trim_ms`
    shifts produced times so the karaoke aligns with the trimmed audio
    (trail trimming is captured via `merged_duration_ms` clamping):
      - Subtract `merged_lead_trim_ms / 1000` from every word's start/end.
      - Drop words whose interval falls entirely outside the trimmed
        clip's [0, effective_duration_s] range; clamp partial overlaps.
    Un-trimmed groups (lead = 0, duration = full) take the no-op path.

    `get_members` strips heavy fields for the list view, so we re-fetch
    each capture to get `words`. Each member's words are run through
    `_per_word_postprocess` so the displayed text matches what's in
    `final` / what gets exported. Cost is bounded — ≤30 members,
    ≤a few hundred words total per ≤28 s group — and only runs on expand."""
    silence_s = max(0, int(silence_ms)) / 1000.0
    lead_s = max(0, int(merged_lead_trim_ms or 0)) / 1000.0
    eff_dur_s: float | None = None
    if merged_duration_ms is not None:
        # merged_duration_ms reflects the TRIMMED duration (we re-measure
        # post-trim in audio_merge.merge_wavs), so use it directly.
        eff_dur_s = max(0.0, float(merged_duration_ms) / 1000.0)
    merged: list[dict[str, Any]] = []
    cum = 0.0
    for i, m in enumerate(members):
        offset = cum + i * silence_s - lead_s
        ws = _align_words_to_final(
            m.get("words") or [],
            m.get("final") or "",
            model_name=m.get("model"),
        )
        for w in ws:
            start = w.get("start")
            end = w.get("end", start)
            word = w.get("word", "")
            if start is None or end is None:
                continue
            s_new = float(start) + offset
            e_new = float(end) + offset
            if e_new <= 0:
                continue
            if eff_dur_s is not None and s_new >= eff_dur_s:
                continue
            s_clamped = max(0.0, s_new)
            e_clamped = e_new
            if eff_dur_s is not None:
                e_clamped = min(eff_dur_s, e_clamped)
            entry = {
                "word":       word,
                "start":      s_clamped,
                "end":        max(s_clamped, e_clamped),
                "member_idx": i,
            }
            if w.get("raw_word"):
                entry["raw_word"] = w["raw_word"]
            if w.get("removed"):
                entry["removed"] = True
            merged.append(entry)
        cum += float(m.get("duration_seconds") or 0.0)
    return merged


@router.patch(
    "/captures/api/groups/{gid}",
    dependencies=[Depends(require_admin_host)],
)
async def patch_group_api(
    gid: str,
    payload: PatchGroupIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    import capture_groups_store
    g = capture_groups_store.get_group(gid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "group not found")
    if not user.get("is_admin") and g["user_id"] != user.get("user_id"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not your group")
    if g["is_locked"] and not user.get("is_admin"):
        raise HTTPException(status.HTTP_409_CONFLICT, "group is locked")

    patch: dict[str, Any] = {}
    rebuild_audio = False
    # Lazily-fetched hydrated members; up to three branches below need
    # this list and used to issue independent get_members calls each.
    _members_cache: list[dict[str, Any]] | None = None
    def _members() -> list[dict[str, Any]]:
        nonlocal _members_cache
        if _members_cache is None:
            _members_cache = capture_groups_store.get_members(gid)
            _hydrate_members(_members_cache)
        return _members_cache

    if payload.join_strategy is not None and \
            payload.join_strategy != g["transcript_join_strategy"]:
        patch["transcript_join_strategy"] = payload.join_strategy
    if payload.silence_ms is not None and \
            payload.silence_ms != g["inter_segment_silence_ms"]:
        patch["inter_segment_silence_ms"] = payload.silence_ms
        rebuild_audio = True
    if payload.is_locked is not None:
        patch["is_locked"] = 1 if payload.is_locked else 0
    if payload.status is not None:
        patch["status"] = payload.status
    if payload.admin_notes is not None:
        patch["admin_notes"] = payload.admin_notes
    if payload.corrections is not None:
        # Fan group-level chip edits DOWN to the owning members. Group
        # corrections are derived from members on every read (see
        # `_enrich_group`); writing to a group-level chip column would
        # be discarded by the next GET.
        #
        # When the client also sends `baseline_corrections` (a snapshot
        # of what it loaded), apply a three-way merge against the
        # current member-projected chips BEFORE the split — that way a
        # concurrent report cascade or cross-tab admin save isn't
        # clobbered by the user's payload, and the user's deltas
        # (additions, removals, edits) are applied on top.
        members_now = _members()
        edited = [c.model_dump() for c in payload.corrections]
        if payload.baseline_corrections is not None:
            baseline = [c.model_dump() for c in payload.baseline_corrections]
            current = _project_member_corrections(members_now)
            edited = text_corrections.three_way_merge_corrections(
                baseline, edited, current,
            )
        by_member = _split_corrections_to_members(edited, members_now)
        # Skip members whose chip set didn't change — a 30-member group
        # with one edited chip otherwise fires 30 UPDATEs where 29 are
        # idempotent rewrites of the same JSON column.
        current_by_id = {m["id"]: (m.get("corrections") or []) for m in members_now}
        for member_id, chips in by_member.items():
            if json.dumps(current_by_id.get(member_id) or [], sort_keys=True) == \
                    json.dumps(chips, sort_keys=True):
                continue
            captures_store.update_capture(member_id, {"corrections": chips})

    if rebuild_audio:
        # Re-run the merge with the new silence. Member set is unchanged.
        members = _members()
        duration_ms, hashes, lead_trim_ms, trail_trim_ms = _build_merged_wav(
            gid=gid,
            member_ids=[m["id"] for m in members],
            silence_ms=int(patch["inter_segment_silence_ms"]),
        )
        patch["member_hashes_json"] = json.dumps(hashes, sort_keys=True)
        patch["merged_duration_ms"] = duration_ms
        patch["merged_lead_trim_ms"] = int(lead_trim_ms or 0)
        patch["merged_trail_trim_ms"] = int(trail_trim_ms or 0)
        patch["is_stale"] = 0

    # Re-derive `transcript` from current members + chips ONLY when the
    # inputs that feed the derivation actually changed (corrections,
    # join_strategy) or when the audio was rebuilt (silence change). The
    # common status/admin_notes/is_locked auto-save click would otherwise
    # trigger a get_members + transcript rebuild + DB write on every click.
    transcript_inputs_changed = (
        payload.corrections is not None
        or rebuild_audio
        or "transcript_join_strategy" in patch
    )
    if transcript_inputs_changed:
        join_for_derive = patch.get(
            "transcript_join_strategy", g["transcript_join_strategy"] or "space",
        )
        patch["transcript"] = _build_default_transcript(_members(), join_for_derive)

    updated = capture_groups_store.update_group(gid, patch)
    return JSONResponse({"group": _enrich_group(updated)})


@router.post(
    "/captures/api/groups/{gid}/regenerate",
    dependencies=[Depends(require_admin_host)],
)
async def regenerate_group_api(
    gid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Rebuild the merged WAV from current member content, refresh
    hashes, clear `is_stale`. Transcript is preserved (admin's edits stay)."""
    import capture_groups_store
    g = capture_groups_store.get_group(gid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "group not found")
    if not user.get("is_admin") and g["user_id"] != user.get("user_id"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not your group")
    members = capture_groups_store.get_members(gid)
    duration_ms, hashes, lead_trim_ms, trail_trim_ms = _build_merged_wav(
        gid=gid,
        member_ids=[m["id"] for m in members],
        silence_ms=g["inter_segment_silence_ms"],
    )
    updated = capture_groups_store.update_group(gid, {
        "is_stale": 0,
        "merged_duration_ms": duration_ms,
        "member_hashes_json": json.dumps(hashes, sort_keys=True),
        "merged_lead_trim_ms": int(lead_trim_ms or 0),
        "merged_trail_trim_ms": int(trail_trim_ms or 0),
    })
    return JSONResponse({"group": _enrich_group(updated)})


@router.delete(
    "/captures/api/groups/{gid}",
    dependencies=[Depends(require_admin_host)],
)
async def dissolve_group_api(
    gid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    import capture_groups_store
    g = capture_groups_store.get_group(gid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "group not found")
    if not user.get("is_admin") and g["user_id"] != user.get("user_id"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not your group")
    if g["is_locked"] and not user.get("is_admin"):
        raise HTTPException(status.HTTP_409_CONFLICT, "group is locked")
    capture_groups_store.dissolve_group(gid)
    return JSONResponse({"ok": True})


# Per-gid lock so a burst of audio requests for the same missing WAV
# at startup doesn't trigger N concurrent merges. The route runs the
# sync merge inside Starlette's threadpool, so a plain threading.Lock
# is the right primitive (asyncio.Lock would only help if the merge
# itself awaited).
_rebuild_locks: dict[str, threading.Lock] = {}
_rebuild_locks_guard = threading.Lock()


def _get_rebuild_lock(gid: str) -> threading.Lock:
    with _rebuild_locks_guard:
        lock = _rebuild_locks.get(gid)
        if lock is None:
            lock = threading.Lock()
            _rebuild_locks[gid] = lock
        return lock


def _ensure_group_wav(g: dict[str, Any]) -> str:
    """Resolve the merged-WAV abs path. If the file is missing on
    disk but every member capture still has its row + audio, rebuild
    the WAV in place and return its abs path. If unrecoverable, raise
    HTTPException(410).

    Merged WAVs are deterministic functions of (members, silence_ms,
    join_strategy) — treating them as cached derived data instead of
    a precious one-shot artifact means no deletion path (clear_all,
    legacy reconcile, crash mid-write, manual cleanup, etc.) surfaces
    as a hard 404 to the user. The "Regenerate" button still exists
    for the legitimate force-rebuild case (user edited silence/join).
    """
    import capture_groups_store
    try:
        abs_p = capture_groups_store.abs_path_for(g["merged_wav_relpath"])
    except ValueError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "merged audio missing")
    if os.path.exists(abs_p):
        return abs_p

    members = capture_groups_store.get_members(g["id"])
    if not members:
        raise HTTPException(
            status.HTTP_410_GONE,
            "members deleted — group is unrecoverable",
        )
    for m in members:
        cap = captures_store.get_capture(m["id"])
        if cap is None:
            raise HTTPException(
                status.HTTP_410_GONE,
                f"member {m['id'][:8]} row deleted — group is unrecoverable",
            )
        member_abs = captures_store.abs_audio_path(cap["audio_relpath"])
        if not os.path.exists(member_abs):
            raise HTTPException(
                status.HTTP_410_GONE,
                f"member {m['id'][:8]} audio is gone — group is unrecoverable",
            )

    member_ids = [m["id"] for m in members]
    lock = _get_rebuild_lock(g["id"])
    with lock:
        if os.path.exists(abs_p):
            return abs_p
        logger.warning(
            "[groups] gid=%s auto-rebuilding missing WAV from %d members",
            g["id"][:8], len(member_ids),
        )
        duration_ms, hashes, lead_trim_ms, trail_trim_ms = _build_merged_wav(
            gid=g["id"],
            member_ids=member_ids,
            silence_ms=int(g["inter_segment_silence_ms"]),
        )
        capture_groups_store.update_group(g["id"], {
            "merged_duration_ms": int(duration_ms),
            "member_hashes_json": json.dumps(hashes, sort_keys=True),
            "is_stale":           0,
            "merged_lead_trim_ms":  int(lead_trim_ms or 0),
            "merged_trail_trim_ms": int(trail_trim_ms or 0),
        })
    return abs_p


@router.get(
    "/captures/api/groups/{gid}/audio",
    dependencies=[Depends(require_admin_host)],
)
async def get_group_audio_api(
    gid: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
):
    """Stream the merged WAV, self-healing if it's missing on disk
    but reconstructable from member captures."""
    _check_audio_rate(request.client.host if request.client else "")
    import capture_groups_store
    g = capture_groups_store.get_group(gid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "group not found")
    if not user.get("is_admin") and g["user_id"] != user.get("user_id"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not your group")
    abs_p = _ensure_group_wav(g)
    return FileResponse(
        abs_p,
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
        },
    )


# ---------------------------------------------------------------------
# Export: streamed tar.gz
# ---------------------------------------------------------------------

_EXPORT_MANIFEST_KEYS = (
    "audio_filepath",
    "text",
    "duration",
    "language",
    "source",
    "user_id",
    "status",
    "created_ts",
    "model",
    "request_id",
    "member_count",
    "admin_notes",
    "corrections",
)


def _build_manifest_row(
    *,
    audio_filepath: str,
    text: str,
    duration: float,
    language: str,
    source: str,
    user_id: str,
    status_value: str,
    created_ts: float,
    model: str,
    request_id: str,
    member_count: int,
    admin_notes: str,
    corrections: list,
) -> dict[str, Any]:
    """Build a single manifest line dict with the unified 13-key schema.

    Same keys for singletons and groups — defaults populate the keys
    that don't apply (e.g. groups → request_id=""; singletons →
    member_count=1). Heterogeneous-field-set was a documented pain
    point for cross-corpus filtering during fine-tuning."""
    return {
        "audio_filepath": audio_filepath,
        "text": text,
        "duration": float(duration or 0.0),
        "language": language or "",
        "source": source,
        "user_id": user_id or "",
        "status": status_value or "",
        "created_ts": float(created_ts or 0.0),
        "model": model or "",
        "request_id": request_id or "",
        "member_count": int(member_count or 0),
        "admin_notes": admin_notes or "",
        "corrections": list(corrections or []),
    }


def _build_export_stream(only_status: str | None, include_audio: bool):
    """Generator that yields tar.gz bytes containing manifest.jsonl and,
    optionally, audio/<id>.wav entries. One manifest entry per training
    unit — a "unit" is either a capture group (≤28 s packed sample) OR
    an ungrouped singleton capture. Group members never appear as
    singletons (no double-counting; the group transcript covers them).

    Hard filters applied regardless of `only_status`:
      - `is_stale=true` (group audio/text drift) → skipped
      - `is_locked=true` (admin-flagged anomaly) → skipped
      - `status=audio_missing` (file is gone) → skipped
      - missing WAV on disk → both manifest entry AND tar entry skipped
        (prevents the manifest from referencing files that don't exist
        in the tarball)
    `only_status` (default "ready" from the route) further narrows.
    `audio_missing` rows leak only when `only_status='all'`, and even
    then the hard filter above drops them.
    """
    import capture_groups_store

    buf = io.BytesIO()
    tar = tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6)
    manifest_lines: list[bytes] = []

    # 1. Capture groups (the packed-for-fine-tune training samples).
    user_filter_scope: str | None = None  # admin-only path; no per-user scope
    for g in capture_groups_store.list_groups(user_id=user_filter_scope):
        # Status gate — groups have the same status field as captures.
        # The caller has already mapped "all" → None upstream, so a
        # truthy only_status here is a concrete status to match.
        if only_status:
            if (g.get("status") or "new") != only_status:
                continue
        # Hard filters (apply even on `only_status=all`). These rows are
        # never valid training data: stale = audio/text drift, locked =
        # admin-flagged anomaly.
        if g.get("is_stale") or g.get("is_locked"):
            continue
        # Always rebuild the transcript at export time from members +
        # chips, so the exported text reflects current corrections even
        # if the stored snapshot is stale. Source from the training-form
        # column so reviewers see — and the trainer learns from — the
        # same text.
        gid = g["id"]
        members = capture_groups_store.get_members(gid)
        text = _build_default_transcript(
            members, g.get("transcript_join_strategy") or "space",
        ).strip()
        if not text:
            continue
        # Audio existence gate: skip the manifest entry entirely if the
        # WAV isn't on disk, to avoid manifest pointing at missing files.
        try:
            abs_p = capture_groups_store.abs_path_for(g["merged_wav_relpath"])
        except ValueError:
            continue
        if not os.path.isfile(abs_p):
            continue

        audio_name = f"audio/{gid}.wav"
        # Group `model` and `request_id` are intentionally empty — a
        # group has multiple members each with their own model id. Per-
        # member audit is reachable via the group's GET /members endpoint.
        manifest_lines.append(json.dumps(_build_manifest_row(
            audio_filepath=audio_name,
            text=text,
            duration=float(g.get("merged_duration_ms") or 0) / 1000.0,
            language=g.get("language") or "",
            source="group",
            user_id=g.get("user_id") or "",
            status_value=g.get("status") or "new",
            created_ts=float(g.get("created_ts") or 0.0),
            model="",
            request_id="",
            member_count=len(members),
            admin_notes=g.get("admin_notes") or "",
            corrections=[],
        ), ensure_ascii=False).encode("utf-8") + b"\n")

        if include_audio:
            info = tarfile.TarInfo(audio_name)
            info.size = os.path.getsize(abs_p)
            info.mtime = int(g.get("created_ts") or time.time())
            with open(abs_p, "rb") as af:
                tar.addfile(info, af)
            chunk = buf.getvalue()
            buf.seek(0); buf.truncate()
            if chunk:
                yield chunk

    # 2. Ungrouped captures (no group_id).
    for row in captures_store.iter_captures_for_export(status=only_status):
        if row.get("group_id"):
            continue
        # Hard filter: `audio_missing` rows have no WAV; never valid
        # training data. (Caught here even when only_status='all'.)
        if (row.get("status") or "") == "audio_missing":
            continue
        cid = row["id"]
        # Source training-form text first so the export matches what
        # reviewers see on /captures. Chip-applied on top. `final` and
        # `raw` fall-backs cover captures from before the
        # text_for_training column existed.
        base = (row.get("text_for_training")
                or row.get("final")
                or row.get("raw") or "")
        text = _apply_chips_to_text(base, row.get("corrections") or [])
        if not text.strip():
            continue
        # Audio path: prefer the trimmed companion if one was produced.
        # Either way, the manifest line is skipped if the file isn't on
        # disk (defense against the audio_missing leak path).
        rel = row.get("audio_trimmed_relpath") or row.get("audio_relpath")
        if not rel:
            continue
        try:
            abs_p = captures_store.abs_audio_path(rel)
        except ValueError:
            continue
        if not os.path.isfile(abs_p):
            continue
        ext = os.path.splitext(rel)[1].lstrip(".").lower() or "wav"
        audio_name = f"audio/{cid}.{ext}"
        manifest_lines.append(json.dumps(_build_manifest_row(
            audio_filepath=audio_name,
            text=text,
            duration=float(row.get("duration_seconds") or 0.0),
            language=row.get("language") or "",
            source="singleton",
            user_id=row.get("user_id") or "",
            status_value=row.get("status") or "",
            created_ts=float(row.get("created_ts") or 0.0),
            model=row.get("model") or "",
            request_id=row.get("request_id") or "",
            member_count=1,
            admin_notes=row.get("admin_notes") or "",
            corrections=row.get("corrections") or [],
        ), ensure_ascii=False).encode("utf-8") + b"\n")

        if include_audio:
            info = tarfile.TarInfo(audio_name)
            info.size = os.path.getsize(abs_p)
            info.mtime = int(row.get("created_ts") or time.time())
            with open(abs_p, "rb") as af:
                tar.addfile(info, af)
            chunk = buf.getvalue()
            buf.seek(0); buf.truncate()
            if chunk:
                yield chunk

    # Manifest last so it's written in row order matching the audio.
    manifest_blob = b"".join(manifest_lines)
    info = tarfile.TarInfo("manifest.jsonl")
    info.size = len(manifest_blob)
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(manifest_blob))

    tar.close()
    final_chunk = buf.getvalue()
    if final_chunk:
        yield final_chunk


# ---------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------
# Single-page admin: list view + per-row expand. The expanded row plays
# the audio karaoke-style (active word highlights with playback), lets
# the admin shift-click to mark wrong words, and edits the ground-truth
# `corrected_text` (free-form textarea). Status flow: new → reviewed →
# ready (export-eligible) | dismissed (omitted).
#
# IMPORTANT (CLAUDE memory note): never place a `{{...}}` placeholder
# inside a /* */, //, or <!-- --> comment — render_page() does a literal
# string replace and the substitution corrupts the surrounding context.

_CAPTURES_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Whisper — Captures</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
{{SCALE_BOOTSTRAP_HEAD}}
<style>
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --fg: #c9d1d9;
    --dim: #6e7681;
    --help: #8b949e;
    --bold: #f0f6fc;
    --border: #30363d;
    --input-bg: #0d1117;
    --cyan: #79c0ff;
    --green: #7ee787;
    --yellow: #f2cc60;
    --red: #ff7b72;
    --magenta: #d2a8ff;
    --active-word-bg: rgba(121, 192, 255, 0.22);
    --active-word-color: #f0f6fc;
    --selected-word-bg: rgba(255, 123, 114, 0.22);
  }
  *, *::before, *::after { box-sizing: border-box; }
  html, body { background: var(--bg); color: var(--fg);
    font-family: var(--font-sans); font-size: var(--fs-lg);
    margin: 0; padding: 0; }
  main { max-width: 75rem; margin: 0 auto; padding: 1rem 1.25rem 4rem; }
  h2 { font-size: var(--fs-xl); margin: 0 0 0.5rem; color: var(--bold); }
  .toolbar {
    display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
    margin: 0.5rem 0 0.875rem; padding: 0.5rem 0.625rem;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px;
  }
  .toolbar label { font-size: var(--fs-sm); color: var(--help); }
  /* Used in place of <label> when the wrapped control is a button
     group; avoids HTML's auto-click-through to first labelable. */
  .toolbar .filt-label, .cc-actions .cc-status-label {
    font-size: var(--fs-sm); color: var(--help);
    display: inline-flex; align-items: center; gap: 0.35rem;
  }
  .toolbar select, .toolbar input[type="text"] {
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.25rem 0.4rem; font-size: var(--fs-md);
    font-family: var(--font-sans);
  }
  .toolbar input[type="text"] { min-width: 14rem; }
  .toolbar .spacer { flex: 1; }
  .toolbar .counts { color: var(--help); font-size: var(--fs-sm);
    margin-right: 0.5rem; }
  .toolbar .counts .n { color: var(--bold); font-weight: 600; }
  .toolbar .capture-state {
    font-size: var(--fs-sm); padding: 0.15rem 0.5rem; border-radius: 4px;
    border: 1px solid var(--border);
  }
  .toolbar .capture-state.on  { color: var(--green); border-color: #2d5a37; }
  .toolbar .capture-state.off { color: var(--dim);   border-color: var(--border); }

  /* Radio-style status button group. Used in the toolbar filter and
     in the per-row + per-group action rows. Replaces the previous
     <select> dropdowns; a 1-click switch is faster than open-pick. */
  .status-btn-group {
    display: inline-flex; gap: 0;
    border: 1px solid var(--border); border-radius: 0.375rem;
    overflow: hidden; font-family: var(--font-sans);
    vertical-align: middle;
  }
  .status-btn {
    background: var(--input-bg); color: var(--fg);
    border: 0; border-right: 1px solid var(--border);
    padding: 0.25rem 0.625rem; font-size: var(--fs-sm);
    font-family: var(--font-sans); cursor: pointer; line-height: 1.4;
    border-radius: 0;
    transition: background-color 0.1s ease, color 0.1s ease;
  }
  .status-btn:last-child { border-right: 0; }
  .status-btn:hover:not(.active):not(:disabled) {
    background: #21262d; color: var(--bold);
  }
  .status-btn.active {
    background: var(--cyan); color: #0d1117; font-weight: 600;
  }
  .status-btn:focus-visible {
    outline: 2px solid var(--cyan); outline-offset: -2px;
  }
  .status-btn:disabled { color: var(--dim); cursor: not-allowed; }

  button {
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.25rem 0.625rem; font-size: var(--fs-md);
    font-family: var(--font-sans); cursor: pointer;
  }
  button:hover { background: #21262d; color: var(--bold); }
  button.primary { color: var(--green); border-color: var(--green); }
  button.danger  { color: var(--red);   border-color: var(--red); }
  button:disabled { color: var(--dim); cursor: not-allowed; background: var(--input-bg); }

  .empty-state {
    color: var(--dim); font-size: var(--fs-md);
    padding: 3rem 1rem; text-align: center;
  }
  .empty-state strong { color: var(--bold); }
  .empty-state .help-doc {
    margin-top: 1rem; font-size: var(--fs-sm);
    text-align: left; max-width: 36rem; margin-inline: auto;
  }

  .capture-card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.75rem;
  }
  .cc-head {
    display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; align-items: center;
    font-size: var(--fs-sm); color: var(--help);
    cursor: pointer; user-select: none;
  }
  .cc-head .when { color: var(--bold); font-size: var(--fs-md); }
  .cc-head .pill {
    border: 1px solid var(--border); border-radius: 999px;
    padding: 0.05rem 0.5rem; font-family: var(--font-mono);
    font-size: var(--fs-xs);
  }
  .cc-head .pill.status-new        { color: var(--yellow); }
  .cc-head .pill.status-reviewed   { color: var(--cyan); }
  .cc-head .pill.status-ready      { color: var(--green); }
  .cc-head .pill.status-dismissed  { color: var(--dim); }
  .cc-head .pill.status-audio_missing {
    color: var(--red); border-color: #5a2424;
  }
  .cc-head .req {
    font-family: var(--font-mono); color: var(--dim);
    font-size: var(--fs-xs);
  }
  .cc-head .duration { color: var(--bold); font-family: var(--font-mono); }
  .cc-head .spacer { flex: 1; }
  .cc-head .expand-arrow {
    color: var(--dim); transition: transform 0.15s;
  }
  .capture-card.open .cc-head .expand-arrow { transform: rotate(90deg); }

  .cc-body { display: none; margin-top: 0.75rem; }
  .capture-card.open .cc-body { display: block; }

  audio { width: 100%; outline: none; }

  .word-strip {
    margin: 0.75rem 0; padding: 0.6rem 0.7rem;
    background: var(--input-bg); border: 1px solid var(--border);
    border-radius: 4px; line-height: 2;
    font-family: var(--font-mono); font-size: var(--fs-md);
    max-height: 20rem; overflow-y: auto;
  }
  /* inline-block lets each word wrap to a new line when the row fills,
     while `white-space: pre` keeps the leading space of words like
     " Vena" intact (so the active-word highlight covers it).  */
  .word-strip .word {
    display: inline-block;
    padding: 0.1rem 0.15rem; border-radius: 3px;
    cursor: pointer; transition: background 0.08s;
    white-space: pre;
  }
  .word-strip .word:hover { background: #21262d; }
  /* Alternating per-member tints inside a GROUP's karaoke band. The hue
     is deliberately subtle — must read as "same row, different segment",
     not "different speaker". Single-capture word strips don't add either
     class, so they look unchanged. The .active / .selected rules below
     override these via their own background, so the karaoke + selection
     highlights always win visually over the segment tint. */
  .word-strip .word.mem-even { background: rgba(80, 160, 220, 0.10); }
  .word-strip .word.mem-odd  { background: rgba(220, 180, 100, 0.10); }
  .word-strip .word.mem-even:hover { background: rgba(80, 160, 220, 0.22); }
  .word-strip .word.mem-odd:hover  { background: rgba(220, 180, 100, 0.22); }
  .word-strip .word.active {
    background: var(--active-word-bg); color: var(--active-word-color);
  }
  .word-strip .word.selected {
    background: var(--selected-word-bg); color: var(--red);
    text-decoration: line-through;
  }
  .word-strip .word.selected.active {
    background: var(--selected-word-bg); color: var(--bold);
  }
  /* Pipeline rules changed this word's text — dotted cyan underline
     signals to the user. The raw form is on the title attribute. */
  .word-strip .word.post-edited {
    border-bottom: 1px dotted var(--cyan);
  }
  /* Pipeline rule deleted this raw token from `final`. Faded +
     struck-through so the user sees what the rule cut, but the
     span stays clickable so audio seek still works at this
     timestamp. */
  .word-strip .word.rule-removed {
    color: var(--dim);
    text-decoration: line-through;
    opacity: 0.55;
  }
  .word-strip .word.rule-removed:hover { opacity: 0.85; }
  /* Inline green replacement next to a struck-through word. Same
     palette as /reports' .diff-ins so the track-changes look reads
     consistently across pages. Lives as a sibling span after the
     last word in the chip's idx..idx_end span. */
  .word-strip .word-replacement {
    color: var(--green); font-weight: 600;
    background: rgba(126, 231, 135, 0.10);
    padding: 0 0.25rem; margin-left: 0.25rem;
    border-radius: 2px;
    font-family: var(--font-mono);
  }

  .cc-textline {
    display: grid; grid-template-columns: 4rem 1fr;
    gap: 0.5rem; padding: 0.15rem 0; align-items: baseline;
    font-family: var(--font-mono); font-size: var(--fs-sm);
    word-break: break-word;
  }
  .cc-textline .tag {
    text-transform: lowercase; color: var(--help);
    font-size: var(--fs-xs); font-family: var(--font-sans);
  }
  .cc-textline.raw   .val { color: var(--fg); }
  .cc-textline.final .val { color: var(--bold); }

  .cc-section {
    margin: 0.625rem 0 0.25rem; padding: 0.5rem 0.625rem;
    background: var(--input-bg); border: 1px solid var(--border);
    border-radius: 4px;
  }
  .cc-section h3 {
    font-size: var(--fs-sm); margin: 0 0 0.35rem; color: var(--help);
    font-family: var(--font-sans); font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .cc-section .help {
    color: var(--help); font-size: var(--fs-xs);
    font-family: var(--font-sans);
  }

  .cc-corrections {
    display: flex; flex-wrap: wrap; gap: 0.375rem; align-items: center;
  }
  .cc-correction {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.15rem 0.5rem;
    font-family: var(--font-mono); font-size: var(--fs-md);
  }
  .cc-correction .wrong   { color: var(--red); text-decoration: line-through; }
  .cc-correction .arrow   { color: var(--dim); }
  .cc-correction .correct-input {
    background: transparent; color: var(--green); font-weight: 600;
    border: 0; outline: 0; font-family: var(--font-mono);
    font-size: var(--fs-md); min-width: 6rem;
  }
  .cc-correction .correct-input:focus {
    background: var(--input-bg); border: 1px solid var(--cyan);
    border-radius: 3px; padding: 0 0.2rem;
  }
  .cc-correction .remove {
    background: transparent; border: 0; color: var(--dim);
    cursor: pointer; padding: 0 0.2rem;
    font-size: var(--fs-md);
  }
  .cc-correction .remove:hover { color: var(--red); }

  .cc-ground {
    width: 100%; min-height: 5rem;
    background: var(--input-bg); color: var(--bold);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.5rem 0.6rem; font-family: var(--font-mono);
    font-size: var(--fs-md); margin-top: 0.25rem;
    white-space: pre-wrap; word-wrap: break-word;
    user-select: text; cursor: text;
  }

  .cc-notes textarea {
    width: 100%; min-height: 3rem; resize: vertical;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.4rem 0.55rem; font-size: var(--fs-md);
    font-family: var(--font-sans);
  }

  .cc-actions {
    display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
    margin-top: 0.75rem; padding-top: 0.5rem;
    border-top: 1px solid var(--border);
  }
  .cc-actions label { font-size: var(--fs-sm); color: var(--help); }
  .cc-actions select {
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.2rem 0.4rem; font-size: var(--fs-md);
  }
  .cc-actions .spacer { flex: 1; }
  .cc-actions .dirty { color: var(--yellow); font-size: var(--fs-sm); }
  .cc-actions .dirty.hidden { display: none; }

  /* Clear-all confirm modal */
  .modal {
    position: fixed; inset: 0; background: rgba(0,0,0,0.65);
    display: none; align-items: center; justify-content: center; z-index: 30;
  }
  .modal.show { display: flex; }
  .modal .box {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.25rem; min-width: 22rem;
    max-width: 32rem;
  }
  .modal h3 { margin: 0 0 0.5rem; color: var(--bold); font-size: var(--fs-xl); }
  .modal p { margin: 0.25rem 0; font-size: var(--fs-md); }
  .modal input {
    width: 100%; background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.35rem 0.5rem; font-family: var(--font-mono);
    font-size: var(--fs-md); margin-top: 0.5rem;
  }
  .modal .actions {
    display: flex; gap: 0.5rem; justify-content: flex-end;
    margin-top: 0.875rem;
  }

  #toast {
    position: fixed; bottom: 1rem; left: 50%; transform: translateX(-50%);
    background: var(--panel); color: var(--bold);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.5rem 0.875rem; font-size: var(--fs-md);
    box-shadow: 0 0.25rem 0.75rem rgba(0,0,0,0.4);
    opacity: 0; transition: opacity 0.2s ease;
    pointer-events: none; z-index: 40;
  }
  #toast.show { opacity: 1; }
  #toast.err { border-color: var(--red); color: var(--red); }

  #token-modal {
    position: fixed; inset: 0; background: rgba(0,0,0,0.65);
    display: none; align-items: center; justify-content: center; z-index: 30;
  }
  #token-modal.show { display: flex; }
  #token-modal .box {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.4rem 1.5rem 1.2rem;
    width: 30rem; max-width: 92vw;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
  }
  #token-modal h3 {
    margin: 0 0 0.5rem 0; color: var(--bold); font-size: var(--fs-xl);
  }
  #token-modal p {
    margin: 0 0 0.9rem 0; line-height: 1.45;
    color: var(--help); font-size: var(--fs-sm);
  }
  #token-modal p code { color: var(--fg); }
  #token-modal input[type=password] {
    box-sizing: border-box; width: 100%;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.55rem 0.7rem; font: inherit; font-size: var(--fs-md);
    line-height: 1.4;
  }
  #token-modal input[type=password]:focus {
    outline: none; border-color: var(--cyan);
  }
  #token-modal .actions {
    display: flex; gap: 0.6rem; justify-content: flex-end;
    margin-top: 1.1rem; padding-top: 0.85rem;
    border-top: 1px solid var(--border);
  }
  #token-modal .actions button {
    font: inherit; font-size: var(--fs-md);
    line-height: 1.4;
    padding: 0.45rem 1rem;
    min-height: 2.25rem;
    border-radius: 4px;
    cursor: pointer;
    background: var(--input-bg);
    color: var(--fg);
    border: 1px solid var(--border);
  }
  #token-modal .actions button:hover { background: #21262d; color: var(--bold); }
  #token-modal .actions button.primary {
    color: var(--green); border-color: var(--green);
  }

  /* ---- Capture grouping ----
   * Selection state lives in JS and projects to per-row .selected CSS.
   * The sticky action bar appears when ≥1 row is selected; Σ duration
   * goes amber at 24 s and red at 28 s — visual feedback for the
   * Whisper-encoder hard cap (≤30 s). */
  .cc-head .sel-checkbox {
    margin: 0 0.4rem 0 0; cursor: pointer;
  }
  .capture-card.selected { outline: 2px solid var(--cyan); }
  .capture-card.is-group {
    border-left: 4px solid var(--magenta);
  }
  .cc-head .group-pill {
    color: var(--magenta); border-color: #4d2d73;
  }
  .cc-head .stale-pill {
    color: var(--yellow); border-color: #4d3e1f;
    background: #2d2a14;
  }
  .cc-head .lock-pill {
    color: var(--dim);
  }
  .group-members {
    margin-top: 0.6rem; padding-left: 1.25rem;
    border-left: 2px solid #2d2a14;
  }
  .group-members .capture-card {
    border-left: 3px solid var(--border);
    background: #11151b;
  }

  #action-bar {
    position: sticky; top: 0; z-index: 8;
    background: #1d293d; border: 1px solid #30538a; border-radius: 6px;
    padding: 0.5rem 0.75rem; margin-bottom: 0.75rem;
    display: none; align-items: center; gap: 0.75rem;
    flex-wrap: wrap;
  }
  #action-bar.show { display: flex; }
  #action-bar .meter {
    font-family: var(--font-mono); font-size: var(--fs-md);
    color: var(--bold); padding: 0.15rem 0.5rem;
    border: 1px solid var(--border); border-radius: 4px;
  }
  #action-bar .meter.amber { color: var(--yellow); border-color: #4d3e1f; }
  #action-bar .meter.red   { color: var(--red);    border-color: #5a2424; }
  #action-bar .summary { color: var(--help); font-size: var(--fs-sm); }
  #action-bar .spacer { flex: 1; }

  /* Merge modal */
  #merge-modal {
    position: fixed; inset: 0; background: rgba(0,0,0,0.6);
    z-index: 1000; align-items: center; justify-content: center;
    display: none;
  }
  #merge-modal.show { display: flex; }
  #merge-modal .box {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.25rem; width: 48rem; max-width: 95vw;
    max-height: 88vh; overflow: auto;
  }
  #merge-modal h3 { margin: 0 0 0.5rem 0; color: var(--bold); }
  #merge-modal textarea {
    width: 100%; min-height: 8rem; box-sizing: border-box;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.5rem 0.6rem; font-family: var(--font-mono);
    font-size: var(--fs-md);
  }
  #merge-modal .row {
    display: flex; gap: 0.5rem; align-items: center; margin: 0.5rem 0;
    flex-wrap: wrap;
  }
  #merge-modal label { font-size: var(--fs-sm); color: var(--help); }
  #merge-modal .actions {
    display: flex; gap: 0.5rem; justify-content: flex-end; margin-top: 1rem;
  }
  #merge-modal .seam-marker {
    display: inline-block; margin: 0 0.15rem;
    color: var(--magenta); font-weight: 700;
  }
  #merge-modal .members-preview {
    background: var(--input-bg); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.5rem 0.7rem;
    font-size: var(--fs-sm); color: var(--dim);
    margin-bottom: 0.5rem; max-height: 8rem; overflow: auto;
  }
  #merge-modal .members-preview .seg-line { padding: 0.15rem 0; }
  #merge-modal .members-preview .seg-time {
    color: var(--help); font-family: var(--font-mono);
    font-size: var(--fs-xs); margin-right: 0.4rem;
  }

  {{NAV_CSS}}
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <span class="title">Whisper — Captures</span>
    {{NAV}}
    <span class="spacer"></span>
    {{SCALE_PICKER}}
  </div>
</header>

<main>
  <div class="toolbar">
    <span class="filt-label">status <span id="filt-status-wrap"></span></span>
    <label>model
      <select id="filt-model">
        <option value="all">all</option>
      </select>
    </label>
    <label>search
      <input id="filt-search" type="text" placeholder="text in raw / final / corrected">
    </label>
    <span class="counts" id="counts"></span>
    <span class="spacer"></span>
    <span id="capture-state" class="capture-state off">capture OFF</span>
    <button id="btn-refresh">Refresh</button>
    <button id="btn-reprocess-all" title="Re-run PIPELINE_RULES on every capture's raw text. Use after editing rules.">Reprocess all</button>
    <button id="btn-export" title="Download ready captures as a tar.gz (manifest.jsonl + audio/)">Export ready</button>
    <button id="btn-clear" class="danger" title="Permanently delete every capture">Clear all</button>
  </div>

  <div id="action-bar">
    <span class="summary"><strong id="ab-count">0</strong> selected</span>
    <span class="meter" id="ab-meter">Σ 0.00 s / 28.00 s</span>
    <span class="summary" id="ab-warn"></span>
    <span class="spacer"></span>
    <button id="ab-merge" class="primary" disabled>Merge into group</button>
    <button id="ab-clear">Clear selection</button>
  </div>

  <div id="list"></div>
</main>

<div id="merge-modal">
  <div class="box">
    <h3>Merge into one training sample</h3>
    <p style="color:var(--help);font-size:var(--fs-sm);margin:0 0 0.6rem;">
      Concatenates the selected captures (same speaker, ≤28 s total) into a
      single ≤30 s Whisper training sample. Inter-segment silence is
      preserved at the joins per the Low-Resource Whisper paper.
    </p>
    <div class="members-preview" id="merge-members"></div>
    <div class="row">
      <label>Join style:
        <select id="merge-join">
          <option value="space">space</option>
          <option value="period_space">period + space</option>
        </select>
      </label>
      <label>Silence gap:
        <select id="merge-silence">
          <option value="200">200 ms</option>
          <option value="300" selected>300 ms</option>
          <option value="400">400 ms</option>
          <option value="500">500 ms</option>
        </select>
      </label>
      <span class="summary" id="merge-summary"></span>
    </div>
    <p style="margin: 0.5rem 0 0.25rem; color: var(--help); font-size: var(--fs-sm);">
      Final result preview (derived from members + chips):
    </p>
    <div id="merge-transcript" class="cc-ground" role="textbox" aria-readonly="true"></div>
    <div class="actions">
      <button id="merge-cancel">Cancel</button>
      <button id="merge-commit" class="primary">Commit merge</button>
    </div>
  </div>
</div>

<div id="confirm-modal" class="modal">
  <div class="box">
    <h3>Clear all captures?</h3>
    <p>This permanently deletes every capture row AND every audio file
       under the captures directory. Training data is irrecoverable.</p>
    <p>Type <strong>CAPTURES</strong> to confirm:</p>
    <input id="confirm-input" type="text" autocomplete="off" placeholder="CAPTURES">
    <div class="actions">
      <button id="confirm-cancel">Cancel</button>
      <button id="confirm-ok" class="danger" disabled>Delete all</button>
    </div>
  </div>
</div>

<div id="token-modal">
  <div class="box">
    <h3>API key</h3>
    <p>Paste your <code>wk_…</code> API key. Stored in sessionStorage until
    tab close.</p>
    <input id="token-input" type="password" autocomplete="off" placeholder="wk_…">
    <div class="actions">
      <button id="token-cancel">Cancel</button>
      <button id="token-save" class="primary">Save</button>
    </div>
  </div>
</div>

<div id="toast"></div>

{{SCALE_PICKER_JS}}
{{SEV_POLLER_JS}}
{{TIME_HELPERS_JS}}
<script>
(function() {
  'use strict';

  // -------------------------------------------------------------------
  // Token storage (mirrors /reports)
  // -------------------------------------------------------------------
  var TOKEN_KEY = 'whisper_api_key';
  function getToken() {
    try { return sessionStorage.getItem(TOKEN_KEY) || ''; } catch(_) { return ''; }
  }
  function setToken(v) {
    try { sessionStorage.setItem(TOKEN_KEY, v || ''); } catch(_) {}
  }
  function showTokenModal(onSaved) {
    var m = document.getElementById('token-modal');
    var inp = document.getElementById('token-input');
    inp.value = getToken();
    m.classList.add('show');
    setTimeout(function() { inp.focus(); inp.select(); }, 50);
    function close() { m.classList.remove('show'); }
    document.getElementById('token-cancel').onclick = close;
    document.getElementById('token-save').onclick = function() {
      setToken(inp.value.trim());
      close();
      if (onSaved) onSaved();
    };
    inp.onkeydown = function(e) {
      if (e.key === 'Enter') document.getElementById('token-save').click();
      if (e.key === 'Escape') close();
    };
  }

  // -------------------------------------------------------------------
  // Toast
  // -------------------------------------------------------------------
  var _toastTimer = null;
  function toast(msg, err) {
    var el = document.getElementById('toast');
    el.textContent = msg;
    el.classList.toggle('err', !!err);
    el.classList.add('show');
    if (_toastTimer) clearTimeout(_toastTimer);
    _toastTimer = setTimeout(function() {
      el.classList.remove('show');
    }, err ? 5000 : 2500);
  }

  // -------------------------------------------------------------------
  // API
  // -------------------------------------------------------------------
  async function api(method, url, body) {
    var headers = { 'Content-Type': 'application/json' };
    var tok = getToken();
    if (tok) headers['Authorization'] = 'Bearer ' + tok;
    var resp = await fetch(url, {
      method: method, headers: headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (resp.status === 401) {
      // After the user pastes a key, re-run load() so the page actually
      // populates (and body.role-admin gets added). Without this the
      // modal save closed the dialog but left the page stuck on the
      // static toolbar with admin nav hidden.
      showTokenModal(function() { load(); });
      throw new Error('unauthorized');
    }
    if (resp.status === 403) {
      var rendered = await _renderAdminOnlyIfNonAdmin();
      if (rendered) throw new Error('not-admin');
    }
    if (!resp.ok) {
      var msg = 'HTTP ' + resp.status;
      try { var j = await resp.json(); if (j && j.detail) msg = j.detail; }
      catch(_) {}
      throw new Error(msg);
    }
    return await resp.json();
  }

  {{NOT_ADMIN_LANDING_JS}}

  async function _renderAdminOnlyIfNonAdmin() {
    try {
      var tok = getToken();
      var hdrs = tok ? { Authorization: 'Bearer ' + tok } : {};
      var r = await fetch('/auth/whoami', { headers: hdrs });
      if (r.ok) {
        var j = await r.json();
        if (j && j.is_admin === false) {
          _renderNotAdminLanding();
          return true;
        }
      }
    } catch (_) {}
    return false;
  }

  // -------------------------------------------------------------------
  // Status button group (radio-style)
  // -------------------------------------------------------------------
  // Replaces the per-row <select> + the filter-bar dropdown with a
  // 1-click switch. Returns { root, setValue }: setValue(v) updates the
  // visible selection WITHOUT firing onChange (used on server round-trip
  // echo so the user's click doesn't trigger a second save).
  function buildStatusButtonGroup(options, current, onChange) {
    var root = document.createElement('div');
    root.className = 'status-btn-group';
    root.setAttribute('role', 'radiogroup');
    var buttons = new Map();
    options.forEach(function(opt) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'status-btn';
      b.dataset.value = opt.value;
      b.textContent = opt.label;
      b.setAttribute('role', 'radio');
      var on = (opt.value === current);
      b.setAttribute('aria-checked', String(on));
      if (on) b.classList.add('active');
      b.addEventListener('click', function() {
        if (b.classList.contains('active')) return;
        buttons.forEach(function(other) {
          other.classList.remove('active');
          other.setAttribute('aria-checked', 'false');
        });
        b.classList.add('active');
        b.setAttribute('aria-checked', 'true');
        try { onChange(opt.value); } catch (e) { /* swallow handler errors */ }
      });
      buttons.set(opt.value, b);
      root.appendChild(b);
    });
    function setValue(v) {
      buttons.forEach(function(btn, key) {
        var on = (key === v);
        btn.classList.toggle('active', on);
        btn.setAttribute('aria-checked', String(on));
      });
    }
    return { root: root, setValue: setValue };
  }

  // -------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------
  var _allCaptures = [];
  var _allGroups = [];
  var _counts = {};
  var _openRows = {};   // cid -> { audio, blobUrl, wordEls, words, finalText, dirty, corrections, ... }
  var _openGroups = {}; // gid -> { audio } — for blob-URL cleanup on render() / beforeunload
  var _selection = new Set();   // capture ids currently selected for merge
  var _lastSelectId = null;     // anchor for shift-range select

  // -------------------------------------------------------------------
  // Selection helpers
  // -------------------------------------------------------------------
  function _handleSelectionClick(row, shift) {
    var visibleIds = applyFilters(_allCaptures)
      .filter(function(r) { return !r.group_id; })
      .map(function(r) { return r.id; });
    if (shift && _lastSelectId && _lastSelectId !== row.id) {
      var i = visibleIds.indexOf(_lastSelectId);
      var j = visibleIds.indexOf(row.id);
      if (i >= 0 && j >= 0) {
        var lo = Math.min(i, j), hi = Math.max(i, j);
        for (var k = lo; k <= hi; k++) _selection.add(visibleIds[k]);
      }
    } else {
      if (_selection.has(row.id)) _selection.delete(row.id);
      else _selection.add(row.id);
    }
    _lastSelectId = row.id;
    _updateActionBar();
    // Toggle the .selected class on every visible card without a full
    // re-render so checkbox focus stays.
    document.querySelectorAll('.capture-card').forEach(function(card) {
      var cid = card.dataset.id;
      if (!cid) return;
      var inSel = _selection.has(cid);
      card.classList.toggle('selected', inSel);
      var cb = card.querySelector('.sel-checkbox');
      if (cb) cb.checked = inSel;
    });
  }

  function _selectedRows() {
    return Array.from(_selection)
      .map(function(id) { return _allCaptures.find(function(r) { return r.id === id; }); })
      .filter(Boolean);
  }

  function _updateActionBar() {
    var bar = document.getElementById('action-bar');
    var n = _selection.size;
    document.getElementById('ab-count').textContent = String(n);
    bar.classList.toggle('show', n >= 1);
    if (n === 0) return;
    var rows = _selectedRows();
    var totalSec = rows.reduce(function(s, r) {
      return s + (r.duration_seconds || 0);
    }, 0);
    var meter = document.getElementById('ab-meter');
    var gap_ms = 300;
    var totalWithGaps = totalSec + (gap_ms / 1000) * Math.max(0, n - 1);
    meter.textContent = 'Σ ' + totalWithGaps.toFixed(2) + ' s / 28.00 s';
    meter.classList.remove('amber', 'red');
    if (totalWithGaps > 28) meter.classList.add('red');
    else if (totalWithGaps > 24) meter.classList.add('amber');

    // Warn on cross-speaker mixes — server enforces, UI nudges.
    var userIds = new Set(rows.map(function(r) { return r.user_id || ''; }));
    var warn = document.getElementById('ab-warn');
    var mixedUsers = userIds.size > 1;
    var hasGrouped = rows.some(function(r) { return r.group_id; });
    if (mixedUsers) {
      warn.textContent = '⚠ multiple speakers — merging not allowed';
      warn.style.color = 'var(--red)';
    } else if (hasGrouped) {
      warn.textContent = '⚠ selection includes captures already in a group';
      warn.style.color = 'var(--red)';
    } else {
      warn.textContent = '';
    }

    var canMerge = n >= 2 && !mixedUsers && !hasGrouped && totalWithGaps <= 28;
    document.getElementById('ab-merge').disabled = !canMerge;
  }

  function _clearSelection() {
    _selection.clear();
    _lastSelectId = null;
    _updateActionBar();
    document.querySelectorAll('.capture-card.selected').forEach(function(c) {
      c.classList.remove('selected');
      var cb = c.querySelector('.sel-checkbox');
      if (cb) cb.checked = false;
    });
  }

  // absTime / relTime / fmtWhen / timeTick are injected via TIME_HELPERS_JS.
  function escapeHtml(s) {
    var d = document.createElement('div');
    d.textContent = s == null ? '' : String(s);
    return d.innerHTML;
  }

  // Filter state — replaces the previous <select id="filt-status">.
  // Initialised to 'new' to preserve the original page default. The
  // status filter button-group bootstrap below mirrors this value.
  var _filtStatus = 'new';

  function applyFilters(rows) {
    var s = _filtStatus;
    var m = document.getElementById('filt-model').value;
    var q = (document.getElementById('filt-search').value || '').trim().toLowerCase();
    return rows.filter(function(r) {
      if (s !== 'all' && r.status !== s) return false;
      if (m !== 'all' && r.model !== m) return false;
      if (!q) return true;
      var hay = (
        (r.raw || '') + ' ' + (r.final || '') + ' ' +
        (r.text_for_training || '') + ' ' +
        (r.corrected_text || '') + ' ' + (r.admin_notes || '')
      ).toLowerCase();
      return hay.indexOf(q) !== -1;
    });
  }

  function rebuildModelFilter() {
    var sel = document.getElementById('filt-model');
    var cur = sel.value;
    var seen = {};
    _allCaptures.forEach(function(r) { if (r.model) seen[r.model] = true; });
    var opts = ['<option value="all">all</option>'];
    Object.keys(seen).sort().forEach(function(m) {
      opts.push('<option value="' + escapeHtml(m) + '">' + escapeHtml(m) + '</option>');
    });
    sel.innerHTML = opts.join('');
    if (Object.prototype.hasOwnProperty.call(seen, cur) || cur === 'all') sel.value = cur;
  }

  function updateCounts() {
    var el = document.getElementById('counts');
    var c = _counts || {};
    el.innerHTML =
      '<span class="n">' + (c.new || 0) + '</span> new · ' +
      '<span class="n">' + (c.reviewed || 0) + '</span> reviewed · ' +
      '<span class="n">' + (c.ready || 0) + '</span> ready · ' +
      '<span class="n">' + (c.dismissed || 0) + '</span> dismissed';
  }

  function updateCaptureBadge(enabled) {
    var el = document.getElementById('capture-state');
    el.textContent = enabled ? 'capture ON' : 'capture OFF';
    el.classList.toggle('on', !!enabled);
    el.classList.toggle('off', !enabled);
  }

  // -------------------------------------------------------------------
  // List card (collapsed)
  // -------------------------------------------------------------------
  function renderCard(r) {
    var card = document.createElement('div');
    card.className = 'capture-card';
    card.dataset.id = r.id;
    if (_selection.has(r.id)) card.classList.add('selected');

    var head = document.createElement('div');
    head.className = 'cc-head';
    // Checkbox is OUTSIDE the click-to-expand zone semantically (we
    // stopPropagation on it).
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.className = 'sel-checkbox';
    cb.checked = _selection.has(r.id);
    cb.title = 'Select for merge';
    cb.addEventListener('click', function(ev) {
      ev.stopPropagation();
      _handleSelectionClick(r, ev.shiftKey);
    });
    head.appendChild(cb);
    head.insertAdjacentHTML('beforeend',
      '<span class="expand-arrow">›</span>' +
      '<span class="when" data-ts="' + (r.created_ts || 0) + '" title="' +
        escapeHtml(absTime(r.created_ts)) + '">' +
        escapeHtml(fmtWhen(r.created_ts)) + '</span>' +
      '<span class="pill status-' + escapeHtml(r.status || 'new') + '">' +
        escapeHtml(r.status || 'new') + '</span>' +
      (r.model ? '<span class="pill">' + escapeHtml(r.model) + '</span>' : '') +
      (r.language ? '<span class="pill">' + escapeHtml(r.language) + '</span>' : '') +
      '<span class="duration">'
        + ((r.effective_duration_seconds !== undefined
            ? r.effective_duration_seconds
            : (r.duration_seconds || 0)).toFixed(1))
        + 's</span>' +
      (r.request_id
        ? '<span class="req">req ' + escapeHtml((r.request_id||'').slice(0,8)) + '</span>'
        : '') +
      (r.username
        ? '<span class="pill" title="speaker">' + escapeHtml(r.username) + '</span>'
        : (r.user_id
            ? '<span class="pill" title="speaker (unknown user)">' + escapeHtml((r.user_id||'').slice(0,6)) + '</span>'
            : '')) +
      (r.group_id
        ? '<span class="pill group-pill" title="member of group ' + escapeHtml(r.group_id.slice(0,8)) + '">in group</span>'
        : '') +
      '<span class="spacer"></span>');
    card.appendChild(head);

    var preview = document.createElement('div');
    preview.style.fontFamily = 'var(--font-mono)';
    preview.style.fontSize = 'var(--fs-sm)';
    preview.style.color = 'var(--help)';
    preview.style.marginTop = '0.3rem';
    preview.style.whiteSpace = 'nowrap';
    preview.style.overflow = 'hidden';
    preview.style.textOverflow = 'ellipsis';
    preview.textContent =
      _applyChipsToText(
        r.text_for_training || r.final || r.raw || '', r.corrections || [])
      || r.raw
      || '(empty)';
    card.appendChild(preview);

    var body = document.createElement('div');
    body.className = 'cc-body';
    card.appendChild(body);

    head.addEventListener('click', function() { toggleExpand(card, r); });

    return card;
  }

  // -------------------------------------------------------------------
  // Expanded body (lazy: built on first open)
  // -------------------------------------------------------------------
  async function toggleExpand(card, r) {
    if (card.classList.contains('open')) {
      collapse(card, r.id);
      return;
    }
    card.classList.add('open');
    var body = card.querySelector('.cc-body');
    if (body.dataset.built !== '1') {
      try {
        var resp = await api('GET', '/captures/api/' + encodeURIComponent(r.id));
        var full = resp.capture;
        // Merge full row into our list-state so the toolbar counts stay
        // current when we change status, etc.
        Object.assign(r, full);
        buildBody(body, r);
        body.dataset.built = '1';
      } catch (e) {
        if (e.message !== 'unauthorized') {
          toast('Failed to load capture: ' + e.message, true);
        }
        card.classList.remove('open');
      }
    }
  }

  function collapse(card, cid) {
    card.classList.remove('open');
    var state = _openRows[cid];
    if (state) {
      if (state.blobUrl) {
        try { URL.revokeObjectURL(state.blobUrl); } catch(_) {}
      }
      if (state.audio) {
        try { state.audio.pause(); } catch(_) {}
      }
    }
    delete _openRows[cid];
    // Reset body so the next toggleExpand re-fetches and re-binds —
    // otherwise the audio element keeps its now-revoked blob URL and
    // the player is dead on second open.
    var body = card.querySelector('.cc-body');
    if (body) { body.dataset.built = '0'; body.innerHTML = ''; }
  }

  function buildBody(body, r) {
    body.innerHTML = '';
    // Release prior state's blob URL + audio if buildBody runs again for
    // a row whose card was rebuilt by render() while open (the new card's
    // dataset.built is '0', so toggleExpand re-runs buildBody).
    var prior = _openRows[r.id];
    if (prior) {
      if (prior.blobUrl) { try { URL.revokeObjectURL(prior.blobUrl); } catch(_) {} }
      if (prior.audio) { try { prior.audio.pause(); } catch(_) {} }
    }
    var state = {
      cid: r.id,
      audio: null,
      blobUrl: null,
      words: r.words || [],
      // Training-form text is the canonical column reviewers see and
      // chips operate against — it's what the export emits and what
      // Whisper will be fine-tuned on. Fall back to `final` (runtime
      // form) then `raw` for captures from before the
      // text_for_training column existed.
      finalText: r.text_for_training || r.final || r.raw || '',
      corrections: (r.corrections || []).map(function(c) {
        return {
          wrong: c.wrong || '',
          correct: c.correct || '',
          idx: typeof c.idx === 'number' ? c.idx : null,
          idx_end: typeof c.idx_end === 'number' ? c.idx_end : null,
        };
      }),
      // Frozen snapshot of `corrections` at GET time. Sent back to the
      // server with the PATCH so it can three-way-merge against any
      // concurrent writes that landed between this load and the save.
      baselineCorrections: JSON.parse(JSON.stringify(r.corrections || [])),
      adminNotes: r.admin_notes || '',
      newStatus: r.status || 'new',
      wordEls: [],
      activeWordIdx: -1,
      dirty: false,
    };
    _openRows[r.id] = state;

    // --- audio player ---
    var audio = document.createElement('audio');
    audio.controls = true;
    audio.preload = 'metadata';
    body.appendChild(audio);
    state.audio = audio;

    // Authenticated audio fetch → blob URL. The server always serves
    // RIFF/WAVE 16 kHz mono (every capture is transcoded on write), so
    // browser decode is reliable across Linux/Windows/macOS.
    var tok = getToken();
    fetch('/captures/api/' + encodeURIComponent(r.id) + '/audio', {
      headers: tok ? { 'Authorization': 'Bearer ' + tok } : {},
    }).then(function(resp) {
      if (resp.status === 401) {
        showTokenModal(function() { toast('Re-open the row to load audio.'); });
        throw new Error('unauthorized');
      }
      if (resp.status === 410) throw new Error('audio file is gone');
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      return resp.blob();
    }).then(function(blob) {
      var url = URL.createObjectURL(blob);
      // If the row was collapsed mid-fetch, `state` is no longer in
      // _openRows — assigning blobUrl to it would orphan the URL
      // (no later cleanup sees it). Revoke immediately in that case.
      if (_openRows[r.id] !== state) { URL.revokeObjectURL(url); return; }
      state.blobUrl = url;
      audio.src = url;
    }).catch(function(e) {
      if (e && e.message !== 'unauthorized') {
        toast('Audio load failed: ' + e.message, true);
      }
    });

    // --- Corrections section: karaoke band + chip list inside one card. ---
    var corrSec = document.createElement('div');
    corrSec.className = 'cc-section';
    corrSec.innerHTML = '<h3>Corrections</h3>'
      + '<div class="help">Click a word to mark it; shift-click another '
      + 'to extend the range; type the corrected text in the chip below.</div>';
    var strip = document.createElement('div');
    strip.className = 'word-strip';
    state.words.forEach(function(w, i) {
      var sp = document.createElement('span');
      sp.className = 'word';
      sp.dataset.idx = String(i);
      sp.dataset.start = String(w.start || 0);
      sp.dataset.end = String(w.end || 0);
      sp.textContent = (w.word || '').replace(/^\s+/, ' ');
      if (w.removed) {
        sp.classList.add('rule-removed');
        sp.title = 'removed by pipeline rule';
      } else if (w.raw_word) {
        sp.title = 'raw: ' + w.raw_word;
        sp.classList.add('post-edited');
      }
      sp.addEventListener('click', function(e) {
        onWordClick(state, i, !!e.shiftKey);
      });
      strip.appendChild(sp);
      state.wordEls.push(sp);
    });
    corrSec.appendChild(strip);

    // Karaoke highlighting via timeupdate (~4 Hz; battery-friendly).
    function onTimeUpdate() {
      var t = audio.currentTime;
      var idx = -1;
      // Linear scan is fine for ≤ a few hundred words. If we ever blow
      // past that, a binary search keyed on word.start would still be
      // under 1 ms per tick.
      for (var i = 0; i < state.words.length; i++) {
        var s = state.words[i].start || 0;
        var e = state.words[i].end || 0;
        if (s <= t && t < e) { idx = i; break; }
      }
      if (idx !== state.activeWordIdx) {
        if (state.activeWordIdx >= 0) {
          var prev = state.wordEls[state.activeWordIdx];
          if (prev) prev.classList.remove('active');
        }
        state.activeWordIdx = idx;
        if (idx >= 0) {
          var cur = state.wordEls[idx];
          if (cur) {
            cur.classList.add('active');
            // Keep the active word visible inside the strip.
            cur.scrollIntoView({ block: 'nearest', inline: 'nearest' });
          }
        }
      }
    }
    audio.addEventListener('timeupdate', onTimeUpdate);

    // --- raw / final reference lines ---
    function textLine(klass, label, value) {
      var row = document.createElement('div');
      row.className = 'cc-textline ' + klass;
      var tag = document.createElement('span');
      tag.className = 'tag';
      tag.textContent = label;
      row.appendChild(tag);
      var v = document.createElement('span');
      v.className = 'val';
      v.textContent = value || '(empty)';
      row.appendChild(v);
      return row;
    }
    body.appendChild(textLine('raw',             'raw',             r.raw));
    body.appendChild(textLine(
      'post-processing', 'post-processing (training)',
      state.finalText));
    // Runtime/symbol-form `final` shown as a secondary line so reviewers
    // can compare against what the dictation client sees in real time.
    // Tagged with the rule-difference so the relationship is explicit.
    if (r.final && r.final !== state.finalText) {
      body.appendChild(textLine(
        'post-processing runtime',
        'runtime (dictation-map applied)', r.final));
    }

    // Chip list nests inside the same Corrections section started above,
    // so the word band and its chip editors render as one harmonized
    // surface (matches the group view's layout).
    var chipBox = document.createElement('div');
    chipBox.className = 'cc-corrections';
    corrSec.appendChild(chipBox);
    body.appendChild(corrSec);
    state.chipBox = chipBox;
    renderChips(state);

    // --- Final result (read-only preview) ---
    // Locked output preview: shows what Whisper would learn to emit
    // for this audio = post-processing text + word-chip corrections.
    // Not editable — to change it, edit chips above or change pipeline
    // rules on /quick-config.
    var gtSec = document.createElement('div');
    gtSec.className = 'cc-section';
    gtSec.innerHTML = '<h3>Final result</h3>'
      + '<div class="help">Computed from <em>post-processing</em> + word '
      + 'corrections + current pipeline rules. To change it, edit chips '
      + 'above or update rules on /quick-config.</div>';
    var gtArea = document.createElement('div');
    gtArea.className = 'cc-ground';
    gtArea.setAttribute('role', 'textbox');
    gtArea.setAttribute('aria-readonly', 'true');
    gtArea.textContent = state.finalText;
    gtSec.appendChild(gtArea);
    body.appendChild(gtSec);
    state.gtArea = gtArea;
    // Initial paint: layer existing chip corrections onto finalText so
    // the preview matches what would be exported right now.
    applyCorrectionsToGround(state);

    // --- notes ---
    var notesSec = document.createElement('div');
    notesSec.className = 'cc-section cc-notes';
    notesSec.innerHTML = '<h3>Admin notes</h3>';
    var notesArea = document.createElement('textarea');
    notesArea.value = state.adminNotes;
    notesArea.addEventListener('input', function() {
      state.adminNotes = notesArea.value;
      markDirty(state);
    });
    notesSec.appendChild(notesArea);
    body.appendChild(notesSec);

    // --- action row ---
    var actions = document.createElement('div');
    actions.className = 'cc-actions';
    var statusLbl = document.createElement('span');
    statusLbl.className = 'cc-status-label';
    statusLbl.textContent = 'status ';
    var statusGrp = buildStatusButtonGroup(
      [{value: 'new',       label: 'New'},
       {value: 'reviewed',  label: 'Reviewed'},
       {value: 'ready',     label: 'Ready'},
       {value: 'dismissed', label: 'Dismissed'}],
      state.newStatus,
      async function(v) {
        // Auto-save on click. Narrow PATCH: status only — does NOT
        // include corrections or baseline_corrections, so an unsaved
        // chip edit stays dirty and only persists when the user clicks
        // Save. Also avoids the three-way-merge path; status has no
        // concurrency hazard.
        var prev = state.newStatus;
        state.newStatus = v;
        try {
          var j = await api('PATCH',
            '/captures/api/' + encodeURIComponent(state.cid),
            { status: v });
          if (j && j.capture) {
            Object.assign(r, j.capture);
            // Defensive resync in case server normalized the value.
            if (j.capture.status) {
              state.newStatus = j.capture.status;
              statusGrp.setValue(j.capture.status);
            }
          }
          reloadCounts();
          toast('Status: ' + v);
        } catch (e) {
          statusGrp.setValue(prev);
          state.newStatus = prev;
          if (e.message !== 'unauthorized') {
            toast('Status save failed: ' + e.message, true);
          }
        }
      }
    );
    statusLbl.appendChild(statusGrp.root);
    actions.appendChild(statusLbl);

    var dirty = document.createElement('span');
    dirty.className = 'dirty hidden';
    dirty.textContent = 'unsaved';
    actions.appendChild(dirty);
    state.dirtyEl = dirty;

    var spc = document.createElement('span');
    spc.className = 'spacer';
    actions.appendChild(spc);

    // Trim silence — cuts leading/trailing silence via Silero VAD, writes
    // a sibling `.trimmed.wav` and pivots the audio GET to serve it. The
    // export also picks up the trimmed companion. Opt-in per singleton.
    var trimBtn = document.createElement('button');
    trimBtn.type = 'button';
    trimBtn.textContent = 'Trim silence';
    trimBtn.title = 'Cut leading/trailing silence (Silero VAD)';
    trimBtn.addEventListener('click', function() {
      onTrimAudio(state, r, trimBtn);
    });
    actions.appendChild(trimBtn);

    // Untrim — only shown when a trimmed companion exists. Deletes the
    // trimmed file + clears the offsets so the original audio is served
    // again and the karaoke uses un-shifted word times.
    if (r.audio_trimmed_relpath) {
      var untrimBtn = document.createElement('button');
      untrimBtn.type = 'button';
      untrimBtn.textContent = '↺ Untrim';
      untrimBtn.title = 'Restore the original (un-trimmed) audio';
      untrimBtn.addEventListener('click', function() {
        onUntrimAudio(state, r, untrimBtn);
      });
      actions.appendChild(untrimBtn);
    }

    // Reprocess — re-runs PIPELINE_RULES on the stored `raw` so the
    // training-form text reflects current rule edits without waiting
    // for the bulk reapply job.
    var reBtn = document.createElement('button');
    reBtn.type = 'button';
    reBtn.textContent = 'Reprocess';
    reBtn.title = 'Re-run PIPELINE_RULES on raw → refresh training text';
    reBtn.addEventListener('click', function() {
      onReprocess(state, r, reBtn);
    });
    actions.appendChild(reBtn);

    var saveBtn = document.createElement('button');
    saveBtn.className = 'primary';
    saveBtn.textContent = 'Save';
    saveBtn.addEventListener('click', function() { onSave(state, r); });
    actions.appendChild(saveBtn);

    var delBtn = document.createElement('button');
    delBtn.className = 'danger';
    delBtn.textContent = 'Delete';
    delBtn.addEventListener('click', function() { onDelete(r); });
    actions.appendChild(delBtn);

    body.appendChild(actions);

    // Initial selection paint from existing corrections
    state.corrections.forEach(function(c) {
      if (typeof c.idx !== 'number') return;
      var end = (typeof c.idx_end === 'number') ? c.idx_end : c.idx;
      for (var j = c.idx; j <= end; j++) selectWord(state, j, true);
    });
  }

  // -------------------------------------------------------------------
  // Word selection / chip helpers (shift-click range extends last chip)
  // -------------------------------------------------------------------
  function selectWord(state, idx, on) {
    var el = state.wordEls[idx];
    if (el) el.classList.toggle('selected', on);
  }

  // Recompute a chip's denormalized `wrong` from the current display
  // words at its idx..idx_end. Lets stored chips (whose `wrong` was
  // the raw STT form before the karaoke band started showing
  // post-pipeline words) self-heal — and ensures applyCorrectionsToGround
  // can find the chip's `wrong` in finalText.
  function recomputeWrong(state, chip) {
    if (typeof chip.idx !== 'number') return chip.wrong || '';
    var b = (typeof chip.idx_end === 'number') ? chip.idx_end : chip.idx;
    var parts = [];
    for (var i = chip.idx; i <= b; i++) {
      var w = state.words[i];
      if (w) parts.push((w.word || ''));
    }
    return parts.join('').replace(/^\s+/, '');
  }

  // Build/update/clear an inline `.word-replacement` sibling next to
  // the last word in a chip's span. Mirrors the .diff-ins green look
  // used by /reports' renderDiff so the user sees the correction
  // inline in the strip, not just in the chip panel below.
  function setReplacementInline(state, chip) {
    if (typeof chip.idx !== 'number') return;
    var lastIdx = (typeof chip.idx_end === 'number') ? chip.idx_end : chip.idx;
    var anchor = state.wordEls[lastIdx];
    if (!anchor) return;
    var nxt = anchor.nextSibling;
    var existing = (nxt && nxt.classList && nxt.classList.contains('word-replacement'))
      ? nxt : null;
    var text = (chip.correct || '').trim();
    if (!text) {
      if (existing) existing.parentNode.removeChild(existing);
      return;
    }
    if (!existing) {
      existing = document.createElement('span');
      existing.className = 'word-replacement';
      anchor.parentNode.insertBefore(existing, anchor.nextSibling);
    }
    existing.textContent = text;
  }

  function clearReplacementInline(state, chip) {
    if (typeof chip.idx !== 'number') return;
    var lastIdx = (typeof chip.idx_end === 'number') ? chip.idx_end : chip.idx;
    var anchor = state.wordEls[lastIdx];
    if (!anchor) return;
    var nxt = anchor.nextSibling;
    if (nxt && nxt.classList && nxt.classList.contains('word-replacement')) {
      nxt.parentNode.removeChild(nxt);
    }
  }
  function chipCovers(c, idx) {
    if (typeof c.idx !== 'number') return false;
    var end = (typeof c.idx_end === 'number') ? c.idx_end : c.idx;
    return c.idx <= idx && idx <= end;
  }
  function spanText(state, a, b) {
    // Verbatim concat of the words in [a..b]. Whisper's word.word
    // carries leading space + trailing punctuation, so the joined
    // string preserves the original spacing/punctuation.
    var parts = [];
    for (var i = a; i <= b; i++) parts.push(state.words[i].word || '');
    return parts.join('').replace(/^\s+/, '');
  }
  function focusLastInput(state) {
    setTimeout(function() {
      var inputs = state.chipBox.querySelectorAll('.correct-input');
      var last = inputs[inputs.length - 1];
      if (last) last.focus();
    }, 10);
  }

  function onWordClick(state, idx, shiftKey) {
    var clicked = state.words[idx];
    if (clicked && clicked.removed) {
      // Rule deleted this token from `final` — a chip would have no
      // anchor in the exported text. Click still seeks audio.
      if (state.audio) {
        try { state.audio.currentTime = parseFloat(clicked.start) || 0; }
        catch (_) {}
      }
      return;
    }
    var last = state.corrections[state.corrections.length - 1];
    if (shiftKey && last && typeof last.idx === 'number') {
      extendLastChip(state, idx);
      return;
    }
    var existing = -1;
    for (var i = 0; i < state.corrections.length; i++) {
      if (chipCovers(state.corrections[i], idx)) { existing = i; break; }
    }
    if (existing >= 0) {
      removeChip(state, existing);
      return;
    }
    state.corrections.push({
      wrong: (state.words[idx].word || '').replace(/^\s+/, ''),
      correct: '',
      idx: idx,
      idx_end: idx,
    });
    selectWord(state, idx, true);
    renderChips(state);
    focusLastInput(state);
    markDirty(state);
    applyCorrectionsToGround(state);
  }

  function refreshAllReplacementsInline(state) {
    // Walk every chip; rebuild its inline sibling. Cheap (≤ chip count).
    state.corrections.forEach(function(c) { setReplacementInline(state, c); });
  }

  function extendLastChip(state, idx) {
    var lastI = state.corrections.length - 1;
    var last = state.corrections[lastI];
    var lastEnd = (typeof last.idx_end === 'number') ? last.idx_end : last.idx;
    var newStart = Math.min(last.idx, idx);
    var newEnd = Math.max(lastEnd, idx);
    // Absorb any earlier chip that overlaps the new range.
    state.corrections = state.corrections.filter(function(c, i) {
      if (i === lastI) return true;
      if (typeof c.idx !== 'number') return true;
      var cEnd = (typeof c.idx_end === 'number') ? c.idx_end : c.idx;
      var overlaps = !(cEnd < newStart || c.idx > newEnd);
      if (overlaps) {
        for (var j = c.idx; j <= cEnd; j++) selectWord(state, j, false);
      }
      return !overlaps;
    });
    last.idx = newStart;
    last.idx_end = newEnd;
    last.wrong = spanText(state, newStart, newEnd);
    for (var j = newStart; j <= newEnd; j++) selectWord(state, j, true);
    renderChips(state);
    focusLastInput(state);
    markDirty(state);
    applyCorrectionsToGround(state);
  }

  function removeChip(state, i) {
    var c = state.corrections[i];
    if (c && typeof c.idx === 'number') {
      var end = (typeof c.idx_end === 'number') ? c.idx_end : c.idx;
      for (var j = c.idx; j <= end; j++) selectWord(state, j, false);
      clearReplacementInline(state, c);
    }
    state.corrections.splice(i, 1);
    renderChips(state);
    markDirty(state);
  }

  function renderChips(state) {
    var box = state.chipBox;
    box.innerHTML = '';
    // Self-heal: chips may have been saved with `wrong` = the raw STT
    // form before the karaoke band started showing post-pipeline
    // words. Recompute from the current display words so the chip,
    // strike-through, and applyCorrectionsToGround all agree on what
    // text the chip is replacing.
    state.corrections.forEach(function(c) {
      c.wrong = recomputeWrong(state, c) || c.wrong;
    });
    refreshAllReplacementsInline(state);
    if (state.corrections.length === 0) {
      var empty = document.createElement('span');
      empty.className = 'help';
      empty.textContent = '(none yet — click a word above)';
      box.appendChild(empty);
      return;
    }
    state.corrections.forEach(function(c, i) {
      var chip = document.createElement('div');
      chip.className = 'cc-correction';

      var w = document.createElement('span');
      w.className = 'wrong';
      w.textContent = c.wrong || '?';
      chip.appendChild(w);

      var arrow = document.createElement('span');
      arrow.className = 'arrow';
      arrow.textContent = '→';
      chip.appendChild(arrow);

      var inp = document.createElement('input');
      inp.type = 'text';
      inp.className = 'correct-input';
      inp.value = c.correct || '';
      inp.placeholder = 'correction';
      inp.spellcheck = false;
      inp.autocomplete = 'off';
      inp.addEventListener('input', function() {
        c.correct = inp.value;
        setReplacementInline(state, c);
        markDirty(state);
      });
      inp.addEventListener('blur', function() {
        applyCorrectionsToGround(state);
      });
      inp.addEventListener('keydown', function(ev) {
        if (ev.key === 'Enter') {
          ev.preventDefault();
          applyCorrectionsToGround(state);
          inp.blur();
        }
      });
      chip.appendChild(inp);

      var rm = document.createElement('button');
      rm.className = 'remove';
      rm.type = 'button';
      rm.textContent = '×';
      rm.addEventListener('click', function() {
        removeChip(state, i);
        applyCorrectionsToGround(state);
      });
      chip.appendChild(rm);

      box.appendChild(chip);
    });
  }

  function applyCorrectionsToGround(state) {
    // Substitute each chip's wrong text with its correct text in
    // state.finalText. Walk corrections in `idx` order so multi-word
    // spans get replaced as one unit. If a chip's `wrong` isn't found
    // verbatim in the final text (regex special chars / whitespace
    // drift), it's left alone.
    //
    // The GT element is now a read-only preview (textContent, not
    // value) — markDirty stays off here because the user can only
    // change GT *indirectly*, by editing a chip, which already
    // markDirty'd on its own.
    var ordered = state.corrections.slice().sort(function(a, b) {
      var ai = typeof a.idx === 'number' ? a.idx : 1e9;
      var bi = typeof b.idx === 'number' ? b.idx : 1e9;
      return ai - bi;
    });
    var out = state.finalText;
    ordered.forEach(function(c) {
      if (!c.correct || !c.wrong) return;
      var i = out.indexOf(c.wrong);
      if (i >= 0) out = out.slice(0, i) + c.correct + out.slice(i + c.wrong.length);
    });
    state.gtArea.textContent = out;
  }

  function markDirty(state) {
    state.dirty = true;
    if (state.dirtyEl) state.dirtyEl.classList.remove('hidden');
  }
  function clearDirty(state) {
    state.dirty = false;
    if (state.dirtyEl) state.dirtyEl.classList.add('hidden');
  }

  // -------------------------------------------------------------------
  // Save / Delete
  // -------------------------------------------------------------------
  async function onSave(state, r) {
    try {
      // Strip idx_end when equal to idx — the server tolerates either,
      // but this keeps payloads compact for inspection.
      var corrections = state.corrections
        .filter(function(c) { return (c.correct || '').trim(); })
        .map(function(c) {
          var out = { wrong: c.wrong || '', correct: (c.correct || '').trim() };
          if (typeof c.idx === 'number') {
            out.idx = c.idx;
            if (typeof c.idx_end === 'number' && c.idx_end !== c.idx) {
              out.idx_end = c.idx_end;
            }
          }
          return out;
        });
      var body = {
        status: state.newStatus,
        corrections: corrections,
        // Snapshot the server returned with the original GET. Lets the
        // server three-way-merge our edits against any concurrent
        // writes (report cascade, another admin in another tab) so
        // those changes survive.
        baseline_corrections: state.baselineCorrections || [],
        admin_notes: state.adminNotes,
      };
      var j = await api('PATCH',
        '/captures/api/' + encodeURIComponent(state.cid), body);
      Object.assign(r, j.capture);
      // Refresh baseline from server's authoritative response so the
      // next save measures deltas from this new ground truth.
      state.baselineCorrections =
        JSON.parse(JSON.stringify(j.capture.corrections || []));
      clearDirty(state);
      toast('Saved.');
      reloadCounts();
    } catch (e) {
      if (e.message !== 'unauthorized') toast('Save failed: ' + e.message, true);
    }
  }

  async function onDelete(r) {
    if (!confirm('Delete this capture and its audio file? This is irreversible.'))
      return;
    try {
      await api('DELETE', '/captures/api/' + encodeURIComponent(r.id));
      var card = document.querySelector('.capture-card[data-id="' + r.id + '"]');
      if (card) {
        collapse(card, r.id);
        card.remove();
      }
      _allCaptures = _allCaptures.filter(function(x) { return x.id !== r.id; });
      reloadCounts();
      toast('Deleted.');
    } catch (e) {
      if (e.message !== 'unauthorized') toast('Delete failed: ' + e.message, true);
    }
  }

  // Trim leading/trailing silence on a singleton via the per-capture
  // /trim endpoint. On success, refresh the row entirely so the karaoke
  // band picks up the shifted word timestamps the server now returns.
  // (Audio reload alone isn't enough — word.start/end values changed.)
  async function onTrimAudio(state, r, btn) {
    var origLabel = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Trimming…';
    try {
      var j = await api('POST',
        '/captures/api/' + encodeURIComponent(state.cid) + '/trim', {});
      if (j && j.trimmed) {
        toast('Trimmed silence.');
        _refreshRowAfterTrim(state.cid);
      } else {
        toast('Nothing to trim — already tight or silent.');
      }
    } catch (e) {
      if (e.message !== 'unauthorized') {
        toast('Trim failed: ' + e.message, true);
      }
    } finally {
      btn.disabled = false;
      btn.textContent = origLabel;
    }
  }

  // Restore the untrimmed audio: delete the trimmed companion file +
  // clear the offsets so the audio GET serves audio_relpath again and
  // the karaoke uses un-shifted word times.
  async function onUntrimAudio(state, r, btn) {
    var origLabel = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Restoring…';
    try {
      await api('POST',
        '/captures/api/' + encodeURIComponent(state.cid) + '/untrim', {});
      toast('Restored original audio.');
      _refreshRowAfterTrim(state.cid);
    } catch (e) {
      if (e.message !== 'unauthorized') {
        toast('Untrim failed: ' + e.message, true);
      }
    } finally {
      btn.disabled = false;
      btn.textContent = origLabel;
    }
  }

  // Force-refresh a row by collapsing it, clearing the body-built flag
  // (so the next expand re-fetches), and re-triggering toggleExpand to
  // rebuild from fresh server data. Cleanest way to make trim/untrim
  // changes (audio URL + shifted word times + effective duration)
  // visible without rolling our own re-paint.
  function _refreshRowAfterTrim(cid) {
    var card = document.querySelector('.capture-card[data-id="' + cid + '"]');
    if (!card) return;
    var wasOpen = card.classList.contains('open');
    var body = card.querySelector('.cc-body');
    if (wasOpen) collapse(card, cid);
    if (body) {
      body.dataset.built = '';
      body.innerHTML = '';
    }
    var r = _allCaptures.find(function(x) { return x.id === cid; });
    if (!r) return;
    if (wasOpen) {
      setTimeout(function() { toggleExpand(card, r); }, 0);
    }
  }

  // Re-run the pipeline on the stored `raw` so the training-form text
  // reflects current PIPELINE_RULES edits. Updates the visible word-
  // strip + preview in place when the text changed.
  async function onReprocess(state, r, btn) {
    var origLabel = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Reprocessing…';
    try {
      var j = await api('POST',
        '/captures/api/' + encodeURIComponent(state.cid) + '/reprocess', {});
      var fresh = j && j.capture ? j.capture : null;
      var changed = j && j.changed ? j.changed : [];
      if (fresh) {
        Object.assign(r, fresh);
        // Update local state.finalText so chips/preview reflect new text.
        state.finalText =
          fresh.text_for_training || fresh.final || fresh.raw || '';
        if (state.gtArea) {
          state.gtArea.textContent = state.finalText;
          applyCorrectionsToGround(state);
        }
      }
      if (changed && changed.length) {
        toast('Reprocessed: ' + changed.join(', ') + '.');
      } else {
        toast('No change — pipeline output already up to date.');
      }
    } catch (e) {
      if (e.message !== 'unauthorized') {
        toast('Reprocess failed: ' + e.message, true);
      }
    } finally {
      btn.disabled = false;
      btn.textContent = origLabel;
    }
  }

  // -------------------------------------------------------------------
  // Clear-all (typed confirmation)
  // -------------------------------------------------------------------
  function onClearAll() {
    var m = document.getElementById('confirm-modal');
    var inp = document.getElementById('confirm-input');
    var ok = document.getElementById('confirm-ok');
    inp.value = '';
    ok.disabled = true;
    m.classList.add('show');
    setTimeout(function() { inp.focus(); }, 50);
    inp.oninput = function() { ok.disabled = inp.value.trim() !== 'CAPTURES'; };
    document.getElementById('confirm-cancel').onclick = function() {
      m.classList.remove('show');
    };
    ok.onclick = async function() {
      m.classList.remove('show');
      try {
        var j = await api('POST', '/captures/api/clear', { confirm: 'CAPTURES' });
        toast('Cleared ' + j.deleted + ' capture' +
          (j.deleted === 1 ? '' : 's') + '.');
        // Drop any open blob URLs first — both per-row audio blobs AND
        // group-card audio (their <audio>.src is also a blob: URL).
        Object.keys(_openRows).forEach(function(cid) {
          var st = _openRows[cid];
          if (st && st.blobUrl) URL.revokeObjectURL(st.blobUrl);
        });
        _openRows = {};
        Object.keys(_openGroups).forEach(function(gid) {
          var st = _openGroups[gid];
          if (st && st.audio && st.audio.src && st.audio.src.indexOf('blob:') === 0) {
            try { URL.revokeObjectURL(st.audio.src); } catch(_) {}
          }
        });
        _openGroups = {};
        await load();
      } catch (e) {
        if (e.message !== 'unauthorized') toast('Failed: ' + e.message, true);
      }
    };
  }

  // -------------------------------------------------------------------
  // Export
  // -------------------------------------------------------------------
  // Bulk-reprocess: trigger the background job that re-runs PIPELINE_RULES
  // on every capture's `raw` text and updates `final` + `text_for_training`.
  // Same worker /quick-config/reapply-rules uses; idempotent — a second
  // call while running returns the current state.
  async function onReprocessAll() {
    if (!confirm('Re-run PIPELINE_RULES on every capture? Updates final + training text in place.\n\nThis runs in the background.'))
      return;
    try {
      var j = await api('POST', '/captures/api/reprocess-all', {});
      var st = j && j.status ? j.status : 'started';
      toast('Reprocess-all ' + st + '.');
    } catch (e) {
      if (e.message !== 'unauthorized') {
        toast('Reprocess-all failed: ' + e.message, true);
      }
    }
  }

  function onExport() {
    var tok = getToken();
    fetch('/captures/api/export?only_status=ready&include_audio=1', {
      headers: tok ? { 'Authorization': 'Bearer ' + tok } : {},
    }).then(function(resp) {
      if (resp.status === 401) {
        showTokenModal(function() { onExport(); });
        throw new Error('unauthorized');
      }
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      var fname = 'whisper-captures-' +
        new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19) + '.tar.gz';
      var cd = resp.headers.get('Content-Disposition') || '';
      var m = /filename="([^"]+)"/.exec(cd);
      if (m) fname = m[1];
      return resp.blob().then(function(blob) {
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url; a.download = fname;
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(function() { URL.revokeObjectURL(url); }, 1000);
      });
    }).catch(function(e) {
      if (e && e.message !== 'unauthorized') {
        toast('Export failed: ' + e.message, true);
      }
    });
  }

  // -------------------------------------------------------------------
  // Render loop — assigned (not function-declared) so the groups-aware
  // override further down replaces it without leaving a dead shadowed
  // singleton-only definition in the file.
  // -------------------------------------------------------------------
  var render;

  // -------------------------------------------------------------------
  // Load
  // -------------------------------------------------------------------
  async function load() {
    try {
      var j = await api('GET', '/captures/api/list?status=all&limit=500');
      _allCaptures = j.captures || [];
      _counts = j.counts || {};
      // Pull groups in parallel-shape; failure is non-fatal (admin sees no groups).
      try {
        var jg = await api('GET', '/captures/api/groups');
        _allGroups = jg.groups || [];
      } catch (_) { _allGroups = []; }
      updateCaptureBadge(!!j.enabled);
      rebuildModelFilter();
      updateCounts();
      _clearSelection();
      render();
      if (j.is_admin) document.body.classList.add('role-admin');
    } catch (e) {
      if (e.message === 'unauthorized' || e.message === 'not-admin') return;
      toast('Failed to load captures: ' + e.message, true);
    }
  }

  async function reloadCounts() {
    try {
      var j = await api('GET', '/captures/api/list?status=all&limit=1');
      _counts = j.counts || _counts;
      updateCaptureBadge(!!j.enabled);
      updateCounts();
    } catch (_) {}
  }

  // -------------------------------------------------------------------
  // Merge modal
  // -------------------------------------------------------------------
  var JOIN_STR = { space: ' ', period_space: '. ' };

  function _applyChipsToText(text, corrections) {
    // Mirror of Python _apply_chips_to_text — walk chips in idx order
    // and substitute `wrong` → `correct` in the text. Used to layer
    // each member's chip corrections on top of post-processing before
    // joining into the group's final-result preview.
    if (!corrections || !corrections.length) return text || '';
    var out = text || '';
    var ordered = corrections.slice().filter(function(c) {
      return c && typeof c === 'object';
    });
    ordered.sort(function(a, b) {
      var ai = typeof a.idx === 'number' ? a.idx : 1e9;
      var bi = typeof b.idx === 'number' ? b.idx : 1e9;
      return ai - bi;
    });
    ordered.forEach(function(c) {
      var wrong = c.wrong || '';
      var correct = c.correct || '';
      if (!wrong || !correct) return;
      var i = out.indexOf(wrong);
      if (i >= 0) out = out.slice(0, i) + correct + out.slice(i + wrong.length);
    });
    return out;
  }

  function _buildDefaultTranscript(rows, strategy) {
    // Chips-applied join. Source from `text_for_training` so the preview
    // matches the word-form training text the export will actually emit
    // (falls back to `final` then `raw` for legacy captures predating
    // text_for_training).
    var sep = JOIN_STR[strategy] || ' ';
    return rows.map(function(r) {
      var base = r.text_for_training || r.final || r.raw || '';
      return _applyChipsToText(base, r.corrections || []).trim();
    }).filter(Boolean).join(sep);
  }

  function _renderMergePreview(rows) {
    var el = document.getElementById('merge-members');
    el.innerHTML = '';
    rows.forEach(function(r, i) {
      var line = document.createElement('div');
      line.className = 'seg-line';
      var base = r.text_for_training || r.final || r.raw || '';
      var memberText = _applyChipsToText(base, r.corrections || []);
      line.innerHTML = '<span class="seg-time">[' + (i + 1) + '] '
        + (r.duration_seconds || 0).toFixed(1) + 's</span>'
        + escapeHtml(memberText.slice(0, 200));
      el.appendChild(line);
    });
  }

  function _openMergeModal() {
    var rows = _selectedRows();
    if (rows.length < 2) return;
    var modal = document.getElementById('merge-modal');
    _renderMergePreview(rows);
    var joinSel = document.getElementById('merge-join');
    var silSel = document.getElementById('merge-silence');
    var ta = document.getElementById('merge-transcript');
    function refreshTranscript() {
      ta.textContent = _buildDefaultTranscript(rows, joinSel.value);
    }
    refreshTranscript();
    joinSel.onchange = refreshTranscript;
    // Live summary
    function refreshSummary() {
      var n = rows.length;
      var totalAudio = rows.reduce(function(s, r) { return s + (r.duration_seconds || 0); }, 0);
      var gap = (parseInt(silSel.value, 10) || 0) / 1000;
      var total = totalAudio + gap * Math.max(0, n - 1);
      document.getElementById('merge-summary').textContent =
        n + ' segments · Σ ' + total.toFixed(2) + ' s / 28.00 s';
    }
    refreshSummary();
    silSel.onchange = refreshSummary;
    document.getElementById('merge-cancel').onclick = function() {
      modal.classList.remove('show');
    };
    document.getElementById('merge-commit').onclick = function() {
      // Server derives the transcript from members + chips on create
      // (mirrors the locked preview); we don't send a client-side string.
      var payload = {
        member_ids: rows.map(function(r) { return r.id; }),
        transcript: '',
        join_strategy: joinSel.value,
        silence_ms: parseInt(silSel.value, 10) || 300,
      };
      api('POST', '/captures/api/groups', payload)
        .then(function() {
          modal.classList.remove('show');
          toast('Group created');
          return load();
        })
        .catch(function(e) {
          if (e && e.message !== 'unauthorized') toast(e.message, true);
        });
    };
    modal.classList.add('show');
  }

  // -------------------------------------------------------------------
  // Group row rendering — packed-for-fine-tune training samples
  // -------------------------------------------------------------------
  function _renderGroupCard(g) {
    var card = document.createElement('div');
    card.className = 'capture-card is-group';
    card.dataset.gid = g.id;
    var head = document.createElement('div');
    head.className = 'cc-head';
    head.innerHTML =
      '<span class="expand-arrow">›</span>' +
      '<span class="when" data-ts="' + (g.created_ts || 0) + '" title="' +
        escapeHtml(absTime(g.created_ts)) + '">' +
        escapeHtml(fmtWhen(g.created_ts)) + '</span>' +
      '<span class="pill group-pill">group</span>' +
      '<span class="pill status-' + escapeHtml(g.status || 'new') + '">' +
        escapeHtml(g.status || 'new') + '</span>' +
      (g.username
        ? '<span class="pill" title="speaker">' + escapeHtml(g.username) + '</span>'
        : '<span class="pill" title="speaker (unknown user)">' +
            escapeHtml((g.user_id || '?').slice(0, 6)) + '</span>') +
      '<span class="duration">' +
        ((g.merged_duration_ms || 0) / 1000).toFixed(2) + 's</span>' +
      (g.is_stale
        ? '<span class="pill stale-pill" title="member changed since merge">stale</span>'
        : '') +
      (g.is_locked
        ? '<span class="pill lock-pill" title="locked — protect from edits">locked</span>'
        : '') +
      '<span class="spacer"></span>';
    card.appendChild(head);

    var preview = document.createElement('div');
    preview.style.fontFamily = 'var(--font-mono)';
    preview.style.fontSize = 'var(--fs-sm)';
    preview.style.color = 'var(--help)';
    preview.style.marginTop = '0.3rem';
    preview.style.whiteSpace = 'nowrap';
    preview.style.overflow = 'hidden';
    preview.style.textOverflow = 'ellipsis';
    preview.textContent = g.transcript || '(empty)';
    card.appendChild(preview);

    var body = document.createElement('div');
    body.className = 'cc-body';
    card.appendChild(body);
    head.addEventListener('click', function() { _toggleGroupExpand(card, g); });
    return card;
  }

  function _toggleGroupExpand(card, g) {
    if (card.classList.contains('open')) {
      card.classList.remove('open');
      return;
    }
    card.classList.add('open');
    var body = card.querySelector('.cc-body');
    if (body.dataset.built === '1') return;
    // Guard against concurrent expands on the same group (collapse →
    // re-expand while the first GET is still in flight) — a second build
    // would append a second <audio> element whose blob URL the existing
    // _openGroups tracking can't reach, leaking on render() wipe.
    if (body.dataset.fetching === '1') return;
    body.dataset.fetching = '1';
    api('GET', '/captures/api/groups/' + encodeURIComponent(g.id))
      .then(function(j) {
        var detail = j.group || g;

        // --- Skeleton: audio slot, transcript textarea, karaoke band,
        // chip box, settings row, button row, members list. All built
        // once; their CONTENT is filled (and refilled) by
        // applyServerGroup() below. ---
        var audio = document.createElement('audio');
        audio.controls = true;
        audio.style.marginTop = '0.5rem';
        audio.preload = 'metadata';
        var audioSlot = document.createElement('div');
        audioSlot.appendChild(audio);
        body.appendChild(audioSlot);

        // Per-member raw + post-processing reference rows. Same shape
        // as the single-capture textLine, but iterated per member so
        // the group surface mirrors what each captured segment carried.
        function _textLine(klass, label, value) {
          var row = document.createElement('div');
          row.className = 'cc-textline ' + klass;
          var tag = document.createElement('span');
          tag.className = 'tag';
          tag.textContent = label;
          row.appendChild(tag);
          var v = document.createElement('span');
          v.className = 'val' + (value ? '' : ' dim');
          v.textContent = value || '(empty)';
          row.appendChild(v);
          return row;
        }
        var refsWrap = document.createElement('div');
        body.appendChild(refsWrap);

        // Group "Final result" read-only preview (the gtArea).
        // Identical lock-down to single-capture GT: this is a derived
        // view of (members + chips). To change it, edit chips above.
        var ta = document.createElement('div');
        ta.className = 'cc-ground';
        ta.setAttribute('role', 'textbox');
        ta.setAttribute('aria-readonly', 'true');

        var groupState = {
          gid: detail.id,
          audio: audio,
          words: [],
          finalText: '',
          corrections: [],
          wordEls: [],
          activeWordIdx: -1,
          chipBox: null,
          gtArea: ta,
          adminNotes: '',
          newStatus: 'new',
          dirty: false,
          dirtyEl: null,
        };

        // ---- Corrections section: karaoke band + chip list together. ----
        var corrSec = document.createElement('div');
        corrSec.className = 'cc-section';
        corrSec.innerHTML = '<h3>Corrections</h3>'
          + '<div class="help">Click a word to mark it; shift-click '
          + 'another to extend the range; type the corrected text in '
          + 'the chip below. Edits auto-apply to the final-result '
          + 'preview on blur / Enter.</div>';
        var staleHint = document.createElement('div');
        staleHint.className = 'help';
        staleHint.style.color = 'var(--cyan)';
        corrSec.appendChild(staleHint);
        var strip = document.createElement('div');
        strip.className = 'word-strip';
        corrSec.appendChild(strip);

        // Karaoke highlight via timeupdate (attached once; reads the
        // live groupState.words). Survives applyServerGroup re-renders.
        audio.addEventListener('timeupdate', function() {
          var t = audio.currentTime;
          var idx = -1;
          for (var i = 0; i < groupState.words.length; i++) {
            var s = groupState.words[i].start || 0;
            var e = groupState.words[i].end || 0;
            if (s <= t && t < e) { idx = i; break; }
          }
          if (idx !== groupState.activeWordIdx) {
            if (groupState.activeWordIdx >= 0) {
              var prev = groupState.wordEls[groupState.activeWordIdx];
              if (prev) prev.classList.remove('active');
            }
            groupState.activeWordIdx = idx;
            if (idx >= 0) {
              var cur = groupState.wordEls[idx];
              if (cur) {
                cur.classList.add('active');
                cur.scrollIntoView({ block: 'nearest', inline: 'nearest' });
              }
            }
          }
        });

        // Chip list nests inside the same Corrections section so the
        // word strip and its chips read as one harmonized surface.
        var chipBox = document.createElement('div');
        chipBox.className = 'cc-corrections';
        corrSec.appendChild(chipBox);
        body.appendChild(corrSec);
        groupState.chipBox = chipBox;

        // ---- Final result section (read-only preview). ----
        var gtSec = document.createElement('div');
        gtSec.className = 'cc-section';
        gtSec.innerHTML = '<h3>Final result</h3>'
          + '<div class="help">Computed from members\' post-processing '
          + 'text + word corrections. To change it, edit chips above or '
          + 'update rules on /quick-config.</div>';
        gtSec.appendChild(ta);
        body.appendChild(gtSec);

        // --- Merge settings row (silence gap + join strategy) ---
        var settingsSec = document.createElement('div');
        settingsSec.className = 'cc-section';
        settingsSec.style.cssText = 'display:flex;gap:1rem;align-items:center;'
          + 'flex-wrap:wrap;margin-top:0.5rem;';
        var joinLabel = document.createElement('label');
        joinLabel.style.cssText = 'display:flex;gap:0.4rem;align-items:center;';
        joinLabel.appendChild(document.createTextNode('Join style: '));
        var joinSel = document.createElement('select');
        ['space','period_space'].forEach(function(v) {
          var opt = document.createElement('option');
          opt.value = v;
          opt.textContent = v === 'space' ? 'space' : 'period + space';
          joinSel.appendChild(opt);
        });
        joinLabel.appendChild(joinSel);
        settingsSec.appendChild(joinLabel);

        var silLabel = document.createElement('label');
        silLabel.style.cssText = 'display:flex;gap:0.4rem;align-items:center;';
        silLabel.appendChild(document.createTextNode('Silence gap: '));
        var silSel = document.createElement('select');
        [0,100,200,300,400,500,750,1000].forEach(function(v) {
          var opt = document.createElement('option');
          opt.value = String(v);
          opt.textContent = v + ' ms';
          silSel.appendChild(opt);
        });
        silLabel.appendChild(silSel);
        settingsSec.appendChild(silLabel);

        var settingsHint = document.createElement('span');
        settingsHint.className = 'help';
        settingsHint.style.marginLeft = '0.5rem';
        settingsSec.appendChild(settingsHint);
        body.appendChild(settingsSec);

        // Mutable baselines so applyServerGroup can update them and
        // settingsDirty() reads the latest "saved" state.
        var origJoin = detail.transcript_join_strategy || 'space';
        var origSil  = detail.inter_segment_silence_ms || 300;
        function settingsDirty() {
          return joinSel.value !== origJoin
              || parseInt(silSel.value, 10) !== origSil;
        }
        function refreshSettingsHint() {
          if (settingsDirty()) {
            settingsHint.textContent =
              'changes pending — Save or Regenerate to rebuild audio';
            settingsHint.style.color = 'var(--cyan)';
          } else {
            settingsHint.textContent = '';
          }
        }
        joinSel.addEventListener('change', refreshSettingsHint);
        silSel.addEventListener('change', refreshSettingsHint);

        // --- Admin notes (matches single-capture layout). ---
        var notesSec = document.createElement('div');
        notesSec.className = 'cc-section cc-notes';
        notesSec.innerHTML = '<h3>Admin notes</h3>';
        var notesArea = document.createElement('textarea');
        notesArea.addEventListener('input', function() {
          groupState.adminNotes = notesArea.value;
          markDirty(groupState);
        });
        notesSec.appendChild(notesArea);
        body.appendChild(notesSec);

        // --- Action row: status + save / regenerate / lock / dissolve ---
        var actions = document.createElement('div');
        actions.className = 'cc-actions';
        var statusLbl = document.createElement('span');
        statusLbl.className = 'cc-status-label';
        statusLbl.textContent = 'status ';
        var statusGrp = buildStatusButtonGroup(
          [{value: 'new',       label: 'New'},
           {value: 'reviewed',  label: 'Reviewed'},
           {value: 'ready',     label: 'Ready'},
           {value: 'dismissed', label: 'Dismissed'}],
          groupState.newStatus || 'new',
          async function(v) {
            var prev = groupState.newStatus;
            groupState.newStatus = v;
            try {
              // Narrow PATCH — status only. Concurrent transcript or
              // chip edits in this view stay dirty and persist only on
              // Save click; the three-way-merge path isn't engaged.
              await api('PATCH',
                '/captures/api/groups/' + encodeURIComponent(g.id),
                { status: v });
              // Sync the collapsed-list header so the next render() /
              // status-filter sees the new status without a full reload.
              // Status-only PATCH doesn't need (and must not run) the
              // full applyServerGroup — it would rebuild word strip,
              // chips, audio, etc. for a single-pill change.
              var idx = _allGroups.findIndex(function(x) { return x.id === g.id; });
              if (idx >= 0) _allGroups[idx].status = v;
              reloadCounts();
              toast('Status: ' + v);
            } catch (e) {
              statusGrp.setValue(prev);
              groupState.newStatus = prev;
              if (e.message !== 'unauthorized') {
                toast('Status save failed: ' + e.message, true);
              }
            }
          }
        );
        statusLbl.appendChild(statusGrp.root);
        actions.appendChild(statusLbl);
        var dirtyEl = document.createElement('span');
        dirtyEl.className = 'dirty hidden';
        dirtyEl.textContent = 'unsaved';
        actions.appendChild(dirtyEl);
        groupState.dirtyEl = dirtyEl;
        var spc = document.createElement('span');
        spc.className = 'spacer';
        actions.appendChild(spc);
        var saveTBtn = document.createElement('button');
        saveTBtn.textContent = 'Save';
        saveTBtn.className = 'primary';
        actions.appendChild(saveTBtn);
        var regenBtn = document.createElement('button');
        actions.appendChild(regenBtn);
        var lockBtn = document.createElement('button');
        actions.appendChild(lockBtn);
        var dissolveBtn = document.createElement('button');
        dissolveBtn.textContent = 'Dissolve group';
        dissolveBtn.className = 'danger';
        actions.appendChild(dissolveBtn);
        body.appendChild(actions);

        // --- Members list (replaced on each applyServerGroup) ---
        var membersDiv = document.createElement('div');
        membersDiv.className = 'group-members';
        body.appendChild(membersDiv);

        // --- refreshAudio: re-fetches the blob; on failure shows the
        // missing-audio banner in place of the <audio> element. The
        // banner's button hits applyServerGroup so success heals the
        // open card without a load() wipe. ---
        function refreshAudio() {
          // Reclaim the <audio> element if the banner replaced it.
          if (audio.src && audio.src.indexOf('blob:') === 0) {
            try { URL.revokeObjectURL(audio.src); } catch (_) {}
          }
          audio.removeAttribute('src');
          if (audioSlot.firstChild !== audio) {
            audioSlot.innerHTML = '';
            audioSlot.appendChild(audio);
          }
          fetch('/captures/api/groups/' + encodeURIComponent(g.id) + '/audio',
                { headers: { Authorization: 'Bearer ' + getToken() } })
            .then(function(r) {
              if (!r.ok) throw new Error('HTTP ' + r.status);
              return r.blob();
            })
            .then(function(b) { audio.src = URL.createObjectURL(b); })
            .catch(function(err) {
              audioSlot.innerHTML = '';
              var banner = document.createElement('div');
              banner.className = 'cc-section';
              banner.style.cssText = 'background:#3a2424;border:1px solid '
                + 'var(--border);border-radius:4px;padding:0.5rem 0.75rem;'
                + 'margin-top:0.5rem;display:flex;align-items:center;'
                + 'gap:0.75rem;flex-wrap:wrap;';
              banner.innerHTML = '<span style="color:#ffd1d1;">'
                + '⚠ Audio unavailable for this group ('
                + escapeHtml(err && err.message || 'fetch failed') + ').'
                + '</span>';
              var fixBtn = document.createElement('button');
              fixBtn.textContent = 'Regenerate audio';
              fixBtn.className = 'primary';
              fixBtn.onclick = function() {
                fixBtn.disabled = true;
                api('POST', '/captures/api/groups/'
                    + encodeURIComponent(g.id) + '/regenerate')
                  .then(function(j) {
                    toast('Regenerated');
                    applyServerGroup(j.group || {});
                  })
                  .catch(function(e) {
                    fixBtn.disabled = false;
                    if (e.message !== 'unauthorized') toast(e.message, true);
                  });
              };
              banner.appendChild(fixBtn);
              audioSlot.appendChild(banner);
            });
        }

        // --- applyServerGroup: the single source of truth for "the
        // server told us the group looks like X, now reflect X in the
        // open card." Called once on initial expand and again after
        // every Save / Regenerate. Replaces the previous load()-wipe
        // pattern. ---
        function applyServerGroup(d) {
          // 1. groupState data.
          groupState.words = d.merged_words || [];
          groupState.corrections = (d.corrections || []).map(function(c) {
            return {
              wrong:   c.wrong   || '',
              correct: c.correct || '',
              idx:     typeof c.idx     === 'number' ? c.idx     : null,
              idx_end: typeof c.idx_end === 'number' ? c.idx_end : null,
            };
          });
          // Frozen snapshot of the server's chip set at THIS moment.
          // Sent back with the next PATCH so the server can apply the
          // user's deltas on top of any concurrent member-chip writes
          // (report cascades, cross-tab admin saves).
          groupState.baselineCorrections =
            JSON.parse(JSON.stringify(d.corrections || []));
          groupState.finalText = d.transcript || '';
          groupState.adminNotes = d.admin_notes || '';
          groupState.newStatus = d.status || 'new';
          groupState.wordEls = [];
          groupState.activeWordIdx = -1;
          ta.textContent = d.transcript || '';

          // Per-member raw + post-processing reference rows.
          // "post-processing" line sources from the training-form text
          // so the displayed member transcript matches the export.
          refsWrap.innerHTML = '';
          (d.members || []).forEach(function(m, i) {
            var prefix = '[' + (i + 1) + '] ';
            refsWrap.appendChild(_textLine('raw',
              prefix + 'raw', m.raw || ''));
            var training = m.text_for_training || m.final || '';
            refsWrap.appendChild(_textLine('post-processing',
              prefix + 'post-processing (training)', training));
          });

          // Status button-group + admin notes (sync from server snapshot).
          statusGrp.setValue(d.status || 'new');
          notesArea.value = d.admin_notes || '';
          clearDirty(groupState);

          // 2. Dropdowns + baselines.
          silSel.value = String(d.inter_segment_silence_ms || 300);
          joinSel.value = d.transcript_join_strategy || 'space';
          origSil  = d.inter_segment_silence_ms || 300;
          origJoin = d.transcript_join_strategy || 'space';
          refreshSettingsHint();

          // 3. Stale hint + regen label.
          staleHint.textContent = d.is_stale
            ? '⚠ audio drift detected — regenerate to realign timings'
            : '';
          regenBtn.textContent = d.is_stale
            ? 'Regenerate audio (clear stale)'
            : 'Regenerate audio';

          // 4. Lock / Dissolve label & visibility.
          lockBtn.textContent = d.is_locked ? 'Unlock' : 'Lock';
          dissolveBtn.style.display = d.is_locked ? 'none' : '';

          // 5. Word strip (alternating per-member tints via mem-even /
          // mem-odd classes; styles in the .word-strip CSS block).
          strip.innerHTML = '';
          var prevMember = -1;
          groupState.words.forEach(function(w, i) {
            var sp = document.createElement('span');
            sp.className = 'word ' + ((w.member_idx % 2 === 0) ? 'mem-even' : 'mem-odd');
            sp.dataset.idx = String(i);
            sp.dataset.start = String(w.start || 0);
            sp.dataset.end = String(w.end || 0);
            sp.dataset.member = String(w.member_idx || 0);
            // Small horizontal breathing room at member boundaries (no
            // hard rule — the tint shift does the work visually).
            if (w.member_idx !== prevMember && prevMember !== -1) {
              sp.style.marginLeft = '0.4rem';
            }
            prevMember = w.member_idx;
            sp.textContent = (w.word || '').replace(/^\s+/, ' ');
            if (w.removed) {
              sp.classList.add('rule-removed');
              sp.title = 'removed by pipeline rule';
            } else if (w.raw_word) {
              sp.title = 'raw: ' + w.raw_word;
              sp.classList.add('post-edited');
            }
            sp.addEventListener('click', function(e) {
              onWordClick(groupState, i, !!e.shiftKey);
            });
            strip.appendChild(sp);
            groupState.wordEls.push(sp);
          });

          // 6. Chips + pre-marked word selections from saved corrections.
          renderChips(groupState);
          groupState.corrections.forEach(function(c) {
            if (typeof c.idx !== 'number') return;
            var end = (typeof c.idx_end === 'number') ? c.idx_end : c.idx;
            for (var j = c.idx; j <= end; j++) selectWord(groupState, j, true);
          });

          // 7. Members list.
          membersDiv.innerHTML = '<p style="font-size:var(--fs-sm);'
            + 'color:var(--help);margin:0.4rem 0;">Members ('
            + (d.members || []).length + ')</p>';
          (d.members || []).forEach(function(m) {
            var line = document.createElement('div');
            line.style.cssText = 'font-size:var(--fs-sm);color:var(--dim);padding:0.2rem 0;';
            var base = m.text_for_training || m.final || m.raw || '';
            var memberText = _applyChipsToText(base, m.corrections || []);
            var memDur = (m.effective_duration_seconds !== undefined
              ? m.effective_duration_seconds
              : (m.duration_seconds || 0));
            line.innerHTML = '[' + (m.group_order + 1) + '] '
              + memDur.toFixed(1) + 's · '
              + escapeHtml(memberText.slice(0, 120));
            membersDiv.appendChild(line);
          });

          // 8. Audio (re-fetch; merged WAV may have been rebuilt).
          refreshAudio();

          // 9. Keep the collapsed-list in-memory entry in sync without
          // triggering a full render — this way the group's "stale"
          // pill, created_ts, etc. on the closed card header stays
          // truthful after a save.
          var idx = _allGroups.findIndex(function(x) { return x.id === d.id; });
          if (idx >= 0) Object.assign(_allGroups[idx], d);
        }

        // --- Save handler: PATCH everything; mutate the card in place. ---
        saveTBtn.onclick = function() {
          var payload = {
            join_strategy: joinSel.value,
            silence_ms:    parseInt(silSel.value, 10),
            status:        groupState.newStatus,
            admin_notes:   groupState.adminNotes,
            corrections:   groupState.corrections.map(function(c) {
              return {
                wrong:   c.wrong   || '',
                correct: c.correct || '',
                idx:     typeof c.idx     === 'number' ? c.idx     : null,
                idx_end: typeof c.idx_end === 'number' ? c.idx_end : null,
              };
            }),
            // Snapshot the server returned when we loaded; pairs with
            // `corrections` so the server can three-way-merge our
            // deltas against any concurrent member-chip writes that
            // landed in between.
            baseline_corrections: groupState.baselineCorrections || [],
          };
          saveTBtn.disabled = true;
          api('PATCH', '/captures/api/groups/' + encodeURIComponent(g.id),
              payload)
            .then(function(j) {
              applyServerGroup(j.group || {});
              toast('Saved');
              reloadCounts();
            })
            .catch(function(e) {
              if (e.message !== 'unauthorized') toast(e.message, true);
            })
            .then(function() { saveTBtn.disabled = false; });
        };

        // --- Regenerate handler: when settings are dirty, fold them
        // into a PATCH (server rebuilds audio); otherwise hit the
        // dedicated /regenerate endpoint. Either way, in-place refresh
        // via applyServerGroup. No "Save first" toast. ---
        regenBtn.onclick = function() {
          regenBtn.disabled = true;
          var dirty = settingsDirty();
          var p = dirty
            ? api('PATCH', '/captures/api/groups/' + encodeURIComponent(g.id), {
                join_strategy: joinSel.value,
                silence_ms:    parseInt(silSel.value, 10),
              })
            : api('POST', '/captures/api/groups/' + encodeURIComponent(g.id) + '/regenerate');
          p.then(function(j) {
              applyServerGroup(j.group || {});
              toast(dirty ? 'Settings applied, audio regenerated.' : 'Regenerated');
              reloadCounts();
            })
            .catch(function(e) {
              if (e.message !== 'unauthorized') toast(e.message, true);
            })
            .then(function() { regenBtn.disabled = false; });
        };

        // --- Lock toggle ---
        lockBtn.onclick = function() {
          api('PATCH', '/captures/api/groups/' + encodeURIComponent(g.id),
              { is_locked: !(lockBtn.textContent === 'Unlock') })
            .then(function(j) {
              var d = j.group || {};
              // applyServerGroup would clobber unsaved chip/notes/settings
              // edits. When dirty, apply only the lock-related visuals.
              if (!groupState.dirty) {
                applyServerGroup(d);
              } else {
                var isLocked = !!d.is_locked;
                lockBtn.textContent = isLocked ? 'Unlock' : 'Lock';
                dissolveBtn.style.display = isLocked ? 'none' : '';
                // Sync the collapsed-list header so the lock-pill in
                // _renderGroupCard reflects the new state on next render().
                var idx = _allGroups.findIndex(function(x) { return x.id === g.id; });
                if (idx >= 0) _allGroups[idx].is_locked = d.is_locked;
              }
              toast('Updated');
            })
            .catch(function(e) {
              if (e.message !== 'unauthorized') toast(e.message, true);
            });
        };

        // --- Dissolve (full reload — the group disappears from the list). ---
        dissolveBtn.onclick = function() {
          if (!confirm('Dissolve this group? Members return to the flat list; merged WAV is unlinked.'))
            return;
          api('DELETE', '/captures/api/groups/' + encodeURIComponent(g.id))
            .then(function() { toast('Dissolved'); return load(); })
            .catch(function(e) {
              if (e.message !== 'unauthorized') toast(e.message, true);
            });
        };

        // Initial fill.
        applyServerGroup(detail);
        body.dataset.built = '1';
        body.dataset.fetching = '';
        _openGroups[g.id] = { audio: audio };
      })
      .catch(function(e) {
        body.dataset.fetching = '';
        card.classList.remove('open');
        if (e && e.message !== 'unauthorized') toast(e.message, true);
      });
  }

  // Override render() to also draw group cards interleaved by created_ts.
  render = function() {
    var rows = applyFilters(_allCaptures);
    var list = document.getElementById('list');
    var openIds = Object.keys(_openRows);
    // Group cards don't persist open state across render — innerHTML wipe
    // destroys their <audio>, so revoke any tracked blob URL first.
    Object.keys(_openGroups).forEach(function(gid) {
      var st = _openGroups[gid];
      if (st && st.audio && st.audio.src && st.audio.src.indexOf('blob:') === 0) {
        try { URL.revokeObjectURL(st.audio.src); } catch(_) {}
      }
    });
    _openGroups = {};
    list.innerHTML = '';
    // Build a merged timeline: ungrouped captures + group cards (members
    // are nested inside group cards, so we exclude them from the flat list).
    var ungrouped = rows.filter(function(r) { return !r.group_id; });
    // Apply the same status filter to groups. `audio_missing` is a
    // captures-only system status — groups don't have it, so the
    // groups section renders empty when that filter is active.
    var filteredGroups = (_filtStatus === 'all')
      ? _allGroups.slice()
      : _allGroups.filter(function(g) { return g.status === _filtStatus; });
    var combined = ungrouped.map(function(r) {
      return { kind: 'capture', ts: r.created_ts || 0, data: r };
    }).concat(filteredGroups.map(function(g) {
      return { kind: 'group', ts: g.created_ts || 0, data: g };
    }));
    combined.sort(function(a, b) { return b.ts - a.ts; });

    if (combined.length === 0) {
      var empty = document.createElement('div');
      empty.className = 'empty-state';
      empty.innerHTML = _allCaptures.length === 0
        ? '<strong>No captures yet.</strong> Enable <em>CAPTURE_RECORDINGS_ENABLED</em> in /config and send a transcription request.'
        : 'No captures match the current filters.';
      list.appendChild(empty);
      return;
    }
    combined.forEach(function(item) {
      list.appendChild(item.kind === 'capture'
        ? renderCard(item.data)
        : _renderGroupCard(item.data));
    });
    openIds.forEach(function(cid) {
      var card = list.querySelector('.capture-card[data-id="' + cid + '"]');
      if (card) {
        var r = _allCaptures.find(function(x) { return x.id === cid; });
        if (r) toggleExpand(card, r);
      } else {
        var st = _openRows[cid];
        if (st && st.blobUrl) URL.revokeObjectURL(st.blobUrl);
        delete _openRows[cid];
      }
    });
  };

  // -------------------------------------------------------------------
  // Wire up
  // -------------------------------------------------------------------
  (function() {
    var opts = [
      { value: 'all',           label: 'All' },
      { value: 'new',           label: 'New' },
      { value: 'reviewed',      label: 'Reviewed' },
      { value: 'ready',         label: 'Ready' },
      { value: 'dismissed',     label: 'Dismissed' },
      { value: 'audio_missing', label: 'Audio missing' },
    ];
    var grp = buildStatusButtonGroup(opts, _filtStatus, function(v) {
      _filtStatus = v;
      render();
    });
    document.getElementById('filt-status-wrap').appendChild(grp.root);
  })();
  document.getElementById('filt-model').addEventListener('change', render);
  // Debounce search input: render() does a full list rebuild AND each
  // previously-open row re-runs toggleExpand → triggers a fresh
  // /captures/api/{cid} GET. Per-keystroke firing storms the server
  // when the user is typing.
  var _searchTimer = null;
  document.getElementById('filt-search').addEventListener('input', function() {
    if (_searchTimer) clearTimeout(_searchTimer);
    _searchTimer = setTimeout(function() { _searchTimer = null; render(); }, 150);
  });
  document.getElementById('btn-refresh').addEventListener('click', load);
  document.getElementById('btn-export').addEventListener('click', onExport);
  document.getElementById('btn-clear').addEventListener('click', onClearAll);
  document.getElementById('btn-reprocess-all').addEventListener('click', onReprocessAll);
  document.getElementById('ab-merge').addEventListener('click', _openMergeModal);
  document.getElementById('ab-clear').addEventListener('click', _clearSelection);

  // Revoke any open audio blob URLs on tab close — also handled per-row
  // on collapse, but this is the safety net.
  window.addEventListener('beforeunload', function() {
    Object.keys(_openRows).forEach(function(cid) {
      var st = _openRows[cid];
      if (st && st.blobUrl) {
        try { URL.revokeObjectURL(st.blobUrl); } catch(_) {}
      }
    });
    Object.keys(_openGroups).forEach(function(gid) {
      var st = _openGroups[gid];
      if (st && st.audio && st.audio.src && st.audio.src.indexOf('blob:') === 0) {
        try { URL.revokeObjectURL(st.audio.src); } catch(_) {}
      }
    });
  });

  load();
  timeTick();
})();
</script>
</body>
</html>
"""
