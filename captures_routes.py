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
import re
import tarfile
import tempfile
import threading
import time
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse,
)
from pydantic import BaseModel, Field

import api_keys_store
import captures_merge_proposer
import captures_store
import config as cfg
import text_corrections
import web_common
from admin_routes import require_admin_host
from auth import get_current_user, require_admin, require_page

logger = logging.getLogger("whisper-api")

# Router-level dependency: only the IP gate. The page-permission gate
# (`require_page("captures")`) must NOT live at router level because it
# transitively requires a bearer (via get_current_user), and the HTML
# page is fetched by browser navigation which can't pass Authorization
# headers — the login modal runs in the page's own JS. Page-perm gates
# therefore live per-API-route, where fetch() already attaches the
# bearer. Mutation routes additionally `Depends(require_admin)` for
# system-wide writes (clear, reprocess-all, export).
router = APIRouter(
    dependencies=[Depends(require_admin_host)],
)


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


def _audit_cross_user_read(
    user: dict[str, Any], row: dict[str, Any] | None,
    kind: str, row_id: str,
) -> None:
    """Emit an INFO line when a non-admin viewer reads a row owned by a
    different user (scope=all path). Self-reads + admin-host requests
    that already have access don't audit — only the data-leaving-the-
    user-pool case is interesting. Cheap; makes DSGVO Art. 9 data-
    subject access requests answerable from the standard log stream.
    Silent for admin users (they bypass scope and would otherwise
    audit every read on their own dashboard)."""
    if user.get("is_admin"):
        return
    caller_uid = user.get("user_id") or ""
    owner_uid = (row or {}).get("user_id") or ""
    if not owner_uid or owner_uid == caller_uid:
        return
    logger.info(
        "[audit] cross-user-read user=%s(uid=%s) read %s id=%s owner=%s",
        user.get("username") or "?",
        caller_uid[:8] if caller_uid else "?",
        kind,
        (row_id or "?")[:8],
        owner_uid[:8],
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
    # write (another admin in another tab, or a group save touching this
    # member) doesn't get clobbered by the user's save. Omitted → legacy
    # replace.
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
    dependencies=[Depends(require_page("captures"))],
)
async def list_captures_api(
    status_filter: str = Query("all", alias="status"),
    limit: int = Query(200, ge=1, le=1000),
    before_ts: float | None = Query(None),
    user_filter: str | None = Query(None, alias="user_id"),
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Scope-aware list. `scope=own` users see only their own captures;
    `scope=all` users (incl. admins) see every capture and may narrow
    via the admin-only `?user_id=...` query for the per-user dropdown."""
    perms = user["permissions"]
    caller_uid = user.get("user_id") or ""
    effective_user = perms.effective_user_id_for("captures", caller_uid)
    # Admin-only ?user_id= override (non-admin's query is ignored — they
    # are already scoped to their own data by effective_user_id_for).
    if user.get("is_admin") and user_filter:
        effective_user = user_filter
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
    "/captures/api/propose-merges",
    dependencies=[Depends(require_page("captures"))],
)
async def propose_merges_api(
    user_filter: str | None = Query(None, alias="user_id"),
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Ranked auto-merge proposals for the /captures fine-tuning data UI.

    Scope-aware: `scope=own` users see proposals built from their own
    captures only (the `user_id` query is ignored — that's an admin-
    only override). `scope=all` users (incl. admins) see cross-user
    proposals; admins may further narrow via `?user_id=...`. Results
    are cached per scope with a TTL (cfg.CAPTURES_PROPOSER_CACHE_TTL_S),
    invalidated on any capture/group write."""
    perms = user["permissions"]
    caller_uid = str(user.get("user_id") or "")
    sees_all = perms.scope("captures") == "all"
    proposals, cached = captures_merge_proposer.propose_merges(
        # Only scope=all callers can narrow via ?user_id=; the proposer
        # ignores user_id_filter when is_admin=False (caller scoped to
        # caller_user_id partition).
        user_id_filter=user_filter if sees_all else None,
        is_admin=sees_all,
        caller_user_id=caller_uid,
    )
    if proposals:
        all_uids: set[str] = set()
        for p in proposals:
            if p.get("user_id"):
                all_uids.add(p["user_id"])
            for m in p.get("member_previews", []):
                if m.get("user_id"):
                    all_uids.add(m["user_id"])
        usernames = api_keys_store.get_usernames(list(all_uids)) if all_uids else {}
        for p in proposals:
            p["username"] = usernames.get(p.get("user_id"))
            for m in p.get("member_previews", []):
                m["username"] = usernames.get(m.get("user_id"))
    return JSONResponse({
        "proposals": proposals,
        "generated_ts": time.time(),
        "cached": cached,
    })


@router.get(
    "/captures/api/by-request/{request_id}",
    dependencies=[Depends(require_page("captures"))],
)
async def by_request_id_api(
    request_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Cross-link from /reports → captures sharing a request_id. Non-admin
    callers (scope=own) see only rows they own; admin-equivalent (scope=
    all) sees every match. The endpoint backs the reports-page "show
    capture" jump, so the same scope rules that gate /captures itself
    apply here."""
    rows = captures_store.find_by_request_id(request_id)
    perms = user["permissions"]
    if perms.scope("captures") == "own":
        caller_uid = user.get("user_id")
        rows = [r for r in rows if r.get("user_id") == caller_uid]
    usernames = api_keys_store.get_usernames([r.get("user_id") for r in rows])
    for r in rows:
        _apply_trim_to_capture_row(r)
        r["username"] = usernames.get(r.get("user_id"))
    return JSONResponse({"captures": rows})


# Literal-path GET routes (export, groups) MUST be declared BEFORE the
# parameterized /captures/api/{cid} route — FastAPI/Starlette match in
# declaration order, and the `{cid}` placeholder would otherwise swallow
# any literal-named GET like /captures/api/export with cid="export" or
# /captures/api/samples with cid="samples" (which silently 404s the
# group-list fetch and hides newly created groups from the UI).
@router.get(
    "/captures/api/export",
    dependencies=[
        Depends(require_page("captures")),
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
    "/captures/api/samples",
    dependencies=[Depends(require_page("captures"))],
)
async def list_samples_api(
    user_filter: str | None = Query(None, alias="user_id"),
    status_filter: str | None = Query(None, alias="status"),
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """List packed training-sample groups. `scope=own` users see only
    their own groups; `scope=all` users (incl. admins) see every group
    and may narrow via the admin-only `?user_id=...` query. Optional
    `?status=` filter accepts the same enum as PatchSampleIn
    (new/reviewed/ready/dismissed); unknown values fall through to no
    filter, matching list_captures_api's tolerance.

    Declared above `/captures/api/{cid}` because GET with cid="samples"
    would otherwise resolve to the single-capture handler and 404 — the
    UI's `load()` then silently swallows the failure and renders no
    groups, making merged groups invisible after creation."""
    import capture_samples_store
    perms = user["permissions"]
    caller_uid = user.get("user_id") or ""
    scope = perms.effective_user_id_for("captures", caller_uid)
    if user.get("is_admin") and user_filter:
        scope = user_filter
    groups = capture_samples_store.list_samples(
        user_id=scope, status=status_filter,
    )
    usernames = api_keys_store.get_usernames([g.get("user_id") for g in groups])
    for g in groups:
        # Re-derive transcript + corrections per group so the collapsed
        # card preview reflects chip-applied final text (matches the
        # expanded card + export). Members fetched once per group; no
        # merged_words on the list path — that's expand-only.
        members = capture_samples_store.get_members(g["id"])
        _hydrate_members(members)
        g["transcript"] = _build_default_transcript(
            members, g.get("transcript_join_strategy") or "space",
        )
        g["corrections"] = _project_member_corrections(members)
        g["username"] = usernames.get(g.get("user_id"))
    return JSONResponse({"samples": groups})


@router.get(
    "/captures/api/{cid}",
    dependencies=[Depends(require_page("captures"))],
)
async def get_capture_api(
    cid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    row = captures_store.get_capture(cid)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    # Scope guard. 404 (not 403) on cross-user access — a 403 would
    # confirm the row exists (OWASP IDOR cheatsheet).
    user["permissions"].assert_can_read_row(
        row, "captures", user.get("user_id") or "",
    )
    _audit_cross_user_read(user, row, "capture", cid)
    _refresh_final_if_stale(row)
    # Attach BOTH the runtime-`final` token (`word`) and the EXCLUDE-aware
    # training token (`train_word`/`train_removed`) per raw word — the same
    # shape the merge/proposal path produces. The Corrections strip + chips
    # display the training token so they match what the Final result and the
    # export actually emit (CAPTURES_PIPELINE_RULES_EXCLUDE respected); the
    # runtime form stays visible on the "runtime (dictation-map applied)" line.
    row["words"] = _align_member_words(row)
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
    dependencies=[Depends(require_page("captures"))],
)
async def get_audio_api(
    cid: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> FileResponse:
    _check_audio_rate(request.client.host if request.client else "")
    row = captures_store.get_capture(cid)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    user["permissions"].assert_can_read_row(
        row, "captures", user.get("user_id") or "",
    )
    _audit_cross_user_read(user, row, "audio", cid)
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
    dependencies=[Depends(require_page("captures"))],
)
async def patch_capture_api(
    cid: str,
    payload: PatchCaptureIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Edit a single capture (corrections, status, notes). `scope=own`
    users can edit only their own; `scope=all` users (incl. admins)
    can edit any capture. 404 (not 403) on cross-user access."""
    row = captures_store.get_capture(cid)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    user["permissions"].assert_can_read_row(
        row, "captures", user.get("user_id") or "",
    )
    _audit_cross_user_read(user, row, "capture-patch", cid)
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
            # cross-tab admin saves (and the same member edited from its
            # group view).
            current = row.get("corrections") or []
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
    dependencies=[Depends(require_page("captures"))],
)
async def delete_capture_api(
    cid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Delete a single capture. `scope=own` users can delete only their
    own; `scope=all` users (incl. admins) can delete any. Bulk wipe is
    via /clear which stays admin-only."""
    row = captures_store.get_capture(cid)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    user["permissions"].assert_can_read_row(
        row, "captures", user.get("user_id") or "",
    )
    _audit_cross_user_read(user, row, "capture-delete", cid)
    if not captures_store.delete_capture(cid):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    return JSONResponse({"ok": True})


@router.post(
    "/captures/api/clear",
    dependencies=[
        Depends(require_page("captures")),
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
# Per-capture pipeline reprocess (re-run rules on `raw`)
# ---------------------------------------------------------------------

@router.post(
    "/captures/api/{cid}/reprocess",
    dependencies=[Depends(require_page("captures"))],
)
async def reprocess_capture_api(
    cid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Re-run the post-processing pipeline on the stored `raw` text and
    update both `final` and `text_for_training` to reflect the current
    PIPELINE_RULES (and the captures-specific exclude set).

    Use case: after editing PIPELINE_RULES (e.g. adding a typo-fix or
    a new dictation-map entry), a reviewer wants this specific capture
    re-derived without waiting for the bulk reapply job. The bulk job
    /quick-config/reapply-rules also handles this row eventually, but
    the per-row trigger gives immediate feedback in the UI. `scope=own`
    users can reprocess only their own captures.
    """
    row = captures_store.get_capture(cid)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "capture not found")
    user["permissions"].assert_can_read_row(
        row, "captures", user.get("user_id") or "",
    )
    _audit_cross_user_read(user, row, "capture-reprocess", cid)
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
        Depends(require_page("captures")),
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


@router.get(
    "/captures/api/reprocess-all/status",
    dependencies=[Depends(require_page("captures")), Depends(require_admin)],
)
async def reprocess_all_status_api() -> JSONResponse:
    """Live progress of the pipeline-reapply job (for the Advanced menu's
    progress line)."""
    import captures_reapply
    return JSONResponse(captures_reapply.status())


@router.post(
    "/captures/api/reprocess-vad",
    dependencies=[Depends(require_page("captures")), Depends(require_admin)],
)
async def reprocess_vad_api() -> JSONResponse:
    """Trigger the bulk VAD/silence re-merge job: rebuild every sample's
    merged WAV with the CURRENT global silence settings (skips locked
    samples; over-cap samples are flagged stale, never truncated). Idempotent:
    a second call while running returns the current state. Use after editing
    the global Sample-sizing / Silence-trim settings."""
    import captures_vad_reprocess
    return JSONResponse(captures_vad_reprocess.start())


@router.get(
    "/captures/api/reprocess-vad/status",
    dependencies=[Depends(require_page("captures")), Depends(require_admin)],
)
async def reprocess_vad_status_api() -> JSONResponse:
    """Live progress of the VAD/silence re-merge job."""
    import captures_vad_reprocess
    return JSONResponse(captures_vad_reprocess.status())


# ---------------------------------------------------------------------
# Capture groups (≤28 s packed training samples)
# ---------------------------------------------------------------------

# Join strategy + inter-member silence are GLOBAL admin settings now
# (cfg.CAPTURES_SAMPLE_JOIN_STRATEGY / cfg.CAPTURES_VAD_MARGIN_GROUP_INTERNAL_MS),
# not per-request — so the merge/preview/patch payloads no longer carry them.
class CreateSampleIn(BaseModel):
    model_config = {"extra": "forbid"}
    member_ids: list[str] = Field(min_length=1, max_length=30)


class PreviewMergeIn(BaseModel):
    """Preview the merged audio without creating a sample."""
    model_config = {"extra": "forbid"}
    member_ids: list[str] = Field(min_length=1, max_length=30)


class PreviewSaveChipsIn(BaseModel):
    """Save chip corrections from a not-yet-merged proposal. Chips carry
    GLOBAL word indices into the merged karaoke strip; server fans them
    out to per-member captures via _split_corrections_to_members."""
    model_config = {"extra": "forbid"}
    member_ids: list[str] = Field(min_length=1, max_length=30)
    corrections: list[CorrectionIn] = Field(default_factory=list, max_length=200)


class PatchSampleIn(BaseModel):
    model_config = {"extra": "forbid"}
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


def _global_silence_ms() -> int:
    """Inter-member silence, sourced from the global VAD-internal knob
    (was a per-merge `silence_ms` payload field)."""
    import config as cfg
    try:
        return int(getattr(cfg, "CAPTURES_VAD_MARGIN_GROUP_INTERNAL_MS", 300))
    except (TypeError, ValueError):
        return 300


def _global_join_strategy() -> str:
    """Transcript join strategy, sourced from the global setting."""
    import config as cfg
    j = getattr(cfg, "CAPTURES_SAMPLE_JOIN_STRATEGY", "space")
    return j if j in ("space", "period_space") else "space"


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
    `segments` onto the trimmed-audio timeline. Always sets
    `effective_duration_seconds` (= original `duration_seconds` when
    lead/trail are None or 0) so consumers can read one field
    uniformly without branching on trim presence."""
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
    that chip is left alone. Mirrored byte-for-byte by the JS twin
    `_applyChipsToText` so server- and client-derived transcripts agree."""
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


def _validate_merge_payload(
    member_ids: list[str],
    silence_ms: int,
    user: dict[str, Any],
    *,
    enforce_cap: bool = True,
) -> tuple[list[dict[str, Any]], str, list[str], int]:
    """Shared validation for create_sample_api and the preview-audio endpoint.

    Validates: deduped member_ids, every capture exists, none is already in
    a group, all members belong to the same user (and the caller is either
    that user or admin), audio files are present on disk, and — when
    `enforce_cap` — the TRIMMED merged duration (per-member VAD trim + inter-
    segment silence) ≤ 28 s. The cap is measured on trimmed audio because
    that's what the merged WAV actually is (and what the proposer packs to);
    measuring raw would reject groups that comfortably fit after trimming.

    `enforce_cap=False` skips only the cap (used by the merge-estimate
    endpoint, which needs the totals even when they exceed 28 s).

    Returns (captures, owner_user_id, member_paths, total_trimmed_ms) so
    downstream callers don't re-fetch the same rows."""
    if len(member_ids) != len(set(member_ids)):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "duplicate capture in member_ids",
        )

    captures: list[dict[str, Any]] = []
    member_paths: list[str] = []
    user_ids: set[str] = set()
    for mid in member_ids:
        cap = captures_store.get_capture(mid)
        if cap is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"capture {mid} not found",
            )
        if cap.get("sample_id"):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"capture {mid} is already in a sample",
            )
        abs_p = captures_store.abs_audio_path(cap["audio_relpath"])
        if not os.path.exists(abs_p):
            raise HTTPException(
                status.HTTP_410_GONE, f"capture {mid} audio is missing",
            )
        member_paths.append(abs_p)
        user_ids.add(cap.get("user_id") or "")
        captures.append(cap)
    if len(user_ids) != 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "members must all belong to the same user",
        )
    owner_user_id = next(iter(user_ids))
    # Scope guard via the policy object. scope=all (incl. admin) bypasses;
    # scope=own requires the caller to BE the owner. 404 (not 403) matches
    # the captures detail endpoints — don't leak existence.
    user["permissions"].assert_can_read_row(
        {"user_id": owner_user_id}, "captures", user.get("user_id") or "",
    )
    # Audit the cross-user merge surface so a non-admin scope=all caller
    # reading another user's captures via /groups, /preview-audio,
    # /preview-words or /preview-save-chips is recorded — matches the
    # per-row audit calls at every other capture/group endpoint.
    _audit_cross_user_read(
        user, {"user_id": owner_user_id}, "merge",
        ",".join(member_ids[:3]) + ("+" if len(member_ids) > 3 else ""),
    )

    # Cap on TRIMMED audio — what the merged WAV actually is. Reuses the
    # proposer's cached per-capture trim so the batch flow (already warm) pays
    # nothing here; a cold manual merge trims each member once (then cached).
    import captures_merge_proposer
    total_trimmed_ms = sum(
        int(round(captures_merge_proposer.trimmed_duration_s(c) * 1000))
        for c in captures
    )
    # Real merged length under the uniform layout: 2×outer-edge +
    # Σ trimmed bodies + (N-1)×join silence.
    import config as cfg
    n_members = len(member_ids)
    total_gap_ms = int(silence_ms) * max(0, n_members - 1)
    edge_ms = int(getattr(cfg, "CAPTURES_VAD_MARGIN_GROUP_EDGE_MS", 300))
    total_edge_ms = 2 * edge_ms if n_members >= 1 else 0
    cap_ms = int(float(getattr(cfg, "CAPTURES_SAMPLE_MAX_DURATION_S", 29.9)) * 1000)
    total_ms = total_trimmed_ms + total_gap_ms + total_edge_ms
    if enforce_cap and total_ms > cap_ms:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"merged duration would exceed {cap_ms / 1000:.1f} s "
            f"({total_ms / 1000:.2f}s)",
        )
    return captures, owner_user_id, member_paths, total_trimmed_ms


def _build_merged_wav(
    *,
    sid: str,
    member_ids: list[str],
    silence_ms: int,
    member_paths: list[str] | None = None,
) -> tuple[int, dict[str, str], dict[str, Any]]:
    """Resolve member audio paths (or accept pre-resolved ones), run the
    merge, return (duration_ms, member_hash_map, member_trims). When
    `member_paths` is None, looks them up via captures_store + validates each
    file exists. Caller must have validated member_ids belong to the same
    user and total ≤28 s.

    `member_trims` maps member_id → {lead_ms, new_duration_ms, segments} when
    CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS trims each member; _build_merged_words
    uses it to keep per-member karaoke timestamps in sync with the trimmed
    audio. Empty/identity when trimming is disabled or VAD is unavailable."""
    import audio_merge
    import capture_samples_store
    import config as cfg

    if member_paths is None:
        member_paths = []
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

    hashes: dict[str, str] = {}
    for mid, abs_p in zip(member_ids, member_paths):
        hashes[mid] = audio_merge.hash_wav_pcm(abs_p)

    dst_relpath = capture_samples_store._relpath_for(sid)
    dst_abs = capture_samples_store.abs_path_for(dst_relpath)
    try:
        res = audio_merge.merge_wavs(
            member_paths, dst_abs, gap_ms=silence_ms,
            trim=bool(getattr(cfg, "CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS", False)),
            edge_pad_ms=int(getattr(cfg, "CAPTURES_VAD_MARGIN_GROUP_EDGE_MS", 300)),
            max_internal_gap_ms=int(
                getattr(cfg, "CAPTURES_VAD_MARGIN_GROUP_INTERNAL_MS", 300)),
        )
    except audio_merge.WavFormatError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    duration_ms = int(res["duration_ms"])
    # Re-key the order-parallel per-member trims onto member ids so
    # _build_merged_words can look each member up regardless of ordering.
    member_trims = {
        mid: res["members"][i]
        for i, mid in enumerate(member_ids)
        if i < len(res["members"])
    }
    return duration_ms, hashes, member_trims


def _merged_wav_patch(
    duration_ms: int, hashes: dict[str, str],
    member_trims: dict[str, Any],
) -> dict[str, Any]:
    """Build the capture_samples patch fields from _build_merged_wav outputs.

    merged_lead/trail_trim_ms are forced to 0 under per-member trimming (the
    outer-edge offsets are now folded into the per-member segment maps); they
    survive only so legacy groups without member_trims keep rendering."""
    return {
        "merged_duration_ms":   int(duration_ms),
        "member_hashes_json":   json.dumps(hashes, sort_keys=True),
        "merged_lead_trim_ms":  0,
        "merged_trail_trim_ms": 0,
        "member_trims_json":    json.dumps(member_trims, sort_keys=True),
        "is_stale":             0,
    }


def _preview_member_trims(
    member_ids: list[str], member_paths: list[str],
) -> dict[str, Any]:
    """Compute the same per-member trim map merge_wavs would produce, without
    writing a merged WAV. Used by /preview-words so the karaoke overlay lines
    up with the audio /preview-audio streams for the same payload. Returns {}
    when group trimming is disabled (then _build_merged_words uses the legacy
    full-duration timeline)."""
    import config as cfg
    if not getattr(cfg, "CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS", False):
        return {}
    import audio_merge
    import audio_vad_trim
    edge = int(getattr(cfg, "CAPTURES_VAD_MARGIN_GROUP_EDGE_MS", 300))
    max_gap = int(getattr(cfg, "CAPTURES_VAD_MARGIN_GROUP_INTERNAL_MS", 300))
    join_ms = int(_global_silence_ms())
    trims: dict[str, Any] = {}
    # Mirror merge_wavs' uniform layout: leading edge, then bodies joined by
    # `join_ms`, stamping each member's absolute offset so the preview karaoke
    # lines up with the audio /preview-audio streams.
    cursor_ms = edge
    first = True
    for mid, p in zip(member_ids, member_paths):
        try:
            pcm, n = audio_merge.read_pcm(p)
            res = audio_vad_trim.trim_pcm_for_merge(
                pcm, n, edge_pad_ms=edge, max_internal_gap_ms=max_gap,
            )
        except Exception:
            continue
        if not first:
            cursor_ms += join_ms
        first = False
        trims[mid] = {
            "lead_ms": int(res["lead_ms"]),
            "new_duration_ms": int(res["new_duration_ms"]),
            "segments": res["segments"],
            "offset_ms": int(cursor_ms),
        }
        cursor_ms += int(res["new_duration_ms"])
    return trims


@router.post(
    "/captures/api/samples",
    dependencies=[Depends(require_page("captures"))],
)
async def create_sample_api(
    payload: CreateSampleIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Pack 2+ same-user captures into a ≤28 s training sample.

    Server-enforced invariants:
      - all members exist, are not yet in a group, are all owned by the
        same user (and either the caller is that user OR is admin)
      - total audio + gap silence ≤ 28 s
      - members' audio files match (1 ch, 16 bit, 16 kHz)
    """
    import capture_samples_store
    import uuid as _uuid

    member_ids = payload.member_ids
    captures, owner_user_id, member_paths, _total_audio_ms = (
        _validate_merge_payload(member_ids, _global_silence_ms(), user)
    )

    # Build merged WAV — sid generated upfront so the build path is
    # known before the DB insert (mirrors captures_store).
    sid = _uuid.uuid4().hex
    transcript = _build_default_transcript(captures, _global_join_strategy())
    duration_ms, hashes, member_trims = _build_merged_wav(
        sid=sid,
        member_paths=member_paths,
        member_ids=member_ids,
        silence_ms=_global_silence_ms(),
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
        _insert_sample_with_sid(
            sid=sid,
            user_id=owner_user_id,
            member_ids=member_ids,
            transcript=transcript,
            join_strategy=_global_join_strategy(),
            silence_ms=_global_silence_ms(),
            member_hash_map=hashes,
            duration_ms=duration_ms,
            language=group_language,
            member_trims=member_trims,
        )
    except Exception:
        # Insert failed — roll back the WAV we just wrote so the
        # next merge attempt for the same captures starts clean.
        try:
            os.unlink(capture_samples_store.abs_path_for(
                capture_samples_store._relpath_for(sid)))
        except OSError:
            pass
        raise
    return JSONResponse({"sample_id": sid})


@router.post(
    "/captures/api/samples/preview-audio",
    dependencies=[Depends(require_page("captures"))],
)
async def preview_merge_audio_api(
    payload: PreviewMergeIn,
    request: Request,
    background: BackgroundTasks,
    user: dict[str, Any] = Depends(get_current_user),
) -> FileResponse:
    """Build the merged WAV exactly as create_sample_api would, stream it
    back to the caller as audio/wav, and delete the temp file after the
    response completes. Does NOT persist a capture_samples row.

    Used by the /captures Auto-propose merges modal + the manual merge-
    modal to let users preview the merged audio before committing."""
    import audio_merge

    _check_audio_rate(request.client.host if request.client else "")

    _captures, _owner, member_paths, _total_audio_ms = (
        _validate_merge_payload(payload.member_ids, _global_silence_ms(), user)
    )

    # tempfile.NamedTemporaryFile(delete=False) so FileResponse can stream
    # the closed file; background unlink fires after the response finishes.
    import config as cfg
    fd, tmp_path = tempfile.mkstemp(prefix="preview_merge_", suffix=".wav")
    os.close(fd)
    try:
        audio_merge.merge_wavs(
            member_paths, tmp_path, gap_ms=_global_silence_ms(),
            trim=bool(getattr(cfg, "CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS", False)),
            edge_pad_ms=int(getattr(cfg, "CAPTURES_VAD_MARGIN_GROUP_EDGE_MS", 300)),
            max_internal_gap_ms=int(
                getattr(cfg, "CAPTURES_VAD_MARGIN_GROUP_INTERNAL_MS", 300)),
        )
    except audio_merge.WavFormatError as e:
        try: os.unlink(tmp_path)
        except OSError: pass
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except ValueError as e:
        try: os.unlink(tmp_path)
        except OSError: pass
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except Exception:
        try: os.unlink(tmp_path)
        except OSError: pass
        raise

    def _cleanup():
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    background.add_task(_cleanup)
    return FileResponse(
        path=tmp_path,
        media_type="audio/wav",
        filename="preview.wav",
    )


@router.post(
    "/captures/api/samples/preview-words",
    dependencies=[Depends(require_page("captures"))],
)
async def preview_merge_words_api(
    payload: PreviewMergeIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Return the projected merged words + projected corrections + joined
    transcript for a hypothetical merge — timestamps aligned to the audio
    that POST /preview-audio would stream for the same payload. Lets the
    UI overlay karaoke highlighting AND seed the chip-correction box on
    the preview panel without persisting a group row.

    Pure CPU; no rate-limit (bounded by ≤30 members × few hundred words +
    memoized per-word _postprocess_text). Same validation gates as the
    audio endpoint."""
    captures, _owner, member_paths, _total_audio_ms = (
        _validate_merge_payload(payload.member_ids, _global_silence_ms(), user)
    )
    words = _build_merged_words(
        captures, _global_silence_ms(),
        member_trims=_preview_member_trims(payload.member_ids, member_paths),
    )
    return JSONResponse({
        "words": words,
        "corrections": _project_member_corrections(captures),
        "transcript": _build_default_transcript(captures, _global_join_strategy()),
    })


@router.post(
    "/captures/api/samples/merge-estimate",
    dependencies=[Depends(require_page("captures"))],
)
async def merge_estimate_api(
    payload: PreviewMergeIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Return the raw + TRIMMED merged-duration totals for a hypothetical
    merge so the manual-selection meter can show (and gate on) the real
    post-trim length instead of the raw sum. Skips the 28 s cap so the UI can
    display an over-cap value and disable Merge itself. Same ownership gates
    as the other merge endpoints; reuses the proposer's cached per-capture
    trim."""
    import captures_merge_proposer
    captures, _owner, _paths, trimmed_ms = _validate_merge_payload(
        payload.member_ids, _global_silence_ms(), user, enforce_cap=False,
    )
    raw_ms = sum(
        int(round(float(c.get("duration_seconds") or 0.0) * 1000))
        for c in captures
    )
    n = len(payload.member_ids)
    gap_ms = int(_global_silence_ms()) * max(0, n - 1)
    import config as cfg
    edge_ms = int(getattr(cfg, "CAPTURES_VAD_MARGIN_GROUP_EDGE_MS", 300))
    total_edge_ms = 2 * edge_ms if n >= 1 else 0
    cap_ms = int(float(getattr(cfg, "CAPTURES_SAMPLE_MAX_DURATION_S", 29.9)) * 1000)
    trimmed_total = trimmed_ms + gap_ms + total_edge_ms
    return JSONResponse({
        "raw_total_s": (raw_ms + gap_ms + total_edge_ms) / 1000.0,
        "trimmed_total_s": trimmed_total / 1000.0,
        "hard_cap_s": cap_ms / 1000.0,
        "fits": trimmed_total <= cap_ms,
    })


@router.post(
    "/captures/api/samples/preview-save-chips",
    dependencies=[Depends(require_page("captures"))],
)
async def preview_save_chips_api(
    payload: PreviewSaveChipsIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Persist chip corrections from a not-yet-merged proposal. Fans the
    global-indexed chips out to each member capture's local indices via
    _split_corrections_to_members, then REPLACES each member's
    `corrections` field. Same fan-out semantics as the group-level chip
    save (captures_routes.py:1643+ patch_sample_api path).

    Re-fetches every touched member and returns the canonical chips so
    the client can reproject (via _project_member_corrections) to refresh
    its baseline without a full /preview-words round-trip."""
    captures, owner_user_id, _member_paths, _total_audio_ms = (
        _validate_merge_payload(payload.member_ids, _global_silence_ms(), user)
    )
    chips_in = [c.model_dump(exclude_none=True) for c in payload.corrections]
    per_member = _split_corrections_to_members(chips_in, captures)

    saved: dict[str, int] = {}
    members_corrections: dict[str, list[dict[str, Any]]] = {}
    for cap in captures:
        mid = cap["id"]
        member_chips = per_member.get(mid, [])
        updated = captures_store.update_capture(mid, {"corrections": member_chips})
        canonical = (updated or {}).get("corrections") or []
        members_corrections[mid] = canonical
        saved[mid] = len(canonical)

    captures_merge_proposer.invalidate(owner_user_id)
    return JSONResponse({
        "saved": saved,
        "members_corrections": members_corrections,
    })


def _insert_sample_with_sid(
    *,
    sid: str,
    user_id: str,
    member_ids: list[str],
    transcript: str,
    join_strategy: str,
    silence_ms: int,
    member_hash_map: dict[str, str],
    duration_ms: int,
    language: str | None = None,
    member_trims: dict[str, Any] | None = None,
) -> None:
    """Direct insert that honours a pre-allocated sid (needed because the
    audio file is written at the sid path before this call).

    Group chip state lives on the member captures, not on the group
    row — every read re-projects from members — so no chip plumbing
    appears here."""
    import capture_samples_store

    relpath = capture_samples_store._relpath_for(sid)
    now = time.time()
    conn = capture_samples_store._require_conn()
    with capture_samples_store._lock:
        with conn:
            conn.execute(
                "INSERT INTO capture_samples"
                " (id, user_id, created_ts, merged_wav_relpath,"
                "  merged_duration_ms, transcript,"
                "  transcript_join_strategy, member_hashes_json,"
                "  inter_segment_silence_ms, is_stale, is_locked,"
                "  language, merged_lead_trim_ms, merged_trail_trim_ms,"
                "  member_trims_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,0,0,?,0,0,?)",
                (
                    sid, user_id, now, relpath, int(duration_ms),
                    transcript, join_strategy,
                    json.dumps(member_hash_map, sort_keys=True),
                    int(silence_ms),
                    language or None,
                    json.dumps(member_trims or {}, sort_keys=True),
                ),
            )
            for order, mid in enumerate(member_ids):
                conn.execute(
                    "UPDATE captures SET sample_id = ?, sample_order = ?"
                    " WHERE id = ? AND sample_id IS NULL",
                    (sid, order, mid),
                )
    captures_merge_proposer.invalidate(user_id)
    logger.info(
        "[samples] created sid=%s user=%s n=%d dur=%.1fs",
        sid[:8], (user_id or "?")[:8], len(member_ids), duration_ms / 1000.0,
    )


@router.get(
    "/captures/api/samples/{sid}",
    dependencies=[Depends(require_page("captures"))],
)
async def get_sample_api(
    sid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    import capture_samples_store
    g = capture_samples_store.get_sample(sid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sample not found")
    # 404 (not 403) on cross-user — leaking existence violates OWASP IDOR.
    user["permissions"].assert_can_read_row(
        g, "captures", user.get("user_id") or "",
    )
    _audit_cross_user_read(user, g, "sample", sid)
    return JSONResponse({"sample": _enrich_sample(g)})


def _hydrate_members(members: list[dict[str, Any]]) -> None:
    """Populate `words` (decoded) and `model` on each member dict in
    place by fetching the full capture row once. `capture_samples_store.
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
        # A wordless member has no global anchor to project chips onto.
        # Without this skip the clamp below would collapse to `offset`,
        # which is also the first word index of the NEXT member — the
        # round-trip through _split_corrections_to_members would silently
        # re-attribute the chip to that next member.
        if not words:
            continue
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


def _enrich_sample(g: dict[str, Any]) -> dict[str, Any]:
    """Add `members` + `merged_words` to a group dict and re-derive the
    chip-dependent fields (transcript + corrections) from current
    member state.

    Source of truth for chips is each MEMBER's `corrections` list. With
    every read going through this function, member-chip edits on /captures
    (the member's own singleton card, or another admin tab) flow through
    to the group's Corrections section automatically — no in-DB chip
    storage needed at the group level."""
    import capture_samples_store
    members = capture_samples_store.get_members(g["id"])
    _hydrate_members(members)
    usernames = api_keys_store.get_usernames(
        [m.get("user_id") for m in members] + [g.get("user_id")]
    )
    member_trims = g.get("member_trims") or {}
    for m in members:
        _refresh_final_if_stale(m)
        m["username"] = usernames.get(m.get("user_id"))
        # Per-member trimmed duration so the expanded member list shows the
        # length each clip actually contributes to the merged WAV (mirrors the
        # singleton effective_duration_seconds field). Falls back to raw for
        # legacy groups with no stored per-member trim map.
        info = member_trims.get(m.get("id"))
        if info and info.get("new_duration_ms") is not None:
            m["effective_duration_seconds"] = float(info["new_duration_ms"]) / 1000.0
        else:
            m["effective_duration_seconds"] = float(m.get("duration_seconds") or 0.0)
    g["members"] = members
    g["username"] = usernames.get(g.get("user_id"))
    g["transcript"] = _build_default_transcript(
        members, g.get("transcript_join_strategy") or "space",
    )
    g["corrections"] = _project_member_corrections(members)
    g["merged_words"] = _build_merged_words(
        members, int(g["inter_segment_silence_ms"]),
        member_trims=g.get("member_trims") or {},
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


_ALIGN_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def _align_key(s: str) -> str:
    """Normalise a token for LCS comparison: strip surrounding whitespace
    AND leading/trailing non-word punctuation, then casefold. Internal
    punctuation (apostrophes, internal hyphens) is preserved so "don't"
    or "Sciene-fiction" still compare faithfully.

    Stripping edge punctuation is necessary because the inference pipeline
    (callback:map: "Komma" → ",", "Punkt" → ".") attaches the symbol to
    the preceding token when joining; without normalisation, "existiert"
    (raw) and "existiert," (final after Komma→, glue) won't LCS-match,
    falsely flagging the raw word as removed-by-rule in the corrections
    word-strip.

    The visual diff signal (rule changed the word) is still surfaced via
    `item["raw_word"]` when display != raw (see _align_words_to_final
    L1351-1357) — the user sees the dotted-underline + tooltip without
    the misleading strike-through."""
    s = (s or "").strip()
    s = _ALIGN_PUNCT_RE.sub("", s)
    return s.casefold()


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
      - `word`: the final token(s) attributed to this raw word — the
        display text for the karaoke band + the chip's `wrong` reference.
        A rule that EXPANDS one raw word into several tokens (e.g.
        "Nurtax" → "nur tags") yields the joined run ("nur tags") so the
        band reconstructs `final` losslessly instead of dropping the
        extra tokens.
      - `raw_word`: present when display != raw — powers the dotted
        underline + `title="raw: …"` tooltip
      - `removed`: True only when this raw word ends up owning ZERO final
        tokens (a cross-word rule deleted it, or it's the dropped side of
        a contraction). The UI fades + strikes-through these slots; chip
        creation is suppressed.

    Every final token is assigned to exactly one raw word, so
    `" ".join(item["word"] for non-removed items) == final`. Inserted
    tokens (no direct raw correspondent) attach to the raw word that
    expanded into them — sharing that word's audio timestamp — rather
    than being discarded.
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

    # Assign every final token to exactly one raw word so the per-word band
    # reconstructs `final` losslessly — even when a rule expands one raw word
    # into several tokens ("Nurtax" → "nur tags") or contracts several into one.
    # The matched raw words are monotonic anchors that split `fin_tokens` into
    # segments; the inserted (unmatched) tokens in a segment go to the unmatched
    # raw words sharing that segment, apportioned by each raw word's isolated
    # post-processed token count, with any remainder to the last one.
    owners: list[list[int]] = [[] for _ in range(n)]
    exp_count = [len((post_cache.get(w.get("word") or "") or "").split()) for w in src]
    anchors = [(idx, matches[idx]) for idx in range(n) if matches[idx] >= 0]
    seg_bounds = [(-1, -1)] + anchors + [(n, m)]
    for t in range(len(seg_bounds) - 1):
        ri0, fj0 = seg_bounds[t]
        ri1, fj1 = seg_bounds[t + 1]
        if ri0 >= 0:
            owners[ri0].append(fj0)  # the left anchor owns its matched token
        ins = list(range(fj0 + 1, fj1))       # inserted final tokens in this gap
        if not ins:
            continue
        unmatched = list(range(ri0 + 1, ri1))  # raw words sharing this gap
        if unmatched:
            k = 0
            for ri in unmatched:
                take = exp_count[ri]
                while take > 0 and k < len(ins):
                    owners[ri].append(ins[k]); k += 1; take -= 1
            last = unmatched[-1]
            while k < len(ins):  # leftover insertions → last raw word in the gap
                owners[last].append(ins[k]); k += 1
        else:
            # Pure insertion with no raw word in the gap: attach to an adjacent
            # anchor (it shares that word's audio time). Prefer the left anchor.
            target = ri0 if ri0 >= 0 else (ri1 if ri1 < n else None)
            if target is not None:
                owners[target].extend(ins)

    out: list[dict[str, Any]] = []
    for i, w in enumerate(src):
        item = _clone_word(w)
        raw_w = w.get("word") or ""
        toks = owners[i]
        if toks:
            disp = " ".join(fin_tokens[j] for j in toks)
            item["word"] = disp
            if disp.strip() != raw_w.strip():
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


def _remap_time_ms(tm: float, segments: list[list[int]]) -> float:
    """Map an original member-time (ms) onto trimmed member-local time (ms)
    via the per-member kept-speech `segments` list
    ([orig_start_ms, orig_end_ms, new_start_ms], ascending). A time inside a
    kept span maps linearly; a time in a dropped/collapsed silence region
    snaps to the nearest span boundary (words live in speech, so this only
    fires on the rare straddle/rounding case)."""
    if not segments:
        return tm
    if tm <= segments[0][0]:
        return float(segments[0][2])
    for os_ms, oe_ms, ns_ms in segments:
        if tm <= oe_ms:
            if tm >= os_ms:
                return float(ns_ms + (tm - os_ms))
            return float(ns_ms)  # in collapsed gap before this span
    last = segments[-1]
    return float(last[2] + (last[1] - last[0]))  # past the last span


def _emit_member_words(
    merged: list[dict[str, Any]],
    ws: list[dict[str, Any]],
    *,
    i: int,
    member_offset_s: float,
    eff_dur_s: float | None,
    segments: list[list[int]] | None,
) -> None:
    """Append member i's aligned words to `merged`, placed on the merged
    timeline. When `segments` is given, each word's original time is remapped
    through the per-member trim map first (per-member trimming); otherwise the
    legacy flat offset is used."""
    for w in ws:
        start = w.get("start")
        end = w.get("end", start)
        word = w.get("word", "")
        if start is None or end is None:
            continue
        if segments is None:
            s_new = float(start) + member_offset_s
            e_new = float(end) + member_offset_s
        else:
            s_new = _remap_time_ms(float(start) * 1000.0, segments) / 1000.0 \
                + member_offset_s
            e_new = _remap_time_ms(float(end) * 1000.0, segments) / 1000.0 \
                + member_offset_s
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
        # Training-text token for this raw word (CAPTURES_PIPELINE_RULES_EXCLUDE
        # respected). The Corrections strip shows `word` (runtime `final`); the
        # Final-result karaoke uses `train_word` so it matches what the export
        # emits. Absent when training text == final (no excluded rule differs).
        if w.get("train_word") is not None:
            entry["train_word"] = w["train_word"]
            if w.get("train_removed"):
                entry["train_removed"] = True
        merged.append(entry)


def _align_member_words(m: dict[str, Any]) -> list[dict[str, Any]]:
    """Align a member's raw words to its runtime `final` (for the Corrections
    strip), and — when the training text differs (CAPTURES_PIPELINE_RULES_EXCLUDE
    drops some rules) — also align to `text_for_training` and attach the
    per-word `train_word`/`train_removed`. One entry per raw word, so both
    alignments are index-parallel and chip word-indices stay valid."""
    words = m.get("words") or []
    final = m.get("final") or ""
    training = m.get("text_for_training") or final
    ws = _align_words_to_final(words, final, model_name=m.get("model"))
    if training != final:
        wt = _align_words_to_final(words, training, model_name=m.get("model"))
        for i, w in enumerate(ws):
            tw = wt[i] if i < len(wt) else None
            if tw is None:
                continue
            # Preserve the raw word's leading whitespace in front of the
            # training token. `_align_words_to_final` strips the lead off
            # MATCHED tokens, so without this a multi-word Corrections range
            # would join into a run with no inter-word spaces ("134Schrägstrich92")
            # and fail to match the training text. The Corrections strip's
            # `.replace(/^\s+/,' ')` normalizes the display lead, and
            # `_renderGroundSpans` .trim()s, so neither is affected.
            raw_w = (words[i].get("word") if i < len(words) else "") or ""
            lead = raw_w[: len(raw_w) - len(raw_w.lstrip())]
            core = (tw.get("word") or "").lstrip()
            w["train_word"] = (lead + core) if core else ""
            if tw.get("removed"):
                w["train_removed"] = True
    return ws


def _build_merged_words(
    members: list[dict[str, Any]],
    silence_ms: int,
    *,
    member_trims: dict[str, Any] | None = None,
    merged_lead_trim_ms: int = 0,
    merged_duration_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Project each member's per-word timestamps onto the merged-audio
    timeline. start/end are returned in seconds (matches audio.currentTime
    and the single-capture karaoke band's expectation).

    Two timelines, picked by whether `member_trims` is populated:

    - Per-member trimming (new groups): each member was silence-trimmed
      before concatenation, so member i starts at (Σ_{j<i} new_dur_j) +
      i × silence_s, and each word's original time is remapped through that
      member's kept-speech `segments` map. This is what keeps karaoke aligned
      after the dead-air at member joins is removed.

    - Legacy groups (member_trims empty): member i starts at (Σ_{j<i} dur_j) +
      i × silence_s − merged_lead_trim_ms, using full member durations and the
      single merged-WAV outer-edge offset — i.e. the original behaviour, so
      pre-existing groups render exactly as before.

    `get_members` strips heavy fields for the list view, so callers hydrate
    `words` first. Each member's words go through `_align_words_to_final`
    (LCS-align raw→final). Cost is bounded — ≤30 members, ≤a few hundred words
    per ≤28 s group — and only runs on expand."""
    silence_s = max(0, int(silence_ms)) / 1000.0
    eff_dur_s: float | None = None
    if merged_duration_ms is not None:
        eff_dur_s = max(0.0, float(merged_duration_ms) / 1000.0)
    merged: list[dict[str, Any]] = []

    use_per_member = bool(member_trims)
    # New uniform-silence layout stamps each member with an absolute
    # `offset_ms`; legacy groups don't, and fall back to the cum+i*silence
    # formula so their karaoke renders exactly as before (until reprocessed).
    _first = (member_trims or {}).get(members[0]["id"]) if (use_per_member and members) else None
    new_layout = bool(_first and _first.get("offset_ms") is not None)

    if use_per_member and eff_dur_s is None:
        # Preview path: no stored merged duration — derive it.
        n = len(members)
        if new_layout:
            import config as cfg
            edge_ms = float(getattr(cfg, "CAPTURES_VAD_MARGIN_GROUP_EDGE_MS", 300))
            last_end = 0.0
            for m in members:
                info = (member_trims or {}).get(m["id"]) or {}
                last_end = max(last_end, float(info.get("offset_ms") or 0.0)
                               + float(info.get("new_duration_ms") or 0.0))
            eff_dur_s = (last_end + edge_ms) / 1000.0
        else:
            total_new_ms = 0.0
            for m in members:
                info = member_trims.get(m["id"]) if member_trims else None
                if info and info.get("new_duration_ms") is not None:
                    total_new_ms += float(info["new_duration_ms"])
                else:
                    total_new_ms += float(m.get("duration_seconds") or 0.0) * 1000.0
            eff_dur_s = (total_new_ms + max(0, n - 1) * float(silence_ms)) / 1000.0

    if use_per_member:
        cum_ms = 0.0
        for i, m in enumerate(members):
            info = (member_trims or {}).get(m["id"])
            off_ms = None
            if info and info.get("segments"):
                segments = info["segments"]
                new_dur_ms = float(info.get("new_duration_ms") or 0.0)
                off_ms = info.get("offset_ms")
            else:
                # Member not in the trim map (shouldn't happen) → identity.
                dur_ms = float(m.get("duration_seconds") or 0.0) * 1000.0
                segments = [[0, int(dur_ms), 0]]
                new_dur_ms = dur_ms
            if off_ms is not None:
                member_offset_s = float(off_ms) / 1000.0     # new uniform layout
            else:
                member_offset_s = cum_ms / 1000.0 + i * silence_s  # legacy
            ws = _align_member_words(m)
            _emit_member_words(
                merged, ws, i=i, member_offset_s=member_offset_s,
                eff_dur_s=eff_dur_s, segments=segments,
            )
            cum_ms += new_dur_ms
        return merged

    # Legacy path (no per-member trims): original flat-offset behaviour.
    lead_s = max(0, int(merged_lead_trim_ms or 0)) / 1000.0
    cum = 0.0
    for i, m in enumerate(members):
        member_offset_s = cum + i * silence_s - lead_s
        ws = _align_member_words(m)
        _emit_member_words(
            merged, ws, i=i, member_offset_s=member_offset_s,
            eff_dur_s=eff_dur_s, segments=None,
        )
        cum += float(m.get("duration_seconds") or 0.0)
    return merged


@router.patch(
    "/captures/api/samples/{sid}",
    dependencies=[Depends(require_page("captures"))],
)
async def patch_sample_api(
    sid: str,
    payload: PatchSampleIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    import capture_samples_store
    g = capture_samples_store.get_sample(sid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sample not found")
    user["permissions"].assert_can_read_row(
        g, "captures", user.get("user_id") or "",
    )
    _audit_cross_user_read(user, g, "sample-patch", sid)
    if g["is_locked"] and not user.get("is_admin"):
        raise HTTPException(status.HTTP_409_CONFLICT, "sample is locked")

    patch: dict[str, Any] = {}
    # Lazily-fetched hydrated members; up to three branches below need
    # this list and used to issue independent get_members calls each.
    _members_cache: list[dict[str, Any]] | None = None
    def _members() -> list[dict[str, Any]]:
        nonlocal _members_cache
        if _members_cache is None:
            _members_cache = capture_samples_store.get_members(sid)
            _hydrate_members(_members_cache)
        return _members_cache

    # Join strategy + inter-member silence are GLOBAL settings now; they're no
    # longer patchable per-sample. Changing them and rebuilding audio for
    # existing samples is done via the bulk VAD reprocess action / regenerate.
    if payload.is_locked is not None:
        patch["is_locked"] = 1 if payload.is_locked else 0
    if payload.status is not None:
        patch["status"] = payload.status
    if payload.admin_notes is not None:
        patch["admin_notes"] = payload.admin_notes
    if payload.corrections is not None:
        # Fan group-level chip edits DOWN to the owning members. Group
        # corrections are derived from members on every read (see
        # `_enrich_sample`); writing to a group-level chip column would
        # be discarded by the next GET.
        #
        # When the client also sends `baseline_corrections` (a snapshot
        # of what it loaded), apply a three-way merge against the
        # current member-projected chips BEFORE the split — that way a
        # concurrent cross-tab admin save (or a member edited from its
        # singleton /captures card) isn't clobbered by the user's payload,
        # and the user's deltas (additions, removals, edits) apply on top.
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

    # Re-derive `transcript` from current members + chips ONLY when the
    # corrections changed. The common status/admin_notes/is_locked auto-save
    # click would otherwise trigger a get_members + transcript rebuild + DB
    # write on every click. Join strategy uses the sample's stored value
    # (the global only re-applies on regenerate / bulk reprocess).
    if payload.corrections is not None:
        join_for_derive = g["transcript_join_strategy"] or "space"
        patch["transcript"] = _build_default_transcript(_members(), join_for_derive)

    updated = capture_samples_store.update_sample(sid, patch)
    return JSONResponse({"sample": _enrich_sample(updated)})


@router.post(
    "/captures/api/samples/{sid}/regenerate",
    dependencies=[Depends(require_page("captures"))],
)
async def regenerate_sample_api(
    sid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Rebuild the merged WAV from current member content using the CURRENT
    global silence setting (so regenerate is how an existing sample adopts a
    changed global), refresh hashes, clear `is_stale`. Transcript is preserved
    (admin's edits stay)."""
    import capture_samples_store
    g = capture_samples_store.get_sample(sid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sample not found")
    user["permissions"].assert_can_read_row(
        g, "captures", user.get("user_id") or "",
    )
    _audit_cross_user_read(user, g, "sample-regenerate", sid)
    members = capture_samples_store.get_members(sid)
    silence_ms = _global_silence_ms()
    with _get_rebuild_lock(sid):
        duration_ms, hashes, member_trims = _build_merged_wav(
            sid=sid,
            member_ids=[m["id"] for m in members],
            silence_ms=silence_ms,
        )
        regen_patch = _merged_wav_patch(duration_ms, hashes, member_trims)
        regen_patch["inter_segment_silence_ms"] = silence_ms
        updated = capture_samples_store.update_sample(sid, regen_patch)
    return JSONResponse({"sample": _enrich_sample(updated)})


@router.delete(
    "/captures/api/samples/{sid}",
    dependencies=[Depends(require_page("captures"))],
)
async def dissolve_sample_api(
    sid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    import capture_samples_store
    g = capture_samples_store.get_sample(sid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sample not found")
    user["permissions"].assert_can_read_row(
        g, "captures", user.get("user_id") or "",
    )
    _audit_cross_user_read(user, g, "sample-delete", sid)
    if g["is_locked"] and not user.get("is_admin"):
        raise HTTPException(status.HTTP_409_CONFLICT, "sample is locked")
    capture_samples_store.dissolve_sample(sid)
    return JSONResponse({"ok": True})


# Per-sid lock so a burst of audio requests for the same missing WAV
# at startup doesn't trigger N concurrent merges. The route runs the
# sync merge inside Starlette's threadpool, so a plain threading.Lock
# is the right primitive (asyncio.Lock would only help if the merge
# itself awaited).
_rebuild_locks: dict[str, threading.Lock] = {}
_rebuild_locks_guard = threading.Lock()


def _get_rebuild_lock(sid: str) -> threading.Lock:
    with _rebuild_locks_guard:
        lock = _rebuild_locks.get(sid)
        if lock is None:
            lock = threading.Lock()
            _rebuild_locks[sid] = lock
        return lock


def _ensure_sample_wav(g: dict[str, Any]) -> str:
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
    import capture_samples_store
    try:
        abs_p = capture_samples_store.abs_path_for(g["merged_wav_relpath"])
    except ValueError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "merged audio missing")
    if os.path.exists(abs_p):
        return abs_p

    members = capture_samples_store.get_members(g["id"])
    if not members:
        raise HTTPException(
            status.HTTP_410_GONE,
            "members deleted — sample is unrecoverable",
        )
    for m in members:
        cap = captures_store.get_capture(m["id"])
        if cap is None:
            raise HTTPException(
                status.HTTP_410_GONE,
                f"member {m['id'][:8]} row deleted — sample is unrecoverable",
            )
        member_abs = captures_store.abs_audio_path(cap["audio_relpath"])
        if not os.path.exists(member_abs):
            raise HTTPException(
                status.HTTP_410_GONE,
                f"member {m['id'][:8]} audio is gone — sample is unrecoverable",
            )

    member_ids = [m["id"] for m in members]
    lock = _get_rebuild_lock(g["id"])
    with lock:
        if os.path.exists(abs_p):
            return abs_p
        logger.warning(
            "[samples] sid=%s auto-rebuilding missing WAV from %d members",
            g["id"][:8], len(member_ids),
        )
        duration_ms, hashes, member_trims = _build_merged_wav(
            sid=g["id"],
            member_ids=member_ids,
            silence_ms=int(g["inter_segment_silence_ms"]),
        )
        capture_samples_store.update_sample(
            g["id"], _merged_wav_patch(duration_ms, hashes, member_trims),
        )
    return abs_p


@router.get(
    "/captures/api/samples/{sid}/audio",
    dependencies=[Depends(require_page("captures"))],
)
async def get_sample_audio_api(
    sid: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
):
    """Stream the merged WAV, self-healing if it's missing on disk
    but reconstructable from member captures."""
    _check_audio_rate(request.client.host if request.client else "")
    import capture_samples_store
    g = capture_samples_store.get_sample(sid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sample not found")
    user["permissions"].assert_can_read_row(
        g, "captures", user.get("user_id") or "",
    )
    _audit_cross_user_read(user, g, "sample-audio", sid)
    abs_p = _ensure_sample_wav(g)
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
      - `is_locked=true` (admin lock; mirrors reapply skip) → skipped
      - `status=audio_missing` (file is gone) → skipped
      - missing WAV on disk → both manifest entry AND tar entry skipped
        (prevents the manifest from referencing files that don't exist
        in the tarball)
    `only_status` (default "ready" from the route) further narrows.
    `audio_missing` rows leak only when `only_status='all'`, and even
    then the hard filter above drops them.
    """
    import capture_samples_store

    buf = io.BytesIO()
    tar = tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6)
    manifest_lines: list[bytes] = []

    # 1. Capture groups (the packed-for-fine-tune training samples).
    user_filter_scope: str | None = None  # admin-only path; no per-user scope
    for g in capture_samples_store.list_samples(user_id=user_filter_scope):
        # Status gate — groups have the same status field as captures.
        # The caller has already mapped "all" → None upstream, so a
        # truthy only_status here is a concrete status to match.
        if only_status:
            if (g.get("status") or "new") != only_status:
                continue
        # Hard filters (apply even on `only_status=all`). Stale = audio/
        # text drift; locked = admin lock (same skip as captures_reapply,
        # so this stays consistent with what the trainer last saw).
        if g.get("is_stale") or g.get("is_locked"):
            continue
        # Always rebuild the transcript at export time from members +
        # chips, so the exported text reflects current corrections even
        # if the stored snapshot is stale. Source from the training-form
        # column so reviewers see — and the trainer learns from — the
        # same text.
        sid = g["id"]
        members = capture_samples_store.get_members(sid)
        text = _build_default_transcript(
            members, g.get("transcript_join_strategy") or "space",
        ).strip()
        if not text:
            continue
        # Audio existence gate: skip the manifest entry entirely if the
        # WAV isn't on disk, to avoid manifest pointing at missing files.
        try:
            abs_p = capture_samples_store.abs_path_for(g["merged_wav_relpath"])
        except ValueError:
            continue
        if not os.path.isfile(abs_p):
            continue

        audio_name = f"audio/{sid}.wav"
        # Group `model` and `request_id` are intentionally empty — a
        # group has multiple members each with their own model id. Per-
        # member audit is reachable via the group's GET /members endpoint.
        manifest_lines.append(json.dumps(_build_manifest_row(
            audio_filepath=audio_name,
            text=text,
            duration=float(g.get("merged_duration_ms") or 0) / 1000.0,
            language=g.get("language") or "",
            source="sample",
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

    # 2. Ungrouped captures (no sample_id).
    for row in captures_store.iter_captures_for_export(status=only_status):
        if row.get("sample_id"):
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
# via per-span correction chips. Status flow: new → reviewed → ready
# (export-eligible) | dismissed (omitted).
#
# IMPORTANT (CLAUDE memory note): never place a `{{...}}` placeholder
# inside a /* */, //, or <!-- --> comment — render_page() does a literal
# string replace and the substitution corrupts the surrounding context.

_CAPTURES_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{HEADER_TITLE}}</title>
{{PAGE_META}}
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
  /* The page toolbar (status/model/search filters, capture-state badge +
     actions) now lives in the sticky header subbar — styled by NAV_CSS,
     consistent with every other page. The proposer-action status label is
     the only piece that keeps a page-local rule. */
  .cc-actions .cc-status-label {
    font-size: var(--fs-sm); color: var(--help);
    display: inline-flex; align-items: center; gap: 0.35rem;
  }

  /* Merge-proposer entry points (Batch review / Auto-propose) live in the
     header's row-3 action bar. They keep a blue accent so they read as the
     prominent fast-path workflow, distinct from the grey utility buttons
     beside them. A single class selector (.proposer-action) already outranks
     the centralized `header button` rule on specificity, so these win the
     cascade wherever they sit. */
  .proposer-action {
    background: rgba(88, 166, 255, 0.12); color: var(--accent, #58a6ff);
    border: 1px solid rgba(88, 166, 255, 0.5); border-radius: 4px;
    padding: 0.3rem 0.8rem; font-size: var(--fs-md); font-weight: 600;
    font-family: var(--font-sans); line-height: 1.3; cursor: pointer;
    white-space: nowrap; flex-shrink: 0; transition: background 0.12s;
  }
  .proposer-action:hover:not(:disabled) { background: rgba(88, 166, 255, 0.22); }
  .proposer-action:active:not(:disabled) { transform: translateY(1px); }
  .proposer-action:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Row-3 search sits in the MIDDLE of the action bar, growing to fill the
     gap between the left merge buttons and the right utility cluster (so its
     long placeholder never truncates — the reason it moved off row 2). The
     left cluster shrinks to content width so the search, not the cluster,
     absorbs the free space. input[type="text"] in the selector beats the
     equal-specificity NAV_CSS rule so min-width can drop below its 10rem. */
  header .subbar-actions .subbar-left { flex: 0 0 auto; }
  header .subbar-actions .subbar-search { flex: 1 1 auto; min-width: 0;
    display: inline-flex; align-items: center; gap: 0.35rem; }
  /* The magnifier replaces the "search" label text to save horizontal space
     so the action row stays on one line longer; the input keeps an aria-label
     for assistive tech. Sized in em so it tracks the scale picker. */
  header .subbar-search .search-ico { flex: 0 0 auto; width: 0.95em; height: 0.95em;
    stroke: var(--help); fill: none; stroke-width: 2; }
  /* Search shrinks down to a small floor (input[type="text"] in the selector
     beats the equal-specificity NAV_CSS min-width:10rem) so the buttons keep
     their one-line layout as the viewport narrows. */
  header .subbar-actions .subbar-search input[type="text"] {
    flex: 1 1 auto; min-width: 2rem; max-width: none; }
  /* Safety net for narrow windows: once the two button clusters can no longer
     share a line even with search at its floor, shrinking just orphans the
     utility cluster under a stretched search. Below the breakpoint we instead
     stack cleanly — both clusters on line 1 (merge left, utility pushed
     right), search dropped to its own full-width line (`order:1` +
     `flex-basis:100%`). Root font is --fs-base (15px), so 1rem = 15px.
     Measured one-line cutoff with these labels: ~960px holds, ~940px orphans,
     so we stack at/below 64rem (960px) — just above the orphan onset, no gap. */
  @container hdr (max-width: 64rem) {
    header .subbar-actions .subbar-left { flex: 1 1 auto; }
    header .subbar-actions .subbar-search { order: 1; flex-basis: 100%; }
  }
  /* Counts get their own line below the status/model/capture-state row: the
     live string ("667 new · 0 reviewed · 5 ready · 0 dismissed") is long and
     would otherwise push the capture-state badge to wrap on its own. */
  header .subbar #counts { flex-basis: 100%; }

  /* Advanced ▾ dropdown (bulk reprocess + destructive actions). */
  .adv-wrap { position: relative; display: inline-block; }
  .adv-menu {
    position: absolute; right: 0; top: calc(100% + 0.25rem); z-index: 50;
    min-width: 16rem; padding: 0.35rem;
    display: flex; flex-direction: column; gap: 0.25rem;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 0.375rem; box-shadow: 0 0.5rem 1.25rem rgba(0,0,0,0.45);
  }
  .adv-menu[hidden] { display: none; }
  .adv-menu button { width: 100%; text-align: left; white-space: nowrap; }
  .adv-sep { height: 1px; background: var(--border); margin: 0.15rem 0; }
  .adv-progress {
    font-size: var(--fs-xs); color: var(--help); font-family: var(--font-mono);
    padding: 0.2rem 0.3rem; white-space: normal;
  }
  .adv-progress[hidden] { display: none; }

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
  /* Small phones: tighten the inter-item gap so the timestamp, pills,
     duration and expand arrow keep wrapping cleanly instead of colliding. */
  @media (max-width: 30em) {
    .cc-head { gap: 0.35rem 0.6rem; }
    .cc-head .spacer { flex-basis: 100%; height: 0; }
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
  /* Keyboard-nav cursor — dashed accent outline on the cursored word.
     The cursor persists even when focus leaves the strip (so the user
     knows where ←/→ would resume). The accent tint behind the cursor
     only appears when the strip itself is focused — that's the visual
     cue that keystrokes will land here. */
  .word-strip .word.cursor {
    outline: 1px dashed var(--accent, #58a6ff);
    outline-offset: -1px;
    border-radius: 2px;
  }
  .word-strip { outline: none; }
  .word-strip:focus { outline: none; }
  .word-strip.has-focus .word.cursor {
    background: rgba(88, 166, 255, 0.12);
  }
  .word-strip-hint {
    font-size: var(--fs-xs); color: var(--help);
    font-family: var(--font-mono);
    margin-top: 0.25rem; padding: 0 0.2rem;
    display: none;
  }
  .word-strip-hint.show { display: block; }
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
    /* Don't let double-click word-selection extend from the value onto the
       "raw"/"post-processing"/"runtime" label. */
    -webkit-user-select: none; user-select: none;
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
  /* Final-result karaoke: words render inline (natural text flow); the active
     word lights up in sync with the Corrections strip as audio plays. */
  .cc-ground .word { display: inline; border-radius: 3px; }
  .cc-ground .word.active {
    background: var(--active-word-bg); color: var(--active-word-color);
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
    border-radius: 6px; padding: 1.25rem;
    min-width: min(22rem, 92vw); max-width: min(32rem, 95vw);
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

  /* Auto-propose merges modal */
  #propose-modal {
    position: fixed; inset: 0; background: rgba(0,0,0,0.6);
    z-index: 1000; align-items: center; justify-content: center;
    display: none;
  }
  #propose-modal.show { display: flex; }
  #propose-modal .box {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.25rem; width: 52rem; max-width: 95vw;
    max-height: 88vh; overflow: auto;
  }
  #propose-modal h3 { margin: 0; color: var(--bold); }
  #propose-modal .dim { color: var(--help); font-size: var(--fs-sm); margin: 0.25rem 0 0.5rem 0; }
  #propose-modal .propose-top {
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 0.25rem;
  }
  #propose-modal .propose-top h3 { flex: 1; }
  #propose-modal .actions {
    display: flex; gap: 0.5rem; align-items: center;
    margin-top: 0.75rem;
  }
  #propose-modal .actions .spacer { flex: 1; }
  #propose-modal .empty {
    color: var(--dim); font-size: var(--fs-sm); padding: 1rem 0;
  }
  #propose-list .proposal {
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.6rem 0.75rem; margin-bottom: 0.5rem;
    background: var(--input-bg);
  }
  /* Inner-content styles for .proposal are selector-unscoped so they
     apply both inside #propose-list AND inside #propose-batch's
     batch-card. Only the OUTER container (border / padding / bg) stays
     scoped to #propose-list since batch-card owns its own outer frame. */
  .proposal .row1 {
    display: flex; gap: 0.6rem; align-items: center;
    font-size: var(--fs-sm); margin-bottom: 0.4rem;
    flex-wrap: wrap;
  }
  .proposal .score {
    display: inline-block; min-width: 4rem; text-align: center;
    padding: 0.1rem 0.5rem; border-radius: 999px;
    font-family: var(--font-mono); font-weight: 700;
    font-size: var(--fs-xs); letter-spacing: 0.02em;
  }
  .proposal .score.tier-good { background: #1b3a1b; color: #6acc6a; border: 1px solid #2e5d2e; }
  .proposal .score.tier-ok   { background: #3a321b; color: #d4a14c; border: 1px solid #5d4e2e; }
  .proposal .score.tier-low  { background: #3a1f1f; color: #d46c6c; border: 1px solid #5d2e2e; }
  .proposal .lang-pill,
  .proposal .speaker-pill {
    display: inline-block; padding: 0.05rem 0.4rem; border-radius: 4px;
    background: var(--border); color: var(--help);
    font-size: var(--fs-xs); font-family: var(--font-mono);
  }
  .proposal .speaker-pill {
    background: #1b2a3a; color: #8aaad0; border: 1px solid #2e4566;
  }
  .proposal .reason {
    color: var(--dim); font-size: var(--fs-sm); margin: 0.3rem 0;
  }
  .proposal .members {
    margin: 0.4rem 0;
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.4rem 0.55rem;
    max-height: 8rem; overflow: auto;
    font-size: var(--fs-xs); color: var(--dim);
  }
  .proposal .members .m {
    display: flex; gap: 0.5rem; padding: 0.15rem 0;
    align-items: baseline; flex-wrap: wrap; row-gap: 0.1rem;
  }
  .proposal .members .m .ts,
  .proposal .members .m .dur,
  .proposal .members .m .m-spk {
    color: var(--help); font-family: var(--font-mono); flex-shrink: 0;
    white-space: nowrap;
  }
  .proposal .members .m .m-spk { color: #8aaad0; }
  .proposal .members .m .m-text {
    flex: 1 1 auto; min-width: 0;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    color: var(--fg); font-family: var(--font-sans);
  }
  /* Phone: drop the transcript onto its own full-width line under the
     timestamp/speaker meta and let it wrap in full instead of truncating. */
  @media (max-width: 40em) {
    .proposal .members .m .m-text {
      flex-basis: 100%; white-space: normal;
      overflow: visible; text-overflow: clip;
    }
  }
  .proposal .meter-bar {
    display: inline-block; width: 7rem; height: 0.4rem;
    background: var(--border); border-radius: 2px; overflow: hidden;
    vertical-align: middle;
  }
  .proposal .meter-bar .fill {
    display: block; height: 100%; background: var(--accent, #58a6ff);
  }

  /* Shared merge-preview play button (used in propose-modal and merge-modal) */
  .merge-preview-btn {
    display: inline-flex; align-items: center; justify-content: center;
    gap: 0.3rem; line-height: 1;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 0.15rem 0.5rem; cursor: pointer;
    font-size: var(--fs-sm); font-family: var(--font-mono);
  }
  .merge-preview-btn:hover:not(:disabled) {
    background: #21262d; color: var(--bold);
  }
  .merge-preview-btn:disabled {
    opacity: 0.4; cursor: not-allowed;
  }
  .merge-preview-btn .dur {
    color: var(--help); font-size: var(--fs-xs);
  }
  .merge-preview-panel {
    display: flex; flex-direction: column; gap: 0.3rem;
    margin: 0.4rem 0 0.2rem; width: 100%;
  }
  /* Compact audio player — reused for the merge preview AND for the
     single-capture / group-expand audio rows on /captures. */
  .compact-player {
    display: flex; align-items: center; gap: 0.4rem;
    margin: 0.4rem 0;
  }
  .compact-player .audio-hidden {
    display: none;
  }
  .compact-player-btn {
    display: inline-flex; align-items: center; justify-content: center;
    line-height: 1;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 0.15rem 0.5rem; cursor: pointer;
    font-size: var(--fs-sm); font-family: var(--font-mono);
    min-width: 1.8rem; text-align: center;
  }
  .compact-player-btn:hover:not(:disabled) {
    background: #21262d; color: var(--bold);
  }
  .compact-player-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  /* Inline SVG glyphs (play/pause/skip-to-start/undo) — block so flex
     centering applies cleanly; sized in em so they ride the --fs-* scale. */
  .compact-player-btn svg, .merge-preview-btn svg { display: block; }
  /* Slider matches the page aesthetic: thin rectangular track in the
     same color as .meter-bar, progress fill in accent, a small vertical
     playhead-style thumb (no circles — the UI is all rectangles). */
  .compact-player-scrub {
    flex: 1; margin: 0; padding: 0; cursor: pointer;
    -webkit-appearance: none; -moz-appearance: none; appearance: none;
    background: transparent; height: 0.9rem;
  }
  .compact-player-scrub:focus { outline: none; }
  .compact-player-scrub::-webkit-slider-runnable-track {
    height: 0.25rem; background: var(--border); border-radius: 1px;
  }
  .compact-player-scrub::-moz-range-track {
    height: 0.25rem; background: var(--border); border-radius: 1px; border: none;
  }
  .compact-player-scrub::-moz-range-progress {
    height: 0.25rem; background: var(--accent, #58a6ff); border-radius: 1px;
  }
  .compact-player-scrub::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 0.25rem; height: 0.8rem; border-radius: 1px;
    background: var(--accent, #58a6ff); border: none;
    margin-top: -0.275rem; cursor: pointer;
  }
  .compact-player-scrub::-moz-range-thumb {
    width: 0.25rem; height: 0.8rem; border-radius: 1px;
    background: var(--accent, #58a6ff); border: none; cursor: pointer;
  }
  .compact-player-scrub:disabled { cursor: not-allowed; opacity: 0.4; }
  .compact-player-scrub:disabled::-webkit-slider-thumb { background: var(--help); }
  .compact-player-scrub:disabled::-moz-range-thumb { background: var(--help); }
  .compact-player-time {
    font-family: var(--font-mono); font-size: var(--fs-xs);
    color: var(--help); min-width: 2.2rem; text-align: right;
  }
  .compact-player-time-sep {
    color: var(--help); font-size: var(--fs-xs);
  }
  .merge-preview-panel .word-strip {
    background: var(--input-bg); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.3rem 0.5rem;
    font-size: var(--fs-sm);
    max-height: 8rem; overflow: auto;
  }
  .merge-preview-panel .merge-preview-cc {
    /* Tighter margins than the default .cc-section since we're nested
       inside an already-padded proposal panel. */
    margin: 0.5rem 0 0; padding: 0.4rem 0.6rem;
  }
  .merge-preview-panel .merge-preview-cc h3 {
    font-size: var(--fs-sm); margin: 0 0 0.2rem;
  }
  .merge-preview-panel .merge-preview-cc .help {
    font-size: var(--fs-xs); margin-bottom: 0.3rem;
  }
  .merge-preview-panel .merge-preview-cc .cc-ground {
    max-height: 6rem; overflow: auto;
  }

  /* Batch / Tinder-style review mode */
  #propose-batch {
    margin: 0.5rem 0;
  }
  #propose-batch .batch-banner {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.4rem 0.6rem; margin-bottom: 0.5rem;
    background: var(--input-bg); border: 1px solid var(--border);
    border-radius: 4px; font-size: var(--fs-sm); color: var(--help);
  }
  #propose-batch .batch-banner .spacer { flex: 1; }
  #propose-batch .batch-banner .count { font-family: var(--font-mono); color: var(--bold); }
  #propose-batch .batch-card {
    border: 1px solid var(--border); border-radius: 6px;
    padding: 0.75rem 0.9rem; background: var(--input-bg);
    transition: transform 0.25s ease-out, opacity 0.25s ease-out;
    will-change: transform, opacity;
    /* Inner proposal renders identically to a list-mode card. */
  }
  #propose-batch .batch-card.swiping {
    transition: none;
  }
  #propose-batch .batch-card.gone-right {
    transform: translateX(120%) rotate(8deg); opacity: 0;
  }
  #propose-batch .batch-card.gone-left {
    transform: translateX(-120%) rotate(-8deg); opacity: 0;
  }
  #propose-batch .batch-card .proposal {
    /* Reuse the proposal-card layout from list mode but hide its inline
       Accept button — batch mode uses the big action row instead. */
    border: none; padding: 0; margin: 0; background: transparent;
  }
  #propose-batch .batch-card .proposal .row1 .primary {
    display: none;
  }
  #propose-batch .batch-actions {
    display: flex; gap: 0.5rem; align-items: center;
    margin-top: 0.6rem;
  }
  #propose-batch .batch-actions button {
    flex: 1; padding: 0.5rem 0.75rem;
    font-size: var(--fs-md);
    border-radius: 4px; cursor: pointer;
    border: 1px solid var(--border); background: var(--panel); color: var(--fg);
  }
  #propose-batch .batch-actions .dismiss {
    color: var(--red); border-color: #5a2424;
  }
  #propose-batch .batch-actions .accept {
    color: var(--green); border-color: #2e5d2e; font-weight: 700;
  }
  /* Secondary controls: small, edge/interior, icon-centered. Replay is
     neutral; Revert is amber (echoes Tinder's gold Rewind) and dims when
     there's nothing to undo. */
  #propose-batch .batch-actions .replay,
  #propose-batch .batch-actions .revert {
    flex: 0 0 auto; min-width: 3rem;
    display: inline-flex; align-items: center; justify-content: center;
    line-height: 1;
  }
  #propose-batch .batch-actions .replay svg,
  #propose-batch .batch-actions .revert svg { display: block; }
  #propose-batch .batch-actions .revert {
    color: var(--yellow); border-color: #5a4a1a;
  }
  #propose-batch .batch-actions .revert:disabled {
    color: var(--dim); border-color: var(--border); opacity: 0.5;
    cursor: not-allowed;
  }
  #propose-batch .batch-actions button:hover:not(:disabled) {
    background: #21262d;
  }
  #propose-batch .batch-hint {
    font-size: var(--fs-xs); color: var(--help); text-align: center;
    margin-top: 0.4rem; font-family: var(--font-mono);
  }
  #propose-batch .batch-done {
    text-align: center; padding: 1.5rem 1rem;
    border: 1px solid var(--border); border-radius: 6px;
    background: var(--input-bg);
  }
  #propose-batch .batch-done h4 {
    margin: 0 0 0.5rem; color: var(--bold);
  }
  #propose-batch .batch-done .summary {
    color: var(--help); font-size: var(--fs-sm); margin-bottom: 1rem;
  }
  #propose-batch .batch-done .actions {
    display: flex; gap: 0.5rem; justify-content: center;
  }

  {{NAV_CSS}}
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <span class="title">{{HEADER_BRAND}}</span>
    <span class="brand-sep" aria-hidden="true"></span>
    {{NAV}}
    <span class="spacer"></span>
    <span class="hdr-right">{{SEV_PILLS}}{{SCALE_PICKER}}{{RELOAD}}{{LOGOUT}}</span>
  </div>
  <!-- Row 2 — quick filters + indicators only (status switch, model, counts,
       capture-state). Search lives on row 3 so its long placeholder has room
       and this row never truncates. -->
  <div class="subbar">
    <span class="subbar-title">Captures</span>
    <div class="subbar-left">
      <span class="filt-label">status <span id="filt-status-wrap"></span></span>
      <label>model
        <select id="filt-model">
          <option value="all">all</option>
        </select>
      </label>
      <span id="capture-state" class="capture-state off">capture OFF</span>
    </div>
    <span class="counts" id="counts"></span>
  </div>
  <!-- Row 3 — action bar, three zones: merge-proposer entry points left (blue
       accent, the prominent fast-path workflow), search filling the middle,
       utility buttons right. -->
  <div class="subbar subbar-actions">
    <div class="subbar-left">
      <button id="btn-batch" class="proposer-action" title="Step through proposals one at a time with keyboard / swipe shortcuts (Ctrl+← dismiss / Ctrl+→ accept / Space pause)">✨ Batch propose merges</button>
      <button id="btn-propose" class="proposer-action" title="Suggest ranked merges into ~26 s training samples; review one at a time in a list">⚡ Propose merges</button>
    </div>
    <label class="subbar-search" title="search">
      <svg class="search-ico" viewBox="0 0 24 24" aria-hidden="true" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg>
      <input id="filt-search" type="text" aria-label="search" placeholder="text in raw / final / corrected">
    </label>
    <div class="subbar-right">
      <button id="btn-refresh">Refresh</button>
      <button id="btn-export" title="Download ready captures as a tar.gz (manifest.jsonl + audio/)">Export ready</button>
      <div class="adv-wrap">
        <button id="btn-advanced" aria-haspopup="true" aria-expanded="false"
          title="Bulk reprocessing &amp; destructive actions">Advanced ▾</button>
        <div id="adv-menu" class="adv-menu" hidden role="menu">
          <button id="btn-reprocess-all" role="menuitem" title="Re-run PIPELINE_RULES on every capture's raw text (text only). Use after editing rules.">Reprocess all · Pipeline rules</button>
          <button id="btn-reprocess-vad" role="menuitem" title="Rebuild every sample's audio with the current global silence settings. Use after editing Sample sizing / Silence trim.">Reprocess all · VAD silence</button>
          <div id="adv-progress" class="adv-progress" hidden></div>
          <div class="adv-sep"></div>
          <button id="btn-clear" class="danger" role="menuitem" title="Permanently delete every capture">Clear all</button>
        </div>
      </div>
    </div>
  </div>
</header>

<main>
  <div id="action-bar">
    <span class="summary"><strong id="ab-count">0</strong> selected</span>
    <span class="meter" id="ab-meter">Σ 0.00 s / 28.00 s</span>
    <span class="summary" id="ab-warn"></span>
    <span class="spacer"></span>
    <button id="ab-merge" class="primary" disabled>Merge into sample</button>
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
      <span class="dim small">Join style &amp; inter-member silence use the
        global settings (Settings → Capture &amp; fine-tuning → Sample sizing).</span>
      <span class="summary" id="merge-summary"></span>
      <span id="merge-preview-slot"></span>
    </div>
    <div id="merge-preview-panel-host"></div>
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

<div id="propose-modal">
  <div class="box">
    <div class="propose-top">
      <h3>Proposed merges</h3>
      <button id="propose-refresh-top">Refresh</button>
      <button id="propose-close-top">Close</button>
    </div>
    <p class="dim">Ranked by total duration (~26 s target), session
       homogeneity, and member count. Near-duplicate clips are excluded
       within each proposal. Click <em>Accept</em> to merge directly — the
       popup stays open for the next one. Join style &amp; inter-member
       silence use the global settings (Settings → Sample sizing).</p>
    <div id="propose-list"></div>
    <div id="propose-batch" hidden></div>
    <div class="actions">
      <button id="propose-refresh">Refresh</button>
      <span class="spacer"></span>
      <button id="propose-close">Close</button>
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

<div id="dialog-modal" class="modal">
  <div class="box">
    <h3 id="dialog-title"></h3>
    <p id="dialog-body" style="white-space: pre-line;"></p>
    <div class="actions">
      <button id="dialog-cancel">Cancel</button>
      <button id="dialog-ok" class="primary">Confirm</button>
    </div>
  </div>
</div>

<div id="token-modal">
  <div class="box">
    <h3>API key</h3>
    <p>Paste your <code>wk_…</code> API key. You'll stay signed in on this
    browser until you sign out.</p>
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
  // Sign-in (HttpOnly session cookie; mirrors /reports)
  // -------------------------------------------------------------------
  // Exchange a pasted key for the session cookie. Returns true on success.
  async function doLogin(key) {
    try {
      var r = await fetch('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: key }),
      });
      if (!r.ok) return false;
      try { window.dispatchEvent(new Event('whisper:auth-changed')); } catch(_) {}
      return true;
    } catch (_) { return false; }
  }
  function showTokenModal(onSaved) {
    var m = document.getElementById('token-modal');
    var inp = document.getElementById('token-input');
    inp.value = '';
    m.classList.add('show');
    setTimeout(function() { inp.focus(); inp.select(); }, 50);
    function close() { m.classList.remove('show'); }
    document.getElementById('token-cancel').onclick = close;
    document.getElementById('token-save').onclick = async function() {
      if (await doLogin(inp.value.trim())) {
        close();
        if (onSaved) onSaved();
      } else {
        toast('that key was rejected', true);
      }
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

  // Themed yes/no confirm dialog (replaces the browser-native confirm()).
  // Returns a Promise<bool>. opts: {title, body, confirmLabel, danger}.
  // Enter confirms, Esc / Cancel / backdrop-click reject.
  function _confirm(opts) {
    opts = opts || {};
    return new Promise(function(resolve) {
      var m = document.getElementById('dialog-modal');
      document.getElementById('dialog-title').textContent = opts.title || 'Confirm';
      document.getElementById('dialog-body').textContent = opts.body || '';
      var ok = document.getElementById('dialog-ok');
      var cancel = document.getElementById('dialog-cancel');
      ok.textContent = opts.confirmLabel || 'Confirm';
      ok.className = opts.danger ? 'danger' : 'primary';
      function close(result) {
        m.classList.remove('show');
        ok.onclick = null; cancel.onclick = null; m.onclick = null;
        document.removeEventListener('keydown', onKey);
        resolve(result);
      }
      function onKey(e) {
        if (e.key === 'Escape') { e.preventDefault(); close(false); }
        else if (e.key === 'Enter') { e.preventDefault(); close(true); }
      }
      ok.onclick = function() { close(true); };
      cancel.onclick = function() { close(false); };
      m.onclick = function(e) { if (e.target === m) close(false); };  // backdrop
      document.addEventListener('keydown', onKey);
      m.classList.add('show');
      setTimeout(function() { try { ok.focus(); } catch (_) {} }, 50);
    });
  }

  // -------------------------------------------------------------------
  // API
  // -------------------------------------------------------------------
  async function api(method, url, body) {
    // Session cookie sent automatically; mutations carry the CSRF token.
    var headers = { 'Content-Type': 'application/json' };
    if (method !== 'GET' && method !== 'HEAD') {
      headers['X-CSRF-Token'] = window._csrfToken ? window._csrfToken() : '';
    }
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
    // Renamed-but-kept-for-compat: every 403 on this page means the
    // caller is a non-admin (the API gate is require_page("captures")
    // — a 403 means "valid bearer, no scope on /captures"). Render the
    // shared no-access landing slugged with the current page.
    try {
      var r = await fetch('/auth/whoami');
      if (r.ok) {
        var j = await r.json();
        // Cache whoami so _renderNoAccessLanding can list reachable pages.
        try { window.__whoami = j; } catch(_) {}
        if (j && j.is_admin === false) {
          _renderNoAccessLanding({ page: 'captures' });
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
  var _allSamples = [];
  var _counts = {};
  var _openRows = {};   // cid -> { audio, blobUrl, wordEls, words, finalText, dirty, corrections, ... }
  var _openSamples = {}; // sid -> { audio } — for blob-URL cleanup on render() / beforeunload
  var _selection = new Set();   // capture ids currently selected for merge
  var _lastSelectId = null;     // anchor for shift-range select

  // -------------------------------------------------------------------
  // Selection helpers
  // -------------------------------------------------------------------
  function _handleSelectionClick(row, shift) {
    var visibleIds = applyFilters(_allCaptures)
      .filter(function(r) { return !r.sample_id; })
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
    // Instant raw estimate (prefixed ~), refined to the trimmed total by a
    // debounced merge-estimate fetch below. The 28 s cap is on TRIMMED audio
    // now, so the raw sum must not hard-gate Merge — a selection that trims to
    // ≤28 s is valid even if its raw sum is larger.
    meter.textContent = 'Σ ~' + totalWithGaps.toFixed(2) + ' s / 28.00 s';
    meter.classList.remove('amber', 'red');

    // Warn on cross-speaker mixes — server enforces, UI nudges.
    var userIds = new Set(rows.map(function(r) { return r.user_id || ''; }));
    var warn = document.getElementById('ab-warn');
    var mixedUsers = userIds.size > 1;
    var hasInSample = rows.some(function(r) { return r.sample_id; });
    if (mixedUsers) {
      warn.textContent = '⚠ multiple speakers — merging not allowed';
      warn.style.color = 'var(--red)';
    } else if (hasInSample) {
      warn.textContent = '⚠ selection includes captures already in a sample';
      warn.style.color = 'var(--red)';
    } else {
      warn.textContent = '';
    }

    var baseOk = n >= 2 && !mixedUsers && !hasInSample;
    var mergeBtn = document.getElementById('ab-merge');
    mergeBtn.disabled = !baseOk;
    if (baseOk) {
      _fetchTrimEstimate(rows.map(function(r) { return r.id; }), gap_ms,
                         meter, mergeBtn);
    }
  }

  // Debounced trimmed-duration estimate for the manual-selection meter. Mirrors
  // batch mode (server computes the trimmed total via the same cached helper);
  // here it's fetched lazily for the current selection. A token guards against
  // out-of-order responses when the selection changes mid-flight.
  var _meterEstimateTimer = null;
  var _meterEstimateToken = 0;
  function _fetchTrimEstimate(ids, gap_ms, meter, mergeBtn) {
    if (_meterEstimateTimer) clearTimeout(_meterEstimateTimer);
    var token = ++_meterEstimateToken;
    _meterEstimateTimer = setTimeout(function() {
      api('POST', '/captures/api/samples/merge-estimate',
          { member_ids: ids })
        .then(function(j) {
          if (token !== _meterEstimateToken) return;  // superseded
          var t = j.trimmed_total_s || 0;
          meter.textContent = 'Σ ' + t.toFixed(2) + ' s / 28.00 s';
          meter.classList.remove('amber', 'red');
          if (!j.fits) meter.classList.add('red');
          else if (t > 24) meter.classList.add('amber');
          mergeBtn.disabled = !j.fits;
        })
        .catch(function() {
          // Keep the raw estimate; the server still validates trimmed on merge.
        });
    }, 300);
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
    // German-aware, case-insensitive ordering (model IDs can be mixed-case).
    Object.keys(seen).sort(new Intl.Collator('de', { sensitivity: 'base', numeric: true }).compare).forEach(function(m) {
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
      (r.sample_id
        ? '<span class="pill group-pill" title="member of sample ' + escapeHtml(r.sample_id.slice(0,8)) + '">in sample</span>'
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
    audio.preload = 'metadata';
    body.appendChild(_attachCompactPlayer(audio));
    state.audio = audio;

    // Authenticated audio fetch → blob URL (session cookie auto-sent). The
    // server always serves RIFF/WAVE 16 kHz mono (every capture is
    // transcoded on write), so browser decode is reliable cross-platform.
    fetch('/captures/api/' + encodeURIComponent(r.id) + '/audio').then(function(resp) {
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
      + '<div class="help">Click a word, or use ← / → and type, to mark '
      + 'and correct it; shift-click extends a range; Enter accepts, Del removes.</div>';
    var strip = document.createElement('div');
    strip.className = 'word-strip';
    state.words.forEach(function(w, i) {
      var sp = document.createElement('span');
      sp.className = 'word';
      var dw = _dispWord(w);
      sp.textContent = (dw || '').replace(/^\s+/, ' ');
      if (_dispRemoved(w)) {
        sp.classList.add('rule-removed');
        sp.title = 'removed by pipeline rule';
      } else if (w.raw_word && (w.raw_word || '').trim() !== (dw || '').trim()) {
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
    state.stripEl = strip;
    if (typeof state.cursorIdx !== 'number') state.cursorIdx = -1;
    _bindStripKeyboard(state);

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
      _setActiveWord(state, idx);
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
      v.className = 'val' + (value ? ' ws-region' : '');
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

    // (The manual per-capture silence-trim button was retired — trimming is
    // now applied uniformly when a capture is merged into a sample. A single
    // long capture can be turned into a trimmed sample via "Merge into sample".)

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

  // Display/correction token for a word in the editable Corrections strip.
  // Prefer the EXCLUDE-aware training token (`train_word`) so the strip, the
  // chip `wrong` text, the Final result preview and the server export all
  // agree on what CAPTURES_PIPELINE_RULES_EXCLUDE leaves in the text; fall
  // back to the runtime `final` token when no excluded rule changed this word.
  function _dispWord(w) {
    return (w && w.train_word !== undefined) ? (w.train_word || '')
                                             : ((w && w.word) || '');
  }
  function _dispRemoved(w) {
    return (w && w.train_word !== undefined) ? !!w.train_removed
                                             : !!(w && w.removed);
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
      if (w) parts.push(_dispWord(w));
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
    for (var i = a; i <= b; i++) parts.push(_dispWord(state.words[i]));
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
    // Mirror the keyboard cursor on every click so a mouse user can
    // grab the keyboard mid-flow and resume navigation from where they
    // clicked. _redrawCursor is defined further down (function decl
    // hoisted inside the IIFE) — call only if the strip is keyboard-
    // wired (older code paths leave state.stripEl unset).
    if (state.stripEl && typeof _redrawCursor === 'function') {
      state.cursorIdx = idx;
      _redrawCursor(state);
    }
    if (clicked && _dispRemoved(clicked)) {
      // The token is absent from the EXCLUDE-aware training/export text — a
      // chip would have no anchor in the exported text. Click still seeks
      // audio. (A token merged away only in the runtime `final` but present
      // in the training text is NOT removed here, so it stays correctable.)
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
      wrong: _dispWord(state.words[idx]).replace(/^\s+/, ''),
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
          // Empty input + Enter = remove the chip (per the keyboard
          // spec — "Removing all letters from a edited word + enter
          // removes correction").
          if (!(inp.value || '').trim()) {
            removeChip(state, i);
            applyCorrectionsToGround(state);
          } else {
            applyCorrectionsToGround(state);
            inp.blur();
          }
          // Return focus to the word-strip so navigation resumes
          // immediately. No-op if state.stripEl wasn't wired up (e.g.,
          // older code paths that don't use the keyboard binder).
          if (state.stripEl) {
            try { state.stripEl.focus({ preventScroll: true }); } catch (_) {}
          }
        } else if (ev.key === 'Escape') {
          ev.preventDefault();
          inp.blur();
          if (state.stripEl) {
            try { state.stripEl.focus({ preventScroll: true }); } catch (_) {}
          }
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

  // Render the Final-result preview as per-word spans (not plain text) so it
  // gets the same word-level karaoke highlight as the Corrections strip. Base
  // tokens are state.words (the merged/aligned words); index-based chip
  // corrections collapse their covered range into a single span carrying the
  // chip's `correct` text, and the wordToGround map (merged-word index → its
  // span) lets _setActiveWord light the right span while audio plays. Only
  // used by surfaces that opt in via state.karaokeGround (group + proposal
  // preview); the single-capture card keeps its plain-text ground.
  function _renderGroundSpans(state) {
    var area = state.gtArea;
    if (!area) return;
    var words = state.words || [];
    // First chip covering each word index (chips are word-indexed; idx_end
    // optional). idx-less chips can't be placed positionally — they're shown
    // in the Corrections list and skipped here (rare; word-click chips have idx).
    var cover = new Array(words.length);
    (state.corrections || []).forEach(function(c) {
      if (typeof c.idx !== 'number' || !(c.correct || '').length) return;
      var end = (typeof c.idx_end === 'number') ? c.idx_end : c.idx;
      for (var j = c.idx; j <= end && j < words.length; j++) {
        if (cover[j] == null) cover[j] = c;
      }
    });
    area.innerHTML = '';
    state.wordToGround = new Array(words.length);
    var first = true;
    var i = 0;
    while (i < words.length) {
      var chip = cover[i];
      if (chip) {
        var end = (typeof chip.idx_end === 'number') ? chip.idx_end : chip.idx;
        if (end >= words.length) end = words.length - 1;
        var sp = document.createElement('span');
        sp.className = 'word';
        sp.textContent = (first ? '' : ' ') + chip.correct;
        area.appendChild(sp);
        for (var k = i; k <= end; k++) state.wordToGround[k] = sp;
        first = false;
        i = end + 1;
        continue;
      }
      var w = words[i];
      // Prefer the training-text token (CAPTURES_PIPELINE_RULES_EXCLUDE
      // respected) so the Final result matches the export; fall back to the
      // runtime `word` when no distinct training token was attached.
      var hasTrain = (w && w.train_word !== undefined);
      var removed = hasTrain ? !!w.train_removed : !!(w && w.removed);
      if (removed) { state.wordToGround[i] = null; i++; continue; }
      var txt = (hasTrain ? (w.train_word || '') : ((w && w.word) || '')).trim();
      if (!txt) { state.wordToGround[i] = null; i++; continue; }
      var sp2 = document.createElement('span');
      sp2.className = 'word';
      sp2.textContent = (first ? '' : ' ') + txt;
      area.appendChild(sp2);
      state.wordToGround[i] = sp2;
      first = false;
      i++;
    }
  }

  function applyCorrectionsToGround(state) {
    // Karaoke-enabled surfaces (group + proposal preview) render the ground as
    // per-word spans; the single-capture card keeps the original plain-text
    // substitution preview.
    if (state.karaokeGround) { _renderGroundSpans(state); return; }
    // Substitute each chip's wrong text with its correct text in
    // state.finalText. Walk corrections in `idx` order so multi-word spans get
    // replaced as one unit. If a chip's `wrong` isn't found verbatim, leave it.
    //
    // The GT element is a read-only preview (textContent, not value) — markDirty
    // stays off here because the user changes GT only indirectly via a chip.
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
    // Optional auto-save hook — proposal panels use this to debounce a
    // PATCH to the affected member captures whenever the chip set
    // changes. The capture/group flows leave this unset and rely on
    // their explicit Save button instead.
    if (typeof state.onMarkDirty === 'function') {
      try { state.onMarkDirty(); } catch (_) {}
    }
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
        // writes (another admin in another tab, or a group save touching
        // this member) so those changes survive.
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
    if (!(await _confirm({
        title: 'Delete capture?',
        body: 'Delete this capture and its audio file? This is irreversible.',
        confirmLabel: 'Delete', danger: true })))
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

  // Re-run the pipeline on the stored `raw` so the training-form text
  // reflects current PIPELINE_RULES edits. Updates the preview text in
  // place; the word-strip (post-edit tints, rule-removed markers, raw-
  // word tooltips) only refreshes on the next expand of this row.
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
        Object.keys(_openSamples).forEach(function(sid) {
          var st = _openSamples[sid];
          if (st && st.audio && st.audio.src && st.audio.src.indexOf('blob:') === 0) {
            try { URL.revokeObjectURL(st.audio.src); } catch(_) {}
          }
        });
        _openSamples = {};
        await load();
      } catch (e) {
        if (e.message !== 'unauthorized') toast('Failed: ' + e.message, true);
      }
    };
  }

  // -------------------------------------------------------------------
  // Export
  // -------------------------------------------------------------------
  // --- Advanced menu (bulk reprocess + destructive actions) ---
  function _closeAdvMenu() {
    var m = document.getElementById('adv-menu');
    var b = document.getElementById('btn-advanced');
    if (m) m.hidden = true;
    if (b) b.setAttribute('aria-expanded', 'false');
  }
  function _toggleAdvMenu() {
    var m = document.getElementById('adv-menu');
    var b = document.getElementById('btn-advanced');
    if (!m) return;
    var open = m.hidden;
    m.hidden = !open;
    if (b) b.setAttribute('aria-expanded', open ? 'true' : 'false');
  }

  // Shared progress poller for the two background jobs. Polls the job's
  // status endpoint into the Advanced menu's progress line until it finishes.
  var _jobPollTimer = null;
  function _pollJob(statusUrl, render) {
    if (_jobPollTimer) { clearTimeout(_jobPollTimer); _jobPollTimer = null; }
    var prog = document.getElementById('adv-progress');
    function tick() {
      api('GET', statusUrl).then(function(s) {
        if (prog) { prog.hidden = false; prog.textContent = render(s); }
        if (s && s.status === 'running') {
          _jobPollTimer = setTimeout(tick, 1000);
        } else {
          _jobPollTimer = null;
          if (s && s.status === 'error') toast('Reprocess failed: ' + (s.error || ''), true);
          else toast('Reprocess done.');
          load();  // refresh the list (durations / stale pills may have changed)
        }
      }).catch(function() { _jobPollTimer = null; });
    }
    tick();
  }

  // Bulk-reprocess (text): re-run PIPELINE_RULES on every capture's `raw`,
  // updating final + training text (and unlocked sample transcripts). Audio
  // is NOT rebuilt. Idempotent background job; progress shown in the menu.
  async function onReprocessAll() {
    _closeAdvMenu();
    if (!(await _confirm({
        title: 'Reprocess all · Pipeline rules',
        body: 'Re-run PIPELINE_RULES on every capture? Updates final + '
            + 'training text in place (no audio rebuild).\n\n'
            + 'This runs in the background.',
        confirmLabel: 'Reprocess' })))
      return;
    try {
      await api('POST', '/captures/api/reprocess-all', {});
      _pollJob('/captures/api/reprocess-all/status', function(s) {
        return 'Pipeline rules: ' + (s.processed || 0) + '/' + (s.total || 0)
          + ' · ' + (s.captures_updated || 0) + ' updated'
          + (s.status !== 'running' ? ' — done' : '…');
      });
    } catch (e) {
      if (e.message !== 'unauthorized') toast('Reprocess-all failed: ' + e.message, true);
    }
  }

  // Bulk-reprocess (audio): rebuild every sample's merged WAV with the current
  // global silence settings. Skips locked samples; over-cap → flagged stale.
  async function onReprocessVad() {
    _closeAdvMenu();
    if (!(await _confirm({
        title: 'Reprocess all · VAD silence',
        body: 'Rebuild every sample’s audio with the current global silence '
            + 'settings?\n\nLocked samples are skipped; any that no longer fit '
            + 'the cap are flagged stale (not truncated). Runs in the background.',
        confirmLabel: 'Rebuild audio' })))
      return;
    try {
      await api('POST', '/captures/api/reprocess-vad', {});
      _pollJob('/captures/api/reprocess-vad/status', function(s) {
        return 'VAD silence: ' + (s.processed || 0) + '/' + (s.total || 0)
          + ' · ' + (s.rebuilt || 0) + ' rebuilt'
          + (s.stale ? ', ' + s.stale + ' stale' : '')
          + (s.skipped ? ', ' + s.skipped + ' skipped' : '')
          + (s.status !== 'running' ? ' — done' : '…');
      });
    } catch (e) {
      if (e.message !== 'unauthorized') toast('Reprocess-VAD failed: ' + e.message, true);
    }
  }

  function onExport() {
    fetch('/captures/api/export?only_status=ready&include_audio=1').then(function(resp) {
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
        var jg = await api('GET', '/captures/api/samples');
        _allSamples = jg.samples || [];
      } catch (_) { _allSamples = []; }
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
    // Join style + inter-member silence are global settings now; the local
    // estimate uses a nominal 300 ms gap (the server is authoritative for the
    // real merge + the refined merge-estimate total).
    var GAP_EST_MS = 300;
    var ta = document.getElementById('merge-transcript');
    // Initial static transcript (space-joined); the karaoke preview replaces
    // it with the server-derived (global-join) text on play.
    ta.textContent = _buildDefaultTranscript(rows, 'space');
    // Live summary — instant raw estimate (~), refined to the trimmed total
    // (the real merged length) via the same merge-estimate endpoint the
    // action-bar meter uses.
    var _summaryToken = 0;
    function refreshSummary() {
      var n = rows.length;
      var totalAudio = rows.reduce(function(s, r) { return s + (r.duration_seconds || 0); }, 0);
      var total = totalAudio + (GAP_EST_MS / 1000) * Math.max(0, n - 1);
      var el = document.getElementById('merge-summary');
      el.textContent = n + ' segments · Σ ~' + total.toFixed(2) + ' s';
      var token = ++_summaryToken;
      api('POST', '/captures/api/samples/merge-estimate',
          { member_ids: rows.map(function(r) { return r.id; }) })
        .then(function(j) {
          if (token !== _summaryToken) return;
          var cap = (j.hard_cap_s || 0).toFixed(2);
          el.textContent = n + ' segments · Σ '
            + (j.trimmed_total_s || 0).toFixed(2) + ' s / ' + cap + ' s';
        })
        .catch(function() {});
    }
    refreshSummary();
    // Rebuild the preview controls per open so they capture the current rows.
    var slot = document.getElementById('merge-preview-slot');
    slot.innerHTML = '';
    var panelHost = document.getElementById('merge-preview-panel-host');
    if (panelHost) panelHost.innerHTML = '';
    var previewCtl = _makeMergePreviewBtn(
      function() { return rows.map(function(r) { return r.id; }); },
      function() { return GAP_EST_MS; },
      function() {
        var totalAudio = rows.reduce(function(s, r) { return s + (r.duration_seconds || 0); }, 0);
        var gap = GAP_EST_MS / 1000;
        return totalAudio + gap * Math.max(0, rows.length - 1);
      },
      // Render the karaoke Final result into the modal's prominent
      // #merge-transcript box (no duplicate in-panel Final result), so it
      // highlights word-by-word on playback like the batch view.
      { groundEl: document.getElementById('merge-transcript') }
    );
    slot.appendChild(previewCtl.toggleBtn);
    if (panelHost) panelHost.appendChild(previewCtl.panel);
    document.getElementById('merge-cancel').onclick = function() {
      _stopAnyPreview();
      modal.classList.remove('show');
    };
    document.getElementById('merge-commit').onclick = function() {
      _stopAnyPreview();
      // Server derives the transcript from members + chips on create
      // (mirrors the locked preview).
      var payload = {
        member_ids: rows.map(function(r) { return r.id; }),
      };
      api('POST', '/captures/api/samples', payload)
        .then(function() {
          modal.classList.remove('show');
          toast('Sample created');
          return load();
        })
        .catch(function(e) {
          if (e && e.message !== 'unauthorized') toast(e.message, true);
        });
    };
    modal.classList.add('show');
  }

  // -------------------------------------------------------------------
  // Merge preview audio + karaoke — shared component used by both the
  // propose-modal proposal cards and the manual merge-modal.
  //
  // Architecture (matches the proven a669f05 play/pause pattern):
  //   - ONE module-global detached <audio> (`_previewAudio`) — Firefox
  //     hates per-card attached <audio controls> elements and aborts
  //     the media load on ancestor display:none, surfacing as
  //     "NS_ERROR_DOM_MEDIA_ABORT_ERR". A single shared detached
  //     element bypasses that whole class of failures.
  //   - Per card: a toggle button + a compact panel containing a small
  //     custom scrubber, a "0:00 / 0:00" time label, and the karaoke
  //     word-strip. The shared audio is rebound to whichever card is
  //     currently active.
  //   - Each Activation rebinds the shared audio: revoke old blob,
  //     fetch audio + words in parallel, set new src, swap karaoke
  //     binding to this card's word-strip, play.
  // -------------------------------------------------------------------
  var _previewAudio = null;        // singleton detached <audio>; lazy-init
  var _previewBlobUrl = null;      // current blob URL (revoked on stop)
  var _previewActivePanel = null;  // currently-active panel (or null)
  var _previewBound = null;        // teardown fn for active karaoke + scrubber wiring

  function _getPreviewAudio() {
    if (_previewAudio) return _previewAudio;
    _previewAudio = document.createElement('audio');
    // Intentionally NOT appended to DOM, NOT controls=true — the prior
    // attached + controls implementation broke Firefox's media fetch.
    _previewAudio.preload = 'auto';
    return _previewAudio;
  }

  function _stopAnyPreview() {
    if (_previewBound) {
      try { _previewBound(); } catch (_) {}
      _previewBound = null;
    }
    if (_previewActivePanel) {
      _previewActivePanel.hidden = true;
      var t = _previewActivePanel._toggleBtn;
      if (t) { t._setIcon('play'); t.disabled = false; }
      _previewActivePanel = null;
    }
    if (_previewAudio) {
      try { _previewAudio.pause(); } catch (_) {}
      try { _previewAudio.removeAttribute('src'); _previewAudio.load(); } catch (_) {}
    }
    if (_previewBlobUrl) {
      try { URL.revokeObjectURL(_previewBlobUrl); } catch (_) {}
      _previewBlobUrl = null;
    }
  }

  // onClick (optional): function(idx, shiftKey) called when a word span
  // is clicked. Used by proposal panels to wire onWordClick(state, …)
  // for chip creation; merge-modal (read-only word-strip) passes nothing.
  function _renderWordStrip(stripEl, words, onClick) {
    stripEl.innerHTML = '';
    var els = [];
    words.forEach(function(w, i) {
      var sp = document.createElement('span');
      sp.className = 'word';
      var dw = _dispWord(w);
      sp.textContent = (dw || '').replace(/^\s+/, ' ');
      if (_dispRemoved(w)) {
        sp.classList.add('rule-removed');
        sp.title = 'removed by pipeline rule';
      } else if (w.raw_word && (w.raw_word || '').trim() !== (dw || '').trim()) {
        sp.classList.add('post-edited');
        sp.title = 'raw: ' + w.raw_word;
      }
      if (onClick) {
        sp.addEventListener('click', function(e) {
          onClick(i, !!e.shiftKey);
        });
      }
      stripEl.appendChild(sp);
      els.push(sp);
    });
    return els;
  }

  function _fmtTime(s) {
    if (!isFinite(s) || s < 0) s = 0;
    var m = Math.floor(s / 60);
    var sec = Math.floor(s % 60);
    return m + ':' + (sec < 10 ? '0' : '') + sec;
  }

  // Karaoke highlight shared by every surface (single-capture, group expand,
  // proposal preview): toggles `.active` on both the Corrections word-strip
  // span AND the matching Final-result span (state.wordToGround), so the
  // current word lights up in both places as audio plays. `idx` is an index
  // into state.words; -1 clears.
  function _setActiveWord(state, idx) {
    if (state._activeWordIdx === idx) return;
    var prev = state._activeWordIdx;
    if (prev != null && prev >= 0) {
      if (state.wordEls && state.wordEls[prev]) state.wordEls[prev].classList.remove('active');
      if (state.wordToGround && state.wordToGround[prev]) {
        state.wordToGround[prev].classList.remove('active');
      }
    }
    state._activeWordIdx = idx;
    if (idx >= 0) {
      if (state.wordEls && state.wordEls[idx]) {
        state.wordEls[idx].classList.add('active');
        try { state.wordEls[idx].scrollIntoView({ block: 'nearest', inline: 'nearest' }); }
        catch (_) {}
      }
      if (state.wordToGround && state.wordToGround[idx]) {
        state.wordToGround[idx].classList.add('active');
      }
    }
  }

  // -------------------------------------------------------------------
  // Word-strip keyboard navigation — cursor + arrow keys + type-to-edit
  // for chip corrections. Applied wherever a chip-correction word-strip
  // renders: single-capture expand, group-expand, proposal preview, and
  // batch-review card. Callers set state.stripEl + state.cursorIdx then
  // call _bindStripKeyboard(state).
  // -------------------------------------------------------------------
  function _redrawCursor(state) {
    if (!state.wordEls) return;
    for (var i = 0; i < state.wordEls.length; i++) {
      if (state.wordEls[i]) state.wordEls[i].classList.remove('cursor');
    }
    if (state.cursorIdx >= 0 && state.wordEls[state.cursorIdx]) {
      var el = state.wordEls[state.cursorIdx];
      el.classList.add('cursor');
      try { el.scrollIntoView({ block: 'nearest', inline: 'nearest' }); }
      catch (_) {}
    }
  }
  function _moveCursor(state, idx) {
    state.cursorIdx = idx;
    _redrawCursor(state);
  }
  // Move cursor by one VISUAL line up or down in the wrapped word-strip.
  // Uses getBoundingClientRect to first locate the nearest line in the
  // target direction, then picks the word on that line whose horizontal
  // center is closest to the current cursor's. Falls back to first/last
  // word when there's no line in the chosen direction.
  function _moveCursorLine(state, direction) {
    if (!state.wordEls || state.wordEls.length === 0) return;
    if (state.cursorIdx < 0) {
      _moveCursor(state, direction === 'down' ? 0 : state.words.length - 1);
      return;
    }
    var current = state.wordEls[state.cursorIdx];
    if (!current) return;
    var curRect = current.getBoundingClientRect();
    var curMidX = curRect.left + curRect.width / 2;
    var lineH = Math.max(curRect.height || 0, 12);
    var ahead = direction === 'down';
    // Pass 1: find the nearest line-top in the target direction.
    var nextLineTop = null;
    for (var i = 0; i < state.wordEls.length; i++) {
      var el = state.wordEls[i];
      if (!el) continue;
      var r = el.getBoundingClientRect();
      var dy = r.top - curRect.top;
      var threshold = lineH * 0.5;
      if (ahead) {
        if (dy < threshold) continue;
        if (nextLineTop === null || r.top < nextLineTop) nextLineTop = r.top;
      } else {
        if (dy > -threshold) continue;
        if (nextLineTop === null || r.top > nextLineTop) nextLineTop = r.top;
      }
    }
    if (nextLineTop === null) {
      // No line in that direction → clamp to first / last word.
      _moveCursor(state, ahead ? state.words.length - 1 : 0);
      return;
    }
    // Pass 2: pick the horizontally-closest word on the target line.
    var best = -1, bestDx = Infinity;
    for (var j = 0; j < state.wordEls.length; j++) {
      var ej = state.wordEls[j];
      if (!ej) continue;
      var rj = ej.getBoundingClientRect();
      if (Math.abs(rj.top - nextLineTop) > lineH * 0.5) continue;
      var midX = rj.left + rj.width / 2;
      var dx = Math.abs(midX - curMidX);
      if (dx < bestDx) { bestDx = dx; best = j; }
    }
    if (best >= 0) _moveCursor(state, best);
  }
  // Locate the chip covering state.cursorIdx; create one if absent +
  // requested. Returns the chip's <input> DOM (focused) or null.
  function _focusChipForCursor(state, createIfAbsent) {
    if (state.cursorIdx < 0) return null;
    var chipIdx = -1;
    for (var i = 0; i < state.corrections.length; i++) {
      if (chipCovers(state.corrections[i], state.cursorIdx)) { chipIdx = i; break; }
    }
    if (chipIdx < 0) {
      if (!createIfAbsent) return null;
      onWordClick(state, state.cursorIdx, false);
      for (var j = 0; j < state.corrections.length; j++) {
        if (chipCovers(state.corrections[j], state.cursorIdx)) { chipIdx = j; break; }
      }
    }
    if (chipIdx < 0) return null;
    var inputs = state.chipBox.querySelectorAll('.correct-input');
    var inp = inputs[chipIdx];
    if (inp) {
      try { inp.focus({ preventScroll: true }); } catch (_) { inp.focus(); }
    }
    return inp || null;
  }
  function _removeChipAtCursor(state) {
    if (state.cursorIdx < 0) return;
    for (var i = 0; i < state.corrections.length; i++) {
      if (chipCovers(state.corrections[i], state.cursorIdx)) {
        removeChip(state, i);
        applyCorrectionsToGround(state);
        return;
      }
    }
  }
  function _bindStripKeyboard(state) {
    var strip = state.stripEl;
    if (!strip || strip._kbBound) return;
    strip._kbBound = true;
    strip.setAttribute('tabindex', '0');
    // Hint line: visible only while the strip itself has focus.
    var hint = document.createElement('div');
    hint.className = 'word-strip-hint';
    hint.textContent = '← / → word · ↑ / ↓ line · type to edit · Enter accept · Del remove · Esc release';
    if (strip.parentNode) {
      strip.parentNode.insertBefore(hint, strip.nextSibling);
    }
    strip.addEventListener('focus', function() {
      strip.classList.add('has-focus');
      hint.classList.add('show');
      // Initialize cursor on first focus if not set.
      if (state.cursorIdx < 0 && state.words.length) {
        _moveCursor(state, 0);
      }
    });
    strip.addEventListener('blur', function() {
      strip.classList.remove('has-focus');
      hint.classList.remove('show');
    });
    strip.addEventListener('keydown', function(e) {
      if (e.isComposing) return;   // let IME handle dead keys / compositions
      // Ctrl/Cmd combos belong to the batch-popup handler (accept / dismiss /
      // replay / revert / Ctrl+Space play-pause) — don't consume them here as
      // plain cursor moves; let them bubble untouched to _onBatchKey.
      if (e.ctrlKey || e.metaKey) return;
      var n = state.words.length;
      if (n === 0) return;
      var k = e.key;
      if (k === 'ArrowLeft') {
        e.preventDefault();
        _moveCursor(state, Math.max(0, (state.cursorIdx < 0 ? 0 : state.cursorIdx) - 1));
      } else if (k === 'ArrowRight') {
        e.preventDefault();
        _moveCursor(state, Math.min(n - 1, (state.cursorIdx < 0 ? -1 : state.cursorIdx) + 1));
      } else if (k === 'ArrowUp') {
        e.preventDefault();
        _moveCursorLine(state, 'up');
      } else if (k === 'ArrowDown') {
        e.preventDefault();
        _moveCursorLine(state, 'down');
      } else if (k === 'Home') {
        e.preventDefault();
        _moveCursor(state, 0);
      } else if (k === 'End') {
        e.preventDefault();
        _moveCursor(state, n - 1);
      } else if (k === 'Escape') {
        e.preventDefault();
        strip.blur();
      } else if (k === 'Enter') {
        e.preventDefault();
        _focusChipForCursor(state, true);
      } else if (k === 'Delete' || k === 'Backspace') {
        e.preventDefault();
        _removeChipAtCursor(state);
      } else if (
        !e.ctrlKey && !e.metaKey && !e.altKey &&
        typeof k === 'string' && k.length === 1
      ) {
        // Printable single character — ensure chip exists + focus its
        // input + replay the keystroke into the input (the browser
        // won't auto-deliver it to the now-focused input because the
        // original keydown fired on the strip).
        var inp = _focusChipForCursor(state, true);
        if (inp) {
          e.preventDefault();
          inp.value = (inp.value || '') + k;
          inp.dispatchEvent(new Event('input', { bubbles: true }));
        }
      }
    });
  }

  // Inline-SVG icon factory. Glyphs (▶ ⏸ ⏮ ↶) used to be Unicode text,
  // but the play/pause triangles have asymmetric advance widths (icon
  // drifted off-center + nudged on toggle), and the curved-arrow family
  // (rewind/replay/undo) is visually overloaded. SVG paths render
  // identically everywhere and let us split "replay audio" (media
  // transport ⏮) from "revert action" (undo arrow ↶) into distinct
  // glyph families. Sized in em so they scale with --fs-*.
  var _SVG_NS = 'http://www.w3.org/2000/svg';
  var _SVG_PATHS = {
    play:     'M8 5v14l11-7z',
    pause:    'M6 5h4v14H6zm8 0h4v14h-4z',
    skipprev: 'M6 6h2v12H6zm3.5 6 8.5 6V6z',
    undo:     'M12.5 8c-2.65 0-5.05.99-6.9 2.6L2 7v9h9l-3.62-3.62c1.39-1.16 '
            + '3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78C21.08 '
            + '11.03 17.15 8 12.5 8z',
  };
  function _svgIcon(name) {
    var svg = document.createElementNS(_SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', '1em');
    svg.setAttribute('height', '1em');
    svg.setAttribute('fill', 'currentColor');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    var p = document.createElementNS(_SVG_NS, 'path');
    p.setAttribute('d', _SVG_PATHS[name] || _SVG_PATHS.play);
    svg.appendChild(p);
    return svg;
  }
  // Replace an element's contents with a single SVG glyph.
  function _setBtnIcon(el, name) {
    if (!el) return;
    el.textContent = '';
    el.appendChild(_svgIcon(name));
  }

  // Wrap an <audio> element in a compact play/pause + scrubber + time
  // strip. Replaces native <audio controls> in single-capture and group-
  // expand panels for visual consistency with the merge-preview controls.
  // The audio stays in the DOM (inside the wrapper, hidden) so it can
  // load + play normally; the wrapper exposes the same affordances.
  // Returns the wrapper element; caller still owns the audio reference
  // for setting .src etc.
  function _attachCompactPlayer(audio) {
    audio.removeAttribute('controls');
    audio.classList.add('audio-hidden');

    var wrap = document.createElement('div');
    wrap.className = 'compact-player';

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'compact-player-btn';
    _setBtnIcon(btn, 'play');
    btn.title = 'Play / pause (Space)';
    btn.setAttribute('aria-label', 'Play / pause');

    var scrub = document.createElement('input');
    scrub.type = 'range'; scrub.min = '0'; scrub.max = '1000'; scrub.value = '0';
    scrub.className = 'compact-player-scrub';
    scrub.disabled = true;

    var timeEl = document.createElement('span');
    timeEl.className = 'compact-player-time';
    timeEl.textContent = '0:00';
    var sep = document.createElement('span');
    sep.className = 'compact-player-time-sep'; sep.textContent = '/';
    var totalEl = document.createElement('span');
    totalEl.className = 'compact-player-time';
    totalEl.textContent = '0:00';

    wrap.appendChild(btn);
    wrap.appendChild(scrub);
    wrap.appendChild(timeEl);
    wrap.appendChild(sep);
    wrap.appendChild(totalEl);
    wrap.appendChild(audio);  // hidden, but lives inside the wrapper

    var seeking = false;
    btn.addEventListener('click', function() {
      if (audio.paused) {
        audio.play().catch(function(e) {
          if (e && e.name !== 'AbortError') toast(e.message || 'play failed', true);
        });
      } else {
        audio.pause();
      }
    });
    audio.addEventListener('play',  function() { _setBtnIcon(btn, 'pause'); });
    audio.addEventListener('pause', function() { _setBtnIcon(btn, 'play'); });
    audio.addEventListener('timeupdate', function() {
      if (!seeking && audio.duration) {
        scrub.value = (audio.currentTime / audio.duration) * 1000;
      }
      timeEl.textContent = _fmtTime(audio.currentTime || 0);
    });
    function updateTotal() {
      totalEl.textContent = _fmtTime(audio.duration || 0);
      scrub.disabled = !isFinite(audio.duration) || audio.duration <= 0;
    }
    audio.addEventListener('loadedmetadata', updateTotal);
    audio.addEventListener('durationchange', updateTotal);
    scrub.addEventListener('input', function() {
      seeking = true;
      if (audio.duration) audio.currentTime = (scrub.value / 1000) * audio.duration;
    });
    scrub.addEventListener('change', function() { seeking = false; });
    return wrap;
  }

  // Bind the shared audio's `timeupdate` / scrubber-input / pause / play
  // to this panel's UI. Returns a teardown function that detaches the
  // listeners (called when switching to a different preview).
  function _bindPanelToAudio(audio, panel, words, wordEls) {
    var scrub = panel._scrub;
    var timeEl = panel._time;
    var toggleBtn = panel._toggleBtn;
    var totalEl = panel._total;
    var seeking = false;

    function onTimeUpdate() {
      var t = audio.currentTime || 0;
      if (!seeking && audio.duration) {
        scrub.value = (t / audio.duration) * 1000;
      }
      timeEl.textContent = _fmtTime(t);
      var idx = -1;
      for (var i = 0; i < words.length; i++) {
        var s = words[i].start || 0;
        var e = words[i].end || s;
        if (s <= t && t < e) { idx = i; break; }
      }
      // Highlight in both the Corrections strip and the Final-result spans.
      if (panel._state) _setActiveWord(panel._state, idx);
    }
    function onLoaded() {
      totalEl.textContent = _fmtTime(audio.duration || 0);
      scrub.disabled = false;
      if (toggleBtn && toggleBtn._setDurSeconds) toggleBtn._setDurSeconds(audio.duration);
    }
    function onPause() { toggleBtn._setIcon('play'); }
    function onPlay()  { toggleBtn._setIcon('pause'); }
    function onScrubInput() {
      seeking = true;
      if (audio.duration) {
        audio.currentTime = (scrub.value / 1000) * audio.duration;
      }
    }
    function onScrubChange() { seeking = false; }

    audio.addEventListener('timeupdate', onTimeUpdate);
    audio.addEventListener('loadedmetadata', onLoaded);
    audio.addEventListener('durationchange', onLoaded);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('play', onPlay);
    scrub.addEventListener('input', onScrubInput);
    scrub.addEventListener('change', onScrubChange);

    return function teardown() {
      audio.removeEventListener('timeupdate', onTimeUpdate);
      audio.removeEventListener('loadedmetadata', onLoaded);
      audio.removeEventListener('durationchange', onLoaded);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('play', onPlay);
      scrub.removeEventListener('input', onScrubInput);
      scrub.removeEventListener('change', onScrubChange);
      if (panel._state) _setActiveWord(panel._state, -1);
    };
  }

  // Returns { toggleBtn, panel } — caller appends each to its own host.
  // opts.groundEl (optional): render the karaoke Final result into this
  // external element instead of a Final-result section inside the panel. The
  // manual merge modal passes its #merge-transcript so that prominent box
  // karaokes (no duplicate Final result); batch/proposal cards omit it and get
  // the in-panel Final result.
  function _makeMergePreviewBtn(memberIdsFn, silenceMsFn, durationFn, opts) {
    opts = opts || {};
    var externalGround = opts.groundEl || null;
    var toggleBtn = document.createElement('button');
    toggleBtn.type = 'button';
    toggleBtn.className = 'merge-preview-btn';
    toggleBtn.title = 'Preview the merged audio with word-level karaoke';
    var iconEl = document.createElement('span');
    iconEl.style.display = 'inline-flex';
    _setBtnIcon(iconEl, 'play');
    var durEl = document.createElement('span');
    durEl.className = 'dur';
    toggleBtn.appendChild(iconEl);
    toggleBtn.appendChild(durEl);
    if (opts.inlinePlayer) {
      // Match the single-capture player: a plain ▶ at the LEFT edge of the
      // seek bar (the duration is already shown in the card's meta row, so
      // hide the on-button label here).
      toggleBtn.classList.add('compact-player-btn');
      durEl.style.display = 'none';
      toggleBtn.title = 'Play / pause (Ctrl+Space)';
    }
    // Accepts 'play' | 'pause' | 'loading'. Loading keeps a text ellipsis
    // (no dedicated glyph) while the trimmed preview audio is fetched.
    toggleBtn._setIcon = function(s) {
      if (s === 'loading') { iconEl.textContent = '…'; return; }
      _setBtnIcon(iconEl, s === 'pause' ? 'pause' : 'play');
    };
    toggleBtn._refreshDur = function() {
      var d = durationFn();
      durEl.textContent = (typeof d === 'number' && isFinite(d))
        ? d.toFixed(1) + ' s' : '';
    };
    // Once the trimmed preview audio loads, show its real length instead of
    // the (estimated) durationFn value — keeps "▶ X s" in sync with the
    // scrubber total for every entry point (proposal cards + manual modal).
    toggleBtn._setDurSeconds = function(s) {
      if (typeof s === 'number' && isFinite(s) && s > 0) {
        durEl.textContent = s.toFixed(1) + ' s';
      }
    };
    toggleBtn._refreshDur();

    var panel = document.createElement('div');
    panel.className = 'merge-preview-panel';
    // Inline-player cards show the seek bar (with its left-edge ▶) up front;
    // the Corrections/Final sections stay hidden until the first play loads
    // the words. Other callers keep the whole panel hidden until ▶ is clicked.
    panel.hidden = !opts.inlinePlayer;
    var controls = document.createElement('div');
    controls.className = 'compact-player';
    var scrub = document.createElement('input');
    scrub.type = 'range'; scrub.min = '0'; scrub.max = '1000'; scrub.value = '0';
    scrub.className = 'compact-player-scrub';
    scrub.disabled = true;
    var timeEl = document.createElement('span');
    timeEl.className = 'compact-player-time';
    timeEl.textContent = '0:00';
    var sep = document.createElement('span');
    sep.className = 'compact-player-time-sep'; sep.textContent = '/';
    var totalEl = document.createElement('span');
    totalEl.className = 'compact-player-time';
    totalEl.textContent = '0:00';
    if (opts.inlinePlayer) controls.appendChild(toggleBtn);  // ▶ at seek-bar left
    controls.appendChild(scrub);
    controls.appendChild(timeEl);
    controls.appendChild(sep);
    controls.appendChild(totalEl);

    // Corrections section — mirrors the single-capture / group-expand
    // layout so reviewers get the same word-strip + chip list + final-
    // result preview affordances inside each proposal panel.
    var corrSec = document.createElement('div');
    corrSec.className = 'cc-section merge-preview-cc';
    corrSec.innerHTML = '<h3>Corrections</h3>'
      + '<div class="help">Click a word to mark it; shift-click another to '
      + 'extend the range; type the corrected text in the chip below. '
      + 'Edits save automatically to the source captures.</div>';
    var stripEl = document.createElement('div');
    stripEl.className = 'merge-preview-strip word-strip';
    corrSec.appendChild(stripEl);
    var chipBox = document.createElement('div');
    chipBox.className = 'cc-corrections';
    corrSec.appendChild(chipBox);

    // Final result: an external element (manual modal's #merge-transcript) or
    // an in-panel section (batch/proposal cards).
    var gtArea;
    if (externalGround) {
      gtArea = externalGround;
    } else {
      var gtSec = document.createElement('div');
      gtSec.className = 'cc-section merge-preview-cc';
      gtSec.innerHTML = '<h3>Final result</h3>'
        + '<div class="help">Computed from members\' post-processing text + '
        + 'word corrections. To change it, edit chips above.</div>';
      gtArea = document.createElement('div');
      gtArea.className = 'cc-ground';
      gtArea.setAttribute('role', 'textbox');
      gtArea.setAttribute('aria-readonly', 'true');
      gtSec.appendChild(gtArea);
    }

    panel.appendChild(controls);
    panel.appendChild(corrSec);
    if (!externalGround) panel.appendChild(gtSec);

    if (opts.inlinePlayer) {
      // Keep the editing surfaces collapsed until the first play fetches the
      // words; the inline player row stays visible from the start.
      corrSec.hidden = true;
      if (!externalGround) gtSec.hidden = true;
    }
    panel._corrSec = corrSec;
    panel._gtSec = externalGround ? null : gtSec;

    panel._toggleBtn = toggleBtn;
    panel._scrub = scrub;
    panel._time = timeEl;
    panel._total = totalEl;
    panel._strip = stripEl;
    panel._chipBox = chipBox;
    panel._gtArea = gtArea;

    toggleBtn.addEventListener('click', async function() {
      var audio = _getPreviewAudio();
      // Same card already active → toggle pause/resume; native audio
      // retains currentTime across pause.
      if (_previewActivePanel === panel) {
        if (audio.paused) {
          try { await audio.play(); }
          catch (e) { if (e && e.name !== 'AbortError') toast(e.message || 'play failed', true); }
        } else {
          audio.pause();
        }
        return;
      }
      // Different card → tear down prior, fetch fresh.
      _stopAnyPreview();
      var ids = memberIdsFn() || [];
      if (ids.length < 2) { toast('Need at least 2 members to preview', true); return; }
      toggleBtn.disabled = true;
      toggleBtn._setIcon('loading');
      try {
        var headers = {
          'Content-Type': 'application/json',
          'X-CSRF-Token': window._csrfToken ? window._csrfToken() : '',
        };
        var body = JSON.stringify({ member_ids: ids });
        var [audioResp, wordsResp] = await Promise.all([
          fetch('/captures/api/samples/preview-audio', { method:'POST', headers:headers, body:body }),
          fetch('/captures/api/samples/preview-words', { method:'POST', headers:headers, body:body }),
        ]);
        if (!audioResp.ok) {
          var amsg = 'preview failed (' + audioResp.status + ')';
          try { var aj = await audioResp.json(); if (aj && aj.detail) amsg = aj.detail; } catch (_) {}
          throw new Error(amsg);
        }
        if (!wordsResp.ok) {
          var wmsg = 'word-preview failed (' + wordsResp.status + ')';
          try { var wj = await wordsResp.json(); if (wj && wj.detail) wmsg = wj.detail; } catch (_) {}
          throw new Error(wmsg);
        }
        var blob = await audioResp.blob();
        var wordsJ = await wordsResp.json();
        var wordsArr = wordsJ.words || [];
        var seededChips = wordsJ.corrections || [];
        var transcript = wordsJ.transcript || '';
        // Build a synthetic state object the existing chip helpers
        // (renderChips, onWordClick, applyCorrectionsToGround, etc.) can
        // operate on. baselineCorrections snapshots the server's view so
        // the debounced save sends the user's intent against a stable
        // baseline; the server REPLACES per-member chips on save, so
        // baseline + edits = final state in one round-trip.
        var state = {
          words: wordsArr,
          corrections: JSON.parse(JSON.stringify(seededChips)),
          baselineCorrections: JSON.parse(JSON.stringify(seededChips)),
          finalText: transcript,
          wordEls: [],
          chipBox: chipBox,
          gtArea: gtArea,
          karaokeGround: true,
          stripEl: stripEl,
          cursorIdx: -1,
          dirtyEl: null,
          activeWordIdx: -1,
          dirty: false,
        };
        // Wire UI before src so events fire correctly.
        panel.hidden = false;
        if (panel._corrSec) panel._corrSec.hidden = false;
        if (panel._gtSec) panel._gtSec.hidden = false;
        _previewActivePanel = panel;
        state.wordEls = _renderWordStrip(stripEl, wordsArr, function(i, sh) {
          onWordClick(state, i, sh);
        });
        _bindStripKeyboard(state);
        // Stash state on the panel so the batch-card auto-focus can find
        // it without walking the DOM tree.
        panel._state = state;
        // Debounced save — fires whenever markDirty is invoked (chip
        // input, chip removal, chip extend). 250 ms matches the cadence
        // the rest of /captures uses informally.
        var saveTimer = null;
        state.onMarkDirty = function() {
          if (saveTimer) clearTimeout(saveTimer);
          saveTimer = setTimeout(function() {
            _saveProposalChips(state, ids, silenceMsFn() || 300).catch(function(e) {
              if (e && e.message) toast(e.message, true);
            });
          }, 250);
        };
        renderChips(state);
        applyCorrectionsToGround(state);
        _previewBound = _bindPanelToAudio(audio, panel, wordsArr, state.wordEls);
        _previewBlobUrl = URL.createObjectURL(blob);
        audio.src = _previewBlobUrl;
        toggleBtn._setIcon('pause');
        toggleBtn.disabled = false;
        try { await audio.play(); }
        catch (e) { if (e && e.name !== 'AbortError') throw e; }
      } catch (e) {
        toggleBtn.disabled = false;
        toggleBtn._setIcon('play');
        if (e && e.message) toast(e.message, true);
      }
    });

    return { toggleBtn: toggleBtn, panel: panel };
  }

  // Persist proposal chips: POSTs to /preview-save-chips which fans the
  // global-indexed chips to per-member captures. On success, refresh
  // baselineCorrections so the next save sends the user's current
  // intent against the now-canonical server state.
  async function _saveProposalChips(state, memberIds, silenceMs) {
    var headers = {
      'Content-Type': 'application/json',
      'X-CSRF-Token': window._csrfToken ? window._csrfToken() : '',
    };
    var chips = state.corrections
      .filter(function(c) { return (c.correct || '').trim(); })
      .map(function(c) {
        var o = { wrong: c.wrong || '', correct: (c.correct || '').trim() };
        if (typeof c.idx === 'number') {
          o.idx = c.idx;
          if (typeof c.idx_end === 'number' && c.idx_end !== c.idx) {
            o.idx_end = c.idx_end;
          }
        }
        return o;
      });
    var resp = await fetch('/captures/api/samples/preview-save-chips', {
      method: 'POST', headers: headers,
      body: JSON.stringify({
        member_ids: memberIds, corrections: chips,
      }),
    });
    if (!resp.ok) {
      var msg = 'chip save failed (' + resp.status + ')';
      try { var j = await resp.json(); if (j && j.detail) msg = j.detail; } catch (_) {}
      throw new Error(msg);
    }
    state.baselineCorrections = JSON.parse(JSON.stringify(state.corrections));
  }

  // -------------------------------------------------------------------
  // Auto-propose merges — calls /captures/api/propose-merges, renders a
  // ranked list of candidate groups. Accept commits the merge directly
  // using the join/silence settings at the top of the modal; the modal
  // stays open so the user can rapidly merge several in a row.
  // -------------------------------------------------------------------
  var _proposals = [];   // current list (mutated locally on Accept)

  function _scoreTier(n) {
    if (n >= 70) return 'good';
    if (n >= 50) return 'ok';
    return 'low';
  }

  function _renderProposals(proposals) {
    _proposals = (proposals || []).slice();
    _redrawProposalList();
  }

  function _redrawProposalList() {
    var listEl = document.getElementById('propose-list');
    listEl.innerHTML = '';
    if (!_proposals.length) {
      var empty = document.createElement('div');
      empty.className = 'empty';
      empty.textContent = 'No more proposals — Refresh to re-rank against '
        + 'the current state, or record a few more clips first.';
      listEl.appendChild(empty);
      return;
    }
    _proposals.forEach(function(p) {
      listEl.appendChild(_buildProposalCard(p));
    });
  }

  function _buildProposalCard(p) {
    var card = document.createElement('div');
    card.className = 'proposal';

    var row1 = document.createElement('div');
    row1.className = 'row1';

    var n = Math.round((p.composite_score || 0) * 100);
    var score = document.createElement('span');
    score.className = 'score tier-' + _scoreTier(n);
    score.textContent = n + ' / 100';
    score.title = 'Composite quality score (0-100).\n'
      + 'fill ' + (p.fill_score||0).toFixed(2)
      + ' · density ' + (p.density_score||0).toFixed(2)
      + ' · members ' + (p.member_score||0).toFixed(2)
      + (p.reviewed_count ? ' · reviewed boost +' + (0.1 * p.reviewed_count / (p.member_count||1)).toFixed(2) : '');
    row1.appendChild(score);

    var meterWrap = document.createElement('span');
    meterWrap.title = 'Total packed duration vs 28 s cap';
    var meterBar = document.createElement('span');
    meterBar.className = 'meter-bar';
    var fill = document.createElement('span');
    fill.className = 'fill';
    var pct = Math.min(100, Math.round((p.total_duration_s / 28.0) * 100));
    fill.style.width = pct + '%';
    meterBar.appendChild(fill);
    meterWrap.appendChild(meterBar);
    var durLabel = document.createElement('span');
    durLabel.style.cssText = 'font-family:var(--font-mono);color:var(--help);font-size:var(--fs-xs);margin-left:0.4rem;';
    durLabel.textContent = (p.total_duration_s || 0).toFixed(1) + ' s';
    meterWrap.appendChild(durLabel);
    row1.appendChild(meterWrap);

    var countLabel = document.createElement('span');
    countLabel.style.cssText = 'color:var(--help);font-size:var(--fs-sm);';
    countLabel.textContent = (p.member_count || 0) + ' clips';
    row1.appendChild(countLabel);

    var lang = document.createElement('span');
    lang.className = 'lang-pill';
    lang.textContent = p.language || '?';
    row1.appendChild(lang);

    if (p.username || p.user_id) {
      var spk = document.createElement('span');
      spk.className = 'speaker-pill';
      spk.title = 'speaker';
      spk.textContent = p.username || String(p.user_id).slice(0, 8);
      row1.appendChild(spk);
    }

    var previewCtl = _makeMergePreviewBtn(
      function() { return p.member_ids; },
      function() { return 300; },   // nominal gap; server uses the global
      function() { return p.total_duration_s; },
      // Inline player: the ▶ lives at the left edge of the seek bar inside the
      // panel (mirrors the single-capture player), not in this meta row.
      { inlinePlayer: true }
    );

    var sp = document.createElement('span');
    sp.style.flex = '1';
    row1.appendChild(sp);

    var accept = document.createElement('button');
    accept.className = 'primary';
    accept.type = 'button';
    accept.textContent = 'Accept';
    accept.addEventListener('click', function() { _acceptProposal(p, accept); });
    row1.appendChild(accept);
    card.appendChild(row1);

    var reason = document.createElement('div');
    reason.className = 'reason';
    reason.textContent = p.reason || '';
    card.appendChild(reason);

    // Preview panel host: full-width inline player below the card head. The
    // seek bar (with its left-edge ▶) shows up front; the Corrections/Final
    // sections appear once the user clicks ▶ and the words load.
    card.appendChild(previewCtl.panel);

    var members = document.createElement('div');
    members.className = 'members';
    // Suppress per-line speaker chip when every member shares the proposal's
    // speaker — keeps the line concise. Show it otherwise (mixed-speaker
    // case is currently impossible by construction but we handle defensively).
    var allSameSpeaker = (p.member_previews || []).every(function(m) {
      return (m.user_id || '') === (p.user_id || '');
    });
    (p.member_previews || []).forEach(function(m) {
      var line = document.createElement('div');
      line.className = 'm';
      var ts = document.createElement('span');
      ts.className = 'ts';
      ts.textContent = absTime(m.created_ts || 0);
      line.appendChild(ts);
      var dur = document.createElement('span');
      dur.className = 'dur';
      dur.textContent = (m.duration_s || 0).toFixed(1) + 's';
      line.appendChild(dur);
      if (!allSameSpeaker && (m.username || m.user_id)) {
        var spk = document.createElement('span');
        spk.className = 'm-spk';
        spk.textContent = m.username || String(m.user_id).slice(0, 8);
        line.appendChild(spk);
      }
      var txt = document.createElement('span');
      txt.className = 'm-text';
      txt.title = m.preview || '';
      txt.textContent = m.preview || '(no transcript)';
      line.appendChild(txt);
      members.appendChild(line);
    });
    card.appendChild(members);
    return card;
  }

  async function _loadProposals() {
    var listEl = document.getElementById('propose-list');
    listEl.innerHTML = '<div class="empty">Loading proposals…</div>';
    try {
      var j = await api('GET', '/captures/api/propose-merges');
      _renderProposals(j.proposals || []);
    } catch (e) {
      if (e && e.message !== 'unauthorized' && e.message !== 'not-admin') {
        listEl.innerHTML = '';
        var err = document.createElement('div');
        err.className = 'empty';
        err.style.color = 'var(--red, #f78585)';
        err.textContent = 'Failed to load proposals: ' + (e.message || e);
        listEl.appendChild(err);
      }
    }
  }

  async function _acceptProposal(p, btn) {
    // Validate members are still present in current data.
    var byId = {};
    _allCaptures.forEach(function(r) { byId[r.id] = r; });
    var missing = (p.member_ids || []).filter(function(id) {
      var r = byId[id];
      return !r || r.sample_id;
    });
    if (missing.length) {
      toast(missing.length + ' member(s) no longer eligible — try Refresh', true);
      return;
    }
    _stopAnyPreview();
    var origText = btn ? btn.textContent : '';
    if (btn) { btn.disabled = true; btn.textContent = 'Merging…'; }
    try {
      await api('POST', '/captures/api/samples', {
        member_ids: p.member_ids,
      });
      toast('Sample created (' + (p.member_count || p.member_ids.length) + ' clips)');
      // Drop the accepted proposal + any overlapping ones from the visible
      // list. Now-grouped members invalidate any other proposal that
      // contained them.
      var consumed = {};
      p.member_ids.forEach(function(id) { consumed[id] = true; });
      _proposals = _proposals.filter(function(other) {
        if (other === p) return false;
        return !(other.member_ids || []).some(function(id) { return consumed[id]; });
      });
      _redrawProposalList();
      // Refresh the main captures list so the new group appears outside
      // the modal. We do NOT re-fetch the proposals (per the user's
      // preferred UX); a manual Refresh re-ranks against current state.
      load();
    } catch (e) {
      if (btn) { btn.disabled = false; btn.textContent = origText || 'Accept'; }
      if (e && e.message !== 'unauthorized' && e.message !== 'not-admin') {
        toast(e.message || 'merge failed', true);
      }
    }
  }

  // -------------------------------------------------------------------
  // Batch (Tinder-swipe) review mode — focused single-card review with
  // keyboard + touch shortcuts so the reviewer can blow through a stack
  // of proposals without mouse-hopping. Dismiss is local-only per UX
  // decision; accept fires the same /groups POST as list-mode Accept.
  // -------------------------------------------------------------------
  var _batchActive = false;
  var _batchAccepted = 0;
  var _batchDismissed = 0;
  var _batchCurrentCard = null;
  var _batchKeyHandler = null;
  var _batchTouchState = null;
  // Single-level undo for the last Dismiss/Accept (Ctrl+↓ or the ↶ button).
  // Holds a pre-mutation snapshot of _proposals plus, for accepts, the
  // created sample_id so revert can dissolve it. Null = nothing to revert.
  var _batchUndo = null;

  function _enterBatchMode() {
    if (_batchActive) return;
    _batchActive = true;
    _batchAccepted = 0;
    _batchDismissed = 0;
    _batchUndo = null;
    document.getElementById('propose-list').hidden = true;
    document.getElementById('propose-batch').hidden = false;
    _batchKeyHandler = function(e) { _onBatchKey(e); };
    document.addEventListener('keydown', _batchKeyHandler);
    _renderBatchCard();
  }

  // Batch mode is always entered directly from the /captures page (via
  // the ⚡ Batch review merges button), so exiting always closes the
  // modal. Esc, the modal's Close button, and the end-of-batch "Close"
  // button all route through here.
  function _exitBatchMode() {
    if (!_batchActive) return;
    _batchActive = false;
    _stopAnyPreview();
    document.removeEventListener('keydown', _batchKeyHandler);
    _batchKeyHandler = null;
    _batchCurrentCard = null;
    document.getElementById('propose-list').hidden = false;
    document.getElementById('propose-batch').hidden = true;
    document.getElementById('propose-batch').innerHTML = '';
    document.getElementById('propose-modal').classList.remove('show');
  }

  function _renderBatchCard() {
    var host = document.getElementById('propose-batch');
    host.innerHTML = '';
    _batchCurrentCard = null;
    _stopAnyPreview();

    if (!_proposals.length) {
      _renderBatchDone(host);
      return;
    }
    var p = _proposals[0];

    var banner = document.createElement('div');
    banner.className = 'batch-banner';
    banner.innerHTML = '<span>Reviewing <span class="count">'
      + 1 + ' / ' + _proposals.length + '</span></span>'
      + '<span class="spacer"></span>'
      + '<span><span class="count">' + _batchAccepted + '</span> accepted · '
      + '<span class="count">' + _batchDismissed + '</span> dismissed</span>';
    host.appendChild(banner);

    var card = document.createElement('div');
    card.className = 'batch-card';
    // Reuse the same per-proposal card render as list mode. CSS hides
    // the inline Accept (we use the batch action row below).
    card.appendChild(_buildProposalCard(p));
    host.appendChild(card);
    _batchCurrentCard = card;

    // Action row, left→right: ↶ Revert · ✗ Dismiss · ⏮ Replay · ✓ Accept.
    // Tinder geometry — the undo (Revert) sits at the far-left edge as a
    // small amber secondary control, away from the primary Dismiss/Accept
    // pair so it can't be mis-tapped; Replay is the interior media control
    // (skip-to-start ⏮, deliberately NOT a curved arrow so it can't read
    // as "undo").
    var actions = document.createElement('div');
    actions.className = 'batch-actions';
    var revertBtn = document.createElement('button');
    revertBtn.type = 'button'; revertBtn.className = 'revert';
    revertBtn.title = 'Revert last action (Ctrl+↓)';
    revertBtn.setAttribute('aria-label', 'Revert last action');
    revertBtn.appendChild(_svgIcon('undo'));
    revertBtn.disabled = !_batchUndo;
    revertBtn.addEventListener('click', function() { _revertBatch(); });
    var dismissBtn = document.createElement('button');
    dismissBtn.type = 'button'; dismissBtn.className = 'dismiss';
    dismissBtn.textContent = '✗ Dismiss';
    dismissBtn.addEventListener('click', function() { _advanceBatch('dismiss'); });
    var replayBtn = document.createElement('button');
    replayBtn.type = 'button'; replayBtn.className = 'replay';
    replayBtn.title = 'Replay from start (Ctrl+↑)';
    replayBtn.setAttribute('aria-label', 'Replay audio from start');
    replayBtn.appendChild(_svgIcon('skipprev'));
    replayBtn.addEventListener('click', function() { _batchReplay(); });
    var acceptBtn = document.createElement('button');
    acceptBtn.type = 'button'; acceptBtn.className = 'accept primary';
    acceptBtn.textContent = '✓ Accept';
    acceptBtn.addEventListener('click', function() { _advanceBatch('accept'); });
    actions.appendChild(revertBtn);
    actions.appendChild(dismissBtn);
    actions.appendChild(replayBtn);
    actions.appendChild(acceptBtn);
    host.appendChild(actions);

    var hint = document.createElement('div');
    hint.className = 'batch-hint';
    hint.textContent = 'Ctrl+← Dismiss · Ctrl+→ Accept · Ctrl+↓ Revert · '
      + 'Ctrl+Space pause · Ctrl+↑ Replay · Esc Exit';
    host.appendChild(hint);

    // Touch swipe gestures on the card itself.
    card.addEventListener('touchstart', _onBatchTouchStart, { passive: true });
    card.addEventListener('touchmove',  _onBatchTouchMove,  { passive: true });
    card.addEventListener('touchend',   _onBatchTouchEnd,   { passive: true });

    // Auto-play: click the preview toggle inside the rendered card so
    // audio + karaoke + chip box all wire up. Synchronous click in
    // response to a user gesture (the click that triggered render via
    // _enterBatchMode or _advanceBatch) is allowed by autoplay policy.
    setTimeout(function() {
      var tBtn = card.querySelector('.merge-preview-btn');
      if (tBtn) tBtn.click();
      // Auto-focus the word-strip so keyboard nav (← / →, type to edit)
      // works immediately without a click. Wait for the preview panel
      // to render its strip — _saveProposalChips and the audio fetch
      // both await, so poll briefly.
      var tries = 0;
      var focusTimer = setInterval(function() {
        tries++;
        var panel = card.querySelector('.merge-preview-panel');
        var state = panel && panel._state;
        var strip = state && state.stripEl;
        if (strip && !strip.hidden && state.words && state.words.length) {
          if (state.cursorIdx < 0) {
            state.cursorIdx = 0;
            _redrawCursor(state);
          }
          try { strip.focus({ preventScroll: true }); } catch (_) {}
          clearInterval(focusTimer);
        } else if (tries > 40) {   // ~4 s cap, gives up if preview never loads
          clearInterval(focusTimer);
        }
      }, 100);
    }, 0);
  }

  function _renderBatchDone(host) {
    var done = document.createElement('div');
    done.className = 'batch-done';
    done.innerHTML = '<h4>All done.</h4>'
      + '<div class="summary"><span class="count">' + _batchAccepted
      + '</span> accepted · <span class="count">' + _batchDismissed
      + '</span> dismissed</div>';
    var acts = document.createElement('div');
    acts.className = 'actions';
    var refreshBtn = document.createElement('button');
    refreshBtn.type = 'button'; refreshBtn.className = 'primary';
    refreshBtn.textContent = 'Refresh batch';
    refreshBtn.addEventListener('click', async function() {
      await _loadProposals();
      _batchAccepted = 0; _batchDismissed = 0;
      _renderBatchCard();
    });
    var closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.textContent = 'Close';
    closeBtn.addEventListener('click', _closePropose);
    acts.appendChild(refreshBtn);
    acts.appendChild(closeBtn);
    done.appendChild(acts);
    host.appendChild(done);
  }

  async function _advanceBatch(action) {
    if (!_batchActive || !_proposals.length) return;
    var p = _proposals[0];
    var card = _batchCurrentCard;

    if (action === 'accept') {
      // Mirror _acceptProposal's behavior but adapted for batch — we
      // don't have a per-card Accept button to flip into "Merging…".
      var byId = {};
      _allCaptures.forEach(function(r) { byId[r.id] = r; });
      var missing = (p.member_ids || []).filter(function(id) {
        var r = byId[id]; return !r || r.sample_id;
      });
      if (missing.length) {
        toast(missing.length + ' member(s) no longer eligible — try Refresh', true);
        return;
      }
      _stopAnyPreview();
      // Snapshot the proposal queue BEFORE we splice it so Revert can
      // restore the accepted proposal + any overlapping ones it removed.
      var acceptSnap = _proposals.slice();
      try {
        var resp = await api('POST', '/captures/api/samples', {
          member_ids: p.member_ids,
        });
        toast('Sample created (' + (p.member_count || p.member_ids.length) + ' clips)');
        _batchAccepted++;
        // Splice accepted + any overlapping proposals (their members are
        // now grouped, so they'd be invalid anyway).
        var consumed = {};
        p.member_ids.forEach(function(id) { consumed[id] = true; });
        _proposals = _proposals.filter(function(other) {
          if (other === p) return false;
          return !(other.member_ids || []).some(function(id) { return consumed[id]; });
        });
        // Remember how to undo this accept: dissolve the created group +
        // restore the pre-splice queue. sample_id comes from create_sample_api.
        _batchUndo = {
          action: 'accept',
          proposals: acceptSnap,
          sid: (resp && resp.sample_id) || null,
          clips: p.member_count || p.member_ids.length,
        };
        load();   // refresh main captures list in background
      } catch (e) {
        if (e && e.message !== 'unauthorized' && e.message !== 'not-admin') {
          toast(e.message || 'merge failed', true);
        }
        return;   // don't advance on failure; let user retry
      }
    } else if (action === 'dismiss') {
      // Snapshot before shift so Revert can put the proposal back at front.
      var dismissSnap = _proposals.slice();
      _batchDismissed++;
      _proposals.shift();
      _batchUndo = { action: 'dismiss', proposals: dismissSnap };
    }

    if (card) {
      card.classList.add(action === 'accept' ? 'gone-right' : 'gone-left');
      setTimeout(_renderBatchCard, 260);  // wait for CSS transition
    } else {
      _renderBatchCard();
    }
  }

  function _batchReplay() {
    if (_previewAudio) {
      try { _previewAudio.currentTime = 0; _previewAudio.play(); } catch (_) {}
    }
  }

  // Undo the last Dismiss/Accept (single level). Dismiss is a pure local
  // re-enqueue; Accept dissolves the server-side group it created and
  // restores the spliced queue. Tolerant of a 404 (group already gone).
  async function _revertBatch() {
    if (!_batchActive) return;
    var u = _batchUndo;
    if (!u) { toast('Nothing to revert'); return; }
    if (u.action === 'dismiss') {
      _proposals = u.proposals;
      _batchDismissed = Math.max(0, _batchDismissed - 1);
      _batchUndo = null;
      toast('Reverted — proposal restored');
      _renderBatchCard();
      return;
    }
    // accept → dissolve the created group, then restore the queue.
    // Disable the button while the DELETE is in flight (no double-fire).
    var btn = _batchCurrentCard
      && _batchCurrentCard.parentNode
      && _batchCurrentCard.parentNode.querySelector('.batch-actions .revert');
    if (btn) btn.disabled = true;
    try {
      if (u.sid) {
        await api('DELETE', '/captures/api/samples/' + encodeURIComponent(u.sid));
      }
    } catch (e) {
      // 404 = already dissolved elsewhere; treat as success. Anything else
      // is a real failure — keep the undo entry so the user can retry.
      if (!(e && /404|not found/i.test(e.message || ''))) {
        if (e && e.message !== 'unauthorized') {
          toast('Revert failed: ' + (e.message || 'error'), true);
        }
        if (btn) btn.disabled = false;
        return;
      }
    }
    _proposals = u.proposals;
    _batchAccepted = Math.max(0, _batchAccepted - 1);
    _batchUndo = null;
    toast('Reverted — sample dissolved');
    load();              // refresh main captures list (members freed again)
    _renderBatchCard();
  }

  // Resolve the word-strip state of the card currently on screen, so plain
  // arrow keys can drive Corrections navigation without a prior click.
  function _activeBatchState() {
    if (!_batchCurrentCard) return null;
    var panel = _batchCurrentCard.querySelector('.merge-preview-panel');
    return (panel && panel._state) || null;
  }

  function _onBatchKey(e) {
    if (!_batchActive) return;
    // The focused word-strip handles its own arrows/typing and calls
    // preventDefault — bail so we never double-handle the same event.
    if (e.defaultPrevented) return;
    // Don't hijack keys the browser owns inside editable / native controls
    // (chip inputs, the seek slider, <select>s). Esc still exits.
    var t = e.target;
    if (t && t.closest && t.closest(
        'input, textarea, select, [contenteditable], [role="slider"]')) {
      if (e.key === 'Escape') { e.preventDefault(); _exitBatchMode(); }
      return;
    }
    if (e.key === 'Escape') {
      e.preventDefault(); _exitBatchMode(); return;
    }
    if (e.ctrlKey || e.metaKey) {
      // Ctrl+Space toggles play/pause — plain Space is left free to type
      // into a correction chip. These fire regardless of where focus sits
      // in the popup (incl. the focused word-strip).
      if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault();
        if (_previewAudio) {
          if (_previewAudio.paused) _previewAudio.play().catch(function() {});
          else _previewAudio.pause();
        }
      }
      else if (e.key === 'ArrowRight') { e.preventDefault(); _advanceBatch('accept'); }
      else if (e.key === 'ArrowLeft')  { e.preventDefault(); _advanceBatch('dismiss'); }
      else if (e.key === 'ArrowUp')    { e.preventDefault(); _batchReplay(); }
      else if (e.key === 'ArrowDown')  { e.preventDefault(); _revertBatch(); }
      return;
    }
    if (e.altKey) return;
    // Plain arrows (no modifier) → Corrections word navigation, anywhere in
    // the popup. Focus the strip afterward so typing flows into the chip.
    var st = _activeBatchState();
    if (!st || !st.words || !st.words.length) return;
    var cur = st.cursorIdx < 0 ? -1 : st.cursorIdx;
    var n = st.words.length;
    var handled = true;
    if (e.key === 'ArrowLeft')       { _moveCursor(st, Math.max(0, (cur < 0 ? 0 : cur) - 1)); }
    else if (e.key === 'ArrowRight') { _moveCursor(st, Math.min(n - 1, cur + 1)); }
    else if (e.key === 'ArrowUp')    { _moveCursorLine(st, 'up'); }
    else if (e.key === 'ArrowDown')  { _moveCursorLine(st, 'down'); }
    else if (e.key === 'Home')       { _moveCursor(st, 0); }
    else if (e.key === 'End')        { _moveCursor(st, n - 1); }
    else { handled = false; }
    if (handled) {
      e.preventDefault();
      if (st.stripEl) { try { st.stripEl.focus({ preventScroll: true }); } catch (_) {} }
    }
  }

  function _onBatchTouchStart(e) {
    if (!e.touches || !e.touches.length) return;
    // Don't start a card swipe when the gesture begins on an interactive
    // control — otherwise dragging the seek bar, opening a dropdown, editing a
    // correction chip, or marking a word would also fling the card to the next
    // proposal. The browser handles those drags natively; swipe-navigation only
    // arms on neutral card surface.
    var t = e.target;
    if (t && t.closest && t.closest(
        'input, select, textarea, button, [contenteditable], ' +
        '.word, .compact-player, .cc-corrections, a')) {
      _batchTouchState = null;
      return;
    }
    _batchTouchState = { x0: e.touches[0].clientX, x: e.touches[0].clientX };
    if (_batchCurrentCard) _batchCurrentCard.classList.add('swiping');
  }
  function _onBatchTouchMove(e) {
    if (!_batchTouchState || !e.touches || !e.touches.length) return;
    _batchTouchState.x = e.touches[0].clientX;
    var dx = _batchTouchState.x - _batchTouchState.x0;
    if (_batchCurrentCard) {
      _batchCurrentCard.style.transform =
        'translateX(' + dx + 'px) rotate(' + (dx / 25) + 'deg)';
    }
  }
  function _onBatchTouchEnd() {
    if (!_batchTouchState) return;
    var dx = _batchTouchState.x - _batchTouchState.x0;
    _batchTouchState = null;
    if (_batchCurrentCard) {
      _batchCurrentCard.classList.remove('swiping');
      _batchCurrentCard.style.transform = '';
    }
    if (Math.abs(dx) > 100) {
      _advanceBatch(dx > 0 ? 'accept' : 'dismiss');
    }
  }

  function _openProposeModal(opts) {
    var modal = document.getElementById('propose-modal');
    modal.classList.add('show');
    var startInBatch = opts && opts.startInBatch;
    _loadProposals().then(function() {
      if (startInBatch && _proposals.length > 0 && !_batchActive) {
        _enterBatchMode();
      }
    });
  }
  function _openProposeBatch() { _openProposeModal({ startInBatch: true }); }

  // -------------------------------------------------------------------
  // Group row rendering — packed-for-fine-tune training samples
  // -------------------------------------------------------------------
  function _renderSampleCard(g) {
    var card = document.createElement('div');
    card.className = 'capture-card is-group';
    var head = document.createElement('div');
    head.className = 'cc-head';
    head.innerHTML =
      '<span class="expand-arrow">›</span>' +
      '<span class="when" data-ts="' + (g.created_ts || 0) + '" title="' +
        escapeHtml(absTime(g.created_ts)) + '">' +
        escapeHtml(fmtWhen(g.created_ts)) + '</span>' +
      '<span class="pill group-pill">sample</span>' +
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
    head.addEventListener('click', function() { _toggleSampleExpand(card, g); });
    return card;
  }

  function _toggleSampleExpand(card, g) {
    if (card.classList.contains('open')) {
      card.classList.remove('open');
      // Mirror singleton collapse(): pause + revoke the blob URL and
      // wipe the body so the next expand re-fetches and re-binds —
      // otherwise hidden audio keeps playing and the blob leaks until
      // the next render() wipe.
      var st = _openSamples[g.id];
      if (st && st.audio) {
        try { st.audio.pause(); } catch(_) {}
        if (st.audio.src && st.audio.src.indexOf('blob:') === 0) {
          try { URL.revokeObjectURL(st.audio.src); } catch(_) {}
        }
      }
      delete _openSamples[g.id];
      var bodyEl = card.querySelector('.cc-body');
      if (bodyEl) { bodyEl.dataset.built = '0'; bodyEl.innerHTML = ''; }
      return;
    }
    card.classList.add('open');
    var body = card.querySelector('.cc-body');
    if (body.dataset.built === '1') return;
    // Guard against concurrent expands on the same group (collapse →
    // re-expand while the first GET is still in flight) — a second build
    // would append a second <audio> element whose blob URL the existing
    // _openSamples tracking can't reach, leaking on render() wipe.
    if (body.dataset.fetching === '1') return;
    body.dataset.fetching = '1';
    api('GET', '/captures/api/samples/' + encodeURIComponent(g.id))
      .then(function(j) {
        var detail = j.sample || g;

        // --- Skeleton: audio slot, transcript textarea, karaoke band,
        // chip box, settings row, button row, members list. All built
        // once; their CONTENT is filled (and refilled) by
        // applyServerSample() below. ---
        var audio = document.createElement('audio');
        audio.preload = 'metadata';
        var audioPlayerWrap = _attachCompactPlayer(audio);
        var audioSlot = document.createElement('div');
        audioSlot.appendChild(audioPlayerWrap);
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
          v.className = 'val' + (value ? ' ws-region' : ' dim');
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

        var sampleState = {
          sid: detail.id,
          audio: audio,
          words: [],
          finalText: '',
          corrections: [],
          wordEls: [],
          activeWordIdx: -1,
          chipBox: null,
          gtArea: ta,
          karaokeGround: true,
          adminNotes: '',
          newStatus: 'new',
          dirty: false,
          dirtyEl: null,
        };

        // ---- Corrections section: karaoke band + chip list together. ----
        var corrSec = document.createElement('div');
        corrSec.className = 'cc-section';
        corrSec.innerHTML = '<h3>Corrections</h3>'
          + '<div class="help">Click a word, or use ← / → and type, to '
          + 'mark and correct it; shift-click extends a range; Enter '
          + 'accepts, Del removes.</div>';
        var staleHint = document.createElement('div');
        staleHint.className = 'help';
        staleHint.style.color = 'var(--cyan)';
        corrSec.appendChild(staleHint);
        var strip = document.createElement('div');
        strip.className = 'word-strip';
        corrSec.appendChild(strip);
        sampleState.stripEl = strip;
        if (typeof sampleState.cursorIdx !== 'number') sampleState.cursorIdx = -1;
        _bindStripKeyboard(sampleState);

        // Karaoke highlight via timeupdate (attached once; reads the
        // live sampleState.words). Survives applyServerSample re-renders.
        audio.addEventListener('timeupdate', function() {
          var t = audio.currentTime;
          var idx = -1;
          for (var i = 0; i < sampleState.words.length; i++) {
            var s = sampleState.words[i].start || 0;
            var e = sampleState.words[i].end || 0;
            if (s <= t && t < e) { idx = i; break; }
          }
          _setActiveWord(sampleState, idx);
        });

        // Chip list nests inside the same Corrections section so the
        // word strip and its chips read as one harmonized surface.
        var chipBox = document.createElement('div');
        chipBox.className = 'cc-corrections';
        corrSec.appendChild(chipBox);
        body.appendChild(corrSec);
        sampleState.chipBox = chipBox;

        // ---- Final result section (read-only preview). ----
        var gtSec = document.createElement('div');
        gtSec.className = 'cc-section';
        gtSec.innerHTML = '<h3>Final result</h3>'
          + '<div class="help">Computed from members\' post-processing '
          + 'text + word corrections. To change it, edit chips above or '
          + 'update rules on /quick-config.</div>';
        gtSec.appendChild(ta);
        body.appendChild(gtSec);

        // --- Merge settings (read-only). Join style + inter-member silence
        // are GLOBAL now (Settings → Sample sizing); this row just shows what
        // this sample was built with. Regenerate rebuilds it with the current
        // global silence. ---
        var settingsSec = document.createElement('div');
        settingsSec.className = 'cc-section';
        settingsSec.style.cssText = 'display:flex;gap:1rem;align-items:center;'
          + 'flex-wrap:wrap;margin-top:0.5rem;';
        var settingsHint = document.createElement('span');
        settingsHint.className = 'help';
        settingsSec.appendChild(settingsHint);
        body.appendChild(settingsSec);

        function refreshSettingsHint() {
          var j = detail.transcript_join_strategy || 'space';
          var sil = detail.inter_segment_silence_ms;
          settingsHint.textContent =
            'Built with join "' + j + '"'
            + (sil != null ? ' · silence ' + sil + ' ms' : '')
            + ' · join & silence are global (Settings → Sample sizing); '
            + 'Regenerate rebuilds with the current global silence.';
          settingsHint.style.color = 'var(--help)';
        }

        // --- Admin notes (matches single-capture layout). ---
        var notesSec = document.createElement('div');
        notesSec.className = 'cc-section cc-notes';
        notesSec.innerHTML = '<h3>Admin notes</h3>';
        var notesArea = document.createElement('textarea');
        notesArea.addEventListener('input', function() {
          sampleState.adminNotes = notesArea.value;
          markDirty(sampleState);
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
          sampleState.newStatus || 'new',
          async function(v) {
            var prev = sampleState.newStatus;
            sampleState.newStatus = v;
            try {
              // Narrow PATCH — status only. Concurrent transcript or
              // chip edits in this view stay dirty and persist only on
              // Save click; the three-way-merge path isn't engaged.
              await api('PATCH',
                '/captures/api/samples/' + encodeURIComponent(g.id),
                { status: v });
              // Sync the collapsed-list header so the next render() /
              // status-filter sees the new status without a full reload.
              // Status-only PATCH doesn't need (and must not run) the
              // full applyServerSample — it would rebuild word strip,
              // chips, audio, etc. for a single-pill change.
              var idx = _allSamples.findIndex(function(x) { return x.id === g.id; });
              if (idx >= 0) _allSamples[idx].status = v;
              reloadCounts();
              toast('Status: ' + v);
            } catch (e) {
              statusGrp.setValue(prev);
              sampleState.newStatus = prev;
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
        sampleState.dirtyEl = dirtyEl;
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
        dissolveBtn.textContent = 'Dissolve sample';
        dissolveBtn.className = 'danger';
        actions.appendChild(dissolveBtn);
        body.appendChild(actions);

        // --- Members list (replaced on each applyServerSample) ---
        var membersDiv = document.createElement('div');
        membersDiv.className = 'group-members';
        body.appendChild(membersDiv);

        // --- refreshAudio: re-fetches the blob; on failure shows the
        // missing-audio banner in place of the <audio> element. The
        // banner's button hits applyServerSample so success heals the
        // open card without a load() wipe. ---
        function refreshAudio() {
          // Reclaim the <audio> element if the banner replaced it.
          if (audio.src && audio.src.indexOf('blob:') === 0) {
            try { URL.revokeObjectURL(audio.src); } catch (_) {}
          }
          audio.removeAttribute('src');
          // Reclaim the compact-player wrapper if the banner replaced it.
          if (audioSlot.firstChild !== audioPlayerWrap) {
            audioSlot.innerHTML = '';
            audioSlot.appendChild(audioPlayerWrap);
          }
          fetch('/captures/api/samples/' + encodeURIComponent(g.id) + '/audio')
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
                + '⚠ Audio unavailable for this sample ('
                + escapeHtml(err && err.message || 'fetch failed') + ').'
                + '</span>';
              var fixBtn = document.createElement('button');
              fixBtn.textContent = 'Regenerate audio';
              fixBtn.className = 'primary';
              fixBtn.onclick = function() {
                fixBtn.disabled = true;
                api('POST', '/captures/api/samples/'
                    + encodeURIComponent(g.id) + '/regenerate')
                  .then(function(j) {
                    toast('Regenerated');
                    applyServerSample(j.sample || {});
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

        // --- applyServerSample: the single source of truth for "the
        // server told us the group looks like X, now reflect X in the
        // open card." Called once on initial expand and again after
        // every Save / Regenerate. Replaces the previous load()-wipe
        // pattern. ---
        function applyServerSample(d) {
          // 1. sampleState data.
          sampleState.words = d.merged_words || [];
          sampleState.corrections = (d.corrections || []).map(function(c) {
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
          // (cross-tab admin saves, or a member edited from its singleton
          // /captures card).
          sampleState.baselineCorrections =
            JSON.parse(JSON.stringify(d.corrections || []));
          sampleState.finalText = d.transcript || '';
          sampleState.adminNotes = d.admin_notes || '';
          sampleState.newStatus = d.status || 'new';
          sampleState.wordEls = [];
          sampleState._activeWordIdx = -1;
          // Render Final result as karaoke spans (from merged words + chips)
          // instead of plain transcript text, so it highlights in sync with
          // the Corrections strip as the merged audio plays.
          applyCorrectionsToGround(sampleState);

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
          clearDirty(sampleState);

          // 2. Refresh the read-only settings line from the server snapshot.
          detail.transcript_join_strategy = d.transcript_join_strategy || 'space';
          detail.inter_segment_silence_ms = d.inter_segment_silence_ms;
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
          sampleState.words.forEach(function(w, i) {
            var sp = document.createElement('span');
            sp.className = 'word ' + ((w.member_idx % 2 === 0) ? 'mem-even' : 'mem-odd');
            // Small horizontal breathing room at member boundaries (no
            // hard rule — the tint shift does the work visually).
            if (w.member_idx !== prevMember && prevMember !== -1) {
              sp.style.marginLeft = '0.4rem';
            }
            prevMember = w.member_idx;
            var dw = _dispWord(w);
            sp.textContent = (dw || '').replace(/^\s+/, ' ');
            if (_dispRemoved(w)) {
              sp.classList.add('rule-removed');
              sp.title = 'removed by pipeline rule';
            } else if (w.raw_word && (w.raw_word || '').trim() !== (dw || '').trim()) {
              sp.title = 'raw: ' + w.raw_word;
              sp.classList.add('post-edited');
            }
            sp.addEventListener('click', function(e) {
              onWordClick(sampleState, i, !!e.shiftKey);
            });
            strip.appendChild(sp);
            sampleState.wordEls.push(sp);
          });
          // Word DOM was rebuilt — re-paint the keyboard cursor on the
          // new spans so a mid-edit reload doesn't lose the cursor.
          if (typeof _redrawCursor === 'function') _redrawCursor(sampleState);

          // 6. Chips + pre-marked word selections from saved corrections.
          renderChips(sampleState);
          sampleState.corrections.forEach(function(c) {
            if (typeof c.idx !== 'number') return;
            var end = (typeof c.idx_end === 'number') ? c.idx_end : c.idx;
            for (var j = c.idx; j <= end; j++) selectWord(sampleState, j, true);
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
            line.innerHTML = '[' + (m.sample_order + 1) + '] '
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
          var idx = _allSamples.findIndex(function(x) { return x.id === d.id; });
          if (idx >= 0) Object.assign(_allSamples[idx], d);
        }

        // --- Save handler: PATCH everything; mutate the card in place. ---
        saveTBtn.onclick = function() {
          var payload = {
            status:        sampleState.newStatus,
            admin_notes:   sampleState.adminNotes,
            corrections:   sampleState.corrections.map(function(c) {
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
            baseline_corrections: sampleState.baselineCorrections || [],
          };
          saveTBtn.disabled = true;
          api('PATCH', '/captures/api/samples/' + encodeURIComponent(g.id),
              payload)
            .then(function(j) {
              applyServerSample(j.sample || {});
              toast('Saved');
              reloadCounts();
            })
            .catch(function(e) {
              if (e.message !== 'unauthorized') toast(e.message, true);
            })
            .then(function() { saveTBtn.disabled = false; });
        };

        // --- Regenerate handler: rebuild the merged WAV from current members
        // using the CURRENT global silence (the dedicated /regenerate
        // endpoint). In-place refresh via applyServerSample. ---
        regenBtn.onclick = function() {
          regenBtn.disabled = true;
          api('POST', '/captures/api/samples/' + encodeURIComponent(g.id) + '/regenerate')
            .then(function(j) {
              applyServerSample(j.sample || {});
              toast('Regenerated');
              reloadCounts();
            })
            .catch(function(e) {
              if (e.message !== 'unauthorized') toast(e.message, true);
            })
            .then(function() { regenBtn.disabled = false; });
        };

        // --- Lock toggle ---
        lockBtn.onclick = function() {
          api('PATCH', '/captures/api/samples/' + encodeURIComponent(g.id),
              { is_locked: !(lockBtn.textContent === 'Unlock') })
            .then(function(j) {
              var d = j.sample || {};
              // applyServerSample would clobber unsaved chip/notes/settings
              // edits. When dirty, apply only the lock-related visuals.
              if (!sampleState.dirty) {
                applyServerSample(d);
              } else {
                var isLocked = !!d.is_locked;
                lockBtn.textContent = isLocked ? 'Unlock' : 'Lock';
                dissolveBtn.style.display = isLocked ? 'none' : '';
                // Sync the collapsed-list header so the lock-pill in
                // _renderSampleCard reflects the new state on next render().
                var idx = _allSamples.findIndex(function(x) { return x.id === g.id; });
                if (idx >= 0) _allSamples[idx].is_locked = d.is_locked;
              }
              toast('Updated');
            })
            .catch(function(e) {
              if (e.message !== 'unauthorized') toast(e.message, true);
            });
        };

        // --- Dissolve (full reload — the group disappears from the list). ---
        dissolveBtn.onclick = async function() {
          if (!(await _confirm({
              title: 'Dissolve sample?',
              body: 'Dissolve this sample? Members return to the flat list; '
                  + 'merged WAV is unlinked.',
              confirmLabel: 'Dissolve', danger: true })))
            return;
          api('DELETE', '/captures/api/samples/' + encodeURIComponent(g.id))
            .then(function() { toast('Dissolved'); return load(); })
            .catch(function(e) {
              if (e.message !== 'unauthorized') toast(e.message, true);
            });
        };

        // Initial fill.
        applyServerSample(detail);
        body.dataset.built = '1';
        body.dataset.fetching = '';
        _openSamples[g.id] = { audio: audio };
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
    Object.keys(_openSamples).forEach(function(sid) {
      var st = _openSamples[sid];
      if (st && st.audio && st.audio.src && st.audio.src.indexOf('blob:') === 0) {
        try { URL.revokeObjectURL(st.audio.src); } catch(_) {}
      }
    });
    _openSamples = {};
    list.innerHTML = '';
    // Build a merged timeline: ungrouped captures + group cards (members
    // are nested inside group cards, so we exclude them from the flat list).
    var ungrouped = rows.filter(function(r) { return !r.sample_id; });
    // Apply the same status filter to groups. `audio_missing` is a
    // captures-only system status — groups don't have it, so the
    // groups section renders empty when that filter is active.
    var filteredSamples = (_filtStatus === 'all')
      ? _allSamples.slice()
      : _allSamples.filter(function(g) { return g.status === _filtStatus; });
    var combined = ungrouped.map(function(r) {
      return { kind: 'capture', ts: r.created_ts || 0, data: r };
    }).concat(filteredSamples.map(function(g) {
      return { kind: 'sample', ts: g.created_ts || 0, data: g };
    }));
    combined.sort(function(a, b) { return b.ts - a.ts; });

    if (combined.length === 0) {
      var empty = document.createElement('div');
      empty.className = 'empty-state';
      empty.innerHTML = _allCaptures.length === 0
        ? '<strong>No captures yet.</strong> Enable <em>CAPTURE_RECORDINGS_ENABLED</em> in /settings and send a transcription request.'
        : 'No captures match the current filters.';
      list.appendChild(empty);
      return;
    }
    combined.forEach(function(item) {
      list.appendChild(item.kind === 'capture'
        ? renderCard(item.data)
        : _renderSampleCard(item.data));
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
  document.getElementById('btn-reprocess-vad').addEventListener('click', onReprocessVad);
  document.getElementById('btn-advanced').addEventListener('click', function(e) {
    e.stopPropagation(); _toggleAdvMenu();
  });
  // Close the Advanced menu on an outside click / Escape.
  document.addEventListener('click', function(e) {
    var wrap = document.querySelector('.adv-wrap');
    if (wrap && !wrap.contains(e.target)) _closeAdvMenu();
  });
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') _closeAdvMenu();
  });
  document.getElementById('ab-merge').addEventListener('click', _openMergeModal);
  document.getElementById('ab-clear').addEventListener('click', _clearSelection);
  document.getElementById('btn-propose').addEventListener('click', function() { _openProposeModal(); });
  document.getElementById('btn-batch').addEventListener('click', _openProposeBatch);
  function _closePropose() {
    if (_batchActive) _exitBatchMode();
    _stopAnyPreview();
    document.getElementById('propose-modal').classList.remove('show');
  }
  document.getElementById('propose-refresh').addEventListener('click', _loadProposals);
  document.getElementById('propose-close').addEventListener('click', _closePropose);
  document.getElementById('propose-refresh-top').addEventListener('click', _loadProposals);
  document.getElementById('propose-close-top').addEventListener('click', _closePropose);

  // Revoke any open audio blob URLs on tab close — also handled per-row
  // on collapse, but this is the safety net.
  window.addEventListener('beforeunload', function() {
    Object.keys(_openRows).forEach(function(cid) {
      var st = _openRows[cid];
      if (st && st.blobUrl) {
        try { URL.revokeObjectURL(st.blobUrl); } catch(_) {}
      }
    });
    Object.keys(_openSamples).forEach(function(sid) {
      var st = _openSamples[sid];
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
