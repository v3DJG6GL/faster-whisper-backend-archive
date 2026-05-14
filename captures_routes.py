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
import time
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse,
)
from pydantic import BaseModel, Field

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
    return JSONResponse({"captures": captures_store.find_by_request_id(request_id)})


# Literal-path routes (export, clear) MUST be declared BEFORE the
# parameterized /captures/api/{cid} route — FastAPI/Starlette match in
# declaration order, and the `{cid}` placeholder would otherwise swallow
# any literal-named GET like /captures/api/export with cid="export".
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
        patch["corrections"] = [c.model_dump() for c in payload.corrections]
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
# Capture groups (≤28 s packed training samples)
# ---------------------------------------------------------------------

class CreateGroupIn(BaseModel):
    model_config = {"extra": "forbid"}
    member_ids: list[str] = Field(min_length=2, max_length=30)
    transcript: str = Field(default="", max_length=20_000)
    join_strategy: Literal["space", "period_space", "newline"] = "space"
    silence_ms: int = Field(default=300, ge=0, le=2000)


class PatchGroupIn(BaseModel):
    model_config = {"extra": "forbid"}
    transcript: str | None = Field(default=None, max_length=20_000)
    join_strategy: Literal["space", "period_space", "newline"] | None = None
    silence_ms: int | None = Field(default=None, ge=0, le=2000)
    is_locked: bool | None = None


_JOIN_STR = {"space": " ", "period_space": ". ", "newline": "\n"}


def _build_default_transcript(members: list[dict[str, Any]], strategy: str) -> str:
    """Concatenate member transcripts with the chosen join string. Prefers
    corrected_text > final > raw per-member, matching export ordering."""
    parts: list[str] = []
    for m in members:
        t = (m.get("corrected_text") or m.get("final") or m.get("raw") or "").strip()
        if t:
            parts.append(t)
    return _JOIN_STR.get(strategy, " ").join(parts)


def _build_merged_wav(
    *,
    gid: str,
    member_ids: list[str],
    silence_ms: int,
) -> tuple[int, dict[str, str]]:
    """Resolve member audio paths, run the merge, return (duration_ms,
    member_hash_map). Caller must have validated member_ids belong to
    the same user and total ≤28 s."""
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
        _bytes, n_samples = audio_merge.merge_wavs(
            member_paths, dst_abs, gap_ms=silence_ms,
        )
    except audio_merge.WavFormatError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    duration_ms = int(round(n_samples / 16.0))  # 16 samples/ms at 16 kHz
    return duration_ms, hashes


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
    duration_ms, hashes = _build_merged_wav(
        gid=gid,
        member_ids=member_ids,
        silence_ms=payload.silence_ms,
    )
    # Insert under the gid we already built audio for.
    # capture_groups_store.create_group generates its OWN gid → use a
    # local helper to honour the pre-computed gid instead.
    _insert_group_with_gid(
        gid=gid,
        user_id=owner_user_id,
        member_ids=member_ids,
        transcript=transcript,
        join_strategy=payload.join_strategy,
        silence_ms=payload.silence_ms,
        member_hash_map=hashes,
        duration_ms=duration_ms,
    )
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
) -> None:
    """Direct insert that honours a pre-allocated gid (needed because the
    audio file is written at the gid path before this call). Mirrors the
    transactional shape of capture_groups_store.create_group."""
    import capture_groups_store
    import json as _json
    import time as _time

    relpath = capture_groups_store._relpath_for(gid)
    now = _time.time()
    conn = capture_groups_store._require_conn()
    with capture_groups_store._lock:
        with conn:
            conn.execute(
                "INSERT INTO capture_groups"
                " (id, user_id, created_ts, merged_wav_relpath,"
                "  merged_duration_ms, transcript,"
                "  transcript_join_strategy, member_hashes_json,"
                "  inter_segment_silence_ms, is_stale, is_locked)"
                " VALUES (?,?,?,?,?,?,?,?,?,0,0)",
                (
                    gid, user_id, now, relpath, int(duration_ms),
                    transcript, join_strategy,
                    _json.dumps(member_hash_map, sort_keys=True),
                    int(silence_ms),
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
    "/captures/api/groups",
    dependencies=[Depends(require_admin_host)],
)
async def list_groups_api(
    user_filter: str | None = Query(None, alias="user_id"),
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    import capture_groups_store
    if not user.get("is_admin"):
        scope = user.get("user_id")
    else:
        scope = user_filter
    return JSONResponse({"groups": capture_groups_store.list_groups(user_id=scope)})


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
    g["members"] = capture_groups_store.get_members(gid)
    return JSONResponse({"group": g})


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
    if payload.transcript is not None:
        patch["transcript"] = payload.transcript.strip()
    if payload.join_strategy is not None and \
            payload.join_strategy != g["transcript_join_strategy"]:
        patch["transcript_join_strategy"] = payload.join_strategy
    if payload.silence_ms is not None and \
            payload.silence_ms != g["inter_segment_silence_ms"]:
        patch["inter_segment_silence_ms"] = payload.silence_ms
        rebuild_audio = True
    if payload.is_locked is not None:
        patch["is_locked"] = 1 if payload.is_locked else 0

    if rebuild_audio:
        # Re-run the merge with the new silence. Member set is unchanged.
        members = capture_groups_store.get_members(gid)
        duration_ms, hashes = _build_merged_wav(
            gid=gid,
            member_ids=[m["id"] for m in members],
            silence_ms=int(patch["inter_segment_silence_ms"]),
        )
        import json as _json
        patch["member_hashes_json"] = _json.dumps(hashes, sort_keys=True)
        patch["merged_duration_ms"] = duration_ms
        patch["is_stale"] = 0

    updated = capture_groups_store.update_group(gid, patch)
    return JSONResponse({"group": updated})


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
    import json as _json
    g = capture_groups_store.get_group(gid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "group not found")
    if not user.get("is_admin") and g["user_id"] != user.get("user_id"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not your group")
    members = capture_groups_store.get_members(gid)
    duration_ms, hashes = _build_merged_wav(
        gid=gid,
        member_ids=[m["id"] for m in members],
        silence_ms=g["inter_segment_silence_ms"],
    )
    updated = capture_groups_store.update_group(gid, {
        "is_stale": 0,
        "merged_duration_ms": duration_ms,
        "member_hashes_json": _json.dumps(hashes, sort_keys=True),
    })
    return JSONResponse({"group": updated})


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


@router.get(
    "/captures/api/groups/{gid}/audio",
    dependencies=[Depends(require_admin_host)],
)
async def get_group_audio_api(
    gid: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
):
    """Stream the merged WAV. Same Range-aware handler shape as the
    per-capture audio endpoint."""
    import capture_groups_store
    g = capture_groups_store.get_group(gid)
    if g is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "group not found")
    if not user.get("is_admin") and g["user_id"] != user.get("user_id"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not your group")
    try:
        abs_p = capture_groups_store.abs_path_for(g["merged_wav_relpath"])
    except ValueError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "merged audio missing")
    if not os.path.exists(abs_p):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "merged audio missing")
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

def _build_export_stream(only_status: str | None, include_audio: bool):
    """Generator that yields tar.gz bytes containing manifest.jsonl and,
    optionally, audio/<id>.wav entries. One manifest entry per training
    unit — a "unit" is either a capture group (≤28 s packed sample) OR
    an ungrouped singleton capture. Group members never appear as
    singletons (no double-counting; the group transcript covers them)."""
    import capture_groups_store

    buf = io.BytesIO()
    tar = tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6)
    manifest_lines: list[bytes] = []

    # 1. Capture groups (the packed-for-fine-tune training samples).
    user_filter_scope: str | None = None  # admin-only path; no per-user scope
    for g in capture_groups_store.list_groups(user_id=user_filter_scope):
        text = (g.get("transcript") or "").strip()
        if not text:
            continue
        gid = g["id"]
        audio_name = f"audio/{gid}.wav"
        manifest_lines.append(json.dumps({
            "audio_filepath": audio_name,
            "text": text,
            "language": "",
            "duration": float(g.get("merged_duration_ms") or 0) / 1000.0,
            "source": "group",
            "user_id": g.get("user_id") or "",
            "is_stale": bool(g.get("is_stale")),
            "is_locked": bool(g.get("is_locked")),
            "member_count":
                len(capture_groups_store.get_members(gid)),
            "created_ts": float(g.get("created_ts") or 0.0),
        }, ensure_ascii=False).encode("utf-8") + b"\n")

        if include_audio:
            try:
                abs_p = capture_groups_store.abs_path_for(g["merged_wav_relpath"])
            except ValueError:
                continue
            if not os.path.isfile(abs_p):
                continue
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
        cid = row["id"]
        text = row.get("corrected_text") or row.get("final") or ""
        if not text.strip():
            continue
        ext = (row.get("audio_format") or "bin").lower().lstrip(".")
        audio_name = f"audio/{cid}.{ext}"
        manifest_lines.append(json.dumps({
            "audio_filepath": audio_name,
            "text": text,
            "language": row.get("language") or "",
            "duration": float(row.get("duration_seconds") or 0.0),
            "model": row.get("model") or "",
            "corrections": row.get("corrections") or [],
            "admin_notes": row.get("admin_notes") or "",
            "status": row.get("status") or "",
            "source": "singleton",
            "user_id": row.get("user_id") or "",
            "created_ts": float(row.get("created_ts") or 0.0),
            "request_id": row.get("request_id") or "",
        }, ensure_ascii=False).encode("utf-8") + b"\n")

        if include_audio:
            try:
                abs_p = captures_store.abs_audio_path(row["audio_relpath"])
            except ValueError:
                continue
            if not os.path.isfile(abs_p):
                continue
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
    width: 100%; min-height: 5rem; resize: vertical;
    background: var(--input-bg); color: var(--bold);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.5rem 0.6rem; font-family: var(--font-mono);
    font-size: var(--fs-md); margin-top: 0.25rem;
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
    <label>status
      <select id="filt-status">
        <option value="all">all</option>
        <option value="new" selected>new</option>
        <option value="reviewed">reviewed</option>
        <option value="ready">ready</option>
        <option value="dismissed">dismissed</option>
        <option value="audio_missing">audio_missing</option>
      </select>
    </label>
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
          <option value="newline">newline</option>
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
      Transcript (edit before commit; <span class="seam-marker">¶</span>
      markers show segment boundaries):
    </p>
    <textarea id="merge-transcript" spellcheck="false"></textarea>
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

  // ---- Admin-only landing (shared pattern across admin pages) ----
  function _renderNotAdminLanding() {
    document.body.classList.remove('role-admin');
    var main = document.getElementsByTagName('main')[0];
    if (!main) return;
    main.innerHTML =
      '<div style="max-width:36rem;margin:4rem auto;text-align:center;'
      + 'padding:2rem;background:var(--panel);border:1px solid var(--border);'
      + 'border-radius:6px;">'
      + '<h2 style="margin:0 0 0.5rem;color:var(--bold);">Admin only</h2>'
      + '<p style="color:var(--help);">This page requires an admin API key. '
      + 'Sign in with an admin key or go to your personal page.</p>'
      + '<p style="margin-top:1.2rem;">'
      + '<a href="/quick-config" style="color:var(--cyan);'
      + 'border:1px solid var(--cyan);padding:0.45rem 1rem;'
      + 'border-radius:4px;text-decoration:none;">Open /quick-config</a> '
      + '<button onclick="sessionStorage.removeItem(\'whisper_api_key\');'
      + 'location.reload()" style="margin-left:0.5rem;">Sign out</button>'
      + '</p></div>';
  }
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
  // State
  // -------------------------------------------------------------------
  var _allCaptures = [];
  var _allGroups = [];
  var _counts = {};
  var _openRows = {};   // cid -> { audio, blobUrl, wordEls, words, finalText, dirty, corrections, ... }
  var _selection = new Set();   // capture ids currently selected for merge
  var _lastSelectId = null;     // anchor for shift-range select
  var _isAdmin = false;
  var _selfUserId = null;

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

  function relTime(ts) {
    if (!ts) return '—';
    var sec = Math.max(0, (Date.now() / 1000) - ts);
    if (sec < 5) return 'just now';
    if (sec < 60) return Math.floor(sec) + 's ago';
    if (sec < 3600) return Math.floor(sec / 60) + ' min ago';
    if (sec < 86400) return Math.floor(sec / 3600) + ' h ago';
    return new Date(ts * 1000).toLocaleString();
  }
  function fmtDate(ts) {
    if (!ts) return '';
    return new Date(ts * 1000).toLocaleString();
  }
  function escapeHtml(s) {
    var d = document.createElement('div');
    d.textContent = s == null ? '' : String(s);
    return d.innerHTML;
  }

  function applyFilters(rows) {
    var s = document.getElementById('filt-status').value;
    var m = document.getElementById('filt-model').value;
    var q = (document.getElementById('filt-search').value || '').trim().toLowerCase();
    return rows.filter(function(r) {
      if (s !== 'all' && r.status !== s) return false;
      if (m !== 'all' && r.model !== m) return false;
      if (!q) return true;
      var hay = (
        (r.raw || '') + ' ' + (r.final || '') + ' ' +
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
      '<span class="when" title="' + escapeHtml(fmtDate(r.created_ts)) + '">' +
        escapeHtml(relTime(r.created_ts)) + '</span>' +
      '<span class="pill status-' + escapeHtml(r.status || 'new') + '">' +
        escapeHtml(r.status || 'new') + '</span>' +
      (r.model ? '<span class="pill">' + escapeHtml(r.model) + '</span>' : '') +
      (r.language ? '<span class="pill">' + escapeHtml(r.language) + '</span>' : '') +
      '<span class="duration">' + (r.duration_seconds || 0).toFixed(1) + 's</span>' +
      (r.request_id
        ? '<span class="req">req ' + escapeHtml((r.request_id||'').slice(0,8)) + '</span>'
        : '') +
      (r.user_id
        ? '<span class="pill" title="speaker">' + escapeHtml((r.user_id||'').slice(0,6)) + '</span>'
        : '') +
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
    preview.textContent = r.corrected_text || r.final || r.raw || '(empty)';
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
  }

  function buildBody(body, r) {
    body.innerHTML = '';
    var state = {
      cid: r.id,
      audio: null,
      blobUrl: null,
      words: r.words || [],
      finalText: r.final || '',
      corrections: (r.corrections || []).map(function(c) {
        return {
          wrong: c.wrong || '',
          correct: c.correct || '',
          idx: typeof c.idx === 'number' ? c.idx : null,
          idx_end: typeof c.idx_end === 'number' ? c.idx_end : null,
        };
      }),
      correctedText: r.corrected_text || '',
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
      state.blobUrl = URL.createObjectURL(blob);
      audio.src = state.blobUrl;
    }).catch(function(e) {
      if (e && e.message !== 'unauthorized') {
        toast('Audio load failed: ' + e.message, true);
      }
    });

    // --- word strip ---
    var stripWrap = document.createElement('div');
    stripWrap.className = 'cc-section';
    stripWrap.innerHTML = '<h3>Words (karaoke — click to seek; '
      + 'shift-click another to mark a multi-word span; click a marked '
      + 'word to clear)</h3>';
    var strip = document.createElement('div');
    strip.className = 'word-strip';
    state.words.forEach(function(w, i) {
      var sp = document.createElement('span');
      sp.className = 'word';
      sp.dataset.idx = String(i);
      sp.dataset.start = String(w.start || 0);
      sp.dataset.end = String(w.end || 0);
      sp.textContent = (w.word || '').replace(/^\s+/, ' ');
      sp.addEventListener('click', function(e) {
        onWordClick(state, i, !!e.shiftKey);
      });
      strip.appendChild(sp);
      state.wordEls.push(sp);
    });
    stripWrap.appendChild(strip);
    body.appendChild(stripWrap);

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
    body.appendChild(textLine('raw',   'raw',   r.raw));
    body.appendChild(textLine('final', 'final', r.final));

    // --- corrections section ---
    var corrSec = document.createElement('div');
    corrSec.className = 'cc-section';
    corrSec.innerHTML = '<h3>Word corrections</h3>'
      + '<div class="help">Click a word above to mark it. Shift-click '
      + 'another to extend the range. Type the corrected text in the chip.</div>';
    var chipBox = document.createElement('div');
    chipBox.className = 'cc-corrections';
    corrSec.appendChild(chipBox);
    body.appendChild(corrSec);
    state.chipBox = chipBox;
    renderChips(state);

    // --- ground truth (corrected_text) ---
    var gtSec = document.createElement('div');
    gtSec.className = 'cc-section';
    gtSec.innerHTML = '<h3>Ground truth (exported text)</h3>'
      + '<div class="help">This is the string Whisper learns to emit '
      + 'for this audio. Defaults to the post-pipeline <em>final</em> text; '
      + 'edit freely. <button class="apply-corr">Apply corrections → here</button></div>';
    var gtArea = document.createElement('textarea');
    gtArea.className = 'cc-ground';
    gtArea.value = state.correctedText || state.finalText;
    gtArea.addEventListener('input', function() {
      state.correctedText = gtArea.value;
      markDirty(state);
    });
    gtSec.appendChild(gtArea);
    body.appendChild(gtSec);
    state.gtArea = gtArea;

    gtSec.querySelector('.apply-corr').addEventListener('click', function(e) {
      e.preventDefault();
      applyCorrectionsToGround(state);
    });

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
    var statusLbl = document.createElement('label');
    statusLbl.textContent = 'status';
    var statusSel = document.createElement('select');
    ['new', 'reviewed', 'ready', 'dismissed'].forEach(function(v) {
      var o = document.createElement('option');
      o.value = v; o.textContent = v;
      if (v === state.newStatus) o.selected = true;
      statusSel.appendChild(o);
    });
    statusSel.addEventListener('change', function() {
      state.newStatus = statusSel.value;
      markDirty(state);
    });
    statusLbl.appendChild(statusSel);
    actions.appendChild(statusLbl);

    var dirty = document.createElement('span');
    dirty.className = 'dirty hidden';
    dirty.textContent = 'unsaved';
    actions.appendChild(dirty);
    state.dirtyEl = dirty;

    var spc = document.createElement('span');
    spc.className = 'spacer';
    actions.appendChild(spc);

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
  }

  function removeChip(state, i) {
    var c = state.corrections[i];
    if (c && typeof c.idx === 'number') {
      var end = (typeof c.idx_end === 'number') ? c.idx_end : c.idx;
      for (var j = c.idx; j <= end; j++) selectWord(state, j, false);
    }
    state.corrections.splice(i, 1);
    renderChips(state);
    markDirty(state);
  }

  function renderChips(state) {
    var box = state.chipBox;
    box.innerHTML = '';
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
        markDirty(state);
      });
      chip.appendChild(inp);

      var rm = document.createElement('button');
      rm.className = 'remove';
      rm.type = 'button';
      rm.textContent = '×';
      rm.addEventListener('click', function() { removeChip(state, i); });
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
    state.correctedText = out;
    state.gtArea.value = out;
    markDirty(state);
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
        corrected_text: state.correctedText,
        corrections: corrections,
        admin_notes: state.adminNotes,
      };
      var j = await api('PATCH',
        '/captures/api/' + encodeURIComponent(state.cid), body);
      Object.assign(r, j.capture);
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
        // Drop any open blob URLs first
        Object.keys(_openRows).forEach(function(cid) {
          var st = _openRows[cid];
          if (st && st.blobUrl) URL.revokeObjectURL(st.blobUrl);
        });
        _openRows = {};
        await load();
      } catch (e) {
        if (e.message !== 'unauthorized') toast('Failed: ' + e.message, true);
      }
    };
  }

  // -------------------------------------------------------------------
  // Export
  // -------------------------------------------------------------------
  function onExport() {
    var tok = getToken();
    if (!tok && hasAdminToken) {
      showTokenModal(function() { onExport(); });
      return;
    }
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
  // Render loop
  // -------------------------------------------------------------------
  function render() {
    var rows = applyFilters(_allCaptures);
    var list = document.getElementById('list');
    // Preserve open-state across re-renders: keep DOM nodes for open
    // cards intact, rebuild closed ones.
    var openIds = Object.keys(_openRows);
    list.innerHTML = '';
    if (rows.length === 0) {
      var empty = document.createElement('div');
      empty.className = 'empty-state';
      if (_allCaptures.length === 0) {
        empty.innerHTML = '<strong>No captures yet.</strong><br>'
          + 'Enable <em>CAPTURE_RECORDINGS_ENABLED</em> in /config and send '
          + 'transcription requests; eligible ones land here for review.'
          + '<div class="help-doc">Each capture stores the original audio + '
          + 'word-level timestamps + the model\'s raw and post-pipeline '
          + 'text. Review karaoke-style, correct mistranscriptions, mark as '
          + '<strong>ready</strong>, then <strong>Export ready</strong> to '
          + 'get a HuggingFace-compatible <code>manifest.jsonl + audio/</code> '
          + 'tar.gz.</div>';
      } else {
        empty.innerHTML = 'No captures match the current filters.';
      }
      list.appendChild(empty);
      return;
    }
    rows.forEach(function(r) {
      list.appendChild(renderCard(r));
    });
    // Reopen any rows that were open before re-render
    openIds.forEach(function(cid) {
      var card = list.querySelector('.capture-card[data-id="' + cid + '"]');
      if (card) {
        // Mark open again; lazy-build will skip if data already present
        // (but on re-fetch we want fresh data, so always rebuild).
        var r = _allCaptures.find(function(x) { return x.id === cid; });
        if (r) toggleExpand(card, r);
      } else {
        // Row disappeared from the filter — drop its open state.
        var st = _openRows[cid];
        if (st && st.blobUrl) URL.revokeObjectURL(st.blobUrl);
        delete _openRows[cid];
      }
    });
  }

  // -------------------------------------------------------------------
  // Load
  // -------------------------------------------------------------------
  var hasAdminToken = false;

  async function load() {
    try {
      var j = await api('GET', '/captures/api/list?status=all&limit=500');
      _allCaptures = j.captures || [];
      _counts = j.counts || {};
      _isAdmin = !!j.is_admin;
      _selfUserId = j.user_id || null;
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
      if (_isAdmin) document.body.classList.add('role-admin');
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
  var JOIN_STR = { space: ' ', period_space: '. ', newline: '\n' };
  function _seamMarker() { return '¶'; }   // ¶

  function _buildDefaultTranscript(rows, strategy) {
    var sep = JOIN_STR[strategy] || ' ';
    return rows.map(function(r) {
      return (r.corrected_text || r.final || r.raw || '').trim();
    }).filter(Boolean).join(sep);
  }

  function _renderMergePreview(rows) {
    var el = document.getElementById('merge-members');
    el.innerHTML = '';
    rows.forEach(function(r, i) {
      var line = document.createElement('div');
      line.className = 'seg-line';
      line.innerHTML = '<span class="seg-time">[' + (i + 1) + '] '
        + (r.duration_seconds || 0).toFixed(1) + 's</span>'
        + escapeHtml((r.corrected_text || r.final || r.raw || '').slice(0, 200));
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
      ta.value = _buildDefaultTranscript(rows, joinSel.value);
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
      var payload = {
        member_ids: rows.map(function(r) { return r.id; }),
        transcript: ta.value || '',
        join_strategy: joinSel.value,
        silence_ms: parseInt(silSel.value, 10) || 300,
      };
      api('POST', '/captures/api/groups', payload)
        .then(function() {
          modal.classList.remove('show');
          _selection.clear();
          _lastSelectId = null;
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
      '<span class="when">' + escapeHtml(relTime(g.created_ts)) + '</span>' +
      '<span class="pill group-pill">group</span>' +
      '<span class="pill" title="speaker">' +
        escapeHtml((g.user_id || '?').slice(0, 6)) + '</span>' +
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
    api('GET', '/captures/api/groups/' + encodeURIComponent(g.id))
      .then(function(j) {
        var detail = j.group || g;
        var audio = document.createElement('audio');
        audio.controls = true;
        audio.style.marginTop = '0.5rem';
        api('GET', '/captures/api/groups/' + encodeURIComponent(g.id) + '/audio')
          .catch(function() { return null; });
        audio.src = '/captures/api/groups/' + encodeURIComponent(g.id) + '/audio?t='
          + encodeURIComponent(getToken());
        // Use Authorization header by re-fetching as blob to avoid token in URL.
        audio.removeAttribute('src');
        fetch('/captures/api/groups/' + encodeURIComponent(g.id) + '/audio',
              { headers: { Authorization: 'Bearer ' + getToken() } })
          .then(function(r) { return r.blob(); })
          .then(function(b) { audio.src = URL.createObjectURL(b); })
          .catch(function(_) {});
        body.appendChild(audio);

        var ta = document.createElement('textarea');
        ta.value = detail.transcript || '';
        ta.style.cssText = 'width:100%;min-height:6rem;margin-top:0.5rem;'
          + 'background:var(--input-bg);color:var(--fg);'
          + 'border:1px solid var(--border);border-radius:4px;'
          + 'padding:0.5rem;font-family:var(--font-mono);'
          + 'font-size:var(--fs-md);box-sizing:border-box;';
        body.appendChild(ta);

        var btnRow = document.createElement('div');
        btnRow.style.cssText = 'display:flex;gap:0.5rem;margin-top:0.5rem;flex-wrap:wrap;';
        var saveTBtn = document.createElement('button');
        saveTBtn.textContent = 'Save transcript';
        saveTBtn.className = 'primary';
        saveTBtn.onclick = function() {
          api('PATCH', '/captures/api/groups/' + encodeURIComponent(g.id),
              { transcript: ta.value })
            .then(function() { toast('Saved'); return load(); })
            .catch(function(e) { if (e.message !== 'unauthorized') toast(e.message, true); });
        };
        btnRow.appendChild(saveTBtn);
        if (detail.is_stale) {
          var regenBtn = document.createElement('button');
          regenBtn.textContent = 'Regenerate audio (clear stale)';
          regenBtn.onclick = function() {
            api('POST', '/captures/api/groups/' + encodeURIComponent(g.id) + '/regenerate')
              .then(function() { toast('Regenerated'); return load(); })
              .catch(function(e) { if (e.message !== 'unauthorized') toast(e.message, true); });
          };
          btnRow.appendChild(regenBtn);
        }
        var lockBtn = document.createElement('button');
        lockBtn.textContent = detail.is_locked ? 'Unlock' : 'Lock';
        lockBtn.onclick = function() {
          api('PATCH', '/captures/api/groups/' + encodeURIComponent(g.id),
              { is_locked: !detail.is_locked })
            .then(function() { toast('Updated'); return load(); })
            .catch(function(e) { if (e.message !== 'unauthorized') toast(e.message, true); });
        };
        btnRow.appendChild(lockBtn);

        if (!detail.is_locked) {
          var dissolveBtn = document.createElement('button');
          dissolveBtn.textContent = 'Dissolve group';
          dissolveBtn.className = 'danger';
          dissolveBtn.onclick = function() {
            if (!confirm('Dissolve this group? Members return to the flat list; merged WAV is unlinked.'))
              return;
            api('DELETE', '/captures/api/groups/' + encodeURIComponent(g.id))
              .then(function() { toast('Dissolved'); return load(); })
              .catch(function(e) { if (e.message !== 'unauthorized') toast(e.message, true); });
          };
          btnRow.appendChild(dissolveBtn);
        }
        body.appendChild(btnRow);

        // Member list
        var membersDiv = document.createElement('div');
        membersDiv.className = 'group-members';
        membersDiv.innerHTML = '<p style="font-size:var(--fs-sm);color:var(--help);margin:0.4rem 0;">Members (' + (detail.members||[]).length + ')</p>';
        (detail.members || []).forEach(function(m) {
          var line = document.createElement('div');
          line.style.cssText = 'font-size:var(--fs-sm);color:var(--dim);padding:0.2rem 0;';
          line.innerHTML = '[' + (m.group_order + 1) + '] '
            + (m.duration_seconds || 0).toFixed(1) + 's · '
            + escapeHtml((m.corrected_text || m.final || m.raw || '').slice(0, 120));
          membersDiv.appendChild(line);
        });
        body.appendChild(membersDiv);

        body.dataset.built = '1';
      })
      .catch(function(e) {
        card.classList.remove('open');
        if (e && e.message !== 'unauthorized') toast(e.message, true);
      });
  }

  // Override render() to also draw group cards interleaved by created_ts.
  var _originalRender = render;
  render = function() {
    var rows = applyFilters(_allCaptures);
    var list = document.getElementById('list');
    var openIds = Object.keys(_openRows);
    list.innerHTML = '';
    // Build a merged timeline: ungrouped captures + group cards (members
    // are nested inside group cards, so we exclude them from the flat list).
    var groupsById = {};
    _allGroups.forEach(function(g) { groupsById[g.id] = g; });
    var ungrouped = rows.filter(function(r) { return !r.group_id; });
    var combined = ungrouped.map(function(r) {
      return { kind: 'capture', ts: r.created_ts || 0, data: r };
    }).concat(_allGroups.map(function(g) {
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
  document.getElementById('filt-status').addEventListener('change', render);
  document.getElementById('filt-model').addEventListener('change', render);
  document.getElementById('filt-search').addEventListener('input', render);
  document.getElementById('btn-refresh').addEventListener('click', load);
  document.getElementById('btn-export').addEventListener('click', onExport);
  document.getElementById('btn-clear').addEventListener('click', onClearAll);
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
  });

  load();
})();
</script>
</body>
</html>
"""
