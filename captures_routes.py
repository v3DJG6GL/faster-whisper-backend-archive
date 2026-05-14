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
    dependencies=[
        Depends(require_admin_host),
        Depends(require_admin),
    ],
)
async def list_captures_api(
    status_filter: str = Query("all", alias="status"),
    limit: int = Query(200, ge=1, le=1000),
    before_ts: float | None = Query(None),
) -> JSONResponse:
    rows = captures_store.list_captures(
        status=status_filter, limit=limit, before_ts=before_ts,
    )
    return JSONResponse({
        "captures": rows,
        "counts": captures_store.counts_by_status(),
        "enabled": bool(getattr(cfg, "CAPTURE_RECORDINGS_ENABLED", False)),
        "retention_days": int(getattr(cfg, "CAPTURES_RETENTION_DAYS", 0)),
        "total_count": captures_store.count(),
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
# Export: streamed tar.gz
# ---------------------------------------------------------------------

def _build_export_stream(only_status: str | None, include_audio: bool):
    """Generator that yields tar.gz bytes containing manifest.jsonl and,
    optionally, audio/<cid>.<ext> entries. Streams row-by-row so a
    multi-GB export doesn't load anything but one row's audio into RAM
    at a time."""
    buf = io.BytesIO()
    tar = tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6)

    manifest_lines: list[bytes] = []

    for row in captures_store.iter_captures_for_export(status=only_status):
        cid = row["id"]
        text = row.get("corrected_text") or row.get("final") or ""
        # Skip rows with no usable text. Whisper fine-tuning on empty
        # targets produces garbage; better to omit than ship trash.
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
            # Flush so the consumer sees progress instead of buffering
            # the whole tar in RAM. Trick: read out what's been written
            # so far and yield it; tarfile keeps its own write head.
            chunk = buf.getvalue()
            buf.seek(0); buf.truncate()
            if chunk:
                yield chunk

    # Manifest last so we know the row count it covers (and entries are
    # written in the same order as audio files when present).
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
    border-radius: 6px; padding: 1.25rem; min-width: 22rem; max-width: 32rem;
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

  <div id="list"></div>
</main>

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
    <h3>Admin token</h3>
    <p>Bearer token for /captures/api endpoints. Stored in sessionStorage
       until tab close.</p>
    <input id="token-input" type="password" autocomplete="off" placeholder="paste token">
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
      showTokenModal(function() {});
      throw new Error('unauthorized');
    }
    if (!resp.ok) {
      var msg = 'HTTP ' + resp.status;
      try { var j = await resp.json(); if (j && j.detail) msg = j.detail; }
      catch(_) {}
      throw new Error(msg);
    }
    return await resp.json();
  }

  // -------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------
  var _allCaptures = [];
  var _counts = {};
  var _openRows = {};   // cid -> { audio, blobUrl, wordEls, words, finalText, dirty, corrections, ... }

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

    var head = document.createElement('div');
    head.className = 'cc-head';
    head.innerHTML =
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
      updateCaptureBadge(!!j.enabled);
      rebuildModelFilter();
      updateCounts();
      render();
      document.body.classList.add('role-admin');
    } catch (e) {
      if (e.message !== 'unauthorized') {
        toast('Failed to load captures: ' + e.message, true);
      }
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
  // Wire up
  // -------------------------------------------------------------------
  document.getElementById('filt-status').addEventListener('change', render);
  document.getElementById('filt-model').addEventListener('change', render);
  document.getElementById('filt-search').addEventListener('input', render);
  document.getElementById('btn-refresh').addEventListener('click', load);
  document.getElementById('btn-export').addEventListener('click', onExport);
  document.getElementById('btn-clear').addEventListener('click', onClearAll);

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
