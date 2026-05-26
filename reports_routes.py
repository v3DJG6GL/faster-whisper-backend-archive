"""Transcription error report submission + admin triage.

Two surfaces:

  /quick-config/reports/api/*       — end-user routes. Gated by
    require_admin_host + get_current_user. The form + delete control
    live inline on each .trace-item in /quick-config:
    POST   /quick-config/reports/api/submit                — receiver
    DELETE /quick-config/reports/api/by-request/{request_id} — caller
                                                              undoes
                                                              own row

  /reports                          — admin page + APIs:
    GET   /reports                  HTML triage page
    GET   /reports/api/list         all reports (newest first)
    PATCH /reports/api/{rid}        status + admin_notes
    DELETE /reports/api/{rid}       single delete
    POST  /reports/api/clear        wipe all (confirm dialog)
    GET   /reports/api/export       full JSON dump (envelope-wrapped)
  Mutating routes use Depends(require_admin) — admin-only API keys.

Per-user rate limit on submission: in-memory fixed-window counter
keyed on the resolved user_id from the API key, falling back to
request.client.host when no user_id is present.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

import api_keys_store
import config as cfg
import reports_store
import web_common
from admin_routes import require_admin_host
from auth import get_current_user, require_admin, require_page

router = APIRouter()


# ---------------------------------------------------------------------
# Submission payload
# ---------------------------------------------------------------------

class CorrectionIn(BaseModel):
    model_config = {"extra": "forbid"}
    wrong: str = ""
    correct: str = ""
    idx: int | None = None
    # Inclusive end-of-range for multi-word selections. Omitted (or equal
    # to idx) for single-word corrections. Validation happens server-side
    # in reports_store._clean_corrections.
    idx_end: int | None = None


class ReportSubmitIn(BaseModel):
    model_config = {"extra": "forbid"}
    trace_ts: float = 0.0
    request_id: str | None = None
    model: str = Field(default="", max_length=256)
    raw: str = ""
    final: str = ""
    steps: list[Any] = []
    corrections: list[CorrectionIn] = []
    intended_text: str = ""
    user_comment: str = ""


# ---------------------------------------------------------------------
# Rate limit (per reporter — keyed on user_id, or host fallback)
# ---------------------------------------------------------------------
# Fixed-window counter, reset on roll. ~15 LOC, no third-party dep. The
# threat model is "accidental double-click / runaway script", not a
# motivated attacker — for that the LAN box is already locked down by
# require_admin_host.

_RATE_WINDOW_S = 600.0
_RATE_MAX = 20
_rate: dict[str, tuple[int, float]] = {}


def _check_rate_limit(key: str) -> None:
    key = key or "<unknown>"
    now = time.time()
    n, start = _rate.get(key, (0, now))
    if now - start > _RATE_WINDOW_S:
        n, start = 0, now
    n += 1
    _rate[key] = (n, start)
    if n > _RATE_MAX:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many reports. Try again in a few minutes.",
        )


# ---------------------------------------------------------------------
# Submission endpoint (under /quick-config)
# ---------------------------------------------------------------------

@router.post(
    "/quick-config/reports/api/submit",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_page("quick_config")),
    ],
)
async def submit_report(
    payload: ReportSubmitIn,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    is_admin = bool(user.get("is_admin"))
    if not getattr(cfg, "REPORTS_ALLOW_USER_SUBMIT", True) and not is_admin:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Report submission is disabled by the admin.",
        )

    rate_key = user.get("user_id") or (
        request.client.host if request.client else ""
    )
    _check_rate_limit(rate_key)

    intended = (payload.intended_text or "").strip()
    comment = (payload.user_comment or "").strip()
    corrections = reports_store._clean_corrections(
        [c.model_dump() for c in (payload.corrections or [])]
    )

    if not corrections and not intended and not comment:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Mark a wrong word, write what you meant to say, or leave a comment.",
        )

    host = request.client.host if request.client else ""
    rid, was_updated = reports_store.upsert_report(
        user_id=user.get("user_id"),
        request_id=payload.request_id,
        trace_ts=float(payload.trace_ts or 0.0),
        model=payload.model or "",
        raw=payload.raw or "",
        final=payload.final or "",
        steps=list(payload.steps or []),
        corrections=corrections,
        intended_text=intended,
        user_comment=comment,
        reporter_role="admin" if is_admin else "user",
        reporter_host=host,
    )

    # Apply the report's corrections as visible chip corrections on
    # every capture matching this report's request_id. The chips show
    # up at /captures and are re-editable like any other chip — i.e.
    # the report effectively becomes a per-capture correction, NOT a
    # global PIPELINE_RULES change. This keeps the shared pipeline
    # untouched (important since /quick-config is user-accessible) and
    # the admin still gets a single-action way to "promote a report
    # into a fix" by just clicking Submit.
    captures_updated = 0
    if corrections and payload.request_id:
        import captures_store
        import text_corrections
        matches = captures_store.find_by_request_id(payload.request_id)
        for cap in matches:
            existing = cap.get("corrections") or []
            merged = text_corrections.three_way_merge_corrections(
                baseline=[], edited=corrections, current=existing,
            )
            if merged != existing:
                captures_store.update_capture(
                    cap["id"], {"corrections": merged},
                )
                captures_updated += 1

    return JSONResponse({
        "ok": True,
        "id": rid,
        "was_updated": was_updated,
        "captures_updated": captures_updated,
    })


# ---------------------------------------------------------------------
# Admin page + APIs
# ---------------------------------------------------------------------

@router.get(
    "/reports",
    # HTML page is host-only — the login modal runs in this page's
    # own JS, so the bearer isn't available on the initial navigation.
    # API endpoints below gate by `require_page("reports")`; if the
    # user lacks access, the first list-fetch 403s and the JS renders
    # a "no access" landing.
    dependencies=[Depends(require_admin_host)],
    response_class=HTMLResponse,
)
async def reports_page() -> HTMLResponse:
    if not getattr(cfg, "ADMIN_UI_ENABLED", False):
        return HTMLResponse("Admin UI disabled.", status_code=404)
    return HTMLResponse(
        web_common.render_page(_REPORTS_HTML, current="reports"),
        media_type="text/html",
    )


class PatchReportIn(BaseModel):
    model_config = {"extra": "forbid"}
    status: Literal["open", "resolved", "dismissed"] | None = None
    admin_notes: str | None = None


@router.get(
    "/reports/api/list",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_page("reports")),
    ],
)
async def list_reports_api(
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Scope-aware report list. `scope=own` users see only their own
    reports; `scope=all` users (incl. admins) see every report. Closes
    the previous "list_reports returns ALL rows" leak the moment non-
    admins can reach the page."""
    perms = user["permissions"]
    caller_uid = user.get("user_id") or ""
    effective_user = perms.effective_user_id_for("reports", caller_uid)
    rows = reports_store.list_reports(user_id=effective_user)
    usernames = api_keys_store.get_usernames(
        [r.get("user_id") for r in rows],
    )
    for r in rows:
        r["username"] = usernames.get(r.get("user_id"))
    return JSONResponse({
        "reports": rows,
        "counts": reports_store.counts_by_status(),
        "retention_days": int(getattr(cfg, "REPORTS_RETENTION_DAYS", 0)),
        "is_admin": bool(user.get("is_admin")),
        "scope": perms.scope("reports"),
    })


@router.patch(
    "/reports/api/{rid}",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_page("reports")),
    ],
)
async def patch_report_api(
    rid: str, payload: PatchReportIn,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Mark a report status/notes. `scope=own` users can edit only their
    own; `scope=all` users (incl. admins) can edit any."""
    existing = reports_store.get_report(rid)
    if existing is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "report not found")
    user["permissions"].assert_can_read_row(
        existing, "reports", user.get("user_id") or "",
    )
    patch: dict[str, Any] = {}
    if payload.status is not None:
        patch["status"] = payload.status
    if payload.admin_notes is not None:
        patch["admin_notes"] = payload.admin_notes
    try:
        updated = reports_store.update_report(rid, patch)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "report not found")
    return JSONResponse({"ok": True, "report": updated})


def _delete_report_and_cascade(report: dict[str, Any]) -> int:
    """Delete a single report row AND prune its corrections from every
    capture that currently carries them as chips.

    The prune is keyed on (idx, correct) and filters against the chips
    that OTHER surviving reports for the same request_id still claim —
    so if two reports both put `(idx=3, correct='Test,')` on a capture
    and we delete one of them, the chip stays put because the other
    report still asserts it.

    Returns the count of captures whose chip list shrank. The deleted
    report itself is gone from the DB on return."""
    import captures_store

    rid = report.get("id")
    request_id = report.get("request_id")
    report_chips = report.get("corrections") or []

    reports_store.delete_report(rid)

    # Compute "surviving" chips AFTER the delete: any concurrent submit
    # that landed during this handler is now visible and its chips are
    # protected from the prune. Without this re-snapshot order, a chip
    # from a freshly-submitted report can get stripped because we
    # snapshotted survivors before it became durable.
    survivor_keys: set[tuple[Any, Any]] = set()
    if request_id:
        for other in reports_store.list_reports_for_request_id(request_id):
            for c in (other.get("corrections") or []):
                if isinstance(c, dict):
                    survivor_keys.add((c.get("idx"), c.get("correct")))

    prune_set = [
        c for c in report_chips
        if isinstance(c, dict)
        and (c.get("idx"), c.get("correct")) not in survivor_keys
    ]

    if request_id and prune_set:
        return captures_store.prune_chips_for_request_id(
            request_id, prune_set,
        )
    return 0


@router.delete(
    "/quick-config/reports/api/by-request/{request_id}",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_page("quick_config")),
    ],
)
async def delete_my_report_api(
    request_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Delete the caller's report for a given request_id. One report
    per (user_id, request_id) is enforced by upsert_report, so this
    targets exactly the caller's row. Cascades chip cleanup on every
    capture sharing the request_id (filtered against surviving reports
    from other users, so we don't strip chips someone else still
    claims)."""
    # find_by_request_user returns None when user_id is falsy (e.g. an
    # unauthenticated caller); the 404 path below covers both that and
    # an authenticated caller whose user_id has no matching report row.
    # In open mode user_id is the literal "(open-mode)" sentinel — a
    # real value — so the query runs and matches the admin's own row.
    existing = reports_store.find_by_request_user(
        request_id, user.get("user_id"),
    )
    if not existing:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "no report to delete",
        )
    captures_cleaned = _delete_report_and_cascade(existing)
    return JSONResponse({"ok": True, "captures_cleaned": captures_cleaned})


@router.delete(
    "/reports/api/{rid}",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_page("reports")),
    ],
)
async def delete_report_api(
    rid: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """Delete a single report. `scope=own` users can delete only their
    own; `scope=all` users (incl. admins) can delete any. Bulk wipe is
    via /clear which stays admin-only."""
    report = reports_store.get_report(rid)
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "report not found")
    user["permissions"].assert_can_read_row(
        report, "reports", user.get("user_id") or "",
    )
    captures_cleaned = _delete_report_and_cascade(report)
    return JSONResponse({"ok": True, "captures_cleaned": captures_cleaned})


@router.post(
    "/reports/api/clear",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_admin),
    ],
)
async def clear_reports_api(request: Request) -> JSONResponse:
    import captures_store

    host = request.client.host if request.client else ""
    # Snapshot every report's (request_id, corrections) before the wipe
    # so the cascade has data to act on. No survivor-subtraction needed:
    # we're deleting EVERY report, so no other report can possibly still
    # claim a chip.
    all_reports = reports_store.list_reports()
    by_req: dict[str, list[dict[str, Any]]] = {}
    for r in all_reports:
        req_id = r.get("request_id")
        if not req_id:
            continue
        by_req.setdefault(req_id, []).extend(r.get("corrections") or [])

    n = reports_store.clear_all(reporter_host=host)

    captures_cleaned = 0
    for req_id, chips in by_req.items():
        captures_cleaned += captures_store.prune_chips_for_request_id(
            req_id, chips,
        )
    return JSONResponse({
        "ok": True, "deleted": n, "captures_cleaned": captures_cleaned,
    })


@router.get(
    "/reports/api/export",
    dependencies=[
        Depends(require_admin_host),
        Depends(require_admin),
    ],
)
async def export_reports_api() -> Response:
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "app": "faster-whisper-backend",
        "reports": reports_store.list_reports(),
    }
    blob = json.dumps(payload, ensure_ascii=False, indent=2)
    fname = f"whisper-reports-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    return Response(
        content=blob,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ---------------------------------------------------------------------
# /reports HTML page
# ---------------------------------------------------------------------
# Card list with status/model filter + free-text search. Word-correction
# chips render the wrong word red-strike + the correct word green.
# Sentence rewrite computes a word-level LCS diff between `final` and
# `intended_text` and renders deletions / insertions inline.
#
# IMPORTANT (CLAUDE memory note): never place a `{{...}}` placeholder
# inside a /* */, //, or <!-- --> comment — render_page() does a literal
# string replace and corrupting context kills the page. The placeholders
# below are all at HTML-element scope or inside <style>/<script> as bare
# tokens, never inside a comment.

_REPORTS_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{HEADER_TITLE}}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
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
  }
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
  .toolbar input[type="text"] { min-width: 16rem; }
  .toolbar .spacer { flex: 1; }
  .toolbar .counts { color: var(--help); font-size: var(--fs-sm);
    margin-right: 0.5rem; }
  .toolbar .counts .n { color: var(--bold); font-weight: 600; }
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

  .report-card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.75rem;
  }
  .rc-head {
    display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; align-items: center;
    font-size: var(--fs-sm); color: var(--help); margin-bottom: 0.5rem;
  }
  .rc-head .when { color: var(--bold); font-size: var(--fs-md); }
  .rc-head .pill {
    border: 1px solid var(--border); border-radius: 999px;
    padding: 0.05rem 0.5rem; font-family: var(--font-mono);
    font-size: var(--fs-xs);
  }
  .rc-head .pill.role-user  { color: var(--cyan); }
  .rc-head .pill.role-admin { color: var(--magenta); }
  .rc-head .req { font-family: var(--font-mono); color: var(--dim); }
  .rc-head .spacer { flex: 1; }
  .rc-head select {
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.15rem 0.35rem; font-size: var(--fs-sm);
  }
  .rc-head select.status-open      { color: var(--yellow); }
  .rc-head select.status-resolved  { color: var(--green); }
  .rc-head select.status-dismissed { color: var(--dim); }

  .rc-textline {
    display: grid; grid-template-columns: 4rem 1fr;
    gap: 0.5rem; padding: 0.15rem 0; align-items: baseline;
    font-family: var(--font-mono); font-size: var(--fs-md);
    word-break: break-word;
  }
  .rc-textline .tag {
    text-transform: lowercase; color: var(--help);
    font-size: var(--fs-xs); font-family: var(--font-sans);
  }
  .rc-textline.raw   .val { color: var(--fg); }
  .rc-textline.final .val { color: var(--bold); }
  .rc-textline .val.dim { color: var(--dim); font-style: italic; }

  details.rc-steps { margin: 0.375rem 0 0.5rem; }
  details.rc-steps summary {
    cursor: pointer; color: var(--help); font-size: var(--fs-sm);
  }
  .rc-step {
    display: grid; grid-template-columns: 12rem 1fr;
    gap: 0.5rem; padding: 0.1rem 0 0.1rem 1rem;
    font-family: var(--font-mono); font-size: var(--fs-sm);
  }
  .rc-step .step-label { color: var(--cyan); }
  .rc-step .step-before { color: var(--help); text-decoration: line-through; }
  .rc-step .step-after { color: var(--bold); }

  .rc-section {
    margin: 0.625rem 0 0.25rem; padding: 0.5rem 0.625rem;
    background: var(--input-bg); border: 1px solid var(--border);
    border-radius: 4px;
  }
  .rc-section h3 {
    font-size: var(--fs-sm); margin: 0 0 0.35rem; color: var(--help);
    font-family: var(--font-sans); font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .rc-corrections { display: flex; flex-wrap: wrap; gap: 0.375rem; }
  .rc-correction {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.15rem 0.5rem;
    font-family: var(--font-mono); font-size: var(--fs-md);
  }
  .rc-correction .wrong   { color: var(--red); text-decoration: line-through; }
  .rc-correction .arrow   { color: var(--dim); }
  .rc-correction .correct { color: var(--green); font-weight: 600; }

  .rc-diff {
    font-family: var(--font-mono); font-size: var(--fs-md);
    line-height: 1.5; word-break: break-word;
  }
  .rc-diff .diff-eq  { color: var(--fg); }
  .rc-diff .diff-del { color: var(--red); text-decoration: line-through;
    background: rgba(255, 123, 114, 0.08); padding: 0 0.15rem;
    border-radius: 2px; }
  .rc-diff .diff-ins { color: var(--green); font-weight: 600;
    background: rgba(126, 231, 135, 0.10); padding: 0 0.15rem;
    border-radius: 2px; }

  .rc-comment {
    font-family: var(--font-sans); font-size: var(--fs-md);
    color: var(--fg); white-space: pre-wrap;
  }

  .rc-notes textarea {
    width: 100%; min-height: 4.5rem; resize: vertical;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.4rem 0.55rem; font-size: var(--fs-md);
    font-family: var(--font-sans);
  }
  .rc-notes .row {
    display: flex; align-items: center; gap: 0.5rem;
    margin-top: 0.375rem;
  }
  .rc-notes .dirty {
    color: var(--yellow); font-size: var(--fs-sm); margin-left: auto;
  }
  .rc-notes .dirty.hidden { display: none; }

  .rc-actions {
    display: flex; flex-wrap: wrap; gap: 0.375rem;
    margin-top: 0.5rem;
  }

  /* Clear-all confirm modal — same pattern as token-modal in /quick-config */
  .modal {
    position: fixed; inset: 0; background: rgba(0,0,0,0.65);
    display: none; align-items: center; justify-content: center; z-index: 30;
  }
  .modal.show { display: flex; }
  .modal .box {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.25rem; min-width: 20rem;
    max-width: 30rem;
  }
  .modal h3 { margin: 0 0 0.5rem; color: var(--bold); font-size: var(--fs-xl); }
  .modal p { margin: 0.25rem 0; font-size: var(--fs-md); }
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

  /* API-key login modal — same layout as /captures and /config so the
   * three pages share a consistent look. */
  #token-modal {
    position: fixed; inset: 0; background: rgba(0,0,0,0.65);
    display: none; align-items: center; justify-content: center; z-index: 30;
  }
  #token-modal.show { display: flex; }
  #token-modal .box {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 1.4rem 1.5rem 1.2rem;
    width: 30rem; max-width: 92vw;
    box-shadow: 0 0.75rem 2.5rem rgba(0,0,0,0.5);
  }
  #token-modal h3 {
    margin: 0 0 0.5rem 0; color: var(--bold); font-size: var(--fs-xl);
  }
  #token-modal p {
    margin: 0 0 0.9rem 0; line-height: 1.45;
    color: var(--help); font-size: var(--fs-sm);
  }
  #token-modal p code { color: var(--fg); font-family: var(--font-mono); }
  #token-modal input {
    box-sizing: border-box; width: 100%;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.55rem 0.7rem; font-family: var(--font-mono);
    font-size: var(--fs-md); line-height: 1.4;
  }
  #token-modal input:focus { outline: none; border-color: var(--cyan); }
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

  {{NAV_CSS}}
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <span class="title">{{HEADER_TITLE}}</span>
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
        <option value="open" selected>open</option>
        <option value="resolved">resolved</option>
        <option value="dismissed">dismissed</option>
      </select>
    </label>
    <label>model
      <select id="filt-model">
        <option value="all">all</option>
      </select>
    </label>
    <label>search
      <input id="filt-search" type="text" placeholder="text in raw / final / corrections / comment / notes">
    </label>
    <span class="counts" id="counts"></span>
    <span class="spacer"></span>
    <button id="btn-refresh" title="Reload">Refresh</button>
    <button id="btn-export" title="Download all reports as JSON">Export</button>
    <button id="btn-clear" class="danger" title="Permanently delete every report">Clear all</button>
  </div>

  <div id="list"></div>
</main>

<div id="confirm-modal" class="modal">
  <div class="box">
    <h3>Clear all reports?</h3>
    <p id="confirm-msg">This permanently deletes every report. There is no undo.</p>
    <div class="actions">
      <button id="confirm-cancel">Cancel</button>
      <button id="confirm-ok" class="danger">Delete all</button>
    </div>
  </div>
</div>

<div id="token-modal">
  <div class="box">
    <h3>API key</h3>
    <p>Paste your <code>wk_…</code> admin API key. Stored in sessionStorage until tab close.</p>
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
  // Token (sessionStorage; matches the /quick-config rhythm)
  // -------------------------------------------------------------------
  var TOKEN_KEY = 'whisper_api_key';
  function getToken() {
    try { return sessionStorage.getItem(TOKEN_KEY) || ''; } catch(_) { return ''; }
  }
  function setToken(v) {
    try { sessionStorage.setItem(TOKEN_KEY, v || ''); } catch(_) {}
    // Notify the shared web_common chrome (_refreshAuthChrome in
    // OPEN_MODE_BANNER_JS) so the nav-link visibility updates without a
    // page reload.
    try { window.dispatchEvent(new Event('whisper:auth-changed')); } catch(_) {}
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
  // API helper
  // -------------------------------------------------------------------
  async function api(method, url, body) {
    var headers = { 'Content-Type': 'application/json' };
    var tok = getToken();
    if (tok) headers['Authorization'] = 'Bearer ' + tok;
    var resp = await fetch(url, {
      method: method,
      headers: headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (resp.status === 401) {
      // After the user pastes a key, re-run load() so the page actually
      // populates and body.role-admin gets added. The original "user
      // re-runs the action" comment meant they had to click Refresh
      // manually — but most users don't realise that, so the page stayed
      // stuck on toolbar-only + admin nav hidden.
      showTokenModal(function() { load(); });
      throw new Error('unauthorized');
    }
    if (resp.status === 403) {
      // Probe /auth/whoami to disambiguate "non-admin user" from a
      // server-side authz problem. On non-admin, swap <main> for the
      // Admin-only landing and throw a sentinel the load() catch can
      // suppress (no toast, no console noise).
      var rendered = await _renderAdminOnlyIfNonAdmin();
      if (rendered) throw new Error('not-admin');
    }
    if (!resp.ok) {
      var msg = 'HTTP ' + resp.status;
      try {
        var j = await resp.json();
        if (j && j.detail) msg = j.detail;
      } catch(_) {}
      throw new Error(msg);
    }
    return await resp.json();
  }

  {{NOT_ADMIN_LANDING_JS}}

  async function _renderAdminOnlyIfNonAdmin() {
    // Renamed-but-kept-for-compat: every 403 on this page means the
    // caller is a non-admin (the API gate is require_page("reports")
    // — a 403 means "valid bearer, no scope on /reports"). Render the
    // shared no-access landing slugged with the current page.
    try {
      var tok = getToken();
      var hdrs = tok ? { Authorization: 'Bearer ' + tok } : {};
      var r = await fetch('/auth/whoami', { headers: hdrs });
      if (r.ok) {
        var j = await r.json();
        // Cache whoami so _renderNoAccessLanding can list reachable pages.
        try { window.__whoami = j; } catch(_) {}
        if (j && j.is_admin === false) {
          _renderNoAccessLanding({ page: 'reports' });
          return true;
        }
      }
    } catch (_) {}
    return false;
  }

  // -------------------------------------------------------------------
  // Diff renderer — word-level LCS, hand-rolled. ~30 LOC.
  // -------------------------------------------------------------------
  // Tokenize preserving whitespace runs so the rendered diff reads
  // naturally. Each token is either a word (letters/digits/some Unicode
  // letters) or a whitespace run. Punctuation is a third class kept
  // separate so a "comma vs. period" change shows as a diff op too.
  var WORDISH = /[\\p{L}\\p{N}_]+/u;
  function tokenizeForDiff(s) {
    if (!s) return [];
    var out = [];
    var i = 0;
    while (i < s.length) {
      var ch = s.charCodeAt(i);
      if (ch === 32 || ch === 9 || ch === 10 || ch === 13) {
        var j = i;
        while (j < s.length) {
          var c = s.charCodeAt(j);
          if (c !== 32 && c !== 9 && c !== 10 && c !== 13) break;
          j++;
        }
        out.push(s.slice(i, j));
        i = j;
        continue;
      }
      var c = s[i];
      if (WORDISH.test(c)) {
        var k = i + 1;
        while (k < s.length && WORDISH.test(s[k])) k++;
        out.push(s.slice(i, k));
        i = k;
      } else {
        out.push(s[i]);
        i++;
      }
    }
    return out;
  }

  function lcsDiff(a, b) {
    // Compare token streams ignoring leading/trailing whitespace; treat
    // whitespace runs as their own equivalence class so we don't waste
    // diff ops on minor spacing changes.
    var n = a.length, m = b.length;
    var dp = new Array(n + 1);
    for (var i = 0; i <= n; i++) {
      dp[i] = new Int32Array(m + 1);
    }
    for (var i = n - 1; i >= 0; i--) {
      for (var j = m - 1; j >= 0; j--) {
        if (a[i] === b[j]) dp[i][j] = dp[i + 1][j + 1] + 1;
        else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
    var ops = [];
    var i = 0, j = 0;
    while (i < n && j < m) {
      if (a[i] === b[j]) {
        ops.push(['eq', a[i]]);
        i++; j++;
      } else if (dp[i + 1][j] >= dp[i][j + 1]) {
        ops.push(['del', a[i]]);
        i++;
      } else {
        ops.push(['ins', b[j]]);
        j++;
      }
    }
    while (i < n) { ops.push(['del', a[i++]]); }
    while (j < m) { ops.push(['ins', b[j++]]); }
    return ops;
  }

  function renderDiff(beforeStr, afterStr) {
    var a = tokenizeForDiff(beforeStr || '');
    var b = tokenizeForDiff(afterStr || '');
    var ops = lcsDiff(a, b);
    var div = document.createElement('div');
    div.className = 'rc-diff';
    ops.forEach(function(op) {
      var span = document.createElement('span');
      span.className = 'diff-' + op[0];
      span.textContent = op[1];
      div.appendChild(span);
    });
    return div;
  }

  // -------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------
  var _allReports = [];
  var _counts = { open: 0, resolved: 0, dismissed: 0 };

  // absTime / relTime / fmtWhen / timeTick are injected via TIME_HELPERS_JS.
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
        (r.raw || '') + ' ' +
        (r.final || '') + ' ' +
        (r.intended_text || '') + ' ' +
        (r.user_comment || '') + ' ' +
        (r.admin_notes || '') + ' ' +
        ((r.corrections || []).map(function(c) {
          return (c.wrong || '') + ' ' + (c.correct || '');
        }).join(' '))
      ).toLowerCase();
      return hay.indexOf(q) !== -1;
    });
  }

  function rebuildModelFilter() {
    var sel = document.getElementById('filt-model');
    var cur = sel.value;
    var seen = {};
    _allReports.forEach(function(r) { if (r.model) seen[r.model] = true; });
    var opts = ['<option value="all">all</option>'];
    Object.keys(seen).sort().forEach(function(m) {
      opts.push('<option value="' + escapeHtml(m) + '">' + escapeHtml(m) + '</option>');
    });
    sel.innerHTML = opts.join('');
    if (Object.prototype.hasOwnProperty.call(seen, cur) || cur === 'all') sel.value = cur;
  }

  function updateCounts() {
    var el = document.getElementById('counts');
    el.innerHTML =
      '<span class="n">' + _counts.open + '</span> open · ' +
      '<span class="n">' + _counts.resolved + '</span> resolved · ' +
      '<span class="n">' + _counts.dismissed + '</span> dismissed';
  }

  // -------------------------------------------------------------------
  // Card rendering
  // -------------------------------------------------------------------
  function renderCard(r) {
    var card = document.createElement('div');
    card.className = 'report-card';
    card.dataset.id = r.id;

    // Header row
    var head = document.createElement('div');
    head.className = 'rc-head';
    head.innerHTML =
      '<span class="when" data-ts="' + (r.created_ts || 0) + '" title="' +
        escapeHtml(absTime(r.created_ts)) + '">' +
        escapeHtml(fmtWhen(r.created_ts)) + '</span>' +
      '<span class="pill role-' + escapeHtml(r.reporter_role || 'user') + '">' +
        escapeHtml(r.reporter_role || 'user') + '</span>' +
      (r.username
        ? '<span class="pill" title="reported by">' + escapeHtml(r.username) + '</span>'
        : (r.user_id
            ? '<span class="pill" title="reported by (unknown user)">' + escapeHtml((r.user_id||'').slice(0,6)) + '</span>'
            : '')) +
      (r.model ? '<span class="pill">' + escapeHtml(r.model) + '</span>' : '') +
      (r.request_id
        ? '<span class="req" title="cross-reference key in the log file (grep req=' +
            escapeHtml((r.request_id || '').slice(0, 8)) + ')">req ' +
            escapeHtml((r.request_id || '').slice(0, 8)) + '</span>'
        : '') +
      '<span class="spacer"></span>';
    card.appendChild(head);

    // Status dropdown
    var sel = document.createElement('select');
    sel.className = 'status-' + (r.status || 'open');
    ['open', 'resolved', 'dismissed'].forEach(function(v) {
      var o = document.createElement('option');
      o.value = v; o.textContent = v;
      if (v === r.status) o.selected = true;
      sel.appendChild(o);
    });
    sel.addEventListener('change', function() { onStatusChange(r, sel); });
    head.appendChild(sel);

    // Delete button
    var delBtn = document.createElement('button');
    delBtn.className = 'danger';
    delBtn.textContent = 'Delete';
    delBtn.addEventListener('click', function() { onDelete(r); });
    head.appendChild(delBtn);

    // raw + final
    function textLine(klass, label, value) {
      var row = document.createElement('div');
      row.className = 'rc-textline ' + klass;
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
    card.appendChild(textLine('raw', 'raw', r.raw));
    card.appendChild(textLine('final', 'final', r.final));

    // Pipeline steps (collapsed; mostly noise unless we're debugging the
    // pipeline itself).
    var steps = r.steps || [];
    if (steps.length) {
      var det = document.createElement('details');
      det.className = 'rc-steps';
      var sum = document.createElement('summary');
      var changed = steps.filter(function(s) {
        return Array.isArray(s) && s.length >= 3 && s[1] !== s[2];
      }).length;
      sum.textContent = 'Pipeline steps (' + changed + ' changed, ' +
        (steps.length - changed) + ' unchanged)';
      det.appendChild(sum);
      steps.forEach(function(s) {
        if (!Array.isArray(s) || s.length < 3) return;
        var st = document.createElement('div');
        st.className = 'rc-step';
        st.innerHTML =
          '<span class="step-label">' + escapeHtml(s[0]) + '</span>' +
          '<span><span class="step-before">' + escapeHtml(s[1]) + '</span> ' +
          '<span class="step-after">→ ' + escapeHtml(s[2]) + '</span></span>';
        det.appendChild(st);
      });
      card.appendChild(det);
    }

    // Word corrections section
    var corr = (r.corrections || []).filter(function(c) {
      return c && c.correct;
    });
    if (corr.length) {
      var sec = document.createElement('div');
      sec.className = 'rc-section';
      sec.innerHTML = '<h3>Word corrections</h3>';
      var box = document.createElement('div');
      box.className = 'rc-corrections';
      corr.forEach(function(c) {
        var chip = document.createElement('div');
        chip.className = 'rc-correction';
        chip.innerHTML =
          '<span class="wrong">' + escapeHtml(c.wrong || '?') + '</span>' +
          '<span class="arrow">→</span>' +
          '<span class="correct">' + escapeHtml(c.correct || '') + '</span>';
        box.appendChild(chip);
      });
      sec.appendChild(box);
      card.appendChild(sec);
    }

    // Sentence rewrite diff
    if (r.intended_text && r.intended_text.trim()) {
      var sec2 = document.createElement('div');
      sec2.className = 'rc-section';
      sec2.innerHTML = '<h3>Sentence rewrite (diff vs. final)</h3>';
      sec2.appendChild(renderDiff(r.final || '', r.intended_text));
      card.appendChild(sec2);
    }

    // User comment
    if (r.user_comment && r.user_comment.trim()) {
      var sec3 = document.createElement('div');
      sec3.className = 'rc-section';
      sec3.innerHTML = '<h3>User comment</h3>';
      var p = document.createElement('div');
      p.className = 'rc-comment';
      p.textContent = r.user_comment;
      sec3.appendChild(p);
      card.appendChild(sec3);
    }

    // Admin notes — explicit Save
    var sec4 = document.createElement('div');
    sec4.className = 'rc-section rc-notes';
    sec4.innerHTML = '<h3>Admin notes</h3>';
    var ta = document.createElement('textarea');
    ta.value = r.admin_notes || '';
    ta.placeholder = 'Notes about how this was triaged, what was changed, follow-ups…';
    sec4.appendChild(ta);
    var row = document.createElement('div');
    row.className = 'row';
    var saveBtn = document.createElement('button');
    saveBtn.className = 'primary';
    saveBtn.textContent = 'Save notes';
    saveBtn.disabled = true;
    var dirty = document.createElement('span');
    dirty.className = 'dirty hidden';
    dirty.textContent = 'unsaved changes';
    row.appendChild(saveBtn);
    row.appendChild(dirty);
    sec4.appendChild(row);
    card.appendChild(sec4);

    ta.addEventListener('input', function() {
      var changed = ta.value !== (r.admin_notes || '');
      saveBtn.disabled = !changed;
      dirty.classList.toggle('hidden', !changed);
    });
    saveBtn.addEventListener('click', function() {
      onSaveNotes(r, ta, saveBtn, dirty);
    });
    ta.addEventListener('keydown', function(e) {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        if (!saveBtn.disabled) saveBtn.click();
      }
    });

    return card;
  }

  // -------------------------------------------------------------------
  // Actions
  // -------------------------------------------------------------------
  async function onStatusChange(r, sel) {
    var newStatus = sel.value;
    var prev = r.status;
    sel.className = 'status-' + newStatus;
    try {
      var j = await api('PATCH', '/reports/api/' + encodeURIComponent(r.id),
        { status: newStatus });
      r.status = j.report.status;
      r.resolved_ts = j.report.resolved_ts;
      toast('Status updated.');
      reloadCounts();
    } catch (e) {
      sel.value = prev;
      sel.className = 'status-' + prev;
      toast('Failed: ' + e.message, true);
    }
  }

  async function onSaveNotes(r, ta, btn, dirty) {
    try {
      var j = await api('PATCH', '/reports/api/' + encodeURIComponent(r.id),
        { admin_notes: ta.value });
      r.admin_notes = j.report.admin_notes;
      btn.disabled = true;
      dirty.classList.add('hidden');
      toast('Notes saved.');
    } catch (e) {
      toast('Failed to save notes: ' + e.message, true);
    }
  }

  async function onDelete(r) {
    if (!confirm('Delete this report permanently?')) return;
    try {
      await api('DELETE', '/reports/api/' + encodeURIComponent(r.id));
      _allReports = _allReports.filter(function(x) { return x.id !== r.id; });
      render();
      reloadCounts();
      toast('Deleted.');
    } catch (e) {
      toast('Failed: ' + e.message, true);
    }
  }

  async function onClearAll() {
    var m = document.getElementById('confirm-modal');
    var msg = document.getElementById('confirm-msg');
    msg.textContent = 'This permanently deletes ' + _allReports.length +
      ' report' + (_allReports.length === 1 ? '' : 's') +
      '. There is no undo.';
    m.classList.add('show');
    document.getElementById('confirm-cancel').onclick = function() {
      m.classList.remove('show');
    };
    document.getElementById('confirm-ok').onclick = async function() {
      m.classList.remove('show');
      try {
        var j = await api('POST', '/reports/api/clear', {});
        toast('Cleared ' + j.deleted + ' report' +
          (j.deleted === 1 ? '' : 's') + '.');
        await load();
      } catch (e) {
        toast('Failed: ' + e.message, true);
      }
    };
  }

  function onExport() {
    var tok = getToken();
    // Fetch via JS to send Authorization header; download as blob.
    // No token? Let the 401 branch below trigger the token modal.
    fetch('/reports/api/export', {
      headers: tok ? { 'Authorization': 'Bearer ' + tok } : {},
    }).then(function(resp) {
      if (resp.status === 401) {
        showTokenModal(function() { onExport(); });
        throw new Error('unauthorized');
      }
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      var fname = 'whisper-reports-' +
        new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19) + '.json';
      var cd = resp.headers.get('Content-Disposition') || '';
      var m = /filename="([^"]+)"/.exec(cd);
      if (m) fname = m[1];
      return resp.blob().then(function(blob) {
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url; a.download = fname;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(function() { URL.revokeObjectURL(url); }, 1000);
      });
    }).catch(function(e) {
      if (e && e.message !== 'unauthorized') toast('Export failed: ' + e.message, true);
    });
  }

  // -------------------------------------------------------------------
  // Render loop
  // -------------------------------------------------------------------
  function render() {
    var rows = applyFilters(_allReports);
    var list = document.getElementById('list');
    list.innerHTML = '';
    if (rows.length === 0) {
      var empty = document.createElement('div');
      empty.className = 'empty-state';
      if (_allReports.length === 0) {
        empty.innerHTML = '<strong>No reports yet.</strong><br>' +
          "When users flag a transcription on /quick-config, it lands here.";
      } else {
        empty.innerHTML = 'No reports match the current filters.';
      }
      list.appendChild(empty);
      return;
    }
    rows.forEach(function(r) {
      list.appendChild(renderCard(r));
    });
  }

  // -------------------------------------------------------------------
  // Load
  // -------------------------------------------------------------------
  async function load() {
    try {
      var j = await api('GET', '/reports/api/list');
      _allReports = j.reports || [];
      _counts = j.counts || { open: 0, resolved: 0, dismissed: 0 };
      rebuildModelFilter();
      updateCounts();
      render();
      // role-admin is set by OPEN_MODE_BANNER_JS (single source of truth)
      // when whoami.is_admin=true. Adding it here unconditionally would
      // leak admin chrome to non-admins with reports=own/all scope.
    } catch (e) {
      if (e.message === 'unauthorized' || e.message === 'not-admin') return;
      toast('Failed to load reports: ' + e.message, true);
    }
  }

  function reloadCounts() {
    _counts = { open: 0, resolved: 0, dismissed: 0 };
    _allReports.forEach(function(r) {
      if (_counts.hasOwnProperty(r.status)) _counts[r.status] += 1;
    });
    updateCounts();
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

  // First load. If unauthorized, prompt for the token then retry.
  load();
  timeTick();
})();
</script>
</body>
</html>
"""
