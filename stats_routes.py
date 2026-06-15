"""
/stats system overview dashboard.

Routes (all gated by web_common.require_user_webui_host):

  GET /stats           HTML page (single-file inline-HTML+CSS+JS, mirrors /logs)
  GET /stats/snapshot  one-shot JSON: ts + metrics_snapshot() + system_snapshot() + severity_counts()
  GET /stats/stream    SSE: same JSON, ~1 Hz (1 s data cadence defeats idle-proxy timeouts; no separate keepalive frame)

Access control (user tier): the shell is gated only by the host allowlist
cfg.USER_WEBUI_ALLOWED_HOSTS (loopback always allowed); the data endpoints
stack require_page("stats") so the API key is the inner gate. The dependency
reads cfg at request time so the admin WebUI can broaden access without a
restart.

Live updates: SSE rather than polling so we get free auto-reconnect on
service-restart, matching the /logs page UX.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse

import config as cfg
import metrics
import system_stats
import web_common
from auth import Permissions, require_page, user_from_session_cookie

router = APIRouter()

_require_stats_host = web_common.require_user_webui_host


def _require_stats_page_sse(request: Request) -> dict[str, Any]:
    """SSE-aware variant of `require_page("stats")`. EventSource cannot
    set Authorization, so we accept ?key=<raw_key> as a fallback.

    In OPEN mode (no admin key yet) the synthetic admin sails through;
    in locked-down mode the bearer must resolve to a user with
    scope("stats") != "none"."""
    import api_keys_store
    if not api_keys_store.is_locked_down():
        return dict(api_keys_store.OPEN_MODE_USER)
    auth_header = request.headers.get("authorization") or ""
    raw = ""
    if auth_header.lower().startswith("bearer "):
        raw = auth_header.split(" ", 1)[1].strip()
    rec = api_keys_store.lookup_by_raw_key(raw) if raw else None
    if rec is None:
        rec = user_from_session_cookie(request)
    if rec is None:
        key = request.query_params.get("key") or ""
        rec = api_keys_store.lookup_by_raw_key(key) if key else None
    if rec is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    perms = Permissions(
        rec.get("permissions_raw") or {}, bool(rec.get("is_admin")),
    )
    if not perms.can("stats"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "no access to /stats")
    rec["permissions"] = perms
    return rec


def _build_payload() -> dict[str, Any]:
    """Combine request metrics + system snapshot into one payload."""
    return {
        "ts": time.time(),
        **metrics.metrics_snapshot(),
        **system_stats.system_snapshot(),
        "severity": web_common.severity_counts(),
    }


@router.get(
    "/stats",
    response_class=HTMLResponse,
    # HTML page is host-only — the bearer isn't available on initial
    # navigation. API endpoints below gate by `require_page("stats")`;
    # the page's first snapshot fetch 403s for non-permitted users.
    dependencies=[Depends(_require_stats_host)],
)
async def stats_page() -> HTMLResponse:
    """Single-file inline HTML page. `no-store` so a browser never serves a
    stale build after a service restart."""
    return HTMLResponse(
        web_common.render_page(_STATS_VIEWER_HTML, current="stats"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get(
    "/stats/snapshot",
    dependencies=[
        Depends(_require_stats_host),
        Depends(require_page("stats")),
    ],
)
async def stats_snapshot() -> dict[str, Any]:
    """One-shot JSON. Useful for scripts and for the page's initial render."""
    return _build_payload()


@router.get(
    "/stats/usage",
    dependencies=[
        Depends(_require_stats_host),
        Depends(require_page("stats")),
    ],
)
async def stats_usage(
    days: int = 30,
    bucket: str = "day",
    by: str = "user",
    metric: str = "audio_s",
) -> dict[str, Any]:
    """Historical usage: a total-throughput-over-time series plus a
    per-user (or per-key) leaderboard for the window. Served once per page
    load / selector change — NOT part of the 1 Hz SSE payload.

    `days<=0` is lifetime (no lower bound). `bucket` ∈ {day, week}. `by` ∈
    {user, key}. `metric` ranks the leaderboard. Gated by the page's
    `stats` access (which has no 'own' scope — see api_keys_store
    ACCESS_ONLY_PAGES), so the data returned here is global."""
    import api_keys_store
    import usage_store

    bucket = "week" if bucket == "week" else "day"
    by = "key" if by == "key" else "user"
    if metric not in ("requests", "errors", "words", "audio_s"):
        metric = "audio_s"
    # Window in UTC epoch-hours; days reckoned in the SERVER-local timezone
    # (the operator's perspective). days<=0 = lifetime (no lower bound).
    start_hour: int | None = None
    if days and days > 0:
        start_hour = usage_store.local_day_start_hour(days_ago=int(days) - 1)

    # Global series gives the shared x-axis (every bucket with any usage) and
    # the per-bucket totals used to derive the "others" line.
    series_global = usage_store.series(start_hour=start_hour, bucket=bucket)
    day_axis = [int(p["day"]) for p in series_global]
    axis_index = {d: i for i, d in enumerate(day_axis)}
    n = len(day_axis)

    board = usage_store.leaderboard(
        start_hour=start_hour, by=by, metric=metric, limit=50,
    )

    # Resolve display names + a stable `id` server-side (the /stats client has
    # no api-keys data). Revoked users/keys still resolve; sentinels stay literal.
    names = api_keys_store.get_usernames([r["user_id"] for r in board])
    for r in board:
        if by == "user":
            r["id"] = r["user_id"]
            r["label"] = names.get(r["user_id"]) or r["user_id"]
        else:
            kid = r["key_id"]
            r["id"] = kid
            key = (api_keys_store.get_key(kid)
                   if kid and not kid.startswith("(") else None)
            lbl = (key or {}).get("label") or ""
            disp = (key or {}).get("key_prefix")
            r["label"] = (lbl or (disp + "…" if disp else kid))
            r["user_label"] = names.get(r["user_id"]) or r["user_id"]

    # One chart line per top-K entity (by the selected metric), aligned to the
    # shared x-axis with 0-fill for buckets where the entity had no usage.
    K = 8
    lines: list[dict[str, Any]] = []
    sum_top = [0.0] * n
    for r in board[:K]:
        kwargs = {"user_id": r["id"]} if by == "user" else {"key_id": r["id"]}
        s = usage_store.series(start_hour=start_hour, bucket=bucket, **kwargs)
        vals: list[float] = [0] * n
        for p in s:
            i = axis_index.get(int(p["day"]))
            if i is not None:
                vals[i] = p[metric]
        for i in range(n):
            sum_top[i] += vals[i] or 0
        line = {"id": r["id"], "label": r["label"], "values": vals}
        if by == "key":
            line["user_label"] = r.get("user_label")
        lines.append(line)

    # Fold the long tail (entities beyond top-K) into one "others" line =
    # global total minus the top-K, per bucket. Skip if nothing remains.
    if len(board) > K and n:
        others = []
        any_pos = False
        for i, p in enumerate(series_global):
            rem = (p[metric] or 0) - sum_top[i]
            if rem < 0:
                rem = 0  # float-subtraction guard
            if rem > 0:
                any_pos = True
            others.append(rem)
        if any_pos:
            lines.append({
                "id": "__others__",
                "label": f"others ({len(board) - K})",
                "values": others,
                "others": True,
            })

    return {
        "days": day_axis,
        "metric": metric,
        "by": by,
        "bucket": bucket,
        "lines": lines,
        "leaderboard": board,
    }


@router.get(
    "/stats/stream",
    dependencies=[
        Depends(_require_stats_host),
        Depends(_require_stats_page_sse),
    ],
)
async def stats_stream() -> StreamingResponse:
    """1 Hz SSE stream of the snapshot payload. The 1-second data cadence
    already counts as traffic for idle-proxy timeout purposes — no separate
    keepalive frame needed."""
    async def gen():
        while True:
            payload = _build_payload()
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(1.0)

    return web_common.sse_response(gen())


# --- HTML template -----------------------------------------------------------
# Single-file, no build step. Mirrors the /logs and /settings style. uPlot is
# loaded from the local /static mount — no CDN, works offline.

_STATS_VIEWER_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{{HEADER_TITLE}}</title>
{{PAGE_META}}
{{SCALE_BOOTSTRAP_HEAD}}
<link rel="stylesheet" href="/static/uplot.min.css">
<link rel="stylesheet" href="/static/gridstack.min.css">
<script src="/static/uplot.iife.min.js"></script>
<script src="/static/gridstack.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --fg: #c9d1d9; --dim: #6e7681;
    --cyan: #79c0ff; --green: #7ee787; --yellow: #f2cc60;
    --red: #ff7b72; --magenta: #d2a8ff; --bold: #f0f6fc;
    --border: #30363d;
  }
  /* Font tokens, --font-sans, --font-mono and html font-size live in
     NAV_CSS (injected further down). Important: never embed the NAV_CSS
     template placeholder inside another comment block — render_page() does
     a naive string replace and would inject NAV_CSS into this comment,
     prematurely closing it (NAV_CSS contains its own internal comments)
     and silently dropping every CSS rule that follows. Chrome (titles,
     buttons, badges, card headers) uses --font-sans; uPlot's axis labels
     and the spark-head numeric readouts stay in --font-mono so digits
     align (font-variant-numeric: tabular-nums hint relies on the mono
     stack for crisp tabular alignment). */
  html, body { background: var(--bg); color: var(--fg);
    font: 1rem/1.5 var(--font-sans);
    margin: 0; padding: 0; min-height: 100%; }
  input, textarea, select, kbd, code, pre { font-family: var(--font-mono); }
  /* header / .header-inner / .title / page-toolbar controls (buttons,
     pills) are all centralized in NAV_CSS. */
  {{NAV_CSS}}
  .grid { padding: 0.875rem; max-width: 68.75rem; margin: 0 auto;
    box-sizing: border-box; min-height: 60vh; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 0.625rem 0.75rem; min-width: 0; height: 100%; box-sizing: border-box;
    overflow: auto; display: flex; flex-direction: column; min-height: 0; }
  .card h3 { font-size: var(--fs-xs); color: var(--dim); margin: 0 0 0.375rem;
    text-transform: uppercase; letter-spacing: .05em; font-weight: 500; }
  .card .val { color: var(--bold); font-size: var(--fs-xxl); font-weight: 600; line-height: 1.1; }
  .card .val .sub { color: var(--dim); font-size: var(--fs-sm); font-weight: normal; margin-left: 0.375rem; }
  .card .meta { color: var(--dim); font-size: var(--fs-xs); margin-top: 0.25rem; }
  .card .meta b { color: var(--fg); font-weight: 500; }
  .bar { height: 6px; background: #21262d; border-radius: 3px; margin-top: 0.375rem; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: var(--cyan);
    transition: width .3s ease; }
  .bar.warn > i { background: var(--yellow); }
  .bar.crit > i { background: var(--red); }
  .spark-wrap  { margin-top: 0.625rem; min-width: 0;
                 flex: 1 1 0; min-height: 0;
                 display: flex; flex-direction: column; }
  .spark-head  { display: flex; justify-content: space-between; align-items: baseline;
                 font: var(--fs-xs) var(--font-mono);
                 color: var(--dim); margin-bottom: 2px; flex: 0 0 auto; }
  .spark-label { letter-spacing: .03em; text-transform: uppercase; }
  .spark-now   { color: var(--bold); font-weight: 600;
                 font-variant-numeric: tabular-nums; }
  .spark       { flex: 1 1 0; min-height: 4rem; width: 100%; }
  .uplot, .u-wrap { background: transparent !important; }
  .u-legend { display: none; }
  .u-axis { color: var(--dim); }
  table.tbl { width: 100%; border-collapse: collapse; font-size: var(--fs-sm); }
  table.tbl th, table.tbl td { padding: 0.25rem 0.375rem; text-align: left;
    border-bottom: 1px solid #21262d; }
  table.tbl th { color: var(--dim); font-weight: 500; font-size: var(--fs-xs);
    text-transform: uppercase; }
  table.tbl td.num { text-align: right; font-variant-numeric: tabular-nums; }
  table.tbl th.num { text-align: right; }
  .badge { display: inline-block; font-size: 0.667rem; padding: 0.0625rem 0.375rem;
    border-radius: 999px; border: 1px solid var(--border); color: var(--dim); }
  .badge.warm { color: var(--green); border-color: #1f4d2a; }
  .badge.cold { color: var(--yellow); border-color: #4d3e1f; }
  .badge.ok { color: var(--green); border-color: #1f4d2a; }
  .badge.err { color: var(--red); border-color: #5a2424; }
  .ts { color: var(--dim); font-variant-numeric: tabular-nums; }
  .core-strip { display: flex; gap: 2px; margin-top: 0.375rem; height: 1.5rem;
    align-items: flex-end; }
  .core-strip > div { flex: 1; background: var(--cyan); border-radius: 1px;
    min-height: 2px; transition: height .3s ease; }
  .err-strip { display: flex; gap: 0.25rem; margin-top: 0.375rem; }
  .err-strip .seg { flex: 1; text-align: center; padding: 0.375rem;
    background: #21262d; border-radius: 4px; }
  .err-strip .seg b { color: var(--bold); display: block; font-size: var(--fs-xl); }
  .err-strip .seg.hot { background: #2d1414; }
  .err-strip .seg.hot b { color: var(--red); }
  .empty { color: var(--dim); font-style: italic; }
  .hidden { display: none !important; }
  /* Usage-over-time tile — a full-width GridStack item. The card fills the
     whole tile (height:100%) and the chart flexes to absorb any slack, so the
     tile is never taller than the card (no dead clickable space below) and
     resizing the tile grows/shrinks the chart. */
  .usage-card { height: 100%; }
  .usage-toolbar { display: flex; flex-wrap: wrap; align-items: baseline;
    gap: 0.4rem 0.9rem; margin-bottom: 0.5rem; }
  .usage-toolbar h3 { margin: 0; }
  .usage-toolbar .spacer { flex: 1 1 auto; }
  .usage-seg { display: inline-flex; align-items: center; gap: 0.4rem; }
  .usage-seg .seg-label { color: var(--dim); font-size: var(--fs-xs);
    text-transform: uppercase; letter-spacing: .04em; }
  .seg-ctrl { display: inline-flex; border: 1px solid var(--border);
    border-radius: 6px; overflow: hidden; }
  .seg-ctrl button { background: var(--bg); color: var(--dim);
    border: none; border-left: 1px solid var(--border);
    padding: 0.15rem 0.55rem; font: inherit; font-size: var(--fs-sm);
    line-height: 1.3; cursor: pointer; }
  .seg-ctrl button:first-child { border-left: none; }
  .seg-ctrl button:hover { color: var(--fg); }
  .seg-ctrl button.active { background: var(--panel); color: var(--cyan);
    font-weight: 600; }
  .usage-chart { width: 100%; flex: 1 1 auto; min-height: 9rem; min-width: 0;
    position: relative; }
  .usage-plot { width: 100%; height: 100%; min-width: 0; }
  .usage-note { color: var(--dim); font-size: var(--fs-xs);
    margin: 0.15rem 0 0.5rem; }
  /* Floating cursor tooltip over the multi-line usage chart. */
  .usage-tip { position: fixed; z-index: 5; pointer-events: none;
    background: var(--panel); border: 1px solid var(--border); border-radius: 5px;
    padding: 0.3rem 0.45rem; font: var(--fs-xs)/1.35 var(--font-mono);
    color: var(--fg); white-space: nowrap; display: none;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4); }
  .usage-tip .tip-date { color: var(--dim); margin-bottom: 0.15rem; }
  .usage-tip .tip-row { display: flex; align-items: center; gap: 0.35rem; }
  .usage-tip .tip-row.focus { color: var(--bold); font-weight: 600; }
  .usage-tip .tip-row .tip-val { margin-left: auto;
    font-variant-numeric: tabular-nums; }
  .usage-swatch { display: inline-block; width: 0.6rem; height: 0.6rem;
    border-radius: 2px; flex: 0 0 auto; vertical-align: baseline; }
  table.usage-board td.rank { color: var(--dim);
    font-variant-numeric: tabular-nums; width: 2rem; }
  table.usage-board td.name { color: var(--fg); }
  table.usage-board td.name .usage-swatch { margin-right: 0.4rem; }
  table.usage-board td.name .sub { color: var(--dim);
    font-size: var(--fs-xs); margin-left: 0.4rem; }
  /* GridStack integration — drag-to-reorder + click-to-resize tiles. */
  .grid-stack { background: transparent; }
  .grid-stack-item-content { background: transparent; padding: 0; overflow: visible; }
  .grid-stack-item .card { cursor: default; }
  .grid-stack-item .card h3 { cursor: grab; user-select: none; }
  .grid-stack-item .card h3:active { cursor: grabbing; }
  .grid-stack-placeholder > .placeholder-content {
    background: rgba(56, 189, 248, 0.08);
    border: 1px dashed var(--cyan);
    border-radius: 6px;
  }
  .grid-stack > .grid-stack-item > .ui-resizable-handle {
    background-image: none;
    color: var(--dim);
    opacity: 0;
    transition: opacity 120ms ease;
  }
  .grid-stack > .grid-stack-item:hover > .ui-resizable-handle { opacity: 0.6; }
  .grid-stack > .grid-stack-item > .ui-resizable-se {
    width: 12px; height: 12px;
    border-right: 2px solid var(--dim);
    border-bottom: 2px solid var(--dim);
    transform: none;
  }
</style></head>
<body>
<header>
  <div class="header-inner">
    <span class="title">{{HEADER_BRAND}}</span>
    <span class="brand-sep" aria-hidden="true"></span>
    {{NAV}}
    <span class="spacer"></span>
    <span class="hdr-right">{{SEV_PILLS}}{{SCALE_PICKER}}{{RELOAD}}{{LOGOUT}}</span>
  </div>
  <div class="subbar">
    <span class="subbar-title">Stats</span>
    <div class="subbar-right">
      <button id="reset-layout-btn" title="reset stats tile layout to defaults">↺ layout</button>
      <span id="status" class="pill live">live</span>
    </div>
  </div>
</header>

<div id="grid" class="grid">
 <div class="grid-stack">
  <!-- GPU (one GridStack item; inner content swaps between "live" and "no NVML"
       — a second hidden grid-stack-item would still occupy a cell and, under
       float: true, shift to a free slot that overlaps the row below.) -->
  <div class="grid-stack-item" gs-id="gpu" gs-x="0" gs-y="0" gs-w="6" gs-h="9">
   <div class="grid-stack-item-content"><div id="card-gpu" class="card">
    <h3>GPU</h3>
    <div id="gpu-content">
     <div id="gpu-name" class="val">—</div>
     <div id="gpu-meta" class="meta"></div>
     <div id="gpu-mem-bar" class="bar"><i style="width:0"></i></div>
     <div id="gpu-meta2" class="meta"></div>
     <div class="spark-wrap">
       <div class="spark-head"><span class="spark-label">GPU util %</span>
         <span id="gpu-util-now" class="spark-now">—</span></div>
       <div id="gpu-spark-util" class="spark"></div>
     </div>
     <div class="spark-wrap">
       <div class="spark-head"><span class="spark-label">VRAM used %</span>
         <span id="gpu-mem-now" class="spark-now">—</span></div>
       <div id="gpu-spark-mem" class="spark"></div>
     </div>
     <div class="spark-wrap">
       <div class="spark-head"><span class="spark-label">GPU temp °C</span>
         <span id="gpu-temp-now" class="spark-now">—</span></div>
       <div id="gpu-spark-temp" class="spark"></div>
     </div>
    </div>
    <div id="gpu-empty" class="hidden">
     <div class="empty">NVML unavailable — running on CPU or pynvml not installed.</div>
     <div id="gpu-error" class="meta"></div>
    </div>
   </div></div>
  </div>

  <!-- Host CPU -->
  <div class="grid-stack-item" gs-id="cpu" gs-x="6" gs-y="0" gs-w="6" gs-h="5">
   <div class="grid-stack-item-content"><div class="card">
    <h3>CPU (host)</h3>
    <div id="cpu-pct" class="val">—<span class="sub">%</span></div>
    <div id="cpu-cores" class="core-strip"></div>
    <div class="spark-wrap">
      <div class="spark-head"><span class="spark-label">CPU %</span>
        <span id="cpu-now" class="spark-now">—</span></div>
      <div id="cpu-spark" class="spark"></div>
    </div>
   </div></div>
  </div>

  <!-- Host RAM -->
  <div class="grid-stack-item" gs-id="ram" gs-x="6" gs-y="5" gs-w="6" gs-h="4">
   <div class="grid-stack-item-content"><div class="card">
    <h3>RAM</h3>
    <div id="ram-val" class="val">— <span class="sub">/ — GB</span></div>
    <div id="ram-bar" class="bar"><i style="width:0"></i></div>
    <div id="ram-meta" class="meta"></div>
    <div class="spark-wrap">
      <div class="spark-head"><span class="spark-label">RAM used %</span>
        <span id="ram-now" class="spark-now">—</span></div>
      <div id="ram-spark" class="spark"></div>
    </div>
   </div></div>
  </div>

  <!-- Process -->
  <div class="grid-stack-item" gs-id="process" gs-x="0" gs-y="9" gs-w="4" gs-h="3">
   <div class="grid-stack-item-content"><div class="card">
    <h3>Process</h3>
    <div id="proc-rss" class="val">—<span class="sub">MB RSS</span></div>
    <div id="proc-meta" class="meta"></div>
   </div></div>
  </div>

  <!-- In-flight + uptime -->
  <div class="grid-stack-item" gs-id="activity" gs-x="4" gs-y="9" gs-w="4" gs-h="3">
   <div class="grid-stack-item-content"><div class="card">
    <h3>Activity</h3>
    <div id="inflight-val" class="val">0<span class="sub">in flight</span></div>
    <div id="activity-meta" class="meta"></div>
   </div></div>
  </div>

  <!-- Errors window -->
  <div class="grid-stack-item" gs-id="errors" gs-x="8" gs-y="9" gs-w="4" gs-h="3">
   <div class="grid-stack-item-content"><div class="card">
    <h3>Errors (5xx)</h3>
    <div class="err-strip">
      <div id="err-1m" class="seg"><b>0</b>1 min</div>
      <div id="err-5m" class="seg"><b>0</b>5 min</div>
      <div id="err-15m" class="seg"><b>0</b>15 min</div>
    </div>
    <div id="err-meta" class="meta"></div>
   </div></div>
  </div>

  <!-- Latency -->
  <div class="grid-stack-item" gs-id="latency" gs-x="0" gs-y="12" gs-w="6" gs-h="5">
   <div class="grid-stack-item-content"><div class="card">
    <h3>Request latency (last <span id="lat-n">0</span>)</h3>
    <div id="lat-val" class="val">— <span class="sub">ms p50</span></div>
    <div id="lat-meta" class="meta"></div>
    <div class="spark-wrap">
      <div class="spark-head"><span class="spark-label">p50 latency (ms)</span>
        <span id="lat-now" class="spark-now">—</span></div>
      <div id="lat-spark" class="spark"></div>
    </div>
   </div></div>
  </div>

  <!-- Endpoint counters -->
  <div class="grid-stack-item" gs-id="endpoints" gs-x="6" gs-y="12" gs-w="6" gs-h="5">
   <div class="grid-stack-item-content"><div class="card">
    <h3>Endpoint counters</h3>
    <table class="tbl rcards"><thead><tr><th>path</th><th class="num">requests</th><th class="num">5xx</th></tr></thead>
    <tbody id="endpoints-rows"></tbody></table>
   </div></div>
  </div>

  <!-- Loaded models -->
  <div class="grid-stack-item" gs-id="models" gs-x="0" gs-y="17" gs-w="12" gs-h="4">
   <div class="grid-stack-item-content"><div class="card">
    <h3>Loaded models</h3>
    <table class="tbl rcards"><thead><tr>
      <th>name</th><th>device</th><th>compute</th>
      <th class="num">VRAM (MB)</th><th>state</th>
      <th class="num">age</th><th class="num">idle</th>
      <th class="num">cold-load</th>
    </tr></thead><tbody id="models-rows"></tbody></table>
   </div></div>
  </div>

  <!-- Usage over time -->
  <div class="grid-stack-item" gs-id="usage" gs-x="0" gs-y="21" gs-w="12" gs-h="9">
   <div class="grid-stack-item-content"><div class="card usage-card">
    <div class="usage-toolbar">
      <h3>Usage over time</h3>
      <span class="spacer"></span>
      <div class="usage-seg"><span class="seg-label">range</span>
        <div class="seg-ctrl" id="usage-range">
          <button data-v="7">7d</button>
          <button data-v="30" class="active">30d</button>
          <button data-v="90">90d</button>
          <button data-v="0">all</button>
        </div>
      </div>
      <div class="usage-seg"><span class="seg-label">bucket</span>
        <div class="seg-ctrl" id="usage-bucket">
          <button data-v="day" class="active">day</button>
          <button data-v="week">week</button>
        </div>
      </div>
      <div class="usage-seg"><span class="seg-label">metric</span>
        <div class="seg-ctrl" id="usage-metric">
          <button data-v="audio_s" class="active">audio</button>
          <button data-v="words">words</button>
          <button data-v="requests">requests</button>
          <button data-v="errors">errors</button>
        </div>
      </div>
      <div class="usage-seg"><span class="seg-label">by</span>
        <div class="seg-ctrl" id="usage-by">
          <button data-v="user" class="active">user</button>
          <button data-v="key">key</button>
        </div>
      </div>
    </div>
    <div class="usage-chart">
      <div id="usage-plot" class="usage-plot"></div>
      <div id="usage-tip" class="usage-tip"></div>
    </div>
    <div class="usage-note">Buckets are local days. Showing the top 8 by the selected metric; the rest are folded into “others”.</div>
    <table class="tbl usage-board rcards"><thead><tr>
      <th class="rank">#</th><th>name</th>
      <th class="num">requests</th><th class="num">words</th>
      <th class="num">audio</th><th class="num">err</th>
    </tr></thead><tbody id="usage-board-rows">
      <tr><td colspan="6" class="empty">— loading —</td></tr>
    </tbody></table>
   </div></div>
  </div>
  <!-- Recent transcriptions -->
  <div class="grid-stack-item" gs-id="recent" gs-x="0" gs-y="30" gs-w="12" gs-h="6">
   <div class="grid-stack-item-content"><div class="card">
    <h3>Recent transcriptions (last <span id="rt-n">0</span>)</h3>
    <table class="tbl rcards"><thead><tr>
      <th>when</th><th>model</th>
      <th class="num">audio</th><th class="num">wall</th><th class="num">RTF</th>
      <th class="num">words</th><th>status</th>
    </tr></thead><tbody id="rt-rows"><tr><td colspan="7" class="empty">— no requests yet —</td></tr></tbody></table>
   </div></div>
  </div>
 </div>
</div>

<script>
(() => {
'use strict';

// --- GridStack init: drag-to-reorder + click-to-resize tiles ---------------
// Layout state persists in localStorage; [↺ layout] in the header clears it.
// uPlot sparklines re-fit on resizestop via setSize().
const GS_LAYOUT_KEY = 'whisper-stats-layout-v5';
const grid = GridStack.init({
  column: 12,
  // Responsive: collapse to a single stacked column on phones/narrow tablets.
  // breakpointForWindow keys off the viewport width (not the grid container),
  // and layout:'list' keeps tiles in their saved order when reflowing.
  // (GridStack 10+ has responsive OFF by default, so this is required.)
  columnOpts: {
    breakpointForWindow: true,
    breakpoints: [{ w: 700, c: 1 }],
    layout: 'list',
  },
  // String form so cells track --fs-base (the scale picker). At 100% scale,
  // 4rem = 60px (matches the previous fixed value); at 130% it's ~78px.
  // Saved layouts (column units) preserve unchanged across scale changes.
  cellHeight: '4rem',
  margin: 6,
  float: true,
  resizable: { handles: 'se,s,e' },
  draggable: { handle: '.card h3' },
  alwaysShowResizeHandle: false,
});
// On touch devices, freeze drag/resize (the layout stays, but reordering tiles
// by dragging is fiddly on a phone and the dashboard is read-mostly there).
try {
  if (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) {
    grid.setStatic(true);
  }
} catch (_) {}
// Restore saved layout if present (best-effort — schema mismatches are
// silently ignored; user can hit [↺ layout] to recover defaults).
try {
  const saved = localStorage.getItem(GS_LAYOUT_KEY);
  if (saved) grid.load(JSON.parse(saved));
} catch (_) {}
// Persist on every change (debounced via setTimeout to coalesce rapid drags).
let _saveTimer = null;
function _saveLayout() {
  if (_saveTimer) clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    try { localStorage.setItem(GS_LAYOUT_KEY, JSON.stringify(grid.save(false))); } catch (_) {}
  }, 200);
}
grid.on('change added removed', _saveLayout);
// Resize is handled by per-spark ResizeObserver inside makeSpark() — fires on
// GridStack drag-resize, window resize, scale-picker rem changes, and any
// other reflow uniformly. Listening on `resizestop` here would only catch
// GridStack-initiated resizes and would miss the rest.
// Header reset-layout button.
const resetLayoutBtn = document.getElementById('reset-layout-btn');
if (resetLayoutBtn) {
  resetLayoutBtn.addEventListener('click', () => {
    if (!confirm('Reset stats tile layout to defaults?')) return;
    localStorage.removeItem(GS_LAYOUT_KEY);
    location.reload();
  });
}

// --- per-metric history rings ----------------------------------------------
const HISTORY_LEN = 120;     // 2 min @ 1 Hz
const histX = [];            // shared time axis (epoch seconds)
const hist = {
  gpu_util: [], gpu_mem_pct: [], gpu_temp: [],
  cpu: [], ram_pct: [], lat_p50: [],
};

function pushHistory(snap) {
  const now = Math.floor(snap.ts || (Date.now() / 1000));
  histX.push(now);
  hist.gpu_util.push(snap.gpu ? snap.gpu.util_pct ?? null : null);
  hist.gpu_mem_pct.push(snap.gpu && snap.gpu.mem_total_mb
    ? (snap.gpu.mem_used_mb / snap.gpu.mem_total_mb * 100) : null);
  hist.gpu_temp.push(snap.gpu ? snap.gpu.temp_c ?? null : null);
  hist.cpu.push(snap.host ? snap.host.cpu_pct ?? null : null);
  hist.ram_pct.push(snap.host ? snap.host.ram_pct ?? null : null);
  hist.lat_p50.push(snap.latency_ms && snap.latency_ms.n > 0 ? snap.latency_ms.p50 : null);
  if (histX.length > HISTORY_LEN) {
    histX.shift();
    for (const k in hist) hist[k].shift();
  }
}

// --- uPlot factory ---------------------------------------------------------
// Each spark gets:
//   - explicit `splits` to force readable y-axis ticks (uPlot auto-picks one
//     tick on flat/idle data, which renders as a lonely "0").
//   - `unit` suffix on those ticks.
//   - auto-padding (10% top/bottom) when no fixed range — keeps unbounded
//     metrics like temperature / latency from pinning to the bottom.
//
// uPlot's canvas rendering needs px (not rem). These helpers read the
// current --fs-base via getComputedStyle so axis sizing tracks the scale
// picker. `--fs-base` is set by SCALE_BOOTSTRAP_HEAD BEFORE this script
// runs, so on first load the axes match the saved scale. Live picker
// changes don't refit the canvas — switching scale visibly updates HTML
// chrome but axis labels stay at construction-time size until the next
// page load. Acceptable trade-off vs destroying/rebuilding sparks (which
// would blank the chart until the next SSE tick).
function _remPx(n) {
  const base = parseFloat(getComputedStyle(document.documentElement).fontSize) || 15;
  return Math.round(n * base);
}
function _axisFontPx() { return _remPx(0.733); }   // matches --fs-xs
const _MONO_STACK = 'Consolas, "Cascadia Code", "JetBrains Mono", Menlo, ui-monospace, monospace';
const sparks = {};   // name -> uPlot instance
function makeSpark(elId, color, opts={}) {
  const el = document.getElementById(elId);
  if (!el) return null;
  const w = el.clientWidth || 240;
  const h = el.clientHeight || 72;
  const yScale = opts.range
    ? { range: opts.range }
    : { range: { min: { pad: 0.1, mode: 1 }, max: { pad: 0.1, mode: 1 } } };
  // uPlot's canvas API needs px values, not rem. Read them from --fs-base
  // so axis labels track the scale picker — see _axisFontPx below.
  const axisFontPx = _axisFontPx();
  const inst = new uPlot({
    width: w, height: h,
    // [top, right, bottom, left] in px. Top AND bottom both ≥ ½ axis-font
    // height + a small breathing margin so the highest split label
    // ("100%" / "60°") and the lowest ("0%" / "30°") render their full
    // glyph height inside the canvas instead of being clipped by the
    // canvas edges (uPlot draws tick labels centered on the data-area
    // edge — half the glyph extends past the edge, so padding must
    // exceed font-size/2). Left padding plus axis size gives uPlot room
    // to draw "100%" without GridStack's overflow-x clipping the "1".
    padding: [_remPx(0.55), 6, _remPx(0.4), _remPx(0.25)],
    cursor: { show: false },
    legend: { show: false },
    select: { show: false },
    scales: { x: { time: false }, y: yScale },
    axes: [
      { show: false },
      { show: true, size: _remPx(2.6), gap: 4,
        font: axisFontPx + 'px ' + _MONO_STACK,
        stroke: '#6e7681',
        grid:  { stroke: '#21262d', width: 1 },
        ticks: { stroke: '#30363d', width: 1, size: 3 },
        splits: opts.splits,
        values: opts.splits ? (u, splits) => splits.map(v => v + (opts.unit || '')) : null,
      },
    ],
    series: [
      {},
      { stroke: color, width: 1.5, fill: color + '22', spanGaps: true,
        points: { show: false } },
    ],
  }, [[], []], el);
  // Responsive sizing. ResizeObserver on the spark element fires for any
  // size source — GridStack drag-resize, window resize, scale-picker rem
  // changes, .hidden toggle reflow. rAF coalescing avoids thrashing
  // setSize() during a drag (it's a relatively expensive canvas rebuild).
  let raf = 0;
  const ro = new ResizeObserver(() => {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      const cw = el.clientWidth, ch = el.clientHeight;
      if (cw < 1 || ch < 1) return;
      inst.setSize({ width: cw, height: ch });
    });
  });
  ro.observe(el);
  return inst;
}

function ensureSparks() {
  if (sparks.cpu) return;     // already built
  // Percentage sparks: fixed [0, 100] range with 0/50/100 ticks.
  sparks.cpu      = makeSpark('cpu-spark',      '#79c0ff', { range: [0, 100], splits: [0, 50, 100], unit: '%' });
  sparks.ram      = makeSpark('ram-spark',      '#7ee787', { range: [0, 100], splits: [0, 50, 100], unit: '%' });
  sparks.gpu_util = makeSpark('gpu-spark-util', '#79c0ff', { range: [0, 100], splits: [0, 50, 100], unit: '%' });
  sparks.gpu_mem  = makeSpark('gpu-spark-mem',  '#d2a8ff', { range: [0, 100], splits: [0, 50, 100], unit: '%' });
  // Temperature: fixed coarse splits at 30/60/90 °C cover idle through hot.
  sparks.gpu_temp = makeSpark('gpu-spark-temp', '#f2cc60', { splits: [30, 60, 90], unit: '°' });
  // Latency: unbounded, auto-range with 10% padding.
  sparks.lat      = makeSpark('lat-spark',      '#7ee787');
}

function setData(u, ys) {
  if (!u) return;
  // uPlot wants nulls preserved for spanGaps; convert undefined -> null.
  const xs = histX.slice();
  const yClean = ys.map(v => (v == null ? null : v));
  u.setData([xs, yClean], true);
}

// --- DOM helpers -----------------------------------------------------------
const $ = id => document.getElementById(id);
function fmtBytes(mb) {
  if (mb == null) return '—';
  return mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : mb.toFixed(0) + ' MB';
}
function fmtSec(s) {
  if (s == null) return '—';
  if (s < 60) return s.toFixed(0) + ' s';
  if (s < 3600) return (s / 60).toFixed(1) + ' min';
  if (s < 86400) return (s / 3600).toFixed(1) + ' h';
  return (s / 86400).toFixed(1) + ' d';
}
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function setBar(barEl, pct) {
  const bar = barEl.querySelector('i');
  bar.style.width = Math.max(0, Math.min(100, pct)).toFixed(1) + '%';
  barEl.classList.toggle('warn', pct >= 75 && pct < 90);
  barEl.classList.toggle('crit', pct >= 90);
}

// --- Render ----------------------------------------------------------------
function render(snap) {
  ensureSparks();

  // --- GPU ---
  if (snap.gpu) {
    // Swap inner content of the single GPU GridStack item (not the wrapper
    // — a hidden grid-stack-item still occupies a cell and breaks the row
    // below it under float: true).
    $('gpu-content').classList.remove('hidden');
    $('gpu-empty').classList.add('hidden');
    $('gpu-name').textContent = snap.gpu.name || 'GPU';
    $('gpu-meta').innerHTML =
      `<b>util</b> ${snap.gpu.util_pct ?? '—'}% &nbsp; ` +
      `<b>temp</b> ${snap.gpu.temp_c ?? '—'}°C &nbsp; ` +
      `<b>power</b> ${snap.gpu.power_w ?? '—'} / ${snap.gpu.power_limit_w ?? '—'} W &nbsp; ` +
      `<b>state</b> ${snap.gpu.p_state || '—'}`;
    const memPct = snap.gpu.mem_total_mb
      ? snap.gpu.mem_used_mb / snap.gpu.mem_total_mb * 100 : 0;
    setBar($('gpu-mem-bar'), memPct);
    $('gpu-meta2').innerHTML =
      `<b>VRAM</b> ${fmtBytes(snap.gpu.mem_used_mb)} / ${fmtBytes(snap.gpu.mem_total_mb)} ` +
      `(${memPct.toFixed(0)}%) &nbsp; ` +
      `<b>SM clock</b> ${snap.gpu.sm_clock_mhz ?? '—'} MHz &nbsp; ` +
      `<b>driver</b> ${snap.gpu.driver || '—'} &nbsp; <b>CUDA</b> ${snap.gpu.cuda || '—'}`;
    setData(sparks.gpu_util, hist.gpu_util);
    setData(sparks.gpu_mem,  hist.gpu_mem_pct);
    setData(sparks.gpu_temp, hist.gpu_temp);
    $('gpu-util-now').textContent = (snap.gpu.util_pct ?? 0).toFixed(0) + '%';
    $('gpu-mem-now').textContent  = memPct.toFixed(0) + '%';
    $('gpu-temp-now').textContent = (snap.gpu.temp_c ?? 0).toFixed(0) + '°C';
  } else {
    $('gpu-content').classList.add('hidden');
    $('gpu-empty').classList.remove('hidden');
    $('gpu-error').textContent = snap.gpu_error || '';
  }

  // --- Host CPU ---
  $('cpu-pct').innerHTML = (snap.host.cpu_pct ?? 0).toFixed(1) + '<span class="sub">%</span>';
  const stripEl = $('cpu-cores');
  const cores = snap.host.cpu_per_core || [];
  if (stripEl.children.length !== cores.length) {
    stripEl.innerHTML = '';
    for (let i = 0; i < cores.length; i++) stripEl.appendChild(document.createElement('div'));
  }
  for (let i = 0; i < cores.length; i++) {
    stripEl.children[i].style.height = Math.max(2, cores[i]) + '%';
  }
  setData(sparks.cpu, hist.cpu);
  $('cpu-now').textContent = (snap.host.cpu_pct ?? 0).toFixed(0) + '%';

  // --- Host RAM ---
  $('ram-val').innerHTML = `${fmtBytes(snap.host.ram_used_mb)} ` +
    `<span class="sub">/ ${fmtBytes(snap.host.ram_total_mb)}</span>`;
  setBar($('ram-bar'), snap.host.ram_pct);
  $('ram-meta').innerHTML = `<b>${snap.host.ram_pct.toFixed(1)}%</b> used &nbsp; ` +
    `<b>disk free</b> ${snap.host.disk_free_gb ?? '—'} GB (model cache)`;
  setData(sparks.ram, hist.ram_pct);
  $('ram-now').textContent = (snap.host.ram_pct ?? 0).toFixed(0) + '%';

  // --- Process ---
  $('proc-rss').innerHTML = (snap.process.rss_mb ?? 0).toFixed(0) +
    '<span class="sub">MB RSS</span>';
  $('proc-meta').innerHTML =
    `<b>PID</b> ${snap.process.pid} &nbsp; ` +
    `<b>CPU</b> ${(snap.process.cpu_pct ?? 0).toFixed(1)}% &nbsp; ` +
    `<b>threads</b> ${snap.process.threads ?? '—'} &nbsp; ` +
    `<b>uptime</b> ${fmtSec(snap.process.uptime_sec)}`;

  // --- Activity / in-flight ---
  $('inflight-val').innerHTML = `${snap.in_flight_transcriptions}` +
    `<span class="sub">in flight</span>`;
  const totalReq = Object.values(snap.requests || {}).reduce((a, b) => a + b, 0);
  $('activity-meta').innerHTML =
    `<b>uptime</b> ${fmtSec(snap.uptime_sec)} &nbsp; ` +
    `<b>total req</b> ${totalReq}`;

  // --- Latency ---
  const lat = snap.latency_ms || { n: 0, p50: 0, p95: 0, p99: 0 };
  $('lat-n').textContent = lat.n;
  if (lat.n > 0) {
    $('lat-val').innerHTML = lat.p50.toFixed(0) + '<span class="sub">ms p50</span>';
    $('lat-meta').innerHTML =
      `<b>p95</b> ${lat.p95.toFixed(0)} ms &nbsp; ` +
      `<b>p99</b> ${lat.p99.toFixed(0)} ms`;
    $('lat-now').textContent = lat.p50.toFixed(0) + ' ms';
  } else {
    $('lat-val').innerHTML = '—';
    $('lat-meta').innerHTML = '<span class="empty">no requests yet</span>';
    $('lat-now').textContent = '—';
  }
  setData(sparks.lat, hist.lat_p50);

  // --- Errors window ---
  const ew = snap.errors_window || { '1m': 0, '5m': 0, '15m': 0 };
  for (const k of ['1m', '5m', '15m']) {
    const seg = $('err-' + k);
    seg.firstElementChild.textContent = ew[k];
    seg.classList.toggle('hot', ew[k] > 0);
  }
  const errTotal = Object.values(snap.errors_total || {}).reduce((a, b) => a + b, 0);
  $('err-meta').innerHTML = `<b>total</b> ${errTotal} since startup`;

  // --- Endpoint counters ---
  const rows = [];
  const paths = Array.from(new Set([
    ...Object.keys(snap.requests || {}),
    ...Object.keys(snap.errors_total || {}),
  ])).sort(new Intl.Collator('de', { sensitivity: 'base', numeric: true }).compare);
  for (const p of paths) {
    const n = snap.requests[p] || 0;
    const errs = snap.errors_total[p] || 0;
    rows.push(`<tr><td data-label="path">${esc(p)}</td><td class="num" data-label="requests">${n}</td>` +
      `<td class="num" data-label="5xx" style="${errs ? 'color:var(--red)' : ''}">${errs}</td></tr>`);
  }
  $('endpoints-rows').innerHTML = rows.length
    ? rows.join('') : '<tr><td colspan="3" class="empty">— none yet —</td></tr>';

  // --- Loaded models ---
  const modelLoads = snap.model_loads || {};
  const mrows = (snap.models || []).map(m => {
    const warm = m.idle_sec < 60;
    const cold = modelLoads[m.name];
    const coldStr = cold
      ? `${cold.first}s / ~${cold.last5_avg}s avg (${cold.count})`
      : '—';
    return `<tr>
      <td data-label="name">${esc(m.name)}</td>
      <td data-label="device">${esc(m.device || '—')}</td>
      <td data-label="compute">${esc(m.compute_type || '—')}</td>
      <td class="num" data-label="VRAM (MB)">${m.vram_mb != null ? m.vram_mb.toFixed(0) : '—'}</td>
      <td data-label="state"><span class="badge ${warm ? 'warm' : 'cold'}">${warm ? 'warm' : 'cold'}</span></td>
      <td class="num" data-label="age">${fmtSec(m.age_sec)}</td>
      <td class="num" data-label="idle">${fmtSec(m.idle_sec)}</td>
      <td class="num" data-label="cold-load">${coldStr}</td>
    </tr>`;
  });
  $('models-rows').innerHTML = mrows.length
    ? mrows.join('') : '<tr><td colspan="8" class="empty">— no models loaded —</td></tr>';

  // --- Recent transcriptions ---
  const rt = snap.recent_transcriptions || [];
  $('rt-n').textContent = rt.length;
  if (rt.length === 0) {
    $('rt-rows').innerHTML =
      '<tr><td colspan="7" class="empty">— no requests yet —</td></tr>';
  } else {
    $('rt-rows').innerHTML = rt.map(r => `<tr>
      <td class="ts" data-label="when" data-ts="${r.ts || 0}" title="${absTime(r.ts)}">${fmtWhen(r.ts)}</td>
      <td data-label="model">${esc(r.model)}</td>
      <td class="num" data-label="audio">${r.audio_dur.toFixed(1)} s</td>
      <td class="num" data-label="wall">${r.proc_dur.toFixed(2)} s</td>
      <td class="num" data-label="RTF">${r.rtf != null ? r.rtf.toFixed(2) + '×' : '—'}</td>
      <td class="num" data-label="words">${r.words}</td>
      <td data-label="status"><span class="badge ${r.status === 'ok' ? 'ok' : 'err'}">${r.status}</span></td>
    </tr>`).join('');
  }

  // Severity pills are driven by SEV_POLLER_JS injected at body-end
  // (5-s poll of /sev), so no per-tick update needed here.
}

// --- SSE consumer ----------------------------------------------------------
let es = null;
let recoveryTimer = null;
const statusEl = $('status');

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = 'pill ' + cls;
}

function openStream() {
  if (es) { try { es.close(); } catch {} }
  // EventSource sends the session cookie automatically (same-origin).
  es = new EventSource('/stats/stream');
  es.onmessage = (e) => {
    try {
      const snap = JSON.parse(e.data);
      pushHistory(snap);
      render(snap);
      setStatus('live', 'live');
    } catch (err) {
      console.warn('[stats] parse error', err);
    }
  };
  es.onerror = () => {
    setStatus('reconnecting…', 'paused');
    // Service may have restarted. Mirror /settings: poll a cheap idempotent
    // endpoint until it 200s, then force-reopen the SSE. Back off on repeated
    // failures (1.5s → ×1.7 → cap 30s) so a genuine outage doesn't hammer.
    if (recoveryTimer) return;
    let delay = 1500;
    const probe = async () => {
      try {
        const r = await fetch('/v1/models', { cache: 'no-store' });
        if (r.ok) {
          clearTimeout(recoveryTimer);
          recoveryTimer = null;
          // Drop history — server uptime jumped, the gap would be misleading.
          histX.length = 0;
          for (const k in hist) hist[k].length = 0;
          openStream();
          return;
        }
      } catch {}
      delay = Math.min(delay * 1.7, 30000);
      recoveryTimer = setTimeout(probe, delay);
    };
    recoveryTimer = setTimeout(probe, delay);
  };
}

// Visibility handler: closes the SSE on hidden tabs to defeat the browser's
// 6-connection-per-origin cap. Reopens on visible. Also cancels any in-flight
// recovery poll — otherwise a poll that succeeds in the background would
// openStream() concurrently with the visibility re-open, racing two
// EventSources for the same gid until one was orphaned.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    if (es) { try { es.close(); } catch {} es = null; }
    if (recoveryTimer) { clearInterval(recoveryTimer); recoveryTimer = null; }
    setStatus('paused (hidden)', 'paused');
  } else {
    openStream();
  }
});

// Auth rides the HttpOnly session cookie, sent automatically on both the
// fetch and the EventSource (same-origin) — no manual header or ?key= param.

// Initial fetch so the page renders before the first SSE tick arrives.
// role-admin used to be added here unconditionally — that leaked admin
// chrome to non-admins. OPEN_MODE_BANNER_JS is now the single source
// of truth (it sets role-admin iff whoami.is_admin=true).
fetch('/stats/snapshot', { cache: 'no-store' })
  .then(r => r.ok ? r.json() : null)
  .then(snap => {
    if (!snap) return;
    pushHistory(snap); render(snap);
  })
  .catch(err => console.warn('[stats] initial fetch failed', err))
  .finally(openStream);

})();
</script>

<script>
// --- Usage-over-time section (independent of the live SSE dashboard) -------
// Self-contained IIFE: builds its own uPlot time-series chart + leaderboard,
// fetched once on load and on every selector change. The chart formats its own
// axis/tooltip dates inline rather than via the shared TIME_HELPERS_JS fmtWhen
// (now injected lower down for the recent-transcriptions table): its x-values
// are day*86400 = UTC midnight of each bucket's calendar date, so labels read
// the date with getUTC* (correct on any operator timezone) — fmtWhen would
// apply the local clock, which is meaningless for a whole-day bucket.
(() => {
'use strict';
const $ = id => document.getElementById(id);
const chartEl = $('usage-plot');
const tipEl = $('usage-tip');
if (!chartEl || typeof uPlot === 'undefined') return;

// Per-entity line palette; 'others' is a dim dashed grey. curLines/curMetric
// are shared by buildChart, the tooltip, and the leaderboard swatches.
const PALETTE = ['#79c0ff','#7ee787','#f2cc60','#d2a8ff','#ff7b72','#56d4dd','#e3b341','#ff9bce'];
const OTHERS_COLOR = '#6e7681';
let curLines = [];
let curMetric = 'audio_s';

function remPx(n) {
  const base = parseFloat(getComputedStyle(document.documentElement).fontSize) || 15;
  return Math.round(n * base);
}
const MONO = 'Consolas, "Cascadia Code", "JetBrains Mono", Menlo, ui-monospace, monospace';

function fmtCount(n) {
  n = Number(n || 0);
  if (n >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, '') + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'k';
  return String(Math.round(n));
}
function fmtDur(sec) {
  sec = Number(sec || 0);
  if (sec < 60) return sec.toFixed(sec < 10 ? 1 : 0) + 's';
  if (sec < 3600) return (sec / 60).toFixed(1).replace(/\.0$/, '') + 'm';
  if (sec < 86400) return (sec / 3600).toFixed(1).replace(/\.0$/, '') + 'h';
  return (sec / 86400).toFixed(1).replace(/\.0$/, '') + 'd';
}
function fmtMetric(metric, v) {
  return metric === 'audio_s' ? fmtDur(v) : fmtCount(v);
}
function fmtDate(ts) {
  const d = new Date(ts * 1000), p2 = n => ('0' + n).slice(-2);
  return d.getUTCFullYear() + '.' + p2(d.getUTCMonth() + 1) + '.' + p2(d.getUTCDate());
}

// Floating cursor tooltip: bucket date + every line's value, with the line
// nearest the cursor (by data-space y distance) bolded. Driven by uPlot's
// setCursor hook; hidden when the cursor leaves the plot (idx == null).
function updateTip(u) {
  const idx = u.cursor.idx;
  if (idx == null || !curLines.length) { tipEl.style.display = 'none'; return; }
  const xs = u.data[0];
  const cy = u.posToVal(u.cursor.top, 'y');
  let focus = -1, best = Infinity;
  for (let s = 0; s < curLines.length; s++) {
    const v = u.data[s + 1][idx];
    if (v == null) continue;
    const d = Math.abs(v - cy);
    if (d < best) { best = d; focus = s; }
  }
  let html = '<div class="tip-date">' + fmtDate(xs[idx]) + '</div>';
  for (let s = 0; s < curLines.length; s++) {
    const ln = curLines[s];
    html += '<div class="tip-row' + (s === focus ? ' focus' : '') + '">'
      + '<span class="usage-swatch" style="background:' + ln.color + '"></span>'
      + '<span>' + esc(ln.label) + '</span>'
      + '<span class="tip-val">' + fmtMetric(curMetric, u.data[s + 1][idx]) + '</span>'
      + '</div>';
  }
  tipEl.innerHTML = html;
  tipEl.style.display = 'block';
  // Position in VIEWPORT space (the tooltip is position:fixed, so the card's
  // overflow:auto can't clip it near the chart edges). uPlot cursor coords are
  // relative to the plot over-element; add its viewport rect.
  const orect = u.over.getBoundingClientRect();
  const vw = document.documentElement.clientWidth;
  const vh = document.documentElement.clientHeight;
  const tw = tipEl.offsetWidth, th = tipEl.offsetHeight;
  let left = orect.left + u.cursor.left + 14;
  if (left + tw + 4 > vw) left = orect.left + u.cursor.left - tw - 14;  // flip left near right edge
  let top = orect.top + u.cursor.top + 14;
  if (top + th + 4 > vh) top = orect.top + u.cursor.top - th - 14;      // flip up near bottom edge
  tipEl.style.left = Math.max(4, left) + 'px';
  tipEl.style.top = Math.max(4, top) + 'px';
}

// cursor.move hook: snap the vertical guide to the nearest data point's x-pixel
// so guide + highlight dot + tooltip coincide (uPlot's canonical snap, per the
// nearest-non-null demo). mLeft < 0 means the cursor is off-plot — leave it.
function snapToDataX(u, mLeft, mTop) {
  if (mLeft < 0) return [mLeft, mTop];
  const idx = u.valToIdx(u.posToVal(mLeft, 'x'));
  return [Math.round(u.valToPos(u.data[0][idx], 'x')), mTop];
}

let chart = null;
function buildChart() {
  if (chart) { chart.destroy(); chart = null; }
  const w = chartEl.clientWidth || 600;
  const h = chartEl.clientHeight || 220;
  const single = curLines.length === 1;
  const series = [{ value: (u, ts) => ts == null ? '' : fmtDate(ts) }].concat(
    curLines.map(ln => ({
      label: ln.label, stroke: ln.color,
      width: ln.others ? 1.25 : 1.5,
      dash: ln.others ? [4, 3] : undefined,
      fill: single ? ln.color + '22' : undefined,
      points: { show: single, size: 4 }, spanGaps: true,
    })));
  chart = new uPlot({
    width: w, height: h,
    padding: [remPx(0.5), remPx(0.6), remPx(0.2), remPx(0.4)],
    legend: { show: false },
    // drag:{x:false,y:false} removes uPlot's default drag-to-zoom.
    // points: a per-series highlight dot on the hovered bucket via uPlot's
    // .u-cursor-pt — leave `show` at its default (an element-FACTORY); passing
    // the boolean `show:true` makes initCursorPt's `instanceof HTMLElement`
    // check fail, so NO dot is ever created. We only style it (rem-sized disc,
    // --bg ring, fill = each line's colour) so every line gets a clear marker.
    // move: snapToDataX locks the vertical guide onto the nearest data point's
    // x-pixel, so the guide line, the dot, cursor.idx and the tooltip all land
    // on the SAME bucket (without it the guide trails the raw mouse and the
    // tooltip flips to the next day at the column's midpoint).
    cursor: {
      y: false, drag: { x: false, y: false }, move: snapToDataX,
      points: {
        size: remPx(0.6), width: remPx(0.13), stroke: '#0d1117',
        fill: (u, si) => (curLines[si - 1] && curLines[si - 1].color) || '#79c0ff',
      },
    },
    scales: { x: { time: true }, y: { range: { min: { pad: 0.05, mode: 1 }, max: { pad: 0.1, mode: 1 } } } },
    hooks: { setCursor: [updateTip] },
    axes: [
      { stroke: '#6e7681', grid: { stroke: '#21262d', width: 1 },
        ticks: { stroke: '#30363d', width: 1, size: 3 },
        font: remPx(0.733) + 'px ' + MONO,
        // Ticks on a clean whole-day calendar grid, NOT on array indices. The
        // data omits empty buckets, so the old "every Nth point" subsample (a) put
        // labels on uneven dates and (b) HALVED the label set at 64px-width
        // boundaries (ceil(n/floor(px/64))) — a stray scrollbar flipped 12 daily
        // labels to every-other-day. Instead pick a day-step from a curated ladder
        // (smallest whose pixel spacing ≥ ~65px) and emit ticks at t0 + k·step·day.
        // t0 is a UTC midnight, so every tick is a UTC midnight → the getUTC
        // `values` formatter below stays calendar-correct on any operator timezone.
        // Thins gracefully (1→2→3→7…) and is deterministic across rebuilds.
        splits: (u) => {
          const xs = u.data[0] || [];
          if (xs.length < 2) return xs.slice();
          // Width source MUST be u.bbox.width, NOT u.over.clientWidth: on every
          // rebuild (any selector change) uPlot runs splits before .u-over is laid
          // out, so over.clientWidth reads 0 and the old `|| 600` fallback forced a
          // narrow width → every-other-day labels that then stuck. bbox.width is set
          // synchronously from the passed width and is reliable; it's in DEVICE
          // pixels, so divide by devicePixelRatio to get CSS px for the gap test.
          const px = (u.bbox && u.bbox.width)
            ? u.bbox.width / (window.devicePixelRatio || 1)
            : (u.over.clientWidth || 600);
          const maxTicks = Math.max(2, Math.floor(px / remPx(4.3)));
          const t0 = xs[0], t1 = xs[xs.length - 1];
          const spanDays = Math.max(1, Math.round((t1 - t0) / 86400));
          const LADDER = [1, 2, 3, 7, 14, 30, 60, 90, 180, 365];
          let step = LADDER[LADDER.length - 1];
          for (const s of LADDER) { if (spanDays / s <= maxTicks) { step = s; break; } }
          const out = [];
          for (let t = t0; t <= t1 + 1; t += step * 86400) out.push(t);
          return out;
        },
        // Fixed MM.DD, locale-independent (matches the project's YYYY.MM.DD norm).
        // x-values are UTC midnight of each bucket day, so getUTC* yields the
        // server-intended calendar date on every operator timezone. The hover
        // tooltip carries the full year-qualified date.
        values: (u, splits) => splits.map(s => {
          const d = new Date(s * 1000);
          const p2 = n => ('0' + n).slice(-2);
          return p2(d.getUTCMonth() + 1) + '.' + p2(d.getUTCDate());
        }) },
      { stroke: '#6e7681', size: remPx(2.8), gap: 4,
        grid: { stroke: '#21262d', width: 1 },
        ticks: { stroke: '#30363d', width: 1, size: 3 },
        font: remPx(0.733) + 'px ' + MONO,
        values: (u, splits) => splits.map(v => fmtMetric(curMetric, v)) },
    ],
    series,
  }, [[]].concat(curLines.map(() => [])), chartEl);
  chart.over.addEventListener('mouseleave', () => { tipEl.style.display = 'none'; });
}

let _raf = 0;
new ResizeObserver(() => {
  if (_raf || !chart) return;
  _raf = requestAnimationFrame(() => {
    _raf = 0;
    const cw = chartEl.clientWidth, ch = chartEl.clientHeight;
    if (cw > 0 && ch > 0) chart.setSize({ width: cw, height: ch });
  });
}).observe(chartEl);

function renderBoard(board, by) {
  const tb = $('usage-board-rows');
  if (!board || !board.length) {
    tb.innerHTML = '<tr><td colspan="6" class="empty">— no usage in this window —</td></tr>';
    return;
  }
  // Swatch the rows that are charted (top-K) so the table ties to the lines.
  const colorById = {};
  curLines.forEach(ln => { if (!ln.others) colorById[ln.id] = ln.color; });
  tb.innerHTML = board.map((r, i) => {
    const sw = colorById[r.id]
      ? '<span class="usage-swatch" style="background:' + colorById[r.id] + '"></span>' : '';
    const sub = by === 'key' && r.user_label
      ? '<span class="sub">' + esc(r.user_label) + '</span>' : '';
    return '<tr>'
      + '<td class="rank" data-label="#">' + (i + 1) + '</td>'
      + '<td class="name" data-label="name">' + sw + esc(r.label || '?') + sub + '</td>'
      + '<td class="num" data-label="requests">' + fmtCount(r.requests) + '</td>'
      + '<td class="num" data-label="words">' + fmtCount(r.words) + '</td>'
      + '<td class="num" data-label="audio">' + fmtDur(r.audio_s) + '</td>'
      + '<td class="num" data-label="err">' + fmtCount(r.errors) + '</td>'
      + '</tr>';
  }).join('');
}
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function segVal(id) {
  const g = $(id);
  const b = g && g.querySelector('button.active');
  return b ? b.dataset.v : '';
}

let _seq = 0;
function loadUsage() {
  const days = segVal('usage-range');
  const bucket = segVal('usage-bucket');
  const metric = segVal('usage-metric');
  const by = segVal('usage-by');
  const q = '?days=' + encodeURIComponent(days)
          + '&bucket=' + encodeURIComponent(bucket)
          + '&metric=' + encodeURIComponent(metric)
          + '&by=' + encodeURIComponent(by);
  const mine = ++_seq;
  fetch('/stats/usage' + q, { cache: 'no-store' })
    .then(r => r.ok ? r.json() : null)
    .then(j => {
      if (!j || mine !== _seq) return;   // stale response — a newer change won
      curMetric = j.metric || metric;
      curLines = (j.lines || []).map((ln, i) => ({
        id: ln.id, label: ln.label, values: ln.values, others: !!ln.others,
        color: ln.others ? OTHERS_COLOR : PALETTE[i % PALETTE.length],
      }));
      const xs = (j.days || []).map(d => d * 86400);
      if (!curLines.length || !xs.length) {
        if (chart) { chart.destroy(); chart = null; }
        tipEl.style.display = 'none';
        renderBoard(j.leaderboard, by);
        return;
      }
      buildChart();
      chart.setData([xs].concat(curLines.map(l => l.values)));
      renderBoard(j.leaderboard, by);
    })
    .catch(err => {
      console.warn('[stats] usage fetch failed', err);
      $('usage-board-rows').innerHTML =
        '<tr><td colspan="6" class="empty">— usage unavailable —</td></tr>';
    });
}

['usage-range', 'usage-bucket', 'usage-metric', 'usage-by'].forEach(id => {
  const g = $(id);
  if (!g) return;
  g.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b || !g.contains(b) || b.classList.contains('active')) return;
    g.querySelectorAll('button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    loadUsage();
  });
});
loadUsage();
})();
</script>
{{TIME_HELPERS_JS}}
{{SCALE_PICKER_JS}}
{{SEV_POLLER_JS}}
<script>
// Runs AFTER TIME_HELPERS_JS defines timeTick. Ages the relative suffix on the
// recent-transcriptions WHEN cells between SSE snapshots; re-queries [data-ts]
// each tick so it also catches rows added by the next snapshot render.
timeTick('#rt-rows [data-ts]');
</script>
</body></html>"""
