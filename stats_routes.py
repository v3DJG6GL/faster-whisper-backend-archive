"""
/stats system overview dashboard.

Routes (all gated by web_common.require_allowed_host(cfg.STATS_ALLOWED_HOSTS)):

  GET /stats           HTML page (single-file inline-HTML+CSS+JS, mirrors /logs)
  GET /stats/snapshot  one-shot JSON: metrics_snapshot() + system_snapshot()
  GET /stats/stream    SSE: same JSON, ~1 Hz, with periodic ': keepalive\\n\\n'

Access control: loopback always allowed; cfg.STATS_ALLOWED_HOSTS adds
extra IPs/CIDRs. The dependency reads cfg at request time so the admin
WebUI can broaden access without a restart.

Live updates: SSE rather than polling so we get free auto-reconnect on
service-restart, matching the /logs page UX.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, StreamingResponse

import config as cfg
import metrics
import system_stats
import web_common

logger = logging.getLogger("whisper-api")

router = APIRouter()

_require_stats_host = web_common.require_allowed_host(lambda: cfg.STATS_ALLOWED_HOSTS)


def _build_payload() -> dict[str, Any]:
    """Combine request metrics + system snapshot into one payload."""
    return {
        "ts": time.time(),
        **metrics.metrics_snapshot(),
        **system_stats.system_snapshot(),
        "severity": web_common.severity_counts(),
    }


@router.get("/stats", response_class=HTMLResponse, dependencies=[Depends(_require_stats_host)])
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


@router.get("/stats/snapshot", dependencies=[Depends(_require_stats_host)])
async def stats_snapshot() -> dict[str, Any]:
    """One-shot JSON. Useful for scripts and for the page's initial render."""
    return _build_payload()


@router.get("/stats/stream", dependencies=[Depends(_require_stats_host)])
async def stats_stream() -> StreamingResponse:
    """1 Hz SSE stream of the snapshot payload. Sends ': keepalive\\n\\n'
    every 15 s to defeat idle-proxy timeouts."""
    async def gen():
        last_keepalive = time.time()
        while True:
            payload = _build_payload()
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(1.0)
            now = time.time()
            if now - last_keepalive > 15.0:
                yield ": keepalive\n\n"
                last_keepalive = now

    return StreamingResponse(gen(), media_type="text/event-stream")


# --- HTML template -----------------------------------------------------------
# Single-file, no build step. Mirrors the /logs and /config style. uPlot is
# loaded from the local /static mount — no CDN, works offline.

_STATS_VIEWER_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>faster-whisper-backend · stats</title>
<link rel="stylesheet" href="/static/uplot.min.css">
<script src="/static/uplot.iife.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --fg: #c9d1d9; --dim: #6e7681;
    --cyan: #79c0ff; --green: #7ee787; --yellow: #f2cc60;
    --red: #ff7b72; --magenta: #d2a8ff; --bold: #f0f6fc;
    --border: #30363d;
  }
  html { color-scheme: dark; }
  html, body { background: var(--bg); color: var(--fg);
    font: 16px/1.5 ui-monospace, "Cascadia Code", Menlo, Consolas, monospace;
    margin: 0; padding: 0; min-height: 100%; }
  header { position: sticky; top: 0; background: var(--panel); border-bottom: 1px solid var(--border);
    z-index: 10; padding: 0; }
  header > .header-inner { display: flex; gap: 12px; align-items: center;
    max-width: 1100px; margin: 0 auto; width: 100%; padding: 8px 14px;
    box-sizing: border-box; }
  header .title { font-weight: 600; color: var(--bold); }
  header .pill { padding: 2px 8px; border-radius: 999px; background: #21262d; color: var(--dim);
    font-size: 11px; }
  header .pill.live { color: var(--green); border: 1px solid #1f4d2a; }
  header .pill.paused { color: var(--yellow); border: 1px solid #4d3e1f; }
  {{NAV_CSS}}
  .grid { display: grid; gap: 12px;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    padding: 14px; max-width: 1100px; margin: 0 auto;
    box-sizing: border-box; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
    padding: 10px 12px; }
  .card.span2 { grid-column: span 2; }
  .card h3 { font-size: 11px; color: var(--dim); margin: 0 0 6px;
    text-transform: uppercase; letter-spacing: .05em; font-weight: 500; }
  .card .val { color: var(--bold); font-size: 22px; font-weight: 600; line-height: 1.1; }
  .card .val .sub { color: var(--dim); font-size: 12px; font-weight: normal; margin-left: 6px; }
  .card .meta { color: var(--dim); font-size: 11px; margin-top: 4px; }
  .card .meta b { color: var(--fg); font-weight: 500; }
  .bar { height: 6px; background: #21262d; border-radius: 3px; margin-top: 6px; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: var(--cyan);
    transition: width .3s ease; }
  .bar.warn > i { background: var(--yellow); }
  .bar.crit > i { background: var(--red); }
  .spark-wrap  { margin-top: 10px; }
  .spark-head  { display: flex; justify-content: space-between; align-items: baseline;
                 font: 11px ui-monospace, "Cascadia Code", Menlo, monospace;
                 color: var(--dim); margin-bottom: 2px; }
  .spark-label { letter-spacing: .03em; text-transform: uppercase; }
  .spark-now   { color: var(--bold); font-weight: 600;
                 font-variant-numeric: tabular-nums; }
  .spark       { height: 72px; }
  .uplot, .u-wrap { background: transparent !important; }
  .u-legend { display: none; }
  .u-axis { color: var(--dim); }
  table.tbl { width: 100%; border-collapse: collapse; font-size: 12px; }
  table.tbl th, table.tbl td { padding: 4px 6px; text-align: left;
    border-bottom: 1px solid #21262d; }
  table.tbl th { color: var(--dim); font-weight: 500; font-size: 11px;
    text-transform: uppercase; }
  table.tbl td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .badge { display: inline-block; font-size: 10px; padding: 1px 6px;
    border-radius: 999px; border: 1px solid var(--border); color: var(--dim); }
  .badge.warm { color: var(--green); border-color: #1f4d2a; }
  .badge.cold { color: var(--yellow); border-color: #4d3e1f; }
  .badge.ok { color: var(--green); border-color: #1f4d2a; }
  .badge.err { color: var(--red); border-color: #5a2424; }
  .ts { color: var(--dim); font-variant-numeric: tabular-nums; }
  .core-strip { display: flex; gap: 2px; margin-top: 6px; height: 24px;
    align-items: flex-end; }
  .core-strip > div { flex: 1; background: var(--cyan); border-radius: 1px;
    min-height: 2px; transition: height .3s ease; }
  .err-strip { display: flex; gap: 4px; margin-top: 6px; }
  .err-strip .seg { flex: 1; text-align: center; padding: 6px;
    background: #21262d; border-radius: 4px; }
  .err-strip .seg b { color: var(--bold); display: block; font-size: 18px; }
  .err-strip .seg.hot { background: #2d1414; }
  .err-strip .seg.hot b { color: var(--red); }
  .empty { color: var(--dim); font-style: italic; }
  .hidden { display: none !important; }
</style></head>
<body>
<header><div class="header-inner">
  <span class="title">faster-whisper-backend · stats</span>
  {{NAV}}
  <span class="spacer"></span>
  <span id="status" class="pill live">live</span>
</div></header>

<div id="grid" class="grid">
  <!-- GPU -->
  <div id="card-gpu" class="card span2">
    <h3>GPU</h3>
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

  <div id="card-gpu-missing" class="card hidden">
    <h3>GPU</h3>
    <div class="empty">NVML unavailable — running on CPU or pynvml not installed.</div>
    <div id="gpu-error" class="meta"></div>
  </div>

  <!-- Host CPU -->
  <div class="card">
    <h3>CPU (host)</h3>
    <div id="cpu-pct" class="val">—<span class="sub">%</span></div>
    <div id="cpu-cores" class="core-strip"></div>
    <div class="spark-wrap">
      <div class="spark-head"><span class="spark-label">CPU %</span>
        <span id="cpu-now" class="spark-now">—</span></div>
      <div id="cpu-spark" class="spark"></div>
    </div>
  </div>

  <!-- Host RAM -->
  <div class="card">
    <h3>RAM</h3>
    <div id="ram-val" class="val">— <span class="sub">/ — GB</span></div>
    <div id="ram-bar" class="bar"><i style="width:0"></i></div>
    <div id="ram-meta" class="meta"></div>
    <div class="spark-wrap">
      <div class="spark-head"><span class="spark-label">RAM used %</span>
        <span id="ram-now" class="spark-now">—</span></div>
      <div id="ram-spark" class="spark"></div>
    </div>
  </div>

  <!-- Process -->
  <div class="card">
    <h3>Process</h3>
    <div id="proc-rss" class="val">—<span class="sub">MB RSS</span></div>
    <div id="proc-meta" class="meta"></div>
  </div>

  <!-- In-flight + uptime -->
  <div class="card">
    <h3>Activity</h3>
    <div id="inflight-val" class="val">0<span class="sub">in flight</span></div>
    <div id="activity-meta" class="meta"></div>
  </div>

  <!-- Latency -->
  <div class="card">
    <h3>Request latency (last <span id="lat-n">0</span>)</h3>
    <div id="lat-val" class="val">— <span class="sub">ms p50</span></div>
    <div id="lat-meta" class="meta"></div>
    <div class="spark-wrap">
      <div class="spark-head"><span class="spark-label">p50 latency (ms)</span>
        <span id="lat-now" class="spark-now">—</span></div>
      <div id="lat-spark" class="spark"></div>
    </div>
  </div>

  <!-- Errors window -->
  <div class="card">
    <h3>Errors (5xx)</h3>
    <div class="err-strip">
      <div id="err-1m" class="seg"><b>0</b>1 min</div>
      <div id="err-5m" class="seg"><b>0</b>5 min</div>
      <div id="err-15m" class="seg"><b>0</b>15 min</div>
    </div>
    <div id="err-meta" class="meta"></div>
  </div>

  <!-- Endpoint counters -->
  <div class="card span2">
    <h3>Endpoint counters</h3>
    <table class="tbl"><thead><tr><th>path</th><th class="num">requests</th><th class="num">5xx</th></tr></thead>
    <tbody id="endpoints-rows"></tbody></table>
  </div>

  <!-- Loaded models -->
  <div class="card span2">
    <h3>Loaded models</h3>
    <table class="tbl"><thead><tr>
      <th>name</th><th>device</th><th>compute</th>
      <th class="num">VRAM (MB)</th><th>state</th>
      <th class="num">age</th><th class="num">idle</th>
      <th class="num">cold-load</th>
    </tr></thead><tbody id="models-rows"></tbody></table>
  </div>

  <!-- Recent transcriptions -->
  <div class="card span2">
    <h3>Recent transcriptions (last <span id="rt-n">0</span>)</h3>
    <table class="tbl"><thead><tr>
      <th>when</th><th>model</th>
      <th class="num">audio</th><th class="num">wall</th><th class="num">RTF</th>
      <th class="num">words</th><th>status</th>
    </tr></thead><tbody id="rt-rows"><tr><td colspan="7" class="empty">— no requests yet —</td></tr></tbody></table>
  </div>
</div>

<script>
(() => {
'use strict';

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
const sparks = {};   // name -> uPlot instance
function makeSpark(elId, color, opts={}) {
  const el = document.getElementById(elId);
  if (!el) return null;
  const w = el.clientWidth || 240;
  const yScale = opts.range
    ? { range: opts.range }
    : { range: { min: { pad: 0.1, mode: 1 }, max: { pad: 0.1, mode: 1 } } };
  return new uPlot({
    width: w, height: 72, padding: [4, 6, 0, 0],
    cursor: { show: false },
    legend: { show: false },
    select: { show: false },
    scales: { x: { time: false }, y: yScale },
    axes: [
      { show: false },
      { show: true, size: 28, gap: 4,
        font: '10px ui-monospace, Menlo, monospace',
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
function fmtAgo(ts) {
  return fmtSec(Math.max(0, Date.now() / 1000 - ts)) + ' ago';
}
function setBar(barEl, pct) {
  const bar = barEl.querySelector('i');
  bar.style.width = Math.max(0, Math.min(100, pct)).toFixed(1) + '%';
  barEl.classList.toggle('warn', pct >= 75 && pct < 90);
  barEl.classList.toggle('crit', pct >= 90);
}
// Update one of the nav-row severity pills. Keeps the link, only edits the
// inner number span; flashes red on count increase. See web_common.py for
// the server-rendered HTML structure these IDs target.
function setPill(id, n) {
  const el = document.getElementById(id); if (!el) return;
  const numEl = el.querySelector('.n');
  const prev = +numEl.textContent;
  numEl.textContent = n;
  el.classList.toggle('hot', n > 0);
  el.classList.toggle('zero', n === 0);
  if (n > prev) {
    el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
  }
}

// --- Render ----------------------------------------------------------------
function render(snap) {
  ensureSparks();

  // --- GPU ---
  if (snap.gpu) {
    $('card-gpu').classList.remove('hidden');
    $('card-gpu-missing').classList.add('hidden');
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
    $('card-gpu').classList.add('hidden');
    $('card-gpu-missing').classList.remove('hidden');
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
  ])).sort();
  for (const p of paths) {
    const n = snap.requests[p] || 0;
    const errs = snap.errors_total[p] || 0;
    rows.push(`<tr><td>${p}</td><td class="num">${n}</td>` +
      `<td class="num" style="${errs ? 'color:var(--red)' : ''}">${errs}</td></tr>`);
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
      <td>${m.name}</td>
      <td>${m.device || '—'}</td>
      <td>${m.compute_type || '—'}</td>
      <td class="num">${m.vram_mb != null ? m.vram_mb.toFixed(0) : '—'}</td>
      <td><span class="badge ${warm ? 'warm' : 'cold'}">${warm ? 'warm' : 'cold'}</span></td>
      <td class="num">${fmtSec(m.age_sec)}</td>
      <td class="num">${fmtSec(m.idle_sec)}</td>
      <td class="num">${coldStr}</td>
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
    $('rt-rows').innerHTML = rt.slice().reverse().map(r => `<tr>
      <td class="ts">${fmtAgo(r.ts)}</td>
      <td>${r.model}</td>
      <td class="num">${r.audio_dur.toFixed(1)} s</td>
      <td class="num">${r.proc_dur.toFixed(2)} s</td>
      <td class="num">${r.rtf != null ? r.rtf.toFixed(2) + '×' : '—'}</td>
      <td class="num">${r.words}</td>
      <td><span class="badge ${r.status === 'ok' ? 'ok' : 'err'}">${r.status}</span></td>
    </tr>`).join('');
  }

  // --- Severity pills (in nav row) ---
  // Server already computed the 60-s window in snap.severity; just push.
  if (snap.severity) {
    setPill('sev-warn', snap.severity.warn || 0);
    setPill('sev-err',  snap.severity.err  || 0);
    setPill('sev-crit', snap.severity.crit || 0);
  }
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
    // Service may have restarted. Mirror /config: poll a cheap idempotent
    // endpoint until it 200s, then force-reopen the SSE.
    if (recoveryTimer) return;
    recoveryTimer = setInterval(async () => {
      try {
        const r = await fetch('/v1/models', { cache: 'no-store' });
        if (r.ok) {
          clearInterval(recoveryTimer);
          recoveryTimer = null;
          // Drop history — server uptime jumped, the gap would be misleading.
          histX.length = 0;
          for (const k in hist) hist[k].length = 0;
          openStream();
        }
      } catch {}
    }, 1500);
  };
}

// Visibility handler: closes the SSE on hidden tabs to defeat the browser's
// 6-connection-per-origin cap. Reopens on visible.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    if (es) { try { es.close(); } catch {} es = null; }
    setStatus('paused (hidden)', 'paused');
  } else {
    openStream();
  }
});

// Initial fetch so the page renders before the first SSE tick arrives.
fetch('/stats/snapshot', { cache: 'no-store' })
  .then(r => r.json())
  .then(snap => { pushHistory(snap); render(snap); })
  .catch(err => console.warn('[stats] initial fetch failed', err))
  .finally(openStream);

})();
</script>
</body></html>"""
