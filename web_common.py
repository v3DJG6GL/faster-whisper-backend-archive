"""
Shared helpers used by the /logs, /config, and /stats web pages.

  - require_allowed_host(allowlist) — FastAPI dependency that 403s callers
    not in the allowlist. Allowlist accepts bare IPs or CIDRs.
  - nav_html(current, request)      — server-rendered nav row HTML.
  - severity_counts()               — log-level counters (last 60 s).
"""

from __future__ import annotations

import ipaddress
import logging
import time
from collections import deque
from typing import Callable

from fastapi import HTTPException, Request, status

import config as cfg


# IPv4-mapped-in-IPv6 prefix surfaces on Windows dual-stack `::` binds when a
# v4 client connects, e.g. "::ffff:127.0.0.1". `ipaddress.ip_address` already
# parses these, but the .ipv4_mapped attribute is what we actually compare on.
def _to_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _build_networks(allowlist: list[str]) -> list[ipaddress._BaseNetwork]:
    nets: list[ipaddress._BaseNetwork] = []
    for entry in allowlist or []:
        entry = entry.strip()
        if not entry:
            continue
        try:
            nets.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            # Bad entry — skip silently. The /config endpoint validates inputs;
            # this is a runtime defense for handwritten config edits.
            continue
    return nets


def require_allowed_host(allowlist_ref: Callable[[], list[str]]) -> Callable[[Request], None]:
    """Returns a FastAPI dependency that rejects callers outside the allowlist.

    `allowlist_ref` is a zero-arg callable that returns the current allowlist —
    NOT the list itself. This indirection matters: the admin WebUI can edit
    cfg.ADMIN_ALLOWED_HOSTS at runtime, and we want the next request to pick
    up the new value without re-creating the dependency.

    Loopback (`127.0.0.1`, `::1`) is ALWAYS allowed in addition to the
    configured list, so a misconfigured CIDR can never lock the local
    operator out of /config — they can still fix the entry from the box.
    """

    def _dep(request: Request) -> None:
        client = request.client
        if client is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "no client info")
        ip = _to_ip(client.host)
        if ip is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "unparseable client host")
        if ip.is_loopback:
            return
        for net in _build_networks(allowlist_ref()):
            if ip in net:
                return
        raise HTTPException(status.HTTP_403_FORBIDDEN, "host not in allowlist")

    return _dep


# --- Severity ring (in-memory log-level counts, last 60s) ---------------------
# A logging.Handler appends (timestamp, levelno) on every record. The /stats
# page and the nav row read severity_counts() at request time. Bounded ring
# (maxlen=2000) keeps memory predictable under burst logging.
_SEVERITY_LOG: deque[tuple[float, int]] = deque(maxlen=2000)


class SeverityCounter(logging.Handler):
    """Append (time, levelno) to the in-memory severity ring on every record.

    Attached alongside the existing console+file handlers in main.py. WARNING-
    and-up only, so the ring stays small under chatty INFO-level traffic."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _SEVERITY_LOG.append((record.created, record.levelno))
        except Exception:
            # Never let a logging failure kill the request that triggered it.
            pass


def severity_counts(window_sec: float = 60.0) -> dict[str, int]:
    """Return {warn, err, crit} counts from the last `window_sec` seconds."""
    cutoff = time.time() - window_sec
    warn = err = crit = 0
    # Iterate from the right (newest first) and break once we cross the
    # cutoff — the deque is append-ordered, so older entries follow.
    for ts, lvl in reversed(_SEVERITY_LOG):
        if ts < cutoff:
            break
        if lvl >= logging.CRITICAL:
            crit += 1
        elif lvl >= logging.ERROR:
            err += 1
        elif lvl >= logging.WARNING:
            warn += 1
    return {"warn": warn, "err": err, "crit": crit}


# --- Nav row + severity pills ------------------------------------------------

# Inline CSS so each page can drop the nav into its existing <header> without
# duplicating styles. Color tokens reuse the page-level CSS vars.
#
# `header .spacer { flex: 1 }` — single canonical spacer rule. Pages place
# `<span class="spacer"></span>` between the nav block and the action cluster
# so the right side stays right-aligned regardless of how many actions a page
# has.
NAV_CSS = """
/* Global scaling tokens — every page uses these so a single :root knob
   (`--fs-base`) re-scales the WHOLE UI. Bump --fs-base to scale up; the
   scale-picker dropdown writes inline-style to override at runtime.
   Spacing/padding everywhere uses rem so it scales with font. */
:root {
  --fs-base:  15px;
  --fs-xs:    0.733rem;   /* ~11px @ 15px base */
  --fs-sm:    0.8rem;     /* ~12px */
  --fs-md:    0.867rem;   /* ~13px */
  --fs-lg:    1rem;       /* 15px (= base) */
  --fs-xl:    1.2rem;     /* ~18px */
  --fs-xxl:   1.467rem;   /* ~22px */
}
html { font-size: var(--fs-base); color-scheme: dark; }
header .spacer { flex: 1; }
header .navrow { display: flex; gap: 0.25rem; }
header .navlink { padding: 0.1875rem 0.625rem; border-radius: 4px; color: var(--dim);
  text-decoration: none; font-size: var(--fs-sm); border: 1px solid transparent;
  flex-shrink: 0; white-space: nowrap; }
header .navlink:hover { background: #21262d; color: var(--fg); }
header .navlink.active { color: var(--bold); background: #21262d;
  border-color: var(--border); }
header .sevpill { font-size: var(--fs-xs); padding: 0.125rem 0.5rem; border-radius: 4px;
  border: 1px solid var(--border); color: var(--dim); text-decoration: none;
  display: inline-flex; gap: 0.25rem; align-items: baseline;
  flex-shrink: 0; white-space: nowrap; }
header .sevpill .n { font-variant-numeric: tabular-nums; }
header .sevpill.warn.hot { color: var(--yellow); border-color: #4d3e1f; }
header .sevpill.err.hot  { color: var(--red);    border-color: #5a2424; }
header .sevpill.crit.hot { color: var(--red);    border-color: #5a2424;
  background: #2d1414; }
header .sevpill.zero { opacity: 0.45; }
@keyframes sev-flash { 0% { background: #5a2424 } 100% { background: transparent } }
header .sevpill.flash { animation: sev-flash .6s ease-out; }
/* Scale picker dropdown — same dark-themed look as other selects.
   Inline SVG arrow keeps it portable across pages. */
header .scale-picker {
  background: #0d1117 url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'><path fill='%236e7681' d='M0 0l5 6 5-6z'/></svg>")
    no-repeat right 0.375rem center;
  color: var(--fg); border: 1px solid var(--border); border-radius: 4px;
  padding: 0.125rem 1.25rem 0.125rem 0.5rem;
  font: inherit; font-size: var(--fs-xs); cursor: pointer;
  appearance: none; -webkit-appearance: none;
  flex-shrink: 0;
}
"""


# Bootstrap script — applies the persisted UI scale BEFORE the page's CSS
# parses, avoiding a flash-of-default-size on every navigation. Belongs in
# <head> as the very first <script>.
SCALE_BOOTSTRAP_HEAD = (
    "<script>(function(){var v=localStorage.getItem('whisper-ui-fs-base');"
    "if(v)document.documentElement.style.setProperty('--fs-base',v+'px');})();</script>"
)


# Header dropdown HTML — placed just before the action cluster (logout etc.).
SCALE_PICKER_HTML = (
    '<select id="scale-picker" class="scale-picker" title="UI scale">'
    '<option value="13">90%</option>'
    '<option value="15" selected>100%</option>'
    '<option value="17">110%</option>'
    '<option value="18">120%</option>'
    '<option value="20">130%</option>'
    '</select>'
)


# Wire-up JS — placed at the end of <body>. Restores the saved value into
# the dropdown and persists future selections. Independent of the <head>
# bootstrap (which only sets the inline style); this binds the change handler.
SCALE_PICKER_JS = """
<script>(function(){
  var KEY='whisper-ui-fs-base';
  var sel=document.getElementById('scale-picker');
  if(!sel)return;
  var saved=localStorage.getItem(KEY);
  if(saved){sel.value=saved;}
  sel.addEventListener('change',function(){
    document.documentElement.style.setProperty('--fs-base',sel.value+'px');
    localStorage.setItem(KEY,sel.value);
  });
})();</script>
"""


def _nav_items(current: str) -> list[tuple[str, str, bool]]:
    """Return [(label, href, active), ...] honoring cfg.ADMIN_UI_ENABLED."""
    items: list[tuple[str, str, bool]] = [
        ("logs",  "/logs",  current == "logs"),
        ("stats", "/stats", current == "stats"),
    ]
    if getattr(cfg, "ADMIN_UI_ENABLED", False):
        items.append(("config", "/config", current == "config"))
    return items


def nav_html(current: str) -> str:
    """Render the nav row + severity pills as an HTML fragment.

    Pills link to /logs?filter=<level> so a click jumps to the relevant log
    rows. Counts of zero render dimmed; non-zero render colored ("hot")."""
    counts = severity_counts()
    parts: list[str] = ['<span class="navrow">']
    for label, href, active in _nav_items(current):
        cls = "navlink active" if active else "navlink"
        parts.append(f'<a class="{cls}" href="{href}">{label}</a>')
    parts.append("</span>")

    # Stable IDs let JS update just the .n inner span on each SSE tick without
    # rebuilding the link (preserves focus/click state). The initial counts
    # rendered here are a "best effort at page load" — the client takes over
    # immediately, so they're correct for the first render and live thereafter.
    for level, key in (("warn", "WARNING"), ("err", "ERROR"), ("crit", "CRITICAL")):
        n = counts[level]
        cls = f"sevpill {level} {'hot' if n else 'zero'}"
        title = f"{key}+ in the last 60 s — click to filter logs"
        parts.append(
            f'<a id="sev-{level}" class="{cls}" '
            f'href="/logs?filter={key}" title="{title}">'
            f'<span class="lbl">{level}</span> '
            f'<span class="n">{n}</span></a>'
        )
    return "".join(parts)


def render_page(template: str, current: str) -> str:
    """Substitute placeholders in a page template:
      - {{NAV}}                  → nav row + severity pills
      - {{NAV_CSS}}              → shared header/scale-token CSS
      - {{SCALE_PICKER}}         → scale dropdown (header)
      - {{SCALE_PICKER_JS}}      → wire-up script (end of body)
      - {{SCALE_BOOTSTRAP_HEAD}} → tiny pre-paint script (top of <head>)

    Pages that don't include a given placeholder are returned unchanged."""
    return (
        template
        .replace("{{NAV}}", nav_html(current))
        .replace("{{NAV_CSS}}", NAV_CSS)
        .replace("{{SCALE_PICKER}}", SCALE_PICKER_HTML)
        .replace("{{SCALE_PICKER_JS}}", SCALE_PICKER_JS)
        .replace("{{SCALE_BOOTSTRAP_HEAD}}", SCALE_BOOTSTRAP_HEAD)
    )
