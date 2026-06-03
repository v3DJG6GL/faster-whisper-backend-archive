"""
Shared helpers used by the /logs, /settings, /stats, and /quick-config pages.

  - require_allowed_host(allowlist) — FastAPI dependency that 403s callers
    not in the allowlist. Allowlist accepts bare IPs or CIDRs.
  - nav_html(current)               — server-rendered nav row HTML.
  - severity_counts()               — WARNING+ counts since process start
                                      (bounded by the 2000-entry ring).
"""

from __future__ import annotations

import ipaddress
import logging
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
            # Bad entry — skip silently. The /settings endpoint validates inputs;
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
    operator out of /settings — they can still fix the entry from the box.
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


# --- Severity ring (in-memory log-level counts, since process start) ---------
# A logging.Handler appends (timestamp, levelno) on every record. The /stats
# page and the nav row read severity_counts() at request time. Bounded ring
# (maxlen=2000) keeps memory predictable under burst logging — once the ring
# fills, oldest entries fall off and the per-level counters cap accordingly.
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


def severity_counts() -> dict[str, int]:
    """Return {warn, err, crit} counts since process start.

    Reads the entire in-memory _SEVERITY_LOG ring. The ring is bounded at
    2000 entries — under sustained WARNING+ traffic, oldest entries fall off
    and the counter caps. In practice this matches a per-run "session
    counter" the user investigates via the /logs?filter=<level> link on each
    pill. Restart resets to zero."""
    warn = err = crit = 0
    for _ts, lvl in _SEVERITY_LOG:
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
   Spacing/padding everywhere uses rem so it scales with font.
   Font stacks split chrome (sans) from code/values/log (mono): Segoe UI
   ships on every Windows since Vista; Consolas on every Windows; both
   listed first so we never fall through to Times New Roman / Courier New
   on boxes without ui-monospace or Cascadia Code installed. --help is
   one notch brighter than --dim for description text. */
:root {
  --fs-base:  15px;
  --fs-xs:    0.733rem;   /* ~11px @ 15px base */
  --fs-sm:    0.8rem;     /* ~12px */
  --fs-md:    0.867rem;   /* ~13px */
  --fs-lg:    1rem;       /* 15px (= base) */
  --fs-xl:    1.2rem;     /* ~18px */
  --fs-xxl:   1.467rem;   /* ~22px */
  /* system-ui resolves to the OS's actual UI font on every modern browser:
     Segoe UI on Windows, San Francisco on macOS, Plasma's chosen font on
     KDE (typically Noto Sans), Cantarell on GNOME. Explicit Linux names
     (Cantarell / Ubuntu / Noto Sans / DejaVu Sans / Liberation Sans) come
     before the generic `sans-serif` keyword because some Linux fontconfig
     setups alias `sans-serif` to a serif (DejaVu Serif / Liberation Serif),
     which previously rendered the WebUI as a "newspaper". */
  --font-sans: system-ui, -apple-system, "Segoe UI", Roboto, Inter,
               "Helvetica Neue", Cantarell, Ubuntu, "Noto Sans",
               "DejaVu Sans", "Liberation Sans", Arial, sans-serif;
  --font-mono: Consolas, "Cascadia Code", "JetBrains Mono", Menlo,
               ui-monospace, monospace;
  --help: #8b949e;
}
html { font-size: var(--fs-base); color-scheme: dark; }
/* Boundary marker for transcription values: dim brackets around the EXACT
   text, with internal whitespace preserved (pre-wrap) so a leading/trailing
   space shows as a literal gap inside the brackets and the begin/end of the
   transcription is unambiguous. Brackets are pseudo-elements — never part of
   the text — so they can't be mistaken for content. Ordinary spaces stay
   visible only as the gap they occupy; no per-space glyphs. */
.ws-region { white-space: pre-wrap; overflow-wrap: anywhere; }
.ws-region::before { content: "\\27E6"; color: var(--dim); }
.ws-region::after  { content: "\\27E7"; color: var(--dim); }
header .navrow { display: flex; align-items: center; gap: 0.25rem;
  flex-wrap: wrap; row-gap: 0.25rem; min-width: 0; }
header .navlink { padding: 0.3rem 0.7rem; border-radius: 6px; color: var(--dim);
  text-decoration: none; font-size: var(--fs-sm); line-height: 1.2;
  border: 1px solid transparent; flex-shrink: 0; white-space: nowrap;
  transition: background .12s ease, color .12s ease; }
header .navlink:hover { background: #21262d; color: var(--fg); }
/* Active page: accent text + subtle filled pill (redundant cue, not colour-
   only — server also sets aria-current="page"). */
header .navlink.active { color: var(--cyan); background: #1f2630;
  border-color: var(--border); font-weight: 600; }
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
/* ---- Two-tier sticky header (Carbon/Primer "global bar + page toolbar") ----
   Row 1 .header-inner = GLOBAL BAR, identical on every page:
     brand · │ · nav links ............ severity pills · scale picker
   Row 2 .subbar = PAGE TOOLBAR, page-specific controls (search/filter on the
     left, actions on the right). Omitted entirely on pages with no page
     actions, so the global bar never changes shape between pages.
   Container queries read in rem against the rendered header width, so they
   respect the --fs-base scale token (unlike @media). Page-local CSS styles
   `header button` / `header .pill`, which still match inside the subbar. */
header { position: sticky; top: 0; z-index: 10;
  background: var(--panel); border-bottom: 1px solid var(--border);
  box-shadow: 0 6px 20px -14px rgba(0,0,0,0.7);
  container-type: inline-size; container-name: hdr; }

/* row 1 — global bar */
header .header-inner { display: flex; align-items: center; gap: 0.75rem;
  flex-wrap: wrap; row-gap: 0.4rem;
  max-width: 68.75rem; margin: 0 auto; width: 100%;
  padding: 0.55rem 1rem; box-sizing: border-box; }
header .title { display: inline-flex; align-items: center; gap: 0.5rem;
  font-weight: 600; color: var(--bold); white-space: nowrap;
  flex-shrink: 1; min-width: 0; max-width: 22rem; overflow: hidden; }
header .brand-mark { width: 1.6em; height: 1.6em; flex-shrink: 0; display: block; }
header .brand-word { font-family: var(--font-mono); font-weight: 700;
  letter-spacing: 0; white-space: nowrap; }
header .brand-word .bw-a { color: var(--dim);  font-weight: 400; }
header .brand-word .bw-b { color: var(--bold); font-weight: 700; }
header .brand-word .bw-sep { color: var(--green); font-weight: 700; margin: 0 0.28em; }
header .brand-word .bw-c { color: var(--dim);  font-weight: 400; }
/* the requested clear separation between logo and nav */
header .brand-sep { flex-shrink: 0; width: 1px; align-self: stretch;
  margin: 0.15rem 0.35rem; background: var(--border); }
/* spacer pushes the utility cluster to the right edge */
header .spacer { flex: 1 1 0; min-width: 0.5rem; }
/* right-side utility cluster: severity pills + scale picker (same everywhere) */
header .hdr-right { display: flex; align-items: center; gap: 0.5rem; flex-shrink: 0; }
header .sevpills { display: inline-flex; align-items: center; gap: 0.25rem; }
/* Square icon buttons (logout / reload …) — same chrome as text buttons via
   the page-local `header button` rule, just sized to the glyph. Labels live
   in title + aria-label so the icon stays accessible. */
header .icon-btn { display: inline-flex; align-items: center; justify-content: center;
  padding: 0.3rem; line-height: 0; }
header .icon-btn svg { width: 1.1em; height: 1.1em; display: block; }
header .icon-btn:hover { color: var(--cyan); }
header .icon-btn:focus-visible { outline: 2px solid var(--cyan); outline-offset: 1px; }
/* `.auth-action` (the sign-out button) is hidden until a bearer exists; the
   author rule must beat `.icon-btn { display:inline-flex }` for [hidden]. */
header .auth-action[hidden] { display: none; }

/* row 2 — page toolbar. Shares the global bar's --panel surface so the
   whole header reads as ONE cohesive slab; a single internal hairline
   divides the two tiers. The page title anchors the left edge so the
   action cluster never floats alone over the page background (the old
   `background: var(--bg)` made this row the same colour as the page, so
   it looked like buttons stuck to a transparent strip). */
header .subbar { display: flex; align-items: center; gap: 0.6rem;
  flex-wrap: wrap; row-gap: 0.4rem;
  max-width: 68.75rem; margin: 0 auto; width: 100%;
  padding: 0.45rem 1rem; box-sizing: border-box;
  border-top: 1px solid var(--border); }
/* page title — fills the left, with a small green tick echoing the brand
   separator so the toolbar feels designed rather than empty-on-the-left. */
header .subbar-title { flex: 0 0 auto; display: inline-flex; align-items: center;
  gap: 0.5rem; font: 600 var(--fs-md)/1.2 var(--font-sans); color: var(--bold);
  white-space: nowrap; }
header .subbar-title::before { content: ""; flex: 0 0 auto;
  width: 3px; height: 0.95em; border-radius: 2px; background: var(--green); }
header .subbar-left  { display: flex; align-items: center; gap: 0.5rem;
  flex: 1 1 auto; min-width: 0; flex-wrap: wrap; row-gap: 0.4rem; }
header .subbar-right { display: flex; align-items: center; gap: 0.5rem;
  flex: 0 0 auto; margin-left: auto; flex-wrap: wrap; row-gap: 0.4rem;
  justify-content: flex-end; }
/* When a filter group (.subbar-left) is present it grows to push the action
   cluster to the right edge on its own, so `margin-left:auto` is not needed —
   and dropping it means that when the actions can't fit and wrap to a second
   line, they align under the title on the LEFT instead of floating at the
   bottom-right with an empty diagonal above them. */
header .subbar-left ~ .subbar-right { margin-left: 0; }
header .subbar #filter { flex: 1 1 auto; min-width: 8rem; max-width: 32rem; }
header .subbar .sep { align-self: stretch; width: 1px; background: var(--border);
  margin: 0.1rem 0.15rem; }

/* Canonical page-toolbar controls. Every page renders identical buttons,
   pills and filter inputs inside the header; page-local CSS no longer
   redefines `header button` / `header .pill` (they used to drift — e.g.
   /quick-config shrank its buttons to --fs-sm), so the toolbar now matches
   across all pages. This block is injected after each page's own <style>,
   so these rules win the cascade on equal specificity. */
header button { background: #21262d; color: var(--fg);
  border: 1px solid var(--border); border-radius: 4px;
  padding: 0.3rem 0.7rem; font: inherit; font-size: var(--fs-md);
  line-height: 1.3; cursor: pointer; white-space: nowrap; flex-shrink: 0; }
header button:hover:not(:disabled) { background: #30363d; color: var(--bold); }
header button:disabled { opacity: 0.4; cursor: not-allowed; }
header button.primary { background: #238636; border-color: #2ea043; color: var(--bold); }
header button.primary:hover:not(:disabled) { background: #2ea043; }
header button.danger:not(:disabled) { color: var(--red); border-color: #5a2424; }
header button.danger:not(:disabled):hover { background: #3a0d0d; border-color: #7d2d2d; }
header button#discard-btn:not(:disabled) { background: #3a0d0d;
  border-color: #5a2424; color: var(--red); }
header button#discard-btn:not(:disabled):hover { background: #531f1f; border-color: #7d2d2d; }
header .pill { padding: 0.125rem 0.5rem; border-radius: 4px; background: #21262d;
  color: var(--dim); font-size: var(--fs-xs); white-space: nowrap; flex-shrink: 0; }
header .pill.live, header .pill.ok     { color: var(--green);  border: 1px solid #1f4d2a; }
header .pill.paused, header .pill.warn { color: var(--yellow); border: 1px solid #4d3e1f; }
header .pill.err { color: var(--red); border: 1px solid #5a2424; }
header .subbar #status:not(.pill) { color: var(--dim); font-size: var(--fs-sm);
  min-width: 0; overflow: hidden; text-overflow: ellipsis; }
header .subbar label { display: inline-flex; align-items: center; gap: 0.3rem;
  font-size: var(--fs-sm); color: var(--help); white-space: nowrap; }
header .subbar select, header .subbar input[type="text"] {
  background: var(--input-bg); color: var(--fg);
  border: 1px solid var(--border); border-radius: 4px;
  padding: 0.25rem 0.4rem; font: inherit; font-size: var(--fs-sm);
  font-family: var(--font-sans); }
header .subbar input[type="text"] { min-width: 10rem; }
header .subbar .counts { color: var(--help); font-size: var(--fs-sm); white-space: nowrap; }
header .subbar .counts .n { color: var(--bold); font-weight: 600; }
header .subbar .capture-state { font-size: var(--fs-sm);
  padding: 0.15rem 0.5rem; border-radius: 4px; border: 1px solid var(--border); }
header .subbar .capture-state.on  { color: var(--green); border-color: #2d5a37; }
header .subbar .capture-state.off { color: var(--dim);   border-color: var(--border); }
header .subbar .filt-label { display: inline-flex; align-items: center; gap: 0.35rem;
  font-size: var(--fs-sm); color: var(--help); white-space: nowrap; }

/* responsive: shed the sevpill word labels, then tighten nav + drop divider */
@container hdr (max-width: 60rem) {
  header .sevpill .lbl { display: none; }
  header .sevpill { padding: 0.125rem 0.4rem; }
}
@container hdr (max-width: 40rem) {
  header .navlink { padding: 0.25rem 0.5rem; }
  header .brand-sep { display: none; }
}

/* ---- Admin-only nav elements ----
   logs/stats/settings nav links + sev pills are marked .admin-only at
   render time. Hidden by default; revealed when the page's JS adds
   `body.role-admin` after a successful auth-required state fetch.
   /settings, /logs, /stats add the class unconditionally (their state
   endpoints already require admin token). /quick-config adds it only
   when /quick-config/state returns role=admin — non-admin user keys
   never see admin-only chrome. */
header .admin-only { display: none; }
body.role-admin header .admin-only { display: inline-flex; }

/* Per-user-gated nav links (logs/stats/quick/reports/captures).
   Hidden until OPEN_MODE_BANNER_JS sets `.allowed` per link based on
   /auth/whoami → permissions.pages[X]. Admins still pass via the
   server-side is_admin short-circuit. */
header .page-link { display: none; }
header .page-link.allowed { display: inline-flex; }

/* No-access landing card — rendered by _renderNoAccessLanding() when
   the bearer is valid but the caller lacks permission for the current
   page. Replaces the page's <main> content. Lists every page the
   caller CAN reach as a button, plus a Sign-out fallback. */
.no-access-landing {
  max-width: 36rem; margin: 4rem auto; text-align: center;
  padding: 2rem; background: var(--panel);
  border: 1px solid var(--border); border-radius: 0.375rem;
}
.no-access-landing h2 {
  margin: 0 0 0.5rem; color: var(--bold);
}
.no-access-landing p { color: var(--help); }
.no-access-landing .landing-hint {
  margin: 1rem 0 0.25rem; font-size: var(--fs-sm);
}
.no-access-landing .landing-actions {
  margin: 0.5rem 0 0; display: flex; gap: 0.6rem;
  justify-content: center; flex-wrap: wrap;
}
.no-access-landing .landing-btn {
  color: var(--cyan); border: 1px solid var(--cyan);
  padding: 0.45rem 1rem; border-radius: 0.25rem;
  text-decoration: none; font-size: var(--fs-sm);
}
.no-access-landing .landing-btn:hover {
  background: rgba(121, 192, 255, 0.08);
}
.no-access-landing .landing-signout {
  margin-top: 1.1rem; padding-top: 0.8rem;
  border-top: 1px solid var(--border);
}
.no-access-landing button {
  background: var(--panel); border: 1px solid var(--border);
  color: var(--fg); padding: 0.45rem 1rem; border-radius: 4px;
  cursor: pointer; font: inherit; font-size: var(--fs-sm);
}
.no-access-landing button:hover {
  background: #21262d; color: var(--bold);
}

/* Tag-picker widget — shared between /settings rule editor and the
   /settings/api-keys permissions matrix. Renders as a single
   border-bounded row of pills + an inline text input.
   Validation: TAG_RE = lowercase a-z0-9- only, 1-32 chars, no
   leading/trailing hyphen. Bad input gets a red border via .invalid.
   The autocomplete dropdown is position:absolute relative to the
   picker, so each .tag-picker is position:relative. */
.tag-picker { position: relative; display: inline-block;
  min-width: 12rem; max-width: 24rem; vertical-align: middle; }
.tag-picker.disabled { opacity: 0.45; pointer-events: none; }
.tag-picker-pills { display: flex; flex-wrap: wrap; gap: 0.25rem;
  align-items: center; padding: 0.15rem 0.3rem;
  background: var(--input-bg); border: 1px solid var(--border);
  border-radius: 4px; min-height: 1.6rem; }
.tag-picker-pills:focus-within { border-color: var(--cyan, #58a6ff); }
.tag-pill { display: inline-flex; align-items: center; gap: 0.15rem;
  font-size: var(--fs-xs); padding: 0.05rem 0.4rem;
  background: rgba(121, 192, 255, 0.12);
  color: var(--cyan, #58a6ff);
  border-radius: 999px; border: 1px solid rgba(121, 192, 255, 0.3);
  font-family: var(--font-sans); white-space: nowrap; }
.tag-pill-x { background: none; border: 0;
  color: var(--cyan, #58a6ff); cursor: pointer;
  padding: 0 0.15rem; line-height: 1; font: inherit;
  font-size: var(--fs-sm); }
.tag-pill-x:hover { color: var(--red, #ff7b72); }
.tag-picker-input { flex: 1; min-width: 5rem; border: 0;
  background: transparent; color: var(--fg); font: inherit;
  font-size: var(--fs-sm); padding: 0.1rem 0.2rem; outline: 0;
  font-family: var(--font-sans); }
.tag-picker-input.invalid { color: var(--red, #ff7b72); }
.tag-picker-input::placeholder { color: var(--dim); }
.tag-picker-suggest { position: absolute; left: 0; top: 100%;
  z-index: 20; margin-top: 0.15rem;
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 4px; max-height: 12rem; overflow-y: auto;
  font-size: var(--fs-sm); min-width: 10rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4); }
.tag-picker-suggest-item { padding: 0.25rem 0.6rem;
  cursor: pointer; color: var(--fg); }
.tag-picker-suggest-item:hover {
  background: var(--input-bg); color: var(--cyan, #58a6ff);
}

/* Yellow warning badge for exposed-but-untagged rules — admin's clue
   that a rule is currently visible to every user. */
.rule-untagged-badge { display: inline-flex; align-items: center;
  font-size: var(--fs-xs); padding: 0.05rem 0.4rem;
  margin-left: 0.5rem; color: var(--yellow, #f2cc60);
  background: rgba(242, 204, 96, 0.08);
  border: 1px solid rgba(242, 204, 96, 0.3);
  border-radius: 999px; white-space: nowrap; }

/* ===================================================================
   RESPONSIVE FOUNDATION  — shared phone/tablet support, injected on
   every page (after each page's own <style>, so these win on equal
   specificity). Breakpoint convention (em is relative to the BROWSER
   default font, NOT --fs-base):
       tablet  max-width: 64em  (~1024px)
       phone   max-width: 40em  (~640px)   matches @container hdr 40rem
       small   max-width: 30em  (~480px)
   Media-query em/rem do NOT track the --fs-base scale picker, so any
   width-based reflow that must follow the user's scale uses grid
   auto-fit + rem minmax() (scale-aware) instead of a breakpoint — see
   the .rgrid helper below.
   =================================================================== */

/* Scale-aware card/grid deck: columns drop on their own as the viewport
   narrows OR the user raises --fs-base, with no media query (22rem rides
   the scale). Pages opt in with class="rgrid". */
.rgrid { display: grid; gap: 0.75rem;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 22rem), 1fr)); }

/* Touch ergonomics — gated on the PRIMARY pointer being coarse so a
   hybrid touch-laptop driven by a mouse keeps the dense desktop layout. */
@media (pointer: coarse) {
  /* iOS Safari zooms the page when a focused control's font-size < 16px.
     The 16px floor is a hard platform minimum (not decorative px); the
     rem term still rides --fs-base above it. */
  input, select, textarea { font-size: max(1rem, 16px); }
  /* Comfortable hit areas — rem rides the scale picker and clears the
     WCAG 2.5.8 (AA) 24px floor at every scale step. */
  header button, header .navlink, header .icon-btn, header .scale-picker,
  header select, button, select, a.btn, .btn { min-height: 2.5rem; }
  header .icon-btn, .nav-toggle { min-width: 2.5rem; }
}

/* ---- Responsive data tables → stacked label/value cards ----
   Pages opt in: add class "rcards" to the <table> and data-label="…"
   to every <td>. Below the phone breakpoint each row becomes a card
   with its column header shown inline as the label. */
@media (max-width: 40em) {
  table.rcards { border: 0; }
  table.rcards thead { position: absolute; width: 1px; height: 1px;
    padding: 0; margin: -1px; overflow: hidden; clip: rect(0 0 0 0);
    white-space: nowrap; border: 0; }                 /* visually hidden */
  table.rcards tr { display: block; border: 1px solid var(--border);
    border-radius: 6px; margin-bottom: 0.5rem; padding: 0.3rem 0.6rem;
    background: var(--panel); }
  table.rcards td { display: grid;
    grid-template-columns: minmax(6rem, 40%) 1fr; gap: 0.5rem;
    align-items: baseline; border: 0; padding: 0.2rem 0; text-align: left; }
  table.rcards td::before { content: attr(data-label); color: var(--help);
    font-weight: 600; }
  /* empty / loading rows span the whole card with no label column */
  table.rcards td[colspan] { display: block; text-align: center; }
  table.rcards td[colspan]::before { content: none; }
}

/* ---- Mobile hamburger nav drawer ----
   <header> has container-type:inline-size, which makes it the containing
   block for its position:fixed descendants. We rely on the header sitting
   at the viewport top-left and spanning the full width: a fixed child at
   top:0/left:0 lines up with the viewport origin, and we size the drawer
   and backdrop with viewport units (vw / dvh) so they still cover the
   whole screen. */
.nav-toggle { display: none; background: #21262d; color: var(--fg);
  border: 1px solid var(--border); border-radius: 4px; cursor: pointer;
  font: inherit; font-size: var(--fs-lg); line-height: 1;
  padding: 0.35rem 0.55rem; flex-shrink: 0; }
.nav-toggle:hover { background: #30363d; color: var(--bold); }
.nav-toggle:focus-visible { outline: 2px solid var(--cyan); outline-offset: 1px; }
.nav-backdrop { display: none; position: fixed; top: 0; left: 0;
  width: 100vw; height: 100vh; height: 100dvh;
  background: rgba(1, 4, 9, 0.6); z-index: 40; }

@media (max-width: 40em) {
  .nav-toggle { display: inline-flex; align-items: center;
    justify-content: center; }
  header .brand-sep { display: none; }
  /* the existing .navrow becomes the off-canvas panel (links keep their
     admin-only / page-link gating because they stay inside <header>). */
  header #navrow { position: fixed; top: 0; left: 0;
    height: 100vh; height: 100dvh; width: min(80vw, 17rem); z-index: 50;
    flex-direction: column; flex-wrap: nowrap; align-items: stretch;
    gap: 0.15rem; padding: 3.25rem 0.6rem 1rem; box-sizing: border-box;
    background: var(--panel); border-right: 1px solid var(--border);
    box-shadow: 4px 0 24px -8px rgba(0, 0, 0, 0.7); overflow-y: auto;
    visibility: hidden; transform: translateX(-100%);
    transition: transform .18s ease, visibility 0s linear .18s; }
  header #navrow .navlink { width: 100%; box-sizing: border-box;
    justify-content: flex-start; font-size: var(--fs-md);
    padding: 0.55rem 0.7rem; }
  header.nav-open #navrow { transform: translateX(0); visibility: visible;
    transition: transform .18s ease; }
  header.nav-open .nav-backdrop { display: block; }
}
"""


# Injected at the top of every page's <head>:
#   1. the viewport meta — MANDATORY for any responsive CSS to take effect.
#      Without it mobile browsers use a ~980px layout viewport and shrink to
#      fit, so no @media query ever matches. Centralised here (rather than
#      per-page) so a page can never silently ship without it. Standard form
#      only — never add maximum-scale / user-scalable=no (WCAG 1.4.4 failure).
#   2. the favicon links — scalable SVG first (modern browsers), then PNG +
#      ICO fallbacks (Safari/legacy don't render SVG favicons) and an
#      apple-touch-icon. All brand-mark assets live in static/.
#   3. a bootstrap script that applies the persisted UI scale BEFORE the
#      page's CSS parses, avoiding a flash-of-default-size on navigation.
SCALE_BOOTSTRAP_HEAD = (
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    '<link rel="icon" type="image/svg+xml" href="/static/favicon.svg">'
    '<link rel="icon" type="image/png" sizes="32x32" href="/static/favicon-32.png">'
    '<link rel="icon" type="image/png" sizes="16x16" href="/static/favicon-16.png">'
    '<link rel="icon" href="/static/favicon.ico" sizes="any">'
    '<link rel="apple-touch-icon" href="/static/apple-touch-icon.png">'
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


# Global sign-out button — the {{LOGOUT}} fragment, rendered into every page's
# right-hand header utility cluster. Auth is a single HttpOnly session cookie
# shared across all pages, so one logout works everywhere; the click handler +
# visibility toggle live in OPEN_MODE_BANNER_JS. Starts `hidden`
# (class .auth-action) and is revealed only while logged in. Icon-only to
# save space, but with an authoritative title + aria-label (outward door-arrow).
LOGOUT_BTN_HTML = (
    '<button id="logout-btn" class="icon-btn auth-action" type="button" '
    'title="Sign out" aria-label="Sign out" hidden>'
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>'
    '<polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>'
    '</svg></button>'
)


# Global reload button — the {{RELOAD}} fragment, in every page's right-hand
# header utility cluster (left of logout). Wired in OPEN_MODE_BANNER_JS: pages
# that expose a soft refresh set `window._pageReload` (settings/keys re-fetch
# their data); everywhere else it falls back to a full location.reload().
# Always visible (a refresh is meaningful regardless of auth state).
RELOAD_BTN_HTML = (
    '<button id="reload-btn" class="icon-btn" type="button" '
    'title="Reload" aria-label="Reload">'
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>'
    '<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>'
    '</svg></button>'
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


# Mobile nav drawer wire-up — appended to SCALE_PICKER_JS at the end of
# <body> on every page (so it ships without a new per-page placeholder).
# Self-contained IIFE (no reliance on later-defined helpers — see the
# injected-JS-parse-order pitfall). Toggles `header.nav-open`, traps focus
# by marking the rest of the page `inert`, closes on Esc / backdrop / link,
# and restores focus to the toggle on close.
NAV_DRAWER_JS = """
<script>(function(){
  var hdr=document.querySelector('header');
  var btn=document.querySelector('.nav-toggle');
  var nav=document.getElementById('navrow');
  var bd=document.querySelector('.nav-backdrop');
  if(!hdr||!btn||!nav)return;
  var lastFocus=null;
  // Mark everything OUTSIDE the header's branch inert while the drawer is open.
  // Walk header -> body and inert each path node's siblings, so this works
  // whether <header> is a direct <body> child (most pages) or wrapped in a
  // container like /settings' <div id="app-wrap"> (where inerting the wrapper
  // would otherwise cascade onto the header and kill the drawer links). Track
  // only what we set so close() never clears pre-existing inert.
  var _inerted=[];
  function inertRest(on){
    if(on){
      _inerted=[];
      var n=hdr;
      while(n&&n!==document.body){
        var p=n.parentElement; if(!p)break;
        Array.prototype.forEach.call(p.children,function(s){
          if(s!==n&&!s.hasAttribute('inert')){s.setAttribute('inert','');_inerted.push(s);}
        });
        n=p;
      }
    }else{
      _inerted.forEach(function(e){e.removeAttribute('inert');});
      _inerted=[];
    }
  }
  function onKey(e){if(e.key==='Escape'){e.preventDefault();close();}}
  function open(){
    if(hdr.classList.contains('nav-open'))return;
    lastFocus=document.activeElement;
    hdr.classList.add('nav-open');
    btn.setAttribute('aria-expanded','true');
    inertRest(true);
    var links=nav.querySelectorAll('.navlink'),first=null;
    for(var i=0;i<links.length;i++){if(links[i].offsetParent!==null){first=links[i];break;}}
    (first||btn).focus();
    document.addEventListener('keydown',onKey);
  }
  function close(){
    if(!hdr.classList.contains('nav-open'))return;
    hdr.classList.remove('nav-open');
    btn.setAttribute('aria-expanded','false');
    inertRest(false);
    document.removeEventListener('keydown',onKey);
    if(lastFocus&&lastFocus.focus)lastFocus.focus();else btn.focus();
  }
  btn.addEventListener('click',function(){
    hdr.classList.contains('nav-open')?close():open();
  });
  if(bd)bd.addEventListener('click',close);
  nav.addEventListener('click',function(e){if(e.target.closest('.navlink'))close();});
})();</script>
"""


# Severity pill poller — placed at the end of <body> on every page that shows
# the nav. Polls /sev every 5 s and writes the counts into the three pills.
# Server-side severity_counts() is the authoritative source (WARNING+ records
# since process start, ring-bounded). The poller is the sole pill updater:
# /stats SSE explicitly defers to it, and the /logs per-line bumps were
# dropped — so every page just trusts the 5 s tick.
#
# Skips the work if no pills exist on the page (e.g. tests, future pages).
# Open-mode warning banner — JS-injected at the top of <body> on every
# WebUI page. Fetches /auth/whoami; if open_mode=true, prepends a red
# banner reminding the operator to bootstrap an admin key. Auth rides the
# HttpOnly session cookie, sent automatically (no manual header).
OPEN_MODE_BANNER_JS = r"""
<script>(function(){
  // Read the page-key carrier ONCE so any helper (the no-access landing,
  // the SCALE_PICKER, future scope-hint UI) can ask "what page am I on?"
  // without re-querying the DOM.
  //   * __current_page = permission key (for can()/scope() lookups)
  //   * __current_page_path = full URL path (for display in the landing
  //     heading, so a nested page like /settings/api-keys reads correctly
  //     instead of showing just the permission key '/api-keys').
  try {
    var meta = document.querySelector('meta[name=page-key]');
    window.__current_page = meta ? (meta.getAttribute('content') || '') : '';
    window.__current_page_path = meta
      ? (meta.getAttribute('data-page-path') || '') : '';
  } catch(_) {
    window.__current_page = '';
    window.__current_page_path = '';
  }

  var BANNER_ID = 'open-mode-banner';
  var landingRenderedFor = null;   // page-load-only — once a no-access
                                   // landing is rendered, subsequent
                                   // refreshes (post-login) don't re-render.

  // CSRF token for cookie-authenticated mutations (double-submit). Prefer the
  // cached whoami payload (correct even if the cookie was renamed); fall back
  // to the readable whisper_csrf cookie (available synchronously on load).
  // The HttpOnly session cookie itself is sent automatically by the browser.
  window._csrfToken = function() {
    try {
      if (window.__whoami && window.__whoami.csrf_token)
        return window.__whoami.csrf_token;
    } catch(_) {}
    try {
      var m = document.cookie.match(/(?:^|;\s*)whisper_csrf=([^;]+)/);
      if (m) return decodeURIComponent(m[1]);
    } catch(_) {}
    return '';
  };

  // Global sign-out: revoke the server session (CSRF-protected), announce the
  // change, then reload (re-shows the login card / no-access landing). Shared
  // by the header logout button and the no-access landing's Sign-out button.
  window._signOut = function() {
    var done = function() {
      window.dispatchEvent(new Event('whisper:auth-changed'));
      location.reload();
    };
    try {
      fetch('/auth/logout', {
        method: 'POST',
        headers: { 'X-CSRF-Token': window._csrfToken() },
        cache: 'no-store',
      }).then(done, done);
    } catch(_) { done(); }
  };

  // Single source of truth for "what does the current bearer let me see?".
  // Idempotent: clears nav chrome first, then re-applies based on a fresh
  // /auth/whoami. Called once on page load AND on every `whisper:auth-changed`
  // event dispatched by login/logout sites (admin_routes, api_keys_routes,
  // quick_config_routes, reports_routes, captures_routes).
  window._refreshAuthChrome = function() {
    // Clear state before the fetch so logout (401 / removed token) leaves
    // the chrome hidden by default. The old IIFE only ADDed classes — it
    // had no way to recover from a transition admin → non-admin.
    try { document.body.classList.remove('role-admin'); } catch(_) {}
    try {
      document.querySelectorAll('header a.page-link[data-page].allowed')
        .forEach(function(a){ a.classList.remove('allowed'); });
    } catch(_) {}

    // The HttpOnly session cookie (and any Authorization header from a
    // non-browser caller) is sent automatically — no manual header needed.
    fetch('/auth/whoami', {
      headers: { Accept: 'application/json' }, cache: 'no-store',
    })
      .then(function(r){ return r.ok ? r.json() : null; })
      .then(function(j){
        if (!j) {
          // 401 (locked-down + no/invalid session). Leave chrome cleared
          // and drop the cached whoami so stale permissions don't linger.
          try { delete window.__whoami; } catch(_) {}
          _syncAuthActions();
          return;
        }
        // Cache the whoami payload so pages that want to consult
        // permissions later (e.g. for inline scope hints) don't re-fetch.
        try { window.__whoami = j; } catch(_) {}
        // Logout-button visibility tracks login state; the HttpOnly cookie
        // isn't JS-readable, so this is driven by whoami, not storage.
        _syncAuthActions();

        // OPEN-mode warning banner — only when no admin key configured.
        // Idempotent: the banner gets a stable id so re-runs don't stack.
        if (j.open_mode && !document.getElementById(BANNER_ID)) {
          var b = document.createElement('div');
          b.id = BANNER_ID;
          b.setAttribute('role','alert');
          b.style.cssText = 'background:#5a2424;color:#fff;padding:0.5rem 1rem;'
            + 'text-align:center;font-weight:600;font-size:0.95rem;'
            + 'position:sticky;top:0;z-index:20;';
          b.innerHTML = '⚠ No admin API key set — the server is in '
            + 'OPEN mode and anyone reachable can use it. '
            + '<a href="/settings/api-keys" style="color:#ffd1d1;text-decoration:underline">'
            + 'Generate the first admin key</a>.';
          document.body.insertBefore(b, document.body.firstChild);
        }

        var isAdmin = !!j.is_admin;
        var perms = (j.permissions && j.permissions.pages) || {};

        // `body.role-admin` reveals admin-only chrome (/settings +
        // /settings/api-keys nav links, severity pills, in-page admin
        // buttons). Pages used to add it unconditionally after a successful
        // state fetch, which leaked admin chrome to non-admins on /stats,
        // /logs and /reports.
        if (isAdmin) document.body.classList.add('role-admin');

        // Per-page nav-link visibility. The nav renders every link with
        // class `.page-link` default-hidden; add `.allowed` per link the
        // caller can reach. Admins pass on every link via is_admin.
        document.querySelectorAll('header a.page-link[data-page]').forEach(
          function(a) {
            var page = a.getAttribute('data-page');
            var scope = perms[page];
            if (isAdmin || (scope && scope !== 'none')) {
              a.classList.add('allowed');
            }
          }
        );

        // No-access landing — page-load only. The landing replaces <main>
        // content; re-rendering it on every refresh would clobber a page
        // the user just successfully logged into. The per-page 403
        // handlers (admin_routes loadState, quick_config_routes loadState)
        // cover the "valid bearer, wrong scope" case after login.
        var current = window.__current_page;
        if (
          landingRenderedFor === null
          && current && current !== '__admin_only__'
          && !isAdmin
          && (!perms[current] || perms[current] === 'none')
          && typeof _renderNoAccessLanding === 'function'
        ) {
          landingRenderedFor = current;
          _renderNoAccessLanding({ page: current });
        }
      })
      .catch(function(){});
  };

  // Refresh on every login/logout. Each login/logout site dispatches
  // `whisper:auth-changed` after the server sets/clears the session cookie.
  window.addEventListener('whisper:auth-changed', window._refreshAuthChrome);

  // ---- Global sign-out ----
  // Every page renders #logout-btn (.auth-action) in the header utility
  // cluster. The HttpOnly session cookie isn't JS-readable, so visibility is
  // driven by the cached whoami: shown only when locked-down AND logged in
  // (open mode has no session to end). Clicking hits the server /auth/logout.
  function _syncAuthActions() {
    var loggedIn = false;
    try {
      loggedIn = !!(window.__whoami && window.__whoami.open_mode === false);
    } catch(_) {}
    document.querySelectorAll('.auth-action').forEach(function(el){
      el.hidden = !loggedIn;
    });
  }
  var _logoutBtn = document.getElementById('logout-btn');
  if (_logoutBtn) {
    _logoutBtn.addEventListener('click', function(){ window._signOut(); });
  }
  window.addEventListener('whisper:auth-changed', _syncAuthActions);
  _syncAuthActions();

  // ---- Global reload ----
  // Every page renders #reload-btn in the header utility cluster. Pages that
  // can refresh in place set `window._pageReload` (e.g. /settings + /settings/api-
  // keys re-fetch their data); elsewhere we do a full page reload. Read at
  // click time so the page's init can register the hook after this runs.
  var _reloadBtn = document.getElementById('reload-btn');
  if (_reloadBtn) {
    _reloadBtn.addEventListener('click', function(){
      if (typeof window._pageReload === 'function') { window._pageReload(); }
      else { location.reload(); }
    });
  }

  // Initial page-load pass — replaces the old IIFE body.
  window._refreshAuthChrome();
})();</script>
"""


# Shared timestamp helpers — one source of truth across every admin page.
# Pages opt in by inserting `{{TIME_HELPERS_JS}}` at the top of their inline
# <script> block; render_page substitutes the constant below.
#
# Format contract: HH:MM:SS | YYYY.MM.DD — 24-hour clock, dot-separated date,
# space-pipe-space separator. fmtWhen appends a relative suffix (e.g.
# " | 5m ago") while the event is within 24 h, then drops it.
#
# `timeTick(rootSelector, intervalMs=30000)` walks every element matching
# rootSelector (defaults to `[data-ts]`) and re-renders its textContent from
# fmtWhen(parseFloat(el.dataset.ts)). Cheap; pages that want their cards'
# relative suffixes to age in place can call timeTick() once at boot.
TIME_HELPERS_JS = r"""
<script>
function _pad2(n) { return (n < 10 ? '0' : '') + n; }

function absTime(ts) {
  if (!ts) return '—';
  var d = new Date(ts * 1000);
  return _pad2(d.getHours()) + ':' + _pad2(d.getMinutes()) + ':' + _pad2(d.getSeconds())
    + ' | ' + d.getFullYear() + '.' + _pad2(d.getMonth() + 1) + '.' + _pad2(d.getDate());
}

function relTime(ts) {
  if (!ts) return '';
  var sec = Math.max(0, Date.now() / 1000 - ts);
  if (sec < 5)     return 'just now';
  if (sec < 60)    return Math.floor(sec) + 's ago';
  if (sec < 3600)  return Math.floor(sec / 60) + 'm ago';
  if (sec < 86400) return Math.floor(sec / 3600) + 'h ago';
  return '';
}

function fmtWhen(ts) {
  if (!ts) return '—';
  var a = absTime(ts), r = relTime(ts);
  return r ? (a + ' | ' + r) : a;
}

function timeTick(sel, ms) {
  sel = sel || '[data-ts]';
  ms = ms || 30000;
  function paint() {
    document.querySelectorAll(sel).forEach(function(el) {
      var ts = parseFloat(el.dataset.ts);
      if (!isFinite(ts) || ts <= 0) return;
      var next = fmtWhen(ts);
      if (el.textContent !== next) el.textContent = next;
      if (!el.title) el.title = absTime(ts);
    });
  }
  paint();
  setInterval(paint, ms);
}
</script>
"""


# No-access landing card used when the bearer is valid but the caller
# lacks access to the current page. Replaces the old `Admin only`
# hard-coded card. The new version reads `window.__whoami` (set by
# OPEN_MODE_BANNER_JS) to list every other page the caller CAN reach,
# rendering one button per accessible page plus a Sign-out button.
# Falls back to a "no pages available" message when the caller has
# scope=none everywhere.
#
# `_renderNotAdminLanding` is kept as a thin alias so existing callers
# (`reports_routes._renderAdminOnlyIfNonAdmin`, the similar inline helper
# in captures_routes, and api_keys_routes._check403) keep working.
#
# Two flavours:
#   - NOT_ADMIN_LANDING_JS         — raw (no <script> wrapper). Injected
#                                    inside a page's existing <script>
#                                    IIFE via {{NOT_ADMIN_LANDING_JS}}.
#                                    Used by pages that already host a
#                                    big inline IIFE (reports, captures,
#                                    api_keys).
#   - NOT_ADMIN_LANDING_GLOBAL_JS  — full <script> wrapper that puts the
#                                    helpers on the global window. Pulled
#                                    in alongside OPEN_MODE_BANNER_JS so
#                                    every page has access regardless of
#                                    whether it includes the placeholder.
NOT_ADMIN_LANDING_JS = """
// Per-page metadata used to render the landing's pill-style action
// buttons. `href` is what the button navigates to. The button label
// reuses `href` directly (no "Open " verb prefix — stacked verbs read
// like the redundant menu) so the buttons look + behave like links.
var _PAGE_LINK_INFO = {
  quick_config: { href: '/quick-config' },
  captures:     { href: '/captures' },
  reports:      { href: '/reports' },
  stats:        { href: '/stats' },
  logs:         { href: '/logs' }
};

function _renderNoAccessLanding(opts) {
  // Clear admin chrome — same reason it's removed: the landing shows
  // when the caller is NOT entitled to admin UI on this page.
  document.body.classList.remove('role-admin');
  var main = document.getElementsByTagName('main')[0];
  if (!main) return;

  var current = (opts && opts.page) || window.__current_page || '';
  var who = window.__whoami || {};
  var perms = (who.permissions && who.permissions.pages) || {};
  var isAdmin = !!who.is_admin;

  // Pages the caller can reach, minus the one they're on (it's the
  // page we can't reach — listing it would be confusing).
  var allowed = Object.keys(_PAGE_LINK_INFO).filter(function(p) {
    if (p === current) return false;
    if (isAdmin) return true;
    return perms[p] && perms[p] !== 'none';
  });

  var btns = allowed.map(function(p) {
    var info = _PAGE_LINK_INFO[p];
    // Button text is the URL path itself (`/logs`, `/quick-config`,
    // …) — drops the redundant "Open" verb and matches what the user
    // expects to see in the address bar.
    return '<a href="' + info.href + '" class="landing-btn">'
         + info.href + '</a>';
  }).join('');

  // Heading slug: prefer the full URL path stashed by OPEN_MODE_BANNER_JS
  // (e.g. "/settings/api-keys"), then fall back to a derived form, then to
  // an empty suffix. Admin-only pages get the URL too — never the bare
  // sentinel.
  var displayPath = window.__current_page_path || '';
  var slug;
  if (displayPath) {
    slug = ' to ' + displayPath;
  } else if (current && current !== '__admin_only__') {
    slug = ' to /' + current.replace(/_/g, '-');
  } else {
    slug = '';
  }
  var body;
  if (allowed.length) {
    body = '<p>Your API key does not grant access to this page.</p>'
         + '<p class="landing-hint">Pages you can access:</p>'
         + '<div class="landing-actions">' + btns + '</div>';
  } else {
    body = '<p>Your API key does not grant access to this page, and no '
         + 'other pages are available either. Ask an admin to grant '
         + 'access.</p>';
  }

  main.innerHTML =
    '<div class="no-access-landing">'
    + '<h2>No access' + slug + '</h2>'
    + body
    + '<p class="landing-signout">'
    + '<button onclick="window._signOut()">Sign out</button>'
    + '</p></div>';
}

// Backwards-compat alias — old callers used `_renderNotAdminLanding()`.
function _renderNotAdminLanding() {
  _renderNoAccessLanding({ page: window.__current_page || '' });
}
"""


# Global wrapper around NOT_ADMIN_LANDING_JS — puts the helpers on the
# window so every page has access regardless of whether it includes the
# {{NOT_ADMIN_LANDING_JS}} placeholder. Injected via render_page next to
# OPEN_MODE_BANNER_JS so the centralised auto-landing path can call
# `_renderNoAccessLanding` for /logs, /stats, /quick-config without
# requiring their templates to opt in to the older placeholder.
NOT_ADMIN_LANDING_GLOBAL_JS = "<script>" + NOT_ADMIN_LANDING_JS + "</script>"


# Shared tag-picker widget used on /settings (per-rule tag editor) and
# /settings/api-keys (per-user tag editor in the permissions matrix).
# DOM-pure factory: `_renderTagPicker(opts) -> { el, getTags, setTags,
# setAvailable }`. Caller mounts the returned element wherever and
# subscribes to `opts.onChange(newTags)`.
#
# Tag format matches the server-side `config_store.TAG_RE`: lowercase
# letters/digits/hyphens, 1-32 chars, no leading/trailing hyphen.
# Validation happens BOTH client-side (visual red border on bad input)
# AND server-side (set_user_permissions / Pydantic validator) so a
# JS-side bypass can't smuggle malformed tags into the DB.
#
# Autocomplete: caller passes `available` = the union of every tag
# currently in use (rule tags for the matrix UI, or all rule tags for
# the rule editor). The widget suggests matches as the user types;
# clicking a suggestion adds the tag.
TAG_PICKER_JS = r"""
<script>(function(){
  var TAG_RE = /^[a-z0-9][a-z0-9-]{0,31}$/;
  function _norm(s) { return String(s == null ? '' : s).trim().toLowerCase(); }

  function _renderTagPicker(opts) {
    opts = opts || {};
    var tags = Array.isArray(opts.initial) ? opts.initial.slice() : [];
    var available = Array.isArray(opts.available) ? opts.available.slice() : [];
    var disabled = !!opts.disabled;

    var el = document.createElement('div');
    el.className = 'tag-picker' + (disabled ? ' disabled' : '');

    var pills = document.createElement('div');
    pills.className = 'tag-picker-pills';
    el.appendChild(pills);

    var input = document.createElement('input');
    input.type = 'text';
    input.className = 'tag-picker-input';
    input.placeholder = opts.placeholder || '+ tag';
    input.autocomplete = 'off';
    if (disabled) input.disabled = true;

    var suggest = document.createElement('div');
    suggest.className = 'tag-picker-suggest';
    suggest.hidden = true;
    el.appendChild(suggest);

    function _hasTag(s) { return tags.indexOf(s) !== -1; }
    function _notify() {
      if (typeof opts.onChange === 'function') opts.onChange(tags.slice());
    }

    function _render() {
      pills.innerHTML = '';
      tags.forEach(function(t) {
        var pill = document.createElement('span');
        pill.className = 'tag-pill';
        pill.textContent = t;
        if (!disabled) {
          var x = document.createElement('button');
          x.type = 'button';
          x.className = 'tag-pill-x';
          x.textContent = '×';
          x.title = 'Remove "' + t + '"';
          x.addEventListener('click', function(e) {
            e.preventDefault();
            var i = tags.indexOf(t);
            if (i >= 0) {
              tags.splice(i, 1);
              _render();
              _notify();
            }
          });
          pill.appendChild(x);
        }
        pills.appendChild(pill);
      });
      pills.appendChild(input);
    }

    function _tryAdd(raw) {
      var t = _norm(raw);
      if (!t) return false;
      if (!TAG_RE.test(t)) {
        input.classList.add('invalid');
        input.title = 'Invalid tag — lowercase a-z0-9- only, max 32 chars, no leading/trailing hyphen';
        return false;
      }
      input.classList.remove('invalid');
      input.title = '';
      if (_hasTag(t)) { input.value = ''; return false; }
      tags.push(t);
      tags.sort();
      input.value = '';
      _render();
      _notify();
      return true;
    }

    function _hideSuggest() { suggest.hidden = true; suggest.innerHTML = ''; }
    function _updateSuggest(prefix) {
      var matches = available.filter(function(a) {
        return a !== prefix && a.indexOf(prefix) === 0 && !_hasTag(a);
      }).slice(0, 8);
      if (!matches.length) { _hideSuggest(); return; }
      suggest.innerHTML = '';
      matches.forEach(function(a) {
        var item = document.createElement('div');
        item.className = 'tag-picker-suggest-item';
        item.textContent = a;
        // mousedown not click so input's blur doesn't fire first and
        // hide the suggest before the click registers.
        item.addEventListener('mousedown', function(e) {
          e.preventDefault();
          _tryAdd(a);
          _hideSuggest();
          input.focus();
        });
        suggest.appendChild(item);
      });
      suggest.hidden = false;
    }

    input.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' || e.key === ',') {
        e.preventDefault();
        _tryAdd(input.value);
        _hideSuggest();
      } else if (e.key === 'Backspace' && !input.value && tags.length) {
        tags.pop();
        _render();
        _notify();
      } else if (e.key === 'Escape') {
        input.value = '';
        input.classList.remove('invalid');
        _hideSuggest();
      }
    });
    input.addEventListener('input', function() {
      var v = _norm(input.value);
      if (!v) { input.classList.remove('invalid'); _hideSuggest(); return; }
      input.classList.toggle('invalid', !TAG_RE.test(v));
      _updateSuggest(v);
    });
    input.addEventListener('blur', function() {
      // Auto-commit a trailing typed value on blur — easy to forget Enter.
      if (input.value.trim()) _tryAdd(input.value);
      setTimeout(_hideSuggest, 150);
    });

    _render();

    return {
      el: el,
      getTags: function() { return tags.slice(); },
      setTags: function(t) {
        tags = Array.isArray(t) ? t.slice() : [];
        _render();
      },
      setAvailable: function(a) {
        available = Array.isArray(a) ? a.slice() : [];
      },
    };
  }

  window._renderTagPicker = _renderTagPicker;
})();</script>
"""


SEV_POLLER_JS = """
<script>(function(){
  if(!document.getElementById('sev-warn'))return;
  function setPill(id, n){
    var el=document.getElementById(id); if(!el)return;
    var numEl=el.querySelector('.n'); if(!numEl)return;
    var prev=+numEl.textContent || 0;
    numEl.textContent=n;
    el.classList.toggle('hot',  n > 0);
    el.classList.toggle('zero', n === 0);
    if(n > prev){
      el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
    }
  }
  function tick(){
    fetch('/sev', {cache:'no-store'})
      .then(function(r){ return r.ok ? r.json() : null; })
      .then(function(j){
        if(!j) return;
        setPill('sev-warn', j.warn|0);
        setPill('sev-err',  j.err |0);
        setPill('sev-crit', j.crit|0);
      })
      .catch(function(){});
  }
  tick();
  setInterval(tick, 5000);
})();</script>
"""


# Per-rule body editors shared by /settings (full editor with drag-reorder etc.)
# and /quick-config (read-only header + body editor only). Defined as
# top-level functions so both pages can call them with their own
# `commitData` callback. The `commitData` argument is invoked by every
# input/change event inside an editor; each page implements its own dirty-
# tracking on top of that callback.
#
# Keep the per-type rendering here in lockstep with config_store.py rule
# schemas. Adding a new rule type requires:
#   1. New Pydantic class in config_store.py
#   2. New `if (rule.type === '<type>')` branch in renderTypeEditor below
#   3. New entry in _PIPELINE_TYPES for the pill label
RULE_EDITOR_JS = r"""<script>
// German-aware, case-insensitive collator for ordering human-visible lists
// (cb:map keys). sensitivity:'base' folds case + accent so 'anemi' sorts next
// to 'Amoxyzylin' and umlauts land in German position (ä≈a); numeric:true gives
// natural number order (item2 before item10). Display only — never used for
// equality/fingerprint/persistence sorts, which stay byte-stable.
const _coll = new Intl.Collator('de', { sensitivity: 'base', numeric: true });
// Display \n, \r, \t, \\ as literal 2-char escape sequences in <input>
// cells. Single-line inputs strip newlines per WHATWG spec, so without
// this the user sees an empty field for any value containing a newline.
function _esc(s) {
  if (s == null) return '';
  return String(s)
    .replace(/\\/g, '\\\\')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t');
}
function _unesc(s) {
  if (s == null) return '';
  let out = '';
  for (let i = 0; i < s.length; i++) {
    if (s[i] === '\\' && i + 1 < s.length) {
      const nxt = s[++i];
      if (nxt === 'n') out += '\n';
      else if (nxt === 'r') out += '\r';
      else if (nxt === 't') out += '\t';
      else if (nxt === '\\') out += '\\';
      else out += '\\' + nxt;  // keep both chars: \1, \d, \w, \. survive intact
    } else {
      out += s[i];
    }
  }
  return out;
}

const _PIPELINE_TYPES = [
  { type: 'regex',                       pill: 'regex' },
  { type: 'callback:lowercase-wordlist', pill: 'cb:wordlist' },
  { type: 'callback:map',                pill: 'cb:map' },
  { type: 'callback:dedup',              pill: 'cb:dedup' },
  { type: 'callback:upper',              pill: 'cb:upper' },
  { type: 'terminal',                    pill: 'terminal' },
];
const _typePill = (t) => (_PIPELINE_TYPES.find(x => x.type === t) || {}).pill || t;

// Bind Enter-to-commit on a text input. When `onEnter` is provided and the
// user presses Enter, defer one tick before firing — if the input's value
// changed during that tick, the user picked a datalist suggestion (native
// behavior); skip the commit and let the next Enter fire it. Otherwise
// invoke onEnter. Browsers update <input list> values synchronously when a
// suggestion is selected, so a one-tick defer is enough to distinguish.
function _bindEnterCommit(inp, onEnter) {
  if (!onEnter) return;
  inp.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    e.preventDefault();
    const before = inp.value;
    setTimeout(() => {
      if (inp.value !== before) return;  // datalist pick — defer
      onEnter();
    }, 0);
  });
}

function _makeMonoLabeledInput(label, val, onInput, onEnter) {
  const lbl = document.createElement('div');
  lbl.className = 'help';
  lbl.textContent = label + ':';
  const inp = document.createElement('input');
  inp.type = 'text';
  inp.spellcheck = false;
  inp.autocomplete = 'off';
  inp.value = val == null ? '' : val;
  inp.addEventListener('input', () => onInput(inp.value));
  _bindEnterCommit(inp, onEnter);
  const wrap = document.createElement('div');
  wrap.appendChild(lbl); wrap.appendChild(inp);
  return wrap;
}

function _makeMapRow(rule, key, val, commitData, datalistId, onEnter, ts, showDate) {
  const tr = document.createElement('tr');
  const td1 = document.createElement('td');
  const td2 = document.createElement('td');
  const td3 = document.createElement('td');
  td3.style.width = '2.5rem';
  const ki = document.createElement('input');
  ki.type = 'text'; ki.value = _esc(key);
  // Spoken-word cell opts into a datalist on /quick-config so end-users
  // get autocomplete from recent transcription FINALs. Admin /settings
  // doesn't pass datalistId — no autocomplete on the admin page (it
  // would clutter long maps).
  if (datalistId) ki.setAttribute('list', datalistId);
  const vi = document.createElement('input');
  vi.type = 'text'; vi.value = _esc(val);
  // Map keys/values may contain \n etc.; <input> strips real newlines,
  // so we display \n as literal 2-char escape and decode on read.
  function _readMap(parent) {
    const m = {};
    parent.querySelectorAll('tr').forEach(r => {
      const k = _unesc(r.querySelector('td:first-child input').value);
      const v = _unesc(r.querySelector('td:nth-child(2) input').value);
      if (k) m[k] = v;
    });
    return m;
  }
  function rebuild() {
    const parent = tr.parentNode;
    if (!parent) return;
    rule.map = _readMap(parent);
    commitData();
  }
  ki.addEventListener('input', rebuild);
  vi.addEventListener('input', rebuild);
  _bindEnterCommit(ki, onEnter);
  _bindEnterCommit(vi, onEnter);
  const del = document.createElement('button');
  del.type = 'button'; del.textContent = '×';
  del.addEventListener('click', () => {
    const parent = tr.parentNode;
    tr.remove();
    if (parent) {
      rule.map = _readMap(parent);
      commitData();
    }
  });
  td1.appendChild(ki); td2.appendChild(vi); td3.appendChild(del);
  tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3);
  // Inline "added / last-updated" date, /quick-config only (showDate). Appended
  // LAST so the td:first-child / td:nth-child(2) selectors _readMap uses to
  // locate the key/value inputs are unaffected. New rows (no ts yet) show "—"
  // until the next server save stamps map_meta.
  if (showDate) {
    const td4 = document.createElement('td');
    td4.className = 'map-date-cell';
    const span = document.createElement('span');
    span.className = 'map-date';
    if (ts) {
      span.setAttribute('data-ts', String(ts));
      span.textContent = fmtWhen(ts);
      span.title = absTime(ts);
    } else {
      span.textContent = '—';
    }
    td4.appendChild(span);
    tr.appendChild(td4);
  }
  return tr;
}

function renderTypeEditor(rule, commitData, opts) {
  // opts (optional):
  //   datalistId      — passed through to _makeMapRow for cb:map autocomplete
  //                     on /quick-config. Other rule types ignore.
  //   makeSaveBtn     — `() => HTMLButtonElement` factory. When provided, the
  //                     editor appends a per-rule Save button. /quick-config
  //                     passes a factory closed over the global dirty Set;
  //                     /settings (admin) omits this opt → no per-card Save.
  //   commitOnEnter   — `() => void` callback. When provided, pressing Enter
  //                     inside any text input in this editor fires it (after
  //                     a one-tick guard for native datalist picks). Skipped
  //                     for the cb:wordlist textarea (Enter inserts newline)
  //                     and the terminal type (no inputs). /quick-config
  //                     passes `doSave`; admin /settings omits → Enter inert.
  opts = opts || {};
  const onEnter = opts.commitOnEnter;
  const box = document.createElement('div');
  box.className = 'rule-editor';

  // Rule rationale / documentation. Only the admin /settings editor passes
  // `showNote` — /quick-config omits it (non-admin users cannot patch the
  // `note` field, see _PATCH_ALLOWED_FIELDS in quick_config_routes.py).
  if (opts.showNote) {
    const noteLbl = document.createElement('div');
    noteLbl.className = 'help';
    noteLbl.textContent = 'note (rationale / documentation):';
    box.appendChild(noteLbl);
    const noteTa = document.createElement('textarea');
    noteTa.className = 'rule-note';
    noteTa.rows = 2;
    noteTa.spellcheck = false;
    noteTa.style.width = '100%';
    noteTa.style.boxSizing = 'border-box';
    noteTa.value = rule.note == null ? '' : rule.note;
    noteTa.addEventListener('input', () => { rule.note = noteTa.value; commitData(); });
    box.appendChild(noteTa);
  }

  if (rule.type === 'terminal') {
    const note = document.createElement('div');
    note.className = 'help';
    note.textContent = 'Hardcoded terminal step: lstrip(" \\t\\r") + rstrip(" \\t\\r"). '
      + 'Always runs last. Preserves a leading or trailing newline ("\\n") emitted by '
      + '"neue Zeile" / "neuer Absatz" at the edges of the utterance.';
    box.appendChild(note);
    return box;
  }

  // Right-align the per-card Save button on rule types that don't have a
  // sibling "+ add entry" bar (cb:map handles its own pairing below).
  function _appendSaveRow(parent) {
    if (!opts.makeSaveBtn) return;
    const saveRow = document.createElement('div');
    saveRow.style.cssText = 'display:flex;justify-content:flex-end;margin-top:0.6rem;';
    const btn = opts.makeSaveBtn();
    btn.style.minWidth = '6rem';
    saveRow.appendChild(btn);
    parent.appendChild(saveRow);
  }

  if (rule.type === 'regex') {
    box.appendChild(_makeMonoLabeledInput('pattern', rule.pattern, (v) => {
      rule.pattern = v; commitData();
    }, onEnter));
    box.appendChild(_makeMonoLabeledInput('replacement', rule.replacement, (v) => {
      rule.replacement = v; commitData();
    }, onEnter));
    _appendSaveRow(box);
    return box;
  }

  if (rule.type === 'callback:lowercase-wordlist') {
    box.appendChild(_makeMonoLabeledInput('pattern', rule.pattern, (v) => {
      rule.pattern = v; commitData();
    }, onEnter));
    const wlLbl = document.createElement('div');
    wlLbl.className = 'help';
    wlLbl.textContent = 'Wordlist (one entry per line, case-insensitive):';
    box.appendChild(wlLbl);
    const ta = document.createElement('textarea');
    ta.value = (rule.wordlist || []).join('\n');
    ta.rows = 6;
    ta.addEventListener('input', () => {
      rule.wordlist = ta.value.split('\n').map(s => s.trim()).filter(Boolean);
      commitData();
    });
    // Intentionally NO Enter binding on the wordlist textarea — Enter must
    // insert a newline so users can edit multi-line lists.
    box.appendChild(ta);
    _appendSaveRow(box);
    return box;
  }

  if (rule.type === 'callback:map') {
    const note = document.createElement('div');
    note.className = 'help';
    note.textContent = 'Pattern auto-built from map keys (longest-first, '
      + 'word-bounded, case-insensitive). Edit entries below.';
    box.appendChild(note);
    const showDate = !!opts.showMapDates;
    const meta = rule.map_meta || {};
    const tbl = document.createElement('table');
    tbl.className = 'map-table';
    tbl.style.width = '100%';
    // Order by map_meta (added / last-updated), oldest first → newest last, so
    // the freshest entries sit next to the "+ add entry" bar. Un-stamped keys
    // (factory entries never edited here) sort as oldest, then alphabetically.
    const rows = Object.entries(rule.map || {}).sort((a, b) => {
      const ta = meta[a[0]] || 0, tb = meta[b[0]] || 0;
      if (ta !== tb) return ta - tb;
      return _coll.compare(a[0], b[0]);
    });
    const collapseAfter = opts.collapseMapAfter || 0;
    const hiddenCount = (collapseAfter && rows.length > collapseAfter)
      ? rows.length - collapseAfter : 0;
    // Toggle for the older (collapsed) head — only when there's an overflow.
    if (hiddenCount) {
      const toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'map-toggle';
      let shown = false;
      const paint = () => {
        tbl.classList.toggle('show-all', shown);
        toggle.textContent = shown
          ? '▾ Hide ' + hiddenCount + ' older'
          : '▸ Show ' + hiddenCount + ' older mapping' + (hiddenCount === 1 ? '' : 's');
      };
      toggle.addEventListener('click', () => { shown = !shown; paint(); });
      paint();
      box.appendChild(toggle);
    }
    rows.forEach(([k, v], i) => {
      const tr = _makeMapRow(rule, k, v, commitData, opts.datalistId, onEnter, meta[k] || 0, showDate);
      // Hide the oldest `hiddenCount` rows behind the toggle (CSS display:none,
      // so _readMap still reads every row when building the patch).
      if (i < hiddenCount) tr.classList.add('map-row-collapsed');
      tbl.appendChild(tr);
    });
    box.appendChild(tbl);
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.textContent = '+ add entry';
    addBtn.style.flex = '1';
    addBtn.addEventListener('click', () => {
      // Append a new <tr> directly so the surrounding row body stays
      // expanded and other expanded rows keep their input state.
      // Pass empty key/val: _readMap rebuilds the dict from DOM inputs
      // on every change and skips empty-key rows, so the new row's
      // blank input doesn't need a placeholder slug in rule.map. No
      // commitData() here — the first keystroke triggers it via the
      // input's rebuild listener (an empty row contributes nothing
      // to the saved dict).
      if (!rule.map) rule.map = {};
      const newTr = _makeMapRow(rule, '', '', commitData, opts.datalistId, onEnter, 0, opts.showMapDates);
      tbl.appendChild(newTr);
      // Focus + open the autocomplete dropdown immediately so the user
      // sees recent-transcription candidates without a second click.
      // Synchronous showPicker() inside this user-initiated handler
      // preserves transient activation (Chrome 99+, Firefox 149+, no-op
      // on older browsers — degrades to focus-only).
      const ki = newTr.querySelector('td:first-child input');
      if (ki) {
        ki.focus();
        try { ki.showPicker(); } catch (_) { /* unsupported */ }
      }
    });
    // Pair addBtn + saveBtn in a flex row so they sit side-by-side near
    // the bottom of the map table. addBtn fills available width; saveBtn
    // sits on the right with a min-width so its label reads comfortably.
    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display:flex;gap:0.5rem;align-items:stretch;margin-top:0.6rem;';
    btnRow.appendChild(addBtn);
    if (opts.makeSaveBtn) {
      const saveBtn = opts.makeSaveBtn();
      saveBtn.style.minWidth = '6rem';
      btnRow.appendChild(saveBtn);
    }
    box.appendChild(btnRow);
    return box;
  }

  if (rule.type === 'callback:dedup' || rule.type === 'callback:upper') {
    box.appendChild(_makeMonoLabeledInput('pattern', rule.pattern, (v) => {
      rule.pattern = v; commitData();
    }, onEnter));
    const note = document.createElement('div');
    note.className = 'help';
    note.textContent = rule.type === 'callback:dedup'
      ? 'Callback: collapse each match — last non-comma wins; pure-comma run → single comma.'
      : 'Callback: uppercase group(2) (or whole match if pattern has fewer than 2 groups).';
    box.appendChild(note);
    _appendSaveRow(box);
    return box;
  }

  return box;
}
</script>
"""


def _nav_items(current: str) -> list[tuple[str, str, bool]]:
    """Return [(label, href, active), ...] in left-to-right display order,
    honoring cfg.ADMIN_UI_ENABLED. `_NAV_SPEC` is the single source of truth
    for nav order — edit it to reorder the bar. The trailing flag marks links
    that only exist when the admin UI is registered (quick-config / reports /
    captures / settings / api-keys); logs + stats are always served and so
    render unconditionally."""
    admin = getattr(cfg, "ADMIN_UI_ENABLED", False)
    return [
        (label, href, current == key)
        for label, href, key, admin_gated in _NAV_SPEC
        if admin or not admin_gated
    ]


# (label, href, current-key, admin_gated) — left-to-right nav order.
_NAV_SPEC: list[tuple[str, str, str, bool]] = [
    ("quick",    "/quick-config",      "quick-config", True),
    ("captures", "/captures",          "captures",     True),
    ("reports",  "/reports",           "reports",      True),
    ("stats",    "/stats",             "stats",        False),
    ("logs",     "/logs",              "logs",         False),
    ("settings", "/settings",          "settings",     True),
    ("keys",     "/settings/api-keys", "api-keys",     True),
]


def nav_html(current: str) -> str:
    """Render the primary nav links as the {{NAV}} fragment. The severity
    pills are rendered separately by sev_pills_html() ({{SEV_PILLS}}) so they
    can live in the header's right-hand utility cluster."""
    # Two visibility tracks:
    #
    #   - "admin-only" — settings + api-keys + sev pills. These stay all-or-
    #     nothing on the existing `body.role-admin` class (CSS hides by
    #     default; pages add the class after a successful admin-API ping).
    #
    #   - "page-link" + data-page="<X>" — logs / stats / quick-config /
    #     reports / captures. Per-user gated by OPEN_MODE_BANNER_JS via
    #     /auth/whoami → permissions.pages[X]. CSS default-hides; the JS
    #     adds `.allowed` per link the caller can reach. Admins always
    #     pass via the is_admin short-circuit on the server side.
    page_link_labels: dict[str, str] = {
        "logs":     "logs",
        "stats":    "stats",
        "quick":    "quick_config",
        "reports":  "reports",
        "captures": "captures",
    }
    admin_only_labels = {"settings", "keys"}
    # Hamburger toggle (shown only ≤40em via NAV_CSS) + the nav links. The
    # links keep their admin-only / page-link gating classes; on narrow
    # screens NAV_CSS turns the same #navrow into an off-canvas drawer and
    # NAV_DRAWER_JS wires open/close + focus handling.
    parts: list[str] = [
        '<button class="nav-toggle" type="button" aria-label="Menu" '
        'aria-expanded="false" aria-controls="navrow">☰</button>',
        '<span class="navrow" id="navrow">',
    ]
    for label, href, active in _nav_items(current):
        classes = ["navlink"]
        extra_attr = ""
        if active:
            classes.append("active")
            extra_attr += ' aria-current="page"'
        if label in admin_only_labels:
            classes.append("admin-only")
        elif label in page_link_labels:
            classes.append("page-link")
            extra_attr += f' data-page="{page_link_labels[label]}"'
        parts.append(
            f'<a class="{" ".join(classes)}" href="{href}"{extra_attr}>'
            f'{label}</a>'
        )
    parts.append("</span>")
    parts.append('<div class="nav-backdrop"></div>')
    return "".join(parts)


def sev_pills_html() -> str:
    """Render the three severity-count pills (warn/err/crit) as the
    {{SEV_PILLS}} fragment, grouped in a `.sevpills` wrapper for the header's
    right-hand utility cluster. Split out of nav_html so the pills sit with
    the other status/utility chrome (scale picker) rather than beside the
    nav links.

    Stable IDs let SEV_POLLER_JS update just the `.n` inner span on each /sev
    tick without rebuilding the link (preserves focus/click state). The counts
    rendered here are best-effort at page load; the client takes over
    immediately, so they're correct for the first render and live thereafter."""
    counts = severity_counts()
    parts: list[str] = ['<span class="sevpills">']
    for level, key in (("warn", "WARNING"), ("err", "ERROR"), ("crit", "CRITICAL")):
        n = counts[level]
        cls = f"sevpill admin-only {level} {'hot' if n else 'zero'}"
        title = f"{key}+ since process start — click to filter logs"
        parts.append(
            f'<a id="sev-{level}" class="{cls}" '
            f'href="/logs?filter={key}" title="{title}">'
            f'<span class="lbl">{level}</span> '
            f'<span class="n">{n}</span></a>'
        )
    parts.append("</span>")
    return "".join(parts)


def render_page(template: str, current: str) -> str:
    """Substitute placeholders in a page template:
      - {{NAV}}                  → primary nav links (left of global bar)
      - {{SEV_PILLS}}            → severity pills (right utility cluster)
      - {{NAV_CSS}}              → shared header/scale-token CSS
      - {{SCALE_PICKER}}         → scale dropdown (header)
      - {{SCALE_PICKER_JS}}      → wire-up script (end of body)
      - {{SEV_POLLER_JS}}        → 5-s pill re-sync + open-mode admin-
                                   key warning banner (end of body)
      - {{SCALE_BOOTSTRAP_HEAD}} → tiny pre-paint script (top of <head>)
      - {{RULE_EDITOR_JS}}       → shared per-rule body editors
      - {{TIME_HELPERS_JS}}      → absTime / relTime / fmtWhen / timeTick
      - {{NOT_ADMIN_LANDING_JS}} → _renderNoAccessLanding() helper
                                   (+ _renderNotAdminLanding alias)
      - {{PAGE_META}}            → <meta name="page-key" ...> carrier so
                                   shared JS knows which page it's on
      - {{TAG_PICKER_JS}}        → window._renderTagPicker(opts) widget
                                   shared by /settings rule editor +
                                   /settings/api-keys permissions matrix
      - {{HEADER_TITLE}}         → uniform page-title string —
                                   "faster-whisper-backend · <slug>"
                                   plain text, used inside <title>
      - {{HEADER_BRAND}}         → branded header lockup (waveform mark +
                                   wordmark + slug) for <span class="title">

    Pages that don't include a given placeholder are returned unchanged."""
    # Resolve the /logs DOM cap: 0 in config means "auto = initial × 4".
    # Computed here so the JS gets a final integer and doesn't need its
    # own resolver. Per-render lookup is fine — render_page runs once
    # per page load and the cost is two attribute reads.
    _log_initial = int(getattr(cfg, "LOG_VIEWER_INITIAL_LINES", 2000))
    _log_dom = int(getattr(cfg, "LOG_VIEWER_DOM_MAX", 0)) or (_log_initial * 4)
    return (
        template
        .replace("{{NAV}}", nav_html(current))
        .replace("{{SEV_PILLS}}", sev_pills_html())
        .replace("{{NAV_CSS}}", NAV_CSS)
        .replace("{{LOG_VIEWER_INITIAL_LINES}}", str(_log_initial))
        .replace("{{LOG_VIEWER_DOM_MAX}}", str(_log_dom))
        .replace("{{SCALE_PICKER}}", SCALE_PICKER_HTML)
        .replace("{{RELOAD}}", RELOAD_BTN_HTML)
        .replace("{{LOGOUT}}", LOGOUT_BTN_HTML)
        .replace("{{SCALE_PICKER_JS}}", SCALE_PICKER_JS + NAV_DRAWER_JS)
        .replace(
            "{{SEV_POLLER_JS}}",
            # Order matters: the global landing helpers must be defined
            # BEFORE OPEN_MODE_BANNER_JS runs, because the central script
            # calls `_renderNoAccessLanding` once whoami resolves.
            SEV_POLLER_JS + NOT_ADMIN_LANDING_GLOBAL_JS + OPEN_MODE_BANNER_JS,
        )
        .replace("{{SCALE_BOOTSTRAP_HEAD}}", SCALE_BOOTSTRAP_HEAD)
        .replace("{{RULE_EDITOR_JS}}", RULE_EDITOR_JS)
        .replace("{{TIME_HELPERS_JS}}", TIME_HELPERS_JS)
        .replace("{{NOT_ADMIN_LANDING_JS}}", NOT_ADMIN_LANDING_JS)
        .replace("{{PAGE_META}}", _page_meta_tag(current))
        .replace("{{TAG_PICKER_JS}}", TAG_PICKER_JS)
        .replace("{{HEADER_TITLE}}", _header_title_for(current))
        .replace("{{HEADER_BRAND}}", _header_brand_for(current))
    )


# Maps the `current=` argument passed by each page route into the
# permission-key used by api_keys_store.PAGES. Pages absent from this
# map (api-keys, settings) are admin-only — they don't participate in
# per-page scope gating; the central JS short-circuits for them.
_PAGE_KEY_BY_CURRENT: dict[str, str] = {
    "logs":         "logs",
    "stats":        "stats",
    "quick-config": "quick_config",
    "reports":      "reports",
    "captures":     "captures",
    # Admin-only pages: emit a sentinel so the JS can distinguish
    # "no per-page perm to check" from "we're on a data page that maps
    # to a key in the permissions dict".
    "settings":     "__admin_only__",
    "api-keys":     "__admin_only__",
}

# Display-URL for each page, used as the heading slug in the no-access
# landing. Keeps the rendered slug in lockstep with what the user typed
# in the address bar (the permission-key alone is misleading for nested
# routes like /settings/api-keys — without this, the landing would say
# "No access to /api-keys").
_PAGE_PATH_BY_CURRENT: dict[str, str] = {
    "logs":         "/logs",
    "stats":        "/stats",
    "quick-config": "/quick-config",
    "reports":      "/reports",
    "captures":     "/captures",
    "settings":     "/settings",
    "api-keys":     "/settings/api-keys",
}


# Human-readable slug per page, used in the uniform header string
# `faster-whisper-backend · <slug>`. Centralising this here means
# adding a page or renaming one is a single-line change instead of
# touching every template's <title> + <span class="title">.
_HEADER_SLUG_BY_CURRENT: dict[str, str] = {
    "logs":         "logs",
    "stats":        "stats",
    "settings":     "settings",
    "api-keys":     "API keys",
    "quick-config": "quick config",
    "reports":      "reports",
    "captures":     "captures",
}


def _header_title_for(current: str) -> str:
    """Build the uniform page-title string substituted into every page's
    <title> via {{HEADER_TITLE}} (plain text — no markup). Format:
    `faster-whisper-backend · <slug>`. Unknown `current` values fall
    back to the app name alone."""
    slug = _HEADER_SLUG_BY_CURRENT.get(current, "")
    if not slug:
        return "faster-whisper-backend"
    return f"faster-whisper-backend · {slug}"


# Inline brand mark: the forward-skewed audio-waveform tile (same geometry
# as static/logo.svg). Sized in em so it tracks the --fs-base UI scale.
_BRAND_MARK_SVG = (
    '<svg class="brand-mark" viewBox="0 0 120 120" aria-hidden="true" focusable="false">'
    '<defs><linearGradient id="fw-hdr" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#79c0ff"/><stop offset="1" stop-color="#7ee787"/>'
    '</linearGradient></defs>'
    '<rect x="6" y="6" width="108" height="108" rx="26" fill="#161b22" '
    'stroke="#30363d" stroke-width="2"/>'
    '<g transform="translate(13 2) skewX(-9)" fill="url(#fw-hdr)">'
    '<rect x="16" y="74" width="11" height="20" rx="5.5"/>'
    '<rect x="35" y="52" width="11" height="42" rx="5.5"/>'
    '<rect x="54" y="22" width="11" height="72" rx="5.5"/>'
    '<rect x="73" y="44" width="11" height="50" rx="5.5"/>'
    '<rect x="92" y="66" width="11" height="28" rx="5.5"/>'
    '</g></svg>'
)


def _header_brand_for(current: str) -> str:
    """Build the branded header lockup substituted into every page's
    <span class="title"> via {{HEADER_BRAND}} (HTML — kept separate from
    the plain-text {{HEADER_TITLE}} used inside <title>). Renders the
    waveform mark + compact wordmark `fasterwhisper › backend`. The current
    page is conveyed by the active nav link (aria-current), so the brand no
    longer repeats it as a slug. `current` is unused but kept for a uniform
    {{...}}-builder signature."""
    return (
        f"{_BRAND_MARK_SVG}"
        '<span class="brand-word">'
        '<span class="bw-a">faster</span><span class="bw-b">whisper</span>'
        '<span class="bw-sep">&gt;</span><span class="bw-c">backend</span>'
        "</span>"
    )


def _page_meta_tag(current: str) -> str:
    """Render a `<meta name="page-key" ...>` tag describing the current
    page. Carries both the permission-key (which the central script uses
    to decide whether to auto-render the landing) and the full URL path
    (which the landing renders as the heading slug — matches the address
    bar instead of just the permission slug)."""
    key = _PAGE_KEY_BY_CURRENT.get(current, "")
    if not key:
        return ""
    path = _PAGE_PATH_BY_CURRENT.get(current, "")
    extra = f' data-page-path="{path}"' if path else ""
    return f'<meta name="page-key" content="{key}"{extra}>'
