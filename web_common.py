"""
Shared helpers used by the /logs, /config, /stats, and /quick-config pages.

  - require_allowed_host(allowlist) — FastAPI dependency that 403s callers
    not in the allowlist. Allowlist accepts bare IPs or CIDRs.
  - nav_html(current, request)      — server-rendered nav row HTML.
  - severity_counts()               — log-level counters (last 60 s).
  - TokenWithGrace                  — bearer token with 60 s rotation grace.
"""

from __future__ import annotations

import ipaddress
import logging
import secrets
import time
from collections import deque
from typing import Callable

from fastapi import HTTPException, Request, status

import config as cfg


class TokenWithGrace:
    """Bearer token whose previous value is honored for 60 s after rotation,
    so an editing session can refresh its stored token mid-flight without
    locking itself out.

    `current_ref` is a zero-arg callable returning the current token (or
    None / empty for "no token set"). The indirection lets the WebUI edit
    the value on cfg at runtime and the next match() call picks up the new
    one. Loopback bypass is handled by the IP-gate dependency, not here.
    """
    GRACE_S = 60.0

    def __init__(self, current_ref: Callable[[], "str | None"]) -> None:
        self._current_ref = current_ref
        self._previous: "str | None" = None
        self._previous_expires_at: float = 0.0

    def record_rotation(self, old: "str | None") -> None:
        """Stash the pre-rotate token + expiry. Empty old token recorded as
        None (== bypass). Call from the admin save handler when the value
        changes."""
        self._previous = old or None
        self._previous_expires_at = time.monotonic() + self.GRACE_S

    def _previous_valid(self) -> "str | None":
        if not self._previous:
            return None
        if time.monotonic() > self._previous_expires_at:
            self._previous = None
            self._previous_expires_at = 0.0
            return None
        return self._previous

    def is_set(self) -> bool:
        """Return True if the current value is set (not None / not empty)."""
        return bool(self._current_ref())

    def matches(self, presented: str) -> bool:
        """Constant-time match against the current value, with 60 s grace
        for the previous value. If the current value is unset, returns
        False (caller should treat this as "auth disabled" via is_set())."""
        expected = self._current_ref()
        if not expected:
            return False
        if secrets.compare_digest(presented, expected):
            return True
        grace = self._previous_valid()
        if grace and secrets.compare_digest(presented, grace):
            return True
        return False


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


# --- Severity ring (in-memory log-level counts, since process start) ---------
# A logging.Handler appends (timestamp, levelno) on every record. The /stats
# page and the nav row read severity_counts() at request time. Bounded ring
# (maxlen=2000) keeps memory predictable under burst logging — once the ring
# fills, oldest entries fall off and the per-level counters cap accordingly.
# (We keep the timestamp tuple in case a future window-based view wants it.)
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
/* ---- Responsive header ----
   The header is a single flex row that, at narrow widths or scaled-up
   --fs-base, would otherwise push its right-edge items (save, status)
   off-screen. We let it wrap onto multiple rows and shrink the title
   intrinsically, with container queries dropping low-priority labels
   before resorting to wrap. Container size queries are evaluated in
   rem against the actual rendered header width, so they respect the
   --fs-base scale token (unlike @media). */
/* Shared header layout — every page renders the same flex row with
   title · nav · pills · spacer · scale-picker · page-specific buttons.
   Page-local CSS used to redeclare these; consolidated here so a single
   change keeps every admin page consistent. */
header { position: sticky; top: 0; background: var(--panel);
  border-bottom: 1px solid var(--border); z-index: 10; padding: 0; }
header > .header-inner { display: flex; gap: 0.75rem; align-items: center;
  max-width: 1100px; margin: 0 auto; width: 100%;
  padding: 0.5rem 0.875rem; box-sizing: border-box; }
header .title { font-weight: 600; color: var(--bold);
  white-space: nowrap; flex-shrink: 0; }
header .spacer { flex: 1; }

header { container-type: inline-size; container-name: hdr; }
header > .header-inner { flex-wrap: wrap; row-gap: 0.4rem; }
/* Spacer collapses to zero on a wrapped row; on a single row it still
   does its "push the action cluster right" job. */
header .spacer { flex: 1 1 0; min-width: 0; }
/* Title may shrink and ellipsise instead of forcing the row to grow.
   max-width caps how much it can take before truncating so it doesn't
   dominate small windows. Overrides the page-local `flex-shrink: 0`
   rule because NAV_CSS is injected later in the cascade. */
header .title {
  flex-shrink: 1; min-width: 0; max-width: 22rem;
  overflow: hidden; text-overflow: ellipsis;
}
/* Status pill is informational; let it shrink and ellipsise. */
header #status {
  flex-shrink: 1; min-width: 0; max-width: 12rem;
  overflow: hidden; text-overflow: ellipsis;
}
/* Wrap-anchor: zero-size sentinel placed before the action cluster.
   Hidden by default; at stage 3 it expands to flex-basis:100% to force
   the actions onto their own row, keeping title+nav+pills clean. */
header .wrap-anchor { flex-basis: 0; height: 0; display: none; }
/* Stage 1 (≤ 60rem): drop sevpill text label, keep the count. */
@container hdr (max-width: 60rem) {
  header .sevpill .lbl { display: none; }
  header .sevpill { padding: 0.125rem 0.4rem; }
}
/* Stage 2 (≤ 46rem): hide informational status pill, tighten nav. */
@container hdr (max-width: 46rem) {
  header #status { display: none; }
  header .navlink { padding: 0.1875rem 0.4rem; }
}
/* Stage 3 (≤ 36rem): drop logout, force action cluster to its own row. */
@container hdr (max-width: 36rem) {
  header #logout-btn { display: none; }
  header .wrap-anchor { display: block; flex-basis: 100%; }
}

/* ---- Admin-only nav elements ----
   logs/stats/config nav links + sev pills are marked .admin-only at
   render time. Hidden by default; revealed when the page's JS adds
   `body.role-admin` after a successful auth-required state fetch.
   /config, /logs, /stats add the class unconditionally (their state
   endpoints already require admin token). /quick-config adds it only
   when /quick-config/state returns role=admin — USER_TOKEN sessions
   never see admin-only chrome. */
header .admin-only { display: none; }
body.role-admin header .admin-only { display: inline-flex; }
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


# Severity pill poller — placed at the end of <body> on every page that shows
# the nav. Polls /sev every 5 s and writes the counts into the three pills.
# Server-side severity_counts() is the authoritative source (true 60 s window
# from real log-record timestamps). Pages with their own faster updates
# (e.g. /stats SSE, /logs per-line bumps) overwrite the same pills more
# often — the poller is a backstop that keeps every page consistent.
#
# Skips the work if no pills exist on the page (e.g. tests, future pages).
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


# Per-rule body editors shared by /config (full editor with drag-reorder etc.)
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

function _makeMonoLabeledInput(label, val, onInput, kind) {
  // kind === 'escape' → display \n/\r/\t/\\ as literal 2-char escapes,
  // decode on input. Required for fields like regex `replacement` that
  // can hold real newlines (single-line <input> strips them otherwise).
  const lbl = document.createElement('div');
  lbl.className = 'help';
  lbl.textContent = label + ':';
  const inp = document.createElement('input');
  inp.type = 'text';
  inp.spellcheck = false;
  inp.autocomplete = 'off';
  const raw = val == null ? '' : val;
  inp.value = (kind === 'escape') ? _esc(raw) : raw;
  inp.addEventListener('input', () => onInput(
    kind === 'escape' ? _unesc(inp.value) : inp.value
  ));
  const wrap = document.createElement('div');
  wrap.appendChild(lbl); wrap.appendChild(inp);
  return wrap;
}

function _makeMapRow(rule, key, val, commitData, datalistId) {
  const tr = document.createElement('tr');
  const td1 = document.createElement('td');
  const td2 = document.createElement('td');
  const td3 = document.createElement('td');
  td3.style.width = '2.5rem';
  const ki = document.createElement('input');
  ki.type = 'text'; ki.value = _esc(key);
  // Spoken-word cell opts into a datalist on /quick-config so end-users
  // get autocomplete from recent transcription FINALs. Admin /config
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
  return tr;
}

function renderTypeEditor(rule, commitData, opts) {
  // opts (optional): { datalistId } — passed through to _makeMapRow for
  // cb:map autocomplete on /quick-config. Other rule types ignore opts.
  opts = opts || {};
  const box = document.createElement('div');
  box.className = 'rule-editor';

  if (rule.type === 'terminal') {
    const note = document.createElement('div');
    note.className = 'help';
    note.textContent = 'Hardcoded terminal step: lstrip(" \\t\\r") + rstrip(" \\t\\r"). '
      + 'Always runs last. Preserves a leading or trailing newline ("\\n") emitted by '
      + '"neue Zeile" / "neuer Absatz" at the edges of the utterance.';
    box.appendChild(note);
    return box;
  }

  if (rule.type === 'regex') {
    box.appendChild(_makeMonoLabeledInput('pattern', rule.pattern, (v) => {
      rule.pattern = v; commitData();
    }));
    box.appendChild(_makeMonoLabeledInput('replacement', rule.replacement, (v) => {
      rule.replacement = v; commitData();
    }));
    return box;
  }

  if (rule.type === 'callback:lowercase-wordlist') {
    box.appendChild(_makeMonoLabeledInput('pattern', rule.pattern, (v) => {
      rule.pattern = v; commitData();
    }));
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
    box.appendChild(ta);
    return box;
  }

  if (rule.type === 'callback:map') {
    const note = document.createElement('div');
    note.className = 'help';
    note.textContent = 'Pattern auto-built from map keys (longest-first, '
      + 'word-bounded, case-insensitive). Edit entries below.';
    box.appendChild(note);
    const tbl = document.createElement('table');
    tbl.className = 'map-table';
    tbl.style.width = '100%';
    const rows = Object.entries(rule.map || {});
    rows.forEach(([k, v]) => tbl.appendChild(_makeMapRow(rule, k, v, commitData, opts.datalistId)));
    box.appendChild(tbl);
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.textContent = '+ add entry';
    addBtn.style.marginTop = '0.4rem';
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
      const newTr = _makeMapRow(rule, '', '', commitData, opts.datalistId);
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
    box.appendChild(addBtn);
    return box;
  }

  if (rule.type === 'callback:dedup' || rule.type === 'callback:upper') {
    box.appendChild(_makeMonoLabeledInput('pattern', rule.pattern, (v) => {
      rule.pattern = v; commitData();
    }));
    const note = document.createElement('div');
    note.className = 'help';
    note.textContent = rule.type === 'callback:dedup'
      ? 'Callback: collapse each match — last non-comma wins; pure-comma run → single comma.'
      : 'Callback: uppercase group(2) (or whole match if pattern has fewer than 2 groups).';
    box.appendChild(note);
    return box;
  }

  return box;
}
</script>
"""


def _nav_items(current: str) -> list[tuple[str, str, bool]]:
    """Return [(label, href, active), ...] honoring cfg.ADMIN_UI_ENABLED."""
    items: list[tuple[str, str, bool]] = [
        ("logs",  "/logs",  current == "logs"),
        ("stats", "/stats", current == "stats"),
    ]
    if getattr(cfg, "ADMIN_UI_ENABLED", False):
        items.append(("config", "/config", current == "config"))
        items.append(("quick", "/quick-config", current == "quick-config"))
        items.append(("reports", "/reports", current == "reports"))
        items.append(("captures", "/captures", current == "captures"))
    return items


def nav_html(current: str) -> str:
    """Render the nav row + severity pills as an HTML fragment.

    Pills link to /logs?filter=<level> so a click jumps to the relevant log
    rows. Counts of zero render dimmed; non-zero render colored ("hot")."""
    counts = severity_counts()
    # logs/stats/config + sev pills are admin-only — every page renders
    # them with class "admin-only" which CSS hides by default. The page's
    # JS adds `body.role-admin` after a successful state-load, revealing
    # them. /quick-config in a USER_TOKEN session never gets that class
    # so the admin links stay hidden. The "quick" link is for everyone.
    admin_only_labels = {"logs", "stats", "config", "reports", "captures"}
    parts: list[str] = ['<span class="navrow">']
    for label, href, active in _nav_items(current):
        classes = ["navlink"]
        if active:
            classes.append("active")
        if label in admin_only_labels:
            classes.append("admin-only")
        parts.append(
            f'<a class="{" ".join(classes)}" href="{href}">{label}</a>'
        )
    parts.append("</span>")

    # Stable IDs let JS update just the .n inner span on each SSE tick without
    # rebuilding the link (preserves focus/click state). The initial counts
    # rendered here are a "best effort at page load" — the client takes over
    # immediately, so they're correct for the first render and live thereafter.
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
    return "".join(parts)


def render_page(template: str, current: str) -> str:
    """Substitute placeholders in a page template:
      - {{NAV}}                  → nav row + severity pills
      - {{NAV_CSS}}              → shared header/scale-token CSS
      - {{SCALE_PICKER}}         → scale dropdown (header)
      - {{SCALE_PICKER_JS}}      → wire-up script (end of body)
      - {{SEV_POLLER_JS}}        → 5-s pill re-sync (end of body)
      - {{SCALE_BOOTSTRAP_HEAD}} → tiny pre-paint script (top of <head>)
      - {{RULE_EDITOR_JS}}       → shared per-rule body editors

    Pages that don't include a given placeholder are returned unchanged."""
    return (
        template
        .replace("{{NAV}}", nav_html(current))
        .replace("{{NAV_CSS}}", NAV_CSS)
        .replace("{{SCALE_PICKER}}", SCALE_PICKER_HTML)
        .replace("{{SCALE_PICKER_JS}}", SCALE_PICKER_JS)
        .replace("{{SEV_POLLER_JS}}", SEV_POLLER_JS)
        .replace("{{SCALE_BOOTSTRAP_HEAD}}", SCALE_BOOTSTRAP_HEAD)
        .replace("{{RULE_EDITOR_JS}}", RULE_EDITOR_JS)
    )
