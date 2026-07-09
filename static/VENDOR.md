# Vendored static assets

These files are committed to the repo so the `/stats` dashboard works
fully offline — no CDN fetch at page-load.

## uPlot

- **Version**: 1.6.32
- **Source**: https://github.com/leeoniya/uPlot
- **License**: MIT
- **Files**:
  - `uplot.iife.min.js` (~50 KB) — IIFE build, exposes the global `uPlot`.
  - `uplot.min.css`     (~2 KB)  — default theme (we override colors via CSS vars).

## GridStack

- **Version**: 10.x
- **Source**: https://github.com/gridstack/gridstack.js
- **License**: MIT
- **Files**:
  - `gridstack.min.js`  (~80 KB) — UMD build (gridstack-all), exposes the global `GridStack`.
  - `gridstack.min.css` (~4 KB)  — default theme (we override colors via CSS vars).
- **Used by**: `/stats` dashboard for drag-to-reorder + click-to-resize tiles.

## Brand fonts (Hubot Sans, Geist Mono)

- **Source**: the `@fontsource-variable/hubot-sans` and `@fontsource-variable/geist-mono`
  packages (same pinned files the faster-whisper-frontend app bundles).
- **License**: SIL OFL-1.1 (both).
- **Files**:
  - `hubot-sans-latin-wght-normal.woff2` (~48 KB) — variable weight 200–900.
  - `geist-mono-latin-wght-normal.woff2` (~30 KB) — variable weight 100–900.
- **Used by**: the header brand lockup on every WebUI page (family wordmark
  grammar shared with faster-whisper-frontend), and `docs/brand/lockup.html`
  for rendering the README lockup PNGs.

## How to update

```bash
curl -sL -o uplot.iife.min.js \
  "https://cdn.jsdelivr.net/npm/uplot@<NEW_VERSION>/dist/uPlot.iife.min.js"
curl -sL -o uplot.min.css \
  "https://cdn.jsdelivr.net/npm/uplot@<NEW_VERSION>/dist/uPlot.min.css"
curl -sL -o gridstack.min.js \
  "https://cdn.jsdelivr.net/npm/gridstack@<NEW_VERSION>/dist/gridstack-all.js"
curl -sL -o gridstack.min.css \
  "https://cdn.jsdelivr.net/npm/gridstack@<NEW_VERSION>/dist/gridstack.min.css"
```

Then bump the version in this file. Do not hand-edit the JS or CSS — keep
them byte-identical to the upstream release so `git blame` stays meaningful.
