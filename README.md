<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo-lockup-dark.svg">
    <img src="static/logo-lockup-light.svg" alt="faster-whisper-backend" width="420">
  </picture>
</p>

# faster-whisper-backend

Self-hosted [faster-whisper](https://github.com/SYSTRAN/faster-whisper) API for **German-language medical/clinical dictation**, with a Swiss-German (CH-DE) orthography + dictation post-processing layer. Exposes an **OpenAI-compatible** `/v1/audio/transcriptions` endpoint for use with [vowen.ai](https://vowen.ai) and other Whisper clients.

## Features

- OpenAI-compatible API — drop-in replacement for `client.audio.transcriptions.create(...)`
- GPU-accelerated (CUDA) via faster-whisper + CTranslate2, with **automatic CPU fallback** when no GPU is available
- **Per-request model selection** — clients pass `model="large-v3"` / `"large-v3-turbo"` / any HF repo id; LRU-cached in VRAM
- **CH-DE locale**: ß → ss, Swiss medical vocabulary in default prompt (Spital, Krankenkasse, FMH, CHF)
- **German dictation map**: `"Punkt"` → `.`, `"Komma"` → `,`, `"neue Zeile"` → `\n`, `"Klammer auf"` → `(`, ~60 phrases total
- Auto-capitalize after sentence ends; strips Whisper noise commas; lowercases mid-sentence non-nouns after stripped Whisper terminators
- Live HTML log viewer at `/logs` (Server-Sent Events, color-coded pipeline trace per request)
- Live system overview at `/stats` (loaded models + VRAM, GPU/CPU/RAM, request latency, recent transcriptions, sparklines — works fully offline, no CDN)
- Admin WebUI at `/settings` for editing every setting without redeploying (on by default; allowlist + bearer-token gated)
- **Configure everything via environment** — every setting has a `WHISPER_*` variable; pin them via `.env`, docker-compose, or the service env. Env-pinned settings are shown read-only in the admin WebUI.
- Cross-page nav with severity pills (WARN/ERR/CRIT in the last 60 s) on every page
- Runs anywhere: Windows Service, Linux systemd, Docker, or bare `python main.py`

## Requirements

- Python 3.14 (the Docker image ships `python:3.14-slim`; 3.12+ works for bare installs)
- Linux, macOS, or Windows 10/11
- **GPU (optional)**: NVIDIA GPU + driver supporting CUDA 12.x (WSL2 driver works).

The default config is **GPU-first** (`MODEL_DEVICE = "cuda"`): on a host without a
usable GPU, each model load automatically falls back to CPU (`int8`), logging a
one-time fallback warning per model. To make CPU the primary path (and silence the
warning), set `WHISPER_MODEL_DEVICE=cpu`. The base `requirements.txt` is CPU-capable
and cross-platform; GPU acceleration is additive — `pip install -r requirements-gpu.txt`
on an NVIDIA host.

## Install

### Linux / macOS (development)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt            # CPU, all platforms
# pip install -r requirements-gpu.txt      # add NVIDIA CUDA wheels (GPU box)
python main.py                             # serves on http://0.0.0.0:8000
```

### Linux (production, systemd)

```bash
./install-service.sh        # CPU   (auto-elevates via sudo, creates venv,
./install-service.sh --gpu  # GPU    installs deps, writes + starts the unit)
# manage: systemctl status|restart whisper-api ; journalctl -u whisper-api -f
# remove: ./uninstall-service.sh
```

### Docker (any OS)

CI publishes two images to GHCR on every push to `main` (and on `v*` tags):
`:latest` (CPU) and `:latest-gpu` (adds the CUDA 12 / cuDNN 9 wheels). Both are
also tagged `:v<version>` / `:v<version>-gpu` and `:sha-<short>` / `:sha-<short>-gpu`.

```bash
# CPU — pulls ghcr.io/<owner>/faster-whisper-backend:latest
docker compose up -d

# GPU — pulls the :latest-gpu image and passes through the host NVIDIA GPU(s)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

The GPU path needs an NVIDIA driver (CUDA 12.x) **and** the NVIDIA Container
Toolkit on the host; the device is exposed via `docker-compose.gpu.yml`. With no
GPU visible, model load auto-falls back to CPU/int8. To build locally instead of
pulling, uncomment `build:` in the compose file(s) (the GPU build uses
`Dockerfile.gpu`). If the GHCR package is private, `docker login ghcr.io` first.

### Windows (production, service)

```powershell
# Auto-elevates via UAC, bootstraps the venv, installs requirements, downloads
# WinSW, and registers the Windows Service in one go.
.\install-service.ps1
# On a GPU box, also: venv\Scripts\python -m pip install -r requirements-gpu.txt
```

First server start eagerly preloads the models in `PRELOAD_MODELS` (by default
three: `large-v2`, `large-v3`, and the German finetune
`GalaktischeGurke/primeline-whisper-large-v3-german-ct2` — several GB total) into
the HuggingFace cache (`~/.cache/huggingface/hub/`, or
`%USERPROFILE%\.cache\huggingface\hub\` on Windows; override with
`WHISPER_DOWNLOAD_ROOT`). Set `WHISPER_PRELOAD_MODELS=large-v2` (or empty) to
download/warm fewer models at startup.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

CI runs the suite on Linux and Windows for every push (`.github/workflows/ci.yml`).

## Usage

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
with open("audio.wav", "rb") as f:
    r = client.audio.transcriptions.create(
        model="whisper-1", file=f,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )
print(r.text)
```

## Configuration

Most knobs live in **`config.py`** at the repo root — models, default prompt, server host/port, log paths, faster-whisper transcribe defaults. The **text post-processing pipeline rules** (dictation map, German non-noun list, regex tidy rules) live in a separate committed file, **`config.json`** — see [Post-processing pipeline](#post-processing-pipeline). Edit either file directly to change behavior, then restart the service (`systemctl restart whisper-api` / `Restart-Service WhisperAPI` / the `/settings` restart button) to pick up the changes. The algorithm code in `main.py` doesn't need to be touched.

Layers of overrides, **env wins over file wins over in-repo default**:

1. **`config.py`** / **`config.json`** — committed in-repo defaults (scalar settings / pipeline rules respectively).
2. **`config.local.json`** (gitignored) — runtime overrides written by the admin WebUI; or hand-edited (see `config.local.example.json`). Validated against `config_store.AdminConfig`; unknown keys are rejected.
3. **`WHISPER_*` env vars** — per-machine deployment pins; always win. Source them from a `.env` file (auto-loaded at startup), docker-compose `environment:` / `env_file:`, or the service env (`<env>` elements in `WhisperAPI.xml`, regenerated by `install-service.ps1`).

### Environment variables

**Every** editable setting has a matching `WHISPER_<FIELD>` variable — see
[`.env.example`](.env.example) for the complete, grouped list with defaults.
Copy it to get started:

```bash
cp .env.example .env     # auto-loaded on startup; gitignored
```

Notes:
- An env-pinned setting is **read-only in the admin WebUI** (greyed out, badged
  `env: WHISPER_…`). Unset the variable to make it editable in the UI again.
- Booleans accept `1/true/yes/on`; lists are comma-separated; an empty value
  clears/disables nullable settings (e.g. `WHISPER_NO_SPEECH_THRESHOLD=`).
- **Secrets** (`WHISPER_BOOTSTRAP_ADMIN_KEY`, `WHISPER_USE_AUTH_TOKEN`) also
  accept a `*_FILE` form pointing at a mounted secret file, so the value stays
  out of `docker inspect` / the process environment.
- **Server port**: `WHISPER_SERVER_PORT` is honored, but in Docker you must also
  update the compose `ports:` mapping to match.
- **Structured settings** (`PIPELINE_RULES`, `MODEL_OVERRIDES`) accept a JSON
  string; per-model fields can also be set one at a time via
  `WHISPER_MODEL_OVERRIDE__<id>__<FIELD>` (encode `/`→`__SLASH__`, `.`→`__DOT__`).

A few of the most common variables:

| Env var | Maps to `config.py` | Effect |
|---|---|---|
| `WHISPER_DEFAULT_MODEL` | `DEFAULT_MODEL` | Model used when request sends `whisper-1` or omits `model` |
| `WHISPER_ALLOWED_MODELS` | `ALLOWED_MODELS` | Comma-separated allowlist (default: a curated 3-model set); empty = any model passes |
| `WHISPER_MODEL_DEVICE` | `MODEL_DEVICE` | `cuda` (default) or `cpu` |
| `WHISPER_PRELOAD_MODELS` | `PRELOAD_MODELS` | Comma-separated list to load eagerly at startup (no first-request warm-up) |
| `WHISPER_SERVER_PORT` | `SERVER_PORT` | Listen port (also update the Docker `ports:` mapping) |
| `WHISPER_DEFAULT_PROMPT` | `DEFAULT_PROMPT` | Initial prompt when request omits `prompt` (empty string disables) |
| `WHISPER_ADMIN_UI` | `ADMIN_UI_ENABLED` | `0` unregisters `/settings*` (on by default) |
| `WHISPER_ADMIN_ALLOWED_HOSTS` | `ADMIN_ALLOWED_HOSTS` | Comma-separated IPs/CIDRs allowed to reach `/settings` (loopback always implicit) |
| `WHISPER_STATS_ALLOWED_HOSTS` | `STATS_ALLOWED_HOSTS` | Comma-separated IPs/CIDRs allowed to reach `/stats` (loopback always implicit) |

### Allowed hosts

`/settings` and `/stats` are gated by IP/CIDR allowlists. Defaults are `["127.0.0.1", "::1"]` — loopback only. Loopback is *always* implicitly allowed regardless of the configured list, so a typo can never lock you out from the box itself.

```python
# Allow the local LAN to reach /stats but keep /settings loopback-only:
ADMIN_ALLOWED_HOSTS = ["127.0.0.1", "::1"]
STATS_ALLOWED_HOSTS = ["127.0.0.1", "::1", "192.168.1.0/24"]
```

CIDR is accepted (`192.168.0.0/16`) and so are bare IPs (`10.0.0.5`).

## Endpoints

- `POST /v1/audio/transcriptions` — OpenAI-compatible transcription. Pass `model=<name>` to pick a specific model (any faster-whisper short name or HF repo id).
- `GET  /v1/models` — list currently-loaded models, the configured default, and the allowlist (if set).
- `GET  /logs` — live log viewer (browser).
- `GET  /logs/stream` — raw SSE feed.
- `GET  /stats` — system overview dashboard. Allowlist-gated (`STATS_ALLOWED_HOSTS`; loopback always allowed).
- `GET  /stats/snapshot`, `GET /stats/stream` — JSON snapshot + SSE stream of the same data (~1 Hz).
- `GET  /settings` — admin WebUI (registered by default; set `WHISPER_ADMIN_UI=0` to disable). Allowlist-gated (`ADMIN_ALLOWED_HOSTS`; loopback always allowed) plus API-key auth.
- `GET  /settings/api-keys` — admin UI for per-user API key management.
- `GET/POST /settings/state`, `POST /settings/restart` — admin JSON endpoints; require `Authorization: Bearer <api_key>` resolving to a user with `is_admin=True`.
- `GET  /auth/whoami` — resolve the current bearer to `{open_mode, user_id, username, is_admin}`. WebUI uses this to render the login modal and the OPEN-mode banner.

### Model selection examples

```python
# Use the configured default (Whisper-1 = OpenAI default name)
client.audio.transcriptions.create(model="whisper-1", file=f)

# Pick a specific faster-whisper short name
client.audio.transcriptions.create(model="large-v3-turbo", file=f)

# Use a German finetune from Hugging Face
client.audio.transcriptions.create(model="primeline/whisper-large-v3-turbo-german", file=f)
```

> **Note:** `ALLOWED_MODELS` ships as a curated 3-model set
> (`large-v2`, `large-v3`, `GalaktischeGurke/primeline-whisper-large-v3-german-ct2`),
> so requests for other ids (e.g. `large-v3-turbo`, `primeline/whisper-large-v3-turbo-german`)
> are rejected until you add them to the allowlist (`WHISPER_ALLOWED_MODELS=...`)
> or clear it (`WHISPER_ALLOWED_MODELS=` → any model passes).

First-use of any new model triggers a one-time download (~600 MB to ~1.5 GB depending on the model) into `%USERPROFILE%\.cache\huggingface\hub\`. Subsequent loads come from cache (~5–10 s into VRAM).

## Service control

Linux (systemd):

```bash
sudo systemctl restart whisper-api       # after editing main.py / config
sudo systemctl stop    whisper-api
systemctl status       whisper-api
./uninstall-service.sh                   # remove the service
```

Windows (service):

```powershell
Restart-Service WhisperAPI               # after editing main.py
Stop-Service    WhisperAPI
Get-Service     WhisperAPI
.\uninstall-service.ps1                  # remove the service
.\uninstall-service.ps1 -RemoveLocal     # also delete logs/, WhisperAPI.exe / .xml, any legacy nssm.exe
```

Docker: `docker compose restart` / `docker compose down`. Any deployment can also
self-restart from the admin UI (`/settings` → **restart**) — it re-execs the process
on Linux/macOS and uses WinSW on Windows.

`Get-Content -Wait logs\whisper.log` to tail logs in a terminal, or open `http://localhost:8000/logs`.

## Post-processing pipeline

A single ordered list of rules — `cfg.PIPELINE_RULES` — is applied to each transcript's joined text. Each row is one of:

- `regex` — pattern + replacement (`re.sub`)
- `callback:lowercase-wordlist` — strip terminator and lowercase next word if it's in the wordlist
- `callback:map` — auto-built alternation of map keys (longest-first, case-insensitive); look up replacement
- `callback:dedup` — collapse adjacent punctuation runs (last non-comma wins; pure-comma run → single comma)
- `callback:upper` — capitalize after sentence terminator
- `terminal` — final `lstrip(" \t\r") + rstrip(" \t\r")`; always last (preserves leading/trailing `\n`)

The 13 seeded defaults handle Swiss German orthography (`ß`→`ss`), Whisper noise stripping, dictation (`Punkt`→`.`, `neue Zeile`→`\n`, …), and tidy spacing/newlines/capitalization. They live in the committed **`config.json`** (`{"schema_version": 1, "PIPELINE_RULES": [...]}`); `config.py` loads that file at startup. Each rule carries an optional `note` field documenting its rationale.

**Ordering invariants:** `dictation-map` multi-word phrases must precede their single-word components (the alternation regex is rebuilt longest-first, so the longest phrase wins); the `terminal` trim rule is always last.

**Editing — the unified editor at `/settings`** (Pipeline section). One rule list
shows the **effective** pipeline (config.json overlaid by config.local.json). Edits
save to `config.local.json` via the page Save (gitignored, per-deployment). Each
rule carries an **origin badge**:

- `● factory` — matches `config.json`.
- `◆ edited` — in `config.json` but locally edited; offers `↺ reset` (discard the
  local edit) and `⇪ promote`.
- `✚ local-only` — not in `config.json`; offers `× delete` and `⇪ promote`.

**Promote** writes a rule (or, via *Promote all changes to config.json*, the whole
list) into the committed **`config.json`** — a diff dialog confirms first. Since
`config.json` is git-tracked, `git commit && git push` then ships the change to
every deployment. After *Promote all* you're offered to **clear the local override**
so `config.json` runs directly on this deployment too (otherwise the local snapshot
keeps shadowing it).

Factory rules cannot be deleted; uncheck `enabled` to disable. `config.json` is
required — if it is missing or malformed the service fails fast at startup; restore
it with `git checkout config.json`.

JSON response notes: `text` is the post-processed joined transcript. `segments[].text` and `words[].word` carry **raw** Whisper output (no post-processing applied to per-segment / per-word fields — multi-word dictation phrases like `"neue Zeile"` only resolve cleanly on the joined text).

## Stats dashboard

`http://localhost:8000/stats` shows a live dashboard updated over Server-Sent Events at ~1 Hz:

- **GPU**: name, util %, VRAM used/total, temp, power draw, SM clock, current performance state.
- **Host**: total CPU%, per-core mini-strip, RAM, free disk on the model cache drive.
- **Process**: PID, RSS, threads, uptime.
- **Loaded models** with per-model VRAM (NVML delta sample taken at construction time), warm/cold badge, and the cold-load history.
- **Request metrics**: in-flight transcriptions, p50/p95/p99 latency, endpoint counters, 5xx counts in 1m/5m/15m windows.
- **Recent transcriptions** ring (last 20) with model, audio length, wall-clock, real-time-factor, words emitted.

Sparklines are rendered with [uPlot](https://github.com/leeoniya/uPlot), vendored under `static/` so the page works **fully offline** — no CDN fetch at page-load. To update the bundled version, see `static/VENDOR.md`.

The `/stats` endpoint is allowlist-gated (`STATS_ALLOWED_HOSTS`). On a host without an NVIDIA GPU or with `nvidia-ml-py` missing, the GPU panel hides and the rest of the dashboard still works.

The nav row at the top of every page (logs ↔ stats ↔ config) also surfaces three severity pills counting `WARNING` / `ERROR` / `CRITICAL` records in the last 60 s; clicking any pill jumps to `/logs` with that filter prefilled.

## Admin WebUI (optional)

A second WebUI at `/settings` lets you edit any setting from `config.py` from the browser, with hot-reload for safe knobs (transcribe params, dictation map, prompt) and an automatic service restart for cold ones (server port, log file, preload list).

**On by default** (`ADMIN_UI_ENABLED = True`). Set `ADMIN_UI_ENABLED = False` in `config.py` (or `WHISPER_ADMIN_UI=0`) and restart to unregister `/settings*`. The page opens at `http://localhost:8000/settings` from the server itself or any host in `ADMIN_ALLOWED_HOSTS`. Settings pinned by a `WHISPER_*` env var appear **read-only** (greyed out, badged with the variable name) since the environment takes precedence.

### Authentication: per-user API keys

The transcription endpoint and every WebUI page are gated by **per-user API keys**, not a shared token. Each key looks like `wk_<43-char base64>` (256-bit entropy); raw keys are SHA-256-hashed at rest and shown **once** on creation.

**Bootstrap.** On a fresh install with no admin key in the DB, the server starts in **OPEN mode**: every request is accepted as a synthetic admin, a red banner appears on every WebUI page, and a `WARNING` log line fires every 60 s. This is the operator's prompt to generate the first admin key. Two ways:

1. **In the UI** — open `/settings/api-keys`, click "+ add user" with admin=true, then "+ generate key", and copy the raw key from the show-once modal.
2. **Via env var** — set `WHISPER_BOOTSTRAP_ADMIN_KEY=wk_…` on first start. A `bootstrap-admin` user is created (or skipped if the same key hash is already present) with that exact raw key. Subsequent starts no-op.

Once at least one active admin key exists, the OPEN-mode banner disappears and 401 is returned to unauthenticated callers.

**Using a key.** Clients (Vowen, curl, the WebUI login modal) send `Authorization: Bearer wk_…`. The WebUI stores it in `sessionStorage` (`whisper_api_key`) until tab close. On any 401 the modal re-prompts.

**Lockout protection.** Revoking the last active admin key (or the last admin user) returns 409. Generate a second admin key first.

**Multi-user.** Each capture is tagged with the originating `user_id`. Non-admin users see only their own captures in `/captures`; admins see all and can filter by user. Group merging is locked to a single speaker — the server rejects any merge whose members span more than one user.

Other layers:
- **Feature flag**: `ADMIN_UI_ENABLED = True` (the default; or `WHISPER_ADMIN_UI=1`) registers the routes. Set it `False` / `WHISPER_ADMIN_UI=0` and `/settings*` returns 404.
- **Host allowlist**: `ADMIN_ALLOWED_HOSTS` keeps the admin endpoints reachable only from the configured CIDRs (loopback always implicit).
- **Server-side validation**: every payload is validated against `config_store.AdminConfig` (Pydantic v2).
- **Auto-restart**: when a "cold" setting changes (server port, log file, preload list, …), a confirmation modal asks whether to restart the service. WinSW relaunches the wrapper; the page polls `/v1/models` until back up.

Edits land in **`config.local.json`** at the repo root (gitignored). See `config.local.example.json` for the schema. The one exception is the Pipeline section's **promote** action, which writes the committed **`config.json`** instead (see [Post-processing pipeline](#post-processing-pipeline)).

## Brand

The mark is an **audio waveform skewed forward** — *whisper* (the equalizer bars) meets *faster* (the rightward lean reads as motion / fast-forward) — on a rounded terminal-dark tile. The wordmark is a monospace lockup, `faster` (dim) + `whisper` (bold), with a green `>` prompt marking this as the **backend** service, distinct from the upstream [faster-whisper](https://github.com/SYSTRAN/faster-whisper) library it wraps. The palette mirrors the WebUI's GitHub-dark theme — cyan `#79c0ff` → green `#7ee787` on `#0d1117`.

**Lockups & variants** — each designed to work on dark *and* light backgrounds:

- **Primary lockup** — mark + stacked wordmark (`fasterwhisper` over `> BACKEND`); the hero above.
- **Compact** — single line, `[mark] fasterwhisper > backend`, for headers and tight spaces. The admin WebUI's sticky header uses this form.
- **Icon only** — the waveform tile alone (favicon, app tile, anything ≤ ~80 px).
- **Monochrome** — single-colour, tile-less bars for one-colour or print contexts.

**Type spec:** monospace throughout; hierarchy carried by weight + value (no rainbow of colours); the green `>` is the wordmark's only accent so the colour lives in the mark; the `BACKEND` sub-label is tracked caps at ~50 % cap-height. The same mark is the favicon and sits in the sticky header of every admin page.

Assets (`static/`):

| File | Use |
| --- | --- |
| `logo-lockup-dark.svg` / `logo-lockup-light.svg` | primary lockup, theme-adaptive (used in this README) |
| `logo.svg` | full icon mark (scalable) |
| `favicon.svg` | simplified 3-bar mark for small sizes |
| `favicon.ico`, `favicon-16.png`, `favicon-32.png`, `apple-touch-icon.png` | raster fallbacks (Safari / legacy) |

## Files

```
main.py                    FastAPI app + post-processing pipeline + log viewer
config.py                  User-tunable scalar settings (models, prompt, server, logging, ...)
config.json                Committed factory pipeline rules (PIPELINE_RULES); editable via /settings
config_store.py            Admin-WebUI persistence layer (Pydantic schema, atomic writes)
admin_routes.py            Admin /settings endpoints + HTML page (on by default; disable with WHISPER_ADMIN_UI=0)
stats_routes.py            /stats dashboard endpoints + HTML page (always on, allowlist-gated)
metrics.py                 In-process request metrics (counters, latency ring, recent transcriptions)
system_stats.py            GPU + host snapshot (pynvml + psutil; degrades gracefully if NVML missing)
web_common.py              Shared helpers: allowlist gate, nav HTML, severity counts
restart_service.py         Detached self-restart helper (Windows-only)
auth.py                    HTTPBearer dep — get_current_user / require_admin + OPEN-mode loop
api_keys_store.py          users + api_keys SQLite store (SHA-256 hash, soft revoke, O(1) lookup)
api_keys_routes.py         /settings/api-keys admin UI for per-user key management
audio_merge.py             stdlib-wave PCM splicer for ≤28 s training-sample packing
capture_groups_store.py    capture_groups table + dissolve / stale / regenerate helpers
captures_store.py          Capture rows + audio fanout, retention, eviction
captures_routes.py         /captures admin page + merge/dissolve/regenerate API
reports_store.py / reports_routes.py
                           User-submitted transcription error reports + admin triage
config.local.json          Runtime overrides written by the admin UI (gitignored, optional)
config.local.example.json  Example overrides file
.env.example               Documented list of every WHISPER_* env var + defaults (copy to .env)
test.py                    Manual test client (vowen.ai compatibility)
install-service.ps1        Windows Service installer (WinSW-based, self-elevating, auto-bootstraps venv)
uninstall-service.ps1      Windows Service uninstaller
install-service.sh         Linux systemd installer (self-elevating, auto-bootstraps venv); --gpu adds CUDA wheels
uninstall-service.sh       Linux systemd uninstaller
Dockerfile / Dockerfile.gpu / .dockerignore
docker-compose.yml / docker-compose.gpu.yml   CPU base + GPU overlay (NVIDIA)
                           CPU container image + compose (named volumes for state); run on any OS
requirements.txt           Base (CPU, cross-platform) deps; transitive resolved by pip
requirements-gpu.txt       NVIDIA CUDA wheels (opt-in, additive)
requirements-dev.txt       Test deps (pytest)
requirements-convert.txt   Deps for converting HF models to CTranslate2 (opt-in)
pytest.ini                 Test discovery config (pytest -q from repo root)
.github/workflows/ci.yml   CI: runs the test suite on Linux + Windows
static/                    Brand assets (logo.svg, favicon.*) + vendored uPlot/GridStack (offline /stats)
.gitignore / .gitattributes
logs/                      Created at first run; rotates at 10 MB × 10 files
```
