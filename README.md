<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/logo-lockup-dark.svg">
    <img src="static/logo-lockup-light.svg" alt="faster-whisper-backend" width="420">
  </picture>
</p>

# faster-whisper-backend

Self-hosted [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcription API with a fully configurable dictation post-processing pipeline (find→replace rules, word maps, spoken punctuation, casing). Exposes an **OpenAI-compatible** `/v1/audio/transcriptions` endpoint for any Whisper client.

## Features

- OpenAI-compatible API — drop-in replacement for `client.audio.transcriptions.create(...)`
- **Live streaming dictation** — WebSocket endpoint `/v1/audio/transcriptions/stream` that emits
  flicker-free partial text *while you speak* (LocalAgreement-2 stabilization) and **post-processed**
  final text per utterance (a locked, append-only `committed` prefix plus a revisable `tail` —
  both sent as full strings the client replaces). Reuses the same models, VAD, and post-processing
  pipeline as the batch route (which is unchanged); accepts raw 16 kHz PCM **or** browser Opus/WebM
  (decoded server-side via a bundled `ffmpeg` — `imageio-ffmpeg`, no system install needed; a
  system `ffmpeg` on PATH is used when present); two-tier Silero/energy endpointing. Try it in the browser at `/dictate`.
  On by default (auth-gated); tune everything via `WHISPER_STREAMING_*` / `/settings`. A shared
  `INFERENCE_CONCURRENCY` limiter governs streaming **and** batch so they don't oversubscribe the GPU.
- GPU-accelerated (CUDA) via faster-whisper + CTranslate2, with **automatic CPU fallback** when no GPU is available
- **Per-request model selection** — clients pass `model="large-v3"` / `"large-v3-turbo"` / any HF repo id; LRU-cached in VRAM
- **Dictation phrase map**: `"Punkt"` → `.`, `"Komma"` → `,`, `"neue Zeile"` → `\n`, `"Klammer auf"` → `(`, ~80 phrases total — every rule editable/replaceable in the WebUI
- Auto-capitalize after sentence ends; strips Whisper noise commas; lowercases mid-sentence non-nouns after stripped Whisper terminators
- Live HTML log viewer at `/logs` (Server-Sent Events, color-coded pipeline trace per request)
- Live system overview at `/stats` (loaded models + VRAM, GPU/CPU/RAM, request latency, recent transcriptions, sparklines — works fully offline, no CDN)
- Admin WebUI at `/settings` for editing every setting without redeploying (on by default; allowlist + bearer-token gated)
- **Configure everything via environment** — every setting has a `WHISPER_*` variable; pin them via `.env`, docker-compose, or the service env. Env-pinned settings are shown read-only in the admin WebUI.
- Cross-page nav with severity pills (WARN/ERR/CRIT counts since process start) on every page
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

# GPU — standalone file: pulls :latest-gpu and passes through the host NVIDIA GPU(s)
docker compose -f docker-compose.gpu.yml up -d
```

`docker-compose.gpu.yml` is a self-contained mirror of `docker-compose.yml` —
same ports/env/volumes, differing only in the GPU bits (`:latest-gpu` image +
the NVIDIA device reservation). The GPU path needs an NVIDIA driver (CUDA 12.x)
**and** the NVIDIA Container Toolkit on the host. With no GPU visible, model load
auto-falls back to CPU/int8. To build locally instead of pulling, uncomment
`build:` in the compose file (the GPU build uses `Dockerfile.gpu`). If the GHCR
package is private, `docker login ghcr.io` first.

The container runs as a **non-root user** (default `1000:1000`); set `PUID` /
`PGID` in `.env` (or the environment) to run as a different user/group — no
rebuild needed, volumes work with any UID out of the box.

### Windows (production, service)

```powershell
# Auto-elevates via UAC, bootstraps the venv, installs requirements, downloads
# WinSW, and registers the Windows Service in one go.
.\install-service.ps1
# On a GPU box, also: venv\Scripts\python -m pip install -r requirements-gpu.txt
```

First server start eagerly preloads the models in `PRELOAD_MODELS` (by default
two: `Systran/faster-whisper-large-v2` and `Systran/faster-whisper-large-v3` —
several GB total) into the HuggingFace cache (`~/.cache/huggingface/hub/`, or
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

**Every** factory default — models, default prompt, server host/port, log paths, faster-whisper transcribe defaults, **and** the post-processing pipeline rules — lives in the committed **`config.json`** at the repo root (single source of truth; `config.py` only loads it and layers the overrides below on top). Edit it directly to change a default for every deployment, then restart the service (`systemctl restart whisper-api` / `Restart-Service WhisperAPI` / the `/settings` restart button) to pick up the changes. The algorithm code in `main.py` doesn't need to be touched.

Layers of overrides, **env wins over file wins over in-repo default**:

1. **`config.json`** — committed in-repo defaults (all scalar settings + pipeline rules).
2. **`config.local.json`** (gitignored) — runtime overrides written by the admin WebUI; or hand-edited (see `config.local.example.json`). Validated against `config_store.AdminConfig`; unknown keys are rejected. Defaults next to the code; set **`WHISPER_CONFIG_LOCAL`** to relocate it (e.g. `/data/config.local.json`) when `/app` is a read-only image dir — otherwise the WebUI save fails with `Permission denied`.
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

| Env var | Maps to setting | Effect |
|---|---|---|
| `WHISPER_DEFAULT_MODEL` | `DEFAULT_MODEL` | Model used when request sends `whisper-1` or omits `model` |
| `WHISPER_ALLOWED_MODELS` | `ALLOWED_MODELS` | Comma-separated allowlist (default: the two official Systran large-v2/large-v3 builds); empty = any model passes |
| `WHISPER_MODEL_DEVICE` | `MODEL_DEVICE` | `cuda` (default) or `cpu` |
| `WHISPER_PRELOAD_MODELS` | `PRELOAD_MODELS` | Comma-separated list to load eagerly at startup (no first-request warm-up) |
| `WHISPER_SERVER_PORT` | `SERVER_PORT` | Listen port (also update the Docker `ports:` mapping) |
| `WHISPER_DEFAULT_PROMPT` | `DEFAULT_PROMPT` | Initial prompt when request omits `prompt` (empty string disables) |
| `WHISPER_ADMIN_UI` | `ADMIN_UI_ENABLED` | `0` unregisters `/settings*` + `/quick-config`, `/captures`, `/reports` (on by default) |
| `WHISPER_ADMIN_WEBUI_ALLOWED_HOSTS` | `ADMIN_WEBUI_ALLOWED_HOSTS` | Comma-separated IPs/CIDRs allowed to reach the **admin** pages — `/settings`, `/settings/api-keys`, `/docs` (loopback always implicit; default loopback only) |
| `WHISPER_USER_WEBUI_ALLOWED_HOSTS` | `USER_WEBUI_ALLOWED_HOSTS` | Comma-separated IPs/CIDRs allowed to reach the **user** pages — `/quick-config`, `/captures`, `/reports`, `/stats`, `/logs`, `/dictate`, `/sev` (loopback always implicit; default open `0.0.0.0/0, ::/0`) |

### Allowed hosts

WebUI access is gated by two IP/CIDR allowlists, bucketed by **privilege tier** — each is the outer (host) layer; an API key is still required on the data layer.

- **`ADMIN_WEBUI_ALLOWED_HOSTS`** — admin pages (`/settings`, `/settings/api-keys`, `/docs`). Default `["127.0.0.1", "::1"]` (loopback only); data also requires an **admin** key.
- **`USER_WEBUI_ALLOWED_HOSTS`** — user pages (`/quick-config`, `/captures`, `/reports`, `/stats`, `/logs`, `/dictate`, `/sev`). Default `["0.0.0.0/0", "::/0"]` (**open**) — the per-page API key is the real gate; narrow this to restrict which networks may even reach the pages.

Loopback is *always* implicitly allowed regardless of the configured list, so a typo can never lock you out from the box itself.

```bash
# Lock the admin pages to loopback, narrow the user pages to your LAN
# (the API key still gates the data). Set via env, or edit the same
# fields in the /settings UI (→ config.local.json):
WHISPER_ADMIN_WEBUI_ALLOWED_HOSTS=127.0.0.1,::1
WHISPER_USER_WEBUI_ALLOWED_HOSTS=127.0.0.1,::1,192.168.1.0/24
```

CIDR is accepted (`192.168.0.0/16`) and so are bare IPs (`10.0.0.5`). For a dual-stack "any host" allowlist you need both `0.0.0.0/0` (IPv4) and `::/0` (IPv6).

## Endpoints

**Core API** (`/v1`, bearer API-key auth, no host allowlist — always registered):

- `POST /v1/audio/transcriptions` — OpenAI-compatible transcription. Pass `model=<name>` to pick a specific model (any faster-whisper short name or HF repo id).
- `WS   /v1/audio/transcriptions/stream` — live streaming dictation (raw 16 kHz PCM or browser WebM/Opus); see the Features section.
- `GET  /v1/models` — list currently-loaded models, the configured default, and the allowlist (if set).
- `GET  /v1/me` — the caller's effective request-override capabilities (drives client UI).
- `GET  /v1/override-profiles` (+ `/{name}`) — the override profiles this caller may request per-request; name list / single-profile preview.
- `GET/PATCH /v1/pipeline-rules` — the exposed post-processing rules this caller may view/edit (same gating + semantics as `/quick-config`, for API clients).
- `GET  /v1/recent-words` — recently-transcribed word/phrase suggestions (for rule-editor autocomplete).
- `GET  /v1/usage` — the caller's own transcription usage rollup.

**User pages** (host-gated by `USER_WEBUI_ALLOWED_HOSTS`, loopback always allowed; data endpoints additionally need an API key with the page permission):

- `GET  /logs` — live log viewer; `GET /logs/stream` (SSE feed), `GET /logs/older` (pagination).
- `GET  /stats` — system overview dashboard; `GET /stats/snapshot` + `GET /stats/stream` (JSON one-shot + ~1 Hz SSE), `GET /stats/usage` (per-user/key usage chart data).
- `GET  /quick-config` — end-user rule editor (state/recent/stream/usage/reapply-rules sub-endpoints, incl. error-report submission).
- `GET  /captures` — training-data curation UI (`/captures/api/*`: list/export, per-capture CRUD + audio, samples + merge/preview, reprocess jobs).
- `GET  /reports` — error-report triage UI (`/reports/api/*`).
- `GET  /dictate` — browser demo for the streaming endpoint.
- `GET  /sev` — tiny JSON severity counts powering the nav pills.

**Admin pages** (host-gated by `ADMIN_WEBUI_ALLOWED_HOSTS`, loopback always allowed; data endpoints require an **admin** key):

- `GET  /settings` — admin WebUI; `GET/POST /settings/state`, `POST /settings/restart`.
- `GET  /settings/pipeline` — pipeline-rule editor; `GET/POST /settings/factory-rules` (+ `/clear-local-override`), `POST /settings/test-pipeline`.
- `GET  /settings/api-keys` — per-user API key management (`/settings/api-keys/api/*`).
- `GET  /settings/overrides` — per-identity override editor (`state`, `resolve` explorer, profile rename).
- `GET  /docs`, `GET /redoc`, `GET /openapi.json` — interactive API docs (always registered; `/openapi.json` additionally requires an admin key).

`WHISPER_ADMIN_UI=0` unregisters `/settings*` **and** the WebUI pages that ride the same switch (`/quick-config`, `/captures`, `/reports`); the core API, `/logs`, `/stats`, `/dictate`, `/sev`, and `/docs` stay up.

**Auth** (user-tier host gate):

- `GET  /auth/whoami` — resolve the current credentials to `{open_mode, user_id, username, is_admin}`. The WebUI uses this to render the login modal and the OPEN-mode banner.
- `POST /auth/login`, `POST /auth/logout` — exchange an API key for a browser session cookie / end the session.

### Model selection examples

```python
# Use the configured default (Whisper-1 = OpenAI default name)
client.audio.transcriptions.create(model="whisper-1", file=f)

# Pick a specific faster-whisper short name
client.audio.transcriptions.create(model="large-v3-turbo", file=f)

# Use a German finetune from Hugging Face
client.audio.transcriptions.create(model="primeline/whisper-large-v3-turbo-german", file=f)
```

> **Note:** `ALLOWED_MODELS` ships as a curated 2-model set
> (`Systran/faster-whisper-large-v2`, `Systran/faster-whisper-large-v3`),
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

- `regex-list` — an ordered batch of find→replace entries (each one `re.sub`), edited as a single card
- `callback:lowercase-wordlist` — strip terminator and lowercase next word if it's in the wordlist
- `callback:map` — auto-built alternation of map keys (longest-first, case-insensitive); look up replacement
- `callback:dedup` — collapse adjacent punctuation runs (last non-comma wins; pure-comma run → single comma)
- `callback:upper` — capitalize after sentence terminator
- `terminal` — final `lstrip(" \t\r") + rstrip(" \t\r")`; always last (preserves leading/trailing `\n`)

The 14 seeded defaults handle orthography normalization (`ß`→`ss`), Whisper noise stripping, dictation (`Punkt`→`.`, `neue Zeile`→`\n`, …), and tidy spacing/newlines/capitalization. They live in the committed **`config.json`** (the `PIPELINE_RULES` array, next to all the scalar defaults); `config.py` loads that file at startup. Each rule carries an optional `note` field documenting its rationale.

**Ordering invariants:** `dictation-map` multi-word phrases must precede their single-word components (the alternation regex is rebuilt longest-first, so the longest phrase wins); the `terminal` trim rule is always last.

**Editing — the dedicated editor at `/settings/pipeline`** (`/settings` keeps every
other section and links there in its place). One rule list
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
- **Usage history** (`/stats/usage`): throughput over time plus a per-user / per-key leaderboard.

Sparklines are rendered with [uPlot](https://github.com/leeoniya/uPlot), vendored under `static/` so the page works **fully offline** — no CDN fetch at page-load. To update the bundled version, see `static/VENDOR.md`.

The `/stats` endpoint is user-tier allowlist-gated (`USER_WEBUI_ALLOWED_HOSTS`) plus a `stats` API key on the data endpoints. On a host without an NVIDIA GPU or with `nvidia-ml-py` missing, the GPU panel hides and the rest of the dashboard still works.

The nav row at the top of every page (logs ↔ stats ↔ quick-config ↔ captures ↔ reports ↔ settings) also surfaces three severity pills counting `WARNING` / `ERROR` / `CRITICAL` records since process start (bounded by a 2000-entry ring; restart resets to zero); clicking any pill jumps to `/logs` with that filter prefilled.

## Admin WebUI (optional)

A second WebUI at `/settings` lets you edit every setting from the browser, with hot-reload for safe knobs (transcribe params, dictation map, prompt) and an automatic service restart for cold ones (server port, log file, preload list).

**On by default** (`ADMIN_UI_ENABLED = true`). Set `WHISPER_ADMIN_UI=0` (or flip `ADMIN_UI_ENABLED` in `config.json` / `config.local.json`) and restart to unregister `/settings*` (plus `/quick-config`, `/captures`, `/reports`, which ride the same switch). The page opens at `http://localhost:8000/settings` from the server itself or any host in `ADMIN_WEBUI_ALLOWED_HOSTS`. Settings pinned by a `WHISPER_*` env var appear **read-only** (greyed out, badged with the variable name) since the environment takes precedence.

### Authentication: per-user API keys

The transcription endpoint and every WebUI page are gated by **per-user API keys**, not a shared token. Each key looks like `wk_<43-char base64>` (256-bit entropy); raw keys are SHA-256-hashed at rest and shown **once** on creation.

**Bootstrap.** On a fresh install with no admin key in the DB, the server starts in **OPEN mode**: every request is accepted as a synthetic admin, a red banner appears on every WebUI page, and a `WARNING` log line fires every 60 s. This is the operator's prompt to generate the first admin key. Two ways:

1. **In the UI** — open `/settings/api-keys`, click "+ add user" with admin=true, then "+ generate key", and copy the raw key from the show-once modal.
2. **Via env var** — set `WHISPER_BOOTSTRAP_ADMIN_KEY=wk_…` on first start. A `bootstrap-admin` user is created (or skipped if the same key hash is already present) with that exact raw key. Subsequent starts no-op.

Once at least one active admin key exists, the OPEN-mode banner disappears and 401 is returned to unauthenticated callers.

**Using a key.** API clients and curl send `Authorization: Bearer wk_…` on every request. The WebUI instead exchanges the key **once** for an HttpOnly session cookie via `POST /auth/login` (server-side session rows, 30-day TTL by default — `SESSION_TTL_SECONDS`; CSRF double-submit cookie on mutating requests). On any 401 the full-page login gate re-prompts; `POST /auth/logout` ends the session.

**Lockout protection.** Revoking the last active admin key (or the last admin user) returns 409. Generate a second admin key first.

**Multi-user.** Each capture is tagged with the originating `user_id`. Non-admin users see only their own captures in `/captures`; admins see all and can filter by user. Merging captures into a training sample is locked to a single speaker — the server rejects any merge whose members span more than one user.

Other layers:
- **Feature flag**: `ADMIN_UI_ENABLED` (on by default; pin with `WHISPER_ADMIN_UI=1`) registers the routes. Set `WHISPER_ADMIN_UI=0` and `/settings*` returns 404.
- **Host allowlist**: `ADMIN_WEBUI_ALLOWED_HOSTS` keeps the admin endpoints reachable only from the configured CIDRs (loopback always implicit). User pages use the separate `USER_WEBUI_ALLOWED_HOSTS` (default open).
- **Server-side validation**: every payload is validated against `config_store.AdminConfig` (Pydantic v2).
- **Auto-restart**: when a "cold" setting changes (server port, log file, preload list, …), a confirmation modal asks whether to restart the service. WinSW relaunches the wrapper; the page polls `/v1/models` until back up.

Edits land in **`config.local.json`** at the repo root (gitignored). See `config.local.example.json` for the schema. The one exception is the Pipeline section's **promote** action, which writes the committed **`config.json`** instead (see [Post-processing pipeline](#post-processing-pipeline)).

### Per-identity config overrides

Beyond the global and per-model layers, decode / streaming / output / pipeline-rule settings can be overridden **per user, per API key, and per reusable profile** — so many users can share one deployment without re-flashing every device. Managed on the dedicated **`/settings/overrides`** page; bound to users & keys in-context on **`/settings/api-keys`** (a `⚙ overrides` / `⚙ config` drawer per user / key). Load-time model fields (device, compute type, workers…) are **never** per-identity — a model is loaded once for everyone.

- **Profiles** — named, reusable override bundles (e.g. `low-latency`). Assign an *ordered* list to a user or key; **earlier wins** on a conflicting field.
- **Direct overrides** — a per-user or per-key blob layered on top of its profiles for one-offs.
- **Precedence** (most → least specific): `per-key direct → per-key profiles → per-user direct → per-user profiles → per-model → global → library`. The first identity layer that sets a field owns its value **and** its lock.
- **Per-field locking** — mark a field 🔒 to forbid the client's per-request `decode_overrides` (and `language`/`prompt`) from changing it; the dropped keys are surfaced in `verbose_json.overrides_ignored` (batch) / the `ready` frame (streaming), never silently ignored. A useful compute cap on a shared server (e.g. lock `BEAM_SIZE`).
- **Effective-config Explorer** — the `/settings/overrides` *Explorer* tab is a what-if simulator: pick a user (+ key, + model, + a simulated client override) and see the full resolution **waterfall** per field — which layer won, what was overridden, what is locked.
- **Pipeline rules** resolve analogously (first layer that force-on/off a rule decides; otherwise per-model, then the global `enabled`). Capture **reprocess** re-runs the pipeline under the capture **owner's** rules.
- **Live changes apply without reconnect** — batch requests resolve identity per request; a live dictation **WebSocket** re-resolves at each utterance boundary whenever the config version changes (any binding / profile / settings edit), so edits land on the next utterance. Session-shaping `STREAMING_*` (chunking / VAD / endpointer timing) and word-timestamp gating stay fixed for the connection — change those, then reconnect. Every batch & stream log block carries an **`Identity`** section (resolved user / key + applied profiles, or `overrides (none — inherits …)`) so a missing binding is obvious at a glance.

The same `OVERRIDE_PROFILES` JSON can be pinned via `WHISPER_OVERRIDE_PROFILES` (see `.env.example`). Per-user bindings live in the user's permissions JSON; per-key bindings in the `api_keys.config` column (added by an idempotent migration on first start).

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
config.py                  Loads config.json factory defaults, layers config.local.json + WHISPER_* env on top
config.json                Committed factory defaults for EVERY setting + the pipeline rules (single source of truth)
config_store.py            Admin-WebUI persistence layer (Pydantic schema, atomic writes)
effective_config.py        Layered per-identity config resolution (key/user/profile overrides, locks)
admin_routes.py            Admin /settings + /settings/pipeline pages & endpoints (disable with WHISPER_ADMIN_UI=0)
overrides_routes.py        /settings/overrides admin page & API (profiles, explorer/resolve)
api_keys_store.py          users + api_keys SQLite store (SHA-256 hash, soft revoke, O(1) lookup)
api_keys_routes.py         /settings/api-keys admin UI for per-user key management
sessions_store.py          Durable browser-session store — the cookie layer on top of api_keys_store
auth.py                    Auth deps — get_current_user / require_admin / require_page + OPEN-mode loop
quick_config_routes.py     /quick-config end-user rule editor + the /v1/pipeline-rules client API
quick_config_state.py      Tokenization + SSE broadcast layer for /quick-config recent transcriptions
stats_routes.py            /stats dashboard endpoints + HTML page (always on, allowlist-gated)
metrics.py                 In-process request metrics (counters, latency ring, recent transcriptions)
system_stats.py            GPU + host snapshot (pynvml + psutil; degrades gracefully if NVML missing)
usage_store.py             Durable per-key / per-user usage rollup (/v1/usage, /stats/usage)
transcriptions_store.py    Durable store for recent transcription traces (/quick-config recent)
web_common.py              Shared helpers: allowlist gate, nav HTML + severity pills, login gate / OPEN-mode banner
restart_service.py         Detached self-restart helper (os.execv re-exec on Linux/macOS, WinSW on Windows)
streaming_routes.py        WebSocket /v1/audio/transcriptions/stream + /dictate demo page
streaming_session.py       Per-connection streaming dictation state machine
streaming_transport.py     Streaming audio decoders (raw PCM passthrough, ffmpeg WebM/Opus)
streaming_vad.py           Streaming endpointing (two-tier Silero/energy VAD)
streaming_localagreement.py LocalAgreement-2 hypothesis stabilization
audio_transcode.py         In-process audio transcoder (PyAV — no ffmpeg-on-PATH needed)
audio_vad_trim.py          Silence-trim WAVs with the bundled Silero VAD
audio_merge.py             stdlib-wave PCM splicer for ≤28 s training-sample packing
captures_store.py          Capture rows + audio fanout, retention, eviction
capture_samples_store.py   Packed ≤28 s training samples built from consecutive same-speaker captures
captures_routes.py         /captures page + samples/merge/reprocess API
captures_merge_proposer.py Auto-merge proposer for /captures curation
captures_reapply.py        Background job: re-run current pipeline rules over existing captures
captures_vad_reprocess.py  Background job: re-merge sample audio with current silence settings
reports_store.py / reports_routes.py
                           User-submitted transcription error reports + admin triage
regex_guard.py             Out-of-process guard for user-authored pipeline regexes
text_corrections.py        Shared schema for word-correction chips
config.local.json          Runtime overrides written by the admin UI (gitignored, optional)
config.local.example.json  Example overrides file
.env.example               Documented list of every WHISPER_* env var + defaults (copy to .env)
test.py                    Manual test client (OpenAI SDK compatibility)
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
.github/workflows/ci.yml   CI: test suite on Linux + Windows, then publishes the GHCR images
static/                    Brand assets (logo.svg, favicon.*) + vendored uPlot/GridStack (offline /stats)
.gitignore / .gitattributes
logs/                      Created at first run; rotates at 10 MB × 10 files
```
