"""
Persistence layer for the admin WebUI.

Stores user-edited overrides in <repo>/config.local.json. The file is loaded
by config.py BETWEEN the in-file defaults and the env-var override block, so
precedence stays:  ENV  >  config.local.json  >  config.py defaults.

Validation uses Pydantic v2. Every field is Optional — missing means "use the
config.py default". `model_config = {"extra": "forbid"}` rejects unknown keys
so typos and probing surface as 422 errors instead of silent no-ops.

Atomic writes: tmp file in the same directory, then os.replace. Retry loop
covers Windows sharing-violations from AV scanners briefly holding the file.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import sys
import tempfile
import time
from pathlib import PurePath, PureWindowsPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
OVERRIDES_PATH = os.path.join(_REPO_DIR, "config.local.json")
# Committed factory-default pipeline rules. Unlike config.local.json this file
# IS version-controlled — the admin WebUI's "Defaults" mode edits it so rule
# fixes can be git-pushed to every deployment. See load_factory_rules().
FACTORY_PATH = os.path.join(_REPO_DIR, "config.json")


# Map AdminConfig field name -> env var that pins it. Mirrors the override
# block at the bottom of config.py. Used by the WebUI to mark fields as
# "currently overridden by WHISPER_X" with a badge.
ENV_VAR_MAPPING: dict[str, str] = {
    "DEFAULT_MODEL": "WHISPER_DEFAULT_MODEL",
    "ALLOWED_MODELS": "WHISPER_ALLOWED_MODELS",
    "MAX_LOADED_MODELS": "WHISPER_MAX_LOADED_MODELS",
    "MODEL_IDLE_TIMEOUT_S": "WHISPER_MODEL_IDLE_TIMEOUT_S",
    "PRELOAD_MODELS": "WHISPER_PRELOAD_MODELS",
    "DEFAULT_PROMPT": "WHISPER_DEFAULT_PROMPT",
    "DEFAULT_HOTWORDS": "WHISPER_DEFAULT_HOTWORDS",
    "OUTPUT_PREFIX": "WHISPER_OUTPUT_PREFIX",
    "OUTPUT_SUFFIX": "WHISPER_OUTPUT_SUFFIX",
    "TEMPERATURE": "WHISPER_TEMPERATURE",
    "PATIENCE": "WHISPER_PATIENCE",
    "LENGTH_PENALTY": "WHISPER_LENGTH_PENALTY",
    "REPETITION_PENALTY": "WHISPER_REPETITION_PENALTY",
    "NO_REPEAT_NGRAM_SIZE": "WHISPER_NO_REPEAT_NGRAM_SIZE",
    "PROMPT_RESET_ON_TEMPERATURE": "WHISPER_PROMPT_RESET_ON_TEMPERATURE",
    "MULTILINGUAL": "WHISPER_MULTILINGUAL",
    "LANGUAGE_DETECTION_THRESHOLD": "WHISPER_LANGUAGE_DETECTION_THRESHOLD",
    "LANGUAGE_DETECTION_SEGMENTS": "WHISPER_LANGUAGE_DETECTION_SEGMENTS",
    "HALLUCINATION_SILENCE_THRESHOLD": "WHISPER_HALLUCINATION_SILENCE_THRESHOLD",
    "SUPPRESS_BLANK": "WHISPER_SUPPRESS_BLANK",
    "SUPPRESS_TOKENS": "WHISPER_SUPPRESS_TOKENS",
    "SUPPRESS_CHARS": "WHISPER_SUPPRESS_CHARS",
    "PREPEND_PUNCTUATIONS": "WHISPER_PREPEND_PUNCTUATIONS",
    "APPEND_PUNCTUATIONS": "WHISPER_APPEND_PUNCTUATIONS",
    "DOWNLOAD_ROOT": "WHISPER_DOWNLOAD_ROOT",
    "LOCAL_FILES_ONLY": "WHISPER_LOCAL_FILES_ONLY",
    "USE_AUTH_TOKEN": "WHISPER_USE_AUTH_TOKEN",
    "AUTO_CONVERT_HF_MODELS": "WHISPER_AUTO_CONVERT_HF_MODELS",
    "CONVERT_QUANTIZATION": "WHISPER_CONVERT_QUANTIZATION",
    "CONVERTED_MODELS_DIR": "WHISPER_CONVERTED_MODELS_DIR",
    "CPU_THREADS": "WHISPER_CPU_THREADS",
    "NUM_WORKERS": "WHISPER_NUM_WORKERS",
    "DEVICE_INDEX": "WHISPER_DEVICE_INDEX",
    "TRACE_ENABLED": "WHISPER_TRACE",
    "LOG_FILE": "WHISPER_LOG_FILE",
    "LOG_VIEWER_INITIAL_LINES": "WHISPER_LOG_VIEWER_INITIAL_LINES",
    "LOG_VIEWER_DOM_MAX": "WHISPER_LOG_VIEWER_DOM_MAX",
    "ADMIN_ALLOWED_HOSTS": "WHISPER_ADMIN_ALLOWED_HOSTS",
    "STATS_ALLOWED_HOSTS": "WHISPER_STATS_ALLOWED_HOSTS",
    # Browser session cookies (HttpOnly cookie auth for the WebUI). All hot.
    "SESSION_COOKIE_SECURE": "WHISPER_SESSION_COOKIE_SECURE",
    "SESSION_TTL_SECONDS": "WHISPER_SESSION_TTL_SECONDS",
    "SESSION_COOKIE_NAME": "WHISPER_SESSION_COOKIE_NAME",
    "SESSION_CSRF_COOKIE_NAME": "WHISPER_SESSION_CSRF_COOKIE_NAME",
    # Reports + captures fields — also surfaced in the AdminConfig schema and
    # also env-readable in config.py. Without these the WebUI silently
    # succeeds on saves whose env-set values will revert on restart.
    "REPORTS_DB": "WHISPER_REPORTS_DB",
    "REPORTS_MAX": "WHISPER_REPORTS_MAX",
    "REPORTS_RETENTION_DAYS": "WHISPER_REPORTS_RETENTION_DAYS",
    "REPORTS_ALLOW_USER_SUBMIT": "WHISPER_REPORTS_ALLOW_USER_SUBMIT",
    "RECENT_TRANSCRIPTIONS_DB": "WHISPER_RECENT_TRANSCRIPTIONS_DB",
    "RECENT_TRANSCRIPTIONS_MAX": "WHISPER_RECENT_TRANSCRIPTIONS_MAX",
    "RECENT_TRANSCRIPTIONS_TTL_DAYS": "WHISPER_RECENT_TRANSCRIPTIONS_TTL_DAYS",
    "RECENT_TRANSCRIPTIONS_PAGE_SIZE": "WHISPER_RECENT_TRANSCRIPTIONS_PAGE_SIZE",
    "RECENT_TRANSCRIPTIONS_PRUNE_EVERY": "WHISPER_RECENT_TRANSCRIPTIONS_PRUNE_EVERY",
    "STATS_RECENT_TX_DISPLAY": "WHISPER_STATS_RECENT_TX_DISPLAY",
    "CAPTURE_RECORDINGS_ENABLED": "WHISPER_CAPTURE_RECORDINGS_ENABLED",
    "CAPTURES_DB": "WHISPER_CAPTURES_DB",
    "CAPTURES_DIR": "WHISPER_CAPTURES_DIR",
    "CAPTURES_MAX": "WHISPER_CAPTURES_MAX",
    "CAPTURES_MAX_MB": "WHISPER_CAPTURES_MAX_MB",
    "CAPTURES_RETENTION_DAYS": "WHISPER_CAPTURES_RETENTION_DAYS",
    "CAPTURE_RECORDINGS_SAMPLE_RATE": "WHISPER_CAPTURE_RECORDINGS_SAMPLE_RATE",
    "CAPTURE_RECORDINGS_MIN_DURATION_SEC": "WHISPER_CAPTURE_RECORDINGS_MIN_DURATION_SEC",
    "CAPTURE_RECORDINGS_MAX_DURATION_SEC": "WHISPER_CAPTURE_RECORDINGS_MAX_DURATION_SEC",
    "CAPTURE_RECORDINGS_AUDIO_BYTES_HARD_LIMIT": "WHISPER_CAPTURE_RECORDINGS_AUDIO_BYTES_HARD_LIMIT",
    # Model device / compute (load-time)
    "MODEL_DEVICE": "WHISPER_MODEL_DEVICE",
    "MODEL_COMPUTE_TYPE": "WHISPER_MODEL_COMPUTE_TYPE",
    "MODEL_DEVICE_FALLBACK": "WHISPER_MODEL_DEVICE_FALLBACK",
    "MODEL_COMPUTE_TYPE_FALLBACK": "WHISPER_MODEL_COMPUTE_TYPE_FALLBACK",
    "DEFAULT_LANGUAGE": "WHISPER_DEFAULT_LANGUAGE",
    # Decode quality / VAD
    "BEAM_SIZE": "WHISPER_BEAM_SIZE",
    "BEST_OF": "WHISPER_BEST_OF",
    "VAD_FILTER": "WHISPER_VAD_FILTER",
    "VAD_MIN_SILENCE_MS": "WHISPER_VAD_MIN_SILENCE_MS",
    "VAD_SPEECH_PAD_MS": "WHISPER_VAD_SPEECH_PAD_MS",
    "VAD_THRESHOLD": "WHISPER_VAD_THRESHOLD",
    "CONDITION_ON_PREVIOUS_TEXT": "WHISPER_CONDITION_ON_PREVIOUS_TEXT",
    "WORD_TIMESTAMPS_ENABLED": "WHISPER_WORD_TIMESTAMPS_ENABLED",
    "NO_SPEECH_THRESHOLD": "WHISPER_NO_SPEECH_THRESHOLD",
    "LOG_PROB_THRESHOLD": "WHISPER_LOG_PROB_THRESHOLD",
    "COMPRESSION_RATIO_THRESHOLD": "WHISPER_COMPRESSION_RATIO_THRESHOLD",
    # Log rotation (restart-required)
    "LOG_MAX_BYTES": "WHISPER_LOG_MAX_BYTES",
    "LOG_BACKUP_COUNT": "WHISPER_LOG_BACKUP_COUNT",
    # Server binding (restart-required). Note: changing the port in Docker also
    # requires updating the compose `ports:` mapping.
    "SERVER_HOST": "WHISPER_SERVER_HOST",
    "SERVER_PORT": "WHISPER_SERVER_PORT",
    "SERVER_WORKERS": "WHISPER_SERVER_WORKERS",
    "SERVER_LOG_LEVEL": "WHISPER_SERVER_LOG_LEVEL",
    # Captures: pipeline exclude + VAD trim
    "CAPTURES_PIPELINE_RULES_EXCLUDE": "WHISPER_CAPTURES_PIPELINE_RULES_EXCLUDE",
    "CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS": "WHISPER_CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS",
    "CAPTURES_VAD_MARGIN_GROUP_EDGE_MS": "WHISPER_CAPTURES_VAD_MARGIN_GROUP_EDGE_MS",
    "CAPTURES_VAD_MARGIN_GROUP_INTERNAL_MS": "WHISPER_CAPTURES_VAD_MARGIN_GROUP_INTERNAL_MS",
    "CAPTURES_SAMPLE_MIN_DURATION_S": "WHISPER_CAPTURES_SAMPLE_MIN_DURATION_S",
    "CAPTURES_SAMPLE_MAX_DURATION_S": "WHISPER_CAPTURES_SAMPLE_MAX_DURATION_S",
    "CAPTURES_SAMPLE_JOIN_STRATEGY": "WHISPER_CAPTURES_SAMPLE_JOIN_STRATEGY",
    "CAPTURES_PROPOSER_TARGET_S": "WHISPER_CAPTURES_PROPOSER_TARGET_S",
    "CAPTURES_PROPOSER_SESSION_GAP_S": "WHISPER_CAPTURES_PROPOSER_SESSION_GAP_S",
    "CAPTURES_PROPOSER_DUP_THRESHOLD": "WHISPER_CAPTURES_PROPOSER_DUP_THRESHOLD",
    "CAPTURES_PROPOSER_MAX_PROPOSALS": "WHISPER_CAPTURES_PROPOSER_MAX_PROPOSALS",
    # Structured fields — supplied as a JSON string (config.py parses+validates).
    # The per-model WHISPER_MODEL_OVERRIDE__<id>__<FIELD> convention still works
    # and merges on top of WHISPER_MODEL_OVERRIDES.
    "PIPELINE_RULES": "WHISPER_PIPELINE_RULES",
    "MODEL_OVERRIDES": "WHISPER_MODEL_OVERRIDES",
}

# Cold settings — editing these requires a service restart for the new value
# to take effect. The WebUI shows a 'restart' badge and offers to trigger a
# self-restart after save. Note: MODEL_DEVICE / MODEL_COMPUTE_TYPE were
# previously listed here but are now hot — admin save triggers drain-then-
# evict on the affected loaded models so they reload with the new values.
RESTART_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "SERVER_HOST", "SERVER_PORT", "SERVER_WORKERS", "SERVER_LOG_LEVEL",
    "LOG_FILE", "LOG_MAX_BYTES", "LOG_BACKUP_COUNT",
    "PRELOAD_MODELS",
})

# Load-time fields. Editing these (globally OR per-model in MODEL_OVERRIDES)
# triggers drain-then-evict on the affected loaded models so the next request
# reloads them with the new values. These are read at WhisperModel(...)
# construction time; changes only take effect after re-load.
LOAD_TIME_FIELDS: frozenset[str] = frozenset({
    "MODEL_DEVICE", "MODEL_COMPUTE_TYPE",
    "MODEL_DEVICE_FALLBACK", "MODEL_COMPUTE_TYPE_FALLBACK",
    "REVISION", "NUM_WORKERS", "DEVICE_INDEX",
    "DOWNLOAD_ROOT", "LOCAL_FILES_ONLY", "USE_AUTH_TOKEN", "CPU_THREADS",
    "AUTO_CONVERT_HF_MODELS", "CONVERT_QUANTIZATION", "CONVERTED_MODELS_DIR",
})

# Hot settings whose derived caches need rebuild after edit. The admin route
# calls main.rebuild_caches() when any of these change.
CACHE_REBUILD_FIELDS: frozenset[str] = frozenset({"PIPELINE_RULES", "SUPPRESS_CHARS"})


# =============================================================================
# Single source of truth for field descriptions
# =============================================================================
# Surfaced everywhere a description is shown:
#   - Pydantic Field(description=…) — see _F() helper below
#   - /settings/state payload — admin_routes.py adds .description from the
#     Pydantic model_fields
#   - /settings admin WebUI — fieldRow() renders it as a <div class="help">
#     line under each editor
# Edit a string here, every consumer reflects it on next reload. Wording is
# cross-validated against upstream docs (faster-whisper, OpenAI Whisper,
# Silero VAD, CTranslate2, uvicorn, Python logging) where authoritative.
FIELD_DESCRIPTIONS: dict[str, str] = {
    # --- Models ---
    "DEFAULT_MODEL":
        "Model loaded when a request sends 'whisper-1' or omits 'model'. "
        "Accepts any faster-whisper short name or HF repo id.",
    "ALLOWED_MODELS":
        "Allowlist of model names clients may request. Empty set lets any name "
        "pass — risks unknown multi-GB downloads.",
    "MAX_LOADED_MODELS":
        "Max models kept hot in VRAM (LRU evicts beyond this). large-v3 "
        "~1.5 GB fp16, turbo/distill ~600 MB.",
    "MODEL_IDLE_TIMEOUT_S":
        "Unload a model from VRAM/RAM after this many seconds without use. "
        "0 = disabled (default). Examples: 1800 = 30 min, 3600 = 1 h, "
        "14400 = 4 h. Background task wakes every 30 s to check.",
    "PRELOAD_MODELS":
        "Models eagerly loaded at startup so the first request skips the "
        "5-30 s warm-up. Empty = only DEFAULT_MODEL.",
    "MODEL_DEVICE":
        "Device to use for computation. cuda = NVIDIA GPU; cpu = host CPU. "
        "(faster-whisper)",
    "MODEL_COMPUTE_TYPE":
        "Numerical precision (CTranslate2).\n"
        "• float16: half-precision, weights + layers in FP16 "
        "(NVIDIA Volta+ / CUDA Compute Capability ≥ 7.0).\n"
        "• bfloat16: brain-float, half-precision "
        "(NVIDIA Ampere+ / CUDA Compute Capability ≥ 8.0).\n"
        "• int8: 8-bit weight quantization (smallest, fastest on CPU).\n"
        "• int8_float16: int8 weights + FP16 activations (smallest GPU footprint).\n"
        "• float32: full precision, largest + slowest.",
    "MODEL_DEVICE_FALLBACK":
        "Backup hardware target if the primary device fails to load "
        "(e.g. fall back to 'cpu' if CUDA is unavailable).",
    "MODEL_COMPUTE_TYPE_FALLBACK":
        "Backup precision used when the primary compute type isn't supported "
        "on the fallback device.",

    # --- Decode params (transcribe-time) ---
    "DEFAULT_LANGUAGE":
        "Language code such as 'en' or 'de'. If empty, the language is "
        "detected in the first 30 seconds of audio. (faster-whisper)",
    "DEFAULT_PROMPT":
        "Optional text passed as initial_prompt for the first window — "
        "useful for custom vocabularies or proper nouns to make those "
        "words more likely to be predicted. (OpenAI Whisper)",
    "BEAM_SIZE":
        "Beam size to use for decoding. Higher = better quality but slower. "
        "faster-whisper default 5.",
    "BEST_OF":
        "Number of independent sample trajectories when temperature > 0 "
        "(only on fallback-retry passes; the initial T=0 pass uses beam "
        "search instead). faster-whisper default 5.",
    "VAD_FILTER":
        "Enable voice activity detection (VAD) to filter out parts of the "
        "audio without speech. Uses the Silero VAD model. Reduces "
        "hallucinations in quiet audio.",
    "VAD_MIN_SILENCE_MS":
        "In the end of each speech chunk, wait this long before separating "
        "it. Default 2000 ms (faster-whisper override; tuned to avoid "
        "splitting on short breaths).",
    "VAD_SPEECH_PAD_MS":
        "Final speech chunks are padded by this much on each side. "
        "Default 400 ms (faster-whisper override; prevents word-edge "
        "consonants from being clipped).",
    "VAD_THRESHOLD":
        "Speech threshold. Silero VAD outputs speech probabilities for "
        "each audio chunk; probabilities ABOVE this value are considered "
        "as SPEECH. Default 0.5; tune per dataset if needed.",
    "CONDITION_ON_PREVIOUS_TEXT":
        "If true, the previous output of the model is provided as a "
        "prompt for the next window. Disabling may make the text "
        "inconsistent across windows, but the model becomes less prone "
        "to getting stuck in failure loops (repetition; timestamps "
        "drifting out of sync). Default true.",
    "WORD_TIMESTAMPS_ENABLED":
        "Gate (not toggle): word timestamps run only when this is true AND "
        "the request asks (`timestamp_granularities[]=word` or "
        "`response_format=verbose_json`). False = force-off, even if asked.",
    "NO_SPEECH_THRESHOLD":
        "If the no_speech probability is higher than this value AND the "
        "average log-probability over sampled tokens is below "
        "LOG_PROB_THRESHOLD, the segment is dropped as silent. Default 0.6.",
    "LOG_PROB_THRESHOLD":
        "If the average log-probability over sampled tokens is below "
        "this value, treat the decode as failed (triggers a temperature-"
        "fallback retry). Default -1.0.",
    "COMPRESSION_RATIO_THRESHOLD":
        "If the gzip-compression ratio (zlib in the implementation) of "
        "the decoded text is above this value, treat the decode as "
        "failed (triggers a temperature-fallback retry). Catches "
        "repetition loops. Default 2.4.",
    "DEFAULT_HOTWORDS":
        "Persistent vocabulary biasing — re-injected into the prompt of "
        "every decoder window. Distinct from DEFAULT_PROMPT (which fades "
        "as decoded text accumulates): hotwords stay constant. Useful "
        "for domain terms, drug names, person names. Ignored when prefix "
        "is set per-call. (faster-whisper)",

    # --- Decode params (advanced) ---
    "TEMPERATURE":
        "Fallback ladder for decoding when compression / log-prob checks "
        "fail. Comma-separated floats (e.g. '0.0,0.2,0.4,0.6,0.8,1.0'). "
        "Lower / shorter ladders fail faster on distil models. (faster-whisper)",
    "PATIENCE":
        "Beam-search patience factor; >1 keeps the beam alive longer. "
        "Default 1.0. Try 1.5 if long sentences get clipped.",
    "LENGTH_PENALTY":
        "Beam-scoring length-norm exponent. >1 favors longer outputs, "
        "<1 favors shorter. Default 1.0. Tweak only if outputs are "
        "systematically too short or too padded.",
    "REPETITION_PENALTY":
        "Multiplies logit of already-emitted tokens by 1/penalty. >1 "
        "discourages loops. Default 1.0. Try 1.05–1.2 for stutter audio.",
    "NO_REPEAT_NGRAM_SIZE":
        "Hard ban on n-grams of this size repeating. 0 = off. Try 3 "
        "for stubborn repetition loops. Caveat: blocks legitimate "
        "repeats too.",
    "PROMPT_RESET_ON_TEMPERATURE":
        "When the temperature ladder fallback exceeds this value, drop "
        "the running text prompt to escape bad context. Default 0.5. "
        "Only relevant when CONDITION_ON_PREVIOUS_TEXT=True.",

    # --- Language detection (active when DEFAULT_LANGUAGE is empty) ---
    "MULTILINGUAL":
        "Re-run language detection on every segment instead of once. "
        "Default false. Enable for code-switching audio. (faster-whisper)",
    "LANGUAGE_DETECTION_THRESHOLD":
        "Min probability the top language token must reach for detection "
        "to be accepted. Default 0.5. Raise for stricter detection.",
    "LANGUAGE_DETECTION_SEGMENTS":
        "How many leading 30 s chunks to sample for language detection. "
        "Default 1. Bump to 2-5 if files start with silence/music.",

    # --- Anti-hallucination & token control ---
    "HALLUCINATION_SILENCE_THRESHOLD":
        "With WORD_TIMESTAMPS_ENABLED=true, skip silent stretches longer "
        "than this many seconds when a possible hallucination is detected. "
        "Default disabled. Try 2.0 if Whisper invents 'thanks for watching' "
        "filler in long silences.",
    "SUPPRESS_BLANK":
        "Suppress blank token at start of decoder sampling. Default true. "
        "Almost never disable; only useful when debugging tokenizer behavior.",
    "SUPPRESS_TOKENS":
        "Comma-separated token IDs to ban from output. '-1' = expand to "
        "the model's default non-speech symbol set; '' = no suppression. "
        "Token IDs vary by tokenizer.",
    "SUPPRESS_CHARS":
        "Single chars to hard-mask during decoding. Each char is encoded "
        "via the loaded model's tokenizer (both bare and ' char' variants); "
        "single-token results are added to the effective suppress_tokens "
        "list — the decoder cannot emit them. Use '.,?!:;' for verbatim "
        "dictation: model can't auto-insert punctuation, so spoken 'Punkt' / "
        "'Komma' surface as words for the dictation-map PIPELINE_RULE to "
        "convert. Empty / unset = no extra suppression. Per-model overridable.",
    "PREPEND_PUNCTUATIONS":
        "With WORD_TIMESTAMPS_ENABLED, glue these characters onto the "
        "FOLLOWING word's timing. Locale-specific.",
    "APPEND_PUNCTUATIONS":
        "With WORD_TIMESTAMPS_ENABLED, glue these characters onto the "
        "PRECEDING word's timing. Add ؟ ، for Arabic, etc.",

    # --- Output wrappers ---
    "OUTPUT_PREFIX":
        "Plain text prepended to the final transcript text after the "
        "post-processing pipeline runs (before final whitespace trim). "
        "Empty / unset = no prefix. NOT a faster-whisper param.",
    "OUTPUT_SUFFIX":
        "Plain text appended to the final transcript text after the "
        "post-processing pipeline runs (before final whitespace trim). "
        "Empty / unset = no suffix. NOT a faster-whisper param.",

    # --- Load-time, hardware (advanced) ---
    "DOWNLOAD_ROOT":
        "Directory where HuggingFace model snapshots are cached. Empty = "
        "standard HF cache dir (~/.cache/huggingface).",
    "LOCAL_FILES_ONLY":
        "If true, never hit the network — only resolve from local cache. "
        "Default false. Use for air-gapped deploys.",
    "USE_AUTH_TOKEN":
        "HuggingFace auth token for gated/private repos. Account-scoped.",
    "AUTO_CONVERT_HF_MODELS":
        "Auto-convert HuggingFace transformers Whisper models to CTranslate2 "
        "format on first load when no `model.bin` is present in the repo. "
        "Requires `pip install -r requirements-convert.txt` (transformers + "
        "torch + accelerate, ~2 GB). Output cached under CONVERTED_MODELS_DIR "
        "and loaded directly on subsequent starts. Conversion takes 1–3 min "
        "per model; happens during PRELOAD_MODELS startup or the first "
        "request that hits an unconverted model.",
    "CONVERT_QUANTIZATION":
        "On-disk weight quantisation when auto-converting HF→CT2. The "
        "saved precision is independent of MODEL_COMPUTE_TYPE (CT2 up- or "
        "down-casts at load). float16 = sweet spot for HF Whisper finetunes "
        "(matches source dtype, ~1.6 GB for large-v3-turbo). int8_float16 "
        "halves disk for marginal accuracy loss. Allowed: float32, float16, "
        "bfloat16, int16, int8, int8_float32, int8_float16, int8_bfloat16.",
    "CONVERTED_MODELS_DIR":
        "Output root for auto-converted CT2 models. Layout: "
        "<root>/<sanitized-id>/<quantization>/. Empty = ~/.cache/whisper-ct2.",
    "CPU_THREADS":
        "CPU threads for inference. 0 = library default (typically 4). "
        "Non-zero overrides OMP_NUM_THREADS for the worker pool.",
    "NUM_WORKERS":
        "Replicates the model so concurrent transcribe() calls run in true "
        "parallel. Default 1. Costs ~Nx VRAM for activation buffers.",
    "DEVICE_INDEX":
        "GPU index to bind to. Default 0. Set per-model on multi-GPU boxes "
        "to pin a model to a specific card.",

    # --- Pipeline ---
    "PIPELINE_RULES":
        "Ordered text-cleanup rules applied to the joined transcript. Each "
        "row is a regex or named-callback rule; drag to reorder, edit, "
        "disable, or add custom rules. Reset to defaults if anything breaks. "
        "The final 'trim edges' row always runs last.",
    "PIPELINE_RULES_EXCLUDE":
        "(Per-model only) List of pipeline rule slugs to FORCE-DISABLE when "
        "this model is serving the request — even if the rule is enabled "
        "globally. Use to drop e.g. 'dictation-map' for German fine-tunes "
        "that already emit punctuation symbols.",
    "PIPELINE_RULES_INCLUDE":
        "(Per-model only) List of pipeline rule slugs to FORCE-ENABLE when "
        "this model is serving the request — even if the rule is disabled "
        "globally. Inverse of PIPELINE_RULES_EXCLUDE; a slug cannot appear "
        "in both lists at once.",
    "MODEL_OVERRIDES":
        "Per-model override bundle. Maps model id → override dict. Each "
        "override may set any of the per-model-overrideable fields; "
        "absent fields inherit the global default. Edited via the per-"
        "model pane of the admin UI.",
    "REVISION":
        "(Per-model only) HuggingFace git revision (branch, tag, commit) "
        "to pin the model snapshot to. Empty = HEAD of default branch.",

    # --- Logging ---
    "TRACE_ENABLED":
        "Emit a multi-line trace block per transcription request. Disable "
        "on busy servers to control log volume.",

    "LOG_FILE":
        "Path to the rotating log file. Parent directory is auto-created "
        "at startup if missing.",
    "LOG_MAX_BYTES":
        "Rotate the log file when it reaches this size in bytes. "
        "0 disables rotation.",
    "LOG_BACKUP_COUNT":
        "Number of rotated log files to retain (.1, .2, …). Older files "
        "are deleted. 0 disables rotation.",
    "LOG_VIEWER_INITIAL_LINES":
        "Backlog lines streamed to the /logs page on connect. When the "
        "active log has fewer lines than this (e.g. right after rotation) "
        "the viewer spills into the rotated chain (.1, .2, …) to fill the "
        "backlog. Raise on chatty TRACE_ENABLED deployments where the "
        "default leaves only a handful of requests visible.",
    "LOG_VIEWER_DOM_MAX":
        "Max number of log lines retained in the browser DOM during live "
        "tail. 0 = auto (= LOG_VIEWER_INITIAL_LINES × 4). The cap applies "
        "only to live-tail appends — \"Load older\" pagination is allowed "
        "to grow the DOM beyond it.",

    # --- Server (uvicorn) ---
    "SERVER_HOST":
        "uvicorn bind address. 0.0.0.0 = listen on all interfaces "
        "(LAN-reachable); 127.0.0.1 = loopback only.",
    "SERVER_PORT":
        "uvicorn TCP port to bind. Default 8000.",
    "SERVER_WORKERS":
        "uvicorn worker processes. Keep at 1 — each worker reloads models "
        "into VRAM and multiplies GPU memory.",
    "SERVER_LOG_LEVEL":
        "uvicorn log verbosity: critical | error | warning | info | debug.",

    # --- Access (allowlists) ---
    "ADMIN_ALLOWED_HOSTS":
        "IP/CIDR allowlist for /settings admin endpoints. Loopback "
        "(127.0.0.1, ::1) is always implicitly allowed.",
    "STATS_ALLOWED_HOSTS":
        "IP/CIDR allowlist for /stats endpoints. Loopback always allowed; "
        "default is loopback only.",
    # --- Browser sessions ---
    "SESSION_COOKIE_SECURE":
        "Mark the WebUI session/CSRF cookies 'Secure' (sent only over HTTPS). "
        "Leave OFF for plain-HTTP LAN/VPN access; turn ON when serving over "
        "HTTPS (e.g. behind a TLS reverse proxy), else login silently fails.",
    "SESSION_TTL_SECONDS":
        "Sliding browser-session lifetime in seconds; refreshed on each "
        "authenticated request. Idle longer than this requires re-login. "
        "Default 2592000 (30 days).",
    "SESSION_COOKIE_NAME":
        "Name of the HttpOnly session cookie. Letters, digits, '_' and '-' "
        "only.",
    "SESSION_CSRF_COOKIE_NAME":
        "Name of the JS-readable CSRF cookie echoed back as the X-CSRF-Token "
        "header on cookie-authenticated mutations. Letters, digits, '_', '-'.",
    # --- Reports store ---
    "REPORTS_DB":
        "Path to the SQLite file holding transcription error reports. "
        "Contains plaintext PHI on a medical deployment — keep on an "
        "encrypted volume.",
    "REPORTS_MAX":
        "Soft cap on the report count. On overflow, oldest closed reports "
        "(resolved/dismissed) are evicted first, then oldest open.",
    "REPORTS_RETENTION_DAYS":
        "Auto-delete reports older than this many days. Sweep runs on "
        "startup and hourly thereafter. 0 = retention sweep disabled "
        "(admin must clear manually).",
    "REPORTS_ALLOW_USER_SUBMIT":
        "Master switch for end-user (non-admin API-key) report submission. "
        "Off = only admins can submit; the button stays visible but the "
        "endpoint returns 403 for non-admin callers.",

    # --- Recent transcriptions store ---
    "RECENT_TRANSCRIPTIONS_DB":
        "Path to the SQLite file holding the persistent /quick-config "
        "trace panel + /stats \"Recent transcriptions\" widget data. "
        "Plaintext PHI on a medical deployment — keep on an encrypted "
        "volume.",
    "RECENT_TRANSCRIPTIONS_MAX":
        "Hard row-count cap. 0 = unbounded (TTL-only pruning). Lazy "
        "pruning runs every RECENT_TRANSCRIPTIONS_PRUNE_EVERY inserts, "
        "so on-disk count can briefly exceed this by up to PRUNE_EVERY "
        "rows before the next sweep.",
    "RECENT_TRANSCRIPTIONS_TTL_DAYS":
        "Auto-delete entries older than this many days. 0 = TTL "
        "disabled (count-cap only). Combined with the row cap: "
        "whichever bound is tighter wins.",
    "RECENT_TRANSCRIPTIONS_PAGE_SIZE":
        "Number of entries the browser fetches per page on "
        "/quick-config (initial load + each \"Load older\" click). "
        "Also clamps the server-side LIMIT.",
    "RECENT_TRANSCRIPTIONS_PRUNE_EVERY":
        "Lazy-prune cadence — every Nth insert runs a single DELETE "
        "that enforces both the row cap and the TTL. 0 disables lazy "
        "pruning entirely (rows accumulate until manual /clear).",
    "STATS_RECENT_TX_DISPLAY":
        "/stats dashboard \"Recent transcriptions\" widget row count. "
        "The widget is intentionally a small ticker — bumping this "
        "past ~50 makes it scroll awkwardly without adding signal.",

    # --- Captures (fine-tuning data store) ---
    "CAPTURE_RECORDINGS_ENABLED":
        "Master switch for capturing audio + word-timestamps next to each "
        "transcription, for use as Whisper fine-tuning training data. "
        "Default OFF — recordings are biometric-grade PHI and persist on "
        "disk in plaintext (encrypt the volume). Per-model "
        "WORD_TIMESTAMPS_ENABLED=False overrides this for that model: "
        "capture is skipped to avoid corrupting alignment data on models "
        "(e.g. primeline / tnfru) where DTW is broken.",
    "CAPTURES_DB":
        "Path to the SQLite file holding capture metadata + word "
        "timestamps + admin corrections. Audio files live separately "
        "under CAPTURES_DIR; this DB references them by relative path.",
    "CAPTURES_DIR":
        "Filesystem root for captured audio files. Files use a 4-char "
        "fanout (<dir>/<id[0:2]>/<id[2:4]>/<id>.<ext>) to keep directory "
        "sizes modest. PHI on disk — encrypt the volume.",
    "CAPTURES_MAX":
        "Soft cap on capture row count. On overflow, oldest rows are "
        "evicted in priority order: dismissed → audio_missing → reviewed "
        "→ new → ready (training data is protected).",
    "CAPTURES_MAX_MB":
        "Soft cap on total audio bytes (sum of files under CAPTURES_DIR, "
        "in megabytes). Eviction policy mirrors CAPTURES_MAX.",
    "CAPTURES_RETENTION_DAYS":
        "Auto-delete captures older than this many days. 0 = retention "
        "disabled (admin must clear manually). Sweep runs on startup and "
        "hourly thereafter.",
    "CAPTURE_RECORDINGS_SAMPLE_RATE":
        "Fraction of eligible transcription requests to capture, in "
        "[0.0, 1.0]. 1.0 captures every eligible request; lower values "
        "are useful when you have a lot of traffic and only need a "
        "representative sample for fine-tuning.",
    "CAPTURE_RECORDINGS_MIN_DURATION_SEC":
        "Skip capture for clips shorter than this. Filters out false "
        "starts and silence pings that VAD almost fully suppresses.",
    "CAPTURE_RECORDINGS_MAX_DURATION_SEC":
        "Skip capture for clips longer than this. Whisper fine-tuning "
        "prefers ≤30s samples; long clips can still be captured for "
        "later segmentation via the stored segments_json metadata, but "
        "very long clips are usually not worth the disk cost.",
    "CAPTURE_RECORDINGS_AUDIO_BYTES_HARD_LIMIT":
        "Pre-transcribe upload-size guard. Captures eligibility roll is "
        "skipped for uploads larger than this many bytes, even when "
        "sampling would otherwise pass.",
    "CAPTURES_PIPELINE_RULES_EXCLUDE":
        "Set of PIPELINE_RULES slugs to SKIP when computing each "
        "capture's `text_for_training` (the column /captures shows and "
        "the export emits). All other PIPELINE_RULES still run. Default "
        "skips `dictation-map` + `capitalize-after-terminator` so the "
        "stored training text matches Whisper's raw output under "
        "SUPPRESS_CHARS — \"Komma\"/\"Punkt\" stay as words; sentence-"
        "internal lowercase preserved. /transcribe runtime output is "
        "unaffected (it still applies the full pipeline). Edit + run "
        "Reprocess all to apply changes to existing captures.",
    "CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS":
        "When True, EVERY member of a group is silence-trimmed via Silero "
        "VAD before merge_wavs() concatenates them: outer edges down to "
        "CAPTURES_VAD_MARGIN_GROUP_EDGE_MS and internal gaps capped at "
        "CAPTURES_VAD_MARGIN_GROUP_INTERNAL_MS. Removes the multi-second "
        "dead air that used to stack up at member joins (member i trailing "
        "+ gap + member i+1 leading silence). Applies to newly created / "
        "re-merged groups; mitigates the hallucination failure mode in "
        "arXiv:2505.12969 (Calm-Whisper).",
    "CAPTURES_VAD_MARGIN_GROUP_EDGE_MS":
        "Per-member group trim: silence kept on each member's outer edges "
        "(default 300 ms) so tight VAD boundaries don't clip word onsets. "
        "Lower for tighter merges; raise if you hear clipped starts/ends.",
    "CAPTURES_VAD_MARGIN_GROUP_INTERNAL_MS":
        "All internal silence in a merged sample, in ms (default 300): the "
        "gap inserted BETWEEN members (normalized — added if the members' "
        "trimmed edges are below this, trimmed if above) and the cap on "
        "pauses WITHIN a member. The single inter-utterance silence knob.",
    "CAPTURES_SAMPLE_MIN_DURATION_S":
        "Minimum length of a finished training sample, in seconds "
        "(default 1.0). A junk floor that discards near-empty samples; the "
        "proposer packs the bulk toward the target so this mainly bounds "
        "single-capture samples. Must be ≤ the proposer target.",
    "CAPTURES_SAMPLE_MAX_DURATION_S":
        "Hard maximum length of a finished sample, in seconds (default "
        "29.9; must be < 30, Whisper's window). The single source of truth "
        "for the merge, the pre-merge validation, the merge estimate, and "
        "the proposer cap. Must be ≥ the proposer target.",
    "CAPTURES_SAMPLE_JOIN_STRATEGY":
        "How member transcripts concatenate in a sample: 'space' (single "
        "space) or 'period_space' ('. '). Applies to every new or "
        "regenerated sample.",
    "CAPTURES_PROPOSER_TARGET_S":
        "Length the auto-proposer packs samples toward, in seconds "
        "(default 26). The fill-score peak; keep ≥1 s below the max so the "
        "proposer doesn't camp at the rejection edge. Must sit between the "
        "sample min and max.",
    "CAPTURES_PROPOSER_SESSION_GAP_S":
        "Captures more than this many seconds apart start a new session "
        "bucket for proposal grouping (default 600).",
    "CAPTURES_PROPOSER_DUP_THRESHOLD":
        "Reject pairing two captures in one proposal when their transcript "
        "similarity ratio exceeds this (0–1, default 0.85) — a near-"
        "duplicate / echo guard.",
    "CAPTURES_PROPOSER_MAX_PROPOSALS":
        "Maximum number of merge proposals returned per request "
        "(default 20).",
}


def _F(name: str, **kwargs: Any) -> Any:
    """`Field(default=None, description=FIELD_DESCRIPTIONS[name], **kwargs)`.

    Single-source-of-truth helper: every editable field passes its name to
    this and gets its description wired up automatically. Raises KeyError
    at import time if a name is missing — keeps schema and descriptions
    in lockstep.
    """
    return Field(default=None, description=FIELD_DESCRIPTIONS[name], **kwargs)

# faster-whisper short name OR HuggingFace repo id (org/name).
_MODEL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.\-]*(/[A-Za-z0-9_.\-]+)?$"
ModelId = Annotated[str, Field(min_length=1, max_length=96, pattern=_MODEL_ID_PATTERN)]

LogLevel = Literal["debug", "info", "warning", "error", "critical"]
DeviceLit = Literal["cuda", "cpu"]
ComputeLit = Literal["float16", "int8_float16", "int8", "float32", "bfloat16"]


# =============================================================================
# Pipeline rule schema (discriminated union on `type`)
# =============================================================================
# Each rule is one row in the unified post-processing pipeline. See
# config.py:PIPELINE_RULES for the canonical seeded list.
RuleSlug = Annotated[str, Field(min_length=1, max_length=64,
                                pattern=r"^[a-z0-9-]+$")]
RuleLabel = Annotated[str, Field(min_length=1, max_length=80)]

# Tag format — Kubernetes label-style: lowercase letters/digits/hyphens,
# 1-32 chars, no leading/trailing hyphen. Tags filter which users see
# which rules on /quick-config. Re-used by api_keys_store for the
# per-user `quick_config_tags` validator so admins can't drift the two
# schemas apart.
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")


def normalize_tags(raw: Any) -> list[str]:
    """Canonicalise a raw tag list: trim, lowercase, drop empties, dedup,
    sort. Raises ValueError on any tag that doesn't match TAG_RE. Empty
    list is permitted (semantic varies by call site)."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("tags must be a list of strings")
    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        if not isinstance(t, str):
            raise ValueError(f"tag must be a string, got {type(t).__name__}")
        norm = t.strip().lower()
        if not norm:
            continue
        if not TAG_RE.match(norm):
            raise ValueError(
                f"invalid tag {t!r} — lowercase a-z0-9- only, max 32 chars,"
                " no leading/trailing hyphen"
            )
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    out.sort()
    return out


class _RuleBase(BaseModel):
    """Common fields for every PipelineRule row."""
    model_config = {"extra": "forbid"}
    name: RuleSlug
    label: RuleLabel
    enabled: bool = True
    locked: bool = False
    seeded: bool = False
    # When True, the rule is shown on /quick-config so end-users (non-admin
    # session) can edit its body fields. Toggle is admin-only — see the
    # per-type allow-list enforcement in quick_config_routes.py.
    exposed: bool = False
    # Tag list for per-user visibility. Asymmetric semantics: an empty
    # list means "visible to every authenticated user" (zero-config
    # migration); a populated list means "visible only to users whose
    # `quick_config_tags` intersects this list". Admins always see
    # everything. See auth.Permissions.can_see_rule().
    tags: list[str] = Field(default_factory=list, max_length=32)
    # Free-text rationale for the rule — why it exists, ordering constraints,
    # tradeoffs. Lives in config.json so a rule's "why" travels with it (this
    # was inline config.py commentary before the factory defaults moved to
    # config.json). Optional; defaults to "" so older config.local.json files
    # that predate this field still validate.
    note: Annotated[str, Field(max_length=4000)] = ""

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, v: Any) -> list[str]:
        return normalize_tags(v)


class RegexRule(_RuleBase):
    """Static (pattern, replacement) row. Pure re.sub."""
    type: Literal["regex"]
    pattern: Annotated[str, Field(max_length=512)]
    replacement: Annotated[str, Field(max_length=512)] = ""


class LowercaseWordlistRule(_RuleBase):
    """Strip terminator and lowercase the next word if it's in the wordlist."""
    type: Literal["callback:lowercase-wordlist"]
    pattern: Annotated[str, Field(max_length=512)]
    wordlist: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=32, pattern=r"^[A-Za-zäöüß]+$")]],
        Field(max_length=2000),
    ]


class MapRule(_RuleBase):
    """Spoken-word → symbol lookup. Pattern auto-built from `map` keys
    (longest-first alternation, case-insensitive) at compile time."""
    type: Literal["callback:map"]
    map: dict[
        Annotated[str, Field(min_length=1, max_length=64,
                             pattern=r"^[\w \-.,!?ßẞÄÖÜäöü]{1,64}$")],
        Annotated[str, Field(max_length=64)],
    ] = Field(default_factory=dict, max_length=500)
    # Server-owned: epoch seconds per `map` key, set when an entry is added or
    # its value last changed (see quick_config_routes.post_state). Never written
    # by the client — drives newest-first ordering + the inline date column on
    # /quick-config. Keys not present in `map` are dropped by the validator so
    # the two stay consistent even when an admin edits the map via /settings.
    map_meta: dict[str, int] = Field(default_factory=dict, max_length=500)

    @field_validator("map_meta", mode="after")
    @classmethod
    def _prune_map_meta(cls, v: dict[str, int], info: Any) -> dict[str, int]:
        keys = info.data.get("map") or {}
        return {k: ts for k, ts in v.items() if k in keys}


class DedupRule(_RuleBase):
    """Pattern-only row. Callback collapses each match: prefer last
    non-comma in the run; pure-comma run collapses to a single comma."""
    type: Literal["callback:dedup"]
    pattern: Annotated[str, Field(max_length=512)]


class UpperRule(_RuleBase):
    """Pattern-only row. Callback uppercases group(2) (or the entire match
    if the pattern has fewer than 2 groups)."""
    type: Literal["callback:upper"]
    pattern: Annotated[str, Field(max_length=512)]


class TerminalRule(_RuleBase):
    """Hardcoded final lstrip(' \\t\\r') + rstrip(' \\t\\r'). Always last;
    never user-editable. Exactly one terminal row is required."""
    type: Literal["terminal"]


PipelineRule = Annotated[
    RegexRule | LowercaseWordlistRule | MapRule | DedupRule | UpperRule | TerminalRule,
    Field(discriminator="type"),
]


# =============================================================================
# Per-model overrides
# =============================================================================
# A ModelOverride bundle lives at MODEL_OVERRIDES[model_id]. Every field is
# Optional — absent means "inherit the global default". The runtime helper
# main.cfg_for(model_id, field) walks: per-model override > global > faster-
# whisper default. Same precedence as everywhere else, just with one more
# layer interposed.
#
# Validation: models may only carry override values that pass the same
# constraints as the corresponding global field. Pipeline rule scoping uses
# PIPELINE_RULES_EXCLUDE: a flat list of rule slugs to skip for this model.
# Rule bodies are NEVER per-model — they stay in the single global PIPELINE_RULES
# list, edited in the global pipeline editor. The per-model pane only toggles
# inclusion via a checklist.

class ModelOverride(BaseModel):
    """Per-model override bundle. All fields optional; absent = inherit global."""
    model_config = {"extra": "forbid", "protected_namespaces": ()}

    # --- Load-time (eviction-on-edit) ---
    MODEL_DEVICE: DeviceLit | None = None
    MODEL_COMPUTE_TYPE: ComputeLit | None = None
    MODEL_DEVICE_FALLBACK: DeviceLit | None = None
    MODEL_COMPUTE_TYPE_FALLBACK: ComputeLit | None = None
    REVISION: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    NUM_WORKERS: Annotated[int, Field(ge=1, le=8)] | None = None
    DEVICE_INDEX: Annotated[int, Field(ge=0, le=15)] | None = None

    # --- Decode params (call-time) ---
    DEFAULT_LANGUAGE: Annotated[str, Field(pattern=r"^([a-z]{2})?$")] | None = None
    DEFAULT_PROMPT: Annotated[str, Field(max_length=2048)] | None = None
    DEFAULT_HOTWORDS: Annotated[str, Field(max_length=2048)] | None = None
    BEAM_SIZE: Annotated[int, Field(ge=1, le=20)] | None = None
    BEST_OF: Annotated[int, Field(ge=1, le=20)] | None = None
    CONDITION_ON_PREVIOUS_TEXT: bool | None = None
    WORD_TIMESTAMPS_ENABLED: bool | None = None
    NO_SPEECH_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    LOG_PROB_THRESHOLD: Annotated[float, Field(ge=-10.0, le=0.0)] | None = None
    COMPRESSION_RATIO_THRESHOLD: Annotated[float, Field(ge=0.0, le=10.0)] | None = None
    TEMPERATURE: Annotated[str, Field(max_length=64)] | None = None
    PATIENCE: Annotated[float, Field(ge=0.5, le=5.0)] | None = None
    LENGTH_PENALTY: Annotated[float, Field(ge=0.1, le=5.0)] | None = None
    REPETITION_PENALTY: Annotated[float, Field(ge=0.5, le=5.0)] | None = None
    NO_REPEAT_NGRAM_SIZE: Annotated[int, Field(ge=0, le=10)] | None = None
    PROMPT_RESET_ON_TEMPERATURE: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    # --- VAD ---
    VAD_FILTER: bool | None = None
    VAD_MIN_SILENCE_MS: Annotated[int, Field(ge=0, le=10000)] | None = None
    VAD_SPEECH_PAD_MS: Annotated[int, Field(ge=0, le=2000)] | None = None
    VAD_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    # --- Language detection ---
    MULTILINGUAL: bool | None = None
    LANGUAGE_DETECTION_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    LANGUAGE_DETECTION_SEGMENTS: Annotated[int, Field(ge=1, le=10)] | None = None

    # --- Anti-hallucination & token control ---
    HALLUCINATION_SILENCE_THRESHOLD: Annotated[float, Field(ge=0.0, le=60.0)] | None = None
    SUPPRESS_BLANK: bool | None = None
    SUPPRESS_TOKENS: Annotated[str, Field(max_length=256)] | None = None
    SUPPRESS_CHARS: Annotated[str, Field(max_length=64)] | None = None
    PREPEND_PUNCTUATIONS: Annotated[str, Field(max_length=64)] | None = None
    APPEND_PUNCTUATIONS: Annotated[str, Field(max_length=64)] | None = None

    # --- Output wrappers ---
    OUTPUT_PREFIX: Annotated[str, Field(max_length=512)] | None = None
    OUTPUT_SUFFIX: Annotated[str, Field(max_length=512)] | None = None

    # --- Pipeline scoping (PM-only) ---
    # EXCLUDE: force-DISABLE rules that are enabled globally.
    # INCLUDE: force-ENABLE rules that are disabled globally. Inverse list.
    # A rule slug must not appear in both — enforced by _no_overlap_… validator.
    PIPELINE_RULES_EXCLUDE: Annotated[
        list[RuleSlug],
        Field(max_length=200),
    ] | None = None
    PIPELINE_RULES_INCLUDE: Annotated[
        list[RuleSlug],
        Field(max_length=200),
    ] | None = None

    @model_validator(mode="after")
    def _no_overlap_include_exclude(self) -> "ModelOverride":
        """A rule slug cannot be both force-disabled AND force-enabled for the
        same model — admin must pick one. Catches obvious misconfiguration
        (e.g. typed both lists then forgot to clean one up)."""
        ex = set(self.PIPELINE_RULES_EXCLUDE or [])
        inc = set(self.PIPELINE_RULES_INCLUDE or [])
        overlap = ex & inc
        if overlap:
            raise ValueError(
                f"PIPELINE_RULES_EXCLUDE and PIPELINE_RULES_INCLUDE overlap: "
                f"{sorted(overlap)} — a rule cannot be both force-disabled "
                f"and force-enabled for the same model. Remove from one of "
                f"the lists."
            )
        return self


class AdminConfig(BaseModel):
    """Pydantic schema for config.local.json. Every field is Optional; absent
    means "do not override". Bounds and patterns enforce resource caps and
    cheap input hygiene at validation time. Per-field user-facing descriptions
    live in FIELD_DESCRIPTIONS above (single source of truth — change there,
    every consumer reflects it on next reload)."""

    # `protected_namespaces=()` disables Pydantic's "model_*" reserved-prefix
    # warning so we can use MODEL_DEVICE / MODEL_COMPUTE_TYPE field names.
    model_config = {"extra": "forbid", "protected_namespaces": ()}

    # --- Models ---
    DEFAULT_MODEL: ModelId | None = _F("DEFAULT_MODEL")
    # Sets serialize as JSON arrays; convert back on load. List type here lets
    # us validate per-element via the ModelId Annotated type.
    ALLOWED_MODELS: list[ModelId] | None = _F("ALLOWED_MODELS")
    MAX_LOADED_MODELS: Annotated[int, Field(ge=1, le=8)] | None = _F("MAX_LOADED_MODELS")
    MODEL_IDLE_TIMEOUT_S: Annotated[int, Field(ge=0, le=86400)] | None = _F("MODEL_IDLE_TIMEOUT_S")
    PRELOAD_MODELS: list[ModelId] | None = _F("PRELOAD_MODELS")
    MODEL_DEVICE: DeviceLit | None = _F("MODEL_DEVICE")
    MODEL_COMPUTE_TYPE: ComputeLit | None = _F("MODEL_COMPUTE_TYPE")
    MODEL_DEVICE_FALLBACK: DeviceLit | None = _F("MODEL_DEVICE_FALLBACK")
    MODEL_COMPUTE_TYPE_FALLBACK: ComputeLit | None = _F("MODEL_COMPUTE_TYPE_FALLBACK")

    # --- Decode params (transcribe-time) ---
    DEFAULT_LANGUAGE: Annotated[str, Field(pattern=r"^([a-z]{2})?$")] | None = _F("DEFAULT_LANGUAGE")
    DEFAULT_PROMPT: Annotated[str, Field(max_length=2048)] | None = _F("DEFAULT_PROMPT")
    BEAM_SIZE: Annotated[int, Field(ge=1, le=20)] | None = _F("BEAM_SIZE")
    BEST_OF: Annotated[int, Field(ge=1, le=20)] | None = _F("BEST_OF")
    VAD_FILTER: bool | None = _F("VAD_FILTER")
    VAD_MIN_SILENCE_MS: Annotated[int, Field(ge=0, le=10000)] | None = _F("VAD_MIN_SILENCE_MS")
    VAD_SPEECH_PAD_MS: Annotated[int, Field(ge=0, le=2000)] | None = _F("VAD_SPEECH_PAD_MS")
    VAD_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] | None = _F("VAD_THRESHOLD")
    CONDITION_ON_PREVIOUS_TEXT: bool | None = _F("CONDITION_ON_PREVIOUS_TEXT")
    WORD_TIMESTAMPS_ENABLED: bool | None = _F("WORD_TIMESTAMPS_ENABLED")
    NO_SPEECH_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] | None = _F("NO_SPEECH_THRESHOLD")
    LOG_PROB_THRESHOLD: Annotated[float, Field(ge=-10.0, le=0.0)] | None = _F("LOG_PROB_THRESHOLD")
    COMPRESSION_RATIO_THRESHOLD: Annotated[float, Field(ge=0.0, le=10.0)] | None = _F("COMPRESSION_RATIO_THRESHOLD")

    # --- Decode params (advanced) ---
    DEFAULT_HOTWORDS: Annotated[str, Field(max_length=2048)] | None = _F("DEFAULT_HOTWORDS")
    TEMPERATURE: Annotated[str, Field(max_length=64)] | None = _F("TEMPERATURE")
    PATIENCE: Annotated[float, Field(ge=0.5, le=5.0)] | None = _F("PATIENCE")
    LENGTH_PENALTY: Annotated[float, Field(ge=0.1, le=5.0)] | None = _F("LENGTH_PENALTY")
    REPETITION_PENALTY: Annotated[float, Field(ge=0.5, le=5.0)] | None = _F("REPETITION_PENALTY")
    NO_REPEAT_NGRAM_SIZE: Annotated[int, Field(ge=0, le=10)] | None = _F("NO_REPEAT_NGRAM_SIZE")
    PROMPT_RESET_ON_TEMPERATURE: Annotated[float, Field(ge=0.0, le=1.0)] | None = _F("PROMPT_RESET_ON_TEMPERATURE")

    # --- Language detection (active when DEFAULT_LANGUAGE is empty) ---
    MULTILINGUAL: bool | None = _F("MULTILINGUAL")
    LANGUAGE_DETECTION_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] | None = _F("LANGUAGE_DETECTION_THRESHOLD")
    LANGUAGE_DETECTION_SEGMENTS: Annotated[int, Field(ge=1, le=10)] | None = _F("LANGUAGE_DETECTION_SEGMENTS")

    # --- Anti-hallucination & token control ---
    HALLUCINATION_SILENCE_THRESHOLD: Annotated[float, Field(ge=0.0, le=60.0)] | None = _F("HALLUCINATION_SILENCE_THRESHOLD")
    SUPPRESS_BLANK: bool | None = _F("SUPPRESS_BLANK")
    SUPPRESS_TOKENS: Annotated[str, Field(max_length=256)] | None = _F("SUPPRESS_TOKENS")
    SUPPRESS_CHARS: Annotated[str, Field(max_length=64)] | None = _F("SUPPRESS_CHARS")
    PREPEND_PUNCTUATIONS: Annotated[str, Field(max_length=64)] | None = _F("PREPEND_PUNCTUATIONS")
    APPEND_PUNCTUATIONS: Annotated[str, Field(max_length=64)] | None = _F("APPEND_PUNCTUATIONS")

    # --- Output wrappers (NOT a faster-whisper param; backend-level) ---
    OUTPUT_PREFIX: Annotated[str, Field(max_length=512)] | None = _F("OUTPUT_PREFIX")
    OUTPUT_SUFFIX: Annotated[str, Field(max_length=512)] | None = _F("OUTPUT_SUFFIX")

    # --- Load-time, hardware (advanced) ---
    DOWNLOAD_ROOT: Annotated[str, Field(max_length=512)] | None = _F("DOWNLOAD_ROOT")
    LOCAL_FILES_ONLY: bool | None = _F("LOCAL_FILES_ONLY")
    USE_AUTH_TOKEN: Annotated[str, Field(max_length=256)] | None = _F("USE_AUTH_TOKEN")
    AUTO_CONVERT_HF_MODELS: bool | None = _F("AUTO_CONVERT_HF_MODELS")
    CONVERT_QUANTIZATION: Annotated[str, Field(max_length=32)] | None = _F("CONVERT_QUANTIZATION")
    CONVERTED_MODELS_DIR: Annotated[str, Field(max_length=512)] | None = _F("CONVERTED_MODELS_DIR")
    CPU_THREADS: Annotated[int, Field(ge=0, le=128)] | None = _F("CPU_THREADS")
    NUM_WORKERS: Annotated[int, Field(ge=1, le=8)] | None = _F("NUM_WORKERS")
    DEVICE_INDEX: Annotated[int, Field(ge=0, le=15)] | None = _F("DEVICE_INDEX")

    # --- Per-model overrides ---
    MODEL_OVERRIDES: dict[ModelId, ModelOverride] | None = _F("MODEL_OVERRIDES")

    # --- Pipeline ---
    PIPELINE_RULES: Annotated[list[PipelineRule], Field(max_length=200)] | None = _F("PIPELINE_RULES")
    TRACE_ENABLED: bool | None = _F("TRACE_ENABLED")

    # --- Logging ---
    LOG_FILE: Annotated[str, Field(min_length=1, max_length=512)] | None = _F("LOG_FILE")
    LOG_MAX_BYTES: Annotated[int, Field(ge=1024 * 1024, le=1024 * 1024 * 1024)] | None = _F("LOG_MAX_BYTES")
    LOG_BACKUP_COUNT: Annotated[int, Field(ge=1, le=100)] | None = _F("LOG_BACKUP_COUNT")
    LOG_VIEWER_INITIAL_LINES: Annotated[int, Field(ge=10, le=100_000)] | None = _F("LOG_VIEWER_INITIAL_LINES")
    LOG_VIEWER_DOM_MAX: Annotated[int, Field(ge=0, le=1_000_000)] | None = _F("LOG_VIEWER_DOM_MAX")

    # --- Server ---
    SERVER_HOST: Annotated[str, Field(min_length=1, max_length=64)] | None = _F("SERVER_HOST")
    SERVER_PORT: Annotated[int, Field(ge=1, le=65535)] | None = _F("SERVER_PORT")
    SERVER_WORKERS: Annotated[int, Field(ge=1, le=8)] | None = _F("SERVER_WORKERS")
    SERVER_LOG_LEVEL: LogLevel | None = _F("SERVER_LOG_LEVEL")

    # --- Admin / stats access control ---
    # Each entry must be parseable by ipaddress.ip_network(strict=False) — bare
    # IPs (v4 or v6) and CIDRs are both accepted. See _validate_hosts below.
    ADMIN_ALLOWED_HOSTS: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=64)]],
        Field(max_length=64),
    ] | None = _F("ADMIN_ALLOWED_HOSTS")
    STATS_ALLOWED_HOSTS: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=64)]],
        Field(max_length=64),
    ] | None = _F("STATS_ALLOWED_HOSTS")
    # --- Browser sessions ---
    SESSION_COOKIE_SECURE: bool | None = _F("SESSION_COOKIE_SECURE")
    SESSION_TTL_SECONDS: Annotated[
        int, Field(ge=300, le=31_536_000)
    ] | None = _F("SESSION_TTL_SECONDS")
    SESSION_COOKIE_NAME: Annotated[
        str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    ] | None = _F("SESSION_COOKIE_NAME")
    SESSION_CSRF_COOKIE_NAME: Annotated[
        str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    ] | None = _F("SESSION_CSRF_COOKIE_NAME")
    # --- Reports store ---
    REPORTS_DB: Annotated[str, Field(min_length=1, max_length=512)] | None = _F("REPORTS_DB")
    REPORTS_MAX: Annotated[int, Field(ge=10, le=100_000)] | None = _F("REPORTS_MAX")
    REPORTS_RETENTION_DAYS: Annotated[int, Field(ge=0, le=3650)] | None = _F("REPORTS_RETENTION_DAYS")
    REPORTS_ALLOW_USER_SUBMIT: bool | None = _F("REPORTS_ALLOW_USER_SUBMIT")

    # --- Recent transcriptions store (persistent /quick-config + /stats) ---
    # MAX/TTL/PRUNE_EVERY accept 0 to mean "disabled"; combined bound is
    # "tighter of MAX and TTL wins."
    RECENT_TRANSCRIPTIONS_DB: Annotated[str, Field(min_length=1, max_length=512)] | None = _F("RECENT_TRANSCRIPTIONS_DB")
    RECENT_TRANSCRIPTIONS_MAX: Annotated[int, Field(ge=0, le=100_000)] | None = _F("RECENT_TRANSCRIPTIONS_MAX")
    RECENT_TRANSCRIPTIONS_TTL_DAYS: Annotated[int, Field(ge=0, le=3650)] | None = _F("RECENT_TRANSCRIPTIONS_TTL_DAYS")
    RECENT_TRANSCRIPTIONS_PAGE_SIZE: Annotated[int, Field(ge=10, le=1000)] | None = _F("RECENT_TRANSCRIPTIONS_PAGE_SIZE")
    RECENT_TRANSCRIPTIONS_PRUNE_EVERY: Annotated[int, Field(ge=0, le=10_000)] | None = _F("RECENT_TRANSCRIPTIONS_PRUNE_EVERY")
    STATS_RECENT_TX_DISPLAY: Annotated[int, Field(ge=1, le=100)] | None = _F("STATS_RECENT_TX_DISPLAY")

    # --- Captures (fine-tuning data store) ---
    CAPTURE_RECORDINGS_ENABLED: bool | None = _F("CAPTURE_RECORDINGS_ENABLED")
    CAPTURES_DB: Annotated[str, Field(min_length=1, max_length=512)] | None = _F("CAPTURES_DB")
    CAPTURES_DIR: Annotated[str, Field(min_length=1, max_length=512)] | None = _F("CAPTURES_DIR")
    CAPTURES_MAX: Annotated[int, Field(ge=10, le=1_000_000)] | None = _F("CAPTURES_MAX")
    CAPTURES_MAX_MB: Annotated[int, Field(ge=1, le=10_000_000)] | None = _F("CAPTURES_MAX_MB")
    CAPTURES_RETENTION_DAYS: Annotated[int, Field(ge=0, le=3650)] | None = _F("CAPTURES_RETENTION_DAYS")
    CAPTURE_RECORDINGS_SAMPLE_RATE: Annotated[float, Field(ge=0.0, le=1.0)] | None = _F("CAPTURE_RECORDINGS_SAMPLE_RATE")
    CAPTURE_RECORDINGS_MIN_DURATION_SEC: Annotated[float, Field(ge=0.0, le=3600.0)] | None = _F("CAPTURE_RECORDINGS_MIN_DURATION_SEC")
    CAPTURE_RECORDINGS_MAX_DURATION_SEC: Annotated[float, Field(ge=0.1, le=86400.0)] | None = _F("CAPTURE_RECORDINGS_MAX_DURATION_SEC")
    CAPTURE_RECORDINGS_AUDIO_BYTES_HARD_LIMIT: Annotated[int, Field(ge=1024, le=10_000_000_000)] | None = _F("CAPTURE_RECORDINGS_AUDIO_BYTES_HARD_LIMIT")
    # Captures-specific pipeline-rule exclusion (set of rule slugs).
    # Stored as a list in JSON; coerced back to set at use time. The
    # admin UI surfaces this as the same rule-checklist widget used for
    # per-model PIPELINE_RULES_EXCLUDE so the editing affordance is
    # identical.
    CAPTURES_PIPELINE_RULES_EXCLUDE: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=64)]],
        Field(max_length=64),
    ] | None = _F("CAPTURES_PIPELINE_RULES_EXCLUDE")
    CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS: bool | None = _F("CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS")
    CAPTURES_VAD_MARGIN_GROUP_EDGE_MS: Annotated[int, Field(ge=0, le=2000)] | None = _F("CAPTURES_VAD_MARGIN_GROUP_EDGE_MS")
    CAPTURES_VAD_MARGIN_GROUP_INTERNAL_MS: Annotated[int, Field(ge=0, le=2000)] | None = _F("CAPTURES_VAD_MARGIN_GROUP_INTERNAL_MS")
    CAPTURES_SAMPLE_MIN_DURATION_S: Annotated[float, Field(ge=0, lt=30)] | None = _F("CAPTURES_SAMPLE_MIN_DURATION_S")
    CAPTURES_SAMPLE_MAX_DURATION_S: Annotated[float, Field(gt=0, lt=30)] | None = _F("CAPTURES_SAMPLE_MAX_DURATION_S")
    CAPTURES_SAMPLE_JOIN_STRATEGY: Literal["space", "period_space"] | None = _F("CAPTURES_SAMPLE_JOIN_STRATEGY")
    CAPTURES_PROPOSER_TARGET_S: Annotated[float, Field(gt=0, lt=30)] | None = _F("CAPTURES_PROPOSER_TARGET_S")
    CAPTURES_PROPOSER_SESSION_GAP_S: Annotated[int, Field(ge=1, le=86400)] | None = _F("CAPTURES_PROPOSER_SESSION_GAP_S")
    CAPTURES_PROPOSER_DUP_THRESHOLD: Annotated[float, Field(ge=0, le=1)] | None = _F("CAPTURES_PROPOSER_DUP_THRESHOLD")
    CAPTURES_PROPOSER_MAX_PROPOSALS: Annotated[int, Field(ge=1, le=200)] | None = _F("CAPTURES_PROPOSER_MAX_PROPOSALS")

    @model_validator(mode="after")
    def _validate_sample_sizing(self) -> "AdminConfig":
        # Enforce MIN ≤ TARGET ≤ MAX < 30 on the EFFECTIVE values (a None
        # override means "unchanged", so fall back to the live config default
        # for the comparison — catches e.g. lowering MAX below the target).
        import config as _cfg
        mn = self.CAPTURES_SAMPLE_MIN_DURATION_S
        tg = self.CAPTURES_PROPOSER_TARGET_S
        mx = self.CAPTURES_SAMPLE_MAX_DURATION_S
        mn = mn if mn is not None else float(_cfg.CAPTURES_SAMPLE_MIN_DURATION_S)
        tg = tg if tg is not None else float(_cfg.CAPTURES_PROPOSER_TARGET_S)
        mx = mx if mx is not None else float(_cfg.CAPTURES_SAMPLE_MAX_DURATION_S)
        if not (mn <= tg <= mx):
            raise ValueError(
                "require CAPTURES_SAMPLE_MIN_DURATION_S ≤ "
                "CAPTURES_PROPOSER_TARGET_S ≤ CAPTURES_SAMPLE_MAX_DURATION_S "
                f"(got {mn} ≤ {tg} ≤ {mx})"
            )
        return self

    @field_validator("LOG_FILE")
    @classmethod
    def _safe_log_path(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # Reject UNC and \\?\ extended paths; cheap to enforce, removes a class
        # of footguns where an admin types a network share by accident.
        if v.startswith("\\\\") or v.startswith("//"):
            raise ValueError("UNC / network paths are not allowed")
        # Reject path traversal segments. We use both PurePath and PureWindowsPath
        # because the deploy target is Windows but the dev machine may be Linux.
        if ".." in PurePath(v).parts or ".." in PureWindowsPath(v).parts:
            raise ValueError("'..' segments are not allowed")
        return v

    @field_validator("SERVER_HOST")
    @classmethod
    def _safe_host(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # IPv4 / IPv6 / hostname / 0.0.0.0 / ::. Loose check — the actual bind
        # error will surface on restart if the address is invalid.
        if not re.match(r"^[A-Za-z0-9._:\-\[\]]+$", v):
            raise ValueError("invalid host string")
        return v

    @field_validator("ALLOWED_MODELS", "PRELOAD_MODELS")
    @classmethod
    def _cap_list(cls, v: list[Any] | None) -> list[Any] | None:
        if v is None:
            return v
        if len(v) > 1000:
            raise ValueError(f"capped at 1000 entries (got {len(v)})")
        return v

    @field_validator("PIPELINE_RULES")
    @classmethod
    def _validate_pipeline_rules(cls, v: list[Any] | None) -> list[Any] | None:
        """Validate the unified pipeline rules list:
          1. Each rule's regex pattern compiles AND survives a 2 s catastrophic-
             backtracking guard against a 1 KB fixture. (Empty patterns are OK
             for a regex rule — that's just a no-op.)
          2. Each `regex` rule's replacement string survives `re.sub` with the
             pattern (catches bad backrefs like `\\3` when only 2 groups exist).
          3. Slug uniqueness across the list.
          4. Exactly one terminal rule, and it must be the last entry.
        """
        if v is None:
            return v
        import threading
        fixture = "Hallo. Wie geht's? 10.23 Uhr! Bitte. " * 32   # ~1 KB
        seen: set[str] = set()
        terminal_idx: int | None = None
        for idx, rule in enumerate(v):
            slug = getattr(rule, "name", None)
            rtype = getattr(rule, "type", None)
            if slug in seen:
                raise ValueError(f"duplicate rule name '{slug}' at index {idx}")
            if slug is not None:
                seen.add(slug)
            if rtype == "terminal":
                if terminal_idx is not None:
                    raise ValueError(f"only one terminal rule allowed (already at index {terminal_idx})")
                terminal_idx = idx
                continue
            # Pattern compile + 2 s timeout-guarded run. callback:map has no
            # pattern field — pattern is auto-built from map keys at compile
            # time; skip it here.
            if rtype == "callback:map":
                continue
            pattern = getattr(rule, "pattern", None)
            if not pattern:
                continue
            try:
                compiled = re.compile(pattern)
            except re.error as e:
                raise ValueError(f"rule {idx} ({slug!r}): invalid regex: {e}")
            replacement = ""
            if rtype == "regex":
                replacement = getattr(rule, "replacement", "") or ""
            result_holder: dict[str, Any] = {"done": False, "err": None}
            def _run(_compiled=compiled, _repl=replacement) -> None:
                try:
                    _compiled.sub(_repl, fixture)
                    result_holder["done"] = True
                except Exception as e:
                    result_holder["err"] = e
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=2.0)
            # Check err before done: when the regex raises (e.g. invalid
            # backref), _run sets err and leaves done=False, so the
            # done-first check would misreport every regex error as a
            # 2 s timeout.
            if result_holder["err"] is not None:
                raise ValueError(
                    f"rule {idx} ({slug!r}): regex test failed: {result_holder['err']}"
                )
            if not result_holder["done"]:
                raise ValueError(
                    f"rule {idx} ({slug!r}): regex took > 2 s on a 1 KB fixture "
                    "(likely catastrophic backtracking). Simplify the pattern."
                )
        if terminal_idx is not None and terminal_idx != len(v) - 1:
            raise ValueError(
                f"terminal rule must be the last entry "
                f"(found at index {terminal_idx}, list has {len(v)} rules)"
            )
        return v

    @model_validator(mode="after")
    def _no_orphan_overrides(self) -> "AdminConfig":
        """Refuse to save if ALLOWED_MODELS is being shrunk in a way that
        orphans entries in MODEL_OVERRIDES. Admin must clean up overrides
        first (or keep the model in the allowlist). Never silent data loss.

        Only fires when both ALLOWED_MODELS *and* MODEL_OVERRIDES are
        present in the same payload. If only one is being saved, the cross-
        check is skipped — the merged-with-existing payload that
        save_overrides() builds will catch the conflict instead.
        """
        if self.ALLOWED_MODELS is None or self.MODEL_OVERRIDES is None:
            return self
        allowed = set(self.ALLOWED_MODELS)
        if not allowed:
            # Empty allowlist = "anything goes" per config.py convention;
            # we don't need to enforce overrides being a subset.
            return self
        orphans = sorted(set(self.MODEL_OVERRIDES.keys()) - allowed)
        if orphans:
            raise ValueError(
                f"MODEL_OVERRIDES references models not in ALLOWED_MODELS: "
                f"{orphans}. Remove the override(s) first or add them back "
                f"to the allowlist."
            )
        return self

    @model_validator(mode="after")
    def _validate_pipeline_rule_slugs(self) -> "AdminConfig":
        """Reject any per-model EXCLUDE / INCLUDE that references a rule slug
        not present in the canonical PIPELINE_RULES list. Closes the silent-
        typo footgun where 'dictashion-map' would save cleanly and quietly do
        nothing at runtime.

        Only fires when both PIPELINE_RULES *and* MODEL_OVERRIDES are present
        in the same payload — partial saves (just MODEL_OVERRIDES) skip the
        check. The merged-with-existing payload built by save_overrides()
        catches it on the next full validation pass.
        """
        if self.PIPELINE_RULES is None or self.MODEL_OVERRIDES is None:
            return self
        canonical = {r.name for r in self.PIPELINE_RULES}
        if not canonical:
            return self
        for model_id, override in self.MODEL_OVERRIDES.items():
            for list_name in ("PIPELINE_RULES_EXCLUDE", "PIPELINE_RULES_INCLUDE"):
                slugs = getattr(override, list_name, None) or []
                unknown = [s for s in slugs if s not in canonical]
                if unknown:
                    raise ValueError(
                        f"MODEL_OVERRIDES[{model_id!r}].{list_name} "
                        f"references unknown rule slugs: {unknown}. "
                        f"Valid: {sorted(canonical)}."
                    )
        return self

    @field_validator("CONVERT_QUANTIZATION")
    @classmethod
    def _validate_convert_quantisation(cls, v: str | None) -> str | None:
        """Match CT2's `ACCEPTED_MODEL_TYPES` (ctranslate2 specs/model_spec.py).
        Empty / None = use the runtime default (float16)."""
        if v is None or not v.strip():
            return v
        allowed = {
            "float32", "float16", "bfloat16", "int16",
            "int8", "int8_float32", "int8_float16", "int8_bfloat16",
        }
        if v not in allowed:
            raise ValueError(
                f"CONVERT_QUANTIZATION must be one of {sorted(allowed)}; got {v!r}"
            )
        return v

    @field_validator("TEMPERATURE")
    @classmethod
    def _validate_temperature(cls, v: str | None) -> str | None:
        """temperature is stored as a comma-separated string (e.g. '0,0.2,0.4').
        Empty / None = library default. Validate parseable floats, ascending
        order is NOT enforced (faster-whisper accepts any order)."""
        if v is None or not v.strip():
            return v
        for token in v.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                f = float(token)
            except ValueError:
                raise ValueError(
                    f"temperature must be comma-separated floats; got '{token}'"
                )
            if not (0.0 <= f <= 1.0):
                raise ValueError(
                    f"temperature values must be in [0.0, 1.0]; got {f}"
                )
        return v

    @field_validator("SUPPRESS_TOKENS")
    @classmethod
    def _validate_suppress_tokens(cls, v: str | None) -> str | None:
        """suppress_tokens is stored as a comma-separated string of ints.
        '-1' is the library sentinel for default suppression set; '' = no
        suppression."""
        if v is None or not v.strip():
            return v
        for token in v.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                int(token)
            except ValueError:
                raise ValueError(
                    f"suppress_tokens must be comma-separated ints; got '{token}'"
                )
        return v

    @field_validator("ADMIN_ALLOWED_HOSTS", "STATS_ALLOWED_HOSTS")
    @classmethod
    def _validate_hosts(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        for entry in v:
            try:
                ipaddress.ip_network(entry, strict=False)
            except ValueError as e:
                raise ValueError(
                    f"'{entry}' is not a valid IP or CIDR (e.g. '127.0.0.1' or "
                    f"'192.168.1.0/24'): {e}"
                )
        return v


# Types that don't survive JSON round-trip natively. Convert after model_dump
# so consumers (config.py, main.py) get the same Python types as if the values
# were defined inline in config.py.
_POST_LOAD_COERCERS: dict[str, Any] = {
    "ALLOWED_MODELS": set,
}


# Fields removed from the schema. Stripped from raw config dicts before
# validation so on-disk config.local.json files from older versions don't
# break with `extra="forbid"`. Drop after one release cycle.
_DEPRECATED_FIELDS: frozenset[str] = frozenset({
    "TOKEN_RULES",
    # Retired in the sample-silence redesign: the per-capture proposer min is
    # now the ingestion floor; singleton manual trim is replaced by one-member
    # samples. Old config.local.json keys are stripped (not 422'd).
    "CAPTURES_PROPOSER_MIN_CLIP_S",
    "CAPTURES_VAD_MARGIN_SINGLETON_MS",
})
_DEPRECATED_OVERRIDE_FIELDS: frozenset[str] = frozenset({
    "TOKEN_RULES_INCLUDE", "TOKEN_RULES_EXCLUDE",
})


def _strip_deprecated(raw: Any) -> Any:
    """Pre-validation cleanup: drop fields that have been removed from the
    schema. Idempotent. Logs one line per dropped field."""
    if not isinstance(raw, dict):
        return raw
    cleaned = dict(raw)
    for f in _DEPRECATED_FIELDS:
        if f in cleaned:
            cleaned.pop(f, None)
            print(f"[config_store] dropped deprecated field {f!r} from local.json",
                  file=sys.stderr)
    mo = cleaned.get("MODEL_OVERRIDES")
    if isinstance(mo, dict):
        for mid, ov in mo.items():
            if not isinstance(ov, dict):
                continue
            for f in _DEPRECATED_OVERRIDE_FIELDS:
                if f in ov:
                    ov.pop(f, None)
                    print(f"[config_store] dropped deprecated override "
                          f"{mid!r}.{f!r}", file=sys.stderr)
    return cleaned


def load_overrides(path: str = OVERRIDES_PATH) -> dict[str, Any]:
    """Load and validate the overrides file. NEVER raises — returns {} on any
    error (missing file, malformed JSON, validation failure). Logs to stderr
    because the standard logger isn't fully wired at config-import time.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[config_store] cannot read {path}: {e}", file=sys.stderr)
        return {}
    if not isinstance(raw, dict):
        print(f"[config_store] {path} must contain a JSON object", file=sys.stderr)
        return {}
    raw = _strip_deprecated(raw)
    try:
        validated = AdminConfig.model_validate(raw)
    except ValidationError as e:
        print(f"[config_store] {path} failed validation; ignoring overrides:\n{e}",
              file=sys.stderr)
        return {}
    out = validated.model_dump(exclude_none=True)
    for key, coerce in _POST_LOAD_COERCERS.items():
        if key in out:
            out[key] = coerce(out[key])
    return out


def _atomic_write_json(obj: Any, path: str, *, sort_keys: bool, tmp_prefix: str) -> None:
    """Atomically write `obj` as pretty JSON to `path`.

    Write to a tempfile in the same directory, fsync, then os.replace. The
    rename is retried a few times — on Windows an AV scanner can briefly hold
    the destination open and raise PermissionError.

    `sort_keys`: True for config.local.json (stable diff of a flat settings
    dict); False for config.json so the committed factory rules keep their
    authored order and git diffs stay minimal.
    """
    dst_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dst_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=tmp_prefix, suffix=".tmp", dir=dst_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=sort_keys)
            f.flush()
            os.fsync(f.fileno())
        last_err: Exception | None = None
        for _ in range(5):
            try:
                os.replace(tmp, path)
                tmp = ""  # consumed
                break
            except PermissionError as e:
                last_err = e
                time.sleep(0.1)
        else:
            raise last_err if last_err else OSError("os.replace failed")
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def load_factory_rules(path: str = FACTORY_PATH) -> list[dict[str, Any]]:
    """Load and validate the committed factory pipeline rules from config.json.

    Unlike load_overrides(), this RAISES on any problem — config.json is a
    required, committed file and the pipeline has no rules without it. The
    caller surfaces the failure as a fatal startup error.

    Returns the validated PIPELINE_RULES list (list of plain dicts).
    """
    if not os.path.exists(path):
        raise RuntimeError(
            f"factory rules file not found: {path} — it is required and "
            f"committed to the repository. Restore it with "
            f"'git checkout config.json'."
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"cannot read factory rules file {path}: {e}") from e
    if not isinstance(raw, dict) or "PIPELINE_RULES" not in raw:
        raise RuntimeError(
            f"{path} must be a JSON object with a 'PIPELINE_RULES' key"
        )
    try:
        validated = AdminConfig.model_validate({"PIPELINE_RULES": raw["PIPELINE_RULES"]})
    except ValidationError as e:
        raise RuntimeError(f"{path} failed validation:\n{e}") from e
    return validated.model_dump(exclude_none=True, mode="json")["PIPELINE_RULES"]


def save_factory_rules(rules: list[Any], path: str = FACTORY_PATH) -> list[dict[str, Any]]:
    """Validate `rules` and atomically write them to config.json.

    Unlike save_overrides() this is a WHOLE-FILE replace — the WebUI's
    "Defaults" mode always sends the full rule list, not a dirty diff.

    Every rule is normalised to `seeded=True` — a rule living in the committed
    factory file IS a factory default by definition; this keeps the editor's
    seeded/custom distinction consistent and prevents a promoted local rule
    from landing in config.json marked `seeded:false`.

    Returns the validated, coerced rule list. Raises ValidationError on bad
    input — the route handler converts that to a 422 response.
    """
    rules = [{**r, "seeded": True} for r in rules]
    validated = AdminConfig.model_validate({"PIPELINE_RULES": rules})
    out_rules = validated.model_dump(exclude_none=True, mode="json")["PIPELINE_RULES"]
    _atomic_write_json(
        {"schema_version": 1, "PIPELINE_RULES": out_rules},
        path, sort_keys=False, tmp_prefix=".config.",
    )
    return out_rules


def save_overrides(payload: dict[str, Any], path: str = OVERRIDES_PATH) -> dict[str, Any]:
    """Validate `payload` against AdminConfig and atomically write it to disk.

    `payload` may contain ONLY the fields the user just edited — the WebUI
    sends a "dirty" diff, not the full state. We MERGE on top of whatever is
    already in `config.local.json` so partial saves preserve previously-saved
    settings. Without this, saving one field would wipe every other override
    on disk and the next restart would revert those values to the in-repo
    defaults.

    Sentinels in `payload`:
      - any key with value `None`  → REMOVE the override (revert to default)
      - any key absent from payload → KEEP the existing value on disk

    Returns a dict containing ONLY the fields actually changed by THIS call
    (after validation/coercion). The route handler uses this for "what needs
    a restart" / "what needs a cache rebuild" decisions — without this
    distinction, every save would re-flag every previously-saved cold setting
    as "restart required."

    Raises ValidationError on bad input — the route handler converts to a 422
    JSON response.

    Atomicity: write to a tempfile in the same directory, then os.replace. On
    Windows AV scanners can briefly hold the destination open; we retry the
    rename a few times with a short backoff.
    """
    # Read existing file (raw, no Pydantic) so we don't lose fields the caller
    # didn't include in `payload`. load_overrides() applies coercions that
    # don't round-trip through model_validate cleanly (set, frozenset, tuple),
    # so we read raw JSON here.
    existing: dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                existing = raw
        except (OSError, json.JSONDecodeError):
            # Corrupted file — fall through to a clean rewrite. The new payload
            # will be validated below, so we never write garbage.
            existing = {}

    # Merge: payload wins over existing. None means "remove this override."
    merged = dict(existing)
    for k, v in payload.items():
        if v is None:
            merged.pop(k, None)
        else:
            merged[k] = v

    merged = _strip_deprecated(merged)
    validated = AdminConfig.model_validate(merged)
    to_write = validated.model_dump(exclude_none=True, mode="json")

    _atomic_write_json(to_write, path, sort_keys=True, tmp_prefix=".config.local.")

    # Return only the fields that actually changed in this call. Compare
    # against `existing` (what was on disk before) using the validated form
    # so type coercions don't show up as spurious diffs.
    changed: dict[str, Any] = {}
    for k in payload:
        new_v = to_write.get(k)             # post-validation value, or None if removed
        old_v = existing.get(k)
        if new_v != old_v:
            changed[k] = new_v
    return changed


def pipeline_rule_tags(rules: Any) -> list[str]:
    """Return the deduped, sorted union of every tag across the given
    rule list. Used by `/settings/state` + `/settings/api-keys/api/users`
    to populate autocomplete in the tag-picker widget so admins don't
    have to remember the exact spelling.

    Accepts both dicts (post _canon_rules) and Pydantic models."""
    seen: set[str] = set()
    for r in (rules or []):
        if hasattr(r, "model_dump"):
            r = r.model_dump()
        if not isinstance(r, dict):
            continue
        for t in (r.get("tags") or []):
            if isinstance(t, str) and t:
                seen.add(t)
    return sorted(seen)


def env_pinned_fields() -> dict[str, str]:
    """Return {field_name: env_var_name} for fields currently pinned by env.

    The WebUI uses this to render an 'env-pinned' badge so the admin can see
    that their saved value won't take effect until the env var is unset.
    """
    return {
        field: env
        for field, env in ENV_VAR_MAPPING.items()
        if os.environ.get(env) is not None
    }


def format_validation_errors(err: ValidationError) -> list[dict[str, str]]:
    """Shape a Pydantic ValidationError into compact JSON for the WebUI.

    Each entry: {"loc": "FIELD.SUBPATH", "msg": "human-readable explanation"}.
    No traceback or input-value leaking — failure messages stay terse.
    """
    out: list[dict[str, str]] = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e.get("loc", ()))
        msg = e.get("msg", "invalid value")
        out.append({"loc": loc, "msg": msg})
    return out
