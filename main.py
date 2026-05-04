import os
import sys
import ctypes
import logging
import logging.handlers
import re
import tempfile
import time
from contextlib import asynccontextmanager

import config as cfg
# system_stats imports psutil + pynvml at module load and primes psutil's
# non-blocking counters. Imported here (early) so the priming happens before
# any request handler runs.
import system_stats

# =============================================================================
# Logging setup: stderr (with colors when TTY) + rotating file (no colors)
# =============================================================================
# Log path and rotation policy come from config.py / WHISPER_LOG_FILE.
# The file copy strips ANSI escape codes so it stays grep-friendly and the
# /logs web viewer can re-color via CSS based on content.
os.makedirs(os.path.dirname(cfg.LOG_FILE), exist_ok=True)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


class _StripAnsiFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return _ANSI_ESCAPE_RE.sub("", super().format(record))


_root = logging.getLogger()
_root.setLevel(logging.INFO)
# Remove any handlers a previous import (or basicConfig) added so we don't
# double-log on auto-reload.
for _h in list(_root.handlers):
    _root.removeHandler(_h)

_console_handler = logging.StreamHandler(sys.stderr)
_console_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
_root.addHandler(_console_handler)

_file_handler = logging.handlers.RotatingFileHandler(
    cfg.LOG_FILE, maxBytes=cfg.LOG_MAX_BYTES, backupCount=cfg.LOG_BACKUP_COUNT, encoding="utf-8",
)
_file_handler.setFormatter(_StripAnsiFormatter(
    "%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_root.addHandler(_file_handler)

# Tail WARNING+ records into an in-memory ring used by the nav-row severity
# pills and the /stats page. Does no I/O — append to a deque and return.
from web_common import SeverityCounter
_root.addHandler(SeverityCounter())

logger = logging.getLogger("whisper-api")


# =============================================================================
# Hugging Face token propagation
# =============================================================================
# faster-whisper accepts `use_auth_token=` per-WhisperModel-call and forwards
# it to huggingface_hub.snapshot_download(token=...). That covers the model
# weights download. But OTHER HF calls in the process — Silero VAD model
# load, tokenizer fetches, metadata pings — don't see that kwarg and would
# log "unauthenticated requests" warnings + hit the lower anonymous rate
# limit. Promoting cfg.USE_AUTH_TOKEN to os.environ["HF_TOKEN"] silences
# those calls AND lifts the ceiling. Per-model USE_AUTH_TOKEN overrides
# still win at the per-WhisperModel-call kwarg level, so a model that
# needs a different token (rare) still works.
#
# Live edits: admin_routes.post_state re-syncs HF_TOKEN whenever
# USE_AUTH_TOKEN changes via the admin UI, so a save takes effect without
# a service restart. Clearing USE_AUTH_TOKEN unsets HF_TOKEN.
if cfg.USE_AUTH_TOKEN:
    os.environ["HF_TOKEN"] = cfg.USE_AUTH_TOKEN
    logger.info("HF_TOKEN set from cfg.USE_AUTH_TOKEN (silences HF rate-limit "
                "warnings for non-WhisperModel calls)")


def _preload_windows_cuda_dlls() -> None:
    base_path = os.path.dirname(sys.executable)
    if os.path.basename(base_path).lower() == "scripts":
        base_path = os.path.dirname(base_path)

    nvidia_base = os.path.join(base_path, "Lib", "site-packages", "nvidia")
    cudnn_bin = os.path.join(nvidia_base, "cudnn", "bin")
    cublas_bin = os.path.join(nvidia_base, "cublas", "bin")

    logger.info("Base path: %s", base_path)
    logger.info("cuDNN path: %s", cudnn_bin)

    os.environ["PATH"] = cudnn_bin + os.pathsep + cublas_bin + os.pathsep + os.environ.get("PATH", "")

    if hasattr(os, "add_dll_directory"):
        if os.path.exists(cudnn_bin):
            os.add_dll_directory(cudnn_bin)
        if os.path.exists(cublas_bin):
            os.add_dll_directory(cublas_bin)

    dlls = [
        (cublas_bin, "cublas64_12.dll"),
        (cublas_bin, "cublasLt64_12.dll"),
        (cudnn_bin, "cudnn_graph64_9.dll"),
        (cudnn_bin, "cudnn_ops64_9.dll"),
        (cudnn_bin, "cudnn_cnn64_9.dll"),
        (cudnn_bin, "cudnn_adv64_9.dll"),
        (cudnn_bin, "cudnn64_9.dll"),
    ]
    try:
        for directory, name in dlls:
            ctypes.CDLL(os.path.join(directory, name))
        logger.info("NVIDIA DLLs pre-loaded successfully.")
    except OSError as e:
        logger.warning("Failed to pre-load DLLs: %s", e)


if sys.platform == "win32":
    _preload_windows_cuda_dlls()


from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from faster_whisper import WhisperModel


# =============================================================================
# Text post-processing pipeline — unified rules list (cfg.PIPELINE_RULES)
# =============================================================================
# A single ordered list of rules is applied to the joined transcript. Each
# rule is one of: "regex" (pattern + replacement), "callback:lowercase-wordlist"
# (smart German non-noun lowercaser), "callback:map" (dictation word→symbol
# map), "callback:dedup" (collapse adjacent punctuation runs), "callback:upper"
# (capitalize after sentence terminator), or "terminal" (final lstrip+rstrip;
# always last). See config.py:PIPELINE_RULES for the canonical seeded list and
# config_store.py for the Pydantic schema.
#
# rebuild_caches() compiles each rule's regex pattern once at module load and
# again on admin WebUI save (CACHE_REBUILD_FIELDS = {"PIPELINE_RULES"}).
# Disabled rules and skipped types (terminal, empty patterns) are filtered
# out of the compiled list — the runtime walker is just a tight for-loop.

from dataclasses import dataclass


@dataclass(frozen=True)
class _CompiledRule:
    """One row of the compiled pipeline. `payload` carries type-specific data:
      regex                       → replacement string
      callback:lowercase-wordlist → frozenset[str] of lowercase words
      callback:map                → dict[str_lower, str] lookup
      callback:dedup              → None (callback hardcoded)
      callback:upper              → None (callback hardcoded)
    `name` is the rule slug used for per-model EXCLUDE / INCLUDE matching.
    `enabled` mirrors the global `rule.enabled` flag — checked at runtime
    rather than at compile time so per-model PIPELINE_RULES_INCLUDE can
    force-enable a globally-disabled rule.
    """
    name: str
    label: str
    type: str
    pattern: "re.Pattern[str]"
    payload: object
    enabled: bool


_COMPILED_RULES: list[_CompiledRule] = []
# Captured from the terminal rule's label in cfg.PIPELINE_RULES at cache build
# time. Falls back to the constant below if the user removed the terminal row.
_TERMINAL_LABEL: str = "Trim edges (always-last)"


def _dedup_callback(match: "re.Match[str]") -> str:
    """Pick the user-intended punct from a run of 2+ adjacent marks. Whisper
    emits its own commas as soft pauses around dictation keywords; after
    substitution we get ",." (Punkt) or ",;" (Semikolon). Prefer any non-
    comma; within non-commas prefer the LAST (dictation came after the
    Whisper pause). Pure commas → single comma."""
    run = match.group(0)
    non_comma = [c for c in run if c != ","]
    return non_comma[-1] if non_comma else ","


def _upper_callback(match: "re.Match[str]") -> str:
    """Uppercase group(2) if the pattern produced two groups; else uppercase
    the entire match. Default seeded pattern produces ([.?!]\\s+|\\n+\\s*)
    + ([a-zäöüß])."""
    try:
        g1, g2 = match.group(1), match.group(2)
    except IndexError:
        return match.group(0).upper()
    return g1 + g2.upper()


def _make_lowercase_wordlist_replacer(wordlist: frozenset):
    """Returns a regex-sub callback that strips the matched terminator and
    lowercases group(2) IFF (group(2)+group(3)).lower() is in `wordlist`.
    The default seeded pattern produces three groups:
      group(1) = whitespace between terminator and next word
      group(2) = first letter of the next word
      group(3) = rest of the next word
    If the user's pattern has fewer than 3 groups we degrade to plain strip.
    """
    def replace(m: "re.Match[str]") -> str:
        try:
            ws, first, rest = m.group(1), m.group(2), m.group(3)
        except IndexError:
            return ""
        if (first + rest).lower() in wordlist:
            return ws + first.lower() + rest
        return ws + first + rest
    return replace


def _make_map_replacer(lookup: dict):
    """Returns a regex-sub callback that does a case-insensitive dict lookup
    on the entire match. Used by callback:map rules."""
    def replace(m: "re.Match[str]") -> str:
        return lookup.get(m.group(0).lower(), m.group(0))
    return replace


def rebuild_caches() -> None:
    """(Re)compile every rule in cfg.PIPELINE_RULES into _COMPILED_RULES.

    Called once at module load (just below) and again by the admin WebUI
    after a config change to PIPELINE_RULES (CACHE_REBUILD_FIELDS).

    The terminal row is filtered out (it runs as the implicit final trim,
    not via the walker). Globally-DISABLED rules are still compiled — the
    runtime filter consults `rule.enabled` per-call so per-model
    PIPELINE_RULES_INCLUDE can force-enable a globally-disabled rule. Rules
    with invalid regex are logged + skipped (the save-time validator
    usually catches these, but a hand-edited config.py or a runtime
    catastrophic-backtracking case might surface here).
    """
    global _COMPILED_RULES, _TERMINAL_LABEL
    compiled: list[_CompiledRule] = []
    terminal_label = _TERMINAL_LABEL
    for rule in cfg.PIPELINE_RULES:
        rtype = rule.get("type")
        if rtype == "terminal":
            terminal_label = rule.get("label", terminal_label)
            continue
        rule_enabled = bool(rule.get("enabled", True))

        try:
            if rtype == "callback:map":
                # Auto-build alternation regex from map keys, longest-first,
                # word-bounded, case-insensitive — matches the legacy
                # _DICTATION_REGEX behaviour exactly.
                m = rule.get("map", {}) or {}
                if not m:
                    continue
                alternation = "|".join(re.escape(k) for k in sorted(m, key=len, reverse=True))
                cre = re.compile(r"\b(" + alternation + r")\b", re.IGNORECASE)
                payload: object = {k.lower(): v for k, v in m.items()}
            else:
                pattern = rule.get("pattern", "")
                if not pattern:
                    # Empty pattern on a regex rule → skip (no-op).
                    continue
                cre = re.compile(pattern)
                if rtype == "regex":
                    payload = rule.get("replacement", "") or ""
                elif rtype == "callback:lowercase-wordlist":
                    payload = frozenset(w.lower() for w in (rule.get("wordlist", []) or []))
                elif rtype in ("callback:dedup", "callback:upper"):
                    payload = None
                else:
                    logger.warning("[pipeline] unknown rule type %r — skipping", rtype)
                    continue
        except re.error as e:
            logger.warning("[pipeline] rule %r has invalid regex (%s) — skipping",
                           rule.get("name"), e)
            continue
        compiled.append(_CompiledRule(rule.get("name", "?"),
                                       rule.get("label", rule.get("name", "?")),
                                       rtype, cre, payload, rule_enabled))
    _COMPILED_RULES = compiled
    _TERMINAL_LABEL = terminal_label


def _apply_rule(rule: _CompiledRule, text: str) -> str:
    """Dispatch on rule type. Hot path — keep it tight."""
    if rule.type == "regex":
        return rule.pattern.sub(rule.payload, text)  # type: ignore[arg-type]
    if rule.type == "callback:lowercase-wordlist":
        return rule.pattern.sub(_make_lowercase_wordlist_replacer(rule.payload), text)  # type: ignore[arg-type]
    if rule.type == "callback:map":
        return rule.pattern.sub(_make_map_replacer(rule.payload), text)  # type: ignore[arg-type]
    if rule.type == "callback:dedup":
        return rule.pattern.sub(_dedup_callback, text)
    if rule.type == "callback:upper":
        return rule.pattern.sub(_upper_callback, text)
    return text


rebuild_caches()


def _postprocess_text(text: str, model_name: "str | None" = None,
                       trace: "list | None" = None) -> str:
    """Run the unified pipeline rule list on `text`. If `trace` is a list,
    each rule that changes the text appends `(label_with_ordinal, before, after)`
    so the per-request log block can render a diff view.

    Per-model scoping (precedence top-down):
      1. PIPELINE_RULES_EXCLUDE — force-DISABLE for this model (highest priority).
      2. PIPELINE_RULES_INCLUDE — force-ENABLE for this model, even if globally
         disabled.
      3. Otherwise inherit `rule.enabled` from the global PIPELINE_RULES list.

    Effective:  (rule.enabled AND slug NOT in EXCLUDE) OR (slug IN INCLUDE).
    A rule cannot appear in both lists — pydantic validator rejects that.
    """
    exclude: "set[str]" = set()
    include: "set[str]" = set()
    if model_name:
        overrides = getattr(cfg, "MODEL_OVERRIDES", None) or {}
        m_over = overrides.get(model_name) if isinstance(overrides, dict) else None
        if isinstance(m_over, dict):
            ex = m_over.get("PIPELINE_RULES_EXCLUDE") or []
            inc = m_over.get("PIPELINE_RULES_INCLUDE") or []
            if isinstance(ex, list):
                exclude = set(ex)
            if isinstance(inc, list):
                include = set(inc)
    for ordinal, rule in enumerate(_COMPILED_RULES, start=1):
        # Force-EXCLUDE wins outright — admin explicitly turned this off.
        if rule.name in exclude:
            if trace is not None:
                trace.append((f"{ordinal} {rule.label} [EXCLUDED for {model_name}]",
                              text, text))
            continue
        forced_in = rule.name in include
        # Globally disabled and not force-included → skip silently.
        # When tracing, surface the skip so the log explains why a rule
        # didn't run.
        if not rule.enabled and not forced_in:
            if trace is not None:
                trace.append((f"{ordinal} {rule.label} [SKIPPED globally disabled]",
                              text, text))
            continue
        before = text
        text = _apply_rule(rule, before)
        if trace is not None:
            # Force-included rule: tag the trace line so the admin sees the
            # rule ran *because of* the per-model override, not the global
            # state. Always emit even when before == after, to make the
            # override path visible.
            if forced_in and not rule.enabled:
                trace.append(
                    (f"{ordinal} {rule.label} [FORCED on for {model_name}]",
                     before, text)
                )
            elif before != text:
                trace.append((f"{ordinal} {rule.label}", before, text))
    return text


# =============================================================================
# Per-request log block
# =============================================================================
# Always emitted (regardless of cfg.TRACE_ENABLED) — surfaces the decode
# params actually applied + per-segment metadata so empty-output failures
# can be diagnosed from the log alone. The per-pipeline transformation
# trace is folded in only when TRACE_ENABLED.
#
# ANSI color is intentionally dropped: the service runs under WinSW (no TTY)
# and the SSE log viewer reads raw bytes — escape codes hurt both consumers.
_LOG_WIDTH = 78
_NAME_COL = 32        # value column starts at this character
_SEG_TEXT_MAX = 80    # truncate per-segment text in the table (full text in FINAL)
_SEG_ROWS_MAX = 30    # truncate the segment table itself

# Maps decode-kwarg name → cfg-default key in cfg._BASELINE. Used by the
# `*` non-default marker. Only scalar fields are listed; lists/dicts skipped.
_KWARG_TO_CFG = {
    "beam_size": "BEAM_SIZE",
    "best_of": "BEST_OF",
    "vad_filter": "VAD_FILTER",
    "word_timestamps": "WORD_TIMESTAMPS_ENABLED",
    "condition_on_previous_text": "CONDITION_ON_PREVIOUS_TEXT",
    "no_speech_threshold": "NO_SPEECH_THRESHOLD",
    "log_prob_threshold": "LOG_PROB_THRESHOLD",
    "compression_ratio_threshold": "COMPRESSION_RATIO_THRESHOLD",
    "min_silence_duration_ms": "VAD_MIN_SILENCE_MS",
    "speech_pad_ms": "VAD_SPEECH_PAD_MS",
    "threshold": "VAD_THRESHOLD",
}


def _pretty_value(v) -> str:
    """Compact display form for a config value: `true`/`false`, `(none)` for
    None, `(empty)` for "", trimmed-zero floats, repr'd strings."""
    if v is None:
        return "(none)"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, float):
        # Preserve at least one decimal so 0.0 / -1.0 / 0.5 still read as
        # floats (not as ints). Strip extra trailing zeros only.
        s = f"{v:.2f}"
        if "." in s and s.endswith("0"):
            s = s.rstrip("0")
            if s.endswith("."):
                s += "0"
        return s
    if isinstance(v, list):
        return "[" + ", ".join(_pretty_value(x) for x in v) + "]"
    if isinstance(v, str):
        if not v:
            return "(empty)"
        if len(v) > 60:
            return repr(v[:57] + "...")
        return repr(v)
    return str(v)


def _is_non_default(key: str, value) -> bool:
    """`*` marker test. True iff a known cfg-default exists and the current
    scalar value differs from it. Skips non-scalars to avoid surprises."""
    cfg_key = _KWARG_TO_CFG.get(key)
    if not cfg_key:
        return False
    baseline_dict = getattr(cfg, "_BASELINE", None)
    if baseline_dict is None:
        return False
    baseline = baseline_dict.get(cfg_key)
    scalar = (bool, int, float, str, type(None))
    if not isinstance(value, scalar) or not isinstance(baseline, scalar):
        return False
    return value != baseline


def _param_row(indent: str, key: str, value) -> str:
    """`indent + key + spaces + value [*]` row. Value column lands at _NAME_COL
    regardless of indent depth so top-level and nested rows align."""
    star = " *" if _is_non_default(key, value) else ""
    pretty = _pretty_value(value)
    pad = max(1, _NAME_COL - len(indent) - len(key))
    return f"{indent}{key}{' ' * pad}{pretty}{star}"


def _section_rule(label: str) -> str:
    """`  ─── label ──────…` inner rule, padded to _LOG_WIDTH."""
    head = f"  ─── {label} "
    fill = max(0, _LOG_WIDTH - len(head))
    return head + ("─" * fill)


def _format_decode_params(kwargs: dict) -> list[str]:
    """Render decode params as aligned rows, with VAD parameters indented
    under vad_filter to show the relationship visually."""
    out: list[str] = []
    order = (
        "beam_size", "best_of", "temperature",
        "vad_filter",
        "word_timestamps", "condition_on_previous_text", "initial_prompt",
        "no_speech_threshold", "log_prob_threshold", "compression_ratio_threshold",
    )
    for k in order:
        if k not in kwargs:
            continue
        out.append(_param_row("    ", k, kwargs[k]))
        if k == "vad_filter" and kwargs[k] and kwargs.get("vad_parameters"):
            for vk, vv in kwargs["vad_parameters"].items():
                out.append(_param_row("      ", vk, vv))
    return out


def _format_segments_section(seg_diag: list[dict], info, kwargs: dict) -> list[str]:
    """Either a fixed-width segments table OR an empty-output diagnostic
    banner whose hint depends on `info.duration_after_vad` and `kwargs`."""
    n = len(seg_diag)
    if n == 0:
        out = [_section_rule("Segments  (n=0)  [!] no output produced")]
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        dav = getattr(info, "duration_after_vad", None)
        ip = kwargs.get("initial_prompt")
        if duration > 0 and dav is not None and float(dav) < 0.3 * duration:
            out.append(f"    likely cause: VAD ate audio  "
                       f"(duration_after_vad={float(dav):.2f}s vs {duration:.2f}s)")
            out.append("    next step:    set VAD_FILTER=false or "
                       "VAD_MIN_SILENCE_MS=250 in /config")
        elif ip:
            out.append("    likely cause: initial_prompt may be poisoning decode")
            out.append("                  (tnfru/primeline finetunes); "
                       "clear DEFAULT_PROMPT in /config")
        else:
            out.append("    likely cause: thresholds suppressed all segments")
            out.append("                  try disabling NO_SPEECH / LOG_PROB / "
                       "COMPRESSION_RATIO thresholds in /config")
        return out

    out = [_section_rule(f"Segments  (n={n})")]
    out.append(
        f"    {'#':>3}  {'start':>7}  {'end':>7}  "
        f"{'alp':>6}  {'nsp':>5}  {'cr':>5}  {'T':>4}   text"
    )
    rows = min(n, _SEG_ROWS_MAX)
    for i in range(rows):
        s = seg_diag[i]
        text = s["text"]
        if len(text) > _SEG_TEXT_MAX:
            text = text[:_SEG_TEXT_MAX - 3] + "..."
        out.append(
            f"    {s['id']:>3d}  "
            f"{s['start']:>6.2f}s  {s['end']:>6.2f}s  "
            f"{s['alp']:>+6.2f}  {s['nsp']:>5.2f}  {s['cr']:>5.2f}  "
            f"{s['temp']:>4.1f}   {text}"
        )
    if n > rows:
        out.append(f"    … (+{n - rows} more)")
    return out


def _model_compute_device(name: str) -> "tuple[str | None, str | None]":
    """Look up the actual device + compute_type a model was loaded with —
    these may differ from cfg.MODEL_* if the fallback path was taken."""
    for entry in system_stats.loaded_models_snapshot():
        if entry.get("name") == name:
            return entry.get("compute_type"), entry.get("device")
    return None, None


def _format_request_block(
    *,
    file_label: str,
    model_name: str,
    info,
    kwargs: dict,
    seg_diag: list[dict],
    raw: str,
    final: str,
    steps: "list | None" = None,
) -> str:
    """Full per-request log block. `steps` is the per-pipeline trace; passed
    in only when cfg.TRACE_ENABLED so the block stays a single message."""
    title_rule = "═" * _LOG_WIDTH
    rule = "─" * _LOG_WIDTH

    status = "[!] empty output" if len(seg_diag) == 0 else "✓ ok"
    title = "  /v1/audio/transcriptions"
    pad = max(1, _LOG_WIDTH - len(title) - len(status))
    title_line = f"{title}{' ' * pad}{status}"

    lines: list[str] = ["", title_rule, title_line, title_rule]

    lines.append(f"  file   {file_label}")
    model_line = f"  model  {model_name}"
    compute, device = _model_compute_device(model_name)
    extras = []
    if compute:
        extras.append(f"compute={compute}")
    if device:
        extras.append(f"device={device}")
    if extras:
        model_line += "   " + "  ".join(extras)
    lines.append(model_line)

    lines.append(_section_rule("Audio"))
    lang = getattr(info, "language", "?")
    lang_prob = getattr(info, "language_probability", None)
    lang_str = f"{lang}  (prob={lang_prob:.2f})" if lang_prob is not None else str(lang)
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    lines.append(f"    {'language':<{_NAME_COL - 4}}{lang_str}")
    lines.append(f"    {'duration':<{_NAME_COL - 4}}{duration:.2f}s")
    dav = getattr(info, "duration_after_vad", None)
    if dav is not None:
        retained = (float(dav) / duration * 100) if duration > 0 else 0.0
        lines.append(
            f"    {'duration_after_vad':<{_NAME_COL - 4}}"
            f"{float(dav):.2f}s   ({retained:.0f} % retained)"
        )

    lines.append(_section_rule("Decode params  (* = non-default)"))
    lines.extend(_format_decode_params(kwargs))

    lines.extend(_format_segments_section(seg_diag, info, kwargs))

    lines.append(rule)
    lines.append(f"  RAW WHISPER  {raw!r}")
    lines.append(rule)
    if steps:
        plural = "s" if len(steps) != 1 else ""
        lines.append(f"  PIPELINE  ({len(steps)} step{plural} changed text)")
        for name, before, after in steps:
            lines.append(f"    ▸ {name}")
            lines.append(f"        {before!r}")
            lines.append(f"     →  {after!r}")
        lines.append(rule)
    lines.append(f"  FINAL        {final!r}")
    lines.append(title_rule)

    return "\n".join(lines)


# =============================================================================
# Per-request model selection with LRU cache
# =============================================================================
# Clients can ask for any faster-whisper-compatible model via the OpenAI
# `model` form param. We resolve the OpenAI default `whisper-1` (and empty)
# to WHISPER_DEFAULT_MODEL, lazy-load on first use, and keep up to
# WHISPER_MAX_LOADED_MODELS hot in VRAM (LRU eviction).
#
# Examples a client can pass:
#   "whisper-1"                                        OpenAI default -> our default
#   "large-v2"                                         faster-whisper short name
#   "large-v3" / "large-v3-turbo" / "distil-large-v3"
#   "Systran/faster-whisper-large-v3"                  full HF repo id
#   "primeline/whisper-large-v3-turbo-german"          German-finetuned
#
# Set WHISPER_ALLOWED_MODELS to restrict which model names are accepted (a
# comma-separated allowlist; empty = anything goes, useful on a private LAN).
import asyncio as _asyncio_for_models
from collections import OrderedDict

# Source: cfg.DEFAULT_MODEL / cfg.ALLOWED_MODELS / cfg.MAX_LOADED_MODELS.

# Insertion order = LRU order (oldest at front). move_to_end on hit.
_loaded_models: "OrderedDict[str, WhisperModel]" = OrderedDict()
_model_load_lock = _asyncio_for_models.Lock()


def _resolve_model_name(requested: str) -> str:
    """Map OpenAI-compatible 'whisper-1' (or empty) to our configured default;
    pass anything else through as a faster-whisper / HF model identifier."""
    if not requested or requested == "whisper-1":
        return cfg.DEFAULT_MODEL
    return requested


# =============================================================================
# Per-model config resolution (per-model override > global default)
# =============================================================================
# cfg_for(model_id, field) is the canonical reader for any G/PM-scoped setting.
# It walks: cfg.MODEL_OVERRIDES[model_id][field] (if set and not None) → cfg.X
# (global default). Pure-G fields (DEFAULT_MODEL, ALLOWED_MODELS, server, log)
# are read with plain cfg.X — they have no per-model meaning.
#
# Precedence (highest to lowest):
#   request-arg  >  per-model override  >  global default  >  faster-whisper
# The first three are this function's business; the last is whatever
# faster-whisper itself defaults to when we omit a kwarg.

def cfg_for(model_id: "str | None", field: str):
    """Resolve a G/PM config field for the given model_id.

    Returns the per-model override if present and non-None, else the global
    cfg.X. Pass model_id=None to skip the override layer (useful at startup
    paths where no specific model is known).
    """
    overrides = getattr(cfg, "MODEL_OVERRIDES", None) or {}
    if model_id and isinstance(overrides, dict):
        m_over = overrides.get(model_id)
        if isinstance(m_over, dict):
            v = m_over.get(field)
            if v is not None:
                return v
    return getattr(cfg, field)


async def _get_or_load_model(name: str) -> WhisperModel:
    cached = _loaded_models.get(name)
    if cached is not None:
        _loaded_models.move_to_end(name)
        system_stats.touch_loaded_model(name)
        return cached

    if cfg.ALLOWED_MODELS and name not in cfg.ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{name}' is not in the allowed list. "
                   f"Allowed: {sorted(cfg.ALLOWED_MODELS)}",
        )

    async with _model_load_lock:
        # Re-check under the lock — another request may have loaded it.
        cached = _loaded_models.get(name)
        if cached is not None:
            _loaded_models.move_to_end(name)
            system_stats.touch_loaded_model(name)
            return cached

        # Evict the least-recently-used model(s) until we have room.
        while len(_loaded_models) >= cfg.MAX_LOADED_MODELS:
            evicted_name, _ = _loaded_models.popitem(last=False)
            logger.info("Evicting model from VRAM (LRU, max=%d): %s",
                        cfg.MAX_LOADED_MODELS, evicted_name)
            system_stats.unregister_loaded_model(evicted_name)

        logger.info("Loading model: %s", name)
        loop = _asyncio_for_models.get_running_loop()
        # NVML delta sampling: compare GPU memory before/after construction
        # to estimate this model's VRAM footprint. Done under
        # _model_load_lock so concurrent loads can't pollute the delta.
        # Subsequent loads of the same size may under-report due to
        # CTranslate2's caching allocator (cached freed memory gets reused).
        vram_before = system_stats.gpu_mem_used_bytes()
        load_t0 = time.perf_counter()
        # Per-model override > global default. Each loaded model can pin its
        # own device/compute_type/etc. independently.
        primary_device = cfg_for(name, "MODEL_DEVICE")
        primary_compute = cfg_for(name, "MODEL_COMPUTE_TYPE")
        fallback_device = cfg_for(name, "MODEL_DEVICE_FALLBACK")
        fallback_compute = cfg_for(name, "MODEL_COMPUTE_TYPE_FALLBACK")
        # Load-time hardware kwargs (also per-model overrideable).
        load_kwargs = {
            "device": primary_device,
            "compute_type": primary_compute,
            "device_index": cfg_for(name, "DEVICE_INDEX"),
            "cpu_threads": cfg_for(name, "CPU_THREADS"),
            "num_workers": cfg_for(name, "NUM_WORKERS"),
        }
        # Optional load-time fields — only forwarded if non-default to keep
        # WhisperModel(...) clean for the common path.
        _download_root = cfg_for(name, "DOWNLOAD_ROOT")
        if _download_root:
            load_kwargs["download_root"] = _download_root
        if cfg_for(name, "LOCAL_FILES_ONLY"):
            load_kwargs["local_files_only"] = True
        _auth_token = cfg_for(name, "USE_AUTH_TOKEN")
        if _auth_token:
            load_kwargs["use_auth_token"] = _auth_token
        # PM-only field (no global counterpart): read directly from override.
        _overrides = getattr(cfg, "MODEL_OVERRIDES", None) or {}
        _m_over = _overrides.get(name) if isinstance(_overrides, dict) else None
        _revision = _m_over.get("REVISION") if isinstance(_m_over, dict) else None
        if _revision:
            load_kwargs["revision"] = _revision

        loaded_device = primary_device
        loaded_compute = primary_compute
        try:
            new_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(name, **load_kwargs),
            )
            logger.info("Model loaded on %s: %s", primary_device, name)
        except Exception as e:
            logger.error("%s load failed for %s, falling back to %s: %s",
                         primary_device, name, fallback_device, e)
            fallback_kwargs = {
                **load_kwargs,
                "device": fallback_device,
                "compute_type": fallback_compute,
            }
            new_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(name, **fallback_kwargs),
            )
            loaded_device = fallback_device
            loaded_compute = fallback_compute
            logger.info("Model loaded on %s: %s", fallback_device, name)

        load_secs = time.perf_counter() - load_t0
        metrics.record_model_load(name, load_secs)
        vram_after = system_stats.gpu_mem_used_bytes()
        vram_delta = (vram_after - vram_before
                      if vram_before is not None and vram_after is not None
                      else None)
        # Negative deltas can happen if another process freed VRAM during load
        # (or the CT2 allocator did). Clamp to 0 rather than store nonsense.
        if vram_delta is not None and vram_delta < 0:
            vram_delta = 0
        system_stats.register_loaded_model(
            name,
            vram_bytes=vram_delta,
            device=loaded_device,
            compute_type=loaded_compute,
        )

        _loaded_models[name] = new_model
        return new_model


async def drain_then_evict(model_id: "str | None" = None) -> list[str]:
    """Drain-then-evict pattern. Drops the cached entry for `model_id` (or all
    entries when None) so the next request for that id reloads the model with
    current cfg / per-model settings.

    "Drain" comes for free from Python reference counting: in-flight transcribe
    requests already hold their own `model` reference (captured via `_get_or_
    load_model` before the executor call), so they continue running on the
    old WhisperModel instance until they finish. Only NEW requests for the
    evicted id pay the reload cost. Returns the list of evicted ids.

    Called from admin_routes.post_state when a load-time field (MODEL_DEVICE,
    MODEL_COMPUTE_TYPE, NUM_WORKERS, DEVICE_INDEX, …) changes either globally
    or in a per-model override. Either case can require reload to take
    effect; this helper makes that reload lazy and non-disruptive.
    """
    evicted: list[str] = []
    async with _model_load_lock:
        if model_id is None:
            names = list(_loaded_models.keys())
        else:
            names = [model_id] if model_id in _loaded_models else []
        for name in names:
            logger.info("[evict-on-edit] dropping %s from cache; "
                        "reload on next request", name)
            _loaded_models.pop(name, None)
            system_stats.unregister_loaded_model(name)
            evicted.append(name)
    return evicted


async def _idle_evictor() -> None:
    """Periodically unload models that haven't been touched for
    cfg.MODEL_IDLE_TIMEOUT_S seconds. Wakes every 30 s; cheap when
    timeout is 0 (early return) or no models are loaded. Acquires the
    same _model_load_lock used by _get_or_load_model so concurrent loads
    can't race with eviction.

    VRAM reclamation: pop the WhisperModel reference from _loaded_models
    so its CT2 destructor can run, then gc.collect() to break any
    remaining cycles. If torch is importable and CUDA is active, also
    call torch.cuda.empty_cache() to release pool-cached blocks.
    """
    import gc
    try:
        while True:
            await _asyncio_for_models.sleep(30)
            timeout = getattr(cfg, "MODEL_IDLE_TIMEOUT_S", 0) or 0
            if timeout <= 0 or not _loaded_models:
                continue
            now = time.monotonic()
            stale: list[str] = []
            for name, info in list(system_stats._loaded_models.items()):
                if name not in _loaded_models:
                    continue
                last = info.get("last_used_monotonic", now)
                if now - last >= timeout:
                    stale.append(name)
            if not stale:
                continue
            async with _model_load_lock:
                for name in stale:
                    if name not in _loaded_models:
                        continue   # raced with another path
                    logger.info("[idle-evict] unloading %s after %ds idle",
                                name, timeout)
                    _loaded_models.pop(name, None)
                    system_stats.unregister_loaded_model(name)
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    except _asyncio_for_models.CancelledError:
        return


@asynccontextmanager
async def lifespan(app: FastAPI):
    # If PRELOAD_MODELS is empty, fall back to preloading just DEFAULT_MODEL
    # so a fresh start always has at least one ready-to-serve model.
    to_preload = list(dict.fromkeys(cfg.PRELOAD_MODELS or [cfg.DEFAULT_MODEL]))

    if len(to_preload) > cfg.MAX_LOADED_MODELS:
        logger.warning(
            "PRELOAD_MODELS has %d entries but MAX_LOADED_MODELS=%d; "
            "LRU eviction will discard the earliest preloaded models. "
            "Bump MAX_LOADED_MODELS to at least %d to keep them all hot.",
            len(to_preload), cfg.MAX_LOADED_MODELS, len(to_preload),
        )

    for name in to_preload:
        if cfg.ALLOWED_MODELS and name not in cfg.ALLOWED_MODELS:
            logger.error(
                "Cannot preload '%s' - it is not in ALLOWED_MODELS. "
                "Add it to the allowlist or remove from PRELOAD_MODELS.", name,
            )
            continue
        try:
            logger.info("Preloading model: %s", name)
            await _get_or_load_model(name)
        except Exception as e:
            logger.error("Failed to preload model '%s': %s", name, e)

    evictor_task = _asyncio_for_models.create_task(_idle_evictor())

    yield

    evictor_task.cancel()
    try:
        await evictor_task
    except (_asyncio_for_models.CancelledError, Exception):
        pass

    _loaded_models.clear()
    # Best-effort NVML shutdown so the service exit doesn't leak driver
    # handles. Safe to call when NVML didn't init.
    system_stats.shutdown()


app = FastAPI(title="Faster Whisper API", version="1.0.0", lifespan=lifespan)

# Static assets for the /stats dashboard (vendored uPlot, etc). Local-only —
# do not put anything sensitive under static/.
from fastapi.staticfiles import StaticFiles
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)

# Per-request metrics middleware. Records (path, status, duration) for every
# HTTP request — bumps in_flight tracked separately by the transcribe handler.
import metrics


@app.middleware("http")
async def _metrics_mw(request: Request, call_next):
    path = request.url.path
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        metrics.record_request(path, status,
                               (time.perf_counter() - start) * 1000.0)


@app.post("/v1/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form("whisper-1", alias="model"),
    response_format: str = Form("json"),
    language: str = Form(None),
    temperature: float = Form(0.0),
    prompt: str = Form(""),
):
    resolved_model = _resolve_model_name(model_name)

    # Bracket the entire request with metrics.in_flight + record_transcription
    # so failed loads / failed transcriptions still surface in the dashboard.
    metrics.in_flight_transcriptions += 1
    _t0 = time.perf_counter()
    _status = "ok"
    _audio_dur: float = 0.0
    _words: int = 0
    tmp_path = None
    try:
        model = await _get_or_load_model(resolved_model)

        form_data = await request.form()
        timestamp_granularities = form_data.getlist("timestamp_granularities[]")
        if not timestamp_granularities:
            timestamp_granularities = form_data.getlist("timestamp_granularities")

        include_words = "word" in timestamp_granularities or (
            response_format == "verbose_json" and not timestamp_granularities
        )

        try:
            audio_content = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                tmp_file.write(audio_content)
                tmp_path = tmp_file.name

            # word_timestamps: AND of the (per-model-overrideable) global
            # config knob and the per-request ask. Disabled (False) bypasses
            # the DTW alignment path entirely — required for primeline-style
            # finetunes that hit faster-whisper#1212.
            want_word_ts = cfg_for(resolved_model, "WORD_TIMESTAMPS_ENABLED") and include_words

            # Empty string is NOT equivalent to None for tnfru / primeline
            # finetunes — passing "" to model.transcribe(initial_prompt=...)
            # triggers the failure mode their model card warns about. Coerce.
            _prompt = prompt or cfg_for(resolved_model, "DEFAULT_PROMPT")
            initial_prompt_arg = _prompt if _prompt else None

            _vad_filter = cfg_for(resolved_model, "VAD_FILTER")
            vad_parameters = dict(
                min_silence_duration_ms=cfg_for(resolved_model, "VAD_MIN_SILENCE_MS"),
                speech_pad_ms=cfg_for(resolved_model, "VAD_SPEECH_PAD_MS"),
                threshold=cfg_for(resolved_model, "VAD_THRESHOLD"),
            ) if _vad_filter else None

            transcribe_kwargs = dict(
                language=language or cfg_for(resolved_model, "DEFAULT_LANGUAGE"),
                beam_size=cfg_for(resolved_model, "BEAM_SIZE"),
                best_of=cfg_for(resolved_model, "BEST_OF"),
                temperature=temperature,
                vad_filter=_vad_filter,
                vad_parameters=vad_parameters,
                word_timestamps=want_word_ts,
                condition_on_previous_text=cfg_for(resolved_model, "CONDITION_ON_PREVIOUS_TEXT"),
                initial_prompt=initial_prompt_arg,
                no_speech_threshold=cfg_for(resolved_model, "NO_SPEECH_THRESHOLD"),
                log_prob_threshold=cfg_for(resolved_model, "LOG_PROB_THRESHOLD"),
                compression_ratio_threshold=cfg_for(resolved_model, "COMPRESSION_RATIO_THRESHOLD"),
            )
            # Optional advanced kwargs — only forwarded when set, so the
            # transcribe_kwargs dict stays clean for the common path.
            _hotwords = cfg_for(resolved_model, "DEFAULT_HOTWORDS")
            if _hotwords:
                transcribe_kwargs["hotwords"] = _hotwords
            _temp_str = cfg_for(resolved_model, "TEMPERATURE")
            if _temp_str:
                # Per-model override of the temperature ladder. Comma-separated
                # floats; falls back to the per-request `temperature` (default
                # 0.0) when unset.
                try:
                    ladder = tuple(float(t.strip()) for t in _temp_str.split(",") if t.strip())
                    if ladder:
                        transcribe_kwargs["temperature"] = ladder
                except ValueError:
                    pass
            _patience = cfg_for(resolved_model, "PATIENCE")
            if _patience and _patience != 1.0:
                transcribe_kwargs["patience"] = _patience
            _length_penalty = cfg_for(resolved_model, "LENGTH_PENALTY")
            if _length_penalty and _length_penalty != 1.0:
                transcribe_kwargs["length_penalty"] = _length_penalty
            _repetition_penalty = cfg_for(resolved_model, "REPETITION_PENALTY")
            if _repetition_penalty and _repetition_penalty != 1.0:
                transcribe_kwargs["repetition_penalty"] = _repetition_penalty
            _no_repeat_ngram = cfg_for(resolved_model, "NO_REPEAT_NGRAM_SIZE")
            if _no_repeat_ngram:
                transcribe_kwargs["no_repeat_ngram_size"] = _no_repeat_ngram
            _prompt_reset_t = cfg_for(resolved_model, "PROMPT_RESET_ON_TEMPERATURE")
            if _prompt_reset_t is not None and _prompt_reset_t != 0.5:
                transcribe_kwargs["prompt_reset_on_temperature"] = _prompt_reset_t
            if cfg_for(resolved_model, "MULTILINGUAL"):
                transcribe_kwargs["multilingual"] = True
            _lang_thresh = cfg_for(resolved_model, "LANGUAGE_DETECTION_THRESHOLD")
            if _lang_thresh is not None and _lang_thresh != 0.5:
                transcribe_kwargs["language_detection_threshold"] = _lang_thresh
            _lang_segs = cfg_for(resolved_model, "LANGUAGE_DETECTION_SEGMENTS")
            if _lang_segs and _lang_segs != 1:
                transcribe_kwargs["language_detection_segments"] = _lang_segs
            _hallu_silence = cfg_for(resolved_model, "HALLUCINATION_SILENCE_THRESHOLD")
            if _hallu_silence is not None:
                transcribe_kwargs["hallucination_silence_threshold"] = _hallu_silence
            _suppress_blank = cfg_for(resolved_model, "SUPPRESS_BLANK")
            if _suppress_blank is False:
                transcribe_kwargs["suppress_blank"] = False
            _suppress_tokens_str = cfg_for(resolved_model, "SUPPRESS_TOKENS")
            if _suppress_tokens_str is not None:
                if _suppress_tokens_str.strip():
                    try:
                        transcribe_kwargs["suppress_tokens"] = [
                            int(t.strip()) for t in _suppress_tokens_str.split(",") if t.strip()
                        ]
                    except ValueError:
                        pass
                else:
                    transcribe_kwargs["suppress_tokens"] = None
            _prepend_p = cfg_for(resolved_model, "PREPEND_PUNCTUATIONS")
            if _prepend_p:
                transcribe_kwargs["prepend_punctuations"] = _prepend_p
            _append_p = cfg_for(resolved_model, "APPEND_PUNCTUATIONS")
            if _append_p:
                transcribe_kwargs["append_punctuations"] = _append_p

            # Run the synchronous CTranslate2 inference in a thread executor
            # so the event loop stays responsive. CT2 releases the GIL
            # internally, so two concurrent requests on different models can
            # decode in parallel (subject to GPU compute scheduling). The
            # generator returned by transcribe() does its work lazily on
            # iteration, so we materialize it inside the executor too.
            def _do_transcribe(_model=model, _path=tmp_path,
                               _kw=transcribe_kwargs):
                _segs, _info = _model.transcribe(_path, **_kw)
                return list(_segs), _info
            loop = _asyncio_for_models.get_running_loop()
            segments_iter, info = await loop.run_in_executor(None, _do_transcribe)

            all_words = []
            segments_list = []
            # Compact per-segment metadata for the log block. Separate from
            # segments_list (the API response shape) so we can include it in
            # the diagnostic output without mutating the wire format.
            seg_diag: list[dict] = []
            # We collect raw segment text so the full transcription can be
            # post-processed in ONE pass — multi-word dictation phrases like
            # "neue Zeile" / "neuer Absatz" frequently get split across Whisper's
            # VAD-based segments, and a per-segment pass would never see them
            # together.
            raw_full_text_parts = []

            for i, segment in enumerate(segments_iter):
                raw_full_text_parts.append(segment.text)

                # segment.temperature reflects CT2's actual after-fallback
                # value (may differ from the request `temperature` if fallback
                # kicked in). segment.compression_ratio is the real gzip ratio
                # used by the suppression check — was previously hardcoded 1.0.
                seg_temp = getattr(segment, "temperature", temperature)
                seg_cr = getattr(segment, "compression_ratio", 1.0)

                # NOTE: segments[].text and words[].word carry RAW Whisper
                # output. Only the joined `text` field below is post-processed.
                # Multi-word dictation phrases ("neue Zeile") frequently get
                # split across VAD segment boundaries, so per-segment post-
                # processing would produce inconsistent results — the joined
                # pass is the authoritative one. Clients that need cleaned
                # per-segment text should read `text` (joined) and split it.
                segments_list.append({
                    "id": i,
                    "seek": 0,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "tokens": [],
                    "temperature": seg_temp,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": seg_cr,
                    "no_speech_prob": segment.no_speech_prob,
                })
                seg_diag.append({
                    "id": i,
                    "start": segment.start,
                    "end": segment.end,
                    "alp": segment.avg_logprob,
                    "nsp": segment.no_speech_prob,
                    "cr": seg_cr,
                    "temp": seg_temp,
                    "text": segment.text,
                })

                if getattr(segment, "words", None):
                    for word in segment.words:
                        all_words.append({
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                        })

            # Terminal trim — symmetric strip of spaces/tabs/CR on BOTH
            # edges. Preserves a leading or trailing "\n" emitted by
            # "neue Zeile" / "neuer Absatz" at the edges of the utterance,
            # since the user explicitly asked for the line break.
            raw_full_text = "".join(raw_full_text_parts)
            trace: "list | None" = [] if cfg.TRACE_ENABLED else None
            full_text_str = _postprocess_text(raw_full_text, model_name=resolved_model, trace=trace)
            # Output wrappers (G/PM): plain prefix/suffix concatenated to
            # the final transcript text after the pipeline runs and BEFORE
            # the final whitespace trim. Per-model overrides win.
            _output_prefix = cfg_for(resolved_model, "OUTPUT_PREFIX") or ""
            _output_suffix = cfg_for(resolved_model, "OUTPUT_SUFFIX") or ""
            if _output_prefix or _output_suffix:
                _wrap_before = full_text_str
                full_text_str = _output_prefix + full_text_str + _output_suffix
                if trace is not None and _wrap_before != full_text_str:
                    trace.append((f"{len(_COMPILED_RULES) + 1} output-wrapper",
                                  _wrap_before, full_text_str))
            before_trim = full_text_str
            full_text_str = full_text_str.lstrip(" \t\r").rstrip(" \t\r")
            if trace is not None and before_trim != full_text_str:
                trace.append((f"{len(_COMPILED_RULES) + 2} {_TERMINAL_LABEL}",
                              before_trim, full_text_str))

            # Always emit the rich diagnostic block — it's how empty-output
            # failures are debugged. The per-pipeline transformation trace
            # is only included when cfg.TRACE_ENABLED is on.
            logger.info(_format_request_block(
                file_label=f"{file.filename}  ({len(audio_content)/1024:.1f} KB, {response_format})",
                model_name=resolved_model,
                info=info,
                kwargs=transcribe_kwargs,
                seg_diag=seg_diag,
                raw=raw_full_text,
                final=full_text_str,
                steps=trace if trace is not None else None,
            ))

            _audio_dur = float(info.duration)
            _words = len(all_words)

            if response_format == "text":
                return full_text_str

            if response_format == "verbose_json":
                response = {
                    "task": "transcribe",
                    "language": info.language,
                    "duration": info.duration,
                    "text": full_text_str,
                    "segments": segments_list,
                }
                if include_words:
                    response["words"] = all_words
                return response

            return {"text": full_text_str}

        except Exception as e:
            _status = "error"
            logger.error("Transcription error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    finally:
        metrics.in_flight_transcriptions -= 1
        metrics.record_transcription(
            model=resolved_model,
            audio_dur=_audio_dur,
            proc_dur=time.perf_counter() - _t0,
            status=_status,
            words=_words,
        )


@app.get("/v1/models")
async def list_models():
    """OpenAI-style model listing — currently-loaded models plus the configured
    default. Useful for clients to discover what's available without trial."""
    import time
    now = int(time.time())
    names: list[str] = list(_loaded_models.keys())
    if cfg.DEFAULT_MODEL not in names:
        names.append(cfg.DEFAULT_MODEL)
    if cfg.ALLOWED_MODELS:
        for n in sorted(cfg.ALLOWED_MODELS):
            if n not in names:
                names.append(n)
    return {
        "object": "list",
        "data": [
            {
                "id": n,
                "object": "model",
                "created": now,
                "owned_by": "local",
                "loaded": n in _loaded_models,
            }
            for n in names
        ],
    }


# =============================================================================
# /logs - live log viewer
# =============================================================================
# A self-contained dark-theme log tailer. Loads recent context from the log
# file, then streams new lines via Server-Sent Events. Color is reapplied
# client-side based on content (since we strip ANSI before writing the file).
import asyncio
import io
from fastapi.responses import HTMLResponse, StreamingResponse

_LOG_VIEWER_INITIAL_LINES = 500


def _read_tail(path: str, n: int) -> list[str]:
    """Return the last n lines of `path` (or fewer if the file is shorter)."""
    if not os.path.exists(path):
        return []
    # Read from the end in chunks to avoid loading huge files into memory.
    with open(path, "rb") as f:
        f.seek(0, io.SEEK_END)
        size = f.tell()
        block = 8192
        data = b""
        while size > 0 and data.count(b"\n") <= n:
            read = min(block, size)
            size -= read
            f.seek(size)
            data = f.read(read) + data
    text = data.decode("utf-8", errors="replace")
    return text.splitlines()[-n:]


async def _stream_log_lines():
    """Yield SSE events: one for each existing tail line, then live tail."""
    for line in _read_tail(cfg.LOG_FILE, _LOG_VIEWER_INITIAL_LINES):
        yield f"data: {line}\n\n"

    # Live tail: open at end-of-file, poll for new lines. Reopen on rotation
    # (when the file shrinks below our last position).
    pos = os.path.getsize(cfg.LOG_FILE) if os.path.exists(cfg.LOG_FILE) else 0
    while True:
        await asyncio.sleep(0.5)
        try:
            size = os.path.getsize(cfg.LOG_FILE)
        except OSError:
            yield ": waiting-for-file\n\n"
            continue
        if size < pos:
            pos = 0  # rotated
        if size == pos:
            yield ": keepalive\n\n"
            continue
        with open(cfg.LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()
        for line in chunk.splitlines():
            yield f"data: {line}\n\n"


_LOG_VIEWER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>faster-whisper-backend · live logs</title>
{{SCALE_BOOTSTRAP_HEAD}}
<script>(function(){var v=localStorage.getItem('whisper-log-zoom');
  if(v)document.documentElement.style.setProperty('--log-zoom',v);})();</script>
<style>
  :root {
    --bg: #0d1117; --fg: #c9d1d9; --dim: #6e7681;
    --cyan: #79c0ff; --green: #7ee787; --yellow: #f2cc60;
    --red: #ff7b72; --magenta: #d2a8ff; --bold: #f0f6fc;
    --border: #30363d;
  }
  /* Font tokens, --font-sans, --font-mono and html font-size live in
     NAV_CSS (injected further down). Important: never embed the NAV_CSS
     template placeholder inside another comment block — render_page() does
     a naive string replace and would inject NAV_CSS into this comment,
     prematurely closing it (NAV_CSS contains its own internal comments)
     and silently dropping every CSS rule that follows. Header chrome
     (title, pills, buttons) gets --font-sans by default; the log lines
     themselves opt into --font-mono via .line so timestamps and tabular
     fields stay aligned. */
  html { height: 100%; }
  body { background: var(--bg); color: var(--fg);
    font: 1rem/1.5 var(--font-sans);
    margin: 0; padding: 0; min-height: 100%; }
  input, textarea, select, kbd, code, pre { font-family: var(--font-mono); }
  header { position: sticky; top: 0; background: #161b22; border-bottom: 1px solid #30363d;
    z-index: 10; padding: 0; }
  header > .header-inner { display: flex; gap: 0.75rem; align-items: center;
    max-width: 1100px; margin: 0 auto; width: 100%; padding: 0.5rem 0.875rem;
    box-sizing: border-box; }
  header .title { font-weight: 600; color: var(--bold);
    white-space: nowrap; flex-shrink: 0; }
  header .pill { padding: 0.125rem 0.5rem; border-radius: 4px; background: #21262d; color: var(--dim);
    font-size: var(--fs-xs); white-space: nowrap; flex-shrink: 0; }
  header .pill.live { color: var(--green); border: 1px solid #1f4d2a; }
  header .pill.paused { color: var(--yellow); border: 1px solid #4d3e1f; }
  header input { flex: 1; background: #0d1117; color: var(--fg); border: 1px solid #30363d;
    padding: 0.25rem 0.5rem; border-radius: 4px; font: inherit; min-width: 0; }
  header button { background: #21262d; color: var(--fg); border: 1px solid #30363d;
    padding: 0.25rem 0.625rem; border-radius: 4px; cursor: pointer; font: inherit;
    flex-shrink: 0; }
  header button:hover { background: #30363d; }
  /* width:100% + box-sizing:border-box are the fix for the "tiny centered
     column with text clipped at the start" rendering — without them the
     container sits on a content-sized box that overflows the viewport.
     No max-width: log content uses the full viewport so long lines (HF
     URLs, model paths) sit on a single line on wide monitors instead of
     wrapping into the empty side-bands. The header bar stays centered at
     1100px (its own .header-inner cap) so controls remain in a predictable
     spot. pre-wrap still wraps lines that genuinely exceed the viewport.
     font-size = global rem * --log-zoom is the multiplicative log-only
     zoom; bumping the global picker grows logs and chrome together, and
     the [-]/[+] buttons in the header then scale logs only on top. */
  #log { padding: 0.5rem 0.875rem;
    width: 100%;
    box-sizing: border-box;
    font-family: var(--font-mono);
    font-size: calc(1rem * var(--log-zoom, 1));
    white-space: pre-wrap; overflow-wrap: anywhere;
    overflow-anchor: none; }
  .line { display: block; word-break: break-word; }
  /* Log-zoom control — independent from the global UI scale picker. */
  .log-zoom { display: inline-flex; align-items: center; gap: 0.25rem;
    border: 1px solid var(--border); border-radius: 4px;
    padding: 0.125rem 0.25rem; flex-shrink: 0;
    font-size: var(--fs-xs); }
  .log-zoom button { background: transparent; border: none; color: var(--fg);
    cursor: pointer; padding: 0 0.375rem; line-height: 1;
    font-size: var(--fs-md); font-family: var(--font-mono); }
  .log-zoom button:hover:not(:disabled) { color: var(--cyan); }
  .log-zoom button:disabled { opacity: 0.35; cursor: not-allowed; }
  .log-zoom #log-zoom-pct { color: var(--dim); font-variant-numeric: tabular-nums;
    min-width: 2.75rem; text-align: center; }
  .line.hidden { display: none; }
  .line.rule    { color: var(--dim); }
  .line.title   { color: var(--bold); font-weight: 600; }
  .line.meta    { color: var(--cyan); }
  .line.raw     { color: var(--bold); }
  .line.step    { color: var(--cyan); }
  .line.before  { color: var(--dim); }
  .line.after   { color: var(--green); }
  .line.final   { color: var(--green); font-weight: 600; }
  .line.warning { color: var(--yellow); }
  .line.error   { color: var(--red); }
  .line.info    { color: var(--fg); }
  {{NAV_CSS}}
</style></head>
<body>
<header><div class="header-inner">
  <span class="title">faster-whisper-backend · logs</span>
  {{NAV}}
  <input id="filter" type="text" placeholder="filter (case-insensitive substring)…">
  <span class="log-zoom" title="zoom log content only">
    <button id="log-zoom-out" type="button" aria-label="decrease log size">−</button>
    <span id="log-zoom-pct">100%</span>
    <button id="log-zoom-in" type="button" aria-label="increase log size">+</button>
  </span>
  {{SCALE_PICKER}}
  <button id="pauseBtn">pause</button>
  <button id="clearBtn">clear</button>
  <span id="status" class="pill live">live</span>
</div></header>
<div id="log"></div>
<script>
  const log = document.getElementById('log');
  const statusEl = document.getElementById('status');
  const filterEl = document.getElementById('filter');
  const pauseBtn = document.getElementById('pauseBtn');
  const clearBtn = document.getElementById('clearBtn');
  let paused = false;
  let filterText = '';

  // Honor ?filter=... so the severity pills in the nav can deep-link.
  const initialFilter = new URLSearchParams(location.search).get('filter');
  if (initialFilter) {
    filterEl.value = initialFilter;
    filterText = initialFilter.toLowerCase();
  }

  function classify(line) {
    if (/^═+$/.test(line.trim()) || /^─+$/.test(line.trim())) return 'rule';
    if (/\\/v1\\/audio\\/transcriptions/.test(line)) return 'title';
    if (/RAW WHISPER/.test(line)) return 'raw';
    if (/FINAL\\s+'/.test(line)) return 'final';
    if (/▸\\s+\\d+\\s+/.test(line)) return 'step';
    if (/^\\s*→\\s/.test(line)) return 'after';
    if (/file=|lang=|duration=|segments=|words=|format=/.test(line)) return 'meta';
    if (/(WARNING|WARN)/.test(line)) return 'warning';
    if (/(ERROR|CRITICAL)/.test(line)) return 'error';
    if (/^\\s+'.*'$/.test(line)) return 'before';
    return 'info';
  }
  function applyFilter(el) {
    if (filterText && !el.textContent.toLowerCase().includes(filterText)) {
      el.classList.add('hidden');
    } else {
      el.classList.remove('hidden');
    }
  }
  function append(line) {
    const el = document.createElement('span');
    const cls = classify(line);
    el.className = 'line ' + cls;
    el.textContent = line + '\\n';
    applyFilter(el);
    log.appendChild(el);
    while (log.childElementCount > 5000) log.firstChild.remove();
    if (!paused) window.scrollTo(0, document.body.scrollHeight);
    // Bump the nav severity pills based on the line's classification.
    if (cls === 'warning') bumpSev('warn');
    else if (cls === 'error') bumpSev(/CRITICAL/.test(line) ? 'crit' : 'err');
  }

  // --- Live severity pills ------------------------------------------------
  // Server-rendered initial values are "best effort at page load"; the client
  // takes over from here. Maintains a sliding 60-s ring that mirrors the
  // server's severity_counts() window.
  const sevWindow = [];   // [{t, kind}]
  function setPill(id, n) {
    const el = document.getElementById(id); if (!el) return;
    const numEl = el.querySelector('.n');
    const prev = +numEl.textContent;
    numEl.textContent = n;
    el.classList.toggle('hot', n > 0);
    el.classList.toggle('zero', n === 0);
    if (n > prev) {
      // Restart the flash animation by toggling the class.
      el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
    }
  }
  function bumpSev(kind) {
    const now = Date.now();
    if (kind) sevWindow.push({ t: now, kind });
    const cutoff = now - 60_000;
    while (sevWindow.length && sevWindow[0].t < cutoff) sevWindow.shift();
    let warn = 0, err = 0, crit = 0;
    for (const e of sevWindow) {
      if (e.kind === 'crit') crit++;
      else if (e.kind === 'err') err++;
      else if (e.kind === 'warn') warn++;
    }
    setPill('sev-warn', warn);
    setPill('sev-err',  err);
    setPill('sev-crit', crit);
  }
  // Reset to 0 on first render (server-baked counts may already be stale).
  bumpSev(null);
  // Tick every 5 s to expire entries that fall out of the 60-s window.
  setInterval(() => bumpSev(null), 5000);

  filterEl.addEventListener('input', () => {
    filterText = filterEl.value.toLowerCase();
    for (const el of log.children) applyFilter(el);
  });
  pauseBtn.addEventListener('click', () => {
    paused = !paused;
    pauseBtn.textContent = paused ? 'resume' : 'pause';
    statusEl.textContent = paused ? 'paused' : 'live';
    statusEl.className = 'pill ' + (paused ? 'paused' : 'live');
    if (!paused) window.scrollTo(0, document.body.scrollHeight);
  });
  clearBtn.addEventListener('click', () => { log.innerHTML = ''; });

  const es = new EventSource('/logs/stream');
  es.onmessage = (e) => append(e.data);
  es.onerror = () => {
    statusEl.textContent = 'reconnecting…';
    statusEl.className = 'pill paused';
  };
  es.onopen = () => {
    if (!paused) {
      statusEl.textContent = 'live';
      statusEl.className = 'pill live';
    }
  };

  // --- Log-only zoom (independent of the global UI scale picker) ---------
  // Multiplies on top of --fs-base via #log { font-size: calc(1rem * --log-zoom) }.
  // Discrete steps so clicks "snap" to recognizable sizes like browser zoom.
  (function(){
    const KEY='whisper-log-zoom';
    const STEPS=[0.7, 0.85, 1, 1.2, 1.4, 1.6, 1.8, 2.0];
    const minus=document.getElementById('log-zoom-out');
    const plus =document.getElementById('log-zoom-in');
    const pct  =document.getElementById('log-zoom-pct');
    if(!minus||!plus||!pct) return;
    function nearestIdx(v){
      let best=2, dist=Infinity;
      STEPS.forEach((s,i)=>{ const d=Math.abs(s-v); if(d<dist){dist=d;best=i;} });
      return best;
    }
    let idx = nearestIdx(parseFloat(localStorage.getItem(KEY)) || 1);
    function apply(){
      const v = STEPS[idx];
      document.documentElement.style.setProperty('--log-zoom', v);
      pct.textContent = Math.round(v*100) + '%';
      minus.disabled = idx === 0;
      plus.disabled  = idx === STEPS.length - 1;
      localStorage.setItem(KEY, v);
    }
    minus.addEventListener('click', () => { if(idx>0){idx--; apply();} });
    plus .addEventListener('click', () => { if(idx<STEPS.length-1){idx++; apply();} });
    apply();
  })();
</script>
{{SCALE_PICKER_JS}}
</body></html>"""


@app.get("/logs", response_class=HTMLResponse)
async def logs_viewer():
    import web_common
    return HTMLResponse(
        web_common.render_page(_LOG_VIEWER_HTML, current="logs"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/logs/stream")
async def logs_stream():
    return StreamingResponse(_stream_log_lines(), media_type="text/event-stream")


# =============================================================================
# /stats - system overview dashboard (always on, allowlist-gated)
# =============================================================================
# Always registered. The route's own require_allowed_host dependency reads
# cfg.STATS_ALLOWED_HOSTS at request time, so the admin UI can broaden access
# without a service restart. Loopback is always allowed.
try:
    from stats_routes import router as _stats_router
    app.include_router(_stats_router)
    logger.info(
        "Stats dashboard at /stats (allowlist=%s; loopback always permitted)",
        cfg.STATS_ALLOWED_HOSTS,
    )
except Exception as _e:
    logger.error("Failed to load stats router: %s", _e)


# =============================================================================
# /config - admin WebUI (opt-in)
# =============================================================================
# Off by default: registered only when cfg.ADMIN_UI_ENABLED is True (set in
# config.py or via WHISPER_ADMIN_UI=1). Loopback-only at the router level;
# bearer-token check is layered on top when cfg.ADMIN_TOKEN is set. See
# admin_routes.py for the full security model.
if cfg.ADMIN_UI_ENABLED:
    try:
        from admin_routes import router as _admin_router
        app.include_router(_admin_router)
        if cfg.ADMIN_TOKEN:
            logger.info("Admin UI enabled at /config (allowlist + bearer token)")
        else:
            logger.warning(
                "Admin UI enabled at /config (allowlist=%s; ADMIN_TOKEN not "
                "set, so any caller from the allowlist can edit config)",
                cfg.ADMIN_ALLOWED_HOSTS,
            )
    except Exception as _e:
        logger.error("Failed to load admin router: %s", _e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",
                host=cfg.SERVER_HOST,
                port=cfg.SERVER_PORT,
                workers=cfg.SERVER_WORKERS,
                log_level=cfg.SERVER_LOG_LEVEL)
