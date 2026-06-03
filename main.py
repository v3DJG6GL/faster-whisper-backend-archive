import os
import random
import sys
import ctypes
import logging
import logging.handlers
import re
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager

# Per-process token, regenerated on every interpreter start. Surfaced via
# /v1/models so the WebUI's restart flow can detect the new process even
# if its 1 s polling missed the brief "service down" window.
BOOT_ID = uuid.uuid4().hex

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
    # Emit file timestamps in UTC (ISO-8601 with a trailing 'Z'). The log file
    # is then unambiguous regardless of the server's timezone; the /logs web
    # viewer converts each line to the reader's local time (like every other
    # timestamp surface). gmtime is a class attribute so it applies to asctime.
    converter = time.gmtime

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
    datefmt="%Y-%m-%dT%H:%M:%SZ",   # UTC (converter=gmtime); viewer localizes
))
_root.addHandler(_file_handler)

# Tail WARNING+ records into an in-memory ring used by the nav-row severity
# pills and the /stats page. Does no I/O — append to a deque and return.
from web_common import SeverityCounter
_root.addHandler(SeverityCounter())

logger = logging.getLogger("whisper-api")

# Surface any env-var coercion problems collected while config.py was imported
# (it runs before logging is configured, so it just stashes messages).
for _msg in getattr(cfg, "_ENV_WARNINGS", ()):
    logger.warning("config env override ignored: %s", _msg)


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


from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response, Depends

# Auth dep used by /v1/audio/transcriptions and /auth/whoami. In open mode
# (no admin key in DB) it returns the synthetic admin so the operator can
# bootstrap; in locked-down mode it 401s on missing/invalid bearer.
from auth import Permissions, get_current_user as _get_current_user_dep
from auth import user_from_session_cookie as _user_from_session_cookie

# faster_whisper pulls the heavy native stack (ctranslate2/onnxruntime/av). It is
# imported lazily at first model load (see _get_or_load_model) so this module
# stays importable for tests/tooling/template rendering on a box without the CUDA
# stack installed. TYPE_CHECKING keeps the WhisperModel annotation resolvable for
# type checkers without importing it at runtime.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
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
# Captured from the terminal rule's name/label in cfg.PIPELINE_RULES at cache
# build time. Falls back to the constants below if the user removed the
# terminal row. The slug is used to honor exclude-set membership (a captures
# pipeline can drop the trim by listing `trim-edges` in
# CAPTURES_PIPELINE_RULES_EXCLUDE).
_TERMINAL_NAME: str = "trim-edges"
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
    global _COMPILED_RULES, _TERMINAL_NAME, _TERMINAL_LABEL
    compiled: list[_CompiledRule] = []
    terminal_name = _TERMINAL_NAME
    terminal_label = _TERMINAL_LABEL
    for rule in cfg.PIPELINE_RULES:
        rtype = rule.get("type")
        if rtype == "terminal":
            terminal_name = rule.get("name", terminal_name)
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
                # Pre-bind the per-rule replacer once at compile time. _apply_rule
                # then becomes a uniform pattern.sub(payload, text) for every rule
                # type — no closure allocation on the hot path (twice per request).
                payload: object = _make_map_replacer({k.lower(): v for k, v in m.items()})
            else:
                pattern = rule.get("pattern", "")
                if not pattern:
                    # Empty pattern on a regex rule → skip (no-op).
                    continue
                cre = re.compile(pattern)
                if rtype == "regex":
                    payload = rule.get("replacement", "") or ""
                elif rtype == "callback:lowercase-wordlist":
                    wordlist = frozenset(w.lower() for w in (rule.get("wordlist", []) or []))
                    payload = _make_lowercase_wordlist_replacer(wordlist)
                elif rtype == "callback:dedup":
                    payload = _dedup_callback
                elif rtype == "callback:upper":
                    payload = _upper_callback
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
    _TERMINAL_NAME = terminal_name
    _TERMINAL_LABEL = terminal_label


def _apply_rule(rule: _CompiledRule, text: str) -> str:
    """Dispatch on rule type. Hot path — payload is pre-bound at
    rebuild_caches() time (a replacement string for `regex`, a pre-built
    callable for every callback:* type), so every type collapses to a
    single pattern.sub call with no per-request closure allocation."""
    return rule.pattern.sub(rule.payload, text)  # type: ignore[arg-type]


rebuild_caches()


def _postprocess_text(text: str, model_name: "str | None" = None,
                       trace: "list | None" = None,
                       extra_excludes: "set[str] | None" = None) -> str:
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

    `extra_excludes` is an additional set of rule slugs to skip on top of
    the per-model EXCLUDE. Used by the /captures storage path to produce
    a training-form transcript (cfg.CAPTURES_PIPELINE_RULES_EXCLUDE) while
    leaving the runtime /transcribe response untouched. Rules in
    `extra_excludes` are skipped even when they appear in INCLUDE — the
    captures-specific intent overrides the per-model force-on.

    The terminal "trim-edges" step (filtered out of _COMPILED_RULES at
    rebuild time) runs as the always-last step here, gated by the same
    exclude set so a trainer can preserve trailing whitespace by adding
    the slug to CAPTURES_PIPELINE_RULES_EXCLUDE. The live /transcribe
    path applies an additional unconditional trim after the output
    wrappers, so per-model exclusion of trim-edges has no effect there.
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
    if extra_excludes:
        exclude = exclude | extra_excludes
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
    term_ordinal = len(_COMPILED_RULES) + 1
    if _TERMINAL_NAME in exclude:
        if trace is not None:
            trace.append(
                (f"{term_ordinal} {_TERMINAL_LABEL} [EXCLUDED for {model_name}]",
                 text, text)
            )
    else:
        before_trim = text
        text = text.lstrip(" \t\r").rstrip(" \t\r")
        if trace is not None and before_trim != text:
            trace.append((f"{term_ordinal} {_TERMINAL_LABEL}", before_trim, text))
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
# `temperature` and `suppress_tokens` are intentionally absent — their cfg
# baselines are strings ("0.0,0.2,…", "-1") while the kwargs are tuples/lists,
# so equality comparison is meaningless without parsing both sides.
_KWARG_TO_CFG = {
    # Search / sampling
    "beam_size": "BEAM_SIZE",
    "best_of": "BEST_OF",
    "patience": "PATIENCE",
    "length_penalty": "LENGTH_PENALTY",
    "repetition_penalty": "REPETITION_PENALTY",
    "no_repeat_ngram_size": "NO_REPEAT_NGRAM_SIZE",
    "prompt_reset_on_temperature": "PROMPT_RESET_ON_TEMPERATURE",
    # VAD
    "vad_filter": "VAD_FILTER",
    "min_silence_duration_ms": "VAD_MIN_SILENCE_MS",
    "speech_pad_ms": "VAD_SPEECH_PAD_MS",
    "threshold": "VAD_THRESHOLD",
    # Output shape
    "word_timestamps": "WORD_TIMESTAMPS_ENABLED",
    # Prompt context
    "condition_on_previous_text": "CONDITION_ON_PREVIOUS_TEXT",
    "initial_prompt": "DEFAULT_PROMPT",
    "hotwords": "DEFAULT_HOTWORDS",
    # Safety / thresholds
    "no_speech_threshold": "NO_SPEECH_THRESHOLD",
    "log_prob_threshold": "LOG_PROB_THRESHOLD",
    "compression_ratio_threshold": "COMPRESSION_RATIO_THRESHOLD",
    "hallucination_silence_threshold": "HALLUCINATION_SILENCE_THRESHOLD",
    # Language detection
    "multilingual": "MULTILINGUAL",
    "language_detection_threshold": "LANGUAGE_DETECTION_THRESHOLD",
    "language_detection_segments": "LANGUAGE_DETECTION_SEGMENTS",
    # Token suppression / punctuation
    "suppress_blank": "SUPPRESS_BLANK",
    "prepend_punctuations": "PREPEND_PUNCTUATIONS",
    "append_punctuations": "APPEND_PUNCTUATIONS",
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
    under vad_filter to show the relationship visually. Fields are only
    printed when present in `kwargs` — most non-default knobs (patience,
    repetition_penalty, etc.) are conditionally added at request build
    time, so absence here means "at faster-whisper / config default".
    Order is grouped by intent (search → sampling → VAD → output → context
    → thresholds → language detection → suppression)."""
    out: list[str] = []
    order = (
        # Search / sampling
        "beam_size", "best_of", "patience", "length_penalty",
        "repetition_penalty", "no_repeat_ngram_size",
        "temperature", "prompt_reset_on_temperature",
        # VAD
        "vad_filter",
        # Output shape
        "word_timestamps",
        # Prompt context
        "condition_on_previous_text", "initial_prompt", "hotwords",
        # Safety / thresholds
        "no_speech_threshold", "log_prob_threshold",
        "compression_ratio_threshold", "hallucination_silence_threshold",
        # Language detection
        "multilingual", "language_detection_threshold",
        "language_detection_segments",
        # Token suppression / punctuation
        "suppress_blank", "suppress_tokens",
        "prepend_punctuations", "append_punctuations",
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
                       "VAD_MIN_SILENCE_MS=250 in /settings")
        elif ip:
            out.append("    likely cause: initial_prompt may be poisoning decode")
            out.append("                  (tnfru/primeline finetunes); "
                       "clear DEFAULT_PROMPT in /settings")
        else:
            out.append("    likely cause: thresholds suppressed all segments")
            out.append("                  try disabling NO_SPEECH / LOG_PROB / "
                       "COMPRESSION_RATIO thresholds in /settings")
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
    request_id: str | None = None,
    captured_id: str | None = None,
) -> str:
    """Full per-request log block. `steps` is the per-pipeline trace; passed
    in only when cfg.TRACE_ENABLED so the block stays a single message.

    `request_id` (uuid4 hex) is the cross-reference key between this
    durable log block and a report submitted via /quick-config. When
    present, the title line carries `req=<id[:8]>` so an admin reading
    a /reports row can grep the log for the matching block.

    `captured_id` is the capture row id when the capture pipeline fired
    for this request — admins can grep for `captured=<id[:8]>` to find
    the audio+timestamps row on /captures."""
    title_rule = "═" * _LOG_WIDTH
    rule = "─" * _LOG_WIDTH

    status = "[!] empty output" if len(seg_diag) == 0 else "✓ ok"
    if request_id:
        status = f"req={request_id[:8]}  {status}"
    if captured_id:
        status = f"captured={captured_id[:8]}  {status}"
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
import asyncio
from collections import OrderedDict

# Source: cfg.DEFAULT_MODEL / cfg.ALLOWED_MODELS / cfg.MAX_LOADED_MODELS.

# Insertion order = LRU order (oldest at front). move_to_end on hit.
_loaded_models: "OrderedDict[str, WhisperModel]" = OrderedDict()
_model_load_lock = asyncio.Lock()


# =============================================================================
# SUPPRESS_CHARS resolution cache
# =============================================================================
# Resolve the user's SUPPRESS_CHARS string to vocabulary token IDs via the
# loaded model's hf_tokenizer. The encoding depends on the model's BPE
# table, so the cache key is (model_id, chars_str). Invalidated on model
# unload (LRU/idle/evict-on-edit) and naturally rekeyed when SUPPRESS_CHARS
# changes.
_suppress_chars_cache: "dict[tuple[str, str], tuple[int, ...]]" = {}


def _resolve_suppress_chars(model_id: str,
                            model: "WhisperModel",
                            chars: "str | None") -> "tuple[int, ...]":
    """Return the sorted tuple of vocab IDs to suppress for the given chars.
    Each char is encoded both bare and with a leading space — Whisper's BPE
    often tokenizes a punct char differently in those positions (mirrors
    faster-whisper's own non_speech_tokens approach). Multi-piece results
    are skipped with a warning (suppressing only the first piece would
    block every word that starts with that piece)."""
    if not chars:
        return ()
    key = (model_id, chars)
    cached = _suppress_chars_cache.get(key)
    if cached is not None:
        return cached
    tok = getattr(model, "hf_tokenizer", None)
    ids: set[int] = set()
    if tok is not None:
        for ch in chars:
            if ch.isspace():
                continue
            for variant in (ch, " " + ch):
                try:
                    enc = tok.encode(variant, add_special_tokens=False)
                except Exception:
                    continue
                raw_ids = getattr(enc, "ids", None)
                if raw_ids is None and isinstance(enc, list):
                    raw_ids = enc
                if raw_ids is None:
                    continue
                if len(raw_ids) == 1:
                    ids.add(int(raw_ids[0]))
                else:
                    logger.warning(
                        "SUPPRESS_CHARS %r tokenises to %d pieces; skipping",
                        variant, len(raw_ids),
                    )
    out = tuple(sorted(ids))
    _suppress_chars_cache[key] = out
    if out:
        logger.info("SUPPRESS_CHARS resolved for %s (%r): %r",
                    model_id, chars, out)
    return out


def _drop_suppress_chars_cache(model_id: str) -> None:
    """Drop all cache entries for a given model. Called from unload paths."""
    for k in list(_suppress_chars_cache):
        if k[0] == model_id:
            _suppress_chars_cache.pop(k, None)


def _drop_loaded_model(name: str) -> None:
    """Single unload entry point: pop the cached WhisperModel, drop its
    suppress-chars entries, and unregister from the system_stats registry.
    Caller is responsible for holding _model_load_lock when the unload is
    racy with loads (LRU eviction and idle eviction paths)."""
    _loaded_models.pop(name, None)
    _drop_suppress_chars_cache(name)
    system_stats.unregister_loaded_model(name)


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


# =============================================================================
# Auto HF→CT2 conversion (opt-in via AUTO_CONVERT_HF_MODELS)
# =============================================================================
# Cache structure: <root>/<sanitised_id>/<quantization>/{model.bin, ...}
# - root: cfg.CONVERTED_MODELS_DIR or ~/.cache/whisper-ct2
# - sanitised_id: model id with "/" replaced by "__"
# - quantization: e.g. "float16" — encoded in the path so changing the cfg
#                 doesn't collide with the previously-saved version.
#
# Locking strategy:
# - Per-model asyncio.Lock (held during conversion, NOT held during the
#   subsequent WhisperModel load — so cached-model fast paths for OTHER
#   models stay snappy).
# - filelock.FileLock for cross-process safety (uvicorn --workers > 1).
# - Atomic publish: write to <output_dir>.tmp, then os.rename to final.
#   Crash mid-conversion leaves no false-positive "model.bin exists" state.

_CT2_QUANTIZATIONS = {
    "float32", "float16", "bfloat16", "int16",
    "int8", "int8_float32", "int8_float16", "int8_bfloat16",
}

# Per-model asyncio locks for conversion. Lazy-populated.
_convert_locks: "dict[str, asyncio.Lock]" = {}
_convert_locks_meta = asyncio.Lock()


def _converted_root() -> str:
    """Resolve the output root for converted models. Honours
    cfg.CONVERTED_MODELS_DIR when set, else ~/.cache/whisper-ct2."""
    return getattr(cfg, "CONVERTED_MODELS_DIR", None) or os.path.join(
        os.path.expanduser("~"), ".cache", "whisper-ct2"
    )


def _converted_dir_for(model_id: str, quantization: str) -> str:
    """Compute the deterministic output directory for `model_id` at the given
    quantisation. Sanitisation: HF repo IDs only contain `[A-Za-z0-9_.-]` plus
    one `/`, so a single replace is enough."""
    sanitised = model_id.replace("/", "__").replace(os.sep, "__")
    return os.path.join(_converted_root(), sanitised, quantization)


def _model_needs_conversion(model_id: str) -> bool:
    """Return True if `model_id` is an HF transformers Whisper checkpoint
    (has model.safetensors / pytorch_model.bin but no model.bin in the repo).
    False for already-CT2 repos and for local paths.

    Implementation: probe the HF Hub file list. Network call (~1 s) but only
    runs when AUTO_CONVERT_HF_MODELS is on AND the converted-output cache
    misses, so it's at worst once per model per process lifetime."""
    # Local path that exists → never convert.
    if os.path.isdir(model_id):
        return not os.path.isfile(os.path.join(model_id, "model.bin"))
    # Heuristic: HF repo id always contains a single "/".
    if "/" not in model_id or model_id.count("/") != 1:
        return False
    try:
        from huggingface_hub import list_repo_files
        files = set(list_repo_files(model_id))
    except Exception as e:
        logger.warning("auto-convert: could not probe %s file list (%s); "
                       "assuming no conversion needed", model_id, e)
        return False
    if "model.bin" in files:
        return False  # already CT2
    if "model.safetensors" in files or "pytorch_model.bin" in files:
        return True
    # Unknown layout — let WhisperModel try and fail naturally.
    return False


def _convert_blocking(model_id: str, output_dir: str, quantization: str) -> None:
    """Synchronous CT2 conversion. Runs in a thread executor so the event
    loop stays responsive. Lazy-imports torch / transformers / ctranslate2
    converter machinery; missing extras → RuntimeError with pip command.

    Atomic publish: writes to `<output_dir>.tmp` then renames to `output_dir`
    so a crash mid-write leaves no false-positive (next start re-detects the
    missing model.bin and retries cleanly)."""
    try:
        from ctranslate2.converters import TransformersConverter
        import transformers  # noqa: F401  ensure dep present
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"AUTO_CONVERT_HF_MODELS=true but the conversion extras are not "
            f"installed (missing {e.name!r}). Run: "
            f"pip install -r requirements-convert.txt"
        ) from e

    tmp_dir = output_dir + ".tmp"
    # Clean any stale tmp from a prior crashed run.
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    logger.info("auto-convert: %s → %s (quantisation=%s)",
                model_id, output_dir, quantization)
    t0 = time.perf_counter()
    converter = TransformersConverter(
        model_name_or_path=model_id,
        # tokenizer.json + preprocessor_config.json are required by faster-
        # whisper at runtime (transcribe.py:700, :732). vocabulary.json is
        # generated by CT2 itself; copying the HF vocab.json is harmless but
        # not necessary.
        copy_files=["tokenizer.json", "preprocessor_config.json"],
        # Loading the source as fp16 keeps RAM ~halved during conversion;
        # HF Whisper checkpoints typically ship as fp16 anyway, no precision
        # loss. low_cpu_mem_usage avoids HF's duplicate-on-CPU intermediate.
        load_as_float16=(quantization in ("float16", "int8_float16")),
        low_cpu_mem_usage=True,
    )
    converter.convert(tmp_dir, quantization=quantization, force=True)
    # Atomic publish. os.replace can't swap onto a non-empty dir on any OS (and
    # os.rename also fails onto an existing dir on Windows), so clear any stale
    # publish first, then replace.
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.replace(tmp_dir, output_dir)
    logger.info("auto-convert: %s completed in %.1fs",
                model_id, time.perf_counter() - t0)


async def _ensure_ct2_model(name: str) -> str:
    """If `name` is an HF transformers Whisper repo and AUTO_CONVERT_HF_MODELS
    is on, ensure a CT2 conversion exists locally and return its path.
    Otherwise return `name` unchanged.

    Locking: per-name asyncio.Lock + filelock.FileLock (cross-process).
    Conversion runs in a thread executor (blocking torch / numpy work)."""
    if not getattr(cfg, "AUTO_CONVERT_HF_MODELS", False):
        return name
    quantization = getattr(cfg, "CONVERT_QUANTIZATION", None) or "float16"
    if quantization not in _CT2_QUANTIZATIONS:
        logger.warning("auto-convert: invalid CONVERT_QUANTIZATION %r; "
                       "falling back to float16", quantization)
        quantization = "float16"
    output_dir = _converted_dir_for(name, quantization)
    # Fast path: already converted (idempotent across restarts).
    if os.path.isfile(os.path.join(output_dir, "model.bin")):
        return output_dir
    # Skip the file-list probe + conversion for already-CT2 repos and
    # local paths.
    if not _model_needs_conversion(name):
        return name

    # Per-model asyncio lock (lazy create). Ensures only one conversion of
    # a given model proceeds within this worker, without serialising loads
    # of OTHER models behind a global lock.
    async with _convert_locks_meta:
        lk = _convert_locks.setdefault(name, asyncio.Lock())
    async with lk:
        # Re-check inside the lock — another coroutine may have just finished.
        if os.path.isfile(os.path.join(output_dir, "model.bin")):
            return output_dir
        # Cross-process file-lock so multi-worker uvicorn doesn't double-convert.
        from filelock import FileLock, Timeout as FileLockTimeout
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        lock_path = output_dir + ".lock"
        try:
            with FileLock(lock_path, timeout=600):
                # Re-check after winning the cross-process race.
                if os.path.isfile(os.path.join(output_dir, "model.bin")):
                    return output_dir
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, _convert_blocking, name, output_dir, quantization,
                )
        except FileLockTimeout:
            raise HTTPException(
                status_code=503,
                detail=f"Auto-convert of {name!r} timed out waiting for "
                       f"a peer worker (>10 min). Check the lock file: {lock_path}",
            )
    return output_dir


async def _get_or_load_model(name: str) -> "WhisperModel":
    # Lazy import (see the TYPE_CHECKING note up top): only when a model is
    # actually loaded do we need the native faster_whisper stack.
    from faster_whisper import WhisperModel  # noqa: F401  (used in executor lambdas below)
    cached = _loaded_models.get(name)
    if cached is not None:
        # Tolerate the race against _drop_loaded_model from _idle_evictor
        # or drain_then_evict, both of which hold _model_load_lock; this
        # cache-hit fast path runs lock-free so move_to_end can KeyError
        # if the entry was popped between .get() and here.
        try:
            _loaded_models.move_to_end(name)
        except KeyError:
            pass
        system_stats.touch_loaded_model(name)
        return cached

    if cfg.ALLOWED_MODELS and name not in cfg.ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{name}' is not in the allowed list. "
                   f"Allowed: {sorted(cfg.ALLOWED_MODELS)}",
        )

    # Auto-convert HF transformers Whisper repos to CT2 format if enabled.
    # Runs OUTSIDE _model_load_lock so loads of OTHER cached models stay
    # snappy during the (rare, slow) conversion step. Returns `name`
    # unchanged if conversion is off, repo is already CT2, or it's a
    # local path with model.bin.
    load_path = await _ensure_ct2_model(name)

    async with _model_load_lock:
        # Re-check under the lock — another request may have loaded it.
        cached = _loaded_models.get(name)
        if cached is not None:
            _loaded_models.move_to_end(name)
            system_stats.touch_loaded_model(name)
            return cached

        # Evict the least-recently-used model(s) until we have room.
        while len(_loaded_models) >= cfg.MAX_LOADED_MODELS:
            evicted_name = next(iter(_loaded_models))
            logger.info("Evicting model from VRAM (LRU, max=%d): %s",
                        cfg.MAX_LOADED_MODELS, evicted_name)
            _drop_loaded_model(evicted_name)

        logger.info("Loading model: %s", name)
        loop = asyncio.get_running_loop()
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
            # `load_path` is `name` for already-CT2 / local repos; for
            # auto-converted HF repos it's the local converted directory.
            new_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(load_path, **load_kwargs),
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
                lambda: WhisperModel(load_path, **fallback_kwargs),
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
            _drop_loaded_model(name)
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
            await asyncio.sleep(30)
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
                    _drop_loaded_model(name)
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    except asyncio.CancelledError:
        return


async def _reports_retention_loop() -> None:
    """Hourly retention sweep for the reports store. Lazy-imports
    cfg.REPORTS_RETENTION_DAYS each tick so admin /settings edits take
    effect on the next cycle without a service restart. Cancellation
    on shutdown is the normal exit path."""
    import reports_store
    while True:
        try:
            await asyncio.sleep(3600)
            reports_store.sweep_retention()
        except asyncio.CancelledError:
            raise
        except Exception as _re:
            logger.error("[reports] retention loop error: %s", _re)


async def _captures_retention_loop() -> None:
    """Hourly retention sweep for the captures store. Same shape as the
    reports loop; lazy reads cfg.CAPTURES_RETENTION_DAYS each tick."""
    import captures_store
    while True:
        try:
            await asyncio.sleep(3600)
            captures_store.sweep_retention()
        except asyncio.CancelledError:
            raise
        except Exception as _ce:
            logger.error("[captures] retention loop error: %s", _ce)


def _bootstrap_admin_from_env(raw_key: str) -> None:
    """If WHISPER_BOOTSTRAP_ADMIN_KEY is set, ensure a `bootstrap-admin`
    user holds that exact key. Idempotent — if the key hash is already in
    the DB we no-op. The raw key never gets persisted in plaintext;
    only the SHA-256 hash hits disk."""
    import api_keys_store
    h = api_keys_store.hash_key(raw_key)
    # If this hash already maps to an active key, nothing to do.
    if api_keys_store._KEY_INDEX.get(h) is not None:
        return
    # Reuse or create the bootstrap-admin user.
    existing = [
        u for u in api_keys_store.list_users()
        if u["username"] == "bootstrap-admin"
    ]
    if existing:
        uid = existing[0]["id"]
        if not existing[0]["is_admin"]:
            logger.warning(
                "[auth] bootstrap-admin user exists but is_admin=False; "
                "leaving as-is. Recreate manually to escalate."
            )
            return
    else:
        uid = api_keys_store.create_user("bootstrap-admin", is_admin=True)
    # Insert the raw key (bypass generate path so we honour the env value).
    import sqlite3 as _sql
    kp, k4 = api_keys_store._split_display_parts(raw_key)
    try:
        with api_keys_store._lock:
            api_keys_store._require_conn().execute(
                "INSERT INTO api_keys"
                " (id, user_id, key_hash, key_prefix, key_last4, label,"
                "  created_ts, revoked_ts, last_used_ts)"
                " VALUES (?,?,?,?,?,?,?,NULL,NULL)",
                (
                    uuid.uuid4().hex, uid, h, kp, k4,
                    "bootstrap (env)", time.time(),
                ),
            )
            api_keys_store._rebuild_index_locked()
        logger.info(
            "[auth] bootstrap admin key registered (prefix=%s)", kp,
        )
    except _sql.IntegrityError:
        # UNIQUE on key_hash — already present in a different user. No-op.
        pass


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

    evictor_task = asyncio.create_task(_idle_evictor())

    # Open the API-keys SQLite store and start the open-mode warning loop.
    # In OPEN mode (no admin key exists yet) the loop nags every 60 s; this
    # is the operator's prompt to bootstrap an admin via /settings/api-keys.
    # Optional WHISPER_BOOTSTRAP_ADMIN_KEY env var creates the very first
    # admin in one shot without any UI.
    open_mode_task = None
    try:
        import api_keys_store
        import auth as _auth
        api_keys_store.init_db(cfg.API_KEYS_DB)
        bootstrap_key = getattr(cfg, "BOOTSTRAP_ADMIN_KEY", None)
        if bootstrap_key:
            # Only inserts if hash isn't already in api_keys. Idempotent.
            _bootstrap_admin_from_env(bootstrap_key)
        logger.info(
            "API keys store initialized at %s (locked_down=%s)",
            cfg.API_KEYS_DB, api_keys_store.is_locked_down(),
        )
        open_mode_task = asyncio.create_task(
            _auth.open_mode_warning_loop()
        )
    except Exception as _ae:
        logger.error("Failed to initialize API keys store: %s", _ae)

    # Open the browser-session store (HttpOnly cookie auth for the WebUI).
    # Non-fatal: if this fails, cookie login is unavailable but bearer auth
    # (API clients) and open mode keep working.
    try:
        import sessions_store
        sessions_store.init_db(cfg.SESSIONS_DB)
        logger.info("Session store initialized at %s", cfg.SESSIONS_DB)
    except Exception as _se:
        logger.error("Failed to initialize session store: %s", _se)

    # Open the reports SQLite store (durable, plaintext PHI on disk) and
    # run an immediate retention sweep before serving traffic. Failure
    # here is non-fatal: the rest of the app must keep working even if
    # the reports surface is broken, but the /reports page will error.
    reports_sweep_task = None
    try:
        import reports_store
        reports_store.init_db(cfg.REPORTS_DB)
        reports_store.sweep_retention()
        logger.info("Reports store initialized at %s", cfg.REPORTS_DB)
        reports_sweep_task = asyncio.create_task(
            _reports_retention_loop()
        )
    except Exception as _re:
        logger.error("Failed to initialize reports store: %s", _re)

    # Open the durable recent-transcriptions store. Replaces the legacy
    # in-memory ring buffers (quick_config_state.recent_traces +
    # metrics.recent_tx) so the /quick-config trace panel + /stats
    # dashboard widget survive service restart and scale beyond 20 rows.
    try:
        import transcriptions_store
        transcriptions_store.init_db(cfg.RECENT_TRANSCRIPTIONS_DB)
        logger.info(
            "Recent-transcriptions store initialized at %s",
            cfg.RECENT_TRANSCRIPTIONS_DB,
        )
    except Exception as _te:
        logger.error("Failed to initialize recent-transcriptions store: %s", _te)

    # Open the durable usage-rollup store. Backs the per-key/per-user usage
    # numbers on /api-keys and the usage-over-time section on /stats. Non-fatal.
    try:
        import usage_store
        usage_store.init_db(cfg.USAGE_DB)
        logger.info("Usage rollup store initialized at %s", cfg.USAGE_DB)
    except Exception as _ue:
        logger.error("Failed to initialize usage store: %s", _ue)

    # Open the captures store. Audio + word-timestamps for Whisper
    # fine-tuning, gated by CAPTURE_RECORDINGS_ENABLED. Reconcile drift
    # before serving (row says audio exists / disk says it doesn't, or
    # vice versa).
    captures_sweep_task = None
    try:
        import captures_store
        captures_store.init(cfg.CAPTURES_DB, cfg.CAPTURES_DIR)
        captures_store.reconcile_on_startup()
        captures_store.sweep_retention()
        logger.info(
            "Captures store initialized at %s (audio dir: %s, enabled=%s)",
            cfg.CAPTURES_DB, cfg.CAPTURES_DIR,
            getattr(cfg, "CAPTURE_RECORDINGS_ENABLED", False),
        )
        # capture_samples_store reuses the captures DB connection — single
        # SQLite file holds both tables.
        import capture_samples_store
        capture_samples_store.init(captures_store._require_conn(), cfg.CAPTURES_DIR)
        capture_samples_store.reconcile_on_startup()
        captures_sweep_task = asyncio.create_task(
            _captures_retention_loop()
        )
    except Exception as _ce:
        logger.error("Failed to initialize captures store: %s", _ce)

    yield

    async def _cancel(task) -> None:
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    await _cancel(evictor_task)
    await _cancel(reports_sweep_task)
    await _cancel(captures_sweep_task)
    await _cancel(open_mode_task)

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


_CSRF_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
# Paths that issue a session and therefore can't carry a CSRF token yet.
_CSRF_EXEMPT_PATHS = frozenset({"/auth/login"})


@app.middleware("http")
async def _csrf_mw(request: Request, call_next):
    """Double-submit CSRF guard for COOKIE-authenticated mutations only.

    Cookies are auto-sent by the browser, so a cross-site POST would ride
    the session cookie — hence we require an X-CSRF-Token header matching
    the session's stored token on unsafe methods. Requests WITHOUT a
    session cookie (Authorization: Bearer API clients — Vowen, curl) are
    untouched: they can't be CSRF'd and must keep working without a token.
    """
    if (
        request.method.upper() not in _CSRF_SAFE_METHODS
        and request.url.path not in _CSRF_EXEMPT_PATHS
    ):
        cookie = request.cookies.get(cfg.SESSION_COOKIE_NAME, "")
        if cookie:
            import hmac
            import sessions_store
            from fastapi.responses import JSONResponse
            sess = sessions_store.lookup_session(cookie)
            header_tok = request.headers.get("x-csrf-token", "")
            if (
                sess is None
                or not header_tok
                or not hmac.compare_digest(header_tok, sess["csrf_token"])
            ):
                return JSONResponse(
                    {"detail": "CSRF token missing or invalid"},
                    status_code=403,
                )
    return await call_next(request)


@app.middleware("http")
async def _metrics_mw(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    response = None
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        # Prefer the route's templated path (e.g. /captures/api/{cid}) so
        # per-ID URLs collapse to a single counter entry; unbounded raw-path
        # keys would otherwise grow the dict forever and turn the /stats
        # endpoint-counters panel into noise. Starlette stores the matched
        # route in the scope after routing — fall back to the raw URL path
        # for 404s and pre-routing failures.
        route = request.scope.get("route") if response is not None else None
        path = route.path if route is not None else request.url.path
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
    user: dict = Depends(_get_current_user_dep),
):
    resolved_model = _resolve_model_name(model_name)

    # Bracket the entire request with metrics.in_flight + record_transcription
    # so failed loads / failed transcriptions still surface in the dashboard.
    # request_id is generated up-front (was deferred to post-transcribe) so
    # the outer finally can correlate timing-only writes to the SQLite store
    # on the error path too.
    metrics.in_flight_transcriptions += 1
    _t0 = time.perf_counter()
    _status = "ok"
    _audio_dur: float = 0.0
    _words: int = 0
    tmp_path = None
    request_id = uuid.uuid4().hex
    _user_id = user.get("user_id")
    _key_id = user.get("key_id")
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
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "")[1]) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_content)

            # word_timestamps: AND of the (per-model-overrideable) global
            # config knob and the per-request ask. Disabled (False) bypasses
            # the DTW alignment path entirely — required for primeline-style
            # finetunes that hit faster-whisper#1212.
            gate_word_ts = cfg_for(resolved_model, "WORD_TIMESTAMPS_ENABLED")
            want_word_ts = gate_word_ts and include_words

            # Capture-for-fine-tuning decision. We gate via gate_word_ts
            # (NOT override): per-model WORD_TIMESTAMPS_ENABLED=False is
            # used on primeline/tnfru-family fine-tunes where DTW is
            # broken — forcing word_timestamps=True there produces empty
            # transcripts. Skip capture instead.
            #
            # Sampling roll + cap check + size guard happen at handler
            # entry so we don't waste DTW CPU on requests that won't
            # land. Duration filter is post-transcribe (we don't know
            # the duration yet).
            will_capture = False
            captured_id: str | None = None
            if (getattr(cfg, "CAPTURE_RECORDINGS_ENABLED", False)
                    and gate_word_ts):
                try:
                    import captures_store as _cap_store
                    cap_max = int(getattr(cfg, "CAPTURES_MAX", 5000))
                    hard_lim = int(getattr(
                        cfg, "CAPTURE_RECORDINGS_AUDIO_BYTES_HARD_LIMIT",
                        100_000_000,
                    ))
                    sample_rate = float(getattr(
                        cfg, "CAPTURE_RECORDINGS_SAMPLE_RATE", 1.0,
                    ))
                    if (_cap_store.count() < cap_max
                            and len(audio_content) < hard_lim
                            and random.random() < sample_rate):
                        will_capture = True
                        want_word_ts = True  # force DTW for capture
                except Exception as _ce:
                    logger.warning("[capture] eligibility check failed: %s", _ce)

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

            # Coerce empty to None — faster-whisper validates the value against
            # its accepted-codes list, so "" raises ValueError; None triggers
            # the first-30s auto-detect path, which is what an empty
            # DEFAULT_LANGUAGE is documented to mean.
            _language = language or cfg_for(resolved_model, "DEFAULT_LANGUAGE")
            transcribe_kwargs = dict(
                language=_language if _language else None,
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
            # SUPPRESS_CHARS — chars resolved to vocab IDs via the loaded
            # model's tokenizer, then merged into the effective suppress_tokens
            # list. Genuinely additive: existing IDs from SUPPRESS_TOKENS are
            # preserved.
            _suppress_chars = cfg_for(resolved_model, "SUPPRESS_CHARS")
            if _suppress_chars:
                extra_ids = _resolve_suppress_chars(resolved_model, model, _suppress_chars)
                if extra_ids:
                    existing = transcribe_kwargs.get("suppress_tokens")
                    if existing is None:
                        merged_ids = sorted({-1, *extra_ids})
                    else:
                        merged_ids = sorted(set(existing) | set(extra_ids))
                    transcribe_kwargs["suppress_tokens"] = merged_ids
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
            loop = asyncio.get_running_loop()
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

            raw_full_text = "".join(raw_full_text_parts)
            trace: "list | None" = [] if cfg.TRACE_ENABLED else None
            full_text_str = _postprocess_text(raw_full_text, model_name=resolved_model, trace=trace)
            # Captures-form text — same pipeline minus the captures-specific
            # exclude set (default-skips `dictation-map` + `capitalize-after-
            # terminator` so the stored text matches Whisper's raw output
            # under SUPPRESS_CHARS for fine-tune training). Only computed when
            # the capture eligibility gate has already passed at handler
            # entry — sampling missed / captures disabled / count cap full
            # are the common case and shouldn't pay for a second pipeline
            # walk per request. No trace participation: the runtime trace
            # describes the user-facing pipeline, not the training-form
            # variant.
            if will_capture:
                training_text_str = _postprocess_text(
                    raw_full_text,
                    model_name=resolved_model,
                    trace=None,
                    extra_excludes=cfg.CAPTURES_PIPELINE_RULES_EXCLUDE,
                )
            # Output wrappers (G/PM): plain prefix/suffix concatenated to
            # the final transcript text after the pipeline runs (including
            # the in-pipeline terminal trim) and BEFORE a defensive
            # post-wrapper trim. Per-model overrides win.
            _output_prefix = cfg_for(resolved_model, "OUTPUT_PREFIX") or ""
            _output_suffix = cfg_for(resolved_model, "OUTPUT_SUFFIX") or ""
            if _output_prefix or _output_suffix:
                _wrap_before = full_text_str
                full_text_str = _output_prefix + full_text_str + _output_suffix
                if trace is not None and _wrap_before != full_text_str:
                    trace.append((f"{len(_COMPILED_RULES) + 2} output-wrapper",
                                  _wrap_before, full_text_str))
            # Post-wrapper trim — strips whitespace that the wrapper config
            # itself may carry. Runs unconditionally (the per-model exclude
            # only governs the in-pipeline trim). Preserves a leading or
            # trailing "\n" emitted by "neue Zeile" / "neuer Absatz" at the
            # edges of the utterance, since the user explicitly asked for
            # the line break.
            before_trim = full_text_str
            full_text_str = full_text_str.lstrip(" \t\r").rstrip(" \t\r")
            if trace is not None and before_trim != full_text_str:
                trace.append((f"{len(_COMPILED_RULES) + 3} {_TERMINAL_LABEL}",
                              before_trim, full_text_str))

            # request_id was generated at handler entry (so the outer
            # finally can record_timing() on the error path too); it is
            # stamped on the log block (req=<id[:8]> in the title line),
            # on each /reports submission, and on the recent-transcriptions
            # store row for the /quick-config trace panel.

            # Persist the capture if eligibility passed at handler entry
            # AND duration falls in the configured window AND we have
            # enough disk free. Done BEFORE the log block so the block
            # can record `captured=<id_prefix>` for traceability. The
            # tmp_path is still on disk — the finally block unlinks it
            # AFTER this. We copy (not move) so the existing cleanup
            # path is unchanged.
            if will_capture:
                try:
                    import captures_store as _cap_store
                    audio_dur_s = float(getattr(info, "duration", 0.0) or 0.0)
                    min_s = float(getattr(cfg, "CAPTURE_RECORDINGS_MIN_DURATION_SEC", 0.5))
                    max_s = float(getattr(cfg, "CAPTURE_RECORDINGS_MAX_DURATION_SEC", 600.0))
                    if not raw_full_text.strip():
                        # Pure-silence clip: Whisper returned no speech, so the
                        # capture would store as "(empty)" with zero training
                        # value. Skip it. The tmp audio is unlinked by the outer
                        # finally, so nothing is orphaned. raw_full_text is the
                        # exact text that would be passed as raw= below.
                        logger.info(
                            "[capture] skipped empty transcription (no speech) req=%s",
                            request_id[:8],
                        )
                    elif min_s <= audio_dur_s <= max_s:
                        # Disk-free guard. Skip on <1 GB free; don't fail
                        # the transcription. Best-effort: a failure to
                        # query free space (e.g. inaccessible dir) is
                        # treated as "OK to try" and the create_capture
                        # path itself surfaces the real error.
                        try:
                            _free = shutil.disk_usage(cfg.CAPTURES_DIR).free
                        except OSError:
                            _free = 1 << 40  # large enough to proceed
                        if _free > 1_000_000_000:
                            captured_id = _cap_store.create_capture(
                                audio_src_path=tmp_path,
                                request_id=request_id,
                                model=resolved_model,
                                language=info.language,
                                duration_seconds=audio_dur_s,
                                raw=raw_full_text,
                                final=full_text_str,
                                text_for_training=training_text_str,
                                words=all_words,
                                segments=seg_diag,
                                user_id=user.get("user_id"),
                            )
                        else:
                            logger.warning(
                                "[capture] skipped due to low disk free "
                                "(%.1f MB free, need >1 GB)",
                                _free / (1024 * 1024),
                            )
                    else:
                        logger.info(
                            "[capture] skipped duration filter: %.1fs "
                            "(window %.1f-%.1f)",
                            audio_dur_s, min_s, max_s,
                        )
                except Exception as _ce:
                    logger.warning("[capture] persistence failed: %s", _ce)

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
                steps=trace,
                request_id=request_id,
                captured_id=captured_id,
            ))

            # Persist the trace to the durable recent-transcriptions store
            # (SQLite, WAL) and broadcast it to /quick-config SSE
            # subscribers in one step. Lazy import keeps main.py decoupled
            # at module-load time. metrics.record_transcription() in the
            # outer finally adds the timing half via UPSERT on the same
            # request_id.
            try:
                import quick_config_state
                quick_config_state.record_trace(
                    request_id=request_id,
                    model=resolved_model,
                    raw=raw_full_text,
                    steps=trace if trace is not None else [],
                    final=full_text_str,
                    language=info.language,
                    user_id=_user_id,
                )
            except Exception as _qc_err:
                logger.error("[quick-config] record_trace failed: %s", _qc_err)

            _audio_dur = float(info.duration)
            # Word count from the final post-processed text — matches what the
            # client actually receives. Counting len(all_words) instead would
            # yield 0 whenever WORD_TIMESTAMPS_ENABLED is off or the request
            # didn't ask for word-level granularity (the common case).
            _words = len(full_text_str.split())

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

    except Exception:
        # Catches failures BEFORE the inner try (e.g. _get_or_load_model
        # raising HTTPException, await request.form() blowing up), which
        # previously bypassed `_status = "error"` and inflated the success
        # counter with failed requests.
        _status = "error"
        raise
    finally:
        metrics.in_flight_transcriptions -= 1
        metrics.record_transcription(
            model=resolved_model,
            audio_dur=_audio_dur,
            proc_dur=time.perf_counter() - _t0,
            status=_status,
            words=_words,
            request_id=request_id,
            user_id=_user_id,
            key_id=_key_id,
        )


@app.get("/v1/models")
async def list_models():
    """OpenAI-style model listing — currently-loaded models plus the configured
    default. Useful for clients to discover what's available without trial."""
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
        "boot_id": BOOT_ID,
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
import io
from fastapi.responses import HTMLResponse, StreamingResponse

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


def _rotated_chain(active_path: str) -> list[str]:
    """Newest→oldest list of paths in the rotation chain: the active log
    followed by .1, .2, … up to LOG_BACKUP_COUNT. Files that don't exist
    are silently skipped — rotation may have produced fewer backups than
    the configured count, and a freshly-deployed service starts with
    just the active file."""
    out = [active_path]
    for i in range(1, int(getattr(cfg, "LOG_BACKUP_COUNT", 10)) + 1):
        p = f"{active_path}.{i}"
        if os.path.exists(p):
            out.append(p)
    return out


def _read_chain_window(active_path: str, skip: int, want: int) -> "tuple[list[str], int | None]":
    """Read up to `want` lines from the rotation chain (newest→oldest),
    starting `skip` lines back from the chain head. Returns
    (lines_oldest_first, next_skip).

    next_skip is None when the chain has no more older content — either
    we returned a partial page (fewer than `want` lines) or we exactly
    reached the chain tail. Otherwise next_skip == skip + len(lines)
    and the caller may re-call with that value to fetch the next
    older window.

    Walks the chain newest-file first (active log → .1 → .2 → …),
    reading each file backward in 8 KB blocks until we've accumulated
    `skip + want` lines across the chain. One file is held in memory
    at a time; ~10 MB worst case for the default LOG_MAX_BYTES."""
    target = skip + want
    # `collected` is built in oldest→newest order: each older file's
    # tail is prepended to the running list as we walk the chain
    # newest→oldest. By construction the OLDEST line in the chain
    # window we've seen so far sits at collected[0].
    collected: list[str] = []
    chain = _rotated_chain(active_path)
    # When we break out of the file loop because `target` is satisfied
    # without opening every file, older rotated files still on disk
    # remain unread — `exhausted` must account for that or the caller
    # ("Load older" UI) loses access to anything beyond the first file
    # whenever its line count meets `target` exactly.
    more_files_after_break = False
    for i, path in enumerate(chain):
        try:
            with open(path, "rb") as f:
                f.seek(0, io.SEEK_END)
                size = f.tell()
                block = 8192
                data = b""
                need = target - len(collected)
                while size > 0 and data.count(b"\n") <= need:
                    read = min(block, size)
                    size -= read
                    f.seek(size)
                    data = f.read(read) + data
        except OSError:
            continue
        collected = data.decode("utf-8", errors="replace").splitlines() + collected
        if len(collected) >= target:
            more_files_after_break = (i + 1) < len(chain)
            break
    # Slice in newest-first frame so `skip` is unambiguous.
    newest_first = list(reversed(collected))
    window = newest_first[skip:skip + want]
    exhausted = ((skip + len(window)) >= len(newest_first)
                 and not more_files_after_break)
    next_skip = None if exhausted else skip + len(window)
    return list(reversed(window)), next_skip


async def _stream_log_lines():
    """Yield SSE events: one for each existing tail line, then live tail."""
    initial = int(getattr(cfg, "LOG_VIEWER_INITIAL_LINES", 2000))
    backlog, _ = _read_chain_window(cfg.LOG_FILE, skip=0, want=initial)
    for line in backlog:
        yield f"data: {line}\n\n"

    # Sentinel — marks the boundary between backlog and the live poll
    # loop. The client's append() early-returns on this line; pill counts
    # are driven entirely by SEV_POLLER_JS against severity_counts().
    yield "data: __LIVE_TAIL__\n\n"

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
<title>{{HEADER_TITLE}}</title>
{{PAGE_META}}
{{SCALE_BOOTSTRAP_HEAD}}
<script>(function(){var v=localStorage.getItem('whisper-log-zoom');
  if(v)document.documentElement.style.setProperty('--log-zoom',v);})();</script>
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
     and silently dropping every CSS rule that follows. Header chrome
     (title, pills, buttons) gets --font-sans by default; the log lines
     themselves opt into --font-mono via .line so timestamps and tabular
     fields stay aligned. */
  html { height: 100%; }
  body { background: var(--bg); color: var(--fg);
    font: 1rem/1.5 var(--font-sans);
    margin: 0; padding: 0; min-height: 100%; }
  input, textarea, select, kbd, code, pre { font-family: var(--font-mono); }
  /* header / .header-inner / .title / page-toolbar controls (buttons,
     pills, the #filter input) are all centralized in NAV_CSS. */
  /* width:100% + box-sizing:border-box are the fix for the "tiny centered
     column with text clipped at the start" rendering — without them the
     container sits on a content-sized box that overflows the viewport.
     No max-width: log content uses the full viewport so long lines (HF
     URLs, model paths) sit on a single line on wide monitors instead of
     wrapping into the empty side-bands. The header bar stays centered at
     68.75rem (its own .header-inner cap) so controls remain in a predictable
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
  #load-older-row { display: flex; justify-content: center;
    padding: 0.5rem 0; border-bottom: 1px solid var(--border); }
  #loadOlderBtn { background: transparent; color: var(--cyan);
    border: 1px solid var(--border); padding: 0.375rem 1rem;
    border-radius: 4px; font: inherit; cursor: pointer; }
  #loadOlderBtn:hover:not([disabled]) { background: var(--panel); }
  #loadOlderBtn[disabled] { opacity: 0.5; cursor: default; }
  .tz-hint { color: var(--dim); font-size: var(--fs-xs); cursor: help; }
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
<header>
  <div class="header-inner">
    <span class="title">{{HEADER_BRAND}}</span>
    <span class="brand-sep" aria-hidden="true"></span>
    {{NAV}}
    <span class="spacer"></span>
    <span class="hdr-right">{{SEV_PILLS}}{{SCALE_PICKER}}{{RELOAD}}{{LOGOUT}}</span>
  </div>
  <div class="subbar">
    <span class="subbar-title">Logs</span>
    <div class="subbar-left">
      <input id="filter" type="text" placeholder="filter (case-insensitive substring)…">
    </div>
    <div class="subbar-right">
      <span class="log-zoom" title="zoom log content only">
        <button id="log-zoom-out" type="button" aria-label="decrease log size">−</button>
        <span id="log-zoom-pct">100%</span>
        <button id="log-zoom-in" type="button" aria-label="increase log size">+</button>
      </span>
      <span class="tz-hint" title="log timestamps are stored in UTC and shown here in your browser's local timezone">local time</span>
      <button id="pauseBtn">pause</button>
      <button id="clearBtn">clear</button>
      <span id="status" class="pill live">live</span>
    </div>
  </div>
</header>
<div id="load-older-row">
  <button id="loadOlderBtn" type="button" style="display:none;">Load older</button>
</div>
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
  function _p2(n) { return (n < 10 ? '0' : '') + n; }
  // Log lines start with a UTC ISO-8601 'Z' timestamp; show it in the reader's
  // local time as 'YYYY-MM-DD HH:MM:SS'. Lines that don't start with that token
  // (continuation/traceback lines, or pre-UTC rotated lines) pass through as-is.
  function localizeLogTs(line) {
    const m = line.match(/^(\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z)\\s/);
    if (!m) return line;
    const d = new Date(m[1]);
    if (isNaN(d.getTime())) return line;
    const local = d.getFullYear() + '-' + _p2(d.getMonth() + 1) + '-' + _p2(d.getDate())
      + ' ' + _p2(d.getHours()) + ':' + _p2(d.getMinutes()) + ':' + _p2(d.getSeconds());
    return local + line.slice(m[1].length);
  }
  // DOM cap for live-tail appends only. The "Load older" path bypasses
  // this so the user can scroll back through arbitrarily many rotated
  // lines without their click silently dropping the freshest content.
  const _LOG_DOM_MAX = {{LOG_VIEWER_DOM_MAX}};
  function append(line) {
    // __LIVE_TAIL__ sentinel marks the boundary between backlog and the
    // live poll loop. After it fires we know the freshest page is in the
    // DOM and the "Load older" button can become active.
    if (line === '__LIVE_TAIL__') {
      const lo = document.getElementById('loadOlderBtn');
      if (lo) lo.style.display = '';
      return;
    }
    const el = document.createElement('span');
    const cls = classify(line);
    el.className = 'line ' + cls;
    el.textContent = localizeLogTs(line) + '\\n';
    applyFilter(el);
    log.appendChild(el);
    while (log.childElementCount > _LOG_DOM_MAX) log.firstChild.remove();
    if (!paused) window.scrollTo(0, document.body.scrollHeight);
  }

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
  clearBtn.addEventListener('click', () => {
    log.innerHTML = '';
    // Resetting the DOM after Clear: the next live append re-fills
    // from the bottom, but the older-chain pointer is unchanged so
    // the user can still walk back if they want.
  });

  // "Load older" cursor: how many lines from the chain head are already
  // in the DOM. Seeded with the initial backlog size; each successful
  // /logs/older response bumps it by the returned-batch length.
  let _logsSkip = {{LOG_VIEWER_INITIAL_LINES}};
  let _logsOlderBusy = false;
  const loadOlderBtn = document.getElementById('loadOlderBtn');
  if (loadOlderBtn) {
    loadOlderBtn.addEventListener('click', async () => {
      if (_logsOlderBusy) return;
      _logsOlderBusy = true;
      loadOlderBtn.disabled = true;
      const prevLabel = loadOlderBtn.textContent;
      loadOlderBtn.textContent = 'Loading…';
      try {
        // Session cookie is sent automatically with the fetch.
        const url = '/logs/older?skip=' + encodeURIComponent(_logsSkip)
                  + '&limit={{LOG_VIEWER_INITIAL_LINES}}';
        const r = await fetch(url);
        if (!r.ok) {
          console.warn('load-older failed', r.status);
          return;
        }
        const j = await r.json();
        const lines = (j && j.lines) || [];
        // Prepend to top of #log, preserving line order (oldest-first
        // batch → first inserted ends up at the very top, last inserted
        // sits just above the existing content). Filter is reapplied per
        // line so the new batch honors any active substring search.
        const frag = document.createDocumentFragment();
        for (const line of lines) {
          const el = document.createElement('span');
          el.className = 'line ' + classify(line);
          el.textContent = localizeLogTs(line) + '\\n';
          applyFilter(el);
          frag.appendChild(el);
        }
        log.insertBefore(frag, log.firstChild);
        _logsSkip += lines.length;
        if (j.next_skip == null) loadOlderBtn.style.display = 'none';
      } catch (e) {
        console.warn('load-older error', e);
      } finally {
        _logsOlderBusy = false;
        loadOlderBtn.disabled = false;
        loadOlderBtn.textContent = prevLabel;
      }
    });
  }

  // EventSource sends the HttpOnly session cookie automatically (same-origin),
  // so the server's _require_logs_page_sse dependency resolves the user
  // without the legacy ?key= fallback.
  let es = null;
  let _logRecoveryTimer = null;
  function openLogStream() {
    if (es) { try { es.close(); } catch (_) {} es = null; }
    es = new EventSource('/logs/stream');
    es.onmessage = (e) => append(e.data);
    es.onerror = () => {
      statusEl.textContent = 'reconnecting…';
      statusEl.className = 'pill paused';
      // EventSource does NOT auto-reconnect after an HTTP error (e.g. an
      // intermittent 401 where the cookie wasn't attached to the SSE
      // handshake). Mirror /stats: poll a cheap endpoint until it 200s,
      // then reopen the stream.
      if (_logRecoveryTimer) return;
      _logRecoveryTimer = setInterval(async () => {
        try {
          const r = await fetch('/v1/models', { cache: 'no-store' });
          if (r.ok) {
            clearInterval(_logRecoveryTimer);
            _logRecoveryTimer = null;
            openLogStream();
          }
        } catch (_) { /* keep polling */ }
      }, 3000);
    };
    es.onopen = () => {
      // role-admin used to be added here unconditionally — that leaked
      // admin chrome to non-admins. OPEN_MODE_BANNER_JS is now the single
      // source of truth (sets role-admin iff whoami.is_admin=true).
      if (_logRecoveryTimer) { clearInterval(_logRecoveryTimer); _logRecoveryTimer = null; }
      if (!paused) {
        statusEl.textContent = 'live';
        statusEl.className = 'pill live';
      }
    };
  }
  openLogStream();

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
{{SEV_POLLER_JS}}
</body></html>"""


def _require_logs_page_sse(request: Request) -> dict:
    """SSE-aware variant of require_page("logs"). EventSource can't set
    Authorization, so accept `?key=<raw_key>` as a fallback. In OPEN
    mode (no admin key yet) the synthetic admin sails through; in
    locked-down mode the bearer must resolve to a user with scope(
    "logs") == "all" — the log file isn't user-partitionable (a single
    request block carries every user's transcripts, filenames, and
    final text via _format_request_block), so "own" can't be enforced
    line-by-line and is rejected as access-only at the schema layer."""
    import api_keys_store
    if not api_keys_store.is_locked_down():
        return dict(api_keys_store.OPEN_MODE_USER)
    auth_header = request.headers.get("authorization") or ""
    raw = ""
    if auth_header.lower().startswith("bearer "):
        raw = auth_header.split(" ", 1)[1].strip()
    rec = api_keys_store.lookup_by_raw_key(raw) if raw else None
    if rec is None:
        rec = _user_from_session_cookie(request)
    if rec is None:
        key = request.query_params.get("key") or ""
        rec = api_keys_store.lookup_by_raw_key(key) if key else None
    if rec is None:
        raise HTTPException(
            401, "invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    perms = Permissions(
        rec.get("permissions_raw") or {}, bool(rec.get("is_admin")),
    )
    # scope("logs") (not can("logs")) — defends against legacy DB rows
    # still storing "own" from before logs joined ACCESS_ONLY_PAGES.
    if perms.scope("logs") != "all":
        raise HTTPException(403, "no access to /logs")
    return rec


@app.get("/logs", response_class=HTMLResponse)
async def logs_viewer():
    # HTML page is open — the bearer isn't available on initial
    # navigation. The SSE /logs/stream endpoint gates by
    # `require_page("logs")` with a ?key= fallback for EventSource;
    # the page's first stream-open 403s for non-permitted users.
    import web_common
    return HTMLResponse(
        web_common.render_page(_LOG_VIEWER_HTML, current="logs"),
        headers={"Cache-Control": "no-store"},
    )


@app.get(
    "/logs/stream",
    dependencies=[Depends(_require_logs_page_sse)],
)
async def logs_stream():
    return StreamingResponse(_stream_log_lines(), media_type="text/event-stream")


@app.get(
    "/logs/older",
    dependencies=[Depends(_require_logs_page_sse)],
)
async def logs_older(skip: int = 0, limit: int = 0):
    """Fetch the next older page from the rotation chain. `skip` is the
    number of lines from the chain head that have already been loaded
    into the browser DOM; `limit` defaults to LOG_VIEWER_INITIAL_LINES
    and is server-clamped to the same value (per-click max page size).

    Response: `{lines: [...], next_skip: <int|null>}`. lines are
    oldest-first so the client can prepend them to the top of the
    log container as a contiguous older window. next_skip=null means
    the rotation chain is exhausted — the client hides the button."""
    initial = int(getattr(cfg, "LOG_VIEWER_INITIAL_LINES", 2000))
    want = max(1, min(limit or initial, initial))
    skip = max(0, int(skip))
    lines, next_skip = _read_chain_window(cfg.LOG_FILE, skip=skip, want=want)
    return {"lines": lines, "next_skip": next_skip}


@app.get("/auth/whoami")
async def whoami(
    request: Request,
    user: dict = Depends(_get_current_user_dep),
):
    """Resolve the caller to a user payload the WebUI uses to render the
    login modal + user-aware chrome.

    Returns `{open_mode, user_id, username, is_admin, permissions,
    csrf_token?}`. The `permissions` object is `{pages: {logs:
    'own'|'all'|'none', ...}}` — used by each page's JS to hide nav links
    the user can't reach and to render scope hints. `csrf_token` is
    present only for cookie-authenticated callers (set by
    user_from_session_cookie on request.state) so the client can attach
    X-CSRF-Token without parsing the cookie. A 401 means no valid
    credential AND the server is locked down — the WebUI re-prompts."""
    import api_keys_store as _ak
    perms = user.get("permissions")
    out = {
        "open_mode": not _ak.is_locked_down(),
        "user_id": user.get("user_id"),
        "username": user.get("username"),
        "is_admin": bool(user.get("is_admin")),
        "permissions": perms.to_dict() if perms is not None else {"pages": {}},
    }
    csrf = getattr(request.state, "session_csrf", None)
    if csrf:
        out["csrf_token"] = csrf
    return out


@app.post("/auth/login")
async def login(request: Request, response: Response):
    """Exchange a pasted API key for an HttpOnly session cookie.

    Open mode → no-op (everyone is already the synthetic admin). Locked
    down → validate the key via api_keys_store, create a server-side
    session, and set two cookies: the HttpOnly session token and a
    JS-readable CSRF token (double-submit). Returns the same shape as
    /auth/whoami so the client can populate chrome without a second
    round-trip. CSRF-exempt (no session exists yet)."""
    import api_keys_store as _ak
    import sessions_store
    if not _ak.is_locked_down():
        return {"open_mode": True}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — malformed/empty body → treat as no key
        body = {}
    key = body.get("key") if isinstance(body, dict) else None
    rec = _ak.lookup_by_raw_key(key or "")
    if rec is None:
        raise HTTPException(
            401, "invalid API key", headers={"WWW-Authenticate": "Bearer"},
        )
    raw_token, csrf_token = sessions_store.create_session(
        rec["user_id"], cfg.SESSION_TTL_SECONDS,
    )
    ttl = int(cfg.SESSION_TTL_SECONDS)
    secure = bool(cfg.SESSION_COOKIE_SECURE)
    response.set_cookie(
        cfg.SESSION_COOKIE_NAME, raw_token, max_age=ttl,
        httponly=True, samesite="lax", secure=secure, path="/",
    )
    response.set_cookie(
        cfg.SESSION_CSRF_COOKIE_NAME, csrf_token, max_age=ttl,
        httponly=False, samesite="lax", secure=secure, path="/",
    )
    perms = Permissions(rec.get("permissions_raw") or {}, bool(rec.get("is_admin")))
    return {
        "open_mode": False,
        "csrf_token": csrf_token,
        "user_id": rec.get("user_id"),
        "username": rec.get("username"),
        "is_admin": bool(rec.get("is_admin")),
        "permissions": perms.to_dict(),
    }


@app.post("/auth/logout")
async def logout(request: Request, response: Response):
    """Revoke the current session and clear its cookies. CSRF-protected
    like any other cookie-authenticated mutation (the WebUI sends the
    X-CSRF-Token header)."""
    import sessions_store
    raw = request.cookies.get(cfg.SESSION_COOKIE_NAME, "")
    if raw:
        sessions_store.revoke_session(raw)
    response.delete_cookie(cfg.SESSION_COOKIE_NAME, path="/")
    response.delete_cookie(cfg.SESSION_CSRF_COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/sev")
async def severity_snapshot():
    """Tiny JSON endpoint polled by every page's nav-row pill poller.

    Returns the same `severity_counts()` the server uses everywhere else
    (nav HTML render, /stats payload) — WARNING+ records since process
    start, bounded by the 2000-entry ring. Wide-open like /logs — three
    integers, no PII. The poller in web_common.SEV_POLLER_JS hits this
    every 5 s on /logs, /stats, and /settings so all three pages stay
    synced to the server-side truth."""
    import web_common
    return web_common.severity_counts()


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
# /settings - admin WebUI (opt-in)
# =============================================================================
# Off by default: registered only when cfg.ADMIN_UI_ENABLED is True (set in
# config.py or via WHISPER_ADMIN_UI=1). Auth on the endpoints themselves is
# per-user API keys (require_admin) layered on top of cfg.ADMIN_ALLOWED_HOSTS.
# In OPEN mode (no admin key in DB) every caller is the synthetic admin so the
# operator can bootstrap.
if cfg.ADMIN_UI_ENABLED:
    try:
        from admin_routes import router as _admin_router
        app.include_router(_admin_router)
        # /settings/api-keys — admin UI for per-user key management. Same
        # auth shape (admin host + admin key) as /settings.
        from api_keys_routes import router as _api_keys_router
        app.include_router(_api_keys_router)
        logger.info(
            "Admin UI enabled at /settings (allowlist=%s; auth=API key)",
            cfg.ADMIN_ALLOWED_HOSTS,
        )
        # /quick-config piggybacks on the admin UI: same allowlist, same
        # per-user API key auth.
        from quick_config_routes import router as _quick_router
        app.include_router(_quick_router)
        logger.info("Quick-config UI enabled at /quick-config")
        # /reports: admin-only triage page for user-submitted transcription
        # error reports. The submission endpoint /quick-config/reports/api/submit
        # lives on the same router and accepts any active API key.
        from reports_routes import router as _reports_router
        app.include_router(_reports_router)
        logger.info(
            "Reports UI enabled at /reports (admin key required for triage; "
            "user submissions %s)",
            "enabled" if getattr(cfg, "REPORTS_ALLOW_USER_SUBMIT", True)
            else "disabled",
        )
        # /captures: admin-only Whisper fine-tuning data capture + review.
        # Master switch is cfg.CAPTURE_RECORDINGS_ENABLED — the page is
        # always registered so the admin can browse existing rows even
        # after disabling new capture.
        from captures_routes import router as _captures_router
        app.include_router(_captures_router)
        logger.info(
            "Captures UI enabled at /captures (admin token required; "
            "new capture %s)",
            "enabled" if getattr(cfg, "CAPTURE_RECORDINGS_ENABLED", False)
            else "disabled",
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
