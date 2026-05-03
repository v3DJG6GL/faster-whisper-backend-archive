import os
import sys
import ctypes
import logging
import logging.handlers
import re
import string
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
# Text post-processing pipeline (Swiss-German / CH-DE)
# =============================================================================
# Whisper's raw output looks like "Hallo Komma der Patient zeigt Besserung
# Punkt." — a mix of the model's own punctuation and spoken-out symbol words.
# Whisper emits standard German orthography (with ß), but our target is Swiss
# German, which uses "ss" instead. `_postprocess_text()` reshapes the output
# in ten ordered steps:
#
#   0. REPLACE        Apply cfg.CHARACTER_REPLACEMENTS (ordered str.replace
#                     pairs). Default rules are ß → ss / ẞ → SS (Swiss
#                     orthography). User-extensible for other 1→N character
#                     substitutions.
#   1. STRIP          Remove most punctuation, keep date/time/number separators.
#   2. NORMALIZE      Turn "10-23" into "10/23" so number ranges aren't broken.
#   3. STRIP TERMS    Drop Whisper-emitted "."/"?"/"!" at audio pauses, AND
#                     lowercase the following word if it's a known non-noun
#                     (interrogatives, conjunctions, articles, etc.) so
#                     "Hallo. Wie geht's" → "Hallo wie geht's". Keeps dots
#                     inside numbers (10.23, 11.).
#   4. STRIP COMMAS   In dictation mode, the user controls all commas via
#                     "Komma". Drop Whisper-emitted soft-pause commas now —
#                     except those between digits ("1,000"), which we keep.
#   5. DICTATION MAP  Replace German words with symbols ("Komma" → ",").
#   6. TIDY SPACING   Remove stray spaces around the inserted punctuation.
#   7. DEDUP PUNCT    Collapse runs of adjacent punctuation. Whisper emits its
#                     own commas around dictation keywords ("..., Punkt."),
#                     which after substitution leaves "...,." — keep only the
#                     dictation-emitted (user-intended) mark.
#   8. TIDY NEWLINES  Drop Whisper-emitted commas / stray whitespace around the
#                     newlines inserted by "neue Zeile" / "neuer Absatz".
#   9. CAPITALIZE     Capitalize the first letter after a dictation-emitted
#                     ".", "?", "!", or newline. Whisper transcribes audio
#                     pauses; it doesn't know "Punkt" will become a sentence
#                     end, so it leaves the next word lowercase.
#
# Each step's regex/lookup is precompiled at module load (it would otherwise be
# rebuilt on every transcription request).
# -----------------------------------------------------------------------------

# --- Step 0: ordered character replacements ---------------------------------
# Source data lives in cfg.CHARACTER_REPLACEMENTS. (str.translate can't do
# 1->N char mapping, hence the tuple-of-replace pattern.)

# --- Step 1: punctuation strip -----------------------------------------------
# Source: cfg.PUNCTUATION_TO_KEEP. Anything else from string.punctuation is
# removed; we also strip German low + high quotes („ ") since Whisper emits
# them inconsistently.
# Built once below in rebuild_caches(); admin-WebUI edits to cfg.PUNCTUATION_TO_KEEP
# or cfg.DICTATION_MAP call rebuild_caches() to refresh.
_PUNCTUATION_TO_REMOVE: str = ""
_PUNCTUATION_STRIP_TABLE: dict = {}

# --- Step 2: number-range normalization --------------------------------------
# Whisper writes ranges as "10-23"; downstream consumers (vowen.ai) want "10/23".
_NUMBER_RANGE_HYPHEN_PATTERN = re.compile(r"(\d)\s*-\s*(\d)")

# --- Step 3: strip Whisper sentence punctuation + lowercase non-nouns -------
# Source data: cfg.LOWERCASE_AFTER_STRIPPED_TERMINATOR (the whitelist of
# German non-noun words that may be lowercased mid-sentence after stripping
# a Whisper-emitted terminator).
#
# Whisper terminator (`.` `?` `!`) NOT preceded by digit, optionally followed
# by whitespace + a capitalized word. We strip the terminator and conditionally
# lowercase the captured word.
_WHISPER_TERMINATOR_AND_NEXT = re.compile(r"(?<!\d)[.?!](\s*)([A-ZÄÖÜ])(\w*)")
# Catches lone .?! (end of text, before non-letter) that the above didn't handle.
_WHISPER_TERMINATOR = re.compile(r"(?<!\d)[.?!]")

# --- Step 4: German dictation map --------------------------------------------
# Source data: cfg.DICTATION_MAP (spoken word -> literal symbol). Multi-word
# phrases must be matched before their single-word components, which is why
# we sort by length (longest-first) when building the alternation regex.

# Regex that matches any cfg.DICTATION_MAP key, longest-first so multi-word
# phrases beat their single-word prefixes. Built in rebuild_caches() below.
_DICTATION_REGEX: "re.Pattern[str]" = re.compile(r"(?!x)x")  # placeholder, never matches
# Lowercase index of the same map, used to look up the replacement after a
# case-insensitive match (e.g. "PUNKT" or "punkt" both resolve to ".").
_DICTATION_LOWERCASE_LOOKUP: dict[str, str] = {}


def rebuild_caches() -> None:
    """Rebuild module-level caches that are derived from cfg.* values.

    Called once at module load (just below) and again by the admin WebUI
    after a config change to DICTATION_MAP or PUNCTUATION_TO_KEEP. The other
    cfg references in this file are read live per-request, so they don't
    need a rebuild step.
    """
    global _DICTATION_REGEX, _DICTATION_LOWERCASE_LOOKUP
    global _PUNCTUATION_TO_REMOVE, _PUNCTUATION_STRIP_TABLE

    _DICTATION_REGEX = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted(cfg.DICTATION_MAP, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    _DICTATION_LOWERCASE_LOOKUP = {k.lower(): v for k, v in cfg.DICTATION_MAP.items()}
    _PUNCTUATION_TO_REMOVE = (
        "".join(c for c in string.punctuation if c not in cfg.PUNCTUATION_TO_KEEP) + "„"
    )
    _PUNCTUATION_STRIP_TABLE = str.maketrans("", "", _PUNCTUATION_TO_REMOVE)


rebuild_caches()

# --- Step 5: whitespace tidy -------------------------------------------------
# After dictation substitution, output looks like "Müller , der" or "( siehe".
# These two patterns collapse the stray spaces. We use [ \t] (not \s) so the
# "\n\n" emitted by "neuer Absatz" → paragraph break is left intact.
_SPACE_BEFORE_CLOSING_PUNCT = re.compile(r"[ \t]+([,.:;!?\)\]\}])")
_SPACE_AFTER_OPENING_PUNCT = re.compile(r"([\(\[\{])[ \t]+")

# --- Step 6: deduplicate adjacent punctuation --------------------------------
# Whisper emits its own commas as soft pauses around dictation keywords. After
# substitution we end up with runs like ",." (Punkt), ",;" (Semikolon), ",:,"
# (Doppelpunkt). The user's intended mark wins: prefer any non-comma in the
# run, and within non-commas prefer the LAST one (dictation came after the
# Whisper pause). Pure commas collapse to a single comma.
_PUNCTUATION_RUN_PATTERN = re.compile(r"[,.:;!?]{2,}")

# --- Step 4: strip Whisper noise commas (dictation mode only) ---------------
# In dictation mode the user explicitly says "Komma" when they want a comma,
# so anything else is a Whisper soft-pause comma we should drop. We keep
# commas BETWEEN digits ("1,000") in case numerical content shows up.
_NOISE_COMMA_PATTERN = re.compile(r"(?<!\d),|,(?!\d)")

# --- Step 8: tidy newline neighborhood ---------------------------------------
# Around a dictation-emitted "\n" / "\n\n" we often have residue: a Whisper-
# emitted comma ("Müller, \n"), trailing space (" \n"), or punctuation that
# leaked through ("\n , Welt"). Collapse the whole neighborhood — optional
# whitespace + optional comma — on each side of the newline(s) into nothing.
# This both removes the noise AND gives clean line starts (no leading space
# on the next line).
_NEWLINE_NEIGHBORHOOD_PATTERN = re.compile(r"[ \t]*,?[ \t]*(\n+)[ \t]*,?[ \t]*")

# --- Step 9: capitalize after sentence-ending punctuation -------------------
# After our dictation map inserts ".", "?", "!", "\n", "\n\n", the following
# letter should start a new sentence/paragraph in proper case. Whisper has no
# idea about our substitution, so it leaves it lowercase.
_CAPITALIZE_AFTER_SENTENCE_PATTERN = re.compile(r"([.?!]\s+|\n+\s*)([a-zäöüß])")


def _collapse_punctuation_run(match: "re.Match[str]") -> str:
    run = match.group(0)
    non_comma = [c for c in run if c != ","]
    return non_comma[-1] if non_comma else ","


def _apply_replacements(text: str) -> str:
    for src, dst in cfg.CHARACTER_REPLACEMENTS:
        text = text.replace(src, dst)
    return text


def _apply_dictation(text: str) -> str:
    return _DICTATION_REGEX.sub(lambda m: _DICTATION_LOWERCASE_LOOKUP[m.group(1).lower()], text)


def _tidy_spacing(text: str) -> str:
    text = _SPACE_BEFORE_CLOSING_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPENING_PUNCT.sub(r"\1", text)
    return text


def _strip_whisper_terminators(text: str) -> str:
    """Step 3: strip Whisper-emitted .?! and lowercase the next word if it's
    a known non-noun. Done before the dictation map so user-emitted punctuation
    (from "Punkt"/"Fragezeichen"/etc.) is the only sentence punctuation left."""
    def replace(m: "re.Match[str]") -> str:
        ws, first, rest = m.group(1), m.group(2), m.group(3)
        if (first + rest).lower() in cfg.LOWERCASE_AFTER_STRIPPED_TERMINATOR:
            return ws + first.lower() + rest
        return ws + first + rest
    text = _WHISPER_TERMINATOR_AND_NEXT.sub(replace, text)
    text = _WHISPER_TERMINATOR.sub("", text)
    return text


def _capitalize_after_sentence(text: str) -> str:
    text = _CAPITALIZE_AFTER_SENTENCE_PATTERN.sub(
        lambda m: m.group(1) + m.group(2).upper(), text
    )
    # Also capitalize the very first letter (Whisper usually does, but VAD
    # splits or mid-utterance starts can produce a stray lowercase).
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _postprocess_text(text: str, trace: "list | None" = None) -> str:
    """Run the 10-step pipeline above on a single piece of Whisper output.

    If `trace` is provided (a list), each step that *changes* the text appends
    a `(step_name, before, after)` tuple — the request handler uses this to
    log a fancy diff view of what each step did.
    """
    def step(name: str, transform):
        nonlocal text
        before = text
        text = transform(before)
        if trace is not None and before != text:
            trace.append((name, before, text))

    step("0 REPLACE",        _apply_replacements)
    step("1 STRIP",          lambda t: t.translate(_PUNCTUATION_STRIP_TABLE))
    step("2 NORMALIZE",      lambda t: _NUMBER_RANGE_HYPHEN_PATTERN.sub(r"\1/\2", t))
    step("3 STRIP TERMS",    _strip_whisper_terminators)
    if cfg.DICTATION_ENABLED:
        step("4 STRIP COMMAS",   lambda t: _NOISE_COMMA_PATTERN.sub("", t))
        step("5 DICTATION",      _apply_dictation)
        step("6 TIDY SPACING",   _tidy_spacing)
        step("7 DEDUP PUNCT",    lambda t: _PUNCTUATION_RUN_PATTERN.sub(_collapse_punctuation_run, t))
        step("8 TIDY NEWLINES",  lambda t: _NEWLINE_NEIGHBORHOOD_PATTERN.sub(r"\1", t))
        step("9 CAPITALIZE",     _capitalize_after_sentence)
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
        loaded_device = cfg.MODEL_DEVICE
        loaded_compute = cfg.MODEL_COMPUTE_TYPE
        try:
            new_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(name, device=cfg.MODEL_DEVICE, compute_type=cfg.MODEL_COMPUTE_TYPE),
            )
            logger.info("Model loaded on %s: %s", cfg.MODEL_DEVICE, name)
        except Exception as e:
            logger.error("%s load failed for %s, falling back to %s: %s",
                         cfg.MODEL_DEVICE, name, cfg.MODEL_DEVICE_FALLBACK, e)
            new_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(name, device=cfg.MODEL_DEVICE_FALLBACK, compute_type=cfg.MODEL_COMPUTE_TYPE_FALLBACK),
            )
            loaded_device = cfg.MODEL_DEVICE_FALLBACK
            loaded_compute = cfg.MODEL_COMPUTE_TYPE_FALLBACK
            logger.info("Model loaded on %s: %s", cfg.MODEL_DEVICE_FALLBACK, name)

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

    yield

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

            # word_timestamps: AND of the global config knob and the per-request
            # ask. Disabled globally (cfg.WORD_TIMESTAMPS_ENABLED=False) bypasses
            # the DTW alignment path entirely — required for primeline-style
            # finetunes that hit faster-whisper#1212.
            want_word_ts = cfg.WORD_TIMESTAMPS_ENABLED and include_words

            # Empty string is NOT equivalent to None for tnfru / primeline
            # finetunes — passing "" to model.transcribe(initial_prompt=...)
            # triggers the failure mode their model card warns about. Coerce.
            _prompt = prompt or cfg.DEFAULT_PROMPT
            initial_prompt_arg = _prompt if _prompt else None

            vad_parameters = dict(
                min_silence_duration_ms=cfg.VAD_MIN_SILENCE_MS,
                speech_pad_ms=cfg.VAD_SPEECH_PAD_MS,
                threshold=cfg.VAD_THRESHOLD,
            ) if cfg.VAD_FILTER else None

            transcribe_kwargs = dict(
                language=language or cfg.DEFAULT_LANGUAGE,
                beam_size=cfg.BEAM_SIZE,
                best_of=cfg.BEST_OF,
                temperature=temperature,
                vad_filter=cfg.VAD_FILTER,
                vad_parameters=vad_parameters,
                word_timestamps=want_word_ts,
                condition_on_previous_text=cfg.CONDITION_ON_PREVIOUS_TEXT,
                initial_prompt=initial_prompt_arg,
                no_speech_threshold=cfg.NO_SPEECH_THRESHOLD,
                log_prob_threshold=cfg.LOG_PROB_THRESHOLD,
                compression_ratio_threshold=cfg.COMPRESSION_RATIO_THRESHOLD,
            )
            segments_iter, info = model.transcribe(tmp_path, **transcribe_kwargs)

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
                clean_segment_text = _postprocess_text(segment.text)

                # segment.temperature reflects CT2's actual after-fallback
                # value (may differ from the request `temperature` if fallback
                # kicked in). segment.compression_ratio is the real gzip ratio
                # used by the suppression check — was previously hardcoded 1.0.
                seg_temp = getattr(segment, "temperature", temperature)
                seg_cr = getattr(segment, "compression_ratio", 1.0)

                segments_list.append({
                    "id": i,
                    "seek": 0,
                    "start": segment.start,
                    "end": segment.end,
                    "text": clean_segment_text,
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
                            "word": _postprocess_text(word.word),
                            "start": word.start,
                            "end": word.end,
                        })

            # Strip leading whitespace fully (no leading paragraph break before
            # the first character ever makes sense) but only trim trailing
            # spaces/tabs — preserve a trailing "\n" emitted by "neue Zeile" /
            # "neuer Absatz" at the end of the dictation, since the user
            # explicitly asked for it.
            raw_full_text = "".join(raw_full_text_parts)
            trace: "list | None" = [] if cfg.TRACE_ENABLED else None
            full_text_str = _postprocess_text(raw_full_text, trace=trace)
            before_trim = full_text_str
            full_text_str = full_text_str.lstrip().rstrip(" \t\r")
            if trace is not None and before_trim != full_text_str:
                trace.append(("10 TRIM EDGES", before_trim, full_text_str))

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
<title>faster-whisper-backend · live logs</title>
<style>
  :root {
    --bg: #0d1117; --fg: #c9d1d9; --dim: #6e7681;
    --cyan: #79c0ff; --green: #7ee787; --yellow: #f2cc60;
    --red: #ff7b72; --magenta: #d2a8ff; --bold: #f0f6fc;
  }
  html { height: 100%; }
  body { background: var(--bg); color: var(--fg);
    font: 13px/1.45 ui-monospace, "Cascadia Code", Menlo, Consolas, monospace;
    margin: 0; padding: 0; min-height: 100%; }
  header { position: sticky; top: 0; background: #161b22; border-bottom: 1px solid #30363d;
    padding: 8px 14px; display: flex; gap: 12px; align-items: center; z-index: 10; }
  header .title { font-weight: 600; color: var(--bold); }
  header .pill { padding: 2px 8px; border-radius: 999px; background: #21262d; color: var(--dim);
    font-size: 11px; }
  header .pill.live { color: var(--green); border: 1px solid #1f4d2a; }
  header .pill.paused { color: var(--yellow); border: 1px solid #4d3e1f; }
  header input { flex: 1; background: #0d1117; color: var(--fg); border: 1px solid #30363d;
    padding: 4px 8px; border-radius: 4px; font: inherit; }
  header button { background: #21262d; color: var(--fg); border: 1px solid #30363d;
    padding: 4px 10px; border-radius: 4px; cursor: pointer; font: inherit; }
  header button:hover { background: #30363d; }
  #log { padding: 8px 14px; white-space: pre; overflow-anchor: none; }
  .line { display: block; }
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
  <span class="title">faster-whisper-backend · logs</span>
  {{NAV}}
  <input id="filter" type="text" placeholder="filter (case-insensitive substring)…">
  <button id="pauseBtn">pause</button>
  <button id="clearBtn">clear</button>
  <span id="status" class="pill live">live</span>
</header>
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
</script>
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
