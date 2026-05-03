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

from pydantic import BaseModel, Field, ValidationError, field_validator


_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
OVERRIDES_PATH = os.path.join(_REPO_DIR, "config.local.json")


# Map AdminConfig field name -> env var that pins it. Mirrors the override
# block at the bottom of config.py. Used by the WebUI to mark fields as
# "currently overridden by WHISPER_X" with a badge.
ENV_VAR_MAPPING: dict[str, str] = {
    "DEFAULT_MODEL": "WHISPER_DEFAULT_MODEL",
    "ALLOWED_MODELS": "WHISPER_ALLOWED_MODELS",
    "MAX_LOADED_MODELS": "WHISPER_MAX_LOADED_MODELS",
    "PRELOAD_MODELS": "WHISPER_PRELOAD_MODELS",
    "DEFAULT_PROMPT": "WHISPER_DEFAULT_PROMPT",
    "DICTATION_ENABLED": "WHISPER_DICTATION_MAP",
    "TRACE_ENABLED": "WHISPER_TRACE",
    "LOG_FILE": "WHISPER_LOG_FILE",
    "ADMIN_ALLOWED_HOSTS": "WHISPER_ADMIN_ALLOWED_HOSTS",
    "STATS_ALLOWED_HOSTS": "WHISPER_STATS_ALLOWED_HOSTS",
}

# Cold settings — editing these requires a service restart for the new value
# to take effect. The WebUI shows a 'restart' badge and offers to trigger a
# self-restart after save.
RESTART_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "SERVER_HOST", "SERVER_PORT", "SERVER_WORKERS", "SERVER_LOG_LEVEL",
    "LOG_FILE", "LOG_MAX_BYTES", "LOG_BACKUP_COUNT",
    "PRELOAD_MODELS",
    "MODEL_DEVICE", "MODEL_COMPUTE_TYPE",
    "MODEL_DEVICE_FALLBACK", "MODEL_COMPUTE_TYPE_FALLBACK",
})

# Hot settings whose derived caches need rebuild after edit. The admin route
# calls main.rebuild_caches() when any of these change.
CACHE_REBUILD_FIELDS: frozenset[str] = frozenset({
    "DICTATION_MAP", "PUNCTUATION_TO_KEEP",
})

# faster-whisper short name OR HuggingFace repo id (org/name).
_MODEL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.\-]*(/[A-Za-z0-9_.\-]+)?$"
ModelId = Annotated[str, Field(min_length=1, max_length=96, pattern=_MODEL_ID_PATTERN)]

# Loose pattern for dictation-map keys: letters, digits, spaces, basic
# punctuation, and German diacritics. Capped length keeps the regex small.
# Note: main.py calls re.escape() on every key before regex compile, so the
# pattern here is for hygiene (catching typos) rather than injection defense.
_DICTATION_KEY_PATTERN = r"^[\w \-.,!?ßẞÄÖÜäöü]{1,64}$"
DictKey = Annotated[str, Field(min_length=1, max_length=64, pattern=_DICTATION_KEY_PATTERN)]
DictVal = Annotated[str, Field(max_length=8)]

LogLevel = Literal["debug", "info", "warning", "error", "critical"]
DeviceLit = Literal["cuda", "cpu"]
ComputeLit = Literal["float16", "int8_float16", "int8", "float32", "bfloat16"]


class AdminConfig(BaseModel):
    """Pydantic schema for config.local.json. Every field is Optional; absent
    means "do not override". Bounds and patterns enforce resource caps and
    cheap input hygiene at validation time."""

    # `protected_namespaces=()` disables Pydantic's "model_*" reserved-prefix
    # warning so we can use MODEL_DEVICE / MODEL_COMPUTE_TYPE field names.
    model_config = {"extra": "forbid", "protected_namespaces": ()}

    # --- Models ---
    DEFAULT_MODEL: ModelId | None = None
    # Sets serialize as JSON arrays; convert back on load. List type here lets
    # us validate per-element via the ModelId Annotated type.
    ALLOWED_MODELS: list[ModelId] | None = None
    MAX_LOADED_MODELS: Annotated[int, Field(ge=1, le=8)] | None = None
    PRELOAD_MODELS: list[ModelId] | None = None
    MODEL_DEVICE: DeviceLit | None = None
    MODEL_COMPUTE_TYPE: ComputeLit | None = None
    MODEL_DEVICE_FALLBACK: DeviceLit | None = None
    MODEL_COMPUTE_TYPE_FALLBACK: ComputeLit | None = None

    # --- Locale ---
    DEFAULT_LANGUAGE: Annotated[str, Field(pattern=r"^[a-z]{2}$")] | None = None
    DEFAULT_PROMPT: Annotated[str, Field(max_length=2048)] | None = None
    CHARACTER_REPLACEMENTS: list[tuple[
        Annotated[str, Field(min_length=1, max_length=4)],
        Annotated[str, Field(max_length=8)],
    ]] | None = None

    # --- Pipeline ---
    DICTATION_ENABLED: bool | None = None
    TRACE_ENABLED: bool | None = None
    PUNCTUATION_TO_KEEP: Annotated[str, Field(max_length=32)] | None = None
    DICTATION_MAP: dict[DictKey, DictVal] | None = None
    LOWERCASE_AFTER_STRIPPED_TERMINATOR: list[
        Annotated[str, Field(min_length=1, max_length=32, pattern=r"^[A-Za-zäöüß]+$")]
    ] | None = None
    BEAM_SIZE: Annotated[int, Field(ge=1, le=20)] | None = None
    BEST_OF: Annotated[int, Field(ge=1, le=20)] | None = None
    VAD_FILTER: bool | None = None
    VAD_MIN_SILENCE_MS: Annotated[int, Field(ge=0, le=10000)] | None = None
    VAD_SPEECH_PAD_MS: Annotated[int, Field(ge=0, le=2000)] | None = None
    VAD_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    CONDITION_ON_PREVIOUS_TEXT: bool | None = None
    WORD_TIMESTAMPS_ENABLED: bool | None = None
    NO_SPEECH_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    LOG_PROB_THRESHOLD: Annotated[float, Field(ge=-10.0, le=0.0)] | None = None
    COMPRESSION_RATIO_THRESHOLD: Annotated[float, Field(ge=0.0, le=10.0)] | None = None

    # --- Logging ---
    LOG_FILE: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    LOG_MAX_BYTES: Annotated[int, Field(ge=1024 * 1024, le=1024 * 1024 * 1024)] | None = None
    LOG_BACKUP_COUNT: Annotated[int, Field(ge=1, le=100)] | None = None

    # --- Server ---
    SERVER_HOST: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    SERVER_PORT: Annotated[int, Field(ge=1, le=65535)] | None = None
    SERVER_WORKERS: Annotated[int, Field(ge=1, le=8)] | None = None
    SERVER_LOG_LEVEL: LogLevel | None = None

    # --- Admin / stats access control ---
    # Each entry must be parseable by ipaddress.ip_network(strict=False) — bare
    # IPs (v4 or v6) and CIDRs are both accepted. See _validate_hosts below.
    ADMIN_ALLOWED_HOSTS: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=64)]],
        Field(max_length=64),
    ] | None = None
    STATS_ALLOWED_HOSTS: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=64)]],
        Field(max_length=64),
    ] | None = None

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

    @field_validator("DICTATION_MAP")
    @classmethod
    def _cap_dict(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if v is None:
            return v
        if len(v) > 500:
            raise ValueError(f"DICTATION_MAP capped at 500 entries (got {len(v)})")
        return v

    @field_validator("ALLOWED_MODELS", "PRELOAD_MODELS", "LOWERCASE_AFTER_STRIPPED_TERMINATOR")
    @classmethod
    def _cap_list(cls, v: list[Any] | None) -> list[Any] | None:
        if v is None:
            return v
        if len(v) > 1000:
            raise ValueError(f"capped at 1000 entries (got {len(v)})")
        return v

    @field_validator("CHARACTER_REPLACEMENTS")
    @classmethod
    def _cap_replacements(cls, v: list[Any] | None) -> list[Any] | None:
        if v is None:
            return v
        if len(v) > 32:
            raise ValueError("CHARACTER_REPLACEMENTS capped at 32 entries")
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
    "LOWERCASE_AFTER_STRIPPED_TERMINATOR": frozenset,
    "CHARACTER_REPLACEMENTS": lambda items: tuple(tuple(p) for p in items),
}


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

    validated = AdminConfig.model_validate(merged)
    to_write = validated.model_dump(exclude_none=True, mode="json")  # JSON-friendly tuples

    dst_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dst_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".config.local.", suffix=".tmp", dir=dst_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(to_write, f, indent=2, ensure_ascii=False, sort_keys=True)
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
