"""In-process PCM-WAV concatenation utility for Whisper training-data
packing.

We pack 2+ short same-speaker captures into a single ≤28 s WAV (Whisper's
encoder hard-caps at 30 s; 28 s leaves a safety margin). Per the
Low-Resource Whisper paper (arXiv 2412.15726) we insert a short silence
between segments to preserve the encoder's noise-floor model at the
joins; butt-splicing degrades transition quality.

Because every input is already 16 kHz mono signed-16-bit PCM (produced by
`audio_transcode.py` at capture time), we don't need PyAV here — pure
stdlib `wave` byte splicing is faster, smaller, and lossless. The
output is a valid RIFF/WAVE file every browser plays and `datasets.Audio()`
loads natively.
"""
from __future__ import annotations

import logging
import os
import time
import wave
from hashlib import sha256

logger = logging.getLogger("whisper-api")

# Required source format. Anything else gets rejected (defense-in-depth
# against an upstream pipeline change). The capture-time transcode is the
# only writer; if it ever produces a different format we want to know.
_REQ_CHANNELS = 1
_REQ_SAMPWIDTH_BYTES = 2     # signed 16-bit
_REQ_RATE = 16000

# Hard cap on the merged WAV duration. The LIVE value is
# cfg.CAPTURES_SAMPLE_MAX_DURATION_S (read per-merge by _max_merged_samples()
# so an admin change applies immediately); MAX_MERGED_SAMPLES below is only a
# fallback when config is unavailable. Whisper truncates >30 s; the configured
# value keeps a margin.
MAX_MERGED_SAMPLES = 448_000  # 28 s × 16 kHz (fallback default)


def _max_merged_samples() -> int:
    """Live merged-WAV sample cap from config (samples = seconds × 16 kHz)."""
    try:
        import config as cfg
        cap_s = float(getattr(cfg, "CAPTURES_SAMPLE_MAX_DURATION_S", 29.9))
        return int(cap_s * _REQ_RATE)
    except Exception:
        return MAX_MERGED_SAMPLES

# Bytes per sample of audio at our fixed format (channels * sampwidth).
BYTES_PER_SAMPLE = _REQ_CHANNELS * _REQ_SAMPWIDTH_BYTES


class WavFormatError(ValueError):
    """Raised when a source WAV doesn't match the required 16 kHz mono s16
    format. The transcode pipeline normalises all internal files; a
    mismatch indicates upstream drift."""


def read_pcm(src_path: str) -> tuple[bytes, int]:
    """Read the raw PCM frames + sample count from a WAV file. Validates
    format. Returns `(pcm_bytes, n_samples)`."""
    with wave.open(src_path, "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (
            _REQ_CHANNELS, _REQ_SAMPWIDTH_BYTES, _REQ_RATE,
        ):
            raise WavFormatError(
                f"{os.path.basename(src_path)}: expected "
                f"({_REQ_CHANNELS}, {_REQ_SAMPWIDTH_BYTES}, {_REQ_RATE}), "
                f"got ({w.getnchannels()}, {w.getsampwidth()}, "
                f"{w.getframerate()})"
            )
        n = w.getnframes()
        return w.readframes(n), n


def silence_bytes(ms: int) -> bytes:
    """Generate `ms` milliseconds of digital silence (all-zero PCM bytes)
    in our fixed format. Silence and feature-extractor zero-padding are
    byte-identical, so we don't need anything fancier than this."""
    n_samples = int(round(ms / 1000.0 * _REQ_RATE))
    return bytes(n_samples * BYTES_PER_SAMPLE)


def merge_wavs(
    src_paths: list[str],
    dst_path: str,
    *,
    gap_ms: int = 300,
    trim: bool = False,
    edge_pad_ms: int = 50,
    max_internal_gap_ms: int = 300,
    threshold: float = 0.5,
) -> dict:
    """Concatenate the given WAVs into a single PCM WAV at dst_path.

    When `trim` is set (audio_vad_trim.trim_pcm_for_merge available), the
    output uses a UNIFORM-silence layout: each member is trimmed to its
    speech body (no edge padding, internal gaps capped at
    `max_internal_gap_ms`), then concatenated as
    `[edge_pad_ms] body0 [gap_ms] body1 [gap_ms] … bodyN [edge_pad_ms]`.
    So every member join carries exactly `gap_ms` of silence and both outer
    ends carry `edge_pad_ms` — replacing the old additive layout where a join
    stacked member-i-trailing + gap + member-i+1-leading into multi-second
    dead air. Each member dict carries `offset_ms`, its absolute start in the
    merged WAV, so the route layer can re-place per-word karaoke timestamps.

    When `trim` is False (VAD unavailable / disabled), falls back to the
    legacy layout: full members joined by `gap_ms`, no outer margin.

    Returns a dict:
      {
        "bytes":        int,           # size of the written WAV on disk
        "n_samples":    int,           # total samples in the merged WAV
        "duration_ms":  int,
        "members": [                   # one entry per src_paths, in order
          {"lead_ms": int, "new_duration_ms": int,
           "segments": [[orig_start_ms, orig_end_ms, new_start_ms], ...],
           "offset_ms": int},          # absolute body start in merged WAV
          ...                          #   (trim path only; omitted when trim=False)
        ],
      }

    The per-member `segments` map is what the route layer persists
    (member_trims_json) so group word-level karaoke timestamps stay in sync
    with the trimmed audio. When `trim` is False (or VAD is unavailable) each
    member carries an identity map over its full duration.

    Raises:
      WavFormatError: any source doesn't match (1ch, 16-bit, 16 kHz).
      ValueError:     <2 sources, total duration > 28 s, or empty source.
      OSError:        disk write failure (atomic .tmp + os.replace).
    """
    if len(src_paths) < 1:
        raise ValueError("need at least 1 source")
    if gap_ms < 0:
        raise ValueError("gap_ms must be ≥ 0")
    _max_samples = _max_merged_samples()

    trimmer = None
    if trim:
        try:
            import audio_vad_trim
            trimmer = audio_vad_trim.trim_pcm_for_merge
        except Exception as _e:  # pragma: no cover - import guard
            logger.warning("[merge] per-member trim unavailable: %s", _e)

    pieces: list[bytes] = []
    total_samples = 0
    members: list[dict] = []

    def _over_cap(reserve: int, i: int) -> None:
        if total_samples + reserve > _max_samples:
            raise ValueError(
                f"merged duration would exceed the sample cap "
                f"({_max_samples / _REQ_RATE:.2f} s) — got "
                f"{(total_samples + reserve) / _REQ_RATE:.2f} s after "
                f"{i+1} of {len(src_paths)} sources"
            )

    if trimmer is not None:
        # Uniform-silence layout. Per-member bodies carry NO edge padding (the
        # trim returns speech-only). We add a uniform outer margin
        # (edge_pad_ms) at both ends and `gap_ms` of silence at each join, so
        # every join == gap_ms and both outer ends == edge_pad_ms — all silence
        # in the merged clip is uniform and bounded. Each member records its
        # absolute offset (ms) in the merged WAV for word-timestamp re-placement.
        edge = silence_bytes(edge_pad_ms)
        edge_samples = len(edge) // BYTES_PER_SAMPLE
        join = silence_bytes(gap_ms)
        join_samples = len(join) // BYTES_PER_SAMPLE
        pieces.append(edge)                 # leading outer margin
        total_samples += edge_samples
        for i, sp in enumerate(src_paths):
            pcm, n = read_pcm(sp)
            if n == 0:
                raise ValueError(f"source {sp} is empty")
            res = trimmer(
                pcm, n,
                edge_pad_ms=edge_pad_ms,
                max_internal_gap_ms=max_internal_gap_ms,
                threshold=threshold,
            )
            body = res["pcm"]
            bn = res["new_n_samples"]
            if i > 0:
                pieces.append(join)
                total_samples += join_samples
            offset_ms = int(round(total_samples * 1000 / _REQ_RATE))
            members.append({
                "lead_ms": int(res["lead_ms"]),
                "new_duration_ms": int(res["new_duration_ms"]),
                "segments": res["segments"],
                "offset_ms": offset_ms,
            })
            pieces.append(body)
            total_samples += bn
            _over_cap(edge_samples, i)       # reserve the trailing margin
        pieces.append(edge)                  # trailing outer margin
        total_samples += edge_samples
    else:
        # Legacy / no-VAD path: full members joined by gap_ms, no outer margin
        # (unchanged behaviour for groups built without per-member trimming).
        gap = silence_bytes(gap_ms)
        gap_samples = len(gap) // BYTES_PER_SAMPLE
        for i, sp in enumerate(src_paths):
            pcm, n = read_pcm(sp)
            if n == 0:
                raise ValueError(f"source {sp} is empty")
            dur_ms = int(round(n * 1000 / _REQ_RATE))
            members.append({
                "lead_ms": 0,
                "new_duration_ms": dur_ms,
                "segments": [[0, dur_ms, 0]],
            })
            if i > 0:
                pieces.append(gap)
                total_samples += gap_samples
            pieces.append(pcm)
            total_samples += n
            _over_cap(0, i)

    out_pcm = b"".join(pieces)

    # Atomic write: tmp + fsync + os.replace, with the same 3-retry
    # Windows-AV-lock loop captures_store uses.
    tmp_path = dst_path + ".tmp"
    # `or "."` so a bare-filename dst_path (no dirname) doesn't pass "" to
    # os.makedirs, which raises FileNotFoundError on some platforms.
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    try:
        with wave.open(tmp_path, "wb") as w:
            w.setnchannels(_REQ_CHANNELS)
            w.setsampwidth(_REQ_SAMPWIDTH_BYTES)
            w.setframerate(_REQ_RATE)
            w.writeframes(out_pcm)
        try:
            with open(tmp_path, "rb") as fp:
                os.fsync(fp.fileno())
        except OSError:
            pass
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                os.replace(tmp_path, dst_path)
                last_err = None
                break
            except OSError as e:
                last_err = e
                time.sleep(0.1)
        if last_err is not None:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise last_err
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Silence trimming now happens per-member BEFORE concatenation (above),
    # so the merged WAV needs no separate outer trim: member 0's leading and
    # the last member's trailing silence were already cut, and internal joins
    # carry only the bounded inter-segment gap.
    return {
        "bytes": os.path.getsize(dst_path),
        "n_samples": total_samples,
        "duration_ms": int(round(total_samples * 1000 / _REQ_RATE)),
        "members": members,
    }


def hash_wav_pcm(src_path: str) -> str:
    """Return SHA-256 hex of just the PCM frame bytes (excluding the WAV
    header) of a source file. Used by capture_samples_store to detect when
    a member's audio content changed under a group (vs. an innocent
    re-encoding of the header)."""
    pcm, _ = read_pcm(src_path)
    return sha256(pcm).hexdigest()
