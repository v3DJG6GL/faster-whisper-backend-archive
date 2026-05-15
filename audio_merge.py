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

# Hard cap on the merged WAV duration in samples. 28 s × 16 kHz = 448_000.
# Whisper's feature extractor truncates anything >30 s; we leave a margin.
MAX_MERGED_SAMPLES = 448_000

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
) -> tuple[int, int]:
    """Concatenate the given WAVs into a single PCM WAV at dst_path, with
    `gap_ms` of silence between each adjacent pair (no leading or
    trailing silence — the feature extractor zero-pads to 30 s anyway).

    Returns `(bytes_written, n_samples)`.

    Raises:
      WavFormatError: any source doesn't match (1ch, 16-bit, 16 kHz).
      ValueError:     <2 sources, total duration > 28 s, or empty source.
      OSError:        disk write failure (atomic .tmp + os.replace).
    """
    if len(src_paths) < 2:
        raise ValueError("need at least 2 sources to merge")
    if gap_ms < 0:
        raise ValueError("gap_ms must be ≥ 0")

    pieces: list[bytes] = []
    total_samples = 0
    gap = silence_bytes(gap_ms) if gap_ms > 0 else b""
    gap_samples = len(gap) // BYTES_PER_SAMPLE

    for i, sp in enumerate(src_paths):
        pcm, n = read_pcm(sp)
        if n == 0:
            raise ValueError(f"source {sp} is empty")
        if i > 0:
            pieces.append(gap)
            total_samples += gap_samples
        pieces.append(pcm)
        total_samples += n
        if total_samples > MAX_MERGED_SAMPLES:
            raise ValueError(
                f"merged duration would exceed 28 s "
                f"(got {total_samples / _REQ_RATE:.2f} s after "
                f"{i+1} of {len(src_paths)} sources)"
            )

    out_pcm = b"".join(pieces)

    # Atomic write: tmp + fsync + os.replace, with the same 3-retry
    # Windows-AV-lock loop captures_store uses.
    tmp_path = dst_path + ".tmp"
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
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

    # Optional: trim leading/trailing silence from the merged WAV via
    # Silero VAD. Same disk path is overwritten (merged WAV is already
    # derivative; the un-trimmed version isn't independently useful).
    # On VAD unavailability or no-speech, we keep the merged-as-written
    # WAV. After a successful trim we re-measure the sample count so
    # capture_groups.merged_duration_ms reflects the trimmed length.
    try:
        import config as _cfg
        if getattr(_cfg, "CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS", False):
            import audio_vad_trim
            margin_ms = int(getattr(_cfg, "CAPTURES_VAD_TRIM_MARGIN_MS", 300))
            if audio_vad_trim.trim_wav(dst_path, dst_path, margin_ms=margin_ms):
                with wave.open(dst_path, "rb") as _w:
                    total_samples = _w.getnframes()
    except Exception as _ve:
        logger.warning("[merge] VAD trim skipped: %s", _ve)

    return os.path.getsize(dst_path), total_samples


def hash_wav_pcm(src_path: str) -> str:
    """Return SHA-256 hex of just the PCM frame bytes (excluding the WAV
    header) of a source file. Used by capture_groups_store to detect when
    a member's audio content changed under a group (vs. an innocent
    re-encoding of the header)."""
    pcm, _ = read_pcm(src_path)
    return sha256(pcm).hexdigest()
