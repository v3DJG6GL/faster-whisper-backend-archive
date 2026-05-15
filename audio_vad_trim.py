"""Silence-trim a WAV file using the Silero VAD that ships with
faster-whisper.

Used by the /captures training-data path:
  - Groups: auto-trim after merge_wavs() (cfg.CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS)
  - Singletons: manual button on the /captures detail page

The trim is non-destructive at the API surface: callers pass a `dst_path`
distinct from the source when they want to preserve the original. Format
constraints match audio_merge.py — 16 kHz mono signed-16-bit PCM RIFF/WAVE
— so a Whisper fine-tune loader sees identical audio shape for trimmed
and untrimmed samples.

Only leading/trailing silence is trimmed; inter-utterance gaps inside the
clip are preserved (groups deliberately insert ~300 ms silence between
members per the Low-Resource Whisper paper). Calm-Whisper (arXiv:2505.12969)
documents long silence as a hallucination trigger during fine-tune; this
is the captures-side mitigation.
"""
from __future__ import annotations

import logging
import os
import time
import wave

import audio_merge

logger = logging.getLogger("whisper-api")

_REQ_RATE = 16000


def trim_wav(
    src_path: str,
    dst_path: str,
    *,
    margin_ms: int = 300,
    threshold: float = 0.5,
) -> bool:
    """Trim leading/trailing silence from src_path, write to dst_path.

    Returns True if the trim ran and produced a different file, False if
    no speech was detected or the file was already tight (output written
    anyway when src != dst, so callers can rely on dst_path existing
    after a True OR False return when src != dst).

    `margin_ms` is the silence preserved on each side of detected speech
    (default 300 ms — matches the merge inter-segment gap so a trimmed
    sample sounds natural). `threshold` is Silero VAD's speech-probability
    cut-off; 0.5 is the library default.

    Format requirements: src must be 16 kHz mono signed-16-bit PCM
    (audio_merge.read_pcm enforces this — same RIFF/WAVE shape every
    other capture pipeline uses). Same format is written out.

    Failure modes:
      - import of faster_whisper.vad fails → log warning, return False
        without writing dst (caller should treat this as "trim
        unavailable" and use src as-is).
      - VAD finds no speech (silent clip) → return False, no write.
      - Source/dest IO error → propagated.

    Atomic write: writes to dst_path + ".tmp" and os.replace.
    """
    # Lazy import — VAD pulls numpy + the Silero model on first call;
    # we don't want the module to be unimportable on hosts without
    # faster-whisper installed (e.g. CI lint).
    try:
        import numpy as np
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as e:
        logger.warning(
            "[vad-trim] faster_whisper.vad unavailable (%s); skipping trim", e,
        )
        return False

    pcm, n_samples = audio_merge.read_pcm(src_path)
    if n_samples == 0:
        return False

    # int16 PCM bytes → float32 [-1, 1] numpy array. Silero VAD's
    # threshold function operates on normalised float audio.
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    # Use a relatively tight min_silence_duration to detect genuine
    # leading/trailing silence; speech_pad_ms here is the VAD's internal
    # pad (not the same as our margin) — keep it small so VAD reports
    # tight boundaries and we apply our margin ourselves.
    opts = VadOptions(
        threshold=float(threshold),
        min_silence_duration_ms=200,
        speech_pad_ms=0,
        min_speech_duration_ms=0,
    )
    speeches = get_speech_timestamps(audio, opts, sampling_rate=_REQ_RATE)
    if not speeches:
        logger.info(
            "[vad-trim] no speech detected in %s; skipping",
            os.path.basename(src_path),
        )
        return False

    first_start = int(speeches[0]["start"])
    last_end = int(speeches[-1]["end"])
    margin_samples = int(margin_ms * _REQ_RATE / 1000)
    start_sample = max(0, first_start - margin_samples)
    end_sample = min(n_samples, last_end + margin_samples)
    if start_sample >= end_sample:
        return False

    # Bail out if the trim wouldn't actually save anything meaningful
    # (under ~50 ms on either side) AND we're overwriting in place.
    # When dst != src we still write so the caller can rely on dst
    # existing.
    leading_trimmed = start_sample
    trailing_trimmed = n_samples - end_sample
    if (leading_trimmed + trailing_trimmed) < int(_REQ_RATE * 0.05):
        if os.path.abspath(src_path) == os.path.abspath(dst_path):
            return False

    out_bytes = pcm[start_sample * audio_merge.BYTES_PER_SAMPLE:
                    end_sample * audio_merge.BYTES_PER_SAMPLE]

    tmp_path = dst_path + ".tmp"
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    try:
        with wave.open(tmp_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(audio_merge.BYTES_PER_SAMPLE)
            w.setframerate(_REQ_RATE)
            w.writeframes(out_bytes)
        try:
            with open(tmp_path, "rb") as fp:
                os.fsync(fp.fileno())
        except OSError:
            pass
        last_err: Exception | None = None
        for _attempt in range(3):
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

    logger.info(
        "[vad-trim] %s: trimmed %d ms leading, %d ms trailing (margin=%d ms)",
        os.path.basename(src_path),
        int(leading_trimmed * 1000 / _REQ_RATE),
        int(trailing_trimmed * 1000 / _REQ_RATE),
        margin_ms,
    )
    return True
