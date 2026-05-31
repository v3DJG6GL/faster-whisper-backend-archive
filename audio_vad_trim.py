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
) -> dict:
    """Trim leading/trailing silence from src_path, write to dst_path.

    Returns a dict with the trim outcome — callers need both the success
    flag AND the offsets to keep stored word-level timestamps in sync
    with the trimmed audio:

      {
        "trimmed":           bool,   # True iff dst was written
        "lead_ms":           int,    # samples cut from front (0 if not trimmed)
        "trail_ms":          int,    # samples cut from back  (0 if not trimmed)
        "orig_duration_ms":  int,    # original WAV duration in ms
        "new_duration_ms":   int,    # post-trim duration (equals orig when
                                     # trimmed=False)
      }

    `margin_ms` is the silence preserved on each side of detected speech
    (default 300 ms — matches the merge inter-segment gap so a trimmed
    sample sounds natural). `threshold` is Silero VAD's speech-probability
    cut-off; 0.5 is the library default.

    Format requirements: src must be 16 kHz mono signed-16-bit PCM
    (audio_merge.read_pcm enforces this — same RIFF/WAVE shape every
    other capture pipeline uses). Same format is written out.

    Failure modes:
      - import of faster_whisper.vad fails → log warning, return a dict
        with trimmed=False (caller should treat this as "trim
        unavailable" and use src as-is).
      - VAD finds no speech (silent clip) → trimmed=False, no write.
      - Source/dest IO error → propagated.

    Atomic write: writes to dst_path + ".tmp" and os.replace.
    """
    def _no_trim(orig_ms: int) -> dict:
        return {
            "trimmed": False, "lead_ms": 0, "trail_ms": 0,
            "orig_duration_ms": orig_ms,
            "new_duration_ms": orig_ms,
        }

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
        return _no_trim(0)

    pcm, n_samples = audio_merge.read_pcm(src_path)
    orig_duration_ms = int(round(n_samples * 1000 / _REQ_RATE))
    if n_samples == 0:
        return _no_trim(0)

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
        return _no_trim(orig_duration_ms)

    first_start = int(speeches[0]["start"])
    last_end = int(speeches[-1]["end"])
    margin_samples = int(margin_ms * _REQ_RATE / 1000)
    start_sample = max(0, first_start - margin_samples)
    end_sample = min(n_samples, last_end + margin_samples)
    if start_sample >= end_sample:
        return _no_trim(orig_duration_ms)

    # Bail out if the trim wouldn't actually save anything meaningful
    # (under ~50 ms on either side) AND we're overwriting in place.
    # When dst != src we still write so the caller can rely on dst
    # existing.
    leading_trimmed = start_sample
    trailing_trimmed = n_samples - end_sample
    if (leading_trimmed + trailing_trimmed) < int(_REQ_RATE * 0.05):
        if os.path.abspath(src_path) == os.path.abspath(dst_path):
            return _no_trim(orig_duration_ms)

    out_bytes = pcm[start_sample * audio_merge.BYTES_PER_SAMPLE:
                    end_sample * audio_merge.BYTES_PER_SAMPLE]

    tmp_path = dst_path + ".tmp"
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    try:
        with wave.open(tmp_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(audio_merge._REQ_SAMPWIDTH_BYTES)
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

    lead_ms = int(leading_trimmed * 1000 / _REQ_RATE)
    trail_ms = int(trailing_trimmed * 1000 / _REQ_RATE)
    new_samples = end_sample - start_sample
    new_duration_ms = int(round(new_samples * 1000 / _REQ_RATE))
    logger.info(
        "[vad-trim] %s: trimmed %d ms leading, %d ms trailing (margin=%d ms)",
        os.path.basename(src_path), lead_ms, trail_ms, margin_ms,
    )
    return {
        "trimmed": True,
        "lead_ms": lead_ms,
        "trail_ms": trail_ms,
        "orig_duration_ms": orig_duration_ms,
        "new_duration_ms": new_duration_ms,
    }


def _samples_to_ms(samples: int) -> int:
    return int(round(samples * 1000 / _REQ_RATE))


def trim_pcm_for_merge(
    pcm: bytes,
    n_samples: int,
    *,
    edge_pad_ms: int = 50,
    max_internal_gap_ms: int = 300,
    threshold: float = 0.5,
) -> dict:
    """Plan + apply a per-member silence trim for the group merge path.

    Unlike `trim_wav` (singleton button: outer leading/trailing only), this
    trims a member's outer edges down to `edge_pad_ms` AND collapses any
    internal silence longer than `max_internal_gap_ms` down to that cap. The
    result feeds `audio_merge.merge_wavs`, which then joins trimmed members
    with the inter-segment gap — so all silence in the final clip is uniform
    and bounded (the multi-second dead air at member joins is removed).

    Operates on in-memory PCM only; the caller never touches the original
    capture file on disk (so ungroup still restores raw audio).

    Returns:
      {
        "trimmed":          bool,        # True iff any silence was removed
        "pcm":              bytes,       # trimmed PCM (orig pcm when not trimmed)
        "new_n_samples":    int,
        "lead_ms":          int,         # leading silence removed (info)
        "new_duration_ms":  int,
        "segments": [[orig_start_ms, orig_end_ms, new_start_ms], ...],
      }

    `segments` is the piecewise time-map from original member time to trimmed
    member-local time — one entry per kept speech span. The route layer uses
    it to re-place per-word karaoke timestamps. When nothing is trimmed (VAD
    unavailable, no speech, or a clip already tight) the map is the identity
    span `[[0, dur_ms, 0]]` so callers can treat every member uniformly.
    """
    orig_duration_ms = _samples_to_ms(n_samples)

    def _identity() -> dict:
        return {
            "trimmed": False,
            "pcm": pcm,
            "new_n_samples": n_samples,
            "lead_ms": 0,
            "new_duration_ms": orig_duration_ms,
            "segments": [[0, orig_duration_ms, 0]],
        }

    if n_samples == 0:
        return _identity()

    try:
        import numpy as np
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as e:
        logger.warning(
            "[vad-trim] faster_whisper.vad unavailable (%s); skipping "
            "per-member trim", e,
        )
        return _identity()

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    opts = VadOptions(
        threshold=float(threshold),
        min_silence_duration_ms=200,
        speech_pad_ms=0,
        min_speech_duration_ms=0,
    )
    speeches = get_speech_timestamps(audio, opts, sampling_rate=_REQ_RATE)
    if not speeches:
        return _identity()

    pad = max(0, int(edge_pad_ms)) * _REQ_RATE // 1000
    max_gap = max(0, int(max_internal_gap_ms)) * _REQ_RATE // 1000
    n_spans = len(speeches)

    # Build a SPEECH-BODY-ONLY result in ORIGINAL samples: outer leading and
    # trailing silence is removed ENTIRELY (no outer pad), internal speech
    # segments keep `pad` of context on their inner sides, and the silence
    # between two kept spans is capped at `max_gap`. The merge layer
    # (audio_merge.merge_wavs) then adds the uniform outer margin (EDGE) and
    # the inter-member join silence, so all silence in the final clip is
    # uniform and the per-member body carries no edge padding of its own.
    bsps = audio_merge.BYTES_PER_SAMPLE
    out_chunks: list[bytes] = []
    segments: list[list[int]] = []
    new_pos = 0          # cursor in trimmed-output samples
    prev_kept_end = 0    # original-sample end of the previous kept span

    for idx, seg in enumerate(speeches):
        # No pad on the clip's outer edges (first span's left, last span's
        # right); internal span sides keep `pad` of context.
        left_pad = 0 if idx == 0 else pad
        right_pad = 0 if idx == n_spans - 1 else pad
        s = max(0, int(seg["start"]) - left_pad)
        e = min(n_samples, int(seg["end"]) + right_pad)
        # Don't let this span's padded start swallow audio already emitted by
        # the previous span (overlapping pads on close segments).
        if s < prev_kept_end:
            s = prev_kept_end
        if e <= s:
            continue

        if not segments:
            # Leading: drop everything before the first kept span.
            gap_keep = 0
        else:
            orig_gap = s - prev_kept_end
            gap_keep = min(orig_gap, max_gap)
            if gap_keep > 0:
                # Emit `gap_keep` silence taken from the original gap bytes.
                out_chunks.append(pcm[prev_kept_end * bsps:
                                      (prev_kept_end + gap_keep) * bsps])
                new_pos += gap_keep

        seg_start_new = new_pos
        out_chunks.append(pcm[s * bsps:e * bsps])
        span_len = e - s
        new_pos += span_len
        segments.append([_samples_to_ms(s), _samples_to_ms(e),
                         _samples_to_ms(seg_start_new)])
        prev_kept_end = e

    if not segments:
        return _identity()

    out_pcm = b"".join(out_chunks)
    new_n = new_pos
    new_duration_ms = _samples_to_ms(new_n)
    # Leading silence fully removed (info only; the merge layer owns outer
    # margins now).
    lead_ms = _samples_to_ms(int(speeches[0]["start"]))

    # Nothing meaningfully removed (under ~50 ms total) → identity, so the
    # merge keeps the original PCM and a clean identity map.
    if (n_samples - new_n) < int(_REQ_RATE * 0.05):
        return _identity()

    logger.info(
        "[vad-trim] member: %d→%d ms over %d span(s) (edge_pad=%d, max_gap=%d)",
        orig_duration_ms, new_duration_ms, len(segments),
        edge_pad_ms, max_internal_gap_ms,
    )
    return {
        "trimmed": True,
        "pcm": out_pcm,
        "new_n_samples": new_n,
        "lead_ms": lead_ms,
        "new_duration_ms": new_duration_ms,
        "segments": segments,
    }
