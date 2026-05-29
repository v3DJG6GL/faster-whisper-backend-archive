"""Tests for audio_vad_trim.trim_wav — the singleton trim path.

trim_pcm_for_merge (the group path) is covered by test_group_trim.py; this
file covers trim_wav's branches with the deterministic fake VAD, plus one
real-Silero smoke test (Silero ships with faster-whisper, installed here).
"""

import os
import wave

import pytest

import audio_vad_trim

RATE = 16000


def _read_wav_params(path):
    with wave.open(path, "rb") as w:
        return (w.getnchannels(), w.getsampwidth(), w.getframerate(),
                w.getnframes())


# ---------------------------------------------------------------------------
# Trimmed case (fake VAD)
# ---------------------------------------------------------------------------

def test_trim_wav_removes_leading_trailing(fake_vad, wav_factory, tmp_path):
    # 500 sil | 1000 speech | 500 sil ; margin 100ms keeps 100ms each side.
    src = wav_factory([("sil", 500), ("speech", 1000), ("sil", 500)])
    dst = str(tmp_path / "trimmed.wav")
    res = audio_vad_trim.trim_wav(src, dst, margin_ms=100)

    assert res["trimmed"] is True
    assert res["orig_duration_ms"] == 2000
    # lead/trail = 500 - 100 margin = 400 ms each.
    assert res["lead_ms"] == 400
    assert res["trail_ms"] == 400
    # kept = 100 + 1000 + 100 = 1200 ms.
    assert res["new_duration_ms"] == 1200
    assert os.path.exists(dst)
    assert not os.path.exists(dst + ".tmp")
    ch, sw, rate, nframes = _read_wav_params(dst)
    assert (ch, sw, rate) == (1, 2, RATE)
    assert nframes == int(1200 * RATE / 1000)


def test_trim_wav_margin_clamped_to_clip_inplace_identity(fake_vad,
                                                          wav_factory, tmp_path):
    # No silence at the edges + in-place (src == dst): margin clamps to the
    # clip so nothing is saved (<50ms) -> identity, source untouched.
    src = wav_factory([("speech", 500)])
    res = audio_vad_trim.trim_wav(src, src, margin_ms=300)
    assert res["trimmed"] is False
    assert res["new_duration_ms"] == 500


def test_trim_wav_margin_clamped_to_clip_outofplace_writes(fake_vad,
                                                           wav_factory, tmp_path):
    # Same all-speech clip but dst != src: even though 0ms is actually saved,
    # the <50ms early-return only fires for in-place writes, so the function
    # still writes dst (the full clip) and reports trimmed=True. This is the
    # documented "caller can rely on dst existing" contract.
    src = wav_factory([("speech", 500)])
    dst = str(tmp_path / "out.wav")
    res = audio_vad_trim.trim_wav(src, dst, margin_ms=300)
    assert res["trimmed"] is True
    assert res["lead_ms"] == 0
    assert res["trail_ms"] == 0
    assert res["new_duration_ms"] == 500
    assert os.path.exists(dst)


# ---------------------------------------------------------------------------
# Identity guards (fake VAD / no VAD)
# ---------------------------------------------------------------------------

def test_trim_wav_vad_unavailable(no_vad, wav_factory, tmp_path):
    src = wav_factory([("sil", 200), ("speech", 200), ("sil", 200)])
    dst = str(tmp_path / "out.wav")
    res = audio_vad_trim.trim_wav(src, dst)
    assert res["trimmed"] is False
    assert res["lead_ms"] == 0 and res["trail_ms"] == 0
    # VAD import fails before reading the WAV -> orig duration reported as 0.
    assert res["orig_duration_ms"] == 0
    assert res["new_duration_ms"] == 0
    assert not os.path.exists(dst)


def test_trim_wav_empty_source(fake_vad, wav_factory, tmp_path):
    src = wav_factory(None, pcm=b"")  # 0-sample WAV
    dst = str(tmp_path / "out.wav")
    res = audio_vad_trim.trim_wav(src, dst)
    assert res["trimmed"] is False
    assert res["orig_duration_ms"] == 0
    assert not os.path.exists(dst)


def test_trim_wav_no_speech_identity(fake_vad, wav_factory, tmp_path):
    src = wav_factory([("sil", 800)])
    dst = str(tmp_path / "out.wav")
    res = audio_vad_trim.trim_wav(src, dst)
    assert res["trimmed"] is False
    assert res["orig_duration_ms"] == 800
    assert res["new_duration_ms"] == 800
    assert not os.path.exists(dst)


def test_trim_wav_tiny_trim_inplace_is_noop(fake_vad, wav_factory, tmp_path):
    # 30ms sil | 500 speech | 30ms sil ; margin 0 -> 30+30=60ms... that's
    # >50ms so it WOULD trim. Use 20ms edges -> 40ms total < 50ms guard.
    src = wav_factory([("sil", 20), ("speech", 500), ("sil", 20)])
    res = audio_vad_trim.trim_wav(src, src, margin_ms=0)  # in place
    # < 50 ms saved AND src == dst -> identity, source untouched.
    assert res["trimmed"] is False
    assert res["orig_duration_ms"] == 540


def test_trim_wav_tiny_trim_outofplace_still_writes(fake_vad, wav_factory,
                                                    tmp_path):
    # Same tiny trim but dst != src -> must still write dst (caller relies on
    # dst existing). The function does NOT early-return for dst != src.
    src = wav_factory([("sil", 20), ("speech", 500), ("sil", 20)])
    dst = str(tmp_path / "copy.wav")
    res = audio_vad_trim.trim_wav(src, dst, margin_ms=0)
    assert res["trimmed"] is True
    assert os.path.exists(dst)
    # Edges (20ms each, but only ~40 total) are cut even though tiny.
    assert res["lead_ms"] == 20
    assert res["trail_ms"] == 20
    assert res["new_duration_ms"] == 500


# ---------------------------------------------------------------------------
# Real Silero VAD smoke test (no fake_vad fixture)
# ---------------------------------------------------------------------------

def _synth_speech_like(dur_s, sr=RATE):
    """Synthesize a voiced, syllabically-modulated, formant-shaped signal
    that the real Silero VAD classifies as speech (white noise / pure tones
    do not reliably trip it). Returns int16 samples."""
    import numpy as np
    t = np.linspace(0, dur_s, int(dur_s * sr), endpoint=False)
    f0 = 120 + 30 * np.sin(2 * np.pi * 3 * t)  # pitch glide
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = np.zeros_like(t)
    for h in range(1, 40):
        fh = h * 120
        w = (np.exp(-((fh - 600) / 400) ** 2)
             + np.exp(-((fh - 1200) / 500) ** 2)
             + 0.5 * np.exp(-((fh - 2400) / 700) ** 2))  # formant peaks
        sig += w * np.sin(h * phase)
    sig *= np.clip(np.sin(2 * np.pi * 4 * t), 0, 1)  # 4 Hz syllabic envelope
    sig = sig / np.abs(sig).max() * 12000
    return sig.astype(np.int16)


def test_trim_wav_real_silero_smoke(wav_factory, tmp_path):
    pytest.importorskip("faster_whisper.vad")
    import numpy as np

    # sil | speech-like | sil with true digital-silence edges. Uses the real
    # Silero model (no fake_vad fixture). threshold=0.2 because our synthetic
    # speech is harder to detect than real voice.
    sr = RATE
    edge = np.zeros(int(0.8 * sr), dtype=np.int16)
    mid = _synth_speech_like(2.0)
    pcm = np.concatenate([edge, mid, edge]).tobytes()
    src = wav_factory(None, pcm=pcm)
    dst = str(tmp_path / "real_trim.wav")

    res = audio_vad_trim.trim_wav(src, dst, margin_ms=100, threshold=0.2)
    assert res["trimmed"] is True
    assert res["new_duration_ms"] < res["orig_duration_ms"]
    assert os.path.exists(dst)
    ch, sw, rate, _ = _read_wav_params(dst)
    assert (ch, sw, rate) == (1, 2, RATE)
