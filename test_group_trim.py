"""Tests for per-member silence trimming of capture groups.

Covers:
  - audio_vad_trim.trim_pcm_for_merge: leading/trailing trim + internal-gap
    collapse, the kept-speech segment time-map, and the no-op/identity guards.
  - audio_merge.merge_wavs: per-member trim wiring + concatenation, and the
    trim=False identity path.
  - captures_routes._remap_time_ms / _build_merged_words: word-timestamp
    re-placement onto the trimmed merged timeline (per-member + legacy).

Silero VAD isn't installed in CI, so the planner tests inject a deterministic
fake `faster_whisper.vad` that treats any non-zero PCM as speech. The
route-layer tests are skipped unless fastapi (the full app stack) is present.

Runnable two ways:
    pytest test_group_trim.py
    python  test_group_trim.py
"""

import sys
import tempfile
import types
import wave

import numpy as np
import pytest

import audio_merge
import audio_vad_trim

RATE = 16000


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _pcm(spec):
    """Build int16 PCM from [(kind, ms), ...] where kind is 'sil' or 'speech'.
    Returns (pcm_bytes, n_samples)."""
    parts = []
    for kind, ms in spec:
        n = int(ms * RATE / 1000)
        val = 0 if kind == "sil" else 4000
        parts.append(np.full(n, val, dtype=np.int16))
    a = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)
    return a.tobytes(), len(a)


def _install_fake_vad(monkeypatch):
    """Inject a faster_whisper.vad whose get_speech_timestamps marks every
    contiguous non-zero run as a speech segment."""
    def get_speech_timestamps(audio, opts, sampling_rate=RATE):
        segs = []
        start = None
        for i, v in enumerate(audio):
            nz = abs(float(v)) > 1e-6
            if nz and start is None:
                start = i
            elif not nz and start is not None:
                segs.append({"start": start, "end": i})
                start = None
        if start is not None:
            segs.append({"start": start, "end": len(audio)})
        return segs

    class VadOptions:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    pkg = types.ModuleType("faster_whisper")
    vad = types.ModuleType("faster_whisper.vad")
    vad.VadOptions = VadOptions
    vad.get_speech_timestamps = get_speech_timestamps
    pkg.vad = vad
    monkeypatch.setitem(sys.modules, "faster_whisper", pkg)
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)


def _write_wav(path, pcm):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


# --------------------------------------------------------------------------
# trim_pcm_for_merge
# --------------------------------------------------------------------------

def test_trim_collapses_internal_gap_and_edges(monkeypatch):
    _install_fake_vad(monkeypatch)
    # 200ms sil | 300ms speech | 1000ms sil (internal) | 300ms speech | 200ms sil
    pcm, n = _pcm([
        ("sil", 200), ("speech", 300), ("sil", 1000),
        ("speech", 300), ("sil", 200),
    ])
    res = audio_vad_trim.trim_pcm_for_merge(
        pcm, n, edge_pad_ms=50, max_internal_gap_ms=300,
    )
    assert res["trimmed"] is True
    # Two kept speech spans, each padded by 50ms on the available sides.
    assert res["segments"] == [[150, 550, 0], [1450, 1850, 700]]
    # 1100ms = span1 (400) + capped gap (300) + span2 (400).
    assert res["new_duration_ms"] == 1100
    assert res["lead_ms"] == 150
    # new_start values are strictly increasing and the trimmed PCM matches.
    new_starts = [s[2] for s in res["segments"]]
    assert new_starts == sorted(new_starts)
    assert len(res["pcm"]) == res["new_n_samples"] * audio_merge.BYTES_PER_SAMPLE
    assert res["new_n_samples"] < n


def test_trim_no_speech_is_identity(monkeypatch):
    _install_fake_vad(monkeypatch)
    pcm, n = _pcm([("sil", 500)])
    res = audio_vad_trim.trim_pcm_for_merge(pcm, n)
    assert res["trimmed"] is False
    assert res["pcm"] == pcm
    assert res["segments"] == [[0, 500, 0]]


def test_trim_all_speech_is_noop(monkeypatch):
    _install_fake_vad(monkeypatch)
    pcm, n = _pcm([("speech", 500)])
    res = audio_vad_trim.trim_pcm_for_merge(pcm, n, edge_pad_ms=50)
    # Nothing meaningful to cut (pad covers the whole clip) → identity.
    assert res["trimmed"] is False
    assert res["new_duration_ms"] == 500


def test_trim_unavailable_vad_is_identity(monkeypatch):
    # Ensure the real (absent) faster_whisper import path returns identity.
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    pcm, n = _pcm([("sil", 100), ("speech", 100)])
    res = audio_vad_trim.trim_pcm_for_merge(pcm, n)
    assert res["trimmed"] is False
    assert res["new_n_samples"] == n


# --------------------------------------------------------------------------
# merge_wavs
# --------------------------------------------------------------------------

def test_merge_trims_each_member(monkeypatch):
    _install_fake_vad(monkeypatch)
    m1, _ = _pcm([("sil", 200), ("speech", 300), ("sil", 200)])
    m2, _ = _pcm([("sil", 300), ("speech", 300), ("sil", 100)])
    with tempfile.TemporaryDirectory() as d:
        p1, p2 = f"{d}/m1.wav", f"{d}/m2.wav"
        out = f"{d}/merged.wav"
        _write_wav(p1, m1)
        _write_wav(p2, m2)
        res = audio_merge.merge_wavs(
            [p1, p2], out, gap_ms=300, trim=True,
            edge_pad_ms=50, max_internal_gap_ms=300,
        )
    assert len(res["members"]) == 2
    # Each member kept its single 300ms speech span padded to 400ms.
    assert res["members"][0]["new_duration_ms"] == 400
    assert res["members"][1]["new_duration_ms"] == 400
    # merged = 400 + 300 gap + 400 = 1100ms.
    assert res["duration_ms"] == 1100


def test_merge_trim_false_is_identity(monkeypatch):
    m1, n1 = _pcm([("sil", 100), ("speech", 100)])
    m2, n2 = _pcm([("speech", 100), ("sil", 100)])
    with tempfile.TemporaryDirectory() as d:
        p1, p2 = f"{d}/m1.wav", f"{d}/m2.wav"
        out = f"{d}/merged.wav"
        _write_wav(p1, m1)
        _write_wav(p2, m2)
        res = audio_merge.merge_wavs([p1, p2], out, gap_ms=300, trim=False)
    assert res["members"][0]["new_duration_ms"] == 200
    assert res["members"][0]["segments"] == [[0, 200, 0]]
    # 200 + 300 gap + 200 = 700ms.
    assert res["duration_ms"] == 700


# --------------------------------------------------------------------------
# Route-layer timestamp re-placement (full app stack only)
# --------------------------------------------------------------------------

def _routes():
    pytest.importorskip("fastapi")
    import captures_routes
    return captures_routes


def test_remap_time_ms():
    cr = _routes()
    segs = [[150, 550, 0], [1450, 1850, 700]]
    # inside first span
    assert cr._remap_time_ms(200, segs) == 50
    # before first span → snaps to span start (0)
    assert cr._remap_time_ms(0, segs) == 0
    # inside the collapsed gap → snaps to the next span's new start
    assert cr._remap_time_ms(1000, segs) == 700
    # inside second span
    assert cr._remap_time_ms(1500, segs) == 750
    # past the last span → end of last span
    assert cr._remap_time_ms(5000, segs) == 1100


def test_build_merged_words_per_member(monkeypatch):
    cr = _routes()
    monkeypatch.setattr(
        cr, "_align_words_to_final",
        lambda words, final, model_name=None: [dict(w) for w in words],
    )
    members = [
        {"id": "a", "words": [{"word": "hi", "start": 0.2, "end": 0.5}],
         "final": "hi", "duration_seconds": 2.0},
        {"id": "b", "words": [{"word": "yo", "start": 0.1, "end": 0.3}],
         "final": "yo", "duration_seconds": 1.5},
    ]
    trims = {
        "a": {"lead_ms": 150, "new_duration_ms": 1100,
              "segments": [[150, 550, 0], [1450, 1850, 700]]},
        "b": {"lead_ms": 100, "new_duration_ms": 900,
              "segments": [[100, 400, 0], [1200, 1500, 500]]},
    }
    out = cr._build_merged_words(members, 300, member_trims=trims)
    assert len(out) == 2
    # a: 0.2s→50ms local, offset 0 → 0.05; 0.5s→350ms → 0.35
    assert out[0]["start"] == pytest.approx(0.05)
    assert out[0]["end"] == pytest.approx(0.35)
    # b: offset = 1100ms + 1*300ms = 1.4s; 0.1s→0ms, 0.3s→200ms
    assert out[1]["start"] == pytest.approx(1.4)
    assert out[1]["end"] == pytest.approx(1.6)
    assert out[1]["member_idx"] == 1


def test_build_merged_words_legacy(monkeypatch):
    cr = _routes()
    monkeypatch.setattr(
        cr, "_align_words_to_final",
        lambda words, final, model_name=None: [dict(w) for w in words],
    )
    members = [
        {"id": "a", "words": [{"word": "hi", "start": 0.2, "end": 0.5}],
         "final": "hi", "duration_seconds": 2.0},
        {"id": "b", "words": [{"word": "yo", "start": 0.1, "end": 0.3}],
         "final": "yo", "duration_seconds": 1.5},
    ]
    # Empty trims → legacy flat-offset timeline using full durations.
    out = cr._build_merged_words(members, 300, member_trims={})
    assert out[0]["start"] == pytest.approx(0.2)
    # b offset = 2.0 + 1*0.3 = 2.3 → start 0.1+2.3
    assert out[1]["start"] == pytest.approx(2.4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
