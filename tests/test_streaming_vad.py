"""Tests for the streaming endpointer: energy gate, the make_endpointer fallback
logic, and frame iteration. The bundled-Silero path needs faster-whisper (not
installed in the deps-free test env), so its availability is monkeypatched to keep
these deterministic on every platform.
"""

import numpy as np
import pytest

import streaming_vad
from streaming_vad import (
    FRAME_SAMPLES,
    EnergyEndpointer,
    iter_frames,
    make_endpointer,
    rms_dbfs,
)


def _frame(level):
    return np.full(FRAME_SAMPLES, level, dtype=np.float32)


def test_rms_dbfs_silence_and_full_scale():
    assert rms_dbfs(np.zeros(FRAME_SAMPLES, dtype=np.float32)) == float("-inf")
    assert rms_dbfs(np.ones(FRAME_SAMPLES, dtype=np.float32)) == pytest.approx(0.0, abs=1e-6)


def test_energy_endpointer_hysteresis():
    ep = EnergyEndpointer(threshold_dbfs=-42.0, hysteresis_db=6.0)
    assert ep.is_speech(_frame(0.0)) is False           # silence
    assert ep.is_speech(_frame(0.3)) is True            # loud → speech (~ -10 dBFS)
    # A small dip stays "speaking" until below the -48 dBFS off-threshold.
    assert ep.is_speech(_frame(0.01)) is True           # ~ -40 dBFS, above off
    assert ep.is_speech(_frame(0.0)) is False           # silence → off
    ep.reset()
    assert ep._speaking is False


def test_make_endpointer_energy_backend():
    ep = make_endpointer("energy")
    assert isinstance(ep, EnergyEndpointer)


def test_make_endpointer_auto_falls_back_when_silero_unavailable(monkeypatch):
    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no bundled silero here")

    monkeypatch.setattr(streaming_vad, "SileroEndpointer", _Boom)
    ep = make_endpointer("auto")
    assert isinstance(ep, EnergyEndpointer)


def test_make_endpointer_silero_backend_raises_when_unavailable(monkeypatch):
    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no bundled silero here")

    monkeypatch.setattr(streaming_vad, "SileroEndpointer", _Boom)
    with pytest.raises(RuntimeError):
        make_endpointer("silero")


def test_make_endpointer_auto_uses_silero_when_available(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(streaming_vad, "SileroEndpointer", lambda **k: sentinel)
    assert make_endpointer("auto") is sentinel


def test_iter_frames_drops_partial_tail():
    samples = np.arange(FRAME_SAMPLES * 2 + 100, dtype=np.float32)
    frames = list(iter_frames(samples))
    assert len(frames) == 2
    assert all(f.shape[0] == FRAME_SAMPLES for f in frames)
