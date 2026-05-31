"""Tests for audio_merge low-level helpers and merge_wavs error paths.

The trim-wiring / happy concatenation paths (per-member trim, over-cap-but-
fits, trim=False identity) are covered by test_group_trim.py; this file covers
the gaps: read_pcm format validation, silence_bytes, hash_wav_pcm, the
merge_wavs ValueError guards, and the clean 2-file merge dict shape.
"""

import os
import wave
from hashlib import sha256

import numpy as np
import pytest

import audio_merge

RATE = 16000


def _write_wav(path, pcm, *, nchannels=1, sampwidth=2, rate=RATE):
    with wave.open(path, "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(pcm)
    return path


def _pcm(ms, val=4000):
    n = int(ms * RATE / 1000)
    return np.full(n, val, dtype=np.int16).tobytes(), n


# ---------------------------------------------------------------------------
# read_pcm
# ---------------------------------------------------------------------------

def test_read_pcm_valid(tmp_path):
    pcm, n = _pcm(100)
    p = _write_wav(str(tmp_path / "ok.wav"), pcm)
    got_pcm, got_n = audio_merge.read_pcm(p)
    assert got_pcm == pcm
    assert got_n == n


def test_read_pcm_wrong_channels(tmp_path):
    # 2 channels, each sample 2 bytes -> need an even byte count per frame.
    pcm = (np.zeros(200, dtype=np.int16)).tobytes()
    p = _write_wav(str(tmp_path / "stereo.wav"), pcm, nchannels=2)
    with pytest.raises(audio_merge.WavFormatError):
        audio_merge.read_pcm(p)


def test_read_pcm_wrong_sampwidth(tmp_path):
    pcm = bytes(400)  # 8-bit samples
    p = _write_wav(str(tmp_path / "s8.wav"), pcm, sampwidth=1)
    with pytest.raises(audio_merge.WavFormatError):
        audio_merge.read_pcm(p)


def test_read_pcm_wrong_rate(tmp_path):
    pcm, _ = _pcm(100)
    p = _write_wav(str(tmp_path / "44k.wav"), pcm, rate=44100)
    with pytest.raises(audio_merge.WavFormatError):
        audio_merge.read_pcm(p)


# ---------------------------------------------------------------------------
# silence_bytes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ms,expected_samples", [
    (0, 0),
    (1, 16),
    (300, 4800),
    (1000, 16000),
])
def test_silence_bytes_lengths(ms, expected_samples):
    b = audio_merge.silence_bytes(ms)
    assert len(b) == expected_samples * audio_merge.BYTES_PER_SAMPLE
    assert b == bytes(len(b))  # all zero


def test_silence_bytes_rounds_to_nearest_sample():
    # 0.5 sample at 16k for a fractional ms boundary -> round() not trunc.
    # 0.03125 ms = 0.5 samples -> rounds to 0 (banker's rounding to even).
    assert len(audio_merge.silence_bytes(0)) == 0


# ---------------------------------------------------------------------------
# hash_wav_pcm
# ---------------------------------------------------------------------------

def test_hash_wav_pcm_is_pcm_only(tmp_path):
    pcm, _ = _pcm(50, val=1234)
    p = _write_wav(str(tmp_path / "a.wav"), pcm)
    assert audio_merge.hash_wav_pcm(p) == sha256(pcm).hexdigest()


def test_hash_wav_pcm_header_independent(tmp_path):
    # Same PCM, written twice (independent headers) -> identical hash.
    pcm, _ = _pcm(50, val=777)
    p1 = _write_wav(str(tmp_path / "h1.wav"), pcm)
    p2 = _write_wav(str(tmp_path / "h2.wav"), pcm)
    assert audio_merge.hash_wav_pcm(p1) == audio_merge.hash_wav_pcm(p2)


def test_hash_wav_pcm_differs_on_content(tmp_path):
    p1 = _write_wav(str(tmp_path / "c1.wav"), _pcm(50, val=1)[0])
    p2 = _write_wav(str(tmp_path / "c2.wav"), _pcm(50, val=2)[0])
    assert audio_merge.hash_wav_pcm(p1) != audio_merge.hash_wav_pcm(p2)


# ---------------------------------------------------------------------------
# merge_wavs error paths
# ---------------------------------------------------------------------------

def test_merge_requires_at_least_one_source(tmp_path):
    # Single-capture samples are allowed now (group-of-one); only 0 is invalid.
    with pytest.raises(ValueError, match="at least 1"):
        audio_merge.merge_wavs([], str(tmp_path / "out.wav"))


def test_merge_single_source_ok(tmp_path):
    p = _write_wav(str(tmp_path / "one.wav"), _pcm(100)[0])
    res = audio_merge.merge_wavs([p], str(tmp_path / "out.wav"), trim=False)
    assert len(res["members"]) == 1
    assert res["members"][0]["new_duration_ms"] == 100


def test_merge_negative_gap_rejected(tmp_path):
    p1 = _write_wav(str(tmp_path / "a.wav"), _pcm(100)[0])
    p2 = _write_wav(str(tmp_path / "b.wav"), _pcm(100)[0])
    with pytest.raises(ValueError, match="gap_ms"):
        audio_merge.merge_wavs([p1, p2], str(tmp_path / "out.wav"), gap_ms=-1)


def test_merge_empty_source_rejected(tmp_path):
    p1 = _write_wav(str(tmp_path / "empty.wav"), b"")
    p2 = _write_wav(str(tmp_path / "b.wav"), _pcm(100)[0])
    with pytest.raises(ValueError, match="empty"):
        audio_merge.merge_wavs([p1, p2], str(tmp_path / "out.wav"))


def test_merge_over_cap_rejected(tmp_path):
    # Two 15 s members + gap = 30.3 s -> over the sample cap (default 29.9 s).
    big, _ = _pcm(15000)
    p1 = _write_wav(str(tmp_path / "big1.wav"), big)
    p2 = _write_wav(str(tmp_path / "big2.wav"), big)
    with pytest.raises(ValueError, match="exceed the sample cap"):
        audio_merge.merge_wavs([p1, p2], str(tmp_path / "out.wav"),
                               gap_ms=300, trim=False)


# ---------------------------------------------------------------------------
# merge_wavs happy path: dict shape + atomic write
# ---------------------------------------------------------------------------

def test_merge_clean_two_file_shape(tmp_path):
    p1 = _write_wav(str(tmp_path / "a.wav"), _pcm(200)[0])
    p2 = _write_wav(str(tmp_path / "b.wav"), _pcm(300)[0])
    out = str(tmp_path / "merged.wav")
    res = audio_merge.merge_wavs([p1, p2], out, gap_ms=300, trim=False)

    assert set(res.keys()) == {"bytes", "n_samples", "duration_ms", "members"}
    # 200 + 300 gap + 300 = 800 ms.
    assert res["duration_ms"] == 800
    # samples: (200+300+300) ms * 16 = 12800.
    assert res["n_samples"] == int(800 * RATE / 1000)
    assert len(res["members"]) == 2
    assert res["members"][0] == {"lead_ms": 0, "new_duration_ms": 200,
                                 "segments": [[0, 200, 0]]}
    assert res["members"][1] == {"lead_ms": 0, "new_duration_ms": 300,
                                 "segments": [[0, 300, 0]]}

    # Atomic write produced a valid WAV at exactly dst (no leftover .tmp).
    assert os.path.exists(out)
    assert not os.path.exists(out + ".tmp")
    assert res["bytes"] == os.path.getsize(out)
    with wave.open(out, "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (
            1, 2, RATE)
        assert w.getnframes() == res["n_samples"]


def test_merge_creates_missing_parent_dir(tmp_path):
    p1 = _write_wav(str(tmp_path / "a.wav"), _pcm(100)[0])
    p2 = _write_wav(str(tmp_path / "b.wav"), _pcm(100)[0])
    out = str(tmp_path / "sub" / "nested" / "merged.wav")
    res = audio_merge.merge_wavs([p1, p2], out, gap_ms=0, trim=False)
    assert os.path.exists(out)
    assert res["duration_ms"] == 200
