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
        # Vectorized contiguous non-zero runs (kept fast for multi-second clips).
        nz = np.abs(audio) > 1e-6
        if not nz.any():
            return []
        d = np.diff(nz.astype(np.int8))
        starts = list(np.where(d == 1)[0] + 1)
        ends = list(np.where(d == -1)[0] + 1)
        if nz[0]:
            starts = [0] + starts
        if nz[-1]:
            ends = ends + [len(audio)]
        return [{"start": int(s), "end": int(e)} for s, e in zip(starts, ends)]

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
    # Body-only: outer edges get NO pad (merge adds the uniform outer margin);
    # internal span sides keep 50ms; the internal gap is capped at 300ms.
    # span1 [200,550) (speech 200-500 + 50 inner-right), span2 [1450,1800)
    # (50 inner-left + speech 1500-1800).
    assert res["segments"] == [[200, 550, 0], [1450, 1800, 650]]
    # 1000ms = span1 (350) + capped gap (300) + span2 (350).
    assert res["new_duration_ms"] == 1000
    # Leading silence fully removed (info only).
    assert res["lead_ms"] == 200
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
    # Body-only: each member is its 300ms speech span (no edge pad).
    assert res["members"][0]["new_duration_ms"] == 300
    assert res["members"][1]["new_duration_ms"] == 300
    # Uniform layout: edge(50) + body(300) + join(300) + body(300) + edge(50).
    assert res["duration_ms"] == 1000
    # Absolute offsets: m0 after the leading edge; m1 after m0 + join.
    assert res["members"][0]["offset_ms"] == 50
    assert res["members"][1]["offset_ms"] == 650


def test_merge_accepts_raw_over_cap_when_trimmed_fits(monkeypatch):
    # Two members of ~15 s raw each (mostly silence) → raw sum ~30 s exceeds the
    # 28 s cap, but each trims to ~13 s so the merged WAV (~26.5 s) fits. This is
    # exactly the proposer/batch case that used to be rejected by the raw cap.
    _install_fake_vad(monkeypatch)
    m1, _ = _pcm([("sil", 1000), ("speech", 13000), ("sil", 1000)])
    m2, _ = _pcm([("sil", 1000), ("speech", 13000), ("sil", 1000)])
    with tempfile.TemporaryDirectory() as d:
        p1, p2 = f"{d}/m1.wav", f"{d}/m2.wav"
        out = f"{d}/merged.wav"
        _write_wav(p1, m1)
        _write_wav(p2, m2)
        res = audio_merge.merge_wavs(
            [p1, p2], out, gap_ms=300, trim=True,
            edge_pad_ms=50, max_internal_gap_ms=300,
        )
    # Body-only: each member is 13 s of speech; merged =
    # edge(50)+13000+join(300)+13000+edge(50) = 26400 ms.
    assert res["members"][0]["new_duration_ms"] == 13000
    assert res["duration_ms"] == 26400
    assert res["duration_ms"] <= 28_000


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


def test_build_merged_words_uniform_offset(monkeypatch):
    # New uniform-silence layout: each member carries an absolute `offset_ms`,
    # so word times are offset_ms + remapped-local (NOT cum+i*silence).
    cr = _routes()
    monkeypatch.setattr(
        cr, "_align_words_to_final",
        lambda words, final, model_name=None: [dict(w) for w in words],
    )
    members = [
        {"id": "a", "words": [{"word": "hi", "start": 0.25, "end": 0.45}],
         "final": "hi", "duration_seconds": 2.0},
        {"id": "b", "words": [{"word": "yo", "start": 1.55, "end": 1.75}],
         "final": "yo", "duration_seconds": 2.0},
    ]
    # edge 300 leading → m0 body at 300ms; m0 body 300ms; join 300 → m1 at 900ms.
    trims = {
        "a": {"lead_ms": 200, "new_duration_ms": 300,
              "segments": [[200, 500, 0]], "offset_ms": 300},
        "b": {"lead_ms": 1500, "new_duration_ms": 300,
              "segments": [[1500, 1800, 0]], "offset_ms": 900},
    }
    out = cr._build_merged_words(members, 300, member_trims=trims)
    assert len(out) == 2
    # a: start 0.25s→50ms local +300ms offset = 0.35s; end 0.45s→250ms +300 = 0.55s
    assert out[0]["start"] == pytest.approx(0.35)
    assert out[0]["end"] == pytest.approx(0.55)
    # b: start 1.55s→50ms local +900ms offset = 0.95s (NOT cum+silence);
    #    end 1.75s→250ms +900 = 1.15s
    assert out[1]["start"] == pytest.approx(0.95)
    assert out[1]["end"] == pytest.approx(1.15)


def test_align_member_words_attaches_training_tokens(monkeypatch):
    cr = _routes()

    # Fake aligner: one item per raw word, display token = target token by index.
    def fake_align(words, final, model_name=None):
        toks = (final or "").split()
        return [
            {"word": toks[i] if i < len(toks) else "",
             "start": w.get("start"), "end": w.get("end")}
            for i, w in enumerate(words)
        ]
    monkeypatch.setattr(cr, "_align_words_to_final", fake_align)

    # final applies dictation-map ("Komma"→"Komma"); training excludes it.
    m = {
        "words": [{"start": 0, "end": 1}, {"start": 1, "end": 2}],
        "final": "Komma World", "text_for_training": "comma world", "model": None,
    }
    ws = cr._align_member_words(m)
    assert ws[0]["word"] == "Komma"
    assert ws[0]["train_word"] == "comma"
    assert ws[1]["word"] == "World"
    assert ws[1]["train_word"] == "world"

    # When training == final, no separate train token is attached (ground falls
    # back to `word`).
    m2 = {"words": [{"start": 0, "end": 1}], "final": "hi",
          "text_for_training": "hi", "model": None}
    ws2 = cr._align_member_words(m2)
    assert "train_word" not in ws2[0]


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


# --------------------------------------------------------------------------
# Proposer trimmed durations + caching
# --------------------------------------------------------------------------

def test_proposer_trimmed_duration_and_caching(monkeypatch, tmp_path):
    _install_fake_vad(monkeypatch)
    import captures_merge_proposer as P
    import captures_store

    # 200ms sil | 600ms speech | 200ms sil → trimmed to speech ± 50ms pad.
    pcm, _ = _pcm([("sil", 200), ("speech", 600), ("sil", 200)])
    wav = tmp_path / "a.wav"
    _write_wav(str(wav), pcm)
    monkeypatch.setattr(captures_store, "abs_audio_path", lambda rel: str(wav))
    monkeypatch.setattr(P.cfg, "CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS", True,
                        raising=False)
    monkeypatch.setattr(P.cfg, "CAPTURES_VAD_MARGIN_GROUP_EDGE_MS", 50, raising=False)
    monkeypatch.setattr(P.cfg, "CAPTURES_VAD_MARGIN_GROUP_INTERNAL_MS", 300,
                        raising=False)
    P._TRIM_DUR_CACHE.clear()

    row = {"id": "cap1", "audio_relpath": "x", "duration_seconds": 1.0}
    d1 = P.trimmed_duration_s(row)
    # Body-only: 600ms speech (outer pad removed; merge adds the margin).
    assert 0.55 <= d1 <= 0.65

    # Second call must hit the cache — make read_pcm explode to prove it.
    def _boom(_p):
        raise AssertionError("read_pcm called on a cache hit")
    monkeypatch.setattr(audio_merge, "read_pcm", _boom)
    assert P.trimmed_duration_s(row) == d1


def test_proposer_trim_disabled_returns_raw(monkeypatch):
    import captures_merge_proposer as P
    monkeypatch.setattr(P.cfg, "CAPTURES_VAD_TRIM_ENABLED_FOR_GROUPS", False,
                        raising=False)
    P._TRIM_DUR_CACHE.clear()
    row = {"id": "c", "audio_relpath": "x", "duration_seconds": 3.5}
    assert P.trimmed_duration_s(row) == 3.5


def test_build_proposal_uses_trimmed_durations():
    import captures_merge_proposer as P
    members = [
        {"id": "a", "created_ts": 1000.0, "duration_seconds": 2.0,
         "_trim_dur_s": 1.2, "status": "", "text_for_training": "hello"},
        {"id": "b", "created_ts": 1001.0, "duration_seconds": 2.0,
         "_trim_dur_s": 1.0, "status": "", "text_for_training": "world"},
    ]
    prop = P._build_proposal(members, 0.3, 26.0, "de", "u1")
    assert prop["member_previews"][0]["duration_s"] == 1.2
    assert prop["member_previews"][1]["duration_s"] == 1.0
    # total = trimmed sum (2.2) + one 0.3s gap.
    assert abs(prop["total_duration_s"] - 2.5) < 1e-6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
