"""Tests for StreamSession: the append-only final-emission mechanics (the core
correctness claim) and the full PCM → partial → final loop with fake decoders.

Async coroutines are driven directly with asyncio.run() so no pytest-asyncio
plugin is required.
"""

import asyncio

import numpy as np

from streaming_session import StreamConfig, StreamSession
from streaming_vad import EnergyEndpointer


def _make_session(*, postprocess, decode_partial=None, decode_final=None, cfg=None):
    msgs: list[dict] = []

    async def emit(m):
        msgs.append(m)

    async def _dp(audio, prompt):
        return []

    async def _df(audio, prompt):
        return ("", [])

    s = StreamSession(
        config=cfg or StreamConfig(),
        endpointer=EnergyEndpointer(),
        decode_partial=decode_partial or _dp,
        decode_final=decode_final or _df,
        postprocess=postprocess,
        emit=emit,
    )
    return s, msgs


# ---- committed/tail emission mechanics ------------------------------------

def test_committed_is_append_only_and_document_equals_postprocess():
    """Across successive finalizes the committed prefix only ever grows (never
    rewrites earlier text), and committed+tail always equals the post-processed
    whole document."""
    s, msgs = _make_session(postprocess=lambda raw: raw)  # identity pipeline

    async def run():
        s.raw_confirmed = "der patient hat fieber."
        await s._emit_document(s.postprocess(s.raw_confirmed))
        s.raw_confirmed += " blutdruck normal"          # no terminator yet → tail
        await s._emit_document(s.postprocess(s.raw_confirmed))
        s.raw_confirmed += "."                            # terminator (commits on close)
        await s._emit_document(s.postprocess(s.raw_confirmed))
        await s.close()

    asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]
    committeds = [m["committed"] for m in finals]
    # append-only: each committed extends the previous, never rewrites it.
    for a, b in zip(committeds, committeds[1:]):
        assert b.startswith(a), f"committed rewritten: {a!r} -> {b!r}"
    # committed+tail reconstructs the post-processed document at every step.
    for m in finals:
        assert m["committed"] + m["tail"] in ("der patient hat fieber.",
                                              "der patient hat fieber. blutdruck normal",
                                              "der patient hat fieber. blutdruck normal.")
    # the final committed (after close) equals the whole post-processed transcript.
    assert committeds[-1] == "der patient hat fieber. blutdruck normal."
    # "blutdruck" is shown live (in some tail) but never committed mid-sentence
    # until it stabilises on close.
    assert all("blutdruck" not in c for c in committeds[:-1])
    assert any("blutdruck" in m["tail"] for m in finals)


def test_committed_never_holds_half_resolved_dictation_phrase():
    """A multi-word dictation phrase split across utterances ('neue' then 'zeile')
    is never committed half-resolved — the literal 'neue' stays out of committed
    text until the phrase completes (it may appear only in the provisional tail)."""
    def pp(raw):                       # toy dictation map: phrase → newline
        return raw.replace("neue zeile", "\n")

    s, msgs = _make_session(postprocess=pp)

    async def run():
        s.raw_confirmed = "bla bla neue"              # incomplete phrase, no terminator
        await s._emit_document(s.postprocess(s.raw_confirmed))
        s.raw_confirmed = "bla bla neue zeile text."  # phrase completes + terminator
        await s._emit_document(s.postprocess(s.raw_confirmed))
        await s.close()

    asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]
    committeds = [m["committed"] for m in finals]
    assert all("neue" not in c for c in committeds)   # literal "neue" never committed
    assert committeds[-1] == "bla bla \n text."


def test_close_commits_unterminated_tail():
    """An utterance with no sentence terminator is shown as a provisional tail
    during the session, then committed on close()."""
    s, msgs = _make_session(postprocess=lambda raw: raw)

    async def run():
        s.raw_confirmed = "hallo welt"                # no terminator
        await s._emit_document(s.postprocess(s.raw_confirmed))
        pre = [m for m in msgs if m["type"] == "final"]
        # shown immediately, but provisional (not committed) — fixes the "text only
        # appears after the next utterance" bug.
        assert pre and pre[-1]["committed"] == "" and pre[-1]["tail"] == "hallo welt"
        await s.close()

    asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]
    assert finals[-1]["committed"] == "hallo welt"
    assert finals[-1]["tail"] == ""
    assert finals[-1].get("last") is True


# ---- full PCM loop --------------------------------------------------------

def _pcm(level: int, ms: int, sample_rate: int = 16000) -> bytes:
    n = sample_rate * ms // 1000
    return (np.full(n, level, dtype="<i2")).tobytes()


def test_pcm_loop_emits_partials_then_a_final_after_silence():
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100, buffer_trim_sec=100,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
    )

    async def decode_partial(audio, prompt):
        return [(0.0, 0.3, " hallo"), (0.3, 0.6, " welt")]

    async def decode_final(audio, prompt):
        return ("hallo welt.", [])

    s, msgs = _make_session(
        postprocess=lambda raw: raw, decode_partial=decode_partial,
        decode_final=decode_final, cfg=cfg,
    )

    async def run():
        await s.feed_pcm(_pcm(8000, 500))   # ~0.5 s speech (loud)
        await s.feed_pcm(_pcm(0, 400))      # ~0.4 s silence → finalize

    asyncio.run(run())
    partials = [m for m in msgs if m["type"] == "partial"]
    finals = [m for m in msgs if m["type"] == "final"]
    assert len(partials) >= 1
    # LocalAgreement commits the repeated hypothesis → committed text appears.
    assert any("welt" in m["committed"] for m in partials)
    assert len(finals) == 1
    # one finalize, no terminator-agreement yet → the text is shown as the
    # provisional tail (committed + tail reconstructs the post-processed utterance).
    assert finals[0]["committed"] + finals[0]["tail"] == "hallo welt."


def test_hard_break_resets_document_after_long_silence():
    """A silence longer than hard_break_silence_ms ends the whole grouping: emit a
    `boundary` marker (carrying the separator) and reset the accumulated document,
    without closing the connection. Fires once per quiet gap."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100, buffer_trim_sec=100,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
        hard_break_silence_ms=500, hard_break_separator="\n",
    )

    async def decode_final(audio, prompt):
        return ("hallo welt.", [])

    s, msgs = _make_session(
        postprocess=lambda raw: raw, decode_final=decode_final, cfg=cfg,
    )

    async def run():
        await s.feed_pcm(_pcm(8000, 300))   # speech → one utterance
        await s.feed_pcm(_pcm(0, 700))      # silence: finalize (192 ms) then hard break (500 ms)

    asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]
    boundaries = [m for m in msgs if m["type"] == "boundary"]
    assert len(finals) == 1                       # one finalize before the break
    assert len(boundaries) == 1                   # exactly one break (raw_confirmed guard)
    assert boundaries[0]["separator"] == "\n"
    assert s.raw_confirmed == ""                  # document reset → fresh grouping next
    assert s._committed_len == 0
    assert s._prev_processed == ""


def test_no_partial_decode_storm_during_trailing_silence():
    """Regression: trailing silence must NOT trigger a partial decode per frame.
    The old inner-pause trigger fired one (synchronous) decode per 32 ms silent
    frame, advancing the silence timer ~1 frame per decode and inflating the
    commit wait to ~20 s. Here ~1 s of silence (≈31 frames) must cost only a
    couple of decodes, not dozens."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=2000,
        min_speech_ms=64, forced_commit_sec=100, rms_gate_dbfs=-60, preroll_keep_ms=100,
    )
    calls = {"partial": 0}

    async def decode_partial(audio, prompt):
        calls["partial"] += 1
        return [(0.0, 0.2, " x")]

    async def decode_final(audio, prompt):
        return ("x.", [])

    s, msgs = _make_session(
        postprocess=lambda raw: raw, decode_partial=decode_partial,
        decode_final=decode_final, cfg=cfg,
    )

    async def run():
        await s.feed_pcm(_pcm(8000, 300))   # 0.3 s speech
        await s.feed_pcm(_pcm(0, 1000))     # 1.0 s silence (≈31 frames) → no finalize yet

    asyncio.run(run())
    # A handful of speech-phase partials only; the silence must add ~none.
    assert calls["partial"] <= 6, f"partial decode storm: {calls['partial']} decodes"
    assert [m for m in msgs if m["type"] == "final"] == []  # held (silence < commit)


def test_silence_only_input_never_finalizes_or_hallucinates():
    cfg = StreamConfig(commit_silence_ms=192, min_speech_ms=64, rms_gate_dbfs=-50)
    called = {"partial": 0, "final": 0}

    async def decode_partial(audio, prompt):
        called["partial"] += 1
        return [(0.0, 0.2, " x")]

    async def decode_final(audio, prompt):
        called["final"] += 1
        return ("x", [])

    s, msgs = _make_session(
        postprocess=lambda raw: raw, decode_partial=decode_partial,
        decode_final=decode_final, cfg=cfg,
    )
    asyncio.run(s.feed_pcm(_pcm(0, 1000)))   # 1 s of pure silence
    assert called["partial"] == 0
    assert called["final"] == 0
    assert msgs == []
