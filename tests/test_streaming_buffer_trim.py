"""Trim-safety: a continuous utterance longer than buffer_trim_sec must not
lose its opening.

_maybe_trim cuts the front of the audio buffer (anchored at a committed word)
to bound partial-decode latency. The cut words' audio is gone, so the final
decode can only re-hear the remaining buffer — its result used to REPLACE the
whole utterance, deleting the already-committed-and-shown opening (observed
live: a 16 s dictation losing its first clause). The fix banks the cut words'
text, folds it into the rolling prompt (seam context), and prepends it to the
final decode's result; captures pair the buffer audio with the buffer-aligned
text only.
"""

import asyncio

import numpy as np

from streaming_session import StreamConfig, StreamSession
from streaming_vad import EnergyEndpointer

SR = 16000
WORD_SEC = 0.2  # deterministic word grid: word i spans [i*0.2, (i+1)*0.2)


def _pcm(level: int, ms: int) -> bytes:
    return (np.full(SR * ms // 1000, level, dtype="<i2")).tobytes()


def _grid_words(start_s: float, end_s: float):
    """Absolute word grid over the utterance timeline."""
    first = int(round(start_s / WORD_SEC))
    last = int(end_s / WORD_SEC)
    return [(i * WORD_SEC, (i + 1) * WORD_SEC, f" w{i}")
            for i in range(first, last)]


def _run_long_utterance(speech_ms: int):
    """Drive a session through one continuous utterance of speech_ms, with a
    buffer trim configured to fire mid-utterance. The fake decoders transcribe
    exactly the buffer they receive (like the real model: they cannot re-hear
    trimmed audio)."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100,
        buffer_trim_sec=2.0, buffer_trim_keep_sec=1.0,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
    )
    finals_meta: list[dict] = []

    async def on_final(info):
        finals_meta.append(info)

    msgs: list[dict] = []

    async def emit(m):
        msgs.append(m)

    session_box: list[StreamSession] = []

    async def decode_partial(audio, prompt):
        off = session_box[0]._buffer_offset
        dur = audio.shape[0] / SR
        return [(a - off, b - off, t) for a, b, t in _grid_words(off, off + dur)]

    async def decode_final(audio, prompt):
        off = session_box[0]._buffer_offset
        dur = audio.shape[0] / SR
        raw = "".join(t for _, _, t in _grid_words(off, off + dur))
        return (raw, [], False)

    s = StreamSession(
        config=cfg, endpointer=EnergyEndpointer(),
        decode_partial=decode_partial, decode_final=decode_final,
        postprocess=lambda raw: raw, emit=emit, on_final=on_final,
    )
    session_box.append(s)

    async def run():
        await s.feed_pcm(_pcm(8000, speech_ms))
        await s.feed_pcm(_pcm(0, 400))  # silence → finalize

    asyncio.run(run())
    return s, msgs, finals_meta


def test_trim_preserves_committed_opening_in_final_document():
    s, msgs, finals_meta = _run_long_utterance(speech_ms=3500)
    # Sanity: the trim actually fired (otherwise this test regressed to the
    # short-utterance case and asserts nothing).
    assert s._buffer_offset == 0.0  # reset after finalize
    assert finals_meta, "no final emitted"
    info = finals_meta[-1]
    # raw_text (document) must contain the utterance's FIRST word — the audio
    # for it was trimmed away mid-utterance, so only the banked committed text
    # can supply it.
    assert " w0" in info["raw_text"], (
        f"opening lost after buffer trim: {info['raw_text']!r}")
    # No duplication at the seam: every grid word appears exactly once.
    words = info["raw_text"].split()
    assert len(words) == len(set(words)), f"duplicated words: {info['raw_text']!r}"
    # The words are in order and contiguous (w0, w1, ..., wN).
    assert words == [f"w{i}" for i in range(len(words))]
    # The emitted document (committed + tail of the last final) matches.
    finals = [m for m in msgs if m["type"] == "final"]
    doc = finals[-1]["committed"] + finals[-1]["tail"]
    assert "w0" in doc

    # Capture alignment: audio_text pairs with the (trimmed) buffer, so it must
    # NOT contain the banked opening, and raw_text = banked prefix + audio_text.
    assert " w0" not in info["audio_text"]
    assert info["raw_text"].endswith(info["audio_text"])
    prefix = info["raw_text"][:len(info["raw_text"]) - len(info["audio_text"])]
    assert prefix.split()[0] == "w0"

    # Duration split: audio_dur is buffer-only (pairs with the capture audio),
    # utterance_dur restores the trimmed seconds — it must cover the ~3.5 s of
    # speech actually fed, while the trimmed buffer alone cannot.
    assert info["utterance_dur"] > info["audio_dur"]
    assert info["utterance_dur"] >= 3.2
    assert info["audio_dur"] < 3.2


def test_trim_folds_cut_words_into_rolling_prompt():
    """After a trim the decodes see a mid-sentence buffer; the cut words must
    ride the prompt so the seam decodes with context."""
    cfg = StreamConfig(
        min_chunk_ms=96, vad_min_silence_ms=96, commit_silence_ms=192,
        min_speech_ms=64, forced_commit_sec=100,
        buffer_trim_sec=2.0, buffer_trim_keep_sec=1.0,
        rms_gate_dbfs=-60, preroll_keep_ms=100,
    )
    final_prompts: list[str] = []
    session_box: list[StreamSession] = []

    async def decode_partial(audio, prompt):
        off = session_box[0]._buffer_offset
        dur = audio.shape[0] / SR
        return [(a - off, b - off, t) for a, b, t in _grid_words(off, off + dur)]

    async def decode_final(audio, prompt):
        final_prompts.append(prompt)
        return (" tail.", [], False)

    async def emit(m):
        pass

    s = StreamSession(
        config=cfg, endpointer=EnergyEndpointer(),
        decode_partial=decode_partial, decode_final=decode_final,
        postprocess=lambda raw: raw, emit=emit,
    )
    session_box.append(s)

    async def run():
        await s.feed_pcm(_pcm(8000, 3500))
        await s.feed_pcm(_pcm(0, 400))

    asyncio.run(run())
    assert final_prompts, "final decode never ran"
    assert "w0" in final_prompts[-1], (
        f"trimmed words missing from decode prompt: {final_prompts[-1]!r}")


def test_short_utterance_unchanged_no_trim():
    """Below buffer_trim_sec nothing is banked: raw_text == audio_text, the
    final decode's text stands alone, and both durations coincide (pre-fix
    behaviour preserved)."""
    s, msgs, finals_meta = _run_long_utterance(speech_ms=800)
    info = finals_meta[-1]
    assert info["raw_text"] == info["audio_text"]
    assert info["utterance_dur"] == info["audio_dur"]
    assert s._trimmed_text == ""
