"""WebSocket endpoint for live (streaming) dictation.

`ws[s]://HOST/v1/audio/transcriptions/stream` — a second entry point alongside the
batch `POST /v1/audio/transcriptions`. It reuses the same model cache
(`_get_or_load_model`), per-model config (`cfg_for`), and post-processing pipeline
(`_postprocess_text`) — none of which are modified — and drives them through
:class:`streaming_session.StreamSession` (LocalAgreement-2 stabilized partials,
append-only post-processed finals).

Protocol (see streaming_session for the emission contract):
  client → server:
    1. first TEXT frame: JSON config
       {"type":"config","model":..,"language":..,"response_format":"json|verbose_json",
        "audio":{"format":"pcm_s16le","sample_rate":16000}}
    2. BINARY frames: raw 16 kHz mono s16le PCM  (encoded formats: phase E)
    3. control TEXT frames: {"type":"flush"} | {"type":"stop"}
  server → client:
    {"type":"ready",..} / {"type":"partial",committed,pending} /
    {"type":"final",committed,tail,last?} / {"type":"error",code,message}
    (final: ``committed`` is append-only/locked, ``tail`` is the provisional
     trailing sentence; both are full strings — the client replaces each region.)

main.py is imported lazily inside the handler to avoid the
main → streaming_routes → main import cycle.
"""

import asyncio
import json
import logging
import os
import random
import shutil
import tempfile
import uuid
import wave

import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials

import auth
import metrics
import web_common
from streaming_session import StreamConfig, StreamSession
from streaming_transport import ENCODED_FORMATS, RAW_FORMATS, make_transport
from streaming_vad import SAMPLE_RATE, make_endpointer

logger = logging.getLogger(__name__)

router = APIRouter()


async def _safe_ws_send(ws: WebSocket, message: dict) -> bool:
    """Send a JSON message, swallowing the errors raised when the peer has already
    disconnected (e.g. the page was reloaded mid-dictation). Without this, the
    session-close drain's final send hits a closed socket and uvicorn raises
    ``RuntimeError: Unexpected ASGI message 'websocket.send' after ... close``,
    surfacing as a noisy traceback. Returns False if the send was dropped."""
    try:
        await ws.send_json(message)
        return True
    except (RuntimeError, WebSocketDisconnect):
        return False


def _write_pcm16_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Write a float32 mono [-1,1] buffer to a temp 16-bit PCM WAV and return its
    path. Used to hand a streamed utterance's audio to the captures pipeline
    (which re-transcodes any source file to its canonical 16 kHz mono WAV)."""
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return path

# Active sessions, for the /stats gauge and the max-session cap (phase D).
_active_sessions: set[str] = set()

# WebSocket close codes (4000-4999 = application-defined).
_WS_UNAUTH = 4401
_WS_DISABLED = 4503
_WS_TOO_MANY = 4429


def _ws_credentials(ws: WebSocket) -> "HTTPAuthorizationCredentials | None":
    """Build bearer credentials from the WS handshake: Authorization header, or a
    ?key= query param (for clients that cannot set WS headers)."""
    header = ws.headers.get("authorization")
    if header:
        scheme, _, token = header.partition(" ")
        return HTTPAuthorizationCredentials(scheme=scheme or "Bearer", credentials=token)
    key = ws.query_params.get("key")
    if key:
        return HTTPAuthorizationCredentials(scheme="Bearer", credentials=key)
    return None


def authenticate_ws(ws: WebSocket) -> "dict | None":
    """Resolve the WS caller to a user record (or None). Reuses the canonical
    non-raising auth core — Starlette's WebSocket exposes .cookies/.headers/.state,
    so the cookie + bearer paths work unchanged. Open mode → synthetic admin."""
    return auth._resolve_user(ws, _ws_credentials(ws))


def _stream_config(cfg, ident=None) -> StreamConfig:
    # Per-identity override (ident) > global. STREAMING_* are not per-model, so
    # ident-or-global is the full resolution — no cfg_for / model_id needed.
    def g(name, default):
        key = "STREAMING_" + name
        if ident is not None and key in ident.values:
            return ident.values[key]
        return getattr(cfg, key, default)
    return StreamConfig(
        sample_rate=SAMPLE_RATE,
        # Public config keys (the g("…") suffix, after STREAMING_) may differ from
        # the internal StreamConfig field names — this adapter is the seam.
        min_chunk_ms=int(g("PARTIAL_INTERVAL_MS", 1000)),
        min_speech_ms=int(g("GATE_MIN_SPEECH_MS", 500)),
        vad_min_silence_ms=int(g("VAD_INNER_SILENCE_MS", 700)),
        commit_silence_ms=int(g("VAD_OUTER_SILENCE_MS", 1200)),
        hard_break_silence_ms=int(g("HARD_BREAK_SILENCE_MS", 5000)),
        hard_break_separator=str(g("HARD_BREAK_SEPARATOR", "")),
        forced_commit_sec=float(g("FORCED_COMMIT_SEC", 25.0)),
        buffer_trim_sec=float(g("BUFFER_TRIM_SEC", 15.0)),
        buffer_trim_keep_sec=float(g("BUFFER_TRIM_KEEP_SEC", 10.0)),
        rms_gate_dbfs=float(g("GATE_RMS_DBFS", -42.0)),
        prompt_words=int(g("PROMPT_WORDS", 200)),
    )


def _build_transcribe_kwargs(main, model_name: str, *, final: bool,
                             prompt: str, want_words: bool,
                             language: str = "", model_obj=None,
                             overrides=None, ident=None) -> dict:
    """Assemble model.transcribe kwargs for a streaming decode.

    Both partial and final decodes pull the SAME per-model config as the batch
    route (via ``main.assemble_transcribe_kwargs``) — hotwords, suppress_tokens/
    chars, prepend/append_punctuations, penalties, thresholds — so streaming
    output matches batch. The FINAL decode (the full committed utterance) is the
    batch decode's exact analogue and uses the assembler verbatim. The PARTIAL
    decode keeps all those quality knobs (they're ~free) and overrides ONLY the
    handful that must stay streaming-specific for latency/stability (see below).

    ``language`` is the per-connection language from the config handshake; it wins
    over the model's DEFAULT_LANGUAGE. Pinning it avoids faster-whisper auto-
    detecting per (short, growing) partial buffer, which is unstable — a brief
    German chunk can be mis-detected as e.g. Swedish."""
    cfg_for = main.cfg_for
    cfg = main.cfg
    lang = (language or cfg_for(model_name, "DEFAULT_LANGUAGE", ident) or "").strip()
    _vad_filter = cfg_for(model_name, "VAD_FILTER", ident)
    vad_parameters = dict(
        min_silence_duration_ms=cfg_for(model_name, "VAD_MIN_SILENCE_MS", ident),
        speech_pad_ms=cfg_for(model_name, "VAD_SPEECH_PAD_MS", ident),
        threshold=cfg_for(model_name, "VAD_THRESHOLD", ident),
    ) if _vad_filter else None
    _prompt = prompt or cfg_for(model_name, "DEFAULT_PROMPT", ident)
    kwargs = main.assemble_transcribe_kwargs(
        model_name, model_obj,
        language=lang, temperature=0.0,
        vad_filter=_vad_filter, vad_parameters=vad_parameters,
        want_word_ts=want_words, initial_prompt=(_prompt or None),
        overrides=overrides, ident=ident,
    )
    if final:
        # Full-utterance decode — identical to the batch route. Nothing to change.
        return kwargs
    # PARTIAL decode: keep every quality knob the final/batch decode applies
    # (hotwords, suppress_tokens/chars, punctuation, penalties, thresholds — all
    # ~free: logit masks / beam shaping / post-processing). Override ONLY the few
    # knobs that genuinely matter for a fast, stable per-partial pass on a growing
    # buffer:
    #   • beam_size → STREAMING_PARTIAL_BEAM: the one real speed knob — partials
    #     re-decode the growing buffer many times per utterance, so the final's
    #     larger beam would roughly double that work.
    #   • temperature → STREAMING_PARTIAL_TEMPERATURE (default 0.0, no ladder): a
    #     fallback re-decode is a mid-stream latency spike; only the final needs
    #     the per-model TEMPERATURE ladder's robustness.
    #   • condition_on_previous_text → STREAMING_PARTIAL_CONDITION_ON_PREVIOUS_TEXT
    #     (default False): documented to loop on German finetunes, worst on short/
    #     growing buffers; the final uses the per-model CONDITION_ON_PREVIOUS_TEXT.
    #   • vad_filter off: the stream is already gated by our own VAD.
    kwargs["beam_size"] = int(cfg_for(model_name, "STREAMING_PARTIAL_BEAM", ident))
    kwargs["temperature"] = float(cfg_for(model_name, "STREAMING_PARTIAL_TEMPERATURE", ident))
    kwargs["condition_on_previous_text"] = bool(
        cfg_for(model_name, "STREAMING_PARTIAL_CONDITION_ON_PREVIOUS_TEXT", ident))
    kwargs["vad_filter"] = False
    kwargs["vad_parameters"] = None
    kwargs.setdefault("no_repeat_ngram_size", 3)  # greedy-safe loop guard
    return kwargs


@router.websocket("/v1/audio/transcriptions/stream")
async def transcribe_stream(ws: WebSocket) -> None:
    import main  # lazy — avoids the import cycle and is loaded by connect time

    cfg = main.cfg
    if not getattr(cfg, "STREAMING_ENABLED", True):
        await ws.close(code=_WS_DISABLED)
        return
    user = authenticate_ws(ws)
    if user is None:
        await ws.close(code=_WS_UNAUTH)
        return
    max_sessions = int(getattr(cfg, "STREAMING_MAX_SESSIONS", 10))
    if len(_active_sessions) >= max_sessions:
        await ws.close(code=_WS_TOO_MANY)
        return

    await ws.accept()
    session_id = uuid.uuid4().hex
    _active_sessions.add(session_id)
    metrics.in_flight_transcriptions += 1
    session: "StreamSession | None" = None
    transport = None
    try:
        # ---- handshake: first message is the JSON config (binary → defaults) ----
        first = await ws.receive()
        if first.get("type") == "websocket.disconnect":
            return
        conf = {}
        pending_audio: "bytes | None" = None
        if first.get("text") is not None:
            try:
                conf = json.loads(first["text"])
            except (ValueError, TypeError):
                conf = {}
        elif first.get("bytes") is not None:
            pending_audio = first["bytes"]

        model_req = conf.get("model") or "whisper-1"
        req_language = (conf.get("language") or "").strip()
        response_format = conf.get("response_format", "json")
        # Per-connection initial prompt (the client's "Vocabulary / prompt"). Empty
        # → fall back to the model's DEFAULT_PROMPT, then to None — same as batch.
        req_prompt = (conf.get("prompt") or "").strip()
        # Optional per-request decode overrides (the client's "decode overrides").
        # Applied to the FINAL decode (the batch analogue); partials keep their
        # streaming-specific beam/temp/condition/vad knobs (see _build_transcribe_kwargs).
        req_overrides = conf.get("decode_overrides")
        if not isinstance(req_overrides, dict):
            req_overrides = {}
        include_words = response_format == "verbose_json"
        audio_fmt = (conf.get("audio") or {}).get("format", "pcm_s16le")
        if audio_fmt not in RAW_FORMATS and audio_fmt not in ENCODED_FORMATS:
            await ws.send_json({"type": "error", "code": "unsupported_format",
                                "message": f"audio format {audio_fmt!r} not supported "
                                           f"(raw: {sorted(RAW_FORMATS)}, "
                                           f"encoded via ffmpeg: {sorted(ENCODED_FORMATS)})"})
            await ws.close()
            return
        # Human-readable transport label for the per-utterance log block.
        audio_source_label = (
            f"{audio_fmt} @ {SAMPLE_RATE} Hz mono (raw PCM, WebSocket)"
            if audio_fmt in RAW_FORMATS
            else f"{audio_fmt} → {SAMPLE_RATE} Hz mono (ffmpeg decode, WebSocket)")

        final_model = main._resolve_model_name(model_req)
        partial_cfg = getattr(cfg, "STREAMING_PARTIAL_MODEL", "") or ""
        partial_model_name = partial_cfg or final_model
        try:
            final_model_obj = await main._get_or_load_model(final_model)
            partial_model_obj = (
                final_model_obj if partial_model_name == final_model
                else await main._get_or_load_model(partial_model_name)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[stream %s] model load failed: %s", session_id[:8], exc)
            await ws.send_json({"type": "error", "code": "model_load_failed",
                                "message": str(exc)})
            await ws.close()
            return

        # Resolve the caller's effective per-identity config ONCE for this
        # connection. ident is built with final_model (per-model rule folding +
        # output wrappers + postprocess use final_model); identity scalar
        # overrides are model-independent, so they apply to the partial decode
        # too via cfg_for's ident layer.
        ident = main.build_ident(user, final_model)
        gate_final_words = bool(main.cfg_for(final_model, "WORD_TIMESTAMPS_ENABLED", ident))
        gate_partial_words = bool(main.cfg_for(partial_model_name, "WORD_TIMESTAMPS_ENABLED", ident))

        # Locked language / prompt: the admin value stands; the client's
        # handshake value is ignored (and surfaced in the ready frame). Locked
        # decode_overrides keys are dropped in the assembler; record them here.
        overrides_ignored = sorted(k for k in req_overrides
                                   if k in ident.locked_client_keys)
        if "DEFAULT_LANGUAGE" in ident.locked:
            _locked_lang = main.cfg_for(final_model, "DEFAULT_LANGUAGE", ident) or ""
            if req_language and req_language != _locked_lang:
                overrides_ignored.append("language")
            req_language = _locked_lang
        if "DEFAULT_PROMPT" in ident.locked:
            _locked_prompt = main.cfg_for(final_model, "DEFAULT_PROMPT", ident) or ""
            if req_prompt and req_prompt != _locked_prompt:
                overrides_ignored.append("prompt")
            req_prompt = _locked_prompt

        async def _transcribe(model_obj, audio, kwargs):
            loop = asyncio.get_running_loop()

            def work():
                segs, info = model_obj.transcribe(audio, **kwargs)
                return list(segs), info

            # Shared GPU limiter (same object the batch route uses).
            async with main.get_inference_semaphore():
                return await loop.run_in_executor(None, work)

        async def decode_partial(audio, prompt):
            kwargs = _build_transcribe_kwargs(
                main, partial_model_name, final=False, prompt=prompt,
                want_words=gate_partial_words, language=req_language,
                model_obj=partial_model_obj, overrides=req_overrides, ident=ident)
            segs, _info = await _transcribe(partial_model_obj, audio, kwargs)
            if gate_partial_words:
                words = [(w.start, w.end, w.word)
                         for seg in segs for w in (getattr(seg, "words", None) or [])]
                if words:
                    return words
            # fallback: segment-level units (coarser LocalAgreement granularity)
            return [(seg.start, seg.end, seg.text) for seg in segs]

        # Captures are eligible only when the model allows the DTW word path
        # (per-model WORD_TIMESTAMPS_ENABLED) — same gate as the batch route.
        cap_enabled = bool(getattr(cfg, "CAPTURE_RECORDINGS_ENABLED", False)) and gate_final_words
        # The final decode stashes its faster-whisper info / segment diagnostics /
        # word list here so on_final (serialized right after, under the session
        # lock) can build the rich log block + the capture row without re-decoding.
        last_decode: dict = {}

        async def decode_final(audio, prompt):
            kwargs = _build_transcribe_kwargs(
                main, final_model, final=True, prompt=prompt,
                want_words=gate_final_words, language=req_language,
                model_obj=final_model_obj, overrides=req_overrides, ident=ident)
            segs, info = await _transcribe(final_model_obj, audio, kwargs)
            raw = "".join(seg.text for seg in segs)
            words_out: list[dict] = []
            seg_diag: list[dict] = []
            for i, seg in enumerate(segs):
                seg_diag.append({
                    "id": i, "start": seg.start, "end": seg.end,
                    "alp": getattr(seg, "avg_logprob", 0.0),
                    "nsp": getattr(seg, "no_speech_prob", 0.0),
                    "cr": getattr(seg, "compression_ratio", 1.0),
                    "temp": getattr(seg, "temperature", 0.0),
                    "text": seg.text,
                })
                for w in (getattr(seg, "words", None) or []):
                    words_out.append({"word": w.word, "start": w.start, "end": w.end})
            last_decode.clear()
            last_decode.update(info=info, seg_diag=seg_diag, kwargs=kwargs)
            return raw, words_out

        def postprocess(raw_text):
            return main._postprocess_text(raw_text, model_name=final_model, ident=ident)

        # Output wrappers: the prefix sits at the very start of the document, the
        # suffix only on the final flush. committed/tail are full authoritative
        # strings (the client replaces each region), so re-applying the prefix on
        # every final is correct — it never accumulates.
        out_prefix = main.cfg_for(final_model, "OUTPUT_PREFIX", ident) or ""
        out_suffix = main.cfg_for(final_model, "OUTPUT_SUFFIX", ident) or ""

        async def emit(message):
            if message.get("type") == "final":
                if not include_words:
                    message.pop("words", None)   # word timestamps only for verbose_json
                if out_prefix:
                    if message.get("committed"):
                        message["committed"] = out_prefix + message["committed"]
                    elif message.get("tail"):
                        message["tail"] = out_prefix + message["tail"]
                if out_suffix and message.get("last"):
                    message["committed"] = (message.get("committed") or "") + out_suffix
            # Peer may have vanished mid-drain (page reload during dictation): the
            # socket is already closed, so swallow the send. Side-effects
            # (metrics/trace/captures) still ran in on_final.
            await _safe_ws_send(ws, message)

        def _maybe_capture(rid, info, raw_text, final_text, words, fw_info):
            """Persist a fine-tuning capture for this utterance, mirroring the batch
            route's eligibility gate (sampling / count cap / size / duration / disk)."""
            try:
                import captures_store as _cap_store
                audio = info["audio"]
                pcm_bytes = int(getattr(audio, "size", 0)) * 2
                cap_max = int(getattr(cfg, "CAPTURES_MAX", 5000))
                hard_lim = int(getattr(cfg, "CAPTURE_RECORDINGS_AUDIO_BYTES_HARD_LIMIT", 100_000_000))
                sample = float(getattr(cfg, "CAPTURE_RECORDINGS_SAMPLE_RATE", 1.0))
                if not (_cap_store.count() < cap_max and pcm_bytes < hard_lim
                        and random.random() < sample):
                    return None
                dur = float(info["audio_dur"])
                min_s = float(getattr(cfg, "CAPTURE_RECORDINGS_MIN_DURATION_SEC", 0.5))
                max_s = float(getattr(cfg, "CAPTURE_RECORDINGS_MAX_DURATION_SEC", 600.0))
                if not (min_s <= dur <= max_s):
                    logger.info("[stream %s] capture skipped duration %.1fs (window %.1f-%.1f)",
                                session_id[:8], dur, min_s, max_s)
                    return None
                try:
                    free = shutil.disk_usage(cfg.CAPTURES_DIR).free
                except OSError:
                    free = 1 << 40
                if free <= 1_000_000_000:
                    logger.warning("[stream %s] capture skipped: low disk (%.0f MB free)",
                                   session_id[:8], free / (1024 * 1024))
                    return None
                training_text = main._postprocess_text(
                    raw_text, model_name=final_model, trace=None,
                    extra_excludes=getattr(cfg, "CAPTURES_PIPELINE_RULES_EXCLUDE", None))
                wav_path = _write_pcm16_wav(audio)
                try:
                    return _cap_store.create_capture(
                        audio_src_path=wav_path, request_id=rid, model=final_model,
                        language=(getattr(fw_info, "language", None) or req_language or ""),
                        duration_seconds=dur, raw=raw_text, final=final_text,
                        text_for_training=training_text, words=words, segments=[],
                        user_id=user.get("user_id"))
                finally:
                    try:
                        os.unlink(wav_path)
                    except OSError:
                        pass
            except Exception as _ce:  # noqa: BLE001 — never let a capture failure break dictation
                logger.warning("[stream %s] capture failed: %s", session_id[:8], _ce)
                return None

        async def on_final(info):
            # One finalized utterance == one mini-transcription: replicate the batch
            # route's per-request side-effects (rich log block, durable trace for
            # /quick-config + /reports, capture, metrics) so streaming has parity.
            rid = uuid.uuid4().hex
            raw_text = info["raw_text"] or ""
            words = info.get("words") or []
            dec = last_decode
            fw_info = dec.get("info")
            seg_diag = dec.get("seg_diag", [])
            kwargs = dec.get("kwargs", {})

            steps: "list | None" = [] if getattr(cfg, "TRACE_ENABLED", False) else None
            final_text = main._postprocess_text(raw_text, model_name=final_model, trace=steps)

            captured_id = None
            if cap_enabled and raw_text.strip():
                captured_id = _maybe_capture(rid, info, raw_text, final_text, words, fw_info)

            # Rich diagnostic block — same formatter the batch route uses, so the
            # VAD-ate-audio / empty-output / pipeline-step diagnostics show up for
            # streaming too. file_label marks it as a streamed utterance.
            try:
                logger.info(main._format_request_block(
                    file_label=f"stream {session_id[:8]} utt#{info['utterance']}  "
                               f"({info['audio_dur']:.2f}s, {response_format})",
                    model_name=final_model, info=fw_info, kwargs=kwargs,
                    seg_diag=seg_diag, raw=raw_text, final=final_text,
                    steps=steps, request_id=rid, captured_id=captured_id,
                    endpoint="/v1/audio/transcriptions/stream",
                    audio_source=audio_source_label,
                    ident=ident, overrides_ignored=overrides_ignored))
            except Exception as _le:  # noqa: BLE001
                logger.warning("[stream %s] log block failed: %s", session_id[:8], _le)

            # Durable trace → /quick-config recent-transcriptions + autocomplete + SSE.
            # source='stream' tags the row so /quick-config can chip it as live
            # dictation vs a file-upload (batch) transcription.
            try:
                import quick_config_state
                quick_config_state.record_trace(
                    request_id=rid, model=final_model, raw=raw_text,
                    steps=steps if steps is not None else [], final=final_text,
                    language=(getattr(fw_info, "language", None) or req_language or None),
                    source="stream", user_id=user.get("user_id"))
            except Exception as _qe:  # noqa: BLE001
                logger.error("[stream %s] record_trace failed: %s", session_id[:8], _qe)

            # Timing/usage half — UPSERTs onto the same request_id row as record_trace.
            metrics.record_transcription(
                model=final_model, audio_dur=info["audio_dur"],
                proc_dur=info["proc_dur"], status="ok",
                words=len(final_text.split()),
                request_id=rid, user_id=user.get("user_id"), key_id=user.get("key_id"))

        session = StreamSession(
            config=_stream_config(cfg, ident),
            endpointer=make_endpointer(
                main.cfg_for(final_model, "STREAMING_VAD_BACKEND", ident),
                threshold=float(main.cfg_for(final_model, "STREAMING_VAD_THRESHOLD", ident)),
                energy_dbfs=float(main.cfg_for(final_model, "STREAMING_GATE_RMS_DBFS", ident)),
            ),
            decode_partial=decode_partial,
            decode_final=decode_final,
            postprocess=postprocess,
            emit=emit,
            base_prompt=(req_prompt or main.cfg_for(final_model, "DEFAULT_PROMPT", ident) or ""),
            on_final=on_final,
        )

        # All session mutation is serialized: the ffmpeg reader task and the
        # control-message handler both touch the session across await points.
        session_lock = asyncio.Lock()

        async def sink(pcm: bytes):
            async with session_lock:
                await session.feed_pcm(pcm)

        transport = make_transport(audio_fmt, sink, sample_rate=SAMPLE_RATE)
        await transport.start()

        ready_msg = {
            "type": "ready", "session": session_id, "model": final_model,
            "partial_model": partial_model_name, "sample_rate": SAMPLE_RATE,
            "response_format": response_format, "audio_format": audio_fmt,
        }
        # Surface (never silently drop) any handshake override the admin config
        # locked out, so the client can see why it had no effect.
        if overrides_ignored:
            ready_msg["overrides_ignored"] = overrides_ignored
        await ws.send_json(ready_msg)
        if pending_audio:
            await transport.feed(pending_audio)

        # ---- main receive loop ----
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await transport.feed(msg["bytes"])
                elif msg.get("text") is not None:
                    try:
                        ctrl = json.loads(msg["text"])
                    except (ValueError, TypeError):
                        continue
                    kind = ctrl.get("type")
                    if kind == "flush":
                        async with session_lock:
                            await session.flush_utterance()
                    elif kind == "stop":
                        break
        finally:
            await transport.aclose()      # flush any encoded tail into the session
        async with session_lock:
            await session.close()
        try:
            await ws.send_json({"type": "closing"})
            await ws.close()
        except RuntimeError:
            pass
    except WebSocketDisconnect:
        if transport is not None:
            try:
                await transport.aclose()
            except Exception:  # noqa: BLE001
                pass
        if session is not None:
            try:
                await session.close()
            except Exception:  # noqa: BLE001 — peer already gone
                pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("[stream %s] error: %s", session_id[:8], exc)
        try:
            await ws.send_json({"type": "error", "code": "internal", "message": str(exc)})
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
    finally:
        metrics.in_flight_transcriptions -= 1
        _active_sessions.discard(session_id)


# --- Demo page ---------------------------------------------------------------
_DICTATE_HTML_PATH = os.path.join(os.path.dirname(__file__), "static", "dictate.html")


@router.get("/dictate", response_class=HTMLResponse,
            dependencies=[Depends(web_common.require_user_webui_host)])
async def dictate_page() -> HTMLResponse:
    """Minimal browser demo for the streaming endpoint: mic → 16 kHz PCM → WS,
    rendering stabilized partials + append-only finals. Gated by the user-WebUI
    host allowlist (loopback always allowed); the WebSocket enforces API auth."""
    try:
        with open(_DICTATE_HTML_PATH, "r", encoding="utf-8") as fh:
            return HTMLResponse(fh.read())
    except OSError as exc:
        logger.error("[dictate] cannot read demo page: %s", exc)
        return HTMLResponse("<h1>dictate demo unavailable</h1>", status_code=500)
