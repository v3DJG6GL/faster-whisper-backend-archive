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
    {"type":"final",text,append:true} / {"type":"error",code,message}

main.py is imported lazily inside the handler to avoid the
main → streaming_routes → main import cycle.
"""

import asyncio
import json
import logging
import os
import uuid

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


def _stream_config(cfg) -> StreamConfig:
    g = lambda name, default: getattr(cfg, "STREAMING_" + name, default)  # noqa: E731
    return StreamConfig(
        sample_rate=SAMPLE_RATE,
        min_chunk_ms=int(g("MIN_CHUNK_MS", 1000)),
        min_speech_ms=int(g("MIN_SPEECH_MS", 500)),
        vad_min_silence_ms=int(g("VAD_MIN_SILENCE_MS", 700)),
        commit_silence_ms=int(g("COMMIT_SILENCE_MS", 1200)),
        forced_commit_sec=float(g("FORCED_COMMIT_SEC", 25.0)),
        buffer_trim_sec=float(g("BUFFER_TRIM_SEC", 15.0)),
        buffer_trim_keep_sec=float(g("BUFFER_TRIM_KEEP_SEC", 10.0)),
        rms_gate_dbfs=float(g("RMS_GATE_DBFS", -42.0)),
        prompt_words=int(g("PROMPT_WORDS", 200)),
    )


def _build_transcribe_kwargs(main, model_name: str, *, final: bool,
                             prompt: str, want_words: bool,
                             language: str = "") -> dict:
    """Assemble model.transcribe kwargs from per-model config + streaming overrides.

    ``language`` is the per-connection language from the config handshake; it wins
    over the model's DEFAULT_LANGUAGE. Pinning it avoids faster-whisper auto-
    detecting per (short, growing) partial buffer, which is unstable — a brief
    German chunk can be mis-detected as e.g. Swedish."""
    cfg_for = main.cfg_for
    cfg = main.cfg
    lang = (language or cfg_for(model_name, "DEFAULT_LANGUAGE") or "").strip()
    kwargs = dict(
        language=lang or None,
        temperature=0.0,
        condition_on_previous_text=False,  # German finetunes loop when True (config.py:245)
        word_timestamps=want_words,
        vad_filter=False,                  # we gate with our own VAD
        initial_prompt=(prompt or cfg_for(model_name, "DEFAULT_PROMPT")) or None,
        no_repeat_ngram_size=3,            # greedy-safe loop guard
    )
    if final:
        kwargs["beam_size"] = cfg_for(model_name, "BEAM_SIZE")
        kwargs["best_of"] = cfg_for(model_name, "BEST_OF")
        kwargs["no_speech_threshold"] = cfg_for(model_name, "NO_SPEECH_THRESHOLD")
        kwargs["log_prob_threshold"] = cfg_for(model_name, "LOG_PROB_THRESHOLD")
        kwargs["compression_ratio_threshold"] = cfg_for(model_name, "COMPRESSION_RATIO_THRESHOLD")
        hallu = cfg_for(model_name, "HALLUCINATION_SILENCE_THRESHOLD")
        if hallu is not None and want_words:
            kwargs["hallucination_silence_threshold"] = hallu
    else:
        kwargs["beam_size"] = int(getattr(cfg, "STREAMING_PARTIAL_BEAM", 5))
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
        include_words = response_format == "verbose_json"
        audio_fmt = (conf.get("audio") or {}).get("format", "pcm_s16le")
        if audio_fmt not in RAW_FORMATS and audio_fmt not in ENCODED_FORMATS:
            await ws.send_json({"type": "error", "code": "unsupported_format",
                                "message": f"audio format {audio_fmt!r} not supported "
                                           f"(raw: {sorted(RAW_FORMATS)}, "
                                           f"encoded via ffmpeg: {sorted(ENCODED_FORMATS)})"})
            await ws.close()
            return

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

        gate_final_words = bool(main.cfg_for(final_model, "WORD_TIMESTAMPS_ENABLED"))
        gate_partial_words = bool(main.cfg_for(partial_model_name, "WORD_TIMESTAMPS_ENABLED"))

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
                want_words=gate_partial_words, language=req_language)
            segs, _info = await _transcribe(partial_model_obj, audio, kwargs)
            if gate_partial_words:
                words = [(w.start, w.end, w.word)
                         for seg in segs for w in (getattr(seg, "words", None) or [])]
                if words:
                    return words
            # fallback: segment-level units (coarser LocalAgreement granularity)
            return [(seg.start, seg.end, seg.text) for seg in segs]

        async def decode_final(audio, prompt):
            want_words = gate_final_words and include_words
            kwargs = _build_transcribe_kwargs(
                main, final_model, final=True, prompt=prompt,
                want_words=gate_final_words, language=req_language)
            segs, _info = await _transcribe(final_model_obj, audio, kwargs)
            raw = "".join(seg.text for seg in segs)
            words_out: list[dict] = []
            if want_words:
                for seg in segs:
                    for w in (getattr(seg, "words", None) or []):
                        words_out.append({"word": w.word, "start": w.start, "end": w.end})
            return raw, words_out

        def postprocess(raw_text):
            return main._postprocess_text(raw_text, model_name=final_model)

        # Output wrappers applied exactly once: prefix on first final, suffix on last.
        out_prefix = main.cfg_for(final_model, "OUTPUT_PREFIX") or ""
        out_suffix = main.cfg_for(final_model, "OUTPUT_SUFFIX") or ""
        wrap_state = {"prefixed": False}

        async def emit(message):
            if message.get("type") == "final":
                if out_prefix and not wrap_state["prefixed"]:
                    message["text"] = out_prefix + message["text"]
                    wrap_state["prefixed"] = True
                if out_suffix and message.get("last"):
                    message["text"] = message["text"] + out_suffix
            await ws.send_json(message)

        async def on_final(info):
            metrics.record_transcription(
                model=final_model, audio_dur=info["audio_dur"],
                proc_dur=info["proc_dur"], status="ok", words=info["words"],
                request_id=uuid.uuid4().hex, user_id=user.get("user_id"),
                key_id=user.get("key_id"))
            logger.info("[stream %s] utt#%d %.2fs audio / %.2fs proc: %r",
                        session_id[:8], info["utterance"], info["audio_dur"],
                        info["proc_dur"], (info["raw_text"] or "")[:120])

        session = StreamSession(
            config=_stream_config(cfg),
            endpointer=make_endpointer(
                getattr(cfg, "STREAMING_VAD_BACKEND", "auto"),
                threshold=float(getattr(cfg, "STREAMING_VAD_THRESHOLD", 0.5)),
                energy_dbfs=float(getattr(cfg, "STREAMING_RMS_GATE_DBFS", -42.0)),
            ),
            decode_partial=decode_partial,
            decode_final=decode_final,
            postprocess=postprocess,
            emit=emit,
            base_prompt=main.cfg_for(final_model, "DEFAULT_PROMPT") or "",
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

        await ws.send_json({
            "type": "ready", "session": session_id, "model": final_model,
            "partial_model": partial_model_name, "sample_rate": SAMPLE_RATE,
            "response_format": response_format, "audio_format": audio_fmt,
        })
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
