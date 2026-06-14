"""In-process TestClient tests for the streaming WebSocket endpoint.

Runs against the real FastAPI app with the conftest fake model (so no
faster-whisper / GPU needed), exercising the config handshake, the partial/final
message contract, and the stop/close drain. Open mode (no API key) → the synthetic
admin passes auth.
"""

import numpy as np
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


def _pcm(level, ms, sr=16000):
    return np.full(sr * ms // 1000, level, dtype="<i2").tobytes()


def _drain(ws, limit=200):
    msgs = []
    try:
        for _ in range(limit):
            msgs.append(ws.receive_json())
    except WebSocketDisconnect:
        pass
    return msgs


def test_stream_happy_path_partials_then_final(app_module, monkeypatch):
    # Force the energy gate: the synthetic constant-amplitude PCM below is "loud"
    # but not speech, so the real Silero VAD would (correctly) reject it. This test
    # exercises the routing/protocol/session flow, not the VAD model.
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({
                "type": "config", "model": "whisper-1", "response_format": "json",
                "audio": {"format": "pcm_s16le", "sample_rate": 16000},
            })
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["sample_rate"] == 16000

            ws.send_bytes(_pcm(8000, 2500))   # ~2.5 s speech → ≥2 partials
            ws.send_bytes(_pcm(0, 1500))      # ~1.5 s silence → finalize (held, no terminator)
            ws.send_json({"type": "stop"})    # → close flushes the held final
            msgs = _drain(ws)

    partials = [m for m in msgs if m["type"] == "partial"]
    finals = [m for m in msgs if m["type"] == "final"]
    assert partials, "expected at least one partial"
    # LocalAgreement-2 commits the repeated 'hallo welt' hypothesis.
    assert any("welt" in m["committed"] for m in partials)
    assert finals, "expected a final after stop/close"
    assert "welt" in "".join(m["committed"] + m.get("tail", "") for m in finals)
    assert finals[-1].get("last") is True


def test_stream_rejects_unsupported_audio_format(app_module):
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({
                "type": "config", "model": "whisper-1",
                "audio": {"format": "g729-telephony"},  # not raw, not an ffmpeg format we allow
            })
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert msg["code"] == "unsupported_format"


def test_dictate_demo_page_served(app_module):
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        r = client.get("/dictate")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "Live Dictation" in body
    assert "/v1/audio/transcriptions/stream" in body
    assert "AudioWorkletNode" in body
    # batch mode: a mode selector + MediaRecorder POST to the file endpoint.
    assert 'id="mode"' in body
    assert "MediaRecorder" in body
    assert "startBatch" in body


def test_stream_disabled_closes_connection(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "STREAMING_ENABLED", False, raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        try:
            with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
                # If it doesn't reject pre-accept, it must close immediately.
                with __import__("pytest").raises(WebSocketDisconnect):
                    ws.receive_json()
        except WebSocketDisconnect:
            pass  # rejected during handshake — also acceptable


def test_stream_records_trace_text_per_utterance(app_module, monkeypatch):
    """Each finalized utterance writes a recent-transcriptions row with non-empty
    raw/final text (drives /quick-config + /reports), not just numeric metrics."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    import transcriptions_store
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        before = transcriptions_store.count()
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({"type": "config", "model": "whisper-1",
                          "audio": {"format": "pcm_s16le", "sample_rate": 16000}})
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(_pcm(8000, 2500))   # speech
            ws.send_bytes(_pcm(0, 1500))      # silence → finalize
            ws.send_json({"type": "stop"})
            _drain(ws)
        rows = transcriptions_store.list_recent(limit=10)
        after = transcriptions_store.count()
    assert after > before, "no recent-transcription row recorded for the utterance"
    assert any((r.get("raw") or "").strip() and (r.get("final") or "").strip() for r in rows), \
        "recorded trace rows have empty raw/final (the /quick-config bug)"
    # streamed utterances are tagged source='stream' so /quick-config can chip them.
    assert any(r.get("source") == "stream" for r in rows), \
        "streamed trace not tagged source='stream'"


def test_safe_ws_send_swallows_dead_socket():
    """A page reload mid-dictation closes the socket; the session-close drain then
    sends a final to a dead socket. uvicorn raises RuntimeError('Unexpected ASGI
    message ... after ... close'); the send must be swallowed, not surface as an
    error traceback."""
    import asyncio
    import streaming_routes

    class DeadWS:
        async def send_json(self, _m):
            raise RuntimeError(
                "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'")

    class LiveWS:
        def __init__(self):
            self.sent = []

        async def send_json(self, m):
            self.sent.append(m)

    assert asyncio.run(streaming_routes._safe_ws_send(DeadWS(), {"type": "final"})) is False
    live = LiveWS()
    assert asyncio.run(streaming_routes._safe_ws_send(live, {"type": "final"})) is True
    assert live.sent == [{"type": "final"}]


def test_stream_handshake_idle_timeout_frees_slot(app_module, monkeypatch):
    # A client that connects + passes auth but never sends its config handshake
    # must not hold a session slot forever: the server abandons the wait after
    # STREAMING_IDLE_TIMEOUT_SEC and closes with the idle close code (4408).
    import pytest
    from streaming_routes import _WS_IDLE_TIMEOUT
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_SEC", 0.3, raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            with pytest.raises(WebSocketDisconnect) as ei:
                ws.receive_json()   # send nothing → idle close
    assert ei.value.code == _WS_IDLE_TIMEOUT


def test_stream_session_idle_timeout_closes_and_notifies(app_module, monkeypatch):
    # After a successful handshake, a connection that goes silent mid-session is
    # closed once the idle timeout elapses, with an idle_timeout notice first.
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_SEC", 0.3, raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({
                "type": "config", "model": "whisper-1", "response_format": "json",
                "audio": {"format": "pcm_s16le", "sample_rate": 16000},
            })
            assert ws.receive_json()["type"] == "ready"
            msgs = _drain(ws)   # send nothing further → idle close
    assert any(m.get("code") == "idle_timeout" for m in msgs)


def test_stream_idle_timeout_zero_disables(app_module, monkeypatch):
    # 0 disables the idle timeout: the normal stop/close flow still works and no
    # idle_timeout notice is emitted (the receive falls through to a plain await).
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_IDLE_TIMEOUT_SEC", 0.0, raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({
                "type": "config", "model": "whisper-1", "response_format": "json",
                "audio": {"format": "pcm_s16le", "sample_rate": 16000},
            })
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "stop"})
            msgs = _drain(ws)
    assert not any(m.get("code") == "idle_timeout" for m in msgs)
