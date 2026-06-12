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
    assert "welt" in "".join(m["text"] for m in finals)
    assert finals[-1].get("append") is True


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
