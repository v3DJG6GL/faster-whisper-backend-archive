"""Tests for the streaming audio transports (raw PCM passthrough + ffmpeg decode).

The ffmpeg tests generate a real WebM/Opus clip with the system ffmpeg and decode
it back to PCM, so they exercise the actual MediaRecorder-style path. Skipped if
ffmpeg (or libopus) is unavailable.
"""

import asyncio
import subprocess

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from streaming_transport import FfmpegTransport, RawPcmTransport, make_transport


def _gen_webm(seconds: float) -> bytes:
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}",
             "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-f", "webm", "pipe:1"],
            capture_output=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("ffmpeg unavailable")
    if p.returncode != 0 or not p.stdout:
        pytest.skip("ffmpeg webm/opus encode unavailable")
    return p.stdout


def test_raw_transport_is_passthrough():
    got = bytearray()

    async def sink(b):
        got.extend(b)

    async def run():
        t = make_transport("pcm_s16le", sink)
        assert isinstance(t, RawPcmTransport)
        await t.start()
        await t.feed(b"\x01\x02\x03\x04")
        await t.aclose()

    asyncio.run(run())
    assert bytes(got) == b"\x01\x02\x03\x04"


def test_ffmpeg_transport_decodes_webm_to_pcm():
    webm = _gen_webm(1.0)
    got = bytearray()

    async def sink(b):
        got.extend(b)

    async def run():
        t = make_transport("webm", sink)
        assert isinstance(t, FfmpegTransport)
        await t.start()
        await t.feed(webm)
        await t.aclose()

    asyncio.run(run())
    # ~1 s of 16 kHz mono s16le ≈ 32000 bytes; allow generous tolerance.
    assert len(got) > 16000


def test_stream_route_accepts_webm_via_ffmpeg(app_module, monkeypatch):
    # Force the energy gate — the synthetic sine tone is not real speech, so the
    # Silero VAD would reject it; this test only checks the ffmpeg decode path.
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    webm = _gen_webm(2.0)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as client:
        with client.websocket_connect("/v1/audio/transcriptions/stream") as ws:
            ws.send_json({"type": "config", "model": "whisper-1",
                          "audio": {"format": "webm"}})
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["audio_format"] == "webm"
            ws.send_bytes(webm)
            ws.send_json({"type": "stop"})
            msgs = []
            try:
                for _ in range(200):
                    msgs.append(ws.receive_json())
            except WebSocketDisconnect:
                pass
    finals = [m for m in msgs if m["type"] == "final"]
    assert finals, "expected a final from the decoded WebM"
    assert "welt" in "".join(m["text"] for m in finals)
