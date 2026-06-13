"""Phase 4: the streaming WebSocket path honours per-identity config — locked
overrides surfaced in the `ready` frame, and profile decode params reach the
final decode. Driven in-process; no faster-whisper needed."""

import numpy as np
from starlette.websockets import WebSocketDisconnect

from tests.conftest import bearer

OV = "/settings/overrides"
PERMS = "/settings/api-keys/api/users"


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


def _profile(client, h, name, **fields):
    r = client.post(f"{OV}/state", headers=h, json={"OVERRIDE_PROFILES": {name: fields}})
    assert r.status_code == 200, r.text


def _bind(client, h, uid, **binding):
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h,
                     json={"pages": {}, "config": {"overrides": {}, "profiles": [], **binding}})
    assert r.status_code == 200, r.text


def test_ready_frame_reports_locked_handshake_override(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profile(client, h, "p", TEMPERATURE="0.0", DEFAULT_LANGUAGE="de",
             locks=["TEMPERATURE", "DEFAULT_LANGUAGE"])
    uid, raw_alice = make_user_key("alice")
    _bind(client, h, uid, profiles=["p"])

    with client.websocket_connect(
            f"/v1/audio/transcriptions/stream?key={raw_alice}") as ws:
        ws.send_json({"type": "config", "model": "whisper-1", "language": "fr",
                      "decode_overrides": {"temperature": 0.7},
                      "audio": {"format": "pcm_s16le"}})
        ready = ws.receive_json()
    assert ready["type"] == "ready"
    # both the locked decode key and the locked language are surfaced
    assert set(ready["overrides_ignored"]) == {"temperature", "language"}


def test_final_decode_uses_profile_beam(client, make_user_key, fake_model,
                                        app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profile(client, h, "p", BEAM_SIZE=7)
    uid, raw_alice = make_user_key("alice")
    _bind(client, h, uid, profiles=["p"])

    with client.websocket_connect(
            f"/v1/audio/transcriptions/stream?key={raw_alice}") as ws:
        ws.send_json({"type": "config", "model": "whisper-1",
                      "audio": {"format": "pcm_s16le", "sample_rate": 16000}})
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))   # speech → partials
        ws.send_bytes(_pcm(0, 1500))      # silence → finalize
        ws.send_json({"type": "stop"})
        _drain(ws)

    # The final decode is the last transcribe call; it uses the assembler with
    # the profile's BEAM_SIZE (partials force STREAMING_PARTIAL_BEAM instead).
    assert fake_model.last_kwargs["beam_size"] == 7
