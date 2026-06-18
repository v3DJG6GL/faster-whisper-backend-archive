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


def test_stream_request_override_profile_applies(
        client, make_user_key, fake_model, app_module, monkeypatch):
    """P11/B0e: a per-request `override_profile` named in the WS handshake is
    honoured — echoed as `profile_applied` in the ready frame and its decode
    params reach the final decode (least-specific layer, no admin lock)."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profile(client, h, "fast", BEAM_SIZE=9)
    _, raw_alice = make_user_key("alice")     # no binding → inherits global gate (on)

    with client.websocket_connect(
            f"/v1/audio/transcriptions/stream?key={raw_alice}") as ws:
        ws.send_json({"type": "config", "model": "whisper-1",
                      "override_profile": "fast",
                      "audio": {"format": "pcm_s16le", "sample_rate": 16000}})
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["profile_applied"] == "fast"
        ws.send_bytes(_pcm(8000, 2500))
        ws.send_bytes(_pcm(0, 1500))
        ws.send_json({"type": "stop"})
        _drain(ws)

    assert fake_model.last_kwargs["beam_size"] == 9


def test_stream_picks_up_binding_change_without_reconnect(
        client, make_user_key, fake_model, app_module, monkeypatch):
    """Regression: a binding change made WHILE a dictation WebSocket is open is
    picked up on the next utterance — ident used to be frozen at handshake for
    the whole session, so live edits silently had no effect until reconnect."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    # Both profiles in ONE save — a second POST would replace OVERRIDE_PROFILES.
    r = client.post(f"{OV}/state", headers=h,
                    json={"OVERRIDE_PROFILES": {"p7": {"BEAM_SIZE": 7},
                                                "p3": {"BEAM_SIZE": 3}}})
    assert r.status_code == 200, r.text
    uid, raw_alice = make_user_key("alice")
    _bind(client, h, uid, profiles=["p7"])

    import api_keys_store
    with client.websocket_connect(
            f"/v1/audio/transcriptions/stream?key={raw_alice}") as ws:
        ws.send_json({"type": "config", "model": "whisper-1",
                      "audio": {"format": "pcm_s16le", "sample_rate": 16000}})
        assert ws.receive_json()["type"] == "ready"
        # utterance 1 under profile p7
        ws.send_bytes(_pcm(8000, 2500))
        ws.send_bytes(_pcm(0, 1500))
        # admin re-binds alice p7 -> p3 mid-session (bumps the config version).
        # Without the per-utterance re-resolve, the connection's frozen ident
        # would keep using p7 and the final assertion below would see beam 7.
        api_keys_store.set_user_permissions(
            uid, {"pages": {}, "config": {"overrides": {}, "profiles": ["p3"], "locks": []}})
        # utterance 2 must pick up p3 (beam 3) on the same connection
        ws.send_bytes(_pcm(8000, 2500))
        ws.send_bytes(_pcm(0, 1500))
        ws.send_json({"type": "stop"})
        _drain(ws)

    assert fake_model.last_kwargs["beam_size"] == 3


def test_final_drops_low_confidence_hallucination_segment(
        client, make_user_key, fake_model, app_module, monkeypatch):
    """B3: a final-decode segment that is BOTH very low confidence AND fell through
    the temperature ladder (a hallucination on near-silence) is dropped from the
    committed text; a normal segment in the same decode is kept."""
    from tests.conftest import FakeSegment
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    make_user_key("admin", is_admin=True)               # lock down
    _, raw_alice = make_user_key("alice")

    bad = FakeSegment(" thank you for watching", 0.0, 1.0)
    bad.avg_logprob, bad.temperature = -2.32, 1.0        # failed decode → drop
    good = FakeSegment(" der patient hat fieber.", 1.0, 2.0)
    good.avg_logprob, good.temperature = -0.2, 0.0       # healthy → keep
    fake_model._segments = [bad, good]

    with client.websocket_connect(
            f"/v1/audio/transcriptions/stream?key={raw_alice}") as ws:
        ws.send_json({"type": "config", "model": "whisper-1",
                      "audio": {"format": "pcm_s16le", "sample_rate": 16000}})
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(_pcm(8000, 2500))
        ws.send_bytes(_pcm(0, 1500))
        ws.send_json({"type": "stop"})
        msgs = _drain(ws)

    finals = [m for m in msgs if m["type"] == "final"]
    doc = (finals[-1]["committed"] + finals[-1]["tail"]) if finals else ""
    assert "fieber" in doc                               # healthy segment kept
    assert "watching" not in doc                         # hallucination dropped


def test_stream_prompt_sentinel_inherit_clear_value(
        client, make_user_key, fake_model, app_module, monkeypatch):
    """B4: the WS handshake prompt is a present-vs-absent sentinel. Absent → inherit
    DEFAULT_PROMPT; explicit "" → CLEAR (no initial_prompt); a value → used verbatim.
    The first utterance's final decode is seeded straight from base_prompt."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    monkeypatch.setattr(app_module.cfg, "DEFAULT_PROMPT", "SERVER PROMPT", raising=False)
    make_user_key("admin", is_admin=True)
    _, raw_alice = make_user_key("alice")

    def _run(conf_extra):
        with client.websocket_connect(
                f"/v1/audio/transcriptions/stream?key={raw_alice}") as ws:
            ws.send_json({"type": "config", "model": "whisper-1",
                          "audio": {"format": "pcm_s16le", "sample_rate": 16000},
                          **conf_extra})
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(_pcm(8000, 2500))
            ws.send_bytes(_pcm(0, 1500))
            ws.send_json({"type": "stop"})
            _drain(ws)
        return fake_model.last_kwargs.get("initial_prompt")

    assert _run({}) == "SERVER PROMPT"            # absent → inherit
    assert _run({"prompt": ""}) is None           # explicit empty → clear
    assert _run({"prompt": "my terms"}) == "my terms"   # value → verbatim


def test_setup_window_error_delivers_internal_error_and_closes(
        client, make_user_key, fake_model, app_module, monkeypatch):
    """A failure in the handshake→loop setup window (after the audio consumer task
    has started, before the receive loop's own cleanup is armed) must still deliver
    an `internal` error frame and close without hanging. The error reply is sent
    under the same send-lock as every other send — hoisted above the try so it is
    bound even on an early failure — and the consumer is cancelled in the finally
    (no orphaned task). Guards the previously-untested outer error path."""
    monkeypatch.setattr(app_module.cfg, "STREAMING_VAD_BACKEND", "energy", raising=False)
    _, raw_alice = make_user_key("alice")

    real_cfg_for = app_module.cfg_for

    def boom(model_id, field, ident=None):
        # The per-identity idle-timeout read happens AFTER the consumer task is
        # created — i.e. squarely inside the setup window this path guards.
        if field == "STREAMING_IDLE_TIMEOUT_SEC":
            raise RuntimeError("kaboom in setup window")
        return real_cfg_for(model_id, field, ident)

    monkeypatch.setattr(app_module, "cfg_for", boom)

    with client.websocket_connect(
            f"/v1/audio/transcriptions/stream?key={raw_alice}") as ws:
        ws.send_json({"type": "config", "model": "whisper-1",
                      "audio": {"format": "pcm_s16le", "sample_rate": 16000}})
        msgs = _drain(ws)

    errors = [m for m in msgs if m.get("type") == "error"]
    assert errors, f"expected an internal error frame, got {msgs!r}"
    assert errors[0]["code"] == "internal"
    # Security: the client frame must NOT leak the raw exception text (it can
    # carry filesystem/internal detail) — it is logged server-side instead.
    assert "kaboom" not in errors[0]["message"]
    assert errors[0]["message"] == "internal error"


def test_model_load_failure_delivers_generic_error_and_closes(
        client, make_user_key, app_module, monkeypatch):
    """A model-load failure during the handshake must deliver a
    `model_load_failed` frame and close — without leaking the raw exception
    text (it can carry the model dir / filesystem path); the detail is logged
    server-side instead. Mirrors the internal-error guard above for the other
    client-facing error reply touched by the same security fix."""
    _, raw_alice = make_user_key("alice")

    async def _boom(name):
        raise RuntimeError("/srv/models/secret-path missing")

    monkeypatch.setattr(app_module, "_get_or_load_model", _boom)

    with client.websocket_connect(
            f"/v1/audio/transcriptions/stream?key={raw_alice}") as ws:
        ws.send_json({"type": "config", "model": "whisper-1",
                      "audio": {"format": "pcm_s16le", "sample_rate": 16000}})
        msgs = _drain(ws)

    errors = [m for m in msgs if m.get("type") == "error"]
    assert errors, f"expected a model_load_failed frame, got {msgs!r}"
    assert errors[0]["code"] == "model_load_failed"
    # Security: the raw exception text (filesystem / model-dir paths) must not
    # reach the client; it is logged server-side instead.
    assert "secret-path" not in errors[0]["message"]
    assert errors[0]["message"] == "model could not be loaded"
