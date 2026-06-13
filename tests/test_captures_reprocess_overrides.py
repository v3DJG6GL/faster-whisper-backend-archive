"""Phase 5: per-capture reprocess resolves the CAPTURE OWNER's effective
pipeline (not the caller's), and tolerates owner-less captures."""

import wave

from tests.conftest import RATE, bearer


def _fake_transcode(monkeypatch):
    import audio_transcode

    def _fake(src_path, dst_path):
        with wave.open(dst_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(b"\x00\x00" * 100)
        return 1234
    monkeypatch.setattr(audio_transcode, "transcode_to_wav_16k_mono", _fake)


def _make_capture(tmp_path, user_id):
    import captures_store
    src = tmp_path / "src.bin"
    src.write_bytes(b"junk")
    return captures_store.create_capture(
        audio_src_path=str(src), request_id="r1", model="whisper-1",
        language="de", duration_seconds=1.0, raw="hallo welt",
        final="hallo welt", words=[], segments=[], user_id=user_id)


def test_reprocess_resolves_owner_not_caller(client, make_user_key, app_module,
                                             monkeypatch, tmp_path):
    _fake_transcode(monkeypatch)
    _, raw_admin = make_user_key("admin", is_admin=True)
    uid_alice, _ = make_user_key("alice")
    cid = _make_capture(tmp_path, user_id=uid_alice)

    calls = []
    orig = app_module.build_ident

    def spy(user, model_id, *a, **k):
        calls.append((dict(user or {}), model_id))
        return orig(user, model_id, *a, **k)
    monkeypatch.setattr(app_module, "build_ident", spy)

    # Admin reprocesses alice's capture → ident must resolve ALICE's config.
    r = client.post(f"/captures/api/{cid}/reprocess", headers=bearer(raw_admin))
    assert r.status_code == 200, r.text
    assert calls, "reprocess did not resolve an ident"
    assert calls[0][0].get("user_id") == uid_alice
    assert calls[0][0].get("key_id") is None      # reprocess is pipeline-only


def test_reprocess_ownerless_capture_ok(client, make_user_key, monkeypatch, tmp_path):
    _fake_transcode(monkeypatch)
    _, raw_admin = make_user_key("admin", is_admin=True)
    cid = _make_capture(tmp_path, user_id=None)
    r = client.post(f"/captures/api/{cid}/reprocess", headers=bearer(raw_admin))
    assert r.status_code == 200, r.text
