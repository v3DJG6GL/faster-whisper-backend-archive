"""Integration tests for /captures routes.

Captures rows require real audio transcode (ffmpeg) to create, so these
tests focus on the read/list/route-ordering/auth surface that works without
fabricating audio blobs.
"""

import os

import pytest
from starlette.testclient import TestClient

from conftest import bearer


def test_captures_page(client):
    r = client.get("/captures")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_captures_list_open_mode(client):
    r = client.get("/captures/api/list")
    assert r.status_code == 200
    body = r.json()
    assert "captures" in body and "counts" in body
    assert "is_admin" in body


def test_reprocess_vad_job_lifecycle(client):
    # Status endpoint is registered and reports a known state.
    s0 = client.get("/captures/api/reprocess-vad/status")
    assert s0.status_code == 200
    assert s0.json()["status"] in ("idle", "running", "done", "error")
    # Start the bulk VAD re-merge on an empty store → runs and finishes clean.
    assert client.post("/captures/api/reprocess-vad").status_code == 200
    import time
    s = s0.json()
    for _ in range(30):
        s = client.get("/captures/api/reprocess-vad/status").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert s["status"] == "done"
    assert s["total"] == 0 and s["rebuilt"] == 0


def test_samples_route_not_swallowed_by_cid(client):
    # Regression: /captures/api/samples must resolve to the sample-list handler,
    # NOT the parameterized /captures/api/{cid} handler (which would 404 with
    # cid="samples"). A 200 with a "samples" key proves correct route ordering.
    r = client.get("/captures/api/samples")
    assert r.status_code == 200
    assert "samples" in r.json()


def test_export_route_not_swallowed_by_cid(client):
    # /captures/api/export is also a literal route declared before /{cid}.
    r = client.get("/captures/api/export")
    assert r.status_code == 200
    assert "application/gzip" in r.headers.get("content-type", "")


def test_unknown_cid_404(client):
    r = client.get("/captures/api/does-not-exist")
    assert r.status_code == 404


def test_propose_merges_ok(client):
    r = client.get("/captures/api/propose-merges")
    assert r.status_code == 200
    assert "proposals" in r.json()


def test_by_request_id_ok(client):
    r = client.get("/captures/api/by-request/unknown-req")
    assert r.status_code == 200
    assert r.json()["captures"] == []  # no captures for an unknown request id


def test_host_gate_rejects_non_loopback(app_module, monkeypatch):
    # /captures is user-tier (require_user_webui_host / USER_WEBUI_ALLOWED_HOSTS).
    # The list defaults OPEN, so narrow it to loopback to exercise the host gate:
    # a non-loopback host is then 403 before the page-permission check.
    import config as cfg
    monkeypatch.setattr(
        cfg, "USER_WEBUI_ALLOWED_HOSTS", ["127.0.0.1", "::1"], raising=False
    )
    with TestClient(app_module.app, client=("8.8.8.8", 1)) as c:
        assert c.get("/captures/api/list").status_code == 403


def test_list_requires_page_when_locked(client, make_user_key):
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"captures": "none"})
    r = client.get("/captures/api/list", headers=bearer(raw))
    assert r.status_code == 403


def test_clear_requires_admin_when_locked(client, make_user_key):
    # POST /captures/api/clear additionally Depends(require_admin).
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"captures": "own"})
    r = client.post("/captures/api/clear", headers=bearer(raw))
    assert r.status_code == 403


def test_merge_member_scope_guard_precedes_state_checks(
        captures_store_db, monkeypatch, tmp_path):
    """A scope=own caller probing ANOTHER user's capture id must get a uniform
    404 from the ownership guard — not a 400/410 that would leak the capture's
    existence + state. Regression guard for _validate_merge_payload: the
    per-member scope check must run BEFORE the already-in-sample / audio-missing
    checks."""
    import wave

    import audio_transcode
    import auth
    import captures_routes
    from fastapi import HTTPException

    cs = captures_store_db

    def _fake_transcode(src_path, dst_path):
        with wave.open(dst_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 100)
        return 1234

    monkeypatch.setattr(
        audio_transcode, "transcode_to_wav_16k_mono", _fake_transcode)

    src = tmp_path / "src.bin"
    src.write_bytes(b"junk")
    cid = cs.create_capture(
        audio_src_path=str(src), request_id="r1", model="small",
        language="de", duration_seconds=1.0, raw="r", final="f",
        words=[], segments=[], user_id="alice",
    )
    # Delete the audio so the OLD ordering would raise 410 ("audio is missing"),
    # leaking that the row exists; the fix must 404 for a non-owner first.
    os.unlink(cs.abs_audio_path(cs.get_capture(cid)["audio_relpath"]))

    # bob: scope=own captures user, NOT the owner and NOT admin → uniform 404.
    bob = {
        "user_id": "bob",
        "permissions": auth.Permissions(
            {"pages": {"captures": "own"}}, is_admin=False),
    }
    with pytest.raises(HTTPException) as ei:
        captures_routes._validate_merge_payload([cid], 0, bob)
    assert ei.value.status_code == 404

    # The OWNER still reaches the real state check (410), proving the guard
    # blocks only cross-user probes — not the owner's own legitimate errors.
    alice = {
        "user_id": "alice",
        "permissions": auth.Permissions(
            {"pages": {"captures": "own"}}, is_admin=False),
    }
    with pytest.raises(HTTPException) as ei2:
        captures_routes._validate_merge_payload([cid], 0, alice)
    assert ei2.value.status_code == 410


def test_member_delete_respects_sample_lock(captures_store_db, groups_store_db):
    """A non-admin cannot mutate/delete a capture that is a member of a LOCKED
    sample — deleting it would auto-dissolve (and destroy the merged WAV of) an
    admin-locked sample, bypassing the same guard dissolve_sample_api enforces.
    Regression guard for _assert_member_sample_not_locked."""
    import auth
    import captures_routes
    from fastapi import HTTPException

    cs = captures_store_db
    gs = groups_store_db
    conn = cs._require_conn()

    def _mk_sample(sid, *, locked):
        conn.execute(
            "INSERT INTO capture_samples (id, user_id, created_ts,"
            " merged_wav_relpath, merged_duration_ms, transcript,"
            " transcript_join_strategy, member_hashes_json,"
            " inter_segment_silence_ms, is_stale, is_locked, status,"
            " admin_notes, language, member_trims_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, "alice", 1.0, gs._relpath_for(sid), 5000, "t", "space",
             "{}", 300, 0, 1 if locked else 0, "new", "", "de", "{}"),
        )

    def _mk_member(cid, sid):
        rel = os.path.join(cid[0:2], cid[2:4], f"{cid}.wav")
        conn.execute(
            "INSERT INTO captures (id, created_ts, request_id, model, language,"
            " duration_seconds, audio_relpath, audio_format, raw, final,"
            " words_json, segments_json, corrections_json, status, user_id,"
            " sample_id, sample_order)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cid, 1.0, None, "m", "de", 2.0, rel, "wav", "r", "f", "[]", "[]",
             "[]", "new", "alice", sid, 0),
        )

    _mk_sample("locked00sid", locked=True)
    _mk_member("locked00cid", "locked00sid")
    _mk_sample("open000sid", locked=False)
    _mk_member("open000cid", "open000sid")
    _mk_member("free0000cid", None)  # no parent sample

    def _user(is_admin):
        return {
            "user_id": "alice",
            "is_admin": is_admin,
            "permissions": auth.Permissions(
                {"pages": {"captures": "own"}}, is_admin=is_admin),
        }

    locked_row = cs.get_capture("locked00cid")
    # Non-admin (even the owner) is refused on a locked sample's member.
    with pytest.raises(HTTPException) as ei:
        captures_routes._assert_member_sample_not_locked(locked_row, _user(False))
    assert ei.value.status_code == 409
    # Admin passes through.
    captures_routes._assert_member_sample_not_locked(locked_row, _user(True))
    # A member of an UNLOCKED sample, and a member of NO sample, pass through.
    captures_routes._assert_member_sample_not_locked(
        cs.get_capture("open000cid"), _user(False))
    captures_routes._assert_member_sample_not_locked(
        cs.get_capture("free0000cid"), _user(False))
