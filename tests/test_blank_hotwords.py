"""Blank hotwords must fully disarm the hotwords mechanism.

Any truthy hotwords string — even a single space — makes faster-whisper emit
<|startofprev|> (the fake previous-transcript slot), which alone biases the
decoder to treat audio that starts mid-speech as a window continuation and
drop its opening words. So a whitespace-only value, whether from a client
decode_overrides "clear" or a blank admin DEFAULT_HOTWORDS, must remove the
kwarg entirely rather than being forwarded.
"""

import json

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


def _post(client, **data):
    data.setdefault("model", "whisper-1")
    return client.post("/v1/audio/transcriptions", files=_FILE, data=data)


def test_space_hotwords_override_clears_admin_hotwords(client, app_module, fake_model):
    app_module.cfg.DEFAULT_HOTWORDS = "Komma, Punkt"
    r = _post(client, decode_overrides=json.dumps({"hotwords": " "}))
    assert r.status_code == 200
    assert "hotwords" not in fake_model.last_kwargs


def test_empty_hotwords_override_clears_admin_hotwords(client, app_module, fake_model):
    app_module.cfg.DEFAULT_HOTWORDS = "Komma, Punkt"
    r = _post(client, decode_overrides=json.dumps({"hotwords": ""}))
    assert r.status_code == 200
    assert "hotwords" not in fake_model.last_kwargs


def test_blank_admin_default_hotwords_not_forwarded(client, app_module, fake_model):
    app_module.cfg.DEFAULT_HOTWORDS = "   "
    r = _post(client)
    assert r.status_code == 200
    assert "hotwords" not in fake_model.last_kwargs


def test_real_hotwords_still_flow(client, app_module, fake_model):
    app_module.cfg.DEFAULT_HOTWORDS = "Komma, Punkt"
    r = _post(client)
    assert r.status_code == 200
    assert fake_model.last_kwargs.get("hotwords") == "Komma, Punkt"
    # ...and a real client override replaces (not clears) them.
    r = _post(client, decode_overrides=json.dumps({"hotwords": "Doppelpunkt"}))
    assert r.status_code == 200
    assert fake_model.last_kwargs.get("hotwords") == "Doppelpunkt"
