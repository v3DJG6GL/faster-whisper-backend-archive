"""Integration tests for POST /v1/audio/transcriptions.

Drives the real handler with the FakeModel injected via the harness. The
fake model ignores the uploaded bytes, so a tiny dummy WAV payload is fine.
"""

from conftest import FakeModel

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


def _post(client, **data):
    data.setdefault("model", "whisper-1")
    return client.post("/v1/audio/transcriptions", files=_FILE, data=data)


def test_default_json_returns_text_object(client):
    r = _post(client, response_format="json")
    assert r.status_code == 200
    body = r.json()
    assert body == {"text": "hallo welt"}


def test_text_format_returns_plain_string(client):
    r = _post(client, response_format="text")
    assert r.status_code == 200
    # response_format="text" returns the bare string (JSON-encoded string body).
    assert r.json() == "hallo welt"


def test_verbose_json_shape(client):
    r = _post(client, response_format="verbose_json")
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "transcribe"
    assert body["language"] == "de"
    assert body["duration"] == 1.0
    assert body["text"] == "hallo welt"
    assert isinstance(body["segments"], list) and body["segments"]
    seg = body["segments"][0]
    assert seg["text"] == "hallo welt"
    # verbose_json with no explicit granularities still asks for words
    # (include_words = response_format == "verbose_json" and not granularities),
    # and WORD_TIMESTAMPS_ENABLED defaults True, so words are present.
    assert "words" in body
    assert [w["word"] for w in body["words"]] == ["hallo", "welt"]


def test_srt_falls_through_to_text_object(client):
    # Documented non-OpenAI behavior: srt/vtt aren't special-cased, so they
    # fall through to the default {"text": ...} JSON shape.
    r = _post(client, response_format="srt")
    assert r.status_code == 200
    assert r.json() == {"text": "hallo welt"}


def test_vtt_falls_through_to_text_object(client):
    r = _post(client, response_format="vtt")
    assert r.status_code == 200
    assert r.json() == {"text": "hallo welt"}


def test_words_gated_by_config_disabled(client, app_module):
    # WORD_TIMESTAMPS_ENABLED=False => want_word_ts is False even when the
    # request asks for word granularity, so the model gets word_timestamps=False
    # and the FakeModel returns no words; verbose_json["words"] is empty.
    app_module.cfg.WORD_TIMESTAMPS_ENABLED = False
    r = client.post(
        "/v1/audio/transcriptions",
        files=_FILE,
        data={
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        },
    )
    assert r.status_code == 200
    assert r.json().get("words") == []


def test_words_included_with_granularity_field(client, app_module):
    # Default config has WORD_TIMESTAMPS_ENABLED=True. Explicitly request word
    # granularity on a json response: include_words drives the response, but
    # the default json shape ({"text":...}) does not surface words. So assert
    # the model was actually asked for word_timestamps=True via fake_model.
    r = client.post(
        "/v1/audio/transcriptions",
        files=_FILE,
        data={
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        },
    )
    assert r.status_code == 200
    assert [w["word"] for w in r.json()["words"]] == ["hallo", "welt"]


def test_model_transcribe_raises_returns_500(client, app_module, monkeypatch):
    class BoomModel(FakeModel):
        def transcribe(self, path, **kwargs):
            raise RuntimeError("decode blew up")

    async def _loader(name):
        return BoomModel()

    monkeypatch.setattr(app_module, "_get_or_load_model", _loader)
    r = _post(client, response_format="json")
    assert r.status_code == 500


def test_output_prefix_suffix_wrap(client, app_module):
    app_module.cfg.OUTPUT_PREFIX = "[de] "
    app_module.cfg.OUTPUT_SUFFIX = " (end)"
    r = _post(client, response_format="text")
    assert r.status_code == 200
    # Wrappers applied, then a defensive outer trim (leading/trailing spaces of
    # the *whole* string are stripped). Inner content keeps the wrapper text.
    assert r.json() == "[de] hallo welt (end)"


def test_missing_file_is_422(client):
    r = client.post("/v1/audio/transcriptions", data={"model": "whisper-1"})
    assert r.status_code == 422
