"""build_info version resolution: env override > git describe > "unknown"."""

import importlib
import subprocess

import build_info


def test_resolves_to_nonempty_string():
    # In any environment (CI, checkout, tarball) the constant must be a
    # usable display string — the exact value depends on the build.
    assert isinstance(build_info.APP_VERSION, str) and build_info.APP_VERSION
    assert build_info.SERVER_NAME == "faster-whisper-backend"


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("WHISPER_BUILD_VERSION", "v9.9.9-test")
    try:
        assert importlib.reload(build_info).APP_VERSION == "v9.9.9-test"
    finally:
        monkeypatch.delenv("WHISPER_BUILD_VERSION")
        importlib.reload(build_info)


def test_no_git_falls_back_to_unknown(monkeypatch):
    monkeypatch.delenv("WHISPER_BUILD_VERSION", raising=False)

    def _no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", _no_git)
    try:
        assert importlib.reload(build_info).APP_VERSION == "unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(build_info)
