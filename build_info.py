"""Build identity — the version string clients and the WebUI can display.

Resolution order (first hit wins):
  1. WHISPER_BUILD_VERSION env — baked into container images by CI
     (docker build --build-arg BUILD_VERSION=$(git describe ...); the image
     has no .git to describe, see .dockerignore).
  2. `git describe --tags --always --dirty` on a bare-metal checkout
     (Linux/Windows) — anchors to the newest v* tag and bumps automatically
     with every commit, e.g. "v0.1.0-3-g1a2b3c4".
  3. "unknown" — tarball without .git, or no git on PATH.

Resolved once at import: the value cannot change while the process runs, and
/v1/models must not fork a git subprocess per request.
"""

import os
import subprocess

SERVER_NAME = "faster-whisper-backend"


def _resolve() -> str:
    env = (os.environ.get("WHISPER_BUILD_VERSION") or "").strip()
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


APP_VERSION = _resolve()
