"""
Self-restart helper for the backend.

Windows (service deployment): spawn `WhisperAPI.exe restart!` (the documented
WinSW self-restart command, note the trailing `!`) as a detached child, then
exit cleanly. WinSW's `restart!` re-execs the wrapper *after* we die,
surviving the SCM child-tree kill that a regular `restart` would not.
Restart latency is ~3-4 s end-to-end on a no-preload deployment.

Other OSes (Linux/macOS, bare / systemd / Docker): re-exec the process in
place via os.execv. Python opens sockets with O_CLOEXEC, so the listen port
is released on exec and the fresh process re-binds cleanly. This works whether
the server is run bare (`python main.py`), supervised by systemd (same-PID
re-exec keeps the unit active), or in a container with a restart policy.

Why not rely on WinSW's <onfailure action="restart"/> on exit code 0?
Because v2's onfailure semantics on graceful exits are unreliable: issue
#699 reports "always restarts", #969 reports "never notifies SCM".
Don't depend on it. <onfailure> stays in the XML purely as defense-in-
depth for crashes.

Why detached subprocess works here when it failed for NSSM (per the
prior comment that lived in this file): NSSM tied the python process
directly into the SCM job; spawning a detached child from inside that
job triggered STATUS_DLL_INIT_FAILED 0xC0000142. WinSW's `restart!`
is *designed* to be invoked from inside a service-jobbed child -- its
whole purpose is to survive the SCM kill. Different mechanism, no
job-object fight.
  docs: https://github.com/winsw/winsw/blob/v2.12.0/doc/selfRestartingService.md
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time


def trigger_self_restart(delay_sec: float = 1.5) -> str:
    """Schedule process exit so WinSW relaunches us via `restart!`.

    Returns immediately with a method label (for the admin UI to display).
    The Timer thread fires after `delay_sec` so the caller's HTTP response
    has time to flush over loopback before the process restarts.
    """
    if sys.platform != "win32":
        # Re-exec the current interpreter with the same argv. Sockets carry
        # O_CLOEXEC so the port frees on exec; the new process re-binds it.
        def do_reexec() -> None:
            try:
                os.execv(sys.executable, [sys.executable, *sys.argv])
            except Exception:
                # If exec fails, exit and let a supervisor (systemd
                # Restart=, Docker restart policy) bring us back.
                os._exit(0)

        threading.Timer(delay_sec, do_reexec).start()
        return "execv-reexec"

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    winsw = os.path.join(repo_dir, "WhisperAPI.exe")

    if not os.path.isfile(winsw):
        # Fallback: bare exit. <onfailure action="restart"/> in WhisperAPI.xml
        # MAY relaunch us on exit 0 (v2 semantics are inconsistent). Don't
        # depend on it, but it's a non-zero chance vs none.
        threading.Timer(delay_sec, lambda: os._exit(0)).start()
        return "winsw-missing-fallback"

    def do_restart() -> None:
        # `restart!` (with the trailing `!`) is WinSW's "restart from inside
        # the wrapped process" command. It detaches a helper that re-execs
        # the wrapper after we die.
        try:
            subprocess.Popen(
                [winsw, "restart!"],
                creationflags=(
                    subprocess.DETACHED_PROCESS
                    | subprocess.CREATE_NEW_PROCESS_GROUP
                ),
                close_fds=True,
            )
        except Exception:
            # If the spawn fails, fall through to os._exit -- <onfailure>
            # in the XML may still relaunch us. Worst case the operator
            # restarts the service manually.
            pass

        # Give WinSW a moment to register the restart command, then exit.
        # os._exit (not sys.exit): bypass uvicorn's signal handlers and
        # Python's atexit hooks. faster-whisper has no on-disk state that
        # needs flushing; the OS reclaims handles and VRAM on process death.
        time.sleep(0.2)
        os._exit(0)

    threading.Timer(delay_sec, do_restart).start()
    return "winsw-restart-bang"
