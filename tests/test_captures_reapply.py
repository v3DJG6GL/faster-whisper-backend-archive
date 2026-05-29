"""Tests for captures_reapply.start()/status() worker management.

_run() lazy-imports the heavy `main` module and walks the whole captures DB,
so we never run the real worker: threading.Thread is monkeypatched to a stub
that records the target but never executes it. We assert start() is
idempotent (a second call while "running" returns state without spawning a
2nd worker) and that status() returns a dict copy of the live state.

The conftest autouse fixture resets _worker/_state between tests, but it
seeds _state with a different key set than the module's real schema, so each
test first restores the canonical idle state (the shape start() expects).
"""

import threading

import pytest

import captures_reapply

# The module's canonical idle state (start() reads _state["status"]).
_IDLE = {
    "status": "idle",
    "started_ts": None,
    "finished_ts": None,
    "total": 0,
    "processed": 0,
    "captures_updated": 0,
    "groups_updated": 0,
    "error": None,
}


@pytest.fixture(autouse=True)
def _canonical_idle():
    """conftest reset uses a foreign key set; restore the real idle schema."""
    captures_reapply._state.clear()
    captures_reapply._state.update(_IDLE)
    captures_reapply._worker = None
    yield


class _FakeThread:
    """Records the target/name but never runs it (no real worker)."""
    created: list["_FakeThread"] = []

    def __init__(self, target=None, daemon=None, name=None, **kw):
        self.target = target
        self.daemon = daemon
        self.name = name
        self.started = False
        _FakeThread.created.append(self)

    def start(self):
        self.started = True


@pytest.fixture
def fake_thread(monkeypatch):
    _FakeThread.created = []
    monkeypatch.setattr(captures_reapply.threading, "Thread", _FakeThread)
    return _FakeThread


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------

def test_status_returns_dict_copy():
    s = captures_reapply.status()
    assert s == _IDLE
    # Mutating the returned dict must not affect module state.
    s["status"] = "tampered"
    assert captures_reapply._state["status"] == "idle"


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------

def test_start_spawns_one_worker(fake_thread):
    state = captures_reapply.start()
    assert state["status"] == "running"
    assert state["started_ts"] is not None
    assert len(fake_thread.created) == 1
    t = fake_thread.created[0]
    assert t.target is captures_reapply._run
    assert t.daemon is True
    assert t.name == "reapply-rules"
    assert t.started is True
    # The module tracks the live worker.
    assert captures_reapply._worker is t


def test_start_resets_counters(fake_thread):
    # Pre-dirty the state to prove start() resets it.
    captures_reapply._state.update({
        "status": "done", "processed": 99, "captures_updated": 5,
        "groups_updated": 3, "error": "old",
    })
    state = captures_reapply.start()
    assert state["status"] == "running"
    assert state["processed"] == 0
    assert state["captures_updated"] == 0
    assert state["groups_updated"] == 0
    assert state["error"] is None
    assert state["finished_ts"] is None


def test_start_is_idempotent_while_running(fake_thread):
    first = captures_reapply.start()
    assert first["status"] == "running"
    assert len(fake_thread.created) == 1
    worker1 = captures_reapply._worker

    # Second call while running: returns current state, no 2nd worker spawned.
    second = captures_reapply.start()
    assert second["status"] == "running"
    assert len(fake_thread.created) == 1  # still just one
    assert captures_reapply._worker is worker1


def test_start_after_done_spawns_again(fake_thread):
    captures_reapply.start()
    assert len(fake_thread.created) == 1
    # Simulate the worker finishing.
    captures_reapply._state["status"] = "done"
    captures_reapply.start()
    assert len(fake_thread.created) == 2


def test_status_reflects_running_after_start(fake_thread):
    captures_reapply.start()
    assert captures_reapply.status()["status"] == "running"
