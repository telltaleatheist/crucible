"""A worker the kernel kills mid-job (Victoria's laptop, 2026-10-10: the guest OOM-killed
a YuE2 worker). The job fails saying how the worker ended - by the out-of-memory killer
when the kernel's count of OOM kills moved while it ran - and the server keeps answering
while it does. The worker is a fake on the CPU, killed with SIGKILL from this test."""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
import pytest
from fastapi import FastAPI

from crucible import workerexit, workers

from .conftest import FAKE_BACKEND
from .live_server import serve
from .test_audio_api import PROMPT, SFX, _envs, _weights
from .test_audio_api import idle_card, transcript  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL is POSIX")

PING_BUDGET_S = 1.0
PINGER = "pinger"


@pytest.fixture
def live(
    make_app: Callable[..., FastAPI],
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    idle_card: None,  # noqa: F811
    transcript: Path,  # noqa: F811
) -> Iterator[tuple[str, FastAPI]]:
    _envs(home, FAKE_BACKEND.kind, monkeypatch)
    _weights(home, SFX, FAKE_BACKEND.kind)
    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_STEP_S", "0.2")
    app = make_app(enable_audio=True)
    with serve(app) as base:
        yield base, app


def _state(base: str, auth: dict[str, str], job_id: str) -> dict[str, Any]:
    return httpx.get(f"{base}/v1/jobs/{job_id}", headers=auth, timeout=30.0).json()


def _until(what: str, check: Callable[[], Any], timeout: float = 30.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = check()
        if found:
            return found
        time.sleep(0.02)
    raise AssertionError(f"never: {what}")


class _Pinger:
    """GET /v1/ping on a fresh connection every 50 ms (each new connection is a
    getLogger call on the loop in uvicorn's protocol) and keep the slowest answer."""

    def __init__(self, base: str) -> None:
        self._base = base
        self._stop = threading.Event()
        self.slowest = 0.0
        self.answered = 0
        self._thread = threading.Thread(target=self._run, name=PINGER, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            assert httpx.get(f"{self._base}/v1/ping", timeout=10.0).status_code == 200
            self.slowest = max(self.slowest, time.monotonic() - started)
            self.answered += 1
            time.sleep(0.05)

    def __enter__(self) -> "_Pinger":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=15)


def _kill_the_worker_mid_job(
    base: str, app: FastAPI, auth: dict[str, str], *, before_kill: Callable[[], None] = lambda: None
) -> tuple[dict[str, Any], _Pinger]:
    job_id = httpx.post(
        f"{base}/v1/jobs", headers=auth, timeout=30.0,
        json={"type": "audio", "model": SFX, "params": {"prompt": PROMPT, "steps": 40}},
    ).json()["job_id"]
    _until("the job runs", lambda: _state(base, auth, job_id)["status"] == "running")
    (pid,) = _until("the worker is up", lambda: app.state.residency.owned_pids())
    _until("the worker is generating", lambda: _state(base, auth, job_id)["progress"] > 0.05)
    with _Pinger(base) as pinger:
        before_kill()
        os.kill(pid, signal.SIGKILL)
        state = _until(
            "the job ends",
            lambda: (s := _state(base, auth, job_id))["status"] in ("failed", "done", "cancelled") and s,
        )
        time.sleep(1.0)
    return state, pinger


def test_a_worker_killed_mid_job_fails_the_job_by_signal_and_the_server_keeps_answering(
    live: tuple[str, FastAPI], auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    base, app = live
    takers: Counter[str] = Counter()
    acquire = logging._acquireLock  # type: ignore[attr-defined]

    def counted() -> None:
        name = threading.current_thread().name
        if name not in ("MainThread", PINGER):  # this test's own HTTP clients
            takers[name] += 1
        acquire()

    monkeypatch.setattr(logging, "_acquireLock", counted)
    state, pinger = _kill_the_worker_mid_job(base, app, auth)

    assert state["status"] == "failed", state
    message = state["error"]["message"]
    assert state["error"]["code"] == "worker_failed"
    assert "fake_audio_worker.py was killed" in message, message
    assert "SIGKILL, signal 9" in message, message
    assert "exited -9" not in message, message
    if workerexit.read_oom_count() is not None:
        assert "It was not the out-of-memory killer" in message, message
    assert pinger.answered > 5
    assert pinger.slowest < PING_BUDGET_S, pinger.slowest
    # The worker's death takes logging's module lock on no thread but the loop's (the
    # one uvicorn's per-connection getLogger runs on): nothing else here could hold it.
    assert set(takers) == {"crucible-test-server"}, takers


def test_a_worker_the_out_of_memory_killer_ended_is_said_to_be_that(
    live: tuple[str, FastAPI], auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    base, app = live
    kills = [3]
    monkeypatch.setattr(
        workers.workerexit, "read_oom_count",
        lambda: workerexit.OomCount(kills[0], "/sys/fs/cgroup/system.slice/crucible.service/memory.events"),
    )
    state, pinger = _kill_the_worker_mid_job(
        base, app, auth, before_kill=lambda: kills.__setitem__(0, 4)
    )
    message = state["error"]["message"]
    assert state["status"] == "failed", state
    assert (
        "fake_audio_worker.py was killed by the out-of-memory killer (SIGKILL, signal 9) "
        "in the middle of a request"
    ) in message, message
    assert "went from 3 to 4" in message and "crucible.service/memory.events" in message
    assert pinger.slowest < PING_BUDGET_S, pinger.slowest
    # The resident worker is gone with it: the next job loads a new one.
    assert app.state.residency.owned_pids() == frozenset()
