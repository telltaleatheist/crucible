from __future__ import annotations

import sys
from pathlib import Path

import pytest

from crucible import accelerator, ladder, procgroup

from .conftest import end_process_tree

DEAF = (
    "import signal, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "if hasattr(signal, 'SIGBREAK'):\n"
    "    signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
    "time.sleep(60)\n"
)


@pytest.fixture(autouse=True)
def quick_clocks(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_card(*args: object) -> list[str]:
        raise accelerator.ProbeError("no card in this test")

    monkeypatch.setattr(accelerator, "_nvidia_smi", no_card)
    monkeypatch.setattr(ladder, "SMOKE_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(procgroup, "STOP_TIMEOUT_SECONDS", 2.0)


def test_a_smoke_past_its_clock_is_asked_to_stop_not_killed() -> None:
    result, why, _ = ladder._run_script(
        Path(sys.executable), "import time\ntime.sleep(60)\n"
    )
    assert result is None
    assert why == "did not finish within 1 s"


def test_a_smoke_deaf_to_sigterm_is_left_running_and_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[int] = []
    real_popen = ladder.subprocess.Popen

    def watched(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process.pid)
        return process

    monkeypatch.setattr(ladder.subprocess, "Popen", watched)
    try:
        result, why, _ = ladder._run_script(Path(sys.executable), DEAF)
        assert result is None
        assert f"`kill {spawned[0]}` (never -9)" in why
        assert "does not SIGKILL" in why
    finally:
        monkeypatch.undo()
        for pid in list(spawned):
            end_process_tree(pid)
