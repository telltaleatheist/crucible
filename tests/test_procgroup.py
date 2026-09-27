from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from crucible import procgroup

from .conftest import end_process_tree

DEAF = (
    "import signal, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "if hasattr(signal, 'SIGBREAK'):\n"
    "    signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
    "print('deaf', flush=True)\n"
    "time.sleep(60)\n"
)


def spawn(script: str) -> "subprocess.Popen[bytes]":
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        **procgroup.own_group(),
    )


def test_a_process_that_already_exited_needs_no_stop(tmp_path: Path) -> None:
    process = spawn("pass")
    process.wait(timeout=30)
    procgroup.stop_gracefully(process, "done", 1.0, tmp_path / "log")


def test_a_process_that_heeds_the_signal_is_stopped(tmp_path: Path) -> None:
    process = spawn("import time\ntime.sleep(60)\n")
    try:
        procgroup.stop_gracefully(process, "sleeper", 30.0, tmp_path / "log")
        assert process.poll() is not None
    finally:
        end_process_tree(process.pid)


def test_a_deaf_process_is_named_with_a_kill_command_never_sigkilled(
    tmp_path: Path,
) -> None:
    process = spawn(DEAF)
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == b"deaf"
        if procgroup.platform_kind() == procgroup.WIN32:
            procgroup.stop_gracefully(process, "deaf", 1.0, tmp_path / "log")
            assert process.poll() is not None
            return
        with pytest.raises(procgroup.ProcessGroupError) as caught:
            procgroup.stop_gracefully(process, "deaf", 1.0, tmp_path / "log")
        said = str(caught.value)
        assert f"`kill {process.pid}` (never -9)" in said
        assert "does not SIGKILL" in said
        assert str(tmp_path / "log") in said
        assert process.poll() is None
    finally:
        end_process_tree(process.pid)


def test_the_stop_budget_counts_the_win32_force_step() -> None:
    budget = procgroup.stop_budget_seconds(30.0)
    if procgroup.platform_kind() == procgroup.WIN32:
        assert budget == 30.0 + 2 * procgroup.KILL_WAIT_SECONDS
    else:
        assert budget == 30.0


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX")
def test_groups_are_asked_to_stop_and_a_gone_pid_is_not_a_failure() -> None:
    process = spawn("import time\ntime.sleep(60)\n")
    gone = spawn("pass")
    gone.wait(timeout=30)
    try:
        failed = procgroup.ask_groups_to_stop(frozenset({process.pid, gone.pid}))
        assert failed == frozenset()
        process.wait(timeout=30)
    finally:
        end_process_tree(process.pid)


@pytest.mark.skipif(sys.platform != "win32", reason="win32 has no groups to signal")
def test_asking_groups_to_stop_off_posix_refuses() -> None:
    with pytest.raises(procgroup.ProcessGroupError):
        procgroup.ask_groups_to_stop(frozenset({1}))
