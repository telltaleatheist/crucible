"""The install's guest restart is `crucible service restart`, which waits for the server.

Before 1.0.115 the migrate-config step ran `service stop && service start`: between the
two the tray's watch saw a stopped unit and started it itself, and the step ended when
systemctl returned, not when the server answered. `service restart` is one systemd
restart and then a wait of up to RESTART_ANSWER_SECONDS for /v1/info, refused
restart_not_answering by name, so the install fails on the server not answering, here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from crucible.cli import service_cmd
from crucible.host import installer
from crucible.platform.errors import HostError
from crucible.platform.runner import RunResult


class Guest:
    def __init__(self, restart: RunResult) -> None:
        self.restart = restart
        self.streamed: list[tuple[list[str], float]] = []

    def stream(self, argv, *, timeout_s, on_line, env=None) -> RunResult:
        self.streamed.append((list(argv), timeout_s))
        for line in self.restart.stdout.splitlines():
            on_line(line, "stdout")
        for line in self.restart.stderr.splitlines():
            on_line(line, "stderr")
        return self.restart


def _walk(tmp_path: Path, runner: Guest) -> installer.EngineInstall:
    return installer.EngineInstall(
        runner, lambda event: None, release="1.0.115", home=tmp_path,
        install_sh_url="https://example.invalid/install.sh",
    )


def test_the_guest_restart_is_one_restart_that_waits_never_stop_then_start(tmp_path: Path) -> None:
    runner = Guest(RunResult(code=0, stdout="restarted crucible.service\n", stderr="", failure=None))
    _walk(tmp_path, runner)._restart_guest_engine()
    [(argv, timeout_s)] = runner.streamed
    script = argv[-1]
    assert "capability --write && " in script
    assert script.endswith('service restart')
    assert "service stop" not in script and "service start" not in script
    assert timeout_s == installer.GUEST_RESTART_TIMEOUT_SECONDS


def test_the_step_outlasts_the_restart_s_own_wait() -> None:
    # The wait for the server to answer belongs to `service restart`; the
    # installer's bound must never be what ends it first, or the install would
    # say "timed out" where the guest was about to say restart_not_answering.
    assert installer.GUEST_RESTART_TIMEOUT_SECONDS > service_cmd.RESTART_ANSWER_SECONDS + 60


def test_a_server_that_never_answers_fails_the_install_by_name(tmp_path: Path) -> None:
    said = (
        "restart_not_answering: the service restarted, and nothing answered on "
        "http://127.0.0.1:7100/v1 within 120 s."
    )
    runner = Guest(RunResult(code=1, stdout="restarted crucible.service\n", stderr=said, failure=None))
    with pytest.raises(HostError) as caught:
        _walk(tmp_path, runner)._restart_guest_engine()
    assert caught.value.code == "guest_restart_failed"
    assert "restart_not_answering" in str(caught.value)
    assert "crucible service restart" in str(caught.value)
