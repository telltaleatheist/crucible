from __future__ import annotations

import sys
from pathlib import Path

import pytest

from crucible import cli, service
from crucible.cli import service_cmd

from .conftest import FAKE_BACKEND
from .test_service import Runner, answer


@pytest.fixture
def user_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "operator-home"
    root.mkdir()
    monkeypatch.setattr(service, "user_home", lambda: root)
    return root


def _a_user_unit(user_home: Path) -> None:
    unit = service.unit_path(user_home, service.USER_SCOPE)
    unit.parent.mkdir(parents=True)
    unit.write_text("[Unit]\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def not_wsl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "in_wsl", lambda: False)


def test_restart_is_one_systemctl_restart(user_home: Path) -> None:
    _a_user_unit(user_home)
    runner = Runner()
    lines = service.restart(service.SYSTEMD, home=user_home, runner=runner)
    assert runner.calls == [("systemctl", "--user", "restart", "crucible.service")]
    assert lines == ["restarted crucible.service"]


def test_restart_refuses_when_nothing_is_installed(user_home: Path) -> None:
    with pytest.raises(service.ServiceError) as caught:
        service.restart(service.SYSTEMD, home=user_home, runner=Runner())
    assert "crucible service install" in str(caught.value)


@pytest.mark.skipif(sys.platform == "win32", reason="launchd's domain is a POSIX uid")
def test_launchd_restart_is_a_kickstart_that_kills_first(user_home: Path) -> None:
    plist = service.plist_path(user_home)
    plist.parent.mkdir(parents=True)
    plist.write_text("<plist/>", encoding="utf-8")
    loaded = Runner({("launchctl", "list"): answer(out="1234 0 com.crucible.serve\n")})
    lines = service.restart(service.LAUNCHD, home=user_home, runner=loaded)
    assert loaded.ran("launchctl", "kickstart", "-k")
    assert not loaded.ran("launchctl", "bootout")
    assert lines == ["restarted com.crucible.serve"]


def test_the_cli_restart_waits_for_the_server_to_answer(
    home: Path,
    user_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init", "--enable-echo"]) == 0
    _a_user_unit(user_home)
    capsys.readouterr()
    runner = Runner()
    monkeypatch.setattr(service, "subprocess_runner", runner)
    answers = iter([None, None, object()])
    monkeypatch.setattr(cli.common, "server_here", lambda _c, _b: next(answers))
    assert cli.main(["service", "restart"]) == 0
    out = capsys.readouterr().out
    assert "restarted crucible.service" in out
    assert "answering on http://127.0.0.1:" in out
    assert runner.ran("systemctl", "--user", "restart")


def test_a_restart_nothing_answers_is_refused_by_name_after_its_budget(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    config, backend = cli.common.here()
    monkeypatch.setattr(cli.common, "server_here", lambda _c, _b: None)
    clock = iter(float(n) for n in range(0, 1000, 5))
    took = service_cmd.wait_for_answer(
        config, backend, budget_s=30.0, say_every_s=10.0,
        clock=lambda: next(clock), sleep=lambda _s: None,
    )
    assert took is None
    assert "still waiting for it to answer" in capsys.readouterr().out


def test_the_service_runner_never_hands_its_child_the_callers_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The root door is `sudo -n` (use_pty relays a terminal stdin in raw mode) or a
    # nested `wsl.exe -u root` (which attaches to the outer session's console). A
    # `crucible service stop` typed into `wsl -d crucible` must give them neither.
    import subprocess

    seen: dict = {}

    def run(argv, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(service.subprocess, "run", run)
    service.subprocess_runner(["systemctl", "stop", service.UNIT_NAME])
    assert seen.get("stdin") is subprocess.DEVNULL
    assert seen.get("capture_output") is True
