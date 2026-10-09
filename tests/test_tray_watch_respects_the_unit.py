from __future__ import annotations

from pathlib import Path

import pytest

from crucible.host import log, presence
from crucible.host.state import Distro, Engine, Owner

from .test_host import Scripted, ok, ticking


@pytest.fixture()
def host_log(tmp_path: Path) -> log.HostLog:
    return log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1")


def _unit(active: str, sub: str, result: str) -> dict:
    return {
        "systemctl show": ok(f"ActiveState={active}\nSubState={sub}\nResult={result}\n"),
        "is-enabled": ok("enabled\n"),
    }


def _started(runner: Scripted) -> bool:
    return any(call[-2:] == ["start", "crucible.service"] for call in runner.calls)


def _watcher(runner: Scripted, host_log: log.HostLog) -> presence.PresenceWatcher:
    return presence.PresenceWatcher(runner, host_log, monotonic=ticking(), sleep=lambda _s: None)


def test_a_unit_stopped_on_purpose_is_left_stopped_and_not_asked_again(
    host_log: log.HostLog,
) -> None:
    runner = Scripted(answers=_unit("inactive", "dead", "success"), pings=[])
    watcher = _watcher(runner, host_log)

    first = watcher.poll(Distro.PRESENT, Owner.WSL_UNIT)
    assert first.engine is Engine.STOPPED
    assert "crucible service stop" in first.detail
    assert not _started(runner), "the operator's stop was undone by a recovery"

    runner.calls.clear()
    second = watcher.poll(Distro.PRESENT, Owner.WSL_UNIT)
    assert second.engine is Engine.STOPPED
    assert second.detail == first.detail
    assert not any("systemctl" in " ".join(call) for call in runner.calls)


@pytest.mark.parametrize(
    "active,sub",
    [("activating", "start"), ("activating", "auto-restart"), ("deactivating", "stop-sigterm"),
     ("active", "running")],
)
def test_a_unit_systemd_is_bringing_up_is_waited_for_not_started_twice(
    host_log: log.HostLog, active: str, sub: str
) -> None:
    runner = Scripted(answers=_unit(active, sub, "success"), pings=[])
    watcher = _watcher(runner, host_log)
    seen = watcher.poll(Distro.PRESENT, Owner.WSL_UNIT)
    assert seen.engine is Engine.STARTING
    assert "systemd is bringing it up" in seen.detail
    assert not _started(runner)


def test_a_restart_that_comes_up_is_running_with_no_recovery_spent(
    host_log: log.HostLog,
) -> None:
    runner = Scripted(answers=_unit("activating", "start", "success"), pings=[])
    watcher = _watcher(runner, host_log)
    assert watcher.poll(Distro.PRESENT, Owner.WSL_UNIT).engine is Engine.STARTING
    runner.pings.append(200)
    assert watcher.poll(Distro.PRESENT, Owner.WSL_UNIT).engine is Engine.RUNNING
    assert not _started(runner)


def test_a_failed_unit_still_gets_its_one_recovery(host_log: log.HostLog) -> None:
    runner = Scripted(answers=_unit("failed", "failed", "exit-code"), pings=[])
    watcher = _watcher(runner, host_log)
    seen = watcher.poll(Distro.PRESENT, Owner.WSL_UNIT)
    assert seen.engine is Engine.STOPPED
    assert _started(runner)


def test_a_unit_systemd_cannot_be_asked_about_still_gets_its_recovery(
    host_log: log.HostLog,
) -> None:
    runner = Scripted(answers={"is-enabled": ok("enabled\n")}, pings=[])
    watcher = _watcher(runner, host_log)
    watcher.poll(Distro.PRESENT, Owner.WSL_UNIT)
    assert _started(runner)


def test_unit_activity_is_read_as_properties() -> None:
    read = presence.parse_unit_activity(ok("Result=success\nActiveState=inactive\nSubState=dead\n"))
    assert read.readable and read.stopped_on_purpose and not read.coming_up
    unreadable = presence.parse_unit_activity(ok(""))
    assert not unreadable.readable
    assert not unreadable.stopped_on_purpose and not unreadable.coming_up
