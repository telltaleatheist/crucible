from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from crucible.host import app as app_module
from crucible.host import presence
from crucible.host.log import HostLog
from crucible.host.state import Engine, Owner
from crucible.platform.runner import RunResult
from crucible.platform.wsl_table import CRUCIBLE_DISTRO

LISTED = f"  NAME   STATE   VERSION\n  {CRUCIBLE_DISTRO}  Running  2\n"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Guest:
    """A guest whose unit reports `state` and whose server binds its port at `answers_at`."""

    def __init__(self, clock: Clock, *, state: str, answers_at: float | None) -> None:
        self.clock = clock
        self.state = state
        self.answers_at = answers_at
        self.calls: list[str] = []

    def run(self, argv: Sequence[str], *, timeout_s: float, env: Mapping[str, str] | None = None) -> RunResult:
        line = " ".join(argv)
        self.calls.append(line)
        if line.endswith("-l -v"):
            return RunResult(code=0, stdout=LISTED, stderr="", failure=None)
        if "systemctl is-active" in line:
            return RunResult(code=0 if self.state == "active" else 3, stdout=self.state + "\n", stderr="", failure=None)
        if "systemctl is-enabled" in line:
            return RunResult(code=0, stdout="enabled\n", stderr="", failure=None)
        return RunResult(code=0, stdout="", stderr="", failure=None)

    def get(self, url: str, *, timeout_s: float) -> object | None:
        if self.answers_at is not None and self.clock.now >= self.answers_at:
            return {"crucible": True}
        return None


def _boot(tmp_path: Path, guest: Guest, clock: Clock) -> tuple[presence.Presence, str]:
    log = HostLog(tmp_path / "host.log", tmp_path / "host.log.1")
    watcher = presence.PresenceWatcher(guest, log, monotonic=clock, sleep=clock.sleep)
    result = watcher.boot()
    return result, (tmp_path / "host.log").read_text(encoding="utf-8")


def _started(guest: Guest) -> bool:
    return any(call.endswith("systemctl start crucible") or " systemctl start " in call for call in guest.calls)


def test_a_first_start_that_binds_a_second_late_is_waited_for_not_recovered(tmp_path: Path) -> None:
    clock = Clock()
    guest = Guest(clock, state="active", answers_at=presence.BOOT_WAIT_SECONDS + 1.0)
    result, written = _boot(tmp_path, guest, clock)
    assert result.engine is Engine.RUNNING
    assert result.owner is Owner.WSL_UNIT
    assert "recovery" not in written, written
    assert not _started(guest)
    assert "is active: the server is still starting" in written
    assert f"up to {presence.UNIT_START_BUDGET_SECONDS}s more" in written


def test_an_activating_unit_is_waited_for_too(tmp_path: Path) -> None:
    clock = Clock()
    guest = Guest(clock, state="activating", answers_at=90.0)
    result, written = _boot(tmp_path, guest, clock)
    assert result.engine is Engine.RUNNING
    assert "recovery" not in written


def test_a_unit_that_is_not_starting_goes_straight_to_the_recovery(tmp_path: Path) -> None:
    clock = Clock()
    guest = Guest(clock, state="inactive", answers_at=None)
    result, written = _boot(tmp_path, guest, clock)
    assert _started(guest), "an inactive unit is something that failed; the recovery starts it"
    assert f"{presence.UNIT_NAME} is inactive, so it is not starting; running the recovery" in " ".join(written.split())
    assert clock.now < presence.BOOT_WAIT_SECONDS + presence.UNIT_START_BUDGET_SECONDS, "no start budget for a unit that is not starting"
    assert result.engine is Engine.FAILED


def test_a_unit_that_never_answers_within_its_budget_is_then_a_recovery(tmp_path: Path) -> None:
    clock = Clock()
    guest = Guest(clock, state="active", answers_at=None)
    result, written = _boot(tmp_path, guest, clock)
    assert _started(guest)
    assert f"within {presence.BOOT_WAIT_SECONDS + presence.UNIT_START_BUDGET_SECONDS}s; running the recovery" in written
    assert clock.now >= presence.BOOT_WAIT_SECONDS + presence.UNIT_START_BUDGET_SECONDS
    assert result.engine is Engine.FAILED


def test_the_settle_ceiling_covers_the_longest_boot() -> None:
    assert app_module.PRESENCE_SETTLE_CEILING_SECONDS >= (
        presence.BOOT_WAIT_SECONDS + presence.RECIPE_TIMEOUT_SECONDS + presence.UNIT_START_BUDGET_SECONDS
    )
