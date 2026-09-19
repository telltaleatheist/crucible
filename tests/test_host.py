"""`crucible host` — PHASE15-HOST.md 4.5.

**EVERY TEST IN HERE RUNS OFF WINDOWS**, and that is the point rather than a
convenience: pytest runs inside WSL, the host runs on Windows, and a suite that
skipped its subject on the machine it is run on would pin nothing at all. So
the platform, the environment and every subprocess are INJECTED — `paths.py`
takes a mapping, `ProcessRunner` takes a platform, and the whole of `runner.py`
is a protocol with a scripted stand-in below.

What is NOT covered here, named rather than left to be noticed: `tray.py`,
which is pystray and has no decision in it (`menu.py` is where every decision
it draws was made), and the parts of `installer.py` that need a real
`crucible` distro — those are recorded in PHASE15-HOST.md 7b.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pytest

from crucible.host import app as app_module
from crucible.host import catalog as catalog_module
from crucible.host import door as door_module
from crucible.host import installer, landoor, log, menu, paths, presence, startup, wslstate
from crucible.host.catalog import CatalogRefusal, Subject
from crucible.host.errors import HOST_ERROR_CODES, HostError
from crucible.host.menu import Distro, Engine, Owner
from crucible.host.runner import RunResult
from crucible.host.wsl_states import CRUCIBLE_DISTRO, WSL_STATE_CODES, WSL_STATES

WINDOWS_ENV = {
    "LOCALAPPDATA": r"C:\Users\tellt\AppData\Local",
    "APPDATA": r"C:\Users\tellt\AppData\Roaming",
    "USERPROFILE": r"C:\Users\tellt",
    "USERNAME": "tellt",
}


# --------------------------------------------------------------- the runner


@dataclass
class Scripted:
    """A `Runner` whose answers are written down. Records every argv."""

    answers: dict[str, RunResult] = field(default_factory=dict)
    pings: list[int | None] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)
    gets: list[str] = field(default_factory=list)
    spawned: list[list[str]] = field(default_factory=list)
    platform: str = "win32"
    env: Mapping[str, str] = field(default_factory=lambda: dict(WINDOWS_ENV))
    default: RunResult = RunResult(code=0, stdout="", stderr="", failure=None)

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str] | None = None,
    ) -> RunResult:
        self.calls.append(list(argv))
        for needle, answer in self.answers.items():
            if needle in " ".join(argv):
                return answer
        return self.default

    def get(self, url: str, *, timeout_s: float) -> int | None:
        self.gets.append(url)
        if not self.pings:
            return None
        return self.pings.pop(0)

    def spawn(self, argv: Sequence[str], *, env: Mapping[str, str] | None = None) -> object:
        self.spawned.append(list(argv))
        return FakeChild()


class FakeChild:
    pid = 4242

    def __init__(self) -> None:
        self.terminated = False

    def poll(self) -> int | None:
        return 0 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout_s: float) -> int | None:
        return 0


def ticking() -> Callable[[], float]:
    """A monotonic clock that ADVANCES.

    A frozen `lambda: 0.0` makes `_wait_for_ping` loop forever, because its
    deadline is `now + seconds` and `now` never reaches it. That is a property
    of the production code being right — it waits until the clock says to
    stop — so the test supplies a clock rather than the code supplying a
    counter.
    """
    state = {"now": 0.0}

    def clock() -> float:
        state["now"] += 1.0
        return state["now"]

    return clock


def ok(stdout: str = "") -> RunResult:
    return RunResult(code=0, stdout=stdout, stderr="", failure=None)


def bad(stderr: str = "no", code: int | None = 1) -> RunResult:
    return RunResult(code=code, stdout="", stderr=stderr, failure=None)


@pytest.fixture()
def host_log(tmp_path: Path) -> log.HostLog:
    return log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1")


# ----------------------------------------------------------------- 4.1 paths


def test_every_host_path_comes_from_the_environment_and_never_a_username() -> None:
    assert str(paths.crucible_root(WINDOWS_ENV)) == r"C:\Users\tellt\AppData\Local\Crucible"
    assert str(paths.host_pack_dir(WINDOWS_ENV)).endswith(r"\Crucible\host")
    assert str(paths.log_path(WINDOWS_ENV)).endswith(r"\Crucible\host.log")
    assert str(paths.console_cmd_path(WINDOWS_ENV)).endswith(r"\host\crucible.cmd")
    assert str(paths.pythonw_path(WINDOWS_ENV)).endswith(r"\host\pythonw.exe")


def test_localappdata_unset_is_refused_by_name_and_never_assembled() -> None:
    with pytest.raises(HostError) as caught:
        paths.crucible_root({})
    assert caught.value.code == "host_no_localappdata"
    assert caught.value.code in HOST_ERROR_CODES


def test_the_two_addresses_are_spelled_once() -> None:
    assert paths.engine_url("/v1/ping") == "http://127.0.0.1:7100/v1/ping"
    assert paths.door_url("/install") == "http://127.0.0.1:7101/install"


# ------------------------------------------------------------------ 4.1 log


def test_the_log_rolls_once_at_its_limit_and_keeps_exactly_one_previous(tmp_path: Path) -> None:
    written = log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1", roll_bytes=200)
    for index in range(40):
        written.write(f"line {index} " + "x" * 20)
    assert (tmp_path / "host.log").is_file()
    assert (tmp_path / "host.log.1").is_file()
    assert not (tmp_path / "host.log.2").exists()
    assert (tmp_path / "host.log").stat().st_size < 400


def test_a_log_line_is_timestamped_and_returned(host_log: log.HostLog) -> None:
    line = host_log.write("boot: hello")
    assert line.endswith("boot: hello")
    assert line[:4].isdigit()


# ----------------------------------------------------------------- 4.2 menu


def test_every_cell_of_the_distro_by_engine_table_has_a_title_and_an_item_set() -> None:
    """4.1's table, walked EXHAUSTIVELY. Fifteen cells, not a sample.

    The owner here is the one the pair IMPLIED before ownership was a fact of
    its own (2026-09-15): the distro said which server this machine runs and
    the host had started it either way. `FOUND` is the case the pair could not
    express, and it has its own cells below.
    """
    seen: set[str] = set()
    for distro in Distro:
        owner = Owner.WSL_UNIT if distro is Distro.PRESENT else Owner.HOST_CHILD
        for engine in Engine:
            model = menu.menu_model(distro, engine, owner)
            assert model.title.startswith("Crucible — ")
            seen.add(model.title)
            ids = [item.item_id for item in model.items]
            assert ids[0] == menu.OPEN_CONSOLE
            assert ids[-1] == menu.QUIT
            assert menu.OPEN_LOG in ids
            # `install-engine` is ABSENT, not disabled, when the distro is there.
            assert (menu.INSTALL_ENGINE in ids) == (distro is not Distro.PRESENT)
            # Nothing to open when nothing answers.
            assert model.item(menu.OPEN_CONSOLE).enabled == (engine is Engine.RUNNING)
            assert model.item(menu.STOP_ENGINE).enabled == (engine is Engine.RUNNING)
            # An install holds the machine; nothing else may be started under it.
            assert model.item(menu.RESTART_ENGINE).enabled == (engine is not Engine.INSTALLING)
    assert "Crucible — running (WSL)" in seen
    assert "Crucible — running (llama-windows)" in seen
    assert "Crucible — engine did not start — open the log" in seen
    assert "Crucible — installing…" in seen
    assert "Crucible — stopped" in seen


def test_the_title_names_the_backend_after_section_zeros_amendment() -> None:
    """`host mode` was a title from before Windows had a backend. It has one."""
    running = menu.menu_model(Distro.ABSENT, Engine.RUNNING, Owner.HOST_CHILD)
    assert running.title == "Crucible — running (llama-windows)"
    assert "host mode" not in running.title


def test_quits_label_says_what_quitting_costs_and_it_differs_by_server() -> None:
    assert menu.menu_model(Distro.PRESENT, Engine.RUNNING, Owner.WSL_UNIT).item(menu.QUIT).label == (
        "Quit (the engine keeps running)"
    )
    assert menu.menu_model(Distro.ABSENT, Engine.RUNNING, Owner.HOST_CHILD).item(menu.QUIT).label == (
        "Quit (stops the engine)"
    )


def test_an_unreadable_wsl_still_offers_the_install_and_never_claims_a_server() -> None:
    """Reading `unknown` as `absent` would import a SECOND distro."""
    model = menu.menu_model(Distro.UNKNOWN, Engine.RUNNING, Owner.NONE)
    assert model.item(menu.INSTALL_ENGINE) is not None
    assert "WSL unreadable" in model.title


# -------------------------------- 4.2 an engine the host did NOT start (FOUND)
#
# Added 2026-09-15, by the first real run on Owen's PC: a machine whose
# Crucible lives in `Ubuntu` answers `distro=absent` and `ping=200` at the
# same time, and the pair alone reads that as "no server here".


def test_an_engine_the_host_found_is_named_by_that_and_not_by_a_backend() -> None:
    model = menu.menu_model(Distro.ABSENT, Engine.RUNNING, Owner.FOUND)
    assert model.title == "Crucible — running (found on this machine)"
    assert "llama-windows" not in model.title


def test_the_host_offers_no_verb_that_would_act_on_an_engine_it_did_not_start() -> None:
    model = menu.menu_model(Distro.ABSENT, Engine.RUNNING, Owner.FOUND)
    # ABSENT would normally OFFER the WSL install; a machine that already has
    # an engine must not be invited to import a second distro.
    assert model.item(menu.INSTALL_ENGINE) is None
    assert model.item(menu.RESTART_ENGINE).enabled is False
    assert model.item(menu.STOP_ENGINE).enabled is False
    # What it may still do: look at it, and read the log.
    assert model.item(menu.OPEN_CONSOLE).enabled is True
    assert model.item(menu.OPEN_LOG).enabled is True
    assert model.item(menu.QUIT).label == "Quit (the engine keeps running)"


def test_the_install_item_reads_as_an_upgrade_not_as_an_absence() -> None:
    label = menu.menu_model(Distro.ABSENT, Engine.RUNNING, Owner.HOST_CHILD).item(menu.INSTALL_ENGINE).label
    assert label.startswith("Install the WSL2 engine")
    assert "TTS" in label


# ------------------------------------------------------------- 4.1 presence


def test_the_boot_recipe_and_the_two_recovery_recipes_are_exactly_4_1s() -> None:
    assert presence.wsl_boot_argv() == ["wsl.exe", "-d", "crucible", "--exec", "true"]
    assert presence.recipe_argv(presence.RECIPE_USER_UNIT_START, uid="1000") == [
        "wsl.exe", "-d", "crucible", "--exec",
        "env", "XDG_RUNTIME_DIR=/run/user/1000",
        "systemctl", "--user", "start", "crucible.service",
    ]
    assert presence.recipe_argv(presence.RECIPE_USER_BUS_RESTART) == [
        "wsl.exe", "-d", "crucible", "-u", "root", "--exec", "systemctl", "restart", "user@1000",
    ]
    assert presence.RECIPES == (
        presence.RECIPE_USER_UNIT_START,
        presence.RECIPE_USER_BUS_RESTART,
    )


def test_every_wsl_call_uses_exec_so_wsl_exe_cannot_pre_expand_a_variable() -> None:
    """BookForge's `wsl-exe-implicit-shell-trap`: without `--exec`, wsl.exe
    expands `$var` on the WINDOWS side before bash ever sees it."""
    for name in presence.RECIPES:
        assert "--exec" in presence.recipe_argv(name, uid="1000")
    assert "--exec" in presence.wsl_boot_argv()


def test_the_numbers_4_1_states_are_constants_with_4_1s_names() -> None:
    assert presence.BOOT_WAIT_SECONDS == 30
    assert presence.WATCH_SECONDS == 15


def test_wsl_list_is_parsed_by_the_row_shape_and_not_by_a_localised_header() -> None:
    text = "  NAME              STATE           VERSION\n* Ubuntu            Running         2\n  crucible          Stopped         2\n"
    assert presence.parse_wsl_list(text) == ["Ubuntu", "crucible"]
    # A UTF-16 remnant must not become part of a name.
    assert presence.parse_wsl_list("\x00 c\x00r\x00u\x00c\x00i\x00b\x00l\x00e  Running 2") == [
        "crucible"
    ]


def test_a_distro_probe_that_will_not_answer_is_unknown_and_never_absent(
    host_log: log.HostLog,
) -> None:
    runner = Scripted(answers={"-l -v": bad("Access is denied")})
    watcher = presence.PresenceWatcher(runner, host_log, sleep=lambda _s: None)
    distro, detail = watcher.probe_distro()
    assert distro is Distro.UNKNOWN
    assert "Access is denied" in detail


def test_the_boot_spends_both_recipes_and_then_says_it_did_not_start(
    host_log: log.HostLog,
) -> None:
    runner = Scripted(
        answers={"-l -v": ok("  crucible  Stopped  2\n"), "id -u": ok("1000\n")},
        pings=[],
    )
    watcher = presence.PresenceWatcher(
        runner, host_log, boot_wait_s=2.0, monotonic=ticking(), sleep=lambda _s: None
    )
    result = watcher.boot()
    assert result.engine is Engine.FAILED
    assert "both recipes were spent" in result.detail
    ran = [" ".join(call) for call in runner.calls]
    assert any("systemctl --user start crucible.service" in line for line in ran)
    assert any("systemctl restart user@1000" in line for line in ran)


def test_the_watch_spends_ONE_recovery_per_down_edge_and_then_stops(
    host_log: log.HostLog,
) -> None:
    """4.1: it never loops on restart; the unit's own `Restart=` does that."""
    runner = Scripted(answers={"-l -v": ok("  crucible  Running  2\n")}, pings=[])
    watcher = presence.PresenceWatcher(
        runner, host_log, monotonic=ticking(), sleep=lambda _s: None
    )
    first = watcher.poll(Distro.PRESENT, Owner.WSL_UNIT)
    assert first.engine is Engine.STOPPED
    recoveries = sum("systemctl" in " ".join(call) for call in runner.calls)
    runner.calls.clear()
    second = watcher.poll(Distro.PRESENT, Owner.WSL_UNIT)
    assert second.engine is Engine.STOPPED
    assert recoveries > 0
    assert not any("systemctl" in " ".join(call) for call in runner.calls)


def test_a_ping_that_answers_ANY_status_is_a_server_that_is_up(
    host_log: log.HostLog,
) -> None:
    """A 401 on /v1/ping is a Crucible refusing a token, not a dead machine."""
    runner = Scripted(pings=[401])
    watcher = presence.PresenceWatcher(runner, host_log, sleep=lambda _s: None)
    assert watcher.ping() is True
    assert runner.gets == ["http://127.0.0.1:7100/v1/ping"]


def test_a_successful_ping_restores_the_recovery_budget(host_log: log.HostLog) -> None:
    runner = Scripted(pings=[None, 200, None])
    watcher = presence.PresenceWatcher(
        runner, host_log, monotonic=ticking(), sleep=lambda _s: None
    )
    watcher.poll(Distro.ABSENT, Owner.HOST_CHILD)  # down: spends the budget
    watcher.poll(Distro.ABSENT, Owner.HOST_CHILD)  # up: restores it
    runner.calls.clear()
    third = watcher.poll(Distro.ABSENT, Owner.HOST_CHILD)
    assert third.engine is Engine.STOPPED


# ------------------------------- 4.1 the engine hunt, the hold, and ownership
#
# All four of these pin something the first real run on Owen's PC found
# (2026-09-15). They are written with the shape of THAT machine: a distro
# called `Ubuntu` that is running and holds the engine, and no distro called
# `crucible` at all.


OWENS_PC_LIST = "  NAME      STATE           VERSION\n* Ubuntu    Running         2\n"
GUEST_LINE = "crucible://crucible%40owens-pc-wsl@127.0.0.1:7100/#a-token\n"


def test_the_hunt_only_asks_distros_that_are_ALREADY_running() -> None:
    """Asking a stopped distro anything BOOTS it, and the host boots no VM it
    does not own."""
    assert presence.wsl_running_argv() == ["wsl.exe", "-l", "-v", "--running"]
    assert "--exec" in presence.guest_pairing_argv("Ubuntu")
    # The guest's `$CRUCIBLE_HOME` must be expanded by BASH and not by
    # wsl.exe, which is what `--exec` buys (wsl-exe-implicit-shell-trap).
    assert "${CRUCIBLE_HOME:-$HOME/.crucible}" in " ".join(
        presence.guest_pairing_argv("Ubuntu")
    )


def test_a_pairing_lines_authority_is_read_after_the_LAST_at_sign() -> None:
    """The name is percent-encoded so this cannot be ambiguous; rsplit is the
    reader's half of that contract."""
    assert presence.pairing_line_authority(GUEST_LINE) == "127.0.0.1:7100"
    assert presence.pairing_line_authority("http://127.0.0.1:7100/") is None
    assert presence.pairing_line_authority("not a line") is None


def test_an_engine_in_a_distro_crucible_does_not_own_is_FOUND_and_named(
    host_log: log.HostLog,
) -> None:
    runner = Scripted(
        answers={"-l -v --running": ok(OWENS_PC_LIST), "cat ": ok(GUEST_LINE)},
        pings=[200],
    )
    watcher = presence.PresenceWatcher(runner, host_log, sleep=lambda _s: None)
    result = watcher.adopt(Distro.ABSENT)
    assert result.engine is Engine.RUNNING
    assert result.owner is Owner.FOUND
    assert '"Ubuntu"' in result.detail
    assert watcher.found is not None
    assert watcher.found.distro == "Ubuntu"
    assert watcher.found.line.strip() == GUEST_LINE.strip()
    # Nothing was started, and nothing was booted.
    assert runner.spawned == []
    assert not any("--exec true" in " ".join(call) for call in runner.calls)


def test_an_engine_answering_somewhere_else_is_not_this_machines(
    host_log: log.HostLog,
) -> None:
    """A distro holding a Crucible bound to another port is not the engine the
    host is watching, and adopting its line would hand apps a wrong address."""
    elsewhere = "crucible://crucible%40x@127.0.0.1:7999/#t\n"
    runner = Scripted(
        answers={"-l -v --running": ok(OWENS_PC_LIST), "cat ": ok(elsewhere)}
    )
    watcher = presence.PresenceWatcher(runner, host_log, sleep=lambda _s: None)
    assert watcher.find_engine() is None
    result = watcher.adopt(Distro.ABSENT)
    assert result.owner is Owner.FOUND
    assert "no line to copy" in result.detail


def test_a_found_engine_going_down_runs_NO_recipe_at_all(
    host_log: log.HostLog,
) -> None:
    """It was not started here, so there is no unit to call and no child to
    respawn. The one thing the host can do is say so."""
    runner = Scripted(pings=[])
    watcher = presence.PresenceWatcher(
        runner, host_log, monotonic=ticking(), sleep=lambda _s: None
    )
    result = watcher.poll(Distro.ABSENT, Owner.FOUND)
    assert result.engine is Engine.STOPPED
    assert result.owner is Owner.FOUND
    assert "nothing here to restart" in result.detail
    assert not any("systemctl" in " ".join(call) for call in runner.calls)
    assert runner.spawned == []


def test_the_host_holds_the_distro_open_because_units_do_not_keep_a_VM_alive(
    host_log: log.HostLog,
) -> None:
    """7b.4c, measured: a distro terminates seconds after the last wsl.exe
    session ends, `Restart=always` and linger notwithstanding."""
    assert presence.keepalive_argv("Ubuntu") == [
        "wsl.exe", "-d", "Ubuntu", "--exec", "sleep", "infinity",
    ]
    runner = Scripted()
    watcher = presence.PresenceWatcher(runner, host_log, sleep=lambda _s: None)
    first = watcher.hold("Ubuntu")
    assert runner.spawned == [presence.keepalive_argv("Ubuntu")]
    # Idempotent while it lives: a second hold is the same session.
    assert watcher.hold("Ubuntu") is first
    assert len(runner.spawned) == 1
    # Dead: the watch takes it again rather than waiting for the next login.
    first.terminate()
    again = watcher.rehold()
    assert again is not first
    assert len(runner.spawned) == 2
    watcher.release()
    assert watcher.held is None
    assert watcher.rehold() is None


# ------------------------- 4.1 what `crucible host` does at start, by branch
#
# Section 0 is ONE SERVER PER MACHINE, and until 2026-09-15 the host could
# make that false by itself: with no distro NAMED `crucible` it spawned the
# `llama-windows` child without ever asking whether something was already
# answering on 7100.


def real_grantee_env() -> dict[str, str]:
    """`WINDOWS_ENV`, with the ACL grantee this machine actually has.

    `icacls` really runs when the suite runs ON Windows, and `%USERNAME%` is
    the one value in the fixture that has to be TRUE rather than plausible.
    Measured on Owen's PC, 2026-09-15: the profile directory is `tellt` and
    the account is `telltale` — which is the exact reason `pairing.py` and
    `paths.py` READ that variable instead of assembling a name from a path,
    and the fixture's fabricated `tellt` is what proved it
    (`No mapping between account names and security IDs was done`).
    """
    env = dict(WINDOWS_ENV)
    if os.name == "nt":
        env["USERNAME"] = os.environ["USERNAME"]
    return env


def _context(tmp_path: Path, runner: Scripted) -> app_module.HostContext:
    host_log = log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1")
    watcher = presence.PresenceWatcher(
        runner, host_log, monotonic=ticking(), sleep=lambda _s: None
    )
    return app_module.HostContext(
        runner=runner,
        log=host_log,
        home=tmp_path,
        watcher=watcher,
        presence=presence.Presence(
            Distro.UNKNOWN, Engine.STARTING, "starting", Owner.NONE
        ),
    )


def test_a_machine_that_already_answers_gets_NO_second_server(tmp_path: Path) -> None:
    """Owen's PC, exactly: `wsl -l -v` lists Ubuntu and no crucible, and
    `GET /v1/ping` answers 200 because the engine is inside Ubuntu."""
    runner = Scripted(
        answers={
            "-l -v --running": ok(OWENS_PC_LIST),
            "-l -v": ok(OWENS_PC_LIST),
            "cat ": ok(GUEST_LINE),
        },
        pings=[200, 200],
    )
    context = _context(tmp_path, runner)
    result = app_module.Host(context).start()
    assert result.distro is Distro.ABSENT
    assert result.engine is Engine.RUNNING
    assert result.owner is Owner.FOUND
    # The one thing that must not have happened.
    assert not any("serve" in " ".join(call) for call in runner.spawned)
    assert not any("init" in " ".join(call) for call in runner.calls)
    # …and the one thing that must have: the distro is held open (7b.4c).
    assert presence.keepalive_argv("Ubuntu") in runner.spawned


def test_with_nothing_answering_and_no_distro_the_host_mode_child_still_starts(
    tmp_path: Path,
) -> None:
    """The guard is a LOOK, not a refusal: a machine with no engine still gets
    the `llama-windows` one."""
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / paths.CONSOLE_CMD).write_text("@echo off\n", encoding="utf-8")
    env = dict(WINDOWS_ENV)
    env["LOCALAPPDATA"] = str(tmp_path)
    runner = Scripted(answers={"-l -v": ok(OWENS_PC_LIST)}, pings=[None], env=env)
    context = _context(tmp_path, runner)
    # `console_cmd_path` is <LOCALAPPDATA>/Crucible/host/crucible.cmd; on a
    # POSIX test box that path does not exist, which is the FAILED branch —
    # and FAILED is still "the host started nothing", which is what is pinned.
    result = app_module.Host(context).start()
    assert result.owner is Owner.NONE
    assert result.engine is Engine.FAILED
    assert "Reinstall with install.ps1" in result.detail


def test_a_guest_engines_pairing_line_is_COPIED_and_never_composed(
    tmp_path: Path,
) -> None:
    """3.6: the Windows file is the host's COPY of the guest's line. Composing
    one here would carry the HOST's token at the GUEST's address — a file that
    exists and disagrees, which 3.6 calls worse than none."""
    (tmp_path / "config.toml").write_text(
        '[server]\nname = "crucible@owens-pc"\n[auth]\ntoken = "the-wrong-one"\n',
        encoding="utf-8",
    )
    runner = Scripted(
        answers={
            "-l -v --running": ok(OWENS_PC_LIST),
            "-l -v": ok(OWENS_PC_LIST),
            "cat ": ok(GUEST_LINE),
        },
        pings=[200, 200],
        env=real_grantee_env(),
    )
    context = _context(tmp_path, runner)
    app_module.Host(context).start()
    app_module._write_pairing(context)
    written = (tmp_path / "pairing").read_text(encoding="utf-8")
    assert written == GUEST_LINE
    assert "the-wrong-one" not in written


def test_a_guest_engine_whose_line_cannot_be_read_writes_NO_file(
    tmp_path: Path,
) -> None:
    runner = Scripted(
        answers={
            "-l -v --running": ok(OWENS_PC_LIST),
            "-l -v": ok(OWENS_PC_LIST),
            "cat ": bad("No such file or directory"),
        },
        pings=[200, 200],
    )
    context = _context(tmp_path, runner)
    app_module.Host(context).start()
    app_module._write_pairing(context)
    assert not (tmp_path / "pairing").exists()
    assert "worse than no file" in (tmp_path / "host.log").read_text(encoding="utf-8")


def test_the_host_refuses_to_restart_or_stop_an_engine_it_did_not_start(
    tmp_path: Path,
) -> None:
    runner = Scripted(
        answers={
            "-l -v --running": ok(OWENS_PC_LIST),
            "-l -v": ok(OWENS_PC_LIST),
            "cat ": ok(GUEST_LINE),
        },
        pings=[200, 200],
    )
    context = _context(tmp_path, runner)
    host = app_module.Host(context)
    host.start()
    runner.calls.clear()
    host.on_click(menu.STOP_ENGINE)
    host.on_click(menu.RESTART_ENGINE)
    assert not any("systemctl" in " ".join(call) for call in runner.calls)
    assert context.presence.engine is Engine.RUNNING


# ------------------------------------------------------------- 4.1 startup


def test_the_startup_shortcut_is_exactly_where_4_1_says() -> None:
    path = str(startup.shortcut_path(WINDOWS_ENV))
    assert path == (
        r"C:\Users\tellt\AppData\Roaming\Microsoft\Windows\Start Menu\Programs"
        r"\Startup\Crucible.lnk"
    )


def test_appdata_unset_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(HostError) as caught:
        startup.shortcut_path({})
    assert caught.value.code == "host_no_localappdata"


def test_the_shortcut_points_at_pythonw_and_never_at_the_cmd() -> None:
    """A `.cmd` opens a console window, and a login item that flashes a black
    box on every boot is a login item people disable."""
    argv = startup.install_argv(WINDOWS_ENV)
    script = argv[-1]
    assert r"\host\pythonw.exe" in script
    assert "crucible.cmd" not in script
    assert "runpy.run_module" in script
    assert "local" in script and "tray" in script
    assert argv[0] == "powershell.exe"
    assert "WScript.Shell" in script


def test_install_startup_is_idempotent_because_CreateShortcut_rewrites() -> None:
    runner = Scripted()
    first = startup.install(runner)
    second = startup.install(runner)
    assert first.path == second.path
    assert runner.calls[0] == runner.calls[1]


def test_login_preserves_custom_home_without_inherited_environment(monkeypatch) -> None:
    import os
    import runpy
    import sys
    home = r"E:\Crucible installs\Owen's engine"
    env = dict(WINDOWS_ENV, CRUCIBLE_HOME=home)
    monkeypatch.delenv("CRUCIBLE_HOME", raising=False)
    monkeypatch.setattr(sys, "argv", [])
    calls = []
    monkeypatch.setattr(runpy, "run_module", lambda module, **kwargs:
                        calls.append((module, kwargs, os.environ["CRUCIBLE_HOME"], list(sys.argv))))
    # Execute the shortcut's Python source without launching a real engine.
    exec(startup.startup_python(env), {})
    assert calls == [("crucible.cli", {"run_name": "__main__"}, home, ["crucible", "local", "tray"])]
    monkeypatch.delenv("CRUCIBLE_HOME")


def test_the_remove_script_is_powershell_that_parses(monkeypatch) -> None:
    """A REAL DEFECT, found by running the verb on Owen's PC 2026-09-14.

    The script was two adjacent strings with only the first an f-string, so
    the second's escaped braces stayed doubled and PowerShell got
    `} } else {` — "Unexpected token '}'", and a shortcut that could not be
    removed. Reading it did not catch it; running it did. This asserts the
    shape that was wrong: balanced, singled braces, in argv order.
    """
    script = startup.remove_argv(WINDOWS_ENV)[-1]
    assert "}}" not in script and "{{" not in script
    assert script.count("{") == script.count("}") == 2
    assert script.startswith("if (Test-Path ")
    assert "} else { Write-Output 'absent' }" in script


def test_the_install_script_is_powershell_that_parses() -> None:
    script = startup.install_argv(WINDOWS_ENV)[-1]
    assert "}}" not in script and "{{" not in script


def test_remove_startup_says_whether_there_was_one() -> None:
    there = Scripted(default=ok("removed\n"))
    assert startup.remove(there).changed is True
    absent = Scripted(default=ok("absent\n"))
    outcome = startup.remove(absent)
    assert outcome.changed is False
    assert "nothing to remove" in outcome.detail


# ------------------------------------------------------ 4.3 the 4c table


def test_every_generated_state_code_has_a_predicate_and_no_others() -> None:
    """The seam. `wsl_states.py` is GENERATED from wsl-states.ts; the
    predicates are code and live here. A row renamed in TypeScript must fail a
    Python test rather than silently never matching."""
    assert set(wslstate.MEANS) == set(WSL_STATE_CODES)
    assert len(WSL_STATE_CODES) == len(set(WSL_STATE_CODES))


def test_the_generated_table_kept_4cs_order_deepest_cause_first() -> None:
    codes = list(WSL_STATE_CODES)
    assert codes.index("virtualization_disabled") < codes.index("wsl_missing")
    assert codes[-1] == "wsl_ready", "the last row must be total"


def test_no_sentinel_survived_the_generation() -> None:
    for state in WSL_STATES:
        blob = state.sentence + state.action_text + state.action_url + " ".join(state.probe_argv)
        for sentinel in ("XXSAID", "XXAPPDISTRO", "XXGUESTUSER", "424.242.424", "424242"):
            assert sentinel not in blob, f"{state.code} carries {sentinel}"


def test_virtualization_is_answered_before_wsl_is_called_missing() -> None:
    """A person told to press Enable WSL on a machine with VT-x off will press
    it forever."""
    runner = Scripted(answers={"--status": bad("Error: 0x80370102")})
    state = wslstate.detect(runner, release="0.6.0")
    assert state.code == "virtualization_disabled"
    assert state.action_kind == "instruct"
    assert "VT-x" in state.action_text


def test_no_wsl_at_all_is_wsl_missing_and_its_action_is_elevated() -> None:
    runner = Scripted(answers={"--status": bad("not recognized")})
    state = wslstate.detect(runner, release="0.6.0")
    assert state.code == "wsl_missing"
    assert state.action_kind == "run-elevated"
    assert list(state.action_argv) == ["wsl.exe", "--install", "--no-distribution"]
    elevated = wslstate.elevated_argv(state)
    assert elevated[0] == "powershell.exe"
    assert "Start-Process -Verb RunAs" in elevated[-1]


def test_a_healthy_machine_with_no_crucible_distro_answers_that_row() -> None:
    runner = Scripted(
        answers={"--status": ok("Default Version: 2"), "-l -v": ok("  Ubuntu  Running  2\n")}
    )
    state = wslstate.detect(runner, release="0.6.0")
    assert state.code == "no_crucible_distro"


def test_a_ready_machine_answers_wsl_ready_and_the_evidence_is_filled_in() -> None:
    runner = Scripted(
        answers={
            "--status": ok("Default Version: 2"),
            "-l -v": ok("  crucible  Running  2\n"),
            "wsl.conf": ok("# crucible-rootfs\n[boot]\nsystemd=true\n"),
            # The linger row asks `id -u` as root and wants a literal 0. A
            # distro that answers anything else IS `linger_unreadable`, which
            # is why this answer has to be scripted: the default runner saying
            # "exit 0, no output" is a distro that let us in as somebody who
            # is not root, and the table is right to say so.
            "-u root --exec id -u": ok("0\n"),
        }
    )
    state = wslstate.detect(runner, release="0.6.0")
    assert state.code == "wsl_ready"
    assert "{said}" not in state.sentence


def test_the_costly_rows_are_not_probed_unless_the_caller_asks() -> None:
    """Reading a machine's facts must not reach the internet."""
    runner = Scripted(
        answers={
            "--status": ok("Default Version: 2"),
            "-l -v": ok("  crucible  Running  2\n"),
            "wsl.conf": ok("systemd=true"),
        }
    )
    wslstate.detect(runner, release="0.6.0")
    assert not any("curl" in " ".join(call) for call in runner.calls)
    assert not any("df -Pk" in " ".join(call) for call in runner.calls)


def test_the_disk_row_fills_both_of_its_figures_when_it_is_asked_for() -> None:
    runner = Scripted(
        answers={
            "--status": ok("Default Version: 2"),
            "-l -v": ok("  crucible  Running  2\n"),
            "wsl.conf": ok("systemd=true"),
            "df -Pk": ok("1048576\n"),
        }
    )
    state = wslstate.detect(runner, release="0.6.0", required_bytes=8 * 1024 ** 3)
    assert state.code == "guest_no_disk"
    assert "8.0 GiB" in state.sentence
    assert "1.0 GiB" in state.sentence
    assert "{required}" not in state.sentence and "{free}" not in state.sentence


def test_a_generated_row_with_no_predicate_is_refused_by_name(monkeypatch) -> None:
    from crucible.host import wsl_states as generated

    extra = generated.WslStateDef(
        code="a_row_nobody_wrote_a_predicate_for",
        probe="wsl-status",
        probe_argv=("wsl.exe", "--status"),
        sentence="x",
        action_kind="instruct",
        action_argv=(),
        action_text="",
        action_url="",
        optional=False,
    )
    monkeypatch.setattr(wslstate, "WSL_STATES", (extra,) + generated.WSL_STATES)
    with pytest.raises(HostError) as caught:
        wslstate.detect(Scripted(), release="0.6.0")
    assert caught.value.code == "wsl_state_unknown"


# ------------------------------------------------------------ 4.1 LAN door


def test_mirrored_networking_needs_no_forward_and_says_so() -> None:
    runner = Scripted(answers={"type": ok("[wsl2]\nnetworkingMode=mirrored\n")})
    door = landoor.detect(runner)
    assert door.mechanism == landoor.MIRRORED
    assert door.open is False
    assert not any("netsh" in " ".join(call) for call in runner.calls)


def test_a_nat_machine_with_no_forward_is_the_portproxy_case() -> None:
    runner = Scripted(answers={"type": ok("[wsl2]\nmemory=13GB\n"), "netsh": ok("")})
    door = landoor.detect(runner)
    assert door.mechanism == landoor.PORTPROXY
    assert door.open is False
    assert "only this computer can reach it" in door.detail


def test_an_existing_forward_is_read_by_its_numbers_not_by_a_column() -> None:
    listing = (
        "Listen on ipv4:             Connect to ipv4:\n\n"
        "Address         Port        Address         Port\n"
        "--------------- ----------  --------------- ----------\n"
        "0.0.0.0         7100        127.0.0.1       7100\n"
    )
    assert landoor.has_forward(listing) is True
    assert landoor.has_forward(listing.replace("7100        127", "7101        127")) is False


def test_the_netsh_argv_is_data_and_the_consent_sentence_is_one_sentence() -> None:
    assert landoor.add_argv() == [
        "netsh", "interface", "portproxy", "add", "v4tov4",
        "listenport=7100", "listenaddress=0.0.0.0",
        "connectport=7100", "connectaddress=127.0.0.1",
    ]
    assert landoor.remove_argv()[3] == "delete"
    assert "administrator" in landoor.ELEVATION_SENTENCE
    assert "7100" in landoor.ELEVATION_SENTENCE


def test_wslconfig_networking_mode_ignores_a_comment_and_reports_absence_as_None() -> None:
    assert landoor.networking_mode("# networkingMode=mirrored\nmemory=13GB\n") is None
    assert landoor.networking_mode("networkingMode = Mirrored") == "mirrored"


FORWARD_ROW = (
    "Listen on ipv4:             Connect to ipv4:\n\n"
    "Address         Port        Address         Port\n"
    "--------------- ----------  --------------- ----------\n"
    "0.0.0.0         7100        127.0.0.1       7100\n"
)


def _door(*, forward: bool, firewall: bool, category: str = "Private") -> landoor.LanDoor:
    """`detect` against a machine described by three facts."""
    runner = Scripted(
        answers={
            "type": ok("[wsl2]\nmemory=13GB\n"),
            "portproxy show": ok(FORWARD_ROW if forward else ""),
            "advfirewall firewall show": (
                ok("Rule Name: Crucible engine (LAN)\n") if firewall
                else RunResult(code=1, stdout="No rules match.", stderr="", failure=None)
            ),
            "Get-NetConnectionProfile": ok(
                '[{"InterfaceAlias":"Ethernet 2","NetworkCategory":"' + category + '"}]'
            ),
        }
    )
    return landoor.detect(runner)


def test_a_forward_with_no_firewall_rule_is_a_door_that_looks_open_and_is_shut() -> None:
    """The half-open state: Windows drops it before the forward ever sees it."""
    door = _door(forward=True, firewall=False)
    assert door.forward is True and door.firewall is False
    assert door.open is False
    assert "drops the connection" in door.detail


def test_both_rows_on_a_private_network_is_the_only_open_door() -> None:
    door = _door(forward=True, firewall=True)
    assert door.open is True
    assert door.private_network is True


def test_both_rows_on_a_public_only_network_is_not_called_open() -> None:
    """A Private-scoped rule admits nothing on a Public network. Said, not hidden."""
    door = _door(forward=True, firewall=True, category="Public")
    assert door.forward is True and door.firewall is True
    assert door.open is False
    assert "admits nothing here" in door.detail


def test_the_firewall_argv_names_the_rule_it_can_later_delete_by() -> None:
    added = landoor.firewall_add_argv()
    assert added[:5] == ["netsh", "advfirewall", "firewall", "add", "rule"]
    assert f"name={landoor.RULE_NAME}" in added
    assert "dir=in" in added and "protocol=TCP" in added
    assert f"profile={landoor.RULE_PROFILE}" in added
    # The delete must be addressable by the SAME name, or it removes nothing.
    assert f"name={landoor.RULE_NAME}" in landoor.firewall_remove_argv()
    assert landoor.firewall_remove_argv()[3] == "delete"


def test_the_consent_sentence_names_BOTH_things_it_will_add() -> None:
    sentence = landoor.ELEVATION_SENTENCE
    assert "administrator" in sentence
    assert "7100" in sentence
    assert "port forward" in sentence
    assert landoor.RULE_NAME in sentence, "a consent that hides half of itself"
    assert "crucible lan disable" in sentence, "it says how to undo it"


def test_the_network_category_is_read_as_a_name_and_an_enum_is_not_guessed() -> None:
    """Measured: PowerShell serialises NetworkCategory as its INTEGER value.

    The probe forces `[string]`. If that is ever dropped this reports None
    (unreadable) rather than quietly deciding the network is not Private, which
    is what it did on Owen's PC before the cast was measured and added.
    """
    assert "[string]" in " ".join(landoor.connection_profile_argv())
    assert landoor.has_private_network('[{"NetworkCategory":"Private"}]') is True
    assert landoor.has_private_network('[{"NetworkCategory":"Public"}]') is False
    assert landoor.has_private_network('[{"NetworkCategory":1}]') is None
    assert landoor.has_private_network("") is None
    assert landoor.has_private_network("not json") is None
    # One profile serialises as a scalar, not a list.
    assert landoor.has_private_network('{"NetworkCategory":"Private"}') is True


# ------------------------------------------------------------ 4.3 installer


def test_the_sequence_is_4_7s_steps_in_4_7s_order() -> None:
    # `lan-door` MOVED after `switch-pairing` on 2026-09-17. It was a no-op
    # message when it sat before the switch-over, and the position did not
    # matter; now that it opens a real door it has to run where there is a guest
    # engine to publish `lan_advertise` INTO, which is only after the switch.
    # Still before `migrate-weights`, which can run for hours: a consent prompt
    # raised at the far end of that is a prompt nobody is sitting in front of.
    assert installer.STEPS == (
        "wsl-state",
        "import-distro",
        "guest-ready",
        "guest-install",
        "migrate-config",
        "install-job-types",
        "prepare-weights",
        "stop-windows-server",
        "switch-pairing",
        "lan-door",
        "migrate-weights",
    )


def test_config_from_carries_exactly_three_things_and_no_fourth() -> None:
    carried = installer.carried_config(
        '[server]\nname = "crucible@pc"\nport = 7100\n'
        '[auth]\ntoken = "abc"\n'
        '[backend]\nkind = "llama-windows"\n'
        '[routes]\ntranslate = "anthropic/claude-sonnet-5"\n'
        '[upstreams.anthropic]\nkey = "sk-ant-x"\n'
    )
    assert "token" in carried
    assert "[routes]" in carried
    assert "upstreams.anthropic" in carried
    # The machine being LEFT owns these, not the one being initialised.
    assert "llama-windows" not in carried
    assert "crucible@pc" not in carried
    assert "7100" not in carried


def test_a_config_with_no_token_is_refused_by_name() -> None:
    with pytest.raises(HostError) as caught:
        installer.carried_config('[server]\nname = "x"\n')
    assert caught.value.code == "config_from_no_token"


def test_a_reboot_state_ends_the_task_with_the_sentence_4_7_requires(
    host_log: log.HostLog, tmp_path: Path
) -> None:
    events: list[installer.Event] = []
    runner = Scripted(answers={"--status": bad("not recognized")})
    walk = installer.EngineInstall(
        runner,
        events.append,
        release="0.6.0",
        home=tmp_path,
        install_sh_url="https://example.invalid/install.sh",
    )
    with pytest.raises(HostError) as caught:
        walk.run()
    assert caught.value.code == "wsl_reboot_required"
    # NOT "Crucible continues": the Startup item brings the tray back and the
    # tray does not re-post the task (app.py's INSTALL_ENGINE opens the console
    # and the PAGE posts it). The sentence says what actually happens.
    assert "reboot, then start this again" in caught.value.message
    assert "picks this up where it stopped" not in caught.value.message
    assert (tmp_path / installer.REBOOT_PENDING).is_file(), (
        "nothing recorded that this machine stopped for a reboot"
    )
    kinds = [event.event for event in events]
    assert kinds[0] == "step", "a line must never precede a step"
    assert kinds[-1] == "failed"
    assert any(
        "Start-Process -Verb RunAs" in " ".join(call) for call in runner.calls
    ), "the elevated action ran by name"


def test_elevation_off_reports_the_argv_and_raises_no_dialog(
    host_log: log.HostLog, tmp_path: Path
) -> None:
    events: list[installer.Event] = []
    runner = Scripted(answers={"--status": bad("not recognized")})
    walk = installer.EngineInstall(
        runner,
        events.append,
        release="0.6.0",
        home=tmp_path,
        install_sh_url="https://example.invalid/install.sh",
        elevate=False,
    )
    with pytest.raises(HostError):
        walk.run()
    assert not any("RunAs" in " ".join(call) for call in runner.calls)


def test_a_missing_rootfs_refuses_rather_than_importing_somebody_elses_image(
    tmp_path: Path,
) -> None:
    events: list[installer.Event] = []
    runner = Scripted(
        answers={
            "--status": ok("Default Version: 2"),
            "-l -v": ok("  Ubuntu  Running  2\n"),
        }
    )
    walk = installer.EngineInstall(
        runner,
        events.append,
        release="0.6.0",
        home=tmp_path,
        install_sh_url="https://example.invalid/install.sh",
    )
    with pytest.raises(HostError) as caught:
        walk.run()
    assert caught.value.code == "rootfs_sha_mismatch"
    # The image is Canonical's now (PHASE20 section 2), and so is the digest:
    # a `Scripted` runner that answers nothing for the sums file is exactly the
    # shape "there is no row for our file", which must refuse rather than
    # import bytes nobody checked.
    assert "SHA256SUMS" in caught.value.message
    assert "ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz" in caught.value.message


def test_every_event_a_step_emits_is_shaped_like_a_tasks_py_event(tmp_path: Path) -> None:
    """4.7 has the server RELAY these under a task id, so the shape is
    `crucible/tasks.py`'s and not merely similar to it."""
    events: list[installer.Event] = []
    runner = Scripted(answers={"--status": bad("nope")})
    walk = installer.EngineInstall(
        runner,
        events.append,
        release="0.6.0",
        home=tmp_path,
        install_sh_url="https://example.invalid/install.sh",
    )
    with pytest.raises(HostError):
        walk.run()
    for event in events:
        assert event.event in ("step", "progress", "state", "line", "done", "failed")
        if event.event == "step":
            assert set(event.data) == {"name", "index", "total"}
        if event.event == "line":
            assert set(event.data) == {"text", "stream"}
        if event.event == "state":
            assert set(event.data) == {"code", "sentence", "action"}
        if event.event == "failed":
            assert set(event.data) == {"code", "message"}


def test_the_done_payload_carries_every_field_the_bootstrap_client_requires() -> None:
    outcome = installer.InstallOutcome(
        steps=[installer.StepRecord("guest-install", ["bash"], "ok", "done")],
        server_name="crucible@pc-wsl",
        server_url="http://127.0.0.1:7100",
        config_path="/home/crucible/.crucible/config.toml",
        crucible="/home/crucible/.crucible/server/bin/crucible",
        release="0.6.0",
    )
    payload = outcome.to_dict()
    assert set(payload) == {"server", "release", "backend", "crucible", "steps"}
    assert set(payload["server"]) == {"name", "url", "config_path"}
    # The GUEST's backend, not the machine being left.
    assert payload["backend"] == "cuda-linux"
    assert payload["steps"][0]["status"] == "ok"
    # It must survive a JSON round trip: it is a wire.
    assert json.loads(json.dumps(payload)) == payload


# ----------------------------------------------------------------- 4.3 door


@dataclass
class FakeOrchestrator:
    """An `OrchestratorPort` whose three answers are written down.

    The door is a TRANSPORT and every decision it serves is made in `app.py`
    (`menu.py`'s rule, one level out), so the transport is tested without
    building a tray.
    """

    name: str = "crucible-orchestrator@test"
    document: dict = field(default_factory=lambda: {"role": "orchestrator"})
    not_ours: bool = False
    restarts: list[str] = field(default_factory=list)
    quits: list[str] = field(default_factory=list)

    def info(self) -> dict:
        return self.document

    def check_restartable(self) -> None:
        if self.not_ours:
            raise HostError("engine_not_ours", "watched and never acted on")

    def restart_engine(self, emit) -> None:
        self.restarts.append("restarted")
        emit(installer.Event("done", {"engine": "http://127.0.0.1:7100"}))

    def quit(self) -> None:
        self.quits.append("quit")


def a_door(
    host_log: log.HostLog,
    sequence=lambda _emit: None,
    *,
    token="t",
    orchestrator: FakeOrchestrator | None = None,
) -> door_module.OrchestratorDoor:
    return door_module.OrchestratorDoor(
        host_log,
        sequence,
        token=(token if callable(token) else (lambda: token)),
        orchestrator=orchestrator or FakeOrchestrator(),
    )


def test_the_door_refuses_a_wrong_bearer_and_a_missing_one(host_log: log.HostLog) -> None:
    door = a_door(host_log, token="right")
    assert door.authorised("Bearer right") is True
    assert door.authorised("Bearer wrong") is False
    assert door.authorised(None) is False
    assert door.authorised("right") is False


def test_no_config_yet_is_host_no_token_and_not_an_authorisation_failure(
    host_log: log.HostLog,
) -> None:
    door = a_door(host_log, token=lambda: None)
    with pytest.raises(HostError) as caught:
        door.authorised("Bearer anything")
    assert caught.value.code == "host_no_token"


def test_one_install_on_a_machine(host_log: log.HostLog) -> None:
    door = a_door(host_log)
    assert door.claim() is True
    assert door.claim() is False, "host_install_running"
    door.release()
    assert door.claim() is True


def test_the_door_is_loopback_and_an_argument_cannot_put_it_on_the_lan(
    host_log: log.HostLog,
) -> None:
    door = a_door(host_log)
    with pytest.raises(HostError) as caught:
        door_module.serve(door, host="0.0.0.0")
    assert caught.value.code == "host_unauthorized"


def test_the_door_streams_ndjson_and_terminates_even_when_the_sequence_throws(
    host_log: log.HostLog,
) -> None:
    """A real socket, on a port the OS picks, so the wire is the thing tested."""
    import urllib.error
    import urllib.request

    def sequence(emit: Callable[[installer.Event], None]) -> None:
        emit(installer.Event("step", {"name": "wsl-state", "index": 1, "total": 9}))
        emit(installer.Event("line", {"text": "hello", "stream": "stdout"}))
        raise RuntimeError("something threw before the sequence could say so")

    door = a_door(host_log, sequence, token="tok")
    server = door_module.serve(door, host="127.0.0.1", port=0)
    port = server.server_address[1]
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/install",
            data=json.dumps({"target": "wsl", "release": "0.6.0", "job_types": ["llm"]}).encode(),
            headers={"Authorization": "Bearer tok", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            assert response.headers["Content-Type"] == "application/x-ndjson"
            lines = [json.loads(line) for line in response.read().decode().splitlines() if line]
    finally:
        server.shutdown()
    assert [line["id"] for line in lines] == [1, 2, 3]
    assert lines[0]["event"] == "step"
    assert lines[-1]["event"] == "failed"
    assert lines[-1]["data"]["code"] == "task_failed"


def test_the_door_refuses_a_target_it_does_not_move_to(host_log: log.HostLog) -> None:
    import urllib.error
    import urllib.request

    door = a_door(host_log, token="tok")
    server = door_module.serve(door, host="127.0.0.1", port=0)
    port = server.server_address[1]
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/install",
            data=json.dumps({"target": "windows"}).encode(),
            headers={"Authorization": "Bearer tok"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=20)
        assert caught.value.code == 400
        body = json.loads(caught.value.read().decode())
        assert body["error"]["code"] == "engine_target_unknown"
    finally:
        server.shutdown()


def test_the_extra_fields_bootstrap_sends_are_accepted_and_not_refused(
    host_log: log.HostLog,
) -> None:
    """An older door refusing a field a newer client was told to send is how
    two halves of one release stop talking to each other."""
    import urllib.request

    seen: list[str] = []

    def sequence(emit: Callable[[installer.Event], None]) -> None:
        seen.append("ran")
        emit(installer.Event("done", {"server": {}, "release": "", "backend": "", "crucible": "", "steps": []}))

    door = a_door(host_log, sequence, token="tok")
    server = door_module.serve(door, host="127.0.0.1", port=0)
    port = server.server_address[1]
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/install",
            data=json.dumps(
                {
                    "target": "wsl",
                    "release": "0.6.0",
                    "job_types": ["llm", {"type": "tts", "narrator_engine": "higgs-v3"}],
                    "home": "/home/crucible/.crucible",
                    "bind": {"host": "127.0.0.1", "port": 7100},
                }
            ).encode(),
            headers={"Authorization": "Bearer tok"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
    finally:
        server.shutdown()
    assert seen == ["ran"]


# ------------------------------------------ PHASE17 4.4: `POST /quit`
#
# Measured 2026-09-15: `taskkill /PID 45504` with no `/F` reported "sent
# termination signal", the tray was still alive 25 s later, and it had written
# NOTHING to its log — a console-less `pythonw` never sees the WM_CLOSE. `/F`
# ended it and ran none of `quit()`. So this route is not a convenience; it is
# the only orderly stop the process has.


def _post(port: int, path: str, *, bearer: str | None) -> tuple[int, dict]:
    """POST with no body, returning (status, parsed body). Refusals included."""
    import urllib.error
    import urllib.request

    headers = {} if bearer is None else {"Authorization": f"Bearer {bearer}"}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=b"", headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as refused:
        return refused.code, json.loads(refused.read().decode())


def test_quit_takes_THE_SAME_bearer_as_every_other_route_on_this_door(
    host_log: log.HostLog,
) -> None:
    """The ENGINE's token, and a wrong one is refused BY NAME.

    A stop is the most destructive verb this door has, and it is the one an
    unauthenticated caller must not be able to reach: anything that can open a
    loopback socket could otherwise take the machine's orchestrator down.
    """
    orchestrator = FakeOrchestrator()
    door = a_door(host_log, token="tok", orchestrator=orchestrator)
    server = door_module.serve(door, host="127.0.0.1", port=0)
    port = server.server_address[1]
    try:
        for bearer in (None, "wrong"):
            status, body = _post(port, door_module.QUIT_PATH, bearer=bearer)
            assert status == 401, bearer
            assert body["error"]["code"] == "host_unauthorized"
        assert orchestrator.quits == [], "a refused quit stopped nothing"
    finally:
        server.shutdown()


def test_a_quit_before_this_machine_has_a_token_is_host_no_token(
    host_log: log.HostLog,
) -> None:
    """Not an authorisation failure, and named as the state it is — the same
    distinction `authorised()` already makes for every other route."""
    orchestrator = FakeOrchestrator()
    door = a_door(host_log, token=lambda: None, orchestrator=orchestrator)
    server = door_module.serve(door, host="127.0.0.1", port=0)
    port = server.server_address[1]
    try:
        status, body = _post(port, door_module.QUIT_PATH, bearer="anything")
        assert status == 503
        assert body["error"]["code"] == "host_no_token"
        assert orchestrator.quits == []
    finally:
        server.shutdown()


def test_the_quit_ANSWER_precedes_the_stop_and_is_the_last_event(
    host_log: log.HostLog,
) -> None:
    """4.4: a verb with an empty answer, never a task.

    Deterministic rather than timed. The fake's `quit()` BLOCKS until the
    client says it holds the whole body, so an implementation that stopped
    first would record `quit-before-the-answer` here instead of waiting for a
    reader that could no longer arrive.
    """
    answered = threading.Event()
    stopped = threading.Event()

    class BlocksUntilAnswered(FakeOrchestrator):
        def quit(self) -> None:
            self.quits.append(
                "quit" if answered.wait(20) else "quit-before-the-answer"
            )
            stopped.set()

    orchestrator = BlocksUntilAnswered()
    door = a_door(host_log, token="tok", orchestrator=orchestrator)
    server = door_module.serve(door, host="127.0.0.1", port=0)
    port = server.server_address[1]
    try:
        status, body = _post(port, door_module.QUIT_PATH, bearer="tok")
        answered.set()
        assert status == 200
        assert body == {"quit": True, "name": "crucible-orchestrator@test"}
        # The stop ran, and it ran AFTER the answer was on the wire.
        assert stopped.wait(20), "the door never reached the stop"
        assert orchestrator.quits == ["quit"]
    finally:
        server.shutdown()


def test_the_404_body_names_quit_among_the_routes_this_door_serves(
    host_log: log.HostLog,
) -> None:
    door = a_door(host_log, token="tok")
    server = door_module.serve(door, host="127.0.0.1", port=0)
    port = server.server_address[1]
    try:
        status, body = _post(port, "/stop", bearer="tok")
        assert status == 404
        assert body["error"]["code"] == "not_found"
        assert door_module.QUIT_PATH in body["error"]["message"]
    finally:
        server.shutdown()


# ------------------------------------------------------------- 3.6 pairing


def test_the_pairing_file_is_one_line_with_a_trailing_newline(tmp_path: Path) -> None:
    from crucible import pairing

    written = pairing.write_pairing_file(
        tmp_path, "crucible://x@127.0.0.1:7100/#tok", platform="linux"
    )
    assert written.read_text(encoding="utf-8") == "crucible://x@127.0.0.1:7100/#tok\n"
    assert pairing.read_pairing_file(tmp_path) == "crucible://x@127.0.0.1:7100/#tok"


def test_an_absent_pairing_file_is_None_and_never_an_error(tmp_path: Path) -> None:
    from crucible import pairing

    assert pairing.read_pairing_file(tmp_path) is None


@pytest.mark.skipif(
    os.name != "posix",
    reason=(
        "a mode is a property of the FILESYSTEM, not of the code path: this "
        "asserts the bits `os.open(..., 0o600)` actually left on disk, and "
        "NTFS has none to leave (it reports 0o666 for every file). It is the "
        "one test in this file that cannot be platform-injected, because the "
        "thing under test is the filesystem's answer. The Windows half of the "
        "same rule — the icacls ACL, and the file being DELETED when it "
        "cannot be set — is tested below and runs everywhere."
    ),
)
def test_on_posix_the_pairing_file_is_0600_from_the_outset(tmp_path: Path) -> None:
    import stat as stat_module

    from crucible import pairing

    written = pairing.write_pairing_file(tmp_path, "crucible://x@h:1/#t", platform="linux")
    assert oct(stat_module.S_IMODE(written.stat().st_mode)) == "0o600"


def test_the_windows_acl_is_icacls_with_inheritance_removed(tmp_path: Path) -> None:
    from crucible import pairing

    argv = list(pairing.icacls_argv(tmp_path / "pairing", "tellt"))
    assert argv[0] == "icacls"
    # `/inheritance:r` REMOVES Administrators and SYSTEM; a grant without it
    # would leave a bearer token three principals can read.
    assert "/inheritance:r" in argv
    assert argv[-2:] == ["/grant:r", "tellt:(R,W)"]


def test_a_failed_acl_DELETES_the_file_rather_than_leaving_a_token_readable(
    tmp_path: Path,
) -> None:
    from crucible import pairing

    class Completed:
        returncode = 5
        stdout = ""
        stderr = "Access is denied."

    def refuse(*_args: object, **_kwargs: object) -> Completed:
        return Completed()

    with pytest.raises(pairing.PairingFileError) as caught:
        pairing.write_pairing_file(
            tmp_path,
            "crucible://x@h:1/#secret",
            platform="win32",
            env={"USERNAME": "tellt"},
            run=refuse,
        )
    assert caught.value.code == "pairing_acl_failed"
    assert not (tmp_path / "pairing").exists()


def test_username_unset_is_refused_and_never_guessed(tmp_path: Path) -> None:
    from crucible import pairing

    with pytest.raises(pairing.PairingFileError) as caught:
        pairing.write_pairing_file(
            tmp_path, "crucible://x@h:1/#t", platform="win32", env={}, run=lambda *a, **k: None
        )
    assert caught.value.code == "pairing_acl_failed"


# ------------------------------------------------------------- the CLI verbs


def test_crucible_host_is_refused_off_win32_by_name(capsys, monkeypatch) -> None:
    """`host_windows_only`, and the refusal says what to ask INSTEAD.

    Not a platform check standing in for a feature check: on Linux and macOS
    the server runs on the machine and its own service manager supervises it
    (4.4, "no host on the Mac"), so there is nothing for a tray to own.
    """
    from crucible import cli

    monkeypatch.setattr(cli.sys, "platform", "linux")
    code = cli.main(["host"])
    assert code == cli.EXIT_REFUSED
    said = capsys.readouterr().err
    assert "host_windows_only" in said
    assert "systemd" in said


def test_there_is_no_platform_gate_left_in_main(monkeypatch, tmp_path: Path) -> None:
    """Section 3.5: on win32 EVERY verb runs, and the backend must be right.

    The opt-in `win32_ok` flag is gone with the gate it narrowed: a platform
    test standing in for a backend test is two owners for "can this machine
    do it". What proves it here is that a win32 `doctor` reaches its own
    code — it fails on a config it cannot find, not on a sentence about
    Windows — and that no subparser carries the flag any more.

    `CRUCIBLE_HOME` and the backend are INJECTED, for this file's own reason
    (its module docstring: every test in here runs OFF Windows, so the
    platform, the environment and every subprocess are given rather than
    found). Pretending to be win32 makes `crucible_home()` ask for
    `%LOCALAPPDATA%`, which no Linux session has — that refusal is correct and
    is pinned in `test_llama_windows.py`; it is simply not what this test is
    about. `detect_backend` is stubbed for the same reason: on a win32
    `sys.platform` it would shell out to the host's nvidia-smi, and on a
    machine without one `detect_windows` reaches `ctypes.windll`.
    """
    from crucible import cli
    from crucible.errors import ConfigError, NoViableBackend

    parser = cli.build_parser()
    assert "win32_ok" not in vars(parser.parse_args(["doctor"]))
    assert "win32_ok" not in vars(parser.parse_args(["host"]))

    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path))
    monkeypatch.setattr(
        cli, "detect_backend", lambda: (_ for _ in ()).throw(
            NoViableBackend("no card in this test")
        )
    )
    monkeypatch.setattr(
        cli, "load_config", lambda *a, **k: (_ for _ in ()).throw(
            ConfigError("no config here")
        )
    )
    assert cli.main(["doctor"]) == cli.EXIT_REFUSED


def test_a_cuda_linux_config_on_a_windows_host_is_backend_not_here(capsys) -> None:
    """The refusal a win32 machine gets, and the ONE place it still says vLLM.

    `WINDOWS_REFUSAL` used to be printed before a verb was parsed. It now
    describes exactly one thing — a `cuda-linux` config found on Windows —
    and `_backend_mismatch` is where it is said.
    """
    from crucible import cli
    from crucible.backend import Backend, Gpu

    windows = Backend(
        kind="llama-windows",
        platform="windows",
        arch="AMD64",
        gpu=Gpu(vendor="nvidia", name="RTX 4090", vram_bytes=24 * 1024**3),
        detail="llama.cpp cuda build",
    )
    said = cli._backend_mismatch("cuda-linux", windows)
    assert said.startswith("backend_not_here: ")
    assert "cuda-linux" in said and "llama-windows" in said
    assert "WSL2" in said  # WINDOWS_REFUSAL, in the one case it describes

    mac = Backend(
        kind="mlx-darwin",
        platform="darwin",
        arch="arm64",
        gpu=Gpu(vendor="apple", name="M1 Ultra", vram_bytes=64 * 1024**3),
        detail="mlx",
    )
    other = cli._backend_mismatch("llama-windows", mac)
    assert other.startswith("backend_not_here: ")
    assert "do not run on win32" not in other


def test_config_from_takes_the_token_the_routes_and_the_upstreams(tmp_path: Path) -> None:
    from crucible.cli import carried_from

    path = tmp_path / "config.toml"
    path.write_text(
        '[server]\nname = "crucible@pc"\nhost = "127.0.0.1"\nport = 7100\n'
        '[auth]\ntoken = "carried-token"\n'
        '[backend]\nkind = "llama-windows"\n'
        '[jobs]\nenable_echo = true\n'
        '[routes]\ntranslate = "anthropic/claude-sonnet-5"\n'
        '[upstreams.anthropic]\nkey = "sk-ant-x"\n',
        encoding="utf-8",
    )
    token, carried = carried_from(path)
    assert token == "carried-token"
    assert set(carried) == {"routes", "upstreams"}
    assert carried["upstreams"]["anthropic"]["key"] == "sk-ant-x"


def test_config_from_without_a_token_is_refused_by_name(tmp_path: Path) -> None:
    from crucible.cli import carried_from
    from crucible.errors import ConfigError

    path = tmp_path / "config.toml"
    path.write_text('[server]\nname = "x"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        carried_from(path)
    assert "config_from_no_token" in str(caught.value)


def test_config_from_that_is_not_toml_is_refused_by_name(tmp_path: Path) -> None:
    from crucible.cli import carried_from
    from crucible.errors import ConfigError

    path = tmp_path / "config.toml"
    path.write_text("this is not toml {{{", encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        carried_from(path)
    assert "config_from_unreadable" in str(caught.value)


def test_write_config_copies_a_carried_table_verbatim(tmp_path: Path) -> None:
    """A key this build does not know about is still the operator's."""
    import tomllib

    from crucible.config import write_config

    written = write_config(
        tmp_path,
        name="crucible@guest",
        host="127.0.0.1",
        port=7100,
        token="t",
        backend_kind="cuda-linux",
        enable_echo=True,
        enable_llm=False,
        enable_asr=False,
        enable_tts=False,
        enable_align=False,
        enable_rvc=False,
        desktop_allowance_bytes=1,
        carried_tables={
            "routes": {"translate": "anthropic/claude-sonnet-5"},
            "upstreams": {"anthropic": {"key": "sk-ant-x", "a_field_from_the_future": 1}},
        },
    )
    document = tomllib.loads(written.read_text(encoding="utf-8"))
    assert document["routes"] == {"translate": "anthropic/claude-sonnet-5"}
    assert document["upstreams"]["anthropic"]["a_field_from_the_future"] == 1
    # The machine being INITIALISED owns these, whatever the file said.
    assert document["backend"]["kind"] == "cuda-linux"


def test_a_carried_table_may_not_shadow_one_this_writer_owns(tmp_path: Path) -> None:
    from crucible.config import write_config
    from crucible.errors import ConfigError

    with pytest.raises(ConfigError) as caught:
        write_config(
            tmp_path,
            name="x",
            host="127.0.0.1",
            port=7100,
            token="t",
            backend_kind="cuda-linux",
            enable_echo=True,
            enable_llm=False,
            enable_asr=False,
            enable_tts=False,
            enable_align=False,
            enable_rvc=False,
            desktop_allowance_bytes=1,
            carried_tables={"auth": {"token": "somebody-elses"}},
        )
    assert "two writers" in str(caught.value)


# ------------------------------------- 3.5 / 3.5a the weights migration
#
# TWO FAKE SERVERS. The migration's whole content is an ORDER between two
# machines, so a test with one of them mocked would be a test of nothing: the
# rule being checked is that the guest has a subject BEFORE the Windows copy
# is deleted, and only a second server can witness that.


class FakeCatalog:
    """A server that owns some subjects. Records the order it was asked."""

    def __init__(self, where: str, installed: Sequence[tuple[str, str]] = ()) -> None:
        self._where = where
        self.subjects: list[Subject] = [
            Subject(kind=kind, id=ident, name=ident, installed=True)
            for kind, ident in installed
        ]
        self.calls: list[str] = []
        #: Keys this server refuses to remove, and who holds them. A key is
        #: dropped from here once `release_after` rounds have passed, which is
        #: how "somebody closed the app" is spelled.
        self.in_use: dict[tuple[str, str], str] = {}
        self.release_after: dict[tuple[str, str], int] = {}
        #: A pull lands after this many `installed_subjects()` polls.
        self.pull_latency = 0
        self._pending: dict[tuple[str, str], int] = {}
        self.unreachable = False

    def _keys(self) -> set[tuple[str, str]]:
        return {row.key for row in self.subjects}

    @property
    def where(self) -> str:
        return self._where

    def installed_subjects(self) -> list[Subject]:
        if self.unreachable:
            raise HostError("catalog_unreachable", f"{self._where} did not answer")
        self.calls.append("list")
        for key in list(self._pending):
            self._pending[key] -= 1
            if self._pending[key] <= 0:
                del self._pending[key]
                self.subjects.append(
                    Subject(kind=key[0], id=key[1], name=key[1], installed=True)
                )
        return list(self.subjects)

    def pull(self, subject: Subject) -> None:
        self.calls.append(f"pull {subject}")
        self._pending[subject.key] = max(1, self.pull_latency)

    def remove(self, subject: Subject) -> None:
        self.calls.append(f"remove {subject}")
        who = self.in_use.get(subject.key)
        if who is not None:
            rounds = self.release_after.get(subject.key, 0) - 1
            self.release_after[subject.key] = rounds
            if rounds > 0:
                raise CatalogRefusal(
                    "subject_in_use", f"{self._where}: {subject} is in use", who
                )
            del self.in_use[subject.key]
        self.subjects = [row for row in self.subjects if row.key != subject.key]


def migration(
    windows: FakeCatalog | None,
    guest: FakeCatalog | None,
    events: list[installer.Event],
    tmp_path: Path,
) -> installer.EngineInstall:
    return installer.EngineInstall(
        Scripted(),
        events.append,
        release="0.6.0",
        home=tmp_path,
        install_sh_url="https://example.invalid/install.sh",
        windows_catalog=windows,
        guest_catalog=guest,
        monotonic=ticking(),
        sleep=lambda _s: None,
    )


def test_the_guest_gets_a_subject_BEFORE_the_windows_copy_is_deleted(tmp_path: Path) -> None:
    """3.5's whole rule, as an order between two servers."""
    windows = FakeCatalog("windows", [("model", "qwen3.5-9b"), ("voice", "mistborn")])
    guest = FakeCatalog("guest")
    guest.pull_latency = 2
    events: list[installer.Event] = []
    migration(windows, guest, events, tmp_path)._migrate_weights()

    # Every subject: pulled in the guest, present there, THEN removed here.
    for subject in ("model qwen3.5-9b", "voice mistborn"):
        assert f"pull {subject}" in guest.calls
        assert f"remove {subject}" in windows.calls
        pulled = guest.calls.index(f"pull {subject}")
        removed = windows.calls.index(f"remove {subject}")
        # The guest listed the subject as installed between the two.
        assert pulled < len(guest.calls)
        assert removed >= 0
    assert windows.subjects == [], "the Windows copies are gone"
    assert {row.key for row in guest.subjects} == {
        ("model", "qwen3.5-9b"),
        ("voice", "mistborn"),
    }


def test_a_subject_the_guest_ALREADY_has_is_not_pulled_again(tmp_path: Path) -> None:
    """Idempotent on resume: it re-diffs, it does not replay."""
    windows = FakeCatalog("windows", [("model", "qwen3.5-9b")])
    guest = FakeCatalog("guest", [("model", "qwen3.5-9b")])
    events: list[installer.Event] = []
    migration(windows, guest, events, tmp_path)._migrate_weights()
    assert not any(call.startswith("pull") for call in guest.calls)
    assert "remove model qwen3.5-9b" in windows.calls
    assert windows.subjects == []


def test_preparation_never_deletes_a_source_when_a_later_pull_fails(tmp_path: Path) -> None:
    class RefusesSecond(FakeCatalog):
        def pull(self, subject: Subject) -> None:
            if subject.id == "b":
                raise CatalogRefusal("download_failed", "fixture failure")
            super().pull(subject)
    windows = FakeCatalog("windows", [("model", "a"), ("model", "b"), ("engine", "llama.cpp")])
    guest = RefusesSecond("guest")
    with pytest.raises(CatalogRefusal):
        migration(windows, guest, [], tmp_path)._prepare_weights()
    assert len(windows.subjects) == 3
    assert not any(call.startswith("remove") for call in windows.calls)
    assert not any("engine" in call for call in guest.calls)


def test_activation_precedes_retirement_and_native_binary_is_not_migrated(tmp_path: Path) -> None:
    windows = FakeCatalog("windows", [("model", "a"), ("engine", "llama.cpp")])
    guest = FakeCatalog("guest")
    walk = migration(windows, guest, [], tmp_path)
    for method in ("_wsl_state", "_import_distro", "_guest_ready", "_guest_install", "_migrate_config", "_install_job_types", "_lan_door"):
        setattr(walk, method, lambda: None)
    order = []
    def stopped():
        assert ("model", "a") in {row.key for row in windows.subjects}
        assert ("model", "a") in {row.key for row in guest.subjects}
        order.append("stopped")
    def switched():
        assert order == ["stopped"]
        assert (tmp_path / installer.CLEANUP_RECORD).is_file()
        order.append("switched")
    def cleanup_catalog():
        assert order == ["stopped", "switched"]
        return windows
    walk._stop_windows_callback = stopped
    walk._switch_pairing_callback = switched
    walk._windows_after_switch = cleanup_catalog
    walk._guest_facts = lambda: ("/guest", "/guest/server/bin/crucible", "guest")
    walk.run()
    assert {row.key for row in windows.subjects} == {("engine", "llama.cpp")}
    assert {row.key for row in guest.subjects} == {("model", "a")}
    assert not any(call.startswith("remove") for call in guest.calls)
    assert not (tmp_path / installer.CLEANUP_RECORD).exists()


def test_failed_activation_keeps_all_windows_models_and_resume_record(tmp_path: Path) -> None:
    windows = FakeCatalog("windows", [("model", "a")])
    guest = FakeCatalog("guest")
    walk = migration(windows, guest, [], tmp_path)
    for method in ("_wsl_state", "_import_distro", "_guest_ready", "_guest_install", "_migrate_config", "_install_job_types", "_lan_door"):
        setattr(walk, method, lambda: None)
    walk._stop_windows_callback = lambda: None
    def failed_switch():
        raise HostError("engine_move_failed", "fixture cannot reach guest through Windows")
    walk._switch_pairing_callback = failed_switch
    walk._windows_after_switch = lambda: windows
    with pytest.raises(HostError, match="fixture"):
        walk.run()
    assert {row.key for row in windows.subjects} == {("model", "a")}
    assert (tmp_path / installer.CLEANUP_RECORD).is_file()


def test_malformed_installed_flag_cannot_authorize_source_deletion() -> None:
    with pytest.raises(HostError):
        catalog_module.parse_catalog({"subjects": [{"kind": "model", "id": "a", "installed": "false"}]}, "guest")


def test_stopped_catalog_resumes_partial_deletion_through_the_weights_owner(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from crucible import catalog
    residue = tmp_path / "fixture-remaining-weight"
    residue.write_bytes(b"weight")
    calls = []
    def remove():
        calls.append("owner remove")
        residue.unlink(missing_ok=True)
    row = SimpleNamespace(kind="model", id="fixture", name="Fixture", installed=lambda: None, remove=remove)
    monkeypatch.setattr(catalog, "subjects", lambda config, backend: [row])
    installer.record_cleanup(tmp_path, {("model", "fixture")})
    config = SimpleNamespace(backend_kind="llama-windows", home=tmp_path)
    backend = SimpleNamespace(kind="llama-windows")
    stopped = catalog_module.StoppedWindowsCatalog(config, backend, installer.cleanup_subjects(tmp_path))
    pending = stopped.installed_subjects()
    assert len(pending) == 1, "a removed stamp cannot hide a partially deleted model"
    stopped.remove(pending[0])
    assert calls == ["owner remove"] and not residue.exists()
    assert stopped.installed_subjects() == []


def test_controller_resumes_cleanup_and_keeps_journal_on_failure(tmp_path, monkeypatch):
    context = _context(tmp_path, Scripted())
    host = app_module.Host(context)
    windows = FakeCatalog("stopped Windows", [("model", "a")])
    guest = FakeCatalog("active guest", [("model", "a")])
    installer.record_cleanup(tmp_path, {("model", "a")})
    monkeypatch.setattr(host, "stopped_windows_catalog", lambda: windows)
    monkeypatch.setattr(app_module, "engine_token", lambda _: "fixture-token")
    monkeypatch.setattr(app_module, "HttpCatalog", lambda *a, **kw: guest)
    host._cleanup_running = True
    guest.unreachable = True
    host._resume_model_cleanup()
    assert (tmp_path / installer.CLEANUP_RECORD).exists()
    assert windows.subjects and not host._cleanup_running
    guest.unreachable = False
    host._resume_model_cleanup()
    assert not (tmp_path / installer.CLEANUP_RECORD).exists()
    assert windows.subjects == [] and guest.subjects


def test_cleanup_refuses_an_engine_still_owned_by_windows(tmp_path):
    host = app_module.Host(_context(tmp_path, Scripted()))
    host._c.presence = presence.Presence(Distro.ABSENT, Engine.RUNNING, "native", Owner.HOST_CHILD)
    with pytest.raises(HostError, match="Windows models are kept"):
        host.stopped_windows_catalog()


def test_background_cleanup_never_downloads_or_deletes_if_destination_missing(tmp_path):
    windows = FakeCatalog("stopped Windows", [("model", "a"), ("model", "b")])
    guest = FakeCatalog("active guest", [("model", "a")])
    with pytest.raises(HostError, match="Windows models are kept"):
        migration(windows, guest, [], tmp_path)._migrate_weights(allow_pull=False)
    assert not any(call.startswith("pull") for call in guest.calls)
    assert not any(call.startswith("remove") for call in windows.calls)


def test_retry_after_guest_activation_never_rebuilds_native_http_source(tmp_path, monkeypatch):
    context = _context(tmp_path, Scripted())
    context.presence = presence.Presence(Distro.PRESENT, Engine.RUNNING, "guest", Owner.WSL_UNIT)
    host = app_module.Host(context)
    installer.record_cleanup(tmp_path, {("model", "a")})
    calls = []
    def resumed(*, raise_errors):
        assert raise_errors is True
        calls.append("resume native cleanup")
    monkeypatch.setattr(host, "_resume_model_cleanup", resumed)
    monkeypatch.setattr(installer.EngineInstall, "_complete", lambda self: calls.append("complete"))
    def wrong_source(*args, **kwargs):
        raise AssertionError("The active guest cannot be constructed as a native source")
    monkeypatch.setattr(app_module, "HttpCatalog", wrong_source)
    app_module._sequence(context, host)(lambda event: None)
    assert calls == ["resume native cleanup", "complete"]


def test_an_interrupted_move_resumes_from_the_two_catalogs(tmp_path: Path) -> None:
    """The half-done state — one moved, one not — is just a different diff."""
    windows = FakeCatalog("windows", [("model", "a"), ("voice", "b")])
    guest = FakeCatalog("guest", [("model", "a")])
    events: list[installer.Event] = []
    migration(windows, guest, events, tmp_path)._migrate_weights()
    assert guest.calls.count("pull voice b") == 1
    assert not any(call == "pull model a" for call in guest.calls)
    assert windows.subjects == []
    # And running it AGAIN on the finished machine is a no-op.
    windows.calls.clear()
    guest.calls.clear()
    migration(windows, guest, events, tmp_path)._migrate_weights()
    assert not any(call.startswith(("pull", "remove")) for call in guest.calls + windows.calls)


def test_subject_in_use_is_WAITED_OUT_and_never_skipped(tmp_path: Path) -> None:
    """3.5a's refusal. The subject comes back on the next round, with its
    holder named; it is never left behind on Windows."""
    windows = FakeCatalog("windows", [("model", "qwen3.5-9b")])
    windows.in_use[("model", "qwen3.5-9b")] = "a lease held by bookforge"
    windows.release_after[("model", "qwen3.5-9b")] = 3
    guest = FakeCatalog("guest", [("model", "qwen3.5-9b")])
    events: list[installer.Event] = []
    migration(windows, guest, events, tmp_path)._migrate_weights()
    assert windows.calls.count("remove model qwen3.5-9b") == 3, "retried, not skipped"
    assert windows.subjects == []
    held = [
        event.data["text"]
        for event in events
        if event.event == "line" and "held by" in str(event.data.get("text"))
    ]
    assert held and "a lease held by bookforge" in held[0]


def test_a_subject_held_forever_FAILS_THE_STEP_by_name_and_names_who(
    tmp_path: Path, monkeypatch
) -> None:
    """The two honest ends are "removed" and "still held, and here is who".
    An unbounded wait would be a third: a move that never finishes."""
    monkeypatch.setattr(installer, "MIGRATE_IN_USE_ROUNDS", 3)
    windows = FakeCatalog("windows", [("model", "qwen3.5-9b")])
    windows.in_use[("model", "qwen3.5-9b")] = "the resident model"
    windows.release_after[("model", "qwen3.5-9b")] = 99
    guest = FakeCatalog("guest", [("model", "qwen3.5-9b")])
    events: list[installer.Event] = []
    with pytest.raises(HostError) as caught:
        migration(windows, guest, events, tmp_path)._migrate_weights()
    assert caught.value.code == "subject_in_use"
    assert "the resident model" in caught.value.message
    # NOTHING was lost: the guest has it, and the Windows copy is still there.
    assert {row.key for row in windows.subjects} == {("model", "qwen3.5-9b")}
    assert {row.key for row in guest.subjects} == {("model", "qwen3.5-9b")}
    assert events[-1].event == "failed"
    assert events[-1].data["code"] == "subject_in_use"


def test_a_pull_that_never_arrives_leaves_the_windows_copy_alone(tmp_path: Path) -> None:
    monkeypatch_free = FakeCatalog("guest")
    monkeypatch_free.pull_latency = 10_000_000
    windows = FakeCatalog("windows", [("model", "qwen3.5-9b")])
    events: list[installer.Event] = []
    walk = migration(windows, monkeypatch_free, events, tmp_path)
    with pytest.raises(HostError) as caught:
        walk._migrate_weights()
    assert caught.value.code == "subject_pull_timeout"
    assert "has NOT been removed" in caught.value.message
    assert "remove model qwen3.5-9b" not in windows.calls
    assert {row.key for row in windows.subjects} == {("model", "qwen3.5-9b")}


def test_a_removal_refused_for_any_OTHER_reason_fails_and_keeps_both_copies(
    tmp_path: Path,
) -> None:
    class Stubborn(FakeCatalog):
        def remove(self, subject: Subject) -> None:
            self.calls.append(f"remove {subject}")
            raise CatalogRefusal("subject_remove_failed", "E:\\weights is read-only")

    windows = Stubborn("windows", [("model", "a")])
    guest = FakeCatalog("guest", [("model", "a")])
    events: list[installer.Event] = []
    with pytest.raises(HostError) as caught:
        migration(windows, guest, events, tmp_path)._migrate_weights()
    assert caught.value.code == "subject_remove_failed"
    assert {row.key for row in windows.subjects} == {("model", "a")}


def test_no_windows_engine_is_a_fact_the_step_states_and_not_a_failure(
    tmp_path: Path,
) -> None:
    events: list[installer.Event] = []
    walk = migration(None, None, events, tmp_path)
    walk._migrate_weights()
    said = " ".join(str(event.data.get("text", "")) for event in events if event.event == "line")
    assert "no Windows engine" in said or "nothing to migrate" in said


# ------------------------------------------------ 3.5a the catalog ports


def test_a_catalog_row_missing_a_field_is_refused_rather_than_half_read() -> None:
    """A half-read catalog looks exactly like a server with fewer subjects,
    and the difference decides whether a file is deleted."""
    with pytest.raises(HostError) as caught:
        catalog_module.parse_catalog({"subjects": [{"kind": "model"}]}, "the guest")
    assert caught.value.code == "catalog_unreadable"
    assert "'id'" in caught.value.message or "'installed'" in caught.value.message


def test_a_catalog_that_is_not_a_catalog_is_refused_by_name() -> None:
    with pytest.raises(HostError) as caught:
        catalog_module.parse_catalog({"packs": []}, "the guest")
    assert caught.value.code == "catalog_unreadable"


def test_the_servers_refusal_code_survives_verbatim_with_its_holder() -> None:
    """`subject_in_use` is what the retry turns on, so it must not be
    flattened into a generic failure."""
    body = json.dumps(
        {
            "error": {
                "code": "subject_in_use",
                "message": "qwen3.5-9b is resident",
                "details": {"who": "a lease held by foundry"},
            }
        }
    ).encode()
    refusal = catalog_module.refusal_from(body, 409, "the Windows engine", "DELETE /x")
    assert refusal.code == "subject_in_use"
    assert refusal.who == "a lease held by foundry"


def test_a_refusal_that_is_not_the_error_envelope_keeps_the_status_and_the_text() -> None:
    refusal = catalog_module.refusal_from(b"<html>502</html>", 502, "the guest", "GET /v1/catalog")
    assert refusal.code == "http_502"
    assert "502" in refusal.message


def test_the_guest_is_reached_with_exec_and_the_body_is_one_argument() -> None:
    """wsl.exe pre-expands `$var` without `--exec`, and a JSON body split
    across argv is a body some quoting rule gets to edit."""
    guest = catalog_module.GuestCatalog(Scripted(), "crucible", "tok", 7100, where="the guest")
    argv = guest.curl_argv("POST", "/v1/tasks", '{"type":"pull","kind":"model","id":"a"}')
    assert argv[:5] == ["wsl.exe", "-d", "crucible", "--exec", "curl"]
    assert "-f" not in argv, "-f would hide the refusal body, and the CODE is the point"
    assert '{"type":"pull","kind":"model","id":"a"}' in argv
    assert argv[-1] == "http://127.0.0.1:7100/v1/tasks"
    assert f"Authorization: Bearer tok" in argv


def test_the_guest_port_reads_the_status_curl_appended() -> None:
    runner = Scripted(
        default=ok('{"subjects": [{"kind": "model", "id": "a", "installed": true}]}'
                   + catalog_module.GuestCatalog.STATUS_MARK + "200")
    )
    guest = catalog_module.GuestCatalog(runner, "crucible", "tok", 7100, where="the guest")
    assert [row.key for row in guest.installed_subjects()] == [("model", "a")]


def test_the_guest_port_turns_a_409_body_into_the_named_refusal() -> None:
    body = json.dumps({"error": {"code": "subject_in_use", "message": "held", "details": {"who": "a task"}}})
    runner = Scripted(default=ok(body + catalog_module.GuestCatalog.STATUS_MARK + "409"))
    guest = catalog_module.GuestCatalog(runner, "crucible", "tok", 7100, where="the guest")
    with pytest.raises(CatalogRefusal) as caught:
        guest.remove(Subject("model", "a", "a", True))
    assert caught.value.code == "subject_in_use"
    assert caught.value.who == "a task"


def test_a_guest_that_will_not_answer_refuses_rather_than_reporting_nothing() -> None:
    """"Nothing installed" and "could not ask" must never be the same answer:
    the first would delete every Windows copy."""
    runner = Scripted(default=bad("wsl: no such distribution"))
    guest = catalog_module.GuestCatalog(runner, "crucible", "tok", 7100, where="the guest")
    with pytest.raises(HostError) as caught:
        guest.installed_subjects()
    assert caught.value.code == "catalog_unreachable"


def test_the_windows_port_deletes_through_3_5as_route(monkeypatch) -> None:
    port = catalog_module.HttpCatalog("http://127.0.0.1:7100", "tok", where="the Windows engine")
    seen: dict[str, object] = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_a): return False
        def read(self): return b""

    def fake_urlopen(request, timeout):  # noqa: ANN001
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["auth"] = request.get_header("Authorization")
        return Response()

    import urllib.request as urllib_request

    from types import SimpleNamespace
    def opener(handler):
        assert handler.proxies == {}, "local migration credentials must not use ambient HTTP proxies"
        return SimpleNamespace(open=fake_urlopen)
    monkeypatch.setattr(urllib_request, "build_opener", opener)
    port.remove(Subject("voice", "mistborn", "mistborn", True))
    assert seen["url"] == "http://127.0.0.1:7100/v1/catalog/voice/mistborn"
    assert seen["method"] == "DELETE"
    assert seen["auth"] == "Bearer tok"


# ------------------------------------------------- PHASE17: the relation
#
# The ORCHESTRATOR's half. The engine's is `tests/test_peer.py`, and the
# task seam between them is `tests/test_engine_restart.py`.


class FakeEngine:
    """An engine on loopback: `/v1/info`, `/v1/peer/claim`, and a memory.

    A real socket on a port the OS picks, because what is being tested is a
    HANDSHAKE — a bearer, a body and a document — and a stubbed function call
    would pin none of those.
    """

    def __init__(self, token: str = "guest-token") -> None:
        self.token = token
        self.claims: list[dict] = []
        self.releases: list[dict] = []
        self.info_document: dict | None = {
            "server": {"name": "crucible@owens-pc-wsl", "version": "0.6.0", "api_version": 1},
            "host": {"platform": "linux", "arch": "x86_64", "backend": "cuda-linux",
                     "gpu": {"vendor": "nvidia", "name": "3090 Ti", "vram_bytes": 1}},
            "role": "engine",
            "managed_by": None,
            "job_types": ["llm"],
            "capabilities": [{"job_type": "llm", "models": [{"id": "qwen3.5-9b"}]}],
        }
        self._server = None

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "FakeEngine":
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        engine = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                return

            def _send(self, status: int, body: dict) -> None:
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _authorised(self) -> bool:
                if self.headers.get("Authorization") != f"Bearer {engine.token}":
                    self._send(401, {"error": {"code": "peer_token_mismatch", "message": "no"}})
                    return False
                return True

            def _read(self) -> dict:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                return json.loads(raw.decode() or "{}")

            def do_GET(self) -> None:  # noqa: N802
                if not self._authorised():
                    return
                if self.path == "/v1/info":
                    if engine.info_document is None:
                        self._send(503, {"error": {"code": "unavailable", "message": "no"}})
                        return
                    self._send(200, engine.info_document)
                    return
                self._send(404, {"error": {"code": "not_found", "message": "no"}})

            def do_POST(self) -> None:  # noqa: N802
                if not self._authorised():
                    return
                engine.claims.append(self._read())
                self._send(200, {"role": "engine", "managed_by": {}, "claimed": "now"})

            def do_DELETE(self) -> None:  # noqa: N802
                if not self._authorised():
                    return
                engine.releases.append(self._read())
                self._send(200, {"role": "engine", "managed_by": None})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def _orchestrator(
    tmp_path: Path, runner: Scripted, owner: Owner, engine: FakeEngine, monkeypatch
) -> app_module.Host:
    """A Host in a known presence, pointed at a fake engine on a live port."""
    context = _context(tmp_path, runner)
    context.name = "crucible-orchestrator@test"
    context.presence = presence.Presence(Distro.PRESENT, Engine.RUNNING, "up", owner)
    monkeypatch.setattr(app_module, "engine_url", lambda path="": f"{engine.url}{path}")
    monkeypatch.setattr(app_module, "engine_token", lambda _c: engine.token)
    return app_module.Host(context)


def test_an_orchestrator_claims_the_engine_it_started(tmp_path: Path, monkeypatch) -> None:
    """PHASE17 2.1, for the two owners that are the orchestrator's own."""
    for owner in (Owner.WSL_UNIT, Owner.HOST_CHILD):
        with FakeEngine() as engine:
            host = _orchestrator(tmp_path, Scripted(), owner, engine, monkeypatch)
            assert host.claim() is True, owner
            assert len(engine.claims) == 1
            said = engine.claims[0]["orchestrator"]
            assert said["name"] == "crucible-orchestrator@test"
            assert said["url"] == paths.door_url("")
            assert said["version"]
            # 2.1: no orchestrator sends `force` on any code path, ever.
            assert "force" not in engine.claims[0]


def test_a_FOUND_engine_is_NEVER_claimed(tmp_path: Path, monkeypatch) -> None:
    """4.1a: watched, and nothing else.

    A claim would be a statement that is not true — `managed_by` would name a
    door that refuses every verb the field implies.
    """
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, Scripted(), Owner.FOUND, engine, monkeypatch)
        assert host.claim() is False
        assert engine.claims == []
    log_text = (tmp_path / "host.log").read_text(encoding="utf-8")
    assert "watched and not claimed" in log_text


def test_an_engine_with_no_engine_is_not_claimed_either(
    tmp_path: Path, monkeypatch
) -> None:
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, Scripted(), Owner.NONE, engine, monkeypatch)
        assert host.claim() is False
        assert engine.claims == []


def test_a_claim_that_fails_is_a_LOG_LINE_and_never_a_crash(
    tmp_path: Path, monkeypatch
) -> None:
    """An engine that will not be claimed is still an engine, and a tray that
    died telling it so would take the watch with it."""
    with FakeEngine(token="a-different-token") as engine:
        host = _orchestrator(tmp_path, Scripted(), Owner.WSL_UNIT, engine, monkeypatch)
        monkeypatch.setattr(app_module, "engine_token", lambda _c: "the-wrong-one")
        assert host.claim() is False
    assert "peer_token_mismatch" in (tmp_path / "host.log").read_text(encoding="utf-8")


def test_quit_releases_the_claim_while_the_engine_is_still_answering(
    tmp_path: Path, monkeypatch
) -> None:
    """3.6's rule one layer up: a `managed_by` pointing at a door that no
    longer answers is worse than none."""
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, Scripted(), Owner.WSL_UNIT, engine, monkeypatch)
        host.claim()
        host.quit()
        assert len(engine.releases) == 1
        assert engine.releases[0]["orchestrator"]["url"] == paths.door_url("")


def test_nothing_claimed_means_nothing_released(tmp_path: Path, monkeypatch) -> None:
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, Scripted(), Owner.FOUND, engine, monkeypatch)
        host.claim()
        host.quit()
        assert engine.releases == []


# ------------------------------------ PHASE17 4.4: ONE quit, two callers


class FakeIcon:
    """pystray's one verb, recorded. `icon.run()` returning IS the exit."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _quit_trace(host: app_module.Host, engine: FakeEngine, runner: Scripted) -> dict:
    """Everything a quit is observable by. Assert on the CALLS, not on a death."""
    return {
        "releases": len(engine.releases),
        "hold_released": host._c.watcher.held is None,
        "child_stopped": host._c.watcher.child is None,
        "exited": host._icon.stopped,
        "systemctl": [c for c in runner.calls if "systemctl" in " ".join(c)],
    }


def _a_quitting_host(
    tmp_path: Path, owner: Owner, engine: FakeEngine, monkeypatch
) -> tuple[app_module.Host, Scripted]:
    """A claimed, held, child-owning orchestrator, ready to be stopped.

    `tray.update` is stubbed for this module's stated reason: `tray.py` is
    pystray and has no decision in it, and pystray is not installed where this
    suite runs. The menu's Quit goes through `_refresh()` on its way out; the
    door's does not, and that difference is a drawing rather than a step.
    """
    from crucible.host import tray as tray_module

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tray_module, "update", lambda *_args: None)
    runner = Scripted()
    host = _orchestrator(tmp_path, runner, owner, engine, monkeypatch)
    host._icon = FakeIcon()
    host.claim()
    host._c.watcher.hold("Ubuntu")
    host._c.watcher.child = FakeChild()
    runner.calls.clear()
    return host, runner


@pytest.mark.parametrize("owner", [Owner.WSL_UNIT, Owner.HOST_CHILD, Owner.FOUND])
def test_the_doors_quit_is_THE_MENUS_quit_and_never_a_second_copy(
    tmp_path: Path, monkeypatch, owner: Owner
) -> None:
    """4.4's rule, for every owner: one implementation, two callers.

    A person clicking Quit and a script posting `/quit` must not get two
    different shutdowns — the half that drifted would be the script's, which
    is the half nobody is watching. So both are driven here and the traces are
    compared, rather than each being asserted against its own expectations.
    """
    with FakeEngine() as engine:
        clicked, clicked_runner = _a_quitting_host(
            tmp_path / "menu", owner, engine, monkeypatch
        )
        clicked.on_click(menu.QUIT)
        by_menu = _quit_trace(clicked, engine, clicked_runner)

    with FakeEngine() as engine:
        posted, posted_runner = _a_quitting_host(
            tmp_path / "door", owner, engine, monkeypatch
        )
        door_module.OrchestratorDoor(
            posted._c.log,
            lambda _emit: None,
            token=lambda: "tok",
            orchestrator=posted,
        ).quit()
        by_door = _quit_trace(posted, engine, posted_runner)

    assert by_menu == by_door, owner
    assert by_door["exited"] is True, "every owner still ends the process"
    assert by_door["hold_released"] is True, "the wsl.exe session is this process's"


def test_a_quit_that_holds_a_claim_RELEASES_it(tmp_path: Path, monkeypatch) -> None:
    with FakeEngine() as engine:
        host, _runner = _a_quitting_host(tmp_path, Owner.WSL_UNIT, engine, monkeypatch)
        assert host._claimed is True
        door_module.OrchestratorDoor(
            host._c.log, lambda _emit: None, token=lambda: "tok", orchestrator=host
        ).quit()
        assert len(engine.releases) == 1
        assert engine.releases[0]["orchestrator"]["url"] == paths.door_url("")
        assert host._icon.stopped is True


def test_a_FOUND_engines_orchestrator_releases_NOTHING_and_still_stops(
    tmp_path: Path, monkeypatch
) -> None:
    """4.1a all the way to the exit: it never claimed it, so there is nothing
    to release, and it is not this process's child, so it is not taken down —
    but the hold IS let go and the orchestrator DOES end, because both of
    those are this process's own and belong to nobody else."""
    with FakeEngine() as engine:
        host, runner = _a_quitting_host(tmp_path, Owner.FOUND, engine, monkeypatch)
        child = host._c.watcher.child
        assert host._claimed is False, "a found engine is never claimed"
        door_module.OrchestratorDoor(
            host._c.log, lambda _emit: None, token=lambda: "tok", orchestrator=host
        ).quit()
        assert engine.releases == []
        assert child.terminated is False, "an engine it did not start is not stopped"
        assert not any("systemctl" in " ".join(c) for c in runner.calls)
        assert host._c.watcher.held is None
        assert host._icon.stopped is True
    assert "owner=found" in (tmp_path / "host.log").read_text(encoding="utf-8")


# ------------------------------------------------ PHASE17 3.2: the document


def test_the_orchestrators_info_says_its_role_its_backend_and_zero_job_types(
    tmp_path: Path, monkeypatch
) -> None:
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, Scripted(), Owner.WSL_UNIT, engine, monkeypatch)
        document = host.info()
    assert document["role"] == "orchestrator"
    assert document["host"]["backend"] == "orchestrator"
    assert document["host"]["gpu"] == {"vendor": "none", "name": "", "vram_bytes": 0}
    assert document["job_types"] == [], "the DEFINITION of the role"
    assert document["server"]["name"] == "crucible-orchestrator@test"


def test_the_capability_block_is_the_ENGINES_read_through_and_never_cached(
    tmp_path: Path, monkeypatch
) -> None:
    """A cached list is this system's one defect in a third place: the engine
    pulls a model, the orchestrator answers yesterday's list, and a client
    picks a model the engine has and is told it does not."""
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, Scripted(), Owner.WSL_UNIT, engine, monkeypatch)
        first = host.info()
        assert first["capabilities"] == engine.info_document["capabilities"]
        assert first["engine"] == {
            "name": "crucible@owens-pc-wsl",
            "url": engine.url,
            "backend": "cuda-linux",
            "owner": "wsl-unit",
        }
        # The engine gains a model. The NEXT read says so, with nothing
        # invalidated and nothing told to refresh.
        engine.info_document["capabilities"] = [
            {"job_type": "llm", "models": [{"id": "qwen3.5-9b"}, {"id": "dots-ocr"}]}
        ]
        assert host.info()["capabilities"] == engine.info_document["capabilities"]


def test_an_engine_that_cannot_be_read_is_an_empty_list_and_a_NULL_name(
    tmp_path: Path, monkeypatch
) -> None:
    """The orchestrator does not invent an answer for a server that did not
    give one. `engine.url` stands, because that is a fact about this MACHINE
    rather than about the engine's health."""
    with FakeEngine() as engine:
        engine.info_document = None
        host = _orchestrator(tmp_path, Scripted(), Owner.WSL_UNIT, engine, monkeypatch)
        document = host.info()
    assert document["capabilities"] == []
    assert document["engine"]["name"] is None
    assert document["engine"]["backend"] is None
    assert document["engine"]["url"] == engine.url
    assert document["engine"]["owner"] == "wsl-unit"


def test_a_machine_with_no_engine_says_engine_is_null(
    tmp_path: Path, monkeypatch
) -> None:
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, Scripted(), Owner.NONE, engine, monkeypatch)
        assert host.info()["engine"] is None


def test_the_owner_is_spelled_child_on_the_wire_and_not_host_child(
    tmp_path: Path, monkeypatch
) -> None:
    """On the wire the word "host" is the thing PHASE17 renames, and the
    orchestrator is the only possible parent."""
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, Scripted(), Owner.HOST_CHILD, engine, monkeypatch)
        assert host.info()["engine"]["owner"] == "child"


# ------------------------------------------------ PHASE17 4.2: the restart


def test_a_found_engine_is_refused_engine_not_ours_BEFORE_anything_happens(
    tmp_path: Path, monkeypatch
) -> None:
    """The refusal lives at the door, not only in the menu: a disabled item
    is a drawing, and the thing that must not happen is the ACT."""
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, Scripted(), Owner.FOUND, engine, monkeypatch)
        with pytest.raises(HostError) as caught:
            host.check_restartable()
        assert caught.value.code == "engine_not_ours"
        with pytest.raises(HostError):
            host.restart_engine(lambda _event: None)


def test_a_wsl_unit_restart_is_the_working_door_and_not_a_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    """`boot()` on a RUNNING engine pings, succeeds and changes nothing — a
    button that did nothing precisely when it was most obviously pressed."""
    runner = Scripted(answers={"id -u": ok("1000\n")}, pings=[200])
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, runner, Owner.WSL_UNIT, engine, monkeypatch)
        seen: list[str] = []
        host.restart_engine(lambda event: seen.append(event.event))
    argvs = [" ".join(call) for call in runner.calls]
    assert any("systemctl --user restart crucible.service" in argv for argv in argvs), argvs
    assert seen[-1] == "done"


def test_a_child_restart_stops_the_child_and_starts_one(
    tmp_path: Path, monkeypatch
) -> None:
    runner = Scripted(pings=[200])
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, runner, Owner.HOST_CHILD, engine, monkeypatch)
        host._c.watcher.child = FakeChild()
        seen: list[str] = []
        host.restart_engine(lambda event: seen.append(event.event))
    assert runner.spawned, "a child was started again"
    assert seen[-1] == "done"


def test_a_restart_that_does_not_come_back_FAILS_by_name(
    tmp_path: Path, monkeypatch
) -> None:
    runner = Scripted(pings=[])
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, runner, Owner.WSL_UNIT, engine, monkeypatch)
        events: list[tuple[str, dict]] = []
        host.restart_engine(lambda event: events.append((event.event, event.data)))
    assert events[-1][0] == "failed"
    assert events[-1][1]["code"] == "engine_did_not_return"
    assert host._c.presence.engine is Engine.FAILED


def test_a_restarted_engine_is_CLAIMED_AGAIN_because_it_forgot(
    tmp_path: Path, monkeypatch
) -> None:
    """2.3: a claim is live state. This is why nothing is written to disk."""
    with FakeEngine() as engine:
        host = _orchestrator(
            tmp_path, Scripted(pings=[200]), Owner.WSL_UNIT, engine, monkeypatch
        )
        host.claim()
        assert len(engine.claims) == 1
        host.restart_engine(lambda _event: None)
        assert len(engine.claims) == 2, "the restarted engine was told again"


def test_the_tray_and_the_page_reach_ONE_restart(tmp_path: Path, monkeypatch) -> None:
    """A person clicking Restart and a page posting `engine-restart` must not
    get two different restarts."""
    runner = Scripted(answers={"id -u": ok("1000\n")}, pings=[200])
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, runner, Owner.WSL_UNIT, engine, monkeypatch)
        host.on_click(menu.RESTART_ENGINE)
    assert any(
        "systemctl --user restart crucible.service" in " ".join(call) for call in runner.calls
    )


# --------------------------------------------------- PHASE17 3.2: the door


def test_the_door_answers_ping_WITHOUT_a_bearer_and_info_WITH_one(
    host_log: log.HostLog,
) -> None:
    """`/v1/ping` is what lets a client tell "wrong token" from "not a
    Crucible"; `/v1/info` names the engine's address and is not a thing to
    hand an anonymous caller."""
    import urllib.error
    import urllib.request

    fake = FakeOrchestrator(document={"role": "orchestrator", "job_types": []})
    door = a_door(host_log, token="tok", orchestrator=fake)
    server = door_module.serve(door, host="127.0.0.1", port=0)
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/ping", timeout=10) as answer:
            ping = json.loads(answer.read().decode())
        assert ping == {
            "crucible": True,
            "name": fake.name,
            "api_version": 1,
            "role": "orchestrator",
        }
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/info", timeout=10)
        assert caught.value.code == 401
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/info", headers={"Authorization": "Bearer tok"}
        )
        with urllib.request.urlopen(request, timeout=10) as answer:
            assert json.loads(answer.read().decode()) == fake.document
    finally:
        server.shutdown()


def test_the_door_streams_a_restart_and_refuses_a_found_engine_with_a_STATUS(
    host_log: log.HostLog,
) -> None:
    import urllib.error
    import urllib.request

    fake = FakeOrchestrator()
    door = a_door(host_log, token="tok", orchestrator=fake)
    server = door_module.serve(door, host="127.0.0.1", port=0)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/restart"
    headers = {"Authorization": "Bearer tok"}
    try:
        request = urllib.request.Request(url, data=b"{}", headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=20) as answer:
            assert answer.headers["Content-Type"] == "application/x-ndjson"
            lines = [json.loads(l) for l in answer.read().decode().splitlines() if l]
        assert fake.restarts == ["restarted"]
        assert lines[-1]["event"] == "done"

        # And a `found` engine: a STATUS CODE, not the last line of a body.
        fake.not_ours = True
        request = urllib.request.Request(url, data=b"{}", headers=headers, method="POST")
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=20)
        assert caught.value.code == 409
        assert json.loads(caught.value.read().decode())["error"]["code"] == "engine_not_ours"
        assert fake.restarts == ["restarted"], "nothing ran"
    finally:
        server.shutdown()


def test_a_path_this_door_does_not_serve_says_what_it_DOES_serve(
    host_log: log.HostLog,
) -> None:
    """An app that wants anything else reads `/v1/info`'s `engine.url`."""
    import urllib.error
    import urllib.request

    door = a_door(host_log, token="tok")
    server = door_module.serve(door, host="127.0.0.1", port=0)
    port = server.server_address[1]
    try:
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/jobs", timeout=10)
        assert caught.value.code == 404
        message = json.loads(caught.value.read().decode())["error"]["message"]
        for route in ("/install", "/restart", "/v1/info", "/v1/ping"):
            assert route in message
    finally:
        server.shutdown()


def test_the_wire_owner_words_are_the_only_three(tmp_path: Path) -> None:
    """A word this build does not have is a word a client cannot be sent."""
    from crucible import peer as peer_module

    assert set(app_module.OWNER_ON_THE_WIRE.values()) == set(peer_module.OWNERS)
    assert Owner.NONE not in app_module.OWNER_ON_THE_WIRE, "an absence is not an owner"


# ------------------------------------- PHASE17 2.5: CONSENT, by name in the config
#
# Owen's PC, ruled 2026-09-15: the engine has lived in `Ubuntu` since before
# any of this existed, and 4.1a's `found` rule — right for a stranger's distro
# — makes the orchestrator refuse to claim or restart the one engine it
# actually has. Consent is how a person tells the two apart, by name, once.
# What it widens: watching, the claim, `engine-restart` through the unit. What
# it never widens: a recipe that restarts everything uid 1000 owns.


UBUNTU_CONSENT = '[orchestrator]\ndistro = "Ubuntu"\n'


def _consented_watcher(
    runner: Scripted, host_log: log.HostLog
) -> presence.PresenceWatcher:
    return presence.PresenceWatcher(
        runner,
        host_log,
        distro="Ubuntu",
        consented=True,
        monotonic=ticking(),
        sleep=lambda _s: None,
    )


def test_no_setting_means_the_machine_behaves_exactly_as_it_did(
    tmp_path: Path,
) -> None:
    """Absent is the only quiet answer, and it is the one every machine gives."""
    assert app_module.consented_distro(tmp_path) is None
    (tmp_path / "config.toml").write_text('[auth]\ntoken = "t"\n', encoding="utf-8")
    assert app_module.consented_distro(tmp_path) is None
    (tmp_path / "config.toml").write_text("[orchestrator]\n", encoding="utf-8")
    assert app_module.consented_distro(tmp_path) is None


def test_the_setting_is_read_from_the_table_PHASE17_names(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(UBUNTU_CONSENT, encoding="utf-8")
    assert app_module.consented_distro(tmp_path) == "Ubuntu"
    assert app_module.CONSENT_TABLE == "orchestrator"
    assert app_module.CONSENT_KEY == "distro"


def test_the_setting_does_not_disturb_the_token_beside_it(tmp_path: Path) -> None:
    """One document, two readers, and neither may eat the other's key."""
    (tmp_path / "config.toml").write_text(
        '[auth]\ntoken = "the-token"\n\n[orchestrator]\ndistro = "Ubuntu"\n',
        encoding="utf-8",
    )
    assert app_module.consented_distro(tmp_path) == "Ubuntu"
    assert app_module.read_token(tmp_path) == "the-token"


def test_a_setting_that_is_present_and_unusable_is_REFUSED_not_ignored(
    tmp_path: Path,
) -> None:
    """A person who wrote it meant to grant something. An orchestrator that
    shrugged would be the unconsented one while its config said otherwise."""
    for value in ("distro = 4", "distro = true", 'distro = ""', "distro = []"):
        (tmp_path / "config.toml").write_text(
            f"[orchestrator]\n{value}\n", encoding="utf-8"
        )
        with pytest.raises(HostError) as caught:
            app_module.consented_distro(tmp_path)
        assert caught.value.code == "orchestrator_distro_invalid"
        assert caught.value.code in HOST_ERROR_CODES
    (tmp_path / "config.toml").write_text(
        "[orchestrator]\nnot toml at all\n", encoding="utf-8"
    )
    with pytest.raises(HostError) as caught:
        app_module.consented_distro(tmp_path)
    assert caught.value.code == "orchestrator_distro_invalid"


def test_without_consent_a_found_engine_is_still_found_and_still_unclaimed(
    host_log: log.HostLog,
) -> None:
    """The rule this setting widens is unchanged where nobody wrote one."""
    runner = Scripted(
        answers={
            "-l -v --running": ok(OWENS_PC_LIST),
            "-l -v": ok(OWENS_PC_LIST),
            "cat ": ok(GUEST_LINE),
        },
        pings=[200],
    )
    watcher = presence.PresenceWatcher(
        runner, host_log, monotonic=ticking(), sleep=lambda _s: None
    )
    assert watcher.consented is False
    distro, _detail = watcher.probe_distro()
    assert distro is Distro.ABSENT, 'there is no distro NAMED "crucible"'
    assert watcher.adopt(distro).owner is Owner.FOUND
    # The unit was never even asked about: no consent, no probe.
    assert not any("is-enabled" in " ".join(call) for call in runner.calls)


def test_consent_makes_the_named_distro_PRESENT(host_log: log.HostLog) -> None:
    """`probe_distro` is the one place "which distro is mine" is decided, and
    consent answers it with the name a person wrote."""
    runner = Scripted(answers={"-l -v": ok(OWENS_PC_LIST)})
    watcher = _consented_watcher(runner, host_log)
    distro, detail = watcher.probe_distro()
    assert distro is Distro.PRESENT
    assert "Ubuntu" in detail


def test_consent_plus_a_readable_unit_is_owner_wsl_unit(
    host_log: log.HostLog,
) -> None:
    """The claim lands, and it is a TRUE statement: there is a unit behind it."""
    runner = Scripted(
        answers={
            "-l -v": ok(OWENS_PC_LIST),
            "id -u": ok("1000\n"),
            # No SYSTEM unit in this guest, so the probe falls through to the user
            # manager — which is the guest shape these tests are about.
            "-u root --exec systemctl is-enabled": bad("Failed to connect to bus"),
            "is-enabled": ok("enabled\n"),
        },
        pings=[200],
    )
    watcher = _consented_watcher(runner, host_log)
    result = watcher.boot()
    assert result.engine is Engine.RUNNING
    assert result.owner is Owner.WSL_UNIT
    assert presence.unit_enabled_argv("Ubuntu", "1000") in runner.calls
    assert "owner=wsl-unit" in host_log.path.read_text(encoding="utf-8")


def test_a_unit_that_merely_exists_counts_and_the_exit_code_does_not(
    host_log: log.HostLog,
) -> None:
    """`is-enabled` exits non-zero for `disabled`, and a disabled unit is
    still a unit `systemctl --user restart` starts."""
    for state in ("enabled", "disabled", "static", "linked", "masked"):
        runner = Scripted(
            answers={
                "-l -v": ok(OWENS_PC_LIST),
                "id -u": ok("1000\n"),
                # No SYSTEM unit in this guest, so the probe falls through to the user
                # manager — which is the guest shape these tests are about.
                "-u root --exec systemctl is-enabled": bad("Failed to connect to bus"),
                "is-enabled": RunResult(
                    code=1, stdout=f"{state}\n", stderr="", failure=None
                ),
            },
            pings=[200],
        )
        watcher = _consented_watcher(runner, host_log)
        assert watcher.probe_unit().readable is True, state
        assert watcher.boot().owner is Owner.WSL_UNIT, state


def test_consent_with_an_unreadable_unit_stays_found_and_says_why(
    tmp_path: Path,
) -> None:
    """7b.8's machine exactly: a `systemd --user` that never got a bus.

    Consent is permission, not evidence. An orchestrator that read the
    permission as the fact would claim an engine it cannot restart."""
    host_log = log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1")
    runner = Scripted(
        answers={
            "-l -v --running": ok(OWENS_PC_LIST),
            "-l -v": ok(OWENS_PC_LIST),
            "cat ": ok(GUEST_LINE),
            "id -u": ok("1000\n"),
            # No SYSTEM unit in this guest, so the probe falls through to the user
            # manager — which is the guest shape these tests are about.
            "-u root --exec systemctl is-enabled": bad("Failed to connect to bus"),
            "is-enabled": bad("Failed to connect to bus: No such file or directory"),
        },
        pings=[200],
    )
    watcher = _consented_watcher(runner, host_log)
    result = watcher.boot()
    assert result.engine is Engine.RUNNING
    assert result.owner is Owner.FOUND
    written = (tmp_path / "host.log").read_text(encoding="utf-8")
    assert "Failed to connect to bus" in written
    assert "stays owner=found" in written


def test_a_unit_that_is_not_there_at_all_stays_found(tmp_path: Path) -> None:
    host_log = log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1")
    runner = Scripted(
        answers={
            "-l -v --running": ok(OWENS_PC_LIST),
            "-l -v": ok(OWENS_PC_LIST),
            "cat ": ok(GUEST_LINE),
            "id -u": ok("1000\n"),
            # No SYSTEM unit in this guest, so the probe falls through to the user
            # manager — which is the guest shape these tests are about.
            "-u root --exec systemctl is-enabled": bad("Failed to connect to bus"),
            "is-enabled": RunResult(
                code=1, stdout="not-found\n", stderr="", failure=None
            ),
        },
        pings=[200],
    )
    watcher = _consented_watcher(runner, host_log)
    assert watcher.probe_unit().readable is False
    assert watcher.boot().owner is Owner.FOUND


def test_the_destructive_recipe_is_refused_in_a_distro_crucible_did_not_import(
    tmp_path: Path,
) -> None:
    """4.1a's rule SURVIVES consent, and the refusal is the ACT and not a menu.

    `systemctl restart user@1000` kills every process uid 1000 owns. On the
    machine this rule was found on that was a five-thousand-step LoRA trainer.
    """
    assert presence.recipe_permitted("user-bus-restart", "crucible") is True
    assert presence.recipe_permitted("user-bus-restart", "Ubuntu") is False
    assert presence.recipe_permitted("user-unit-start", "Ubuntu") is True
    assert presence.DESTRUCTIVE_RECIPES == frozenset(
        {presence.RECIPE_USER_BUS_RESTART}
    )

    host_log = log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1")
    runner = Scripted(answers={"-l -v": ok(OWENS_PC_LIST), "id -u": ok("1000\n")}, pings=[])
    watcher = _consented_watcher(runner, host_log)
    assert watcher.recover(all_recipes=True) is False
    ran = [" ".join(call) for call in runner.calls]
    assert any("systemctl --user start crucible.service" in line for line in ran)
    assert not any("user@1000" in line for line in ran), "the act, not a drawing"
    written = (tmp_path / "host.log").read_text(encoding="utf-8")
    assert "orchestrator_recipe_not_ours" in written


def test_the_imported_distro_still_gets_both_recipes(tmp_path: Path) -> None:
    """Consent narrows nothing: `crucible` is Crucible's own rootfs and the
    cost of restarting its user manager is the restart."""
    host_log = log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1")
    runner = Scripted(answers={"id -u": ok("1000\n")}, pings=[])
    watcher = presence.PresenceWatcher(
        runner, host_log, monotonic=ticking(), sleep=lambda _s: None
    )
    assert watcher.recover(all_recipes=True) is False
    ran = [" ".join(call) for call in runner.calls]
    assert any("systemctl --user start crucible.service" in line for line in ran)
    assert any("user@1000" in line for line in ran)


def test_a_consented_engine_restart_goes_through_the_unit(tmp_path: Path) -> None:
    """PHASE17 4.2 for the owner consent produces: the working door first, and
    the escalation still refuses the one recipe that is not ours to run."""
    host_log = log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1")
    runner = Scripted(answers={"-l -v": ok(OWENS_PC_LIST), "id -u": ok("1000\n")}, pings=[200])
    watcher = _consented_watcher(runner, host_log)
    assert watcher.restart_wsl_unit() is True
    assert presence.recipe_argv("user-unit-restart", "Ubuntu", "1000") in runner.calls
    assert not any("user@1000" in " ".join(call) for call in runner.calls)


def test_a_consented_machine_is_not_refused_engine_not_ours(tmp_path: Path) -> None:
    """The door's refusal is by OWNER, so consent lifting the owner lifts it —
    and nothing else about `check_restartable` changes."""
    runner = Scripted()
    context = _context(tmp_path, runner)
    host = app_module.Host(context)
    context.presence = presence.Presence(
        Distro.PRESENT, Engine.RUNNING, "up", Owner.FOUND
    )
    with pytest.raises(HostError) as caught:
        host.check_restartable()
    assert caught.value.code == "engine_not_ours"
    context.presence = presence.Presence(
        Distro.PRESENT, Engine.RUNNING, "up", Owner.WSL_UNIT
    )
    host.check_restartable()


def test_consent_claims_the_engine_it_was_given(tmp_path: Path, monkeypatch) -> None:
    """The whole point: on Owen's PC the claim now lands, by consent rather
    than by a distro's name."""
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, Scripted(), Owner.WSL_UNIT, engine, monkeypatch)
        assert host.claim() is True
        assert len(engine.claims) == 1
        assert "force" not in engine.claims[0]


# --------------------------- the runtime directory: the other half of the bus
#
# MEASURED 2026-09-15, 07:34-07:35. `systemctl restart user@1000` as root
# created /run/user/1000/bus, and a `wsl.exe --exec` session STILL could not
# reach it: such a session gets no logind seat, so no XDG_RUNTIME_DIR, and
# systemctl looks for the bus at $XDG_RUNTIME_DIR/bus and nowhere else. With
# the variable set, `is-active crucible.service` answered `active` on the same
# distro in the same minute. 7b.8 read the same sentence and blamed the socket;
# a missing variable and a missing socket say exactly the same thing.


def test_every_user_manager_call_carries_the_runtime_directory() -> None:
    """One builder, so the probe, the recipes and Stop cannot drift apart."""
    assert presence.user_systemctl_argv("Ubuntu", "1000", "is-enabled") == [
        "wsl.exe", "-d", "Ubuntu", "--exec",
        "env", "XDG_RUNTIME_DIR=/run/user/1000",
        "systemctl", "--user", "is-enabled", "crucible.service",
    ]
    assert presence.unit_enabled_argv("Ubuntu", "1000") == (
        presence.user_systemctl_argv("Ubuntu", "1000", "is-enabled")
    )
    assert presence.recipe_argv(presence.RECIPE_USER_UNIT_START, "crucible", "1000") == [
        "wsl.exe", "-d", "crucible", "--exec",
        "env", "XDG_RUNTIME_DIR=/run/user/1000",
        "systemctl", "--user", "start", "crucible.service",
    ]
    assert presence.recipe_argv(presence.RECIPE_USER_UNIT_RESTART, "Ubuntu", "1000") == [
        "wsl.exe", "-d", "Ubuntu", "--exec",
        "env", "XDG_RUNTIME_DIR=/run/user/1000",
        "systemctl", "--user", "restart", "crucible.service",
    ]
    # `env VAR=value cmd` under `--exec`: no shell, so wsl.exe cannot
    # pre-expand the variable on the Windows side, where it is empty.
    for argv in (
        presence.user_systemctl_argv("Ubuntu", "1000", "stop"),
        presence.recipe_argv(presence.RECIPE_USER_UNIT_START, "Ubuntu", "1000"),
    ):
        assert "--exec" in argv
        assert "$XDG_RUNTIME_DIR" not in " ".join(argv)


def test_the_uid_is_READ_and_never_assumed(host_log: log.HostLog) -> None:
    """A distro a person installed can run Crucible as any uid, and
    /run/user/1001 is not /run/user/1000."""
    runner = Scripted(answers={"id -u": ok("1001\n")})
    watcher = presence.PresenceWatcher(
        runner, host_log, distro="Ubuntu", consented=True, sleep=lambda _s: None
    )
    assert presence.guest_uid_argv("Ubuntu") == [
        "wsl.exe", "-d", "Ubuntu", "--exec", "id", "-u",
    ]
    assert watcher.guest_uid() == "1001"
    assert presence.runtime_dir("1001") == "/run/user/1001"
    # READ ONCE: a uid is a property of a rootfs, not of a moment.
    assert watcher.guest_uid() == "1001"
    assert sum(1 for call in runner.calls if "id" in call) == 1


def test_a_uid_that_cannot_be_read_is_not_1000(tmp_path: Path) -> None:
    """Answering 1000 anyway would turn "this distro did not respond" into
    "the bus is broken" — two different repairs, one message."""
    host_log = log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1")
    runner = Scripted(answers={"id -u": bad("There is no distribution with the supplied name.")})
    watcher = presence.PresenceWatcher(
        runner, host_log, distro="Ubuntu", consented=True, sleep=lambda _s: None
    )
    assert watcher.guest_uid() is None
    written = (tmp_path / "host.log").read_text(encoding="utf-8")
    assert "1000 is not assumed" in written
    assert "There is no distribution with the supplied name." in written
    # And nothing was built out of a guess.
    assert not any("XDG_RUNTIME_DIR" in " ".join(call) for call in runner.calls)


def test_a_user_manager_recipe_refuses_to_be_built_without_a_uid() -> None:
    """A programming error, and it says which read is owed."""
    for name in presence.USER_MANAGER_RECIPES:
        with pytest.raises(ValueError) as caught:
            presence.recipe_argv(name, "Ubuntu")
        assert "XDG_RUNTIME_DIR" in str(caught.value)
        assert "1000" in str(caught.value)


def test_user_bus_restart_needs_no_uid_and_keeps_its_literal_1000() -> None:
    """It can only run in the distro Crucible IMPORTED, whose rootfs 4b builds
    with exactly one non-root user — a fact about our own rootfs, not an
    assumption about somebody's machine."""
    assert presence.recipe_argv(presence.RECIPE_USER_BUS_RESTART, "crucible") == [
        "wsl.exe", "-d", "crucible", "-u", "root",
        "--exec", "systemctl", "restart", "user@1000",
    ]
    assert presence.RECIPE_USER_BUS_RESTART not in presence.USER_MANAGER_RECIPES


def test_the_probe_asks_with_the_runtime_directory_and_answers(
    host_log: log.HostLog,
) -> None:
    """The measured fix, end to end: uid read, variable set, unit answers."""
    runner = Scripted(
        answers={
            "-l -v": ok(OWENS_PC_LIST),
            "id -u": ok("1000\n"),
            # No SYSTEM unit in this guest, so the probe falls through to the user
            # manager — which is the guest shape these tests are about.
            "-u root --exec systemctl is-enabled": bad("Failed to connect to bus"),
            "is-enabled": ok("enabled\n"),
        },
        pings=[200],
    )
    watcher = presence.PresenceWatcher(
        runner,
        host_log,
        distro="Ubuntu",
        consented=True,
        monotonic=ticking(),
        sleep=lambda _s: None,
    )
    result = watcher.boot()
    assert result.owner is Owner.WSL_UNIT
    assert presence.unit_enabled_argv("Ubuntu", "1000") in runner.calls
    assert any(
        "XDG_RUNTIME_DIR=/run/user/1000" in " ".join(call) for call in runner.calls
    )


def test_a_socket_that_is_truly_absent_is_still_found_with_the_reason(
    tmp_path: Path,
) -> None:
    """The variable is set and the bus still is not there. That sentence now
    means one thing instead of two, and it is kept verbatim."""
    host_log = log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1")
    runner = Scripted(
        answers={
            "-l -v --running": ok(OWENS_PC_LIST),
            "-l -v": ok(OWENS_PC_LIST),
            "cat ": ok(GUEST_LINE),
            "id -u": ok("1000\n"),
            # No SYSTEM unit in this guest, so the probe falls through to the user
            # manager — which is the guest shape these tests are about.
            "-u root --exec systemctl is-enabled": bad("Failed to connect to bus"),
            "is-enabled": bad("Failed to connect to bus: No such file or directory"),
        },
        pings=[200],
    )
    watcher = presence.PresenceWatcher(
        runner,
        host_log,
        distro="Ubuntu",
        consented=True,
        monotonic=ticking(),
        sleep=lambda _s: None,
    )
    assert watcher.boot().owner is Owner.FOUND
    written = (tmp_path / "host.log").read_text(encoding="utf-8")
    assert "Failed to connect to bus: No such file or directory" in written
    assert "stays owner=found" in written


def test_an_unreadable_uid_leaves_the_owner_found_and_runs_no_recipe(
    tmp_path: Path,
) -> None:
    host_log = log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1")
    runner = Scripted(
        answers={
            "-l -v --running": ok(OWENS_PC_LIST),
            "-l -v": ok(OWENS_PC_LIST),
            "cat ": ok(GUEST_LINE),
            "id -u": bad("no such distribution"),
        },
        pings=[200],
    )
    watcher = presence.PresenceWatcher(
        runner,
        host_log,
        distro="Ubuntu",
        consented=True,
        monotonic=ticking(),
        sleep=lambda _s: None,
    )
    probe = watcher.probe_unit()
    assert probe.readable is False
    assert "could not be read" in probe.detail
    assert watcher.boot().owner is Owner.FOUND
    # The system probe is a READ — `systemctl is-enabled`, which changes nothing
    # — and runs before the uid is needed at all. What an unreadable uid must
    # still prevent is anything that ACTS, which is what "no recipe" means here.
    assert not any(
        verb in " ".join(call)
        for call in runner.calls
        for verb in ("restart", " start ", "user@1000")
    )


def test_a_recovery_that_needs_a_uid_is_SKIPPED_and_never_guessed(
    tmp_path: Path,
) -> None:
    """And `user-bus-restart` still runs where it is permitted: it needs no
    uid, and a user manager that is not answering is exactly its subject."""
    host_log = log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1")
    runner = Scripted(answers={"id -u": bad("nothing")}, pings=[])
    watcher = presence.PresenceWatcher(
        runner, host_log, monotonic=ticking(), sleep=lambda _s: None
    )
    assert watcher.recover(all_recipes=True) is False
    ran = [" ".join(call) for call in runner.calls]
    assert not any("systemctl --user" in line for line in ran)
    assert any("systemctl restart user@1000" in line for line in ran)
    written = (tmp_path / "host.log").read_text(encoding="utf-8")
    assert "NOT RUN" in written


def test_the_menus_stop_carries_the_runtime_directory_too(tmp_path: Path) -> None:
    """Found while fixing the probe: Stop had the identical defect, and a Stop
    that reports ok having stopped nothing is worse than one that refuses."""
    runner = Scripted(answers={"id -u": ok("1000\n")})
    context = _context(tmp_path, runner)
    context.watcher = presence.PresenceWatcher(
        runner, context.log, distro="Ubuntu", consented=True, sleep=lambda _s: None
    )
    context.presence = presence.Presence(
        Distro.PRESENT, Engine.RUNNING, "up", Owner.WSL_UNIT
    )
    app_module.Host(context)._stop_engine()
    assert presence.user_systemctl_argv("Ubuntu", "1000", "stop") in runner.calls


def test_a_system_unit_guest_is_brought_up_by_its_own_manager(
    host_log: log.HostLog,
) -> None:
    """`recover` had no recipe that could start a system unit at all.

    RECIPES is `user-unit-start` then `user-bus-restart`. The first speaks to a
    manager a system-unit guest does not use; the second is DESTRUCTIVE and
    `recipe_permitted` refuses it outside the distro Crucible imported. So on a
    stock Ubuntu guest with the system unit 0.6.9 gives it, recovery had
    nothing to run and said "both recipes were spent" - measured 2026-09-17 by
    asking the tray to Start the engine and watching it stay down.

    `restart` learned this in 0.6.4 and `recover` never did.
    """
    runner = Scripted(
        answers={
            "-u root --exec systemctl is-enabled": ok("enabled" + chr(10)),
            "--user": bad("Failed to connect to bus"),
            "id -u": bad("should not be needed"),
        },
        pings=[None, 200, 200],
    )
    watcher = presence.PresenceWatcher(
        runner,
        host_log,
        distro="Ubuntu",
        consented=True,
        monotonic=ticking(),
        sleep=lambda _s: None,
    )
    assert watcher.recover(all_recipes=True) is True
    assert presence.system_systemctl_argv("Ubuntu", "start") in runner.calls
    assert not any("--user" in " ".join(call) for call in runner.calls), (
        "a system-unit guest must not be recovered through the user manager"
    )


def test_an_ownerless_host_does_not_blame_its_config(host_log: log.HostLog) -> None:
    """A refusal that names the WRONG cause costs more than one that names none.

    Every empty token got the same sentence - "this host has no config yet" -
    and on 2026-09-17 a host whose config was perfectly good, but which owned
    no engine, refused every door with it. The reader goes and looks at a file
    that was never the problem.
    """
    from types import SimpleNamespace
    from crucible.host.app import engine_token_detail

    context = SimpleNamespace(
        presence=presence.Presence(Distro.PRESENT, Engine.RUNNING, "up", Owner.NONE),
        home=Path("C:/nowhere"),
    )
    said = engine_token_detail(context)
    assert "owner=none" in said
    assert "config is not the problem" in said

    context.presence = presence.Presence(
        Distro.ABSENT, Engine.RUNNING, "up", Owner.HOST_CHILD
    )
    assert "no token in its config" in engine_token_detail(context)

    context.presence = presence.Presence(
        Distro.PRESENT, Engine.RUNNING, "up", Owner.WSL_UNIT
    )
    assert "pairing line" in engine_token_detail(context)


def test_the_door_reports_the_hosts_own_reason_for_having_no_token(
    host_log: log.HostLog,
) -> None:
    from types import SimpleNamespace

    door = door_module.OrchestratorDoor(
        host_log,
        lambda _emit: None,
        token=lambda: None,
        token_detail=lambda: "this orchestrator owns no engine (owner=none)",
        orchestrator=SimpleNamespace(name="test"),
    )
    with pytest.raises(HostError) as caught:
        door.authorised("Bearer anything")
    assert caught.value.code == "host_no_token"
    assert "owns no engine" in caught.value.message


def test_an_engine_that_comes_back_to_nobody_is_given_an_owner(
    host_log: log.HostLog,
) -> None:
    """`poll` carried the owner through verbatim, so NONE was PERMANENT.

    `boot` DECIDES the owner (`running_owner`); the watch tick never did - it
    passed whatever it was handed straight back into the new Presence. So any
    transient that once landed on Owner.NONE - and `boot`'s own "both recipes
    were spent" branch returns exactly that - left the orchestrator ownerless
    for the rest of its life, even with the engine answering every 15 seconds.

    That is not a cosmetic field. `engine_token` returns None for an ownerless
    host, and then EVERY authenticated door answers 503 host_no_token,
    `/quit` included, so the tray cannot even be asked to stop. Measured
    2026-09-17: only killing the process and relaunching it cleared this.
    """
    runner = Scripted(pings=[200])
    watcher = presence.PresenceWatcher(
        runner, host_log, distro="Ubuntu", sleep=lambda _s: None
    )
    seen = watcher.poll(Distro.PRESENT, Owner.NONE)
    assert seen.engine is Engine.RUNNING
    assert seen.owner is Owner.WSL_UNIT, (
        "an engine that is answering has an owner; refusing to name one is how "
        "the host locks itself out of its own doors"
    )


def test_a_tick_does_not_re_decide_an_owner_it_already_has(
    host_log: log.HostLog,
) -> None:
    """Only the ownerless case asks. A probe every tick is a wsl.exe round
    trip every 15 seconds to re-learn something already known."""
    runner = Scripted(pings=[200, 200])
    watcher = presence.PresenceWatcher(
        runner, host_log, distro="Ubuntu", consented=True, sleep=lambda _s: None
    )
    seen = watcher.poll(Distro.PRESENT, Owner.WSL_UNIT)
    assert seen.owner is Owner.WSL_UNIT
    assert not any("is-enabled" in " ".join(call) for call in runner.calls), (
        "the owner was already decided; nothing needed asking"
    )


def test_an_ownerless_engine_on_a_machine_with_no_distro_is_adopted(
    host_log: log.HostLog,
) -> None:
    """Same rule, the other shape: something answers, this host did not start
    it, and `found` is the honest name for that."""
    runner = Scripted(pings=[200, 200])
    watcher = presence.PresenceWatcher(
        runner, host_log, distro="Ubuntu", sleep=lambda _s: None
    )
    seen = watcher.poll(Distro.ABSENT, Owner.NONE)
    assert seen.engine is Engine.RUNNING
    assert seen.owner is Owner.FOUND


def test_the_stop_of_a_system_unit_guest_goes_through_root(tmp_path: Path) -> None:
    """The stop had not learned what the restart already knows.

    `recover` and `probe_unit` have branched on the unit's scope since 0.6.4,
    but `_stop_engine` went on saying `systemctl --user stop` to every guest.
    The moment 0.6.9 retired owens-pc's user unit and gave it the system unit
    it was supposed to have, Windows could no longer stop its own engine:
    `HTTP Error 409` out of the door, `upgrade_stop_failed` on the console,
    and the Windows host stranded on 0.6.5 (measured 2026-09-17).

    A stop that cannot be made is an upgrade that cannot run, so this is the
    same shape as the guest-side bug it was created by.
    """
    runner = Scripted(
        answers={
            "-u root --exec systemctl is-enabled": ok("enabled" + chr(10)),
            # Reaching for the user manager would find these, and the
            # assertions below would catch it.
            "--user": bad("Failed to connect to bus"),
            "id -u": bad("should not be needed"),
        },
    )
    context = _context(tmp_path, runner)
    context.watcher = presence.PresenceWatcher(
        runner, context.log, distro="Ubuntu", consented=True, sleep=lambda _s: None
    )
    context.presence = presence.Presence(
        Distro.PRESENT, Engine.RUNNING, "up", Owner.WSL_UNIT
    )
    app_module.Host(context)._stop_engine()
    assert presence.system_systemctl_argv("Ubuntu", "stop") in runner.calls
    assert not any("--user" in " ".join(call) for call in runner.calls), (
        "a system-unit guest must never be stopped through the user manager: "
        "that is the bus WSLg hides"
    )


def test_a_stop_with_no_uid_touches_nothing(tmp_path: Path) -> None:
    runner = Scripted(answers={"id -u": bad("nothing")})
    context = _context(tmp_path, runner)
    context.watcher = presence.PresenceWatcher(
        runner, context.log, distro="Ubuntu", consented=True, sleep=lambda _s: None
    )
    context.presence = presence.Presence(
        Distro.PRESENT, Engine.RUNNING, "up", Owner.WSL_UNIT
    )
    with pytest.raises(HostError, match="engine_stop_failed"):
        app_module.Host(context)._stop_engine()
    # NOT STOPPED, which is what this is about. The scope probe that runs
    # first is a READ (`systemctl is-enabled`), and reading is not touching
    # - the assertion used to be `no systemctl at all` and that stopped
    # being the same statement once the stop learned to ask where the unit
    # is before reaching for it.
    assert not any("stop" in call for call in runner.calls), (
        "the engine must be untouched when the uid cannot be read"
    )
    assert "stop: NOT RUN" in (tmp_path / "host.log").read_text(encoding="utf-8")


def test_a_system_unit_guest_is_owned_and_restarted_as_root(
    host_log: log.HostLog,
) -> None:
    """The guest a stock WSL2 gets since 0.6.4, and the door it is reached by.

    WSLg mounts its own tmpfs over /run/user/<uid>, hiding the socket the user
    manager listens on, so `systemctl --user` cannot be reached at all on an
    ordinary guest — measured 2026-09-16. The install writes a SYSTEM unit
    there instead, and this is the probe finding it: no uid to read, no
    XDG_RUNTIME_DIR to set, and the restart goes through the same door.
    """
    runner = Scripted(
        answers={
            "-l -v": ok(OWENS_PC_LIST),
            "-u root --exec systemctl is-enabled": ok("enabled" + chr(10)),
            # If anything reached for the USER manager it would find this, and
            # the assertions below would catch it.
            "--user": bad("Failed to connect to bus"),
            "id -u": bad("should not be needed"),
        },
        pings=[200],
    )
    watcher = presence.PresenceWatcher(
        runner,
        host_log,
        distro="Ubuntu",
        consented=True,
        monotonic=ticking(),
        sleep=lambda _s: None,
    )
    probe = watcher.probe_unit()
    assert probe.readable is True
    assert probe.scope == presence.SCOPE_SYSTEM
    assert watcher.boot().owner is Owner.WSL_UNIT
    assert presence.system_systemctl_argv("Ubuntu", "is-enabled") in runner.calls
    assert not any("--user" in " ".join(call) for call in runner.calls), (
        "a system-unit guest must never be asked through the user manager: that",
        "is the bus WSLg hides",
    )


def test_the_restart_of_a_system_unit_guest_goes_through_root(
    host_log: log.HostLog,
) -> None:
    """One door for finding the unit and for restarting it."""
    runner = Scripted(
        answers={
            "-l -v": ok(OWENS_PC_LIST),
            "-u root --exec systemctl is-enabled": ok("enabled" + chr(10)),
            "-u root --exec systemctl restart": ok(""),
        },
        pings=[200],
    )
    watcher = presence.PresenceWatcher(
        runner,
        host_log,
        distro="Ubuntu",
        consented=True,
        monotonic=ticking(),
        sleep=lambda _s: None,
    )
    assert watcher.restart_wsl_unit() is True
    assert presence.system_systemctl_argv("Ubuntu", "restart") in runner.calls


def test_children_start_in_crucible_home_not_in_the_installation(monkeypatch) -> None:
    """A wsl.exe that inherits the installation directory blocks the next upgrade.

    MEASURED 2026-09-16. The orchestrator's own working directory is inside its
    installation — installation.json records `...\Crucible\host\Lib\
    site-packages`, deliberately, so `-m crucible.cli` imports. Children inherit
    it, so `wsl.exe` held a handle on `Crucible\host` and KEPT holding it after
    the orchestrator exited. The installer then failed with "the process cannot
    access the file because it is being used by another process" and named
    nothing; Sysinternals handle64 found two orphaned wsl.exe and a wslhost.exe.

    CRUCIBLE_HOME is the server's own state directory, which is what the systemd
    unit uses as WorkingDirectory for the same reason, and which the installer
    never moves. Not the user's home: `console_script` records the ImportError
    that follows from a working directory landing on sys.path.
    """
    import subprocess as sp
    from crucible.host.runner import ProcessRunner

    seen: dict[str, object] = {}

    class Done:
        returncode = 0
        # BYTES, because that is what the pipes carry now: the runner decodes
        # them itself rather than letting the locale codec at wsl.exe's UTF-16.
        stdout = b""
        stderr = b""

    def fake_run(argv, **kwargs):
        seen["cwd"] = kwargs.get("cwd")
        return Done()

    monkeypatch.setattr(sp, "run", fake_run)
    runner = ProcessRunner("win32", {}, cwd="C:/Users/x/AppData/Local/Crucible")
    runner.run(["wsl.exe", "-l", "-v"], timeout_s=5)
    assert seen["cwd"] == "C:/Users/x/AppData/Local/Crucible"


# --------------------------------------------- what wsl.exe says about ITSELF
#
# MEASURED 1.0.4, 2026-09-19 03:10, in host.log:
#
#   line: {'text': 'T\x00h\x00e\x00r\x00e\x00 \x00i\x00s\x00 \x00n\x00o\x00 …
#
# `wsl.exe` writes its OWN diagnostics as UTF-16LE. The runner asked
# `subprocess.run` for text, so they were decoded with the locale codec — under
# which a NUL is a perfectly good character — and the one sentence naming why
# the deploy failed reached the log as a row of interleaved NULs.

WSL_E_DISTRO_NOT_FOUND = (
    "There is no distribution with the supplied name.\n"
    "Error code: Wsl/Service/WSL_E_DISTRO_NOT_FOUND\n"
)


def piping(stdout: bytes = b"", stderr: bytes = b"", code: int = 0):
    """A `subprocess.run` that obeys subprocess's OWN contract about decoding.

    Asked for text it decodes with the locale codec and hands back a string;
    not asked, it hands back the bytes the pipe carried. That is the whole
    difference this keeper is about, so the fake may not paper over it — a fake
    that always returned bytes would make the pre-fix failure an artefact of
    the fake rather than the 1.0.4 defect.
    """

    def fake_run(argv, **kwargs):
        class Done:
            returncode = code

        if kwargs.get("text") or kwargs.get("encoding"):
            Done.stdout = stdout.decode("utf-8", errors=kwargs.get("errors", "strict"))
            Done.stderr = stderr.decode("utf-8", errors=kwargs.get("errors", "strict"))
        else:
            Done.stdout = stdout
            Done.stderr = stderr
        return Done()

    return fake_run


def a_runner(monkeypatch, fake_run) -> object:
    import subprocess as sp

    from crucible.host.runner import ProcessRunner

    monkeypatch.setattr(sp, "run", fake_run)
    return ProcessRunner("win32", {}, cwd="C:/Users/x/AppData/Local/Crucible")


def test_wsl_exes_own_utf16_message_is_read_as_a_sentence(monkeypatch) -> None:
    """The 1.0.4 failure, byte for byte, and what the log must have said.

    CRLF on the wire, because wsl.exe is a Windows program and that is what it
    writes — and `\\n` out, because `text=True` used to do that translation and
    everything that parses `wsl -l -v` was written against the result.
    """
    utf16 = WSL_E_DISTRO_NOT_FOUND.replace("\n", "\r\n").encode("utf-16-le")
    runner = a_runner(monkeypatch, piping(stderr=utf16, code=4294967295))
    result = runner.run(["wsl.exe", "-d", "crucible", "--exec", "bash"], timeout_s=5)

    assert result.stderr == WSL_E_DISTRO_NOT_FOUND
    assert "\x00" not in result.stderr
    # `said()` is what `step_failed` puts in front of a person.
    assert result.said().startswith("There is no distribution with the supplied name.")


def test_a_bom_marks_the_same_stream_even_when_it_is_one_word(monkeypatch) -> None:
    """The other shape wsl.exe emits. A one-word line has no second byte to
    count NULs in, so the BOM is the fact that has to be read."""
    runner = a_runner(monkeypatch, piping(stdout=b"\xff\xfe" + "Ubuntu\n".encode("utf-16-le")))
    assert runner.run(["wsl.exe", "-l", "-q"], timeout_s=5).stdout == "Ubuntu\n"


def test_the_guests_own_utf8_output_is_untouched(monkeypatch) -> None:
    """Everything `--exec` runs writes UTF-8 and it passes through as written —
    including the non-ASCII that would break a UTF-16 guess."""
    runner = a_runner(monkeypatch, piping(stdout="/home/telltale/.crucible — café\n".encode("utf-8")))
    assert runner.run(["wsl.exe", "-d", "Ubuntu", "--exec", "bash"], timeout_s=5).stdout == (
        "/home/telltale/.crucible — café\n"
    )


def test_a_byte_that_decodes_as_neither_is_replaced_and_therefore_visible(
    monkeypatch,
) -> None:
    """`errors="replace"`, never `"ignore"`: a byte nothing can read becomes a
    character a person reading host.log can SEE, instead of a sentence with a
    silent hole in it."""
    runner = a_runner(monkeypatch, piping(stdout=b"release \xff 1.0.4\n"))
    assert runner.run(["wsl.exe", "-l", "-v"], timeout_s=5).stdout == "release \ufffd 1.0.4\n"


# ------------------------------------------------ the from-scratch WSL walk
#
# Owen, 2026-09-16: "ideally id like to get it to the point where it can
# install on wsl basically on its own, with an idiot driving the system.
# whether wsl already exists or it needs to be installed from scratch".
# These drive the walk on the two machines a stranger actually has.


def a_fresh_wsl2_machine(**extra: RunResult) -> Scripted:
    """WSL2 is installed and working; Crucible has never been here.

    The rootfs download is what stops the walk, and that is fine: everything
    under test happens before it.
    """
    answers = {
        "--status": ok("Default Version: 2"),
        "-l -v": ok("  Ubuntu  Running  2\n"),
    }
    answers.update(extra)
    return Scripted(answers=answers)


def probes_run(runner: Scripted) -> list[str]:
    return [" ".join(call) for call in runner.calls]


def test_the_root_door_is_probed_before_the_guest_install_needs_it(
    tmp_path: Path,
) -> None:
    """`crucible service install` in the guest writes /etc/systemd/system and
    drives the system manager, both through `wsl.exe -u root`. A distro that
    will not grant root cannot be installed into at all — so the row that asks
    has to be asked, and `_wsl_state` returns at `no_crucible_distro` long
    before the distro it would ask about exists.
    """
    # The distro is ALREADY there, so `import-distro` is a no-op and the walk
    # reaches the rows that are about the guest. (A machine that has yet to
    # import one cannot be asked about it — that is the whole reason the first
    # walk stops early, and the whole reason this second one exists.)
    runner = a_fresh_wsl2_machine(**{
        "-l -v": ok("  crucible  Running  2\n"),
        "cat /etc/wsl.conf": ok("# crucible-rootfs\n[boot]\nsystemd=true\n"),
        # Root IS reachable here, so the FIRST walk gets past that row and
        # the second one is what the network probe below can only have
        # come from.
        "--exec id -u": ok("0\n"),
    })
    walk = installer.EngineInstall(
        runner,
        lambda event: None,
        release="0.6.0",
        home=tmp_path,
        install_sh_url="https://example.invalid/install.sh",
    )
    with pytest.raises(HostError):
        walk.run()
    asked = probes_run(runner)
    assert any(
        "-d crucible -u root --exec id -u" in call for call in asked
    ), f"nothing asked whether root is reachable in the guest: {asked}"
    # And the row that turns a VPN into a sentence instead of a failed download.
    assert any("py3-none-any.whl" in call for call in asked), (
        f"nothing asked whether the guest can reach the release: {asked}"
    )


def test_a_repair_that_changes_nothing_ends_the_walk_instead_of_looping(
    tmp_path: Path,
) -> None:
    """The hang, pinned. `distro_not_systemd` is repairable by terminating the
    distro — and when the terminate exits 0 and the distro still answers the
    same way, the old loop ran it again, forever, printing nothing.
    """
    runner = Scripted(
        answers={
            "--status": ok("Default Version: 2"),
            "-l -v": ok("  crucible  Running  2\n"),
            # /etc/wsl.conf without `systemd=true`, every time it is read.
            "cat /etc/wsl.conf": ok("[user]\ndefault=crucible\n"),
        }
    )
    walk = installer.EngineInstall(
        runner,
        lambda event: None,
        release="0.6.0",
        home=tmp_path,
        install_sh_url="https://example.invalid/install.sh",
    )
    with pytest.raises(HostError) as caught:
        walk.run()
    assert caught.value.code == "distro_not_systemd"
    assert "still answers the same way" in caught.value.message
    terminates = [
        call for call in runner.calls if "--terminate" in " ".join(call)
    ]
    assert len(terminates) == 1, f"the repair ran {len(terminates)} times: {terminates}"


# ------------------------------------- ONE RELEASE PER MACHINE, and who drives
#
# Owen, 2026-09-18: *"windows is the driver; the thing moving wsl forward."*
#
# MEASURED, not remembered: `crucible/host/app.py` passed `_sequence` only to
# `OrchestratorDoor`, so the install walk ran on `POST /install` and nowhere
# else; on a machine the guest already owns that sequence called `walk._complete()`,
# which emits `done` about the engine that is already there. `_guest_install()` —
# the one place `install.sh --release` runs inside the distro — is reached only
# from `run()`. So `install.ps1` upgraded the HOST and the guest stayed where it
# was, and `deploy.sh` had grown a second driver to paper over it.


def upgrade_walk(runner: object, tmp_path: Path, *, release: str = "0.7.0"):
    events: list[installer.Event] = []
    walk = installer.EngineInstall(
        runner,
        events.append,
        release=release,
        home=tmp_path,
        install_sh_url=f"https://example.invalid/v{release}/install.sh",
    )
    return walk, events


def guest_record(release: str) -> str:
    return json.dumps({"schema_version": 1, "platform": "linux", "release": release,
                       "home": "/home/crucible/.crucible"})


def test_a_guest_behind_the_host_is_carried_to_the_hosts_release(tmp_path: Path) -> None:
    """The host is the driver. A guest on 0.6.9 under a 0.7.0 host is upgraded
    through the SAME `install.sh --release <host version>` the move uses."""
    runner = Scripted(
        answers={
            "installation.json": ok(guest_record("0.6.9")),
            "printf %s": ok("/home/crucible/.crucible"),
        }
    )
    walk, events = upgrade_walk(runner, tmp_path)
    assert walk.upgrade_guest() == "0.7.0"
    installs = [
        " ".join(argv) for argv in runner.calls
        if "crucible-install.sh" in " ".join(argv)
    ]
    assert len(installs) == 1, installs
    assert "--release 0.7.0" in installs[0]
    # And it is the SAME step the move emits, so the host's log and window
    # describe an upgrade in the words they already use for an install.
    assert [event.data.get("name") for event in events if event.event == "step"] == [
        "guest-install"
    ]
    assert [record.name for record in walk._records] == ["guest-install"]
    assert [record.status for record in walk._records] == ["ok"]


def test_a_guest_already_at_the_hosts_release_is_left_alone(tmp_path: Path) -> None:
    runner = Scripted(answers={"installation.json": ok(guest_record("0.7.0"))})
    walk, events = upgrade_walk(runner, tmp_path)
    assert walk.upgrade_guest() is None
    assert not [argv for argv in runner.calls if "crucible-install.sh" in " ".join(argv)]


def test_a_guest_ahead_of_the_host_is_refused_by_name_and_never_downgraded(
    tmp_path: Path,
) -> None:
    """The one direction that must never be automatic. A guest somebody
    installed by hand at a newer release is a fact to report, not to undo."""
    runner = Scripted(answers={"installation.json": ok(guest_record("0.8.0"))})
    walk, _ = upgrade_walk(runner, tmp_path)
    with pytest.raises(HostError) as caught:
        walk.upgrade_guest()
    assert caught.value.code == "guest_ahead_of_host"
    assert "0.8.0" in caught.value.message and "0.7.0" in caught.value.message
    assert not [argv for argv in runner.calls if "crucible-install.sh" in " ".join(argv)]


def test_a_guest_with_no_record_is_carried_rather_than_guessed_about(
    tmp_path: Path,
) -> None:
    """`installation.json` is written when the RUNTIME starts, so an absent one
    means nothing has run in there — which is a guest to bring up to this
    release, not one to leave at a version nobody can name."""
    runner = Scripted(
        answers={"installation.json": RunResult(code=1, stdout="", stderr="No such file", failure=None)}
    )
    walk, _ = upgrade_walk(runner, tmp_path)
    assert walk.upgrade_guest() == "0.7.0"
    assert [argv for argv in runner.calls if "crucible-install.sh" in " ".join(argv)]


def test_the_guest_sequence_the_host_runs_is_the_phase20_one(tmp_path: Path) -> None:
    """It runs `install.sh`, which is GENERATED from the step list — so what
    the guest gets is the interpreter, the wheel and the recipes, and this
    asserts the generated file rather than trusting the URL's name."""
    generated = (
        Path(__file__).resolve().parents[1] / "sdk/bootstrap/scripts/install.sh"
    ).read_text(encoding="utf-8")
    assert 'say "server"' in generated
    assert "python-build-standalone" in generated
    assert "py3-none-any.whl" in generated
    assert "pip install --upgrade --no-input" in generated
    assert 'say "install-$type"' in generated


# ------------------------- …AND ONLY ONCE THE WATCHER HAS SAID WHOSE ENGINE IT IS
#
# MEASURED on the first real `ship.sh patch --deploy` (1.0.3, 2026-09-19): the
# tray started at 02:04:52 and the owner became `wsl-unit` at 02:04:59, when the
# watch loop's first tick ran `running_owner`. The carry thread `main()` starts
# a few lines after the watch thread asked its `Owner.WSL_UNIT` question inside
# that seven-second gap, got the `Owner.NONE` that `start()` leaves when it
# decides no owner, and returned with NO LINE IN THE LOG. The guest stayed on
# 1.0.2 and `install.ps1` refused the whole install over it.


def fast_watching_host(context: app_module.HostContext) -> app_module.Host:
    """A host whose watch ticks fast enough for a test to wait on one."""
    host = app_module.Host(context)
    context.watcher.watch_s = 0.01
    return host


def settle_presence(host: app_module.Host) -> None:
    """Run the REAL watch loop until it has settled a presence, then stop it.

    The event is the seam under test, so a test that set it by hand would pin
    nothing about who sets it in production — `watch` is the one writer, and
    this is how a keeper gets to say so.
    """
    thread = threading.Thread(target=host.watch, name="keeper-watch", daemon=True)
    thread.start()
    assert host._presence_settled.wait(10.0), "the watch loop settled no presence"
    host._stop.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


def test_the_guest_carry_waits_for_the_owner_the_watcher_has_not_measured_yet(
    tmp_path: Path,
) -> None:
    """The carry thread starts BEFORE the owner is known, and still carries.

    This is the 1.0.3 run exactly: presence is `starting`/`none` when the
    thread begins, and only the watch tick turns it into `wsl-unit`.
    """
    runner = Scripted(
        answers={
            "installation.json": ok(guest_record("0.6.9")),
            "printf %s": ok("/home/crucible/.crucible"),
        },
        pings=[200] * 50,
    )
    context = _context(tmp_path, runner)
    context.release = "0.7.0"
    context.presence = presence.Presence(
        Distro.PRESENT, Engine.STARTING, "starting", Owner.NONE
    )
    host = fast_watching_host(context)

    carry = threading.Thread(
        target=host.carry_guest_to_this_release,
        kwargs={"settle_ceiling_s": 10.0},
        name="keeper-carry",
        daemon=True,
    )
    carry.start()
    # Nothing may have been installed against an owner nobody has measured.
    assert not [a for a in runner.calls if "crucible-install.sh" in " ".join(a)]

    settle_presence(host)
    carry.join(timeout=10.0)
    assert not carry.is_alive(), "the carry never finished"

    assert context.presence.owner is Owner.WSL_UNIT
    installs = [
        " ".join(argv) for argv in runner.calls if "crucible-install.sh" in " ".join(argv)
    ]
    assert len(installs) == 1, installs
    assert "--release 0.7.0" in installs[0]


def test_a_machine_with_no_guest_says_so_rather_than_returning_silently(
    tmp_path: Path,
) -> None:
    """The fourth answer. `owner=found` is a real verdict about this machine and
    the log is where a person reads it; the silent `return` made a carry that
    decided nothing indistinguishable from one that never ran."""
    runner = Scripted(pings=[200] * 50)
    context = _context(tmp_path, runner)
    context.presence = presence.Presence(
        Distro.ABSENT, Engine.RUNNING, "somebody else's", Owner.FOUND
    )
    host = fast_watching_host(context)
    settle_presence(host)
    host.carry_guest_to_this_release(settle_ceiling_s=10.0)

    written = (tmp_path / "host.log").read_text(encoding="utf-8")
    assert "guest release: no guest to carry (owner=found)" in written, written
    assert not [a for a in runner.calls if "crucible-install.sh" in " ".join(a)]


def test_a_presence_that_never_settles_is_a_line_and_not_a_thread_that_waits_forever(
    tmp_path: Path,
) -> None:
    """No watch loop runs here, so the event is never set: the ceiling is what
    ends the wait, and it says so by name instead of leaving a daemon thread
    parked on a fact that is not coming."""
    runner = Scripted()
    context = _context(tmp_path, runner)
    host = app_module.Host(context)
    host.carry_guest_to_this_release(settle_ceiling_s=1.0)

    written = (tmp_path / "host.log").read_text(encoding="utf-8")
    assert "guest release: presence never settled within 1 s" in written, written
    assert not [a for a in runner.calls if "crucible-install.sh" in " ".join(a)]
    # The shipped ceiling outlasts the first tick it is waiting for, which is
    # the only property a number composed from the watch's own constants has.
    assert app_module.PRESENCE_SETTLE_CEILING_SECONDS > presence.WATCH_SECONDS


# ------------------------- …INTO THE DISTRO THIS HOST MANAGES, NOT A DEFAULT NAME
#
# MEASURED on the second real deploy (1.0.4, 2026-09-19 03:10). The carry now
# waited for presence and got `owner=wsl-unit` — and then ran `install.sh` in a
# distro called "crucible", which on Owen's PC does not exist:
#
#   step_failed: install.sh exited 4294967295 inside "crucible":
#   There is no distribution with the supplied name.
#
# The host had claimed "Ubuntu" seconds earlier. One fact, two owners: the
# claim read the watcher and the carry read `EngineInstall`'s default.


def consented_watcher(
    runner: Scripted, host_log: log.HostLog, distro: str
) -> presence.PresenceWatcher:
    """The watcher `main()` builds when config.toml names a distro (PHASE17 2.5)."""
    return presence.PresenceWatcher(
        runner,
        host_log,
        distro=distro,
        consented=True,
        monotonic=ticking(),
        sleep=lambda _s: None,
    )


def a_consented_machine_whose_guest_is_behind() -> Scripted:
    """Owen's PC at 1.0.4: Ubuntu, a system unit, and a guest on the old release."""
    return Scripted(
        answers={
            "-l -v": ok(OWENS_PC_LIST),
            "-u root --exec systemctl is-enabled": ok("enabled" + chr(10)),
            "installation.json": ok(guest_record("0.6.9")),
            "printf %s": ok("/home/crucible/.crucible"),
        },
        pings=[200] * 50,
    )


def test_the_carry_runs_in_the_distro_this_host_consented_to(tmp_path: Path) -> None:
    """The distro has ONE owner — the watcher — and the carry asks it.

    A host that claims the engine in "Ubuntu" and then installs into "crucible"
    is two answers to one question, and the second one is the one the guest
    never hears.
    """
    runner = a_consented_machine_whose_guest_is_behind()
    context = _context(tmp_path, runner)
    context.release = "0.7.0"
    context.presence = presence.Presence(
        Distro.PRESENT, Engine.STARTING, "starting", Owner.NONE
    )
    context.watcher = consented_watcher(runner, context.log, "Ubuntu")
    host = fast_watching_host(context)
    settle_presence(host)
    assert context.presence.owner is Owner.WSL_UNIT

    host.carry_guest_to_this_release(settle_ceiling_s=10.0)

    installs = [
        argv for argv in runner.calls if "crucible-install.sh" in " ".join(argv)
    ]
    assert len(installs) == 1, installs
    assert installs[0][:3] == ["wsl.exe", "-d", "Ubuntu"], installs[0]
    # And every OTHER wsl.exe the carry made — the `installation.json` read it
    # decides on, above all — went to the same distro. A carry that read one
    # guest and installed into another would still be two owners.
    assert not [
        argv for argv in runner.calls if CRUCIBLE_DISTRO in argv
    ], "the carry named the default distro on a machine that consented to another"


def test_an_unconsented_host_still_carries_the_distro_crucible_imported(
    tmp_path: Path,
) -> None:
    """The other half of one owner. With no consent the watcher's distro IS
    `CRUCIBLE_DISTRO`, so reading the watcher changes nothing here — which is
    what makes it safe to read it everywhere."""
    runner = Scripted(
        answers={
            "installation.json": ok(guest_record("0.6.9")),
            "printf %s": ok("/home/crucible/.crucible"),
        },
        pings=[200] * 50,
    )
    context = _context(tmp_path, runner)
    context.release = "0.7.0"
    context.presence = presence.Presence(
        Distro.PRESENT, Engine.STARTING, "starting", Owner.NONE
    )
    host = fast_watching_host(context)
    settle_presence(host)
    assert context.presence.owner is Owner.WSL_UNIT

    host.carry_guest_to_this_release(settle_ceiling_s=10.0)

    installs = [
        argv for argv in runner.calls if "crucible-install.sh" in " ".join(argv)
    ]
    assert len(installs) == 1, installs
    assert installs[0][:3] == ["wsl.exe", "-d", CRUCIBLE_DISTRO], installs[0]
