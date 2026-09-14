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
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pytest

from crucible.host import door as door_module
from crucible.host import installer, landoor, log, menu, paths, presence, startup, wslstate
from crucible.host.errors import HOST_ERROR_CODES, HostError
from crucible.host.menu import Distro, Engine
from crucible.host.runner import RunResult
from crucible.host.wsl_states import WSL_STATE_CODES, WSL_STATES

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
    """4.1's table, walked EXHAUSTIVELY. Fifteen cells, not a sample."""
    seen: set[str] = set()
    for distro in Distro:
        for engine in Engine:
            model = menu.menu_model(distro, engine)
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
    running = menu.menu_model(Distro.ABSENT, Engine.RUNNING)
    assert running.title == "Crucible — running (llama-windows)"
    assert "host mode" not in running.title


def test_quits_label_says_what_quitting_costs_and_it_differs_by_server() -> None:
    assert menu.menu_model(Distro.PRESENT, Engine.RUNNING).item(menu.QUIT).label == (
        "Quit (the engine keeps running)"
    )
    assert menu.menu_model(Distro.ABSENT, Engine.RUNNING).item(menu.QUIT).label == (
        "Quit (stops the engine)"
    )


def test_an_unreadable_wsl_still_offers_the_install_and_never_claims_a_server() -> None:
    """Reading `unknown` as `absent` would import a SECOND distro."""
    model = menu.menu_model(Distro.UNKNOWN, Engine.RUNNING)
    assert model.item(menu.INSTALL_ENGINE) is not None
    assert "WSL unreadable" in model.title


def test_the_install_item_reads_as_an_upgrade_not_as_an_absence() -> None:
    label = menu.menu_model(Distro.ABSENT, Engine.RUNNING).item(menu.INSTALL_ENGINE).label
    assert label.startswith("Install the WSL2 engine")
    assert "TTS" in label


# ------------------------------------------------------------- 4.1 presence


def test_the_boot_recipe_and_the_two_recovery_recipes_are_exactly_4_1s() -> None:
    assert presence.wsl_boot_argv() == ["wsl.exe", "-d", "crucible", "--exec", "true"]
    assert presence.recipe_argv(presence.RECIPE_USER_UNIT_START) == [
        "wsl.exe", "-d", "crucible", "--exec", "systemctl", "--user", "start", "crucible",
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
        assert "--exec" in presence.recipe_argv(name)
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
    runner = Scripted(answers={"-l -v": ok("  crucible  Stopped  2\n")}, pings=[])
    watcher = presence.PresenceWatcher(
        runner, host_log, boot_wait_s=2.0, monotonic=ticking(), sleep=lambda _s: None
    )
    result = watcher.boot()
    assert result.engine is Engine.FAILED
    assert "both recipes were spent" in result.detail
    ran = [" ".join(call) for call in runner.calls]
    assert any("systemctl --user start crucible" in line for line in ran)
    assert any("systemctl restart user@1000" in line for line in ran)


def test_the_watch_spends_ONE_recovery_per_down_edge_and_then_stops(
    host_log: log.HostLog,
) -> None:
    """4.1: it never loops on restart; the unit's own `Restart=` does that."""
    runner = Scripted(answers={"-l -v": ok("  crucible  Running  2\n")}, pings=[])
    watcher = presence.PresenceWatcher(
        runner, host_log, monotonic=ticking(), sleep=lambda _s: None
    )
    first = watcher.poll(Distro.PRESENT)
    assert first.engine is Engine.STOPPED
    recoveries = sum("systemctl" in " ".join(call) for call in runner.calls)
    runner.calls.clear()
    second = watcher.poll(Distro.PRESENT)
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
    watcher.poll(Distro.ABSENT)  # down: spends the budget
    watcher.poll(Distro.ABSENT)  # up: restores it
    runner.calls.clear()
    third = watcher.poll(Distro.ABSENT)
    assert third.engine is Engine.STOPPED


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
    assert "-m crucible.cli host" in script
    assert argv[0] == "powershell.exe"
    assert "WScript.Shell" in script


def test_install_startup_is_idempotent_because_CreateShortcut_rewrites() -> None:
    runner = Scripted()
    first = startup.install(runner)
    second = startup.install(runner)
    assert first.path == second.path
    assert runner.calls[0] == runner.calls[1]


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
    assert state.code == "pack_disk"
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
    assert door.open is True
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


# ------------------------------------------------------------ 4.3 installer


def test_the_sequence_is_4_7s_steps_in_4_7s_order() -> None:
    assert installer.STEPS == (
        "wsl-state",
        "import-distro",
        "guest-install",
        "migrate-config",
        "install-job-types",
        "migrate-weights",
        "lan-door",
        "stop-windows-server",
        "switch-pairing",
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
    assert "reboot, then Crucible continues" in caught.value.message
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
    assert caught.value.code == "no_crucible_distro"
    assert "rootfs" in caught.value.message


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


def test_the_door_refuses_a_wrong_bearer_and_a_missing_one(host_log: log.HostLog) -> None:
    door = door_module.InstallDoor(host_log, lambda _emit: None, token=lambda: "right")
    assert door.authorised("Bearer right") is True
    assert door.authorised("Bearer wrong") is False
    assert door.authorised(None) is False
    assert door.authorised("right") is False


def test_no_config_yet_is_host_no_token_and_not_an_authorisation_failure(
    host_log: log.HostLog,
) -> None:
    door = door_module.InstallDoor(host_log, lambda _emit: None, token=lambda: None)
    with pytest.raises(HostError) as caught:
        door.authorised("Bearer anything")
    assert caught.value.code == "host_no_token"


def test_one_install_on_a_machine(host_log: log.HostLog) -> None:
    door = door_module.InstallDoor(host_log, lambda _emit: None, token=lambda: "t")
    assert door.claim() is True
    assert door.claim() is False, "host_install_running"
    door.release()
    assert door.claim() is True


def test_the_door_is_loopback_and_an_argument_cannot_put_it_on_the_lan(
    host_log: log.HostLog,
) -> None:
    door = door_module.InstallDoor(host_log, lambda _emit: None, token=lambda: "t")
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

    door = door_module.InstallDoor(host_log, sequence, token=lambda: "tok")
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

    door = door_module.InstallDoor(host_log, lambda _emit: None, token=lambda: "tok")
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

    door = door_module.InstallDoor(host_log, sequence, token=lambda: "tok")
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


def test_every_other_verb_keeps_its_windows_refusal_until_3_5_lands(
    capsys, monkeypatch
) -> None:
    """The win32 gate NARROWED to one opt-in flag rather than opening.

    Letting every verb through before the `llama-windows` backend exists
    would replace one honest refusal with a `NoViableBackend` from somewhere
    deeper — the same "no" with a worse sentence and a stack trace.
    """
    from crucible import cli

    monkeypatch.setattr(cli.sys, "platform", "win32")
    assert cli.main(["doctor"]) == cli.EXIT_REFUSED
    assert "PHASE15" in capsys.readouterr().err


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
