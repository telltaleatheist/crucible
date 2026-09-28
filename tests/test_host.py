from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pytest

from crucible.host import app as app_module
from crucible.host import catalog as catalog_module
from crucible.host import door as door_module
from crucible.host import installer, landoor, log, move_policy, outcome, paths, presence, startup, wslstate
from crucible.host.catalog import CatalogRefusal, Subject
from crucible.host.errors import HOST_ERROR_CODES, HostError
from crucible.host.menu import Distro, Engine, Owner
from crucible.host.runner import RunResult
from crucible.host.wsl_states import CRUCIBLE_DISTRO, WSL_STATE_CODES, WSL_STATES

WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32",
    reason="the install window and the Windows host pack exist only on a Windows host",
)

WINDOWS_ENV = {
    "LOCALAPPDATA": r"C:\Users\tellt\AppData\Local",
    "APPDATA": r"C:\Users\tellt\AppData\Roaming",
    "USERPROFILE": r"C:\Users\tellt",
    "USERNAME": "tellt",
}


@dataclass
class Scripted:

    answers: dict[str, RunResult] = field(default_factory=dict)
    pings: list[int | None] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)
    gets: list[str] = field(default_factory=list)
    spawned: list[list[str]] = field(default_factory=list)
    downloads: list[tuple[str, str, int]] = field(default_factory=list)
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

    def stream(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        on_line: Callable[[str, str], None],
        env: Mapping[str, str] | None = None,
    ) -> RunResult:
        result = self.run(argv, timeout_s=timeout_s, env=env)
        for line in result.stdout.splitlines():
            on_line(line, "stdout")
        for line in result.stderr.splitlines():
            on_line(line, "stderr")
        return result

    def download(
        self,
        url: str,
        destination: Path,
        *,
        timeout_s: float,
        on_progress: Callable[[int, int | None, str], None] | None = None,
        attempts: int = 1,
    ) -> RunResult:
        self.downloads.append((url, str(destination), attempts))
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"")
        if on_progress is not None:
            on_progress(0, None, Path(destination).name)
        return RunResult(code=0, stdout=str(destination), stderr="", failure=None)

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


def test_every_wsl_call_uses_exec_so_wsl_exe_cannot_pre_expand_a_variable() -> None:
    assert "--exec" in presence.wsl_boot_argv()


def test_the_numbers_4_1_states_are_constants_with_4_1s_names() -> None:
    assert presence.BOOT_WAIT_SECONDS == 30
    assert presence.WATCH_SECONDS == 15


def test_wsl_list_is_parsed_by_the_row_shape_and_not_by_a_localised_header() -> None:
    text = "  NAME              STATE           VERSION\n* Ubuntu            Running         2\n  crucible          Stopped         2\n"
    assert presence.parse_wsl_list(text) == ["Ubuntu", "crucible"]
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


def test_the_watch_spends_ONE_recovery_per_down_edge_and_then_stops(
    host_log: log.HostLog,
) -> None:
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
    runner = Scripted(pings=[401])
    watcher = presence.PresenceWatcher(runner, host_log, sleep=lambda _s: None)
    assert watcher.ping() is True
    assert runner.gets == ["http://127.0.0.1:7100/v1/ping"]


def test_a_successful_ping_restores_the_recovery_budget(host_log: log.HostLog) -> None:
    runner = Scripted(pings=[None, 200, None])
    watcher = presence.PresenceWatcher(
        runner, host_log, monotonic=ticking(), sleep=lambda _s: None
    )
    watcher.poll(Distro.ABSENT, Owner.HOST_CHILD)
    watcher.poll(Distro.ABSENT, Owner.HOST_CHILD)
    runner.calls.clear()
    third = watcher.poll(Distro.ABSENT, Owner.HOST_CHILD)
    assert third.engine is Engine.STOPPED


OWENS_PC_LIST = "  NAME      STATE           VERSION\n* Ubuntu    Running         2\n"
GUEST_LINE = "crucible://crucible%40owens-pc-wsl@127.0.0.1:7100/#a-token\n"


def test_the_hunt_only_asks_distros_that_are_ALREADY_running() -> None:
    assert presence.wsl_running_argv() == ["wsl.exe", "-l", "-v", "--running"]
    assert "--exec" in presence.guest_pairing_argv("Ubuntu")
    assert "${CRUCIBLE_HOME:-$HOME/.crucible}" in " ".join(
        presence.guest_pairing_argv("Ubuntu")
    )


def test_a_pairing_lines_authority_is_read_after_the_LAST_at_sign() -> None:
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
    assert runner.spawned == []
    assert not any("--exec true" in " ".join(call) for call in runner.calls)


def test_an_engine_answering_somewhere_else_is_not_this_machines(
    host_log: log.HostLog,
) -> None:
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
    assert presence.keepalive_argv("Ubuntu") == [
        "wsl.exe", "-d", "Ubuntu", "--exec", "sleep", "infinity",
    ]
    runner = Scripted()
    watcher = presence.PresenceWatcher(runner, host_log, sleep=lambda _s: None)
    first = watcher.hold("Ubuntu")
    assert runner.spawned == [presence.keepalive_argv("Ubuntu")]
    assert watcher.hold("Ubuntu") is first
    assert len(runner.spawned) == 1
    first.terminate()
    again = watcher.rehold()
    assert again is not first
    assert len(runner.spawned) == 2
    watcher.release()
    assert watcher.held is None
    assert watcher.rehold() is None


def real_grantee_env() -> dict[str, str]:
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
    assert not any("serve" in " ".join(call) for call in runner.spawned)
    assert not any("init" in " ".join(call) for call in runner.calls)
    assert presence.keepalive_argv("Ubuntu") in runner.spawned


def test_with_nothing_answering_and_no_distro_the_host_mode_child_still_starts(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / paths.CONSOLE_CMD).write_text("@echo off\n", encoding="utf-8")
    env = dict(WINDOWS_ENV)
    env["LOCALAPPDATA"] = str(tmp_path)
    runner = Scripted(answers={"-l -v": ok(OWENS_PC_LIST)}, pings=[None], env=env)
    context = _context(tmp_path, runner)
    result = app_module.Host(context).start()
    assert result.owner is Owner.NONE
    assert result.engine is Engine.FAILED
    assert paths.INSTALL_ONE_LINER in result.detail


def test_a_guest_engines_pairing_line_is_COPIED_and_never_composed(
    tmp_path: Path,
) -> None:
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
    monkeypatch.setenv("CRUCIBLE_HOME", "overwritten by the login item")
    monkeypatch.setattr(sys, "argv", [])
    calls = []
    monkeypatch.setattr(runpy, "run_module", lambda module, **kwargs:
                        calls.append((module, kwargs, os.environ["CRUCIBLE_HOME"], list(sys.argv))))
    exec(startup.startup_python(env), {})
    monkeypatch.delenv("CRUCIBLE_HOME")
    assert calls == [("crucible.cli", {"run_name": "__main__"}, home, ["crucible", "local", "tray"])]


def test_the_remove_script_is_powershell_that_parses(monkeypatch) -> None:
    script = startup.remove_argv(WINDOWS_ENV)[-1]
    assert "}}" not in script and "{{" not in script
    assert script.count("{") == script.count("}") == 2
    assert script.startswith("if (Test-Path ")
    assert "} else { Write-Output 'absent' }" in script


def test_the_install_script_is_powershell_that_parses() -> None:
    script = startup.install_argv(WINDOWS_ENV)[-1]
    assert "}}" not in script and "{{" not in script


def _install_ps1() -> str:
    root = Path(__file__).resolve().parent.parent
    return (root / "sdk" / "bootstrap" / "scripts" / "install.ps1").read_text(encoding="utf-8")


def test_install_ps1_ends_by_READING_the_outcome_and_never_by_asserting_one() -> None:
    script = _install_ps1()
    assert "'-m', 'crucible.host.installwatch'" in script
    assert "'--since', $Began" in script
    assert "$Watch += '--brief'" in script
    assert 'Say "Crucible is ready in your notification area."' in script
    assert "Say \"Crucible is ready in your notification area. The Windows engine works now" not in script
    assert "wsl --install" not in script
    assert script.count("{") == script.count("}")


def test_remove_startup_says_whether_there_was_one() -> None:
    there = Scripted(default=ok("removed\n"))
    assert startup.remove(there).changed is True
    absent = Scripted(default=ok("absent\n"))
    outcome = startup.remove(absent)
    assert outcome.changed is False
    assert "nothing to remove" in outcome.detail


def test_every_generated_state_code_has_a_predicate_and_no_others() -> None:
    assert set(wslstate.MEANS) == set(WSL_STATE_CODES)
    assert len(WSL_STATE_CODES) == len(set(WSL_STATE_CODES))


def test_the_generated_table_kept_4cs_order_deepest_cause_first() -> None:
    codes = list(WSL_STATE_CODES)
    assert codes.index("virtualization_disabled") < codes.index("wsl_missing")
    assert codes[-1] == "wsl_ready", "the last row must be total"


def test_every_row_says_whether_the_tray_can_carry_it_and_agrees_with_its_action() -> None:
    carried = {"run", "run-elevated"}
    for state in WSL_STATES:
        assert isinstance(state.automatic, bool), f"{state.code} has no partition"
        if state.code == "wsl_ready":
            assert state.automatic is True
            continue
        assert state.automatic is (state.action_kind in carried), (
            f"{state.code} says automatic={state.automatic} and its action is "
            f"{state.action_kind!r}"
        )
    assert any(state.automatic for state in WSL_STATES)
    assert any(not state.automatic for state in WSL_STATES)


def test_a_detected_state_carries_the_partition_so_a_caller_never_re_derives_it() -> None:
    runner = Scripted(answers={"--status": bad("not recognized", code=None)})
    state = wslstate.detect(runner, release="1.0.5")
    assert state.code == "wsl_missing"
    assert state.automatic is True, "an elevated action is one the tray carries"


def test_no_sentinel_survived_the_generation() -> None:
    for state in WSL_STATES:
        blob = state.sentence + state.action_text + state.action_url + " ".join(state.probe_argv)
        for sentinel in ("XXSAID", "XXAPPDISTRO", "XXGUESTUSER", "424.242.424", "424242"):
            assert sentinel not in blob, f"{state.code} carries {sentinel}"


def test_virtualization_is_answered_before_wsl_is_called_missing() -> None:
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
            "-u root --exec id -u": ok("0\n"),
        }
    )
    state = wslstate.detect(runner, release="0.6.0")
    assert state.code == "wsl_ready"
    assert "{said}" not in state.sentence


def test_the_costly_rows_are_not_probed_unless_the_caller_asks() -> None:
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
        automatic=False,
        optional=False,
    )
    monkeypatch.setattr(wslstate, "WSL_STATES", (extra,) + generated.WSL_STATES)
    with pytest.raises(HostError) as caught:
        wslstate.detect(Scripted(), release="0.6.0")
    assert caught.value.code == "wsl_state_unknown"


def test_the_ubuntu_image_download_emits_pulls_own_byte_shape(tmp_path: Path) -> None:
    events: list[installer.Event] = []

    class Downloading(Scripted):
        def download(self, url, destination, *, timeout_s, on_progress=None, attempts=1):
            self.downloads.append((url, str(destination), attempts))
            assert on_progress is not None, "the step asked for no bytes at all"
            on_progress(1 << 20, 356515840, Path(destination).name)
            on_progress(356515840, 356515840, Path(destination).name)
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            Path(destination).write_bytes(b"")
            return RunResult(code=0, stdout=str(destination), stderr="", failure=None)

    runner = Downloading(
        answers={
            "-l -v": ok("  NAME   STATE   VERSION\n"),
            "SHA256SUMS": ok("deadbeef  ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz\n"),
            "certutil": ok("nope\n"),
        }
    )
    walk = installer.EngineInstall(
        runner, events.append, release="1.0.5", home=tmp_path,
        install_sh_url="https://example.invalid/install.sh",
    )
    with pytest.raises(HostError):
        walk._import_distro()
    url, _destination, attempts = runner.downloads[0]
    assert url.endswith("ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz")
    assert attempts == installer.IMAGE_DOWNLOAD_ATTEMPTS == 3, "curl's --retry 3, kept"
    progress = [event for event in events if event.event == "progress"]
    assert progress, "the image downloaded and nothing said how far it had got"
    assert set(progress[0].data) == {"bytes_done", "bytes_total", "file"}, (
        "the shape is `pull`'s, in crucible/tasks.py"
    )
    assert progress[-1].data["bytes_done"] == progress[-1].data["bytes_total"]


def test_install_sh_emits_the_progress_wire_interpreter_py_parses() -> None:
    from crucible.interpreter import PROGRESS_PREFIX, parse_progress_line

    root = Path(__file__).resolve().parent.parent
    script = (root / "sdk" / "bootstrap" / "scripts" / "install.sh").read_text(encoding="utf-8")
    line = next(
        (raw.strip() for raw in script.splitlines() if PROGRESS_PREFIX in raw),
        None,
    )
    assert line is not None, "install.sh emits no progress line for the interpreter fetch"
    emitted = (
        line.split("'", 1)[1].rsplit("'", 1)[0]
        .replace("\\n", "")
        .replace("%s", "1", 1)
        .replace("%s", "2", 1)
        .replace("%s", "cpython.tar.gz", 1)
    )
    assert parse_progress_line(emitted) == {
        "bytes_done": 1,
        "bytes_total": 2,
        "file": "cpython.tar.gz",
    }


def test_a_guest_progress_line_becomes_a_progress_EVENT_not_a_log_line(tmp_path: Path) -> None:
    from crucible.interpreter import progress_line

    events: list[installer.Event] = []
    walk = installer.EngineInstall(
        Scripted(), events.append, release="1.0.5", home=tmp_path,
        install_sh_url="https://example.invalid/install.sh",
    )
    walk._line(progress_line(17, 100, "cpython.tar.gz"))
    walk._line("server: python 3.11.16 from python-build-standalone")
    assert [event.event for event in events] == ["progress", "line"]
    assert events[0].data == {"bytes_done": 17, "bytes_total": 100, "file": "cpython.tar.gz"}


def test_the_guest_install_streams_its_lines_rather_than_collecting_them(
    tmp_path: Path,
) -> None:
    events: list[installer.Event] = []
    seen_before_exit: list[str] = []

    class Streaming(Scripted):
        def stream(self, argv, *, timeout_s, on_line, env=None):
            self.calls.append(list(argv))
            on_line("Collecting torch==2.13.0", "stdout")
            seen_before_exit.append(
                "progress" if any(e.event == "line" for e in events) else "nothing"
            )
            on_line("Successfully installed torch", "stdout")
            return RunResult(code=0, stdout="", stderr="", failure=None)

    walk = installer.EngineInstall(
        Streaming(), events.append, release="1.0.5", home=tmp_path,
        install_sh_url="https://example.invalid/install.sh",
    )
    walk._guest_install()
    assert seen_before_exit == ["progress"], (
        "the first line reached the stream before the process had exited"
    )
    assert [event.data["text"] for event in events if event.event == "line"] == [
        "Collecting torch==2.13.0",
        "Successfully installed torch",
    ]


def test_the_network_probe_names_every_index_a_recipe_names() -> None:
    import re as _re

    from crucible import jobenv

    urls = wslstate.install_index_urls("1.0.5")
    assert urls[0] == "https://pypi.org/simple", "a recipe with no index still uses one"
    named: set[str] = set()
    for directory in jobenv.recipe_roots():
        for recipe in directory.glob("*.txt"):
            for line in recipe.read_text(encoding="utf-8").splitlines():
                found = _re.match(
                    r"^\s*(?:--index-url|--extra-index-url|-f|--find-links)[=\s]+(\S+)\s*$", line
                )
                if found is not None:
                    named.add(found.group(1))
    assert named, "no recipe in this build names an index; the probe would prove nothing"
    assert named <= set(urls), f"not probed: {sorted(named - set(urls))}"
    assert any("huggingface" in url for url in urls), "the weights come from somewhere"
    assert any("python-build-standalone" in url for url in urls), "so does the interpreter"
    assert any(url.endswith("crucible-1.0.5-py3-none-any.whl") for url in urls)


def test_the_probe_runs_one_HEAD_per_index_and_names_the_FIRST_it_cannot_reach() -> None:
    runner = Scripted(
        answers={
            "--status": ok("WSL version: 2.3.26.0\nDefault Version: 2\n"),
            "-l -v": ok("  NAME        STATE           VERSION\n* crucible    Running         2\n"),
            "/etc/wsl.conf": ok("# crucible-rootfs\n[boot]\nsystemd=true\n"),
            "for u in": bad("https://download.pytorch.org/whl/cu128 could not be reached"),
        }
    )
    state = wslstate.detect(runner, release="1.0.5", check_network=True)
    assert state.code == "guest_no_network"
    assert "download.pytorch.org" in state.sentence, "the sentence names the one that failed"
    probe = next(call for call in runner.calls if "for u in" in " ".join(call))
    script = probe[-1]
    assert "{indexes}" not in script, "the placeholder was filled from the recipes"
    assert "curl -fsSL -I -m 20" in script, "one cheap HEAD each"
    for url in wslstate.install_index_urls("1.0.5"):
        assert url in script, f"{url} is not probed"


def test_reading_a_machines_facts_still_costs_nothing_it_was_not_asked_for() -> None:
    runner = Scripted(answers={"--status": bad("not recognized")})
    state = wslstate.detect(runner, release="1.0.5")
    assert state.code == "wsl_missing"
    assert not any("for u in" in " ".join(call) for call in runner.calls)


def test_an_outcome_round_trips_every_field_2_2_names(tmp_path: Path) -> None:
    written = outcome.write(
        tmp_path,
        state=outcome.CANNOT,
        code="virtualization_disabled",
        sentence="Windows cannot start a virtual machine: no",
        release="1.0.5",
        attempts=1,
        now=lambda: "2026-09-19T00:00:00+00:00",
    )
    assert (tmp_path / "wsl-outcome.json").is_file()
    read_back = outcome.read(tmp_path)
    assert read_back == written
    assert read_back is not None
    assert read_back.to_dict() == {
        "state": "cannot",
        "code": "virtualization_disabled",
        "sentence": "Windows cannot start a virtual machine: no",
        "at": "2026-09-19T00:00:00+00:00",
        "release": "1.0.5",
        "attempts": 1,
        "restarts": 0,
    }


def test_a_machine_that_never_recorded_one_reads_None_and_never_a_blank(
    tmp_path: Path,
) -> None:
    assert outcome.read(tmp_path) is None


@pytest.mark.parametrize(
    "document",
    [
        "{not json",
        '["a list"]',
        '{"state": "sideways", "at": "x", "release": "1.0.5", "attempts": 0}',
        '{"state": "done", "at": "x", "release": "1.0.5", "attempts": "two"}',
        '{"state": "done", "at": "x", "release": "", "attempts": 0}',
        '{"state": "done", "at": "", "release": "1.0.5", "attempts": 0}',
        '{"state": "failed", "at": "x", "release": "1.0.5", "attempts": 1, "code": 7}',
    ],
)
def test_a_present_outcome_that_cannot_be_read_is_REFUSED_and_never_treated_as_absent(
    tmp_path: Path, document: str
) -> None:
    (tmp_path / "wsl-outcome.json").write_text(document, encoding="utf-8")
    with pytest.raises(HostError) as caught:
        outcome.read(tmp_path)
    assert caught.value.code == "wsl_outcome_invalid"
    assert caught.value.code in HOST_ERROR_CODES


def test_a_state_nobody_defined_is_refused_at_the_WRITE(tmp_path: Path) -> None:
    with pytest.raises(HostError) as caught:
        outcome.write(tmp_path, state="nearly", release="1.0.5", attempts=0)
    assert caught.value.code == "wsl_outcome_invalid"
    assert not (tmp_path / "wsl-outcome.json").exists()


def test_the_classifier_reads_the_TABLES_partition_and_keeps_no_list_of_its_own() -> None:
    for row in WSL_STATES:
        expected = outcome.FAILED if row.automatic else outcome.CANNOT
        assert outcome.classify(row.code) == expected, row.code
    assert outcome.classify("wsl_reboot_required") == outcome.REBOOT_PENDING
    assert outcome.classify("wsl_reboot_again") == outcome.CANNOT
    assert outcome.classify("rootfs_download_failed") == outcome.FAILED
    assert outcome.classify("step_failed") == outcome.FAILED


def test_a_reboot_demand_past_the_budget_is_terminal_and_says_twice(tmp_path: Path) -> None:
    events: list[installer.Event] = []
    runner = Scripted(answers={"--status": bad("not recognized")})
    walk = installer.EngineInstall(
        runner,
        events.append,
        release="1.0.5",
        home=tmp_path,
        install_sh_url="https://example.invalid/install.sh",
        restarts=installer.RESTART_BUDGET,
        rebooted=True,
    )
    with pytest.raises(HostError) as caught:
        walk.run()
    assert caught.value.code == "wsl_reboot_again"
    assert "restarted several times" in caught.value.message
    assert "Windows Update" in caught.value.message
    assert '"Try again"' in caught.value.message
    assert outcome.classify(caught.value.code) == outcome.CANNOT


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
    door = _door(forward=True, firewall=False)
    assert door.forward is True and door.firewall is False
    assert door.open is False
    assert "drops the connection" in door.detail


def test_both_rows_on_a_private_network_is_the_only_open_door() -> None:
    door = _door(forward=True, firewall=True)
    assert door.open is True
    assert door.private_network is True


def test_both_rows_on_a_public_only_network_is_not_called_open() -> None:
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
    assert "[string]" in " ".join(landoor.connection_profile_argv())
    assert landoor.has_private_network('[{"NetworkCategory":"Private"}]') is True
    assert landoor.has_private_network('[{"NetworkCategory":"Public"}]') is False
    assert landoor.has_private_network('[{"NetworkCategory":1}]') is None
    assert landoor.has_private_network("") is None
    assert landoor.has_private_network("not json") is None
    assert landoor.has_private_network('{"NetworkCategory":"Private"}') is True


def test_the_sequence_is_4_7s_steps_in_4_7s_order() -> None:
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
    assert "needs to restart" in caught.value.message
    assert "carries on by itself" in caught.value.message
    assert "press Install" not in caught.value.message
    assert not (tmp_path / "wsl-reboot-pending").exists()
    assert outcome.classify(caught.value.code) == outcome.REBOOT_PENDING
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
    assert "SHA256SUMS" in caught.value.message
    assert "ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz" in caught.value.message


def test_every_event_a_step_emits_is_shaped_like_a_tasks_py_event(tmp_path: Path) -> None:
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
    assert payload["backend"] == "cuda-linux"
    assert payload["steps"][0]["status"] == "ok"
    assert json.loads(json.dumps(payload)) == payload


@dataclass
class FakeOrchestrator:

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

    where: Path | None = None
    seen: dict = field(
        default_factory=lambda: {
            "distro": "absent",
            "engine": "running",
            "owner": "child",
            "detail": "the Windows engine",
        }
    )

    def presence(self) -> dict:
        return dict(self.seen)

    def install_outcome(self) -> dict | None:
        if self.where is None:
            return None
        recorded = outcome.read(self.where)
        return None if recorded is None else recorded.to_dict()


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


def _post(port: int, path: str, *, bearer: str | None) -> tuple[int, dict]:
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


def test_crucible_host_is_refused_off_win32_by_name(capsys, monkeypatch) -> None:
    from crucible import cli

    monkeypatch.setattr(cli.orchestrator.sys, "platform", "linux")
    code = cli.main(["orchestrator"])
    assert code == cli.common.EXIT_REFUSED
    said = capsys.readouterr().err
    assert "host_windows_only" in said
    assert "systemd" in said


def test_there_is_no_platform_gate_left_in_main(monkeypatch, tmp_path: Path) -> None:
    from crucible import cli
    from crucible.errors import ConfigError, NoViableBackend

    parser = cli.build_parser()
    assert "win32_ok" not in vars(parser.parse_args(["doctor"]))
    assert "win32_ok" not in vars(parser.parse_args(["orchestrator"]))

    monkeypatch.setattr(cli.orchestrator.sys, "platform", "win32")
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path))
    monkeypatch.setattr(
        cli.common, "detect_backend", lambda: (_ for _ in ()).throw(
            NoViableBackend("no card in this test")
        )
    )
    monkeypatch.setattr(
        cli.common, "load_config", lambda *a, **k: (_ for _ in ()).throw(
            ConfigError("no config here")
        )
    )
    assert cli.main(["doctor"]) == cli.common.EXIT_REFUSED


def test_a_cuda_linux_config_on_a_windows_host_is_backend_not_here(capsys) -> None:
    from crucible import cli
    from crucible.backend import Backend, Gpu

    windows = Backend(
        kind="llama-windows",
        platform="windows",
        arch="AMD64",
        gpu=Gpu(vendor="nvidia", name="RTX 4090", vram_bytes=24 * 1024**3),
        detail="llama.cpp cuda build",
    )
    said = cli.common._backend_mismatch("cuda-linux", windows)
    assert said.startswith("backend_not_here: ")
    assert "cuda-linux" in said and "llama-windows" in said
    assert "WSL2" in said

    mac = Backend(
        kind="mlx-darwin",
        platform="darwin",
        arch="arm64",
        gpu=Gpu(vendor="apple", name="M1 Ultra", vram_bytes=64 * 1024**3),
        detail="mlx",
    )
    other = cli.common._backend_mismatch("llama-windows", mac)
    assert other.startswith("backend_not_here: ")
    assert "do not run on win32" not in other


def test_config_from_takes_the_token_the_routes_and_the_upstreams(tmp_path: Path) -> None:
    from crucible.cli.init import carried_from

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
    from crucible.cli.init import carried_from
    from crucible.errors import ConfigError

    path = tmp_path / "config.toml"
    path.write_text('[server]\nname = "x"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        carried_from(path)
    assert "config_from_no_token" in str(caught.value)


def test_config_from_that_is_not_toml_is_refused_by_name(tmp_path: Path) -> None:
    from crucible.cli.init import carried_from
    from crucible.errors import ConfigError

    path = tmp_path / "config.toml"
    path.write_text("this is not toml {{{", encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        carried_from(path)
    assert "config_from_unreadable" in str(caught.value)


def test_write_config_copies_a_carried_table_verbatim(tmp_path: Path) -> None:
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
        enable_denoise=False,
        retention_days=7,
        desktop_allowance_basis="stated",
        desktop_allowance_note="",
        carried_tables={
            "routes": {"translate": "anthropic/claude-sonnet-5"},
            "upstreams": {"anthropic": {"key": "sk-ant-x", "a_field_from_the_future": 1}},
        },
    )
    document = tomllib.loads(written.read_text(encoding="utf-8"))
    assert document["routes"] == {"translate": "anthropic/claude-sonnet-5"}
    assert document["upstreams"]["anthropic"]["a_field_from_the_future"] == 1
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
            enable_denoise=False,
            retention_days=7,
            desktop_allowance_basis="stated",
            desktop_allowance_note="",
            carried_tables={"auth": {"token": "somebody-elses"}},
        )
    assert "two writers" in str(caught.value)


class FakeCatalog:

    def __init__(self, where: str, installed: Sequence[tuple[str, str]] = ()) -> None:
        self._where = where
        self.subjects: list[Subject] = [
            Subject(kind=kind, id=ident, name=ident, installed=True)
            for kind, ident in installed
        ]
        self.calls: list[str] = []
        self.in_use: dict[tuple[str, str], str] = {}
        self.release_after: dict[tuple[str, str], int] = {}
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
    windows = FakeCatalog("windows", [("model", "qwen3.5-9b"), ("voice", "mistborn")])
    guest = FakeCatalog("guest")
    guest.pull_latency = 2
    events: list[installer.Event] = []
    migration(windows, guest, events, tmp_path)._migrate_weights()

    for subject in ("model qwen3.5-9b", "voice mistborn"):
        assert f"pull {subject}" in guest.calls
        assert f"remove {subject}" in windows.calls
        pulled = guest.calls.index(f"pull {subject}")
        removed = windows.calls.index(f"remove {subject}")
        assert pulled < len(guest.calls)
        assert removed >= 0
    assert windows.subjects == [], "the Windows copies are gone"
    assert {row.key for row in guest.subjects} == {
        ("model", "qwen3.5-9b"),
        ("voice", "mistborn"),
    }


def test_a_subject_the_guest_ALREADY_has_is_not_pulled_again(tmp_path: Path) -> None:
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
        catalog_module.parse_catalog({"rows": [{"kind": "model", "id": "a", "installed": "false"}]}, "guest")


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
    host._cleanup.running = True
    guest.unreachable = True
    host.resume_model_cleanup()
    assert (tmp_path / installer.CLEANUP_RECORD).exists()
    assert windows.subjects and not host._cleanup.running
    guest.unreachable = False
    host.resume_model_cleanup()
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
    monkeypatch.setattr(host, "resume_model_cleanup", resumed)
    monkeypatch.setattr(installer.EngineInstall, "complete", lambda self: calls.append("complete"))
    def wrong_source(*args, **kwargs):
        raise AssertionError("The active guest cannot be constructed as a native source")
    monkeypatch.setattr(app_module, "HttpCatalog", wrong_source)
    monkeypatch.setattr(move_policy, "HttpCatalog", wrong_source)
    app_module._sequence(context, host)(lambda event: None)
    assert calls == ["resume native cleanup", "complete"]


def test_an_interrupted_move_resumes_from_the_two_catalogs(tmp_path: Path) -> None:
    windows = FakeCatalog("windows", [("model", "a"), ("voice", "b")])
    guest = FakeCatalog("guest", [("model", "a")])
    events: list[installer.Event] = []
    migration(windows, guest, events, tmp_path)._migrate_weights()
    assert guest.calls.count("pull voice b") == 1
    assert not any(call == "pull model a" for call in guest.calls)
    assert windows.subjects == []
    windows.calls.clear()
    guest.calls.clear()
    migration(windows, guest, events, tmp_path)._migrate_weights()
    assert not any(call.startswith(("pull", "remove")) for call in guest.calls + windows.calls)


def test_subject_in_use_is_WAITED_OUT_and_never_skipped(tmp_path: Path) -> None:
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


def test_a_catalog_row_missing_a_field_is_refused_rather_than_half_read() -> None:
    with pytest.raises(HostError) as caught:
        catalog_module.parse_catalog({"rows": [{"kind": "model"}]}, "the guest")
    assert caught.value.code == "catalog_unreadable"
    assert "'id'" in caught.value.message or "'installed'" in caught.value.message


def test_a_catalog_that_is_not_a_catalog_is_refused_by_name() -> None:
    with pytest.raises(HostError) as caught:
        catalog_module.parse_catalog({"packs": []}, "the guest")
    assert caught.value.code == "catalog_unreadable"


def test_the_servers_refusal_code_survives_verbatim_with_its_holder() -> None:
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
    guest = catalog_module.GuestCatalog(Scripted(), "crucible", "tok", 7100, where="the guest")
    argv = guest.curl_argv("POST", "/v1/tasks", '{"type":"pull","kind":"model","id":"a"}')
    assert argv[:5] == ["wsl.exe", "-d", "crucible", "--exec", "curl"]
    assert "-f" not in argv, "-f would hide the refusal body, and the CODE is the point"
    assert '{"type":"pull","kind":"model","id":"a"}' in argv
    assert argv[-1] == "http://127.0.0.1:7100/v1/tasks"
    assert f"Authorization: Bearer tok" in argv


def test_the_guest_port_reads_the_status_curl_appended() -> None:
    runner = Scripted(
        default=ok('{"rows": [{"kind": "model", "id": "a", "installed": true}]}'
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

    def fake_urlopen(request, timeout):
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


class FakeEngine:

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

            def do_GET(self) -> None:
                if not self._authorised():
                    return
                if self.path == "/v1/info":
                    if engine.info_document is None:
                        self._send(503, {"error": {"code": "unavailable", "message": "no"}})
                        return
                    self._send(200, engine.info_document)
                    return
                self._send(404, {"error": {"code": "not_found", "message": "no"}})

            def do_POST(self) -> None:
                if not self._authorised():
                    return
                engine.claims.append(self._read())
                self._send(200, {"role": "engine", "managed_by": {}, "claimed": "now"})

            def do_DELETE(self) -> None:
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
    context = _context(tmp_path, runner)
    context.name = "crucible-orchestrator@test"
    context.presence = presence.Presence(Distro.PRESENT, Engine.RUNNING, "up", owner)
    monkeypatch.setattr(app_module, "engine_url", lambda path="": f"{engine.url}{path}")
    monkeypatch.setattr(app_module, "engine_token", lambda _c: engine.token)
    return app_module.Host(context)


def test_an_orchestrator_claims_the_engine_it_started(tmp_path: Path, monkeypatch) -> None:
    for owner in (Owner.WSL_UNIT, Owner.HOST_CHILD):
        with FakeEngine() as engine:
            host = _orchestrator(tmp_path, Scripted(), owner, engine, monkeypatch)
            assert host.claim() is True, owner
            assert len(engine.claims) == 1
            said = engine.claims[0]["orchestrator"]
            assert said["name"] == "crucible-orchestrator@test"
            assert said["url"] == paths.door_url("")
            assert said["version"]
            assert "force" not in engine.claims[0]


def test_a_FOUND_engine_is_NEVER_claimed(tmp_path: Path, monkeypatch) -> None:
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
    with FakeEngine(token="a-different-token") as engine:
        host = _orchestrator(tmp_path, Scripted(), Owner.WSL_UNIT, engine, monkeypatch)
        monkeypatch.setattr(app_module, "engine_token", lambda _c: "the-wrong-one")
        assert host.claim() is False
    assert "peer_token_mismatch" in (tmp_path / "host.log").read_text(encoding="utf-8")


def test_quit_releases_the_claim_while_the_engine_is_still_answering(
    tmp_path: Path, monkeypatch
) -> None:
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


def _a_quitting_host(
    tmp_path: Path, owner: Owner, engine: FakeEngine, monkeypatch
) -> tuple[app_module.Host, Scripted]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    runner = Scripted()
    host = _orchestrator(tmp_path, runner, owner, engine, monkeypatch)
    host.claim()
    host._c.watcher.hold("Ubuntu")
    host._c.watcher.child = FakeChild()
    runner.calls.clear()
    return host, runner


def test_a_quit_that_holds_a_claim_RELEASES_it(tmp_path: Path, monkeypatch) -> None:
    with FakeEngine() as engine:
        host, _runner = _a_quitting_host(tmp_path, Owner.WSL_UNIT, engine, monkeypatch)
        assert host._claimed is True
        door_module.OrchestratorDoor(
            host._c.log, lambda _emit: None, token=lambda: "tok", orchestrator=host
        ).quit()
        assert len(engine.releases) == 1
        assert engine.releases[0]["orchestrator"]["url"] == paths.door_url("")
        assert host._shutdown_complete.is_set()


def test_a_FOUND_engines_orchestrator_releases_NOTHING_and_still_stops(
    tmp_path: Path, monkeypatch
) -> None:
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
        assert host._shutdown_complete.is_set()
    assert "owner=found" in (tmp_path / "host.log").read_text(encoding="utf-8")


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
        engine.info_document["capabilities"] = [
            {"job_type": "llm", "models": [{"id": "qwen3.5-9b"}, {"id": "dots-ocr"}]}
        ]
        assert host.info()["capabilities"] == engine.info_document["capabilities"]


def test_an_engine_that_cannot_be_read_is_an_empty_list_and_a_NULL_name(
    tmp_path: Path, monkeypatch
) -> None:
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
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, Scripted(), Owner.HOST_CHILD, engine, monkeypatch)
        assert host.info()["engine"]["owner"] == "child"


def test_a_found_engine_is_refused_engine_not_ours_BEFORE_anything_happens(
    tmp_path: Path, monkeypatch
) -> None:
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, Scripted(), Owner.FOUND, engine, monkeypatch)
        with pytest.raises(HostError) as caught:
            host.check_restartable()
        assert caught.value.code == "engine_not_ours"
        with pytest.raises(HostError):
            host.restart_engine(lambda _event: None)


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
    with FakeEngine() as engine:
        host = _orchestrator(
            tmp_path,
            Scripted(answers={"is-enabled": ok("enabled\n")}, pings=[200]),
            Owner.WSL_UNIT,
            engine,
            monkeypatch,
        )
        host.claim()
        assert len(engine.claims) == 1
        host.restart_engine(lambda _event: None)
        assert len(engine.claims) == 2, "the restarted engine was told again"


def test_the_door_answers_ping_WITHOUT_a_bearer_and_info_WITH_one(
    host_log: log.HostLog,
) -> None:
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
    from crucible import peer as peer_module

    assert set(app_module.OWNER_ON_THE_WIRE.values()) == set(peer_module.OWNERS)
    assert Owner.NONE not in app_module.OWNER_ON_THE_WIRE, "an absence is not an owner"


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
    (tmp_path / "config.toml").write_text(
        '[auth]\ntoken = "the-token"\n\n[orchestrator]\ndistro = "Ubuntu"\n',
        encoding="utf-8",
    )
    assert app_module.consented_distro(tmp_path) == "Ubuntu"
    assert app_module.read_token(tmp_path) == "the-token"


def test_a_setting_that_is_present_and_unusable_is_REFUSED_not_ignored(
    tmp_path: Path,
) -> None:
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
    assert not any("is-enabled" in " ".join(call) for call in runner.calls)


def test_consent_makes_the_named_distro_PRESENT(host_log: log.HostLog) -> None:
    runner = Scripted(answers={"-l -v": ok(OWENS_PC_LIST)})
    watcher = _consented_watcher(runner, host_log)
    distro, detail = watcher.probe_distro()
    assert distro is Distro.PRESENT
    assert "Ubuntu" in detail


def test_consent_with_an_unreadable_unit_stays_found_and_says_why(
    tmp_path: Path,
) -> None:
    host_log = log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1")
    runner = Scripted(
        answers={
            "-l -v --running": ok(OWENS_PC_LIST),
            "-l -v": ok(OWENS_PC_LIST),
            "cat ": ok(GUEST_LINE),
            "id -u": ok("1000\n"),
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


def test_a_consented_machine_is_not_refused_engine_not_ours(tmp_path: Path) -> None:
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
    with FakeEngine() as engine:
        host = _orchestrator(tmp_path, Scripted(), Owner.WSL_UNIT, engine, monkeypatch)
        assert host.claim() is True
        assert len(engine.claims) == 1
        assert "force" not in engine.claims[0]


def test_a_socket_that_is_truly_absent_is_still_found_with_the_reason(
    tmp_path: Path,
) -> None:
    host_log = log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1")
    runner = Scripted(
        answers={
            "-l -v --running": ok(OWENS_PC_LIST),
            "-l -v": ok(OWENS_PC_LIST),
            "cat ": ok(GUEST_LINE),
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


def test_an_unreadable_unit_leaves_the_owner_found_and_runs_no_recipe(
    tmp_path: Path,
) -> None:
    host_log = log.HostLog(tmp_path / "host.log", tmp_path / "host.log.1")
    runner = Scripted(
        answers={
            "-l -v --running": ok(OWENS_PC_LIST),
            "-l -v": ok(OWENS_PC_LIST),
            "cat ": ok(GUEST_LINE),
            "is-enabled": bad("no such distribution"),
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
    assert "no system crucible.service" in probe.detail
    assert "no such distribution" in probe.detail
    assert watcher.boot().owner is Owner.FOUND
    assert not any(
        verb in " ".join(call)
        for call in runner.calls
        for verb in ("restart", " start ", "user@1000")
    )


def test_a_system_unit_guest_is_brought_up_by_its_own_manager(
    host_log: log.HostLog,
) -> None:
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
    assert watcher.recover() is True
    assert presence.system_systemctl_argv("Ubuntu", "start") in runner.calls
    assert not any("--user" in " ".join(call) for call in runner.calls), (
        "a system-unit guest must not be recovered through the user manager"
    )


def test_an_ownerless_host_does_not_blame_its_config(host_log: log.HostLog) -> None:
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
    runner = Scripted(pings=[200, 200])
    watcher = presence.PresenceWatcher(
        runner, host_log, distro="Ubuntu", sleep=lambda _s: None
    )
    seen = watcher.poll(Distro.ABSENT, Owner.NONE)
    assert seen.engine is Engine.RUNNING
    assert seen.owner is Owner.FOUND


def test_the_stop_of_a_system_unit_guest_goes_through_root(tmp_path: Path) -> None:
    runner = Scripted(
        answers={
            "-u root --exec systemctl is-enabled": ok("enabled" + chr(10)),
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
    assert not any("stop" in call for call in runner.calls), (
        "the engine must be untouched when the uid cannot be read"
    )
    assert "stop: NOT RUN" in (tmp_path / "host.log").read_text(encoding="utf-8")


def test_a_system_unit_guest_is_owned_and_restarted_as_root(
    host_log: log.HostLog,
) -> None:
    runner = Scripted(
        answers={
            "-l -v": ok(OWENS_PC_LIST),
            "-u root --exec systemctl is-enabled": ok("enabled" + chr(10)),
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
    assert watcher.boot().owner is Owner.WSL_UNIT
    assert presence.system_systemctl_argv("Ubuntu", "is-enabled") in runner.calls
    assert not any("--user" in " ".join(call) for call in runner.calls), (
        "a system-unit guest must never be asked through the user manager: that",
        "is the bus WSLg hides",
    )


def test_the_restart_of_a_system_unit_guest_goes_through_root(
    host_log: log.HostLog,
) -> None:
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
    import subprocess as sp
    from crucible.host.runner import ProcessRunner

    seen: dict[str, object] = {}

    class Done:
        returncode = 0
        stdout = b""
        stderr = b""

    def fake_run(argv, **kwargs):
        seen["cwd"] = kwargs.get("cwd")
        return Done()

    monkeypatch.setattr(sp, "run", fake_run)
    runner = ProcessRunner("win32", {}, cwd="C:/Users/x/AppData/Local/Crucible")
    runner.run(["wsl.exe", "-l", "-v"], timeout_s=5)
    assert seen["cwd"] == "C:/Users/x/AppData/Local/Crucible"


WSL_E_DISTRO_NOT_FOUND = (
    "There is no distribution with the supplied name.\n"
    "Error code: Wsl/Service/WSL_E_DISTRO_NOT_FOUND\n"
)


def piping(stdout: bytes = b"", stderr: bytes = b"", code: int = 0):

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
    utf16 = WSL_E_DISTRO_NOT_FOUND.replace("\n", "\r\n").encode("utf-16-le")
    runner = a_runner(monkeypatch, piping(stderr=utf16, code=4294967295))
    result = runner.run(["wsl.exe", "-d", "crucible", "--exec", "bash"], timeout_s=5)

    assert result.stderr == WSL_E_DISTRO_NOT_FOUND
    assert "\x00" not in result.stderr
    assert result.said().startswith("There is no distribution with the supplied name.")


def test_a_bom_marks_the_same_stream_even_when_it_is_one_word(monkeypatch) -> None:
    runner = a_runner(monkeypatch, piping(stdout=b"\xff\xfe" + "Ubuntu\n".encode("utf-16-le")))
    assert runner.run(["wsl.exe", "-l", "-q"], timeout_s=5).stdout == "Ubuntu\n"


def test_the_guests_own_utf8_output_is_untouched(monkeypatch) -> None:
    runner = a_runner(monkeypatch, piping(stdout="/home/telltale/.crucible — café\n".encode("utf-8")))
    assert runner.run(["wsl.exe", "-d", "Ubuntu", "--exec", "bash"], timeout_s=5).stdout == (
        "/home/telltale/.crucible — café\n"
    )


def test_a_byte_that_decodes_as_neither_is_replaced_and_therefore_visible(
    monkeypatch,
) -> None:
    runner = a_runner(monkeypatch, piping(stdout=b"release \xff 1.0.4\n"))
    assert runner.run(["wsl.exe", "-l", "-v"], timeout_s=5).stdout == "release \ufffd 1.0.4\n"


def a_fresh_wsl2_machine(**extra: RunResult) -> Scripted:
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
    runner = a_fresh_wsl2_machine(**{
        "-l -v": ok("  crucible  Running  2\n"),
        "cat /etc/wsl.conf": ok("# crucible-rootfs\n[boot]\nsystemd=true\n"),
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
    assert any("py3-none-any.whl" in call for call in asked), (
        f"nothing asked whether the guest can reach the release: {asked}"
    )


def test_a_repair_that_changes_nothing_ends_the_walk_instead_of_looping(
    tmp_path: Path,
) -> None:
    runner = Scripted(
        answers={
            "--status": ok("Default Version: 2"),
            "-l -v": ok("  crucible  Running  2\n"),
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
    runner = Scripted(
        answers={"installation.json": RunResult(code=1, stdout="", stderr="No such file", failure=None)}
    )
    walk, _ = upgrade_walk(runner, tmp_path)
    assert walk.upgrade_guest() == "0.7.0"
    assert [argv for argv in runner.calls if "crucible-install.sh" in " ".join(argv)]


def test_the_guest_sequence_the_host_runs_is_the_phase20_one(tmp_path: Path) -> None:
    generated = (
        Path(__file__).resolve().parents[1] / "sdk/bootstrap/scripts/install.sh"
    ).read_text(encoding="utf-8")
    assert 'say "server"' in generated
    assert "python-build-standalone" in generated
    assert "py3-none-any.whl" in generated
    assert "pip install --upgrade --no-input" in generated
    assert 'say "install-$type"' in generated


def fast_watching_host(context: app_module.HostContext) -> app_module.Host:
    host = app_module.Host(context)
    context.watcher.watch_s = 0.01
    return host


def settle_presence(host: app_module.Host) -> None:
    thread = threading.Thread(target=host.watch, name="keeper-watch", daemon=True)
    thread.start()
    assert host._presence_settled.wait(10.0), "the watch loop settled no presence"
    host._stop.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


def test_the_guest_carry_waits_for_the_owner_the_watcher_has_not_measured_yet(
    tmp_path: Path,
) -> None:
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
    runner = Scripted()
    context = _context(tmp_path, runner)
    host = app_module.Host(context)
    host.carry_guest_to_this_release(settle_ceiling_s=1.0)

    written = (tmp_path / "host.log").read_text(encoding="utf-8")
    assert "guest release: presence never settled within 1 s" in written, written
    assert not [a for a in runner.calls if "crucible-install.sh" in " ".join(a)]
    assert app_module.PRESENCE_SETTLE_CEILING_SECONDS > presence.WATCH_SECONDS


def consented_watcher(
    runner: Scripted, host_log: log.HostLog, distro: str
) -> presence.PresenceWatcher:
    return presence.PresenceWatcher(
        runner,
        host_log,
        distro=distro,
        consented=True,
        monotonic=ticking(),
        sleep=lambda _s: None,
    )


def a_consented_machine_whose_guest_is_behind() -> Scripted:
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
    assert not [
        argv for argv in runner.calls if CRUCIBLE_DISTRO in argv
    ], "the carry named the default distro on a machine that consented to another"


def test_an_unconsented_host_still_carries_the_distro_crucible_imported(
    tmp_path: Path,
) -> None:
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


def _native(tmp_path: Path, runner: Scripted, *, owner: Owner = Owner.HOST_CHILD,
            release: str = "1.0.5") -> app_module.HostContext:
    context = _context(tmp_path, runner)
    context.release = release
    context.presence = presence.Presence(
        Distro.ABSENT, Engine.RUNNING, "the Windows engine", owner
    )
    return context


def _decider(
    context: app_module.HostContext, sequence=None
) -> tuple[app_module.Host, list[str]]:
    ran: list[str] = []

    def default(emit: Callable[[installer.Event], None]) -> None:
        ran.append("the move ran")
        emit(installer.Event("step", {"name": "wsl-state", "index": 1, "total": 11}))
        emit(installer.Event("done", {}))

    host = app_module.Host(context)
    host._install_door = door_module.OrchestratorDoor(
        context.log,
        sequence or default,
        token=lambda: "t",
        orchestrator=FakeOrchestrator(),
    )
    return host, ran


def _recorded(tmp_path: Path, state: str, **fields) -> None:
    outcome.write(
        tmp_path,
        state=state,
        release=fields.pop("release", "1.0.5"),
        attempts=fields.pop("attempts", 1),
        **fields,
    )


def test_an_engine_this_orchestrator_did_not_start_is_NEVER_moved(tmp_path: Path) -> None:
    context = _native(tmp_path, Scripted(), owner=Owner.FOUND)
    host, ran = _decider(context)
    assert host.decide_engine() == "found"
    assert ran == []
    assert outcome.read(tmp_path) is None, "nothing was recorded about somebody else's engine"


def test_a_machine_that_declined_stays_native_and_says_so_once(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        '[orchestrator]\nwsl = "never"\n', encoding="utf-8"
    )
    context = _native(tmp_path, Scripted())
    host, ran = _decider(context)
    assert host.decide_engine() == "declined"
    assert ran == []
    recorded = outcome.read(tmp_path)
    assert recorded is not None and recorded.state == "declined"
    was = recorded.at
    assert host.decide_engine() == "declined"
    after = outcome.read(tmp_path)
    assert after is not None and after.at == was


def test_a_wsl_key_nobody_defined_is_refused_and_moves_nothing(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        '[orchestrator]\nwsl = "sometimes"\n', encoding="utf-8"
    )
    context = _native(tmp_path, Scripted())
    host, ran = _decider(context)
    assert host.decide_engine() == "unreadable"
    assert ran == [], "a setting this build cannot carry out is not a licence to move"


def test_cannot_stays_cannot_while_virtualization_is_still_off(tmp_path: Path) -> None:
    _recorded(tmp_path, "cannot", code="virtualization_disabled", sentence="VT-x is off")
    runner = Scripted(answers={"--status": bad("HCS_E_HYPERV_NOT_INSTALLED 0x80370102")})
    context = _native(tmp_path, runner)
    host, ran = _decider(context)
    assert host.decide_engine() == "cannot"
    assert ran == []
    assert any("--status" in call for call in runner.calls), "it looked again at this start"


def test_virtualization_turned_on_since_resumes_the_move_at_the_next_start(
    tmp_path: Path,
) -> None:
    _recorded(tmp_path, "cannot", code="virtualization_disabled", sentence="VT-x is off")
    context = _native(tmp_path, Scripted())
    host, ran = _decider(context)
    assert host.decide_engine() == "done"
    assert ran == ["the move ran"]


def test_a_cannot_that_is_not_the_firmware_is_never_retried(tmp_path: Path) -> None:
    _recorded(tmp_path, "cannot", code="wsl_blocked_by_policy", sentence="policy says no")
    runner = Scripted()
    context = _native(tmp_path, runner)
    host, ran = _decider(context)
    assert host.decide_engine() == "cannot"
    assert ran == []
    assert runner.calls == []


def test_a_failed_move_is_retried_ONCE_and_then_left_alone(tmp_path: Path) -> None:
    _recorded(tmp_path, "failed", code="rootfs_download_failed", sentence="the download died", attempts=1)
    context = _native(tmp_path, Scripted())
    host, ran = _decider(context)
    assert host.decide_engine() == "done"
    assert ran == ["the move ran"], "the first failure earns one retry"

    _recorded(tmp_path, "failed", code="rootfs_download_failed", sentence="again", attempts=2)
    context = _native(tmp_path, Scripted())
    host, again = _decider(context)
    assert host.decide_engine() == "failed"
    assert again == [], "a second consecutive failure stays failed until Try again"


def test_a_machine_with_no_outcome_at_all_is_MOVED_without_anybody_choosing(
    tmp_path: Path,
) -> None:
    context = _native(tmp_path, Scripted())
    host, ran = _decider(context)
    assert host.decide_engine() == "done"
    assert ran == ["the move ran"]


def test_an_unreadable_outcome_is_quarantined_and_the_decision_starts_from_nothing(
    tmp_path: Path,
) -> None:
    (tmp_path / "wsl-outcome.json").write_text("{not json", encoding="utf-8")
    context = _native(tmp_path, Scripted())
    host, ran = _decider(context)
    assert host.decide_engine() == outcome.DONE
    assert ran == ["the move ran"], "a record nobody can read no longer holds the move hostage"
    aside = list(tmp_path.glob("wsl-outcome.json.bad-*"))
    assert len(aside) == 1
    assert str(aside[0]) in context.log.path.read_text(encoding="utf-8")


def test_the_carry_thread_is_the_one_that_decides_and_it_waits_for_the_owner(
    tmp_path: Path,
) -> None:
    runner = Scripted(answers={"-l -v": bad("no distributions")})
    context = _native(tmp_path, runner)
    host = fast_watching_host(context)
    decisions: list[str] = []
    host.decide_engine = lambda: decisions.append("asked") or "found"
    settle_presence(host)
    host.carry_guest_to_this_release(settle_ceiling_s=10.0)
    assert decisions == ["asked"]


def _real_sequence_host(
    tmp_path: Path, runner: Scripted
) -> tuple[app_module.Host, app_module.HostContext]:
    context = _native(tmp_path, runner)
    host = app_module.Host(context)
    host._install_door = door_module.OrchestratorDoor(
        context.log,
        app_module._sequence(context, host),
        token=lambda: "t",
        orchestrator=FakeOrchestrator(),
    )
    return host, context


def test_a_machine_that_cannot_writes_cannot_with_the_tables_own_sentence(
    tmp_path: Path,
) -> None:
    runner = Scripted(
        answers={"--status": bad("HCS_E_HYPERV_NOT_INSTALLED 0x80370102")}
    )
    host, _context = _real_sequence_host(tmp_path, runner)
    assert host.decide_engine() == "cannot"
    recorded = outcome.read(tmp_path)
    assert recorded is not None
    assert recorded.state == "cannot"
    assert recorded.code == "virtualization_disabled"
    assert recorded.sentence is not None and "virtual machine" in recorded.sentence
    assert recorded.release == "1.0.5"
    assert not any("--import" in " ".join(call) for call in runner.calls)


def test_a_reboot_writes_reboot_pending_and_the_NEXT_start_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = Scripted(answers={"--status": bad("not recognized")})
    host, _context = _real_sequence_host(tmp_path, runner)
    assert host.decide_engine() == "reboot-pending"
    first = outcome.read(tmp_path)
    assert first is not None and first.state == "reboot-pending"
    assert first.code == "wsl_reboot_required"
    assert first.restarts == 1

    monkeypatch.setattr(wslstate, "booted_at", lambda: time.time() + 3600)
    for restarts in range(2, installer.RESTART_BUDGET + 1):
        again = Scripted(answers={"--status": bad("not recognized")})
        host_n, _cn = _real_sequence_host(tmp_path, again)
        assert host_n.decide_engine() == "reboot-pending"
        recorded = outcome.read(tmp_path)
        assert recorded is not None and recorded.code == "wsl_reboot_still_owed"
        assert recorded.restarts == restarts
    last = Scripted(answers={"--status": bad("not recognized")})
    host2, _c2 = _real_sequence_host(tmp_path, last)
    assert host2.decide_engine() == "cannot"
    second = outcome.read(tmp_path)
    assert second is not None and second.state == "cannot"
    assert second.code == "wsl_reboot_again"


def _get(port: int, path: str, *, bearer: str | None = "tok") -> tuple[int, object]:
    import urllib.error
    import urllib.request

    headers = {} if bearer is None else {"Authorization": f"Bearer {bearer}"}
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as refused:
        return refused.code, json.loads(refused.read().decode())


def test_get_install_answers_running_outcome_and_presence(
    host_log: log.HostLog, tmp_path: Path
) -> None:
    outcome.write(
        tmp_path, state="cannot", code="wsl1_only", sentence="WSL is version 1",
        release="1.0.5", attempts=1,
    )
    fake = FakeOrchestrator(where=tmp_path)
    door = a_door(host_log, token="tok", orchestrator=fake)
    server = door_module.serve(door, host="127.0.0.1", port=0)
    port = server.server_address[1]
    try:
        status, body = _get(port, "/install")
    finally:
        server.shutdown()
    assert status == 200
    assert isinstance(body, dict)
    assert body["running"] is False
    assert body["outcome"]["state"] == "cannot"
    assert body["outcome"]["code"] == "wsl1_only"
    assert body["presence"]["owner"] == "child"


def test_nothing_to_watch_is_a_404_by_name_and_not_an_empty_success(
    host_log: log.HostLog, tmp_path: Path
) -> None:
    door = a_door(host_log, token="tok", orchestrator=FakeOrchestrator(where=tmp_path))
    server = door_module.serve(door, host="127.0.0.1", port=0)
    port = server.server_address[1]
    try:
        status, body = _get(port, "/install/events")
    finally:
        server.shutdown()
    assert status == 404
    assert isinstance(body, dict)
    assert body["error"]["code"] == "no_install_running"


def test_a_late_attacher_sees_the_step_it_joined_at_and_then_follows(
    host_log: log.HostLog, tmp_path: Path
) -> None:
    import urllib.request

    started = threading.Event()
    go = threading.Event()

    def sequence(emit: Callable[[installer.Event], None]) -> None:
        emit(installer.Event("step", {"name": "wsl-state", "index": 1, "total": 11}))
        emit(installer.Event("state", {"code": "wsl_ready", "sentence": "ready", "action": "instruct"}))
        emit(installer.Event("step", {"name": "import-distro", "index": 2, "total": 11}))
        emit(installer.Event("line", {"text": "Downloading Ubuntu's own WSL image", "stream": "stdout"}))
        started.set()
        assert go.wait(20.0)
        emit(installer.Event("done", {"steps": []}))

    door = a_door(host_log, sequence, token="tok", orchestrator=FakeOrchestrator(where=tmp_path))
    server = door_module.serve(door, host="127.0.0.1", port=0)
    port = server.server_address[1]
    mover = threading.Thread(target=_post_install, args=(port,), daemon=True)
    try:
        mover.start()
        assert started.wait(20.0), "the move never began"
        status, body = _get(port, "/install")
        assert status == 200 and isinstance(body, dict) and body["running"] is True
        refused_status, refused = _post_install(port)
        assert refused_status == 409
        assert isinstance(refused, dict)
        assert refused["error"]["code"] == "host_install_running"
        assert "/install/events" in refused["error"]["message"]

        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/install/events",
            headers={"Authorization": "Bearer tok"},
        )
        response = urllib.request.urlopen(request, timeout=20)
        go.set()
        lines = [json.loads(line) for line in response.read().decode().splitlines() if line]
    finally:
        go.set()
        mover.join(timeout=20.0)
        server.shutdown()
    assert [line["id"] for line in lines] == [1, 2, 3, 4, 5]
    assert [line["event"] for line in lines] == ["step", "state", "step", "line", "done"]
    assert lines[2]["data"]["name"] == "import-distro", "the step it joined at"


def _post_install(port: int) -> tuple[int, object]:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/install",
        data=json.dumps({"target": "wsl"}).encode(),
        headers={"Authorization": "Bearer tok", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as refused:
        return refused.code, json.loads(refused.read().decode())


def test_the_ring_keeps_the_LAST_events_and_not_the_whole_transcript(
    host_log: log.HostLog,
) -> None:
    def sequence(emit: Callable[[installer.Event], None]) -> None:
        emit(installer.Event("step", {"name": "guest-install", "index": 4, "total": 11}))
        for index in range(door_module.MAX_RING_EVENTS + 50):
            emit(installer.Event("line", {"text": f"pip line {index}", "stream": "stdout"}))
        emit(installer.Event("done", {}))

    door = a_door(host_log, sequence, token="tok")
    assert door.claim()
    try:
        door.run_recorded()
    finally:
        door.release()
    backlog, _watcher = door.attach()
    assert len(backlog) == door_module.MAX_RING_EVENTS
    assert backlog[-1]["event"] == "done"
    assert backlog[-1]["id"] == door_module.MAX_RING_EVENTS + 52


def test_a_move_that_throws_records_a_terminal_event_for_every_watcher(
    host_log: log.HostLog,
) -> None:
    def sequence(emit: Callable[[installer.Event], None]) -> None:
        emit(installer.Event("step", {"name": "wsl-state", "index": 1, "total": 11}))
        raise HostError("rootfs_download_failed", "the image would not download")

    door = a_door(host_log, sequence, token="tok")
    assert door.claim()
    with pytest.raises(HostError):
        try:
            door.run_recorded()
        finally:
            door.release()
    backlog, watcher = door.attach()
    assert [event["event"] for event in backlog] == ["step", "failed"]
    assert backlog[-1]["data"]["code"] == "rootfs_download_failed"
    assert watcher.get_nowait() is None, "a finished move closes the queue it hands out"


def test_a_corrupt_outcome_is_quarantined_by_name_and_then_read_as_absent(tmp_path: Path) -> None:
    (tmp_path / "wsl-outcome.json").write_text("{not json", encoding="utf-8")
    said: list[str] = []
    assert outcome.read_or_quarantine(tmp_path, said.append) is None
    assert not (tmp_path / "wsl-outcome.json").exists()
    aside = list(tmp_path.glob("wsl-outcome.json.bad-*"))
    assert len(aside) == 1
    assert aside[0].read_text(encoding="utf-8") == "{not json"
    assert said and str(aside[0]) in said[0]
    assert outcome.read(tmp_path) is None, "the orchestrator decides again from nothing"


def test_a_sound_outcome_is_not_quarantined(tmp_path: Path) -> None:
    written = outcome.write(tmp_path, state=outcome.DONE, release="1.0.5", attempts=0)
    said: list[str] = []
    assert outcome.read_or_quarantine(tmp_path, said.append) == written
    assert said == []
    assert not list(tmp_path.glob("*.bad-*"))


def test_a_corrupt_cleanup_record_is_quarantined_and_the_cleanup_dropped(tmp_path: Path) -> None:
    record = tmp_path / installer.CLEANUP_RECORD
    record.write_text("{not json", encoding="utf-8")
    said: list[str] = []
    aside = installer.quarantine_bad_cleanup_record(tmp_path, said.append)
    assert aside is not None and aside.is_file() and ".bad-" in aside.name
    assert not record.exists()
    assert said and str(aside) in said[0] and "crucible uninstall --purge-weights" in said[0]
    installer.record_cleanup(tmp_path, {("model", "a")})
    assert installer.quarantine_bad_cleanup_record(tmp_path, said.append) is None
    assert record.is_file(), "a sound record stays where it is"


def test_a_cleanup_record_that_is_not_json_is_refused_by_name_not_by_a_traceback(tmp_path: Path) -> None:
    (tmp_path / installer.CLEANUP_RECORD).write_text("{not json", encoding="utf-8")
    with pytest.raises(HostError) as caught:
        installer.cleanup_subjects(tmp_path)
    assert caught.value.code == installer.CLEANUP_RECORD_INVALID
    assert str(tmp_path / installer.CLEANUP_RECORD) in caught.value.message


def _import_walk(tmp_path: Path, runner: Scripted, events: list[installer.Event]) -> installer.EngineInstall:
    return installer.EngineInstall(
        runner, events.append, release="0.6.0", home=tmp_path,
        install_sh_url="https://example.invalid/install.sh",
    )


def test_a_killed_import_that_left_only_the_vhdx_is_cleared_and_imported_again(tmp_path: Path) -> None:
    (tmp_path / "wsl").mkdir()
    (tmp_path / "wsl" / "ext4.vhdx").write_bytes(b"half")
    events: list[installer.Event] = []
    runner = Scripted(answers={"--status": ok("Default Version: 2"), "-l -v": ok("  Ubuntu  Running  2\n")})
    with pytest.raises(HostError) as caught:
        _import_walk(tmp_path, runner, events).run()
    assert caught.value.code == "rootfs_sha_mismatch", "the walk got past the import check"
    assert list((tmp_path / "wsl").iterdir()) == []
    lines = [e.data["text"] for e in events if e.event == "line"]
    assert any("interrupted" in line and "ext4.vhdx" in line for line in lines), lines


def test_a_wsl_directory_with_foreign_files_is_refused_with_the_directory_and_the_command(tmp_path: Path) -> None:
    (tmp_path / "wsl").mkdir()
    (tmp_path / "wsl" / "notes.txt").write_text("mine", encoding="utf-8")
    events: list[installer.Event] = []
    runner = Scripted(answers={"--status": ok("Default Version: 2"), "-l -v": ok("  Ubuntu  Running  2\n")})
    with pytest.raises(HostError) as caught:
        _import_walk(tmp_path, runner, events).run()
    assert caught.value.code == "distro_import_incomplete"
    assert str(tmp_path / "wsl") in caught.value.message
    assert 'Remove-Item -Recurse -Force "%s"' % (tmp_path / "wsl") in caught.value.message
    assert installer.TRY_AGAIN_HINT in caught.value.message
    assert (tmp_path / "wsl" / "notes.txt").exists(), "a file an import never writes is kept"


def test_a_distro_crucible_did_not_make_is_named_with_the_unregister_command_and_its_cost(tmp_path: Path) -> None:
    from crucible.host.wsl_states import WSL_CONF_MARKER

    events: list[installer.Event] = []
    runner = Scripted(answers={
        "--status": ok("Default Version: 2"),
        "-l -v": ok(f"  {CRUCIBLE_DISTRO}  Running  2\n"),
        "/etc/wsl.conf": ok("[boot]\nsystemd=true\n"),
    })
    with pytest.raises(HostError) as caught:
        _import_walk(tmp_path, runner, events)._import_distro()
    assert caught.value.code == "distro_unmarked"
    assert f"wsl --unregister {CRUCIBLE_DISTRO}" in caught.value.message
    assert WSL_CONF_MARKER in caught.value.message
    assert "deletes" in caught.value.message
    assert not any("--unregister" in " ".join(call) for call in runner.calls), "named, never run"


def test_an_unreadable_lan_record_is_quarantined_and_sharing_stays_off(tmp_path: Path) -> None:
    from crucible import lan

    (tmp_path / lan.RECORD).write_text("{not json", encoding="utf-8")
    events: list[installer.Event] = []
    walk = _import_walk(tmp_path, Scripted(), events)
    walk._lan_door()
    assert not (tmp_path / lan.RECORD).exists()
    aside = list(tmp_path.glob(f"{lan.RECORD}.bad-*"))
    assert len(aside) == 1
    lines = [e.data["text"] for e in events if e.event == "line"]
    assert any(str(aside[0]) in line and "crucible lan enable" in line for line in lines), lines
    assert not any(e.event == "failed" for e in events)


NETSTAT_SAMPLE = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1234
  TCP    127.0.0.1:7101         0.0.0.0:0              LISTENING       4242
  TCP    127.0.0.1:7101         127.0.0.1:50000        ESTABLISHED     4242
  TCP    127.0.0.1:50000        127.0.0.1:7101         ESTABLISHED     999
"""

TASKLIST_SAMPLE = '"other.exe","4242","Console","1","10,000 K"\r\n'


def test_the_port_holder_is_read_from_netstat_and_tasklist_and_named_with_its_pid(monkeypatch) -> None:
    from crucible.host import portholder

    assert portholder.listening_pid(NETSTAT_SAMPLE, 7101) == 4242
    assert portholder.listening_pid(NETSTAT_SAMPLE, 7100) is None
    assert portholder.image_name(TASKLIST_SAMPLE) == "other.exe"
    assert portholder.image_name("INFO: No tasks are running which match the specified criteria.") is None

    monkeypatch.setattr(portholder.sys, "platform", "win32")
    asked: list[tuple[str, ...]] = []

    def run(argv):
        asked.append(tuple(argv))
        return NETSTAT_SAMPLE if argv[0] == "netstat" else TASKLIST_SAMPLE

    assert portholder.held_sentence(7101, run) == (
        "port 7101 is held by other.exe (pid 4242); stop it or run `crucible local shutdown`"
    )
    assert asked == [portholder.NETSTAT_ARGV, portholder.tasklist_argv(4242)]
    nobody = portholder.held_sentence(7101, lambda argv: "")
    assert "netstat -ano | findstr :7101" in nobody


def _stepping_clock(step: float) -> Callable[[], float]:
    state = {"now": 0.0}

    def clock() -> float:
        state["now"] += step
        return state["now"]

    return clock


@WINDOWS_ONLY
def test_installwatch_starts_the_controller_before_it_gives_the_sign_in_advice(tmp_path: Path, monkeypatch) -> None:
    import io
    from datetime import datetime, timezone
    from crucible.host import installwatch

    monkeypatch.setattr(installwatch, "door_status", lambda token: None)
    started: list[Path] = []
    out = io.StringIO()
    code = installwatch.watch(
        tmp_path, datetime.now(timezone.utc), brief=False, out=out,
        clock=_stepping_clock(30.0), sleep=lambda seconds: None,
        alive=lambda: False, start=lambda home: started.append(home) or True,
    )
    assert code == 0
    assert started == [tmp_path], "started once, not on every poll"
    text = " ".join(out.getvalue().split())
    assert "this window is starting it" in text
    assert str(tmp_path / "host.log") in text
    assert "Sign out of Windows" in text, "the sign-in advice is the fallback, after the start"


@WINDOWS_ONLY
def test_installwatch_names_the_log_when_the_controller_cannot_be_started(tmp_path: Path, monkeypatch) -> None:
    import io
    from datetime import datetime, timezone
    from crucible.host import installwatch

    monkeypatch.setattr(installwatch, "door_status", lambda token: None)
    out = io.StringIO()
    installwatch.watch(
        tmp_path, datetime.now(timezone.utc), brief=False, out=out,
        clock=_stepping_clock(30.0), sleep=lambda seconds: None,
        alive=lambda: False, start=lambda home: False,
    )
    assert str(tmp_path / "host.log") in out.getvalue()


def _watch_with_no_door(home: Path, monkeypatch) -> str:
    import io
    from datetime import datetime, timezone
    from crucible.host import installwatch

    monkeypatch.setattr(installwatch, "door_status", lambda token: None)
    out = io.StringIO()
    installwatch.watch(
        home, datetime.now(timezone.utc), brief=False, out=out,
        clock=_stepping_clock(30.0), sleep=lambda seconds: None,
        alive=lambda: False, start=lambda home: False,
    )
    return " ".join(out.getvalue().split())


def test_installwatch_names_an_unreadable_config_and_the_command_that_rewrites_it(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "config.toml").write_text("[auth\ntoken = ", encoding="utf-8")
    text = _watch_with_no_door(tmp_path, monkeypatch)
    assert str(tmp_path / "config.toml") in text
    assert "`crucible init --force`" in text
    assert text.count("crucible init --force") == 1, "said once, not on every poll"


def test_installwatch_does_not_let_a_config_error_escape(tmp_path: Path, monkeypatch) -> None:
    from crucible import controller_client
    from crucible.errors import ConfigError

    def unreadable(home: Path) -> str:
        raise ConfigError("the [auth] table is not a table")

    monkeypatch.setattr(controller_client, "bearer", unreadable)
    text = _watch_with_no_door(tmp_path, monkeypatch)
    assert "the [auth] table is not a table" in text
    assert str(tmp_path / "config.toml") in text
    assert "`crucible init --force`" in text


def test_try_again_names_the_log_when_the_controller_will_not_start(tmp_path: Path, monkeypatch) -> None:
    from crucible import local
    from crucible.host import retry

    monkeypatch.setattr(retry, "_controller_up", lambda: False)

    def refuse(home: Path) -> None:
        raise OSError("no pythonw")

    monkeypatch.setattr(local, "_spawn_controller", refuse)
    with pytest.raises(HostError) as caught:
        retry._ensure_controller(tmp_path, lambda text: None)
    assert caught.value.code == "host_door_unavailable"
    assert str(tmp_path / "host.log") in caught.value.message


def test_there_is_one_alive_and_it_reads_access_denied_as_alive() -> None:
    from crucible import processlock, uninstall

    assert app_module._alive is processlock.alive
    assert uninstall._alive is processlock.alive
    assert processlock.alive(os.getpid()) is True
    assert processlock.alive(2 ** 22 + 1) is False


def test_the_install_one_liner_is_the_one_the_readme_documents() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert paths.INSTALL_ONE_LINER in readme
    assert "install.ps1 | iex" in paths.INSTALL_ONE_LINER


@WINDOWS_ONLY
def test_a_missing_console_cmd_names_the_install_one_liner_and_a_silent_child_names_the_log(
    tmp_path: Path, host_log: log.HostLog
) -> None:
    from types import SimpleNamespace

    env = dict(WINDOWS_ENV, CRUCIBLE_HOME=str(tmp_path))
    runner = Scripted(env=env)
    context = app_module.HostContext(
        runner=runner, log=host_log, home=tmp_path,
        watcher=SimpleNamespace(respawn_host_mode=lambda argv, env: None, wait_for_ping=lambda seconds: False),
        presence=presence.Presence(Distro.ABSENT, Engine.STARTING, "starting", Owner.NONE),
    )
    host = app_module.Host(context)
    absent = host._start_host_mode(Distro.ABSENT)
    assert absent.engine is Engine.FAILED
    assert paths.INSTALL_ONE_LINER in absent.detail

    (tmp_path / "host").mkdir()
    (tmp_path / "host" / paths.CONSOLE_CMD).write_text("@echo off\n", encoding="utf-8")
    (tmp_path / "config.toml").write_text("[auth]\ntoken = 't'\n", encoding="utf-8")
    silent = host._start_host_mode(Distro.ABSENT)
    assert silent.engine is Engine.FAILED
    assert str(paths.log_path(env)) in silent.detail
