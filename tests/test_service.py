from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest

from crucible import cli, hosttools, service
from crucible.config import load_config

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "systemd units and launchd agents are POSIX files with POSIX paths; "
        "a Windows host runs its engine under the orchestrator, not a service manager"
    ),
)


PATH_VALUE = "/usr/local/bin:/usr/bin:/bin"


class Runner:

    def __init__(self, answers: dict[tuple[str, ...], service.Ran] | None = None):
        self.calls: list[tuple[str, ...]] = []
        self._answers = answers or {}

    def __call__(self, argv: Sequence[str]) -> service.Ran:
        argv = tuple(argv)
        self.calls.append(argv)
        for prefix, ran in self._answers.items():
            if argv[: len(prefix)] == prefix:
                return service.Ran(
                    argv=argv,
                    returncode=ran.returncode,
                    stdout=ran.stdout,
                    stderr=ran.stderr,
                )
        return service.Ran(argv=argv, returncode=0, stdout="", stderr="")

    def ran(self, *words: str) -> bool:
        return any(call[: len(words)] == words for call in self.calls)


def answer(*, code: int = 0, out: str = "", err: str = "") -> service.Ran:
    return service.Ran(argv=(), returncode=code, stdout=out, stderr=err)


LINGER_ON = {("loginctl", "show-user"): answer(out="Linger=yes\n")}
LINGER_OFF = {("loginctl", "show-user"): answer(out="Linger=no\n")}


@pytest.fixture
def user_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "operator-home"
    root.mkdir()
    monkeypatch.setattr(service, "user_home", lambda: root)
    return root


def env_bin(home: Path, *, with_script: bool = True) -> Path:
    directory = home / "env" / "bin"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    if with_script:
        (directory / "crucible").write_text("#!/bin/sh\n", encoding="utf-8")
    return directory


EXPECTED_UNIT = """[Unit]
Description=Crucible inference server (crucible@owens-pc)
Documentation=https://github.com/telltaleatheist/crucible
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/telltale/.crucible
ExecStart=/home/telltale/anaconda3/envs/crucible/bin/crucible serve --host 127.0.0.1 --port 7100
Environment="CRUCIBLE_HOME=/home/telltale/.crucible"
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Restart=always
RestartSec=2
OOMPolicy=continue

[Install]
WantedBy=default.target
"""


def test_the_systemd_unit_is_exactly_this() -> None:
    assert (
        service.systemd_unit_text(
            server_name="crucible@owens-pc",
            program="/home/telltale/anaconda3/envs/crucible/bin/crucible",
            crucible_home=Path("/home/telltale/.crucible"),
            host="127.0.0.1",
            port=7100,
            path_value=PATH_VALUE,
        )
        == EXPECTED_UNIT
    )


def test_the_unit_never_runs_python_dash_m() -> None:
    assert "-m crucible" not in EXPECTED_UNIT
    assert "WorkingDirectory=/home/telltale/.crucible" in EXPECTED_UNIT


def test_the_unit_restarts_always_and_the_reason_is_the_windows_host() -> None:
    assert "Restart=always\n" in EXPECTED_UNIT
    assert "Restart=on-failure" not in EXPECTED_UNIT
    assert f"RestartSec={service.RESTART_SECONDS}\n" in EXPECTED_UNIT
    assert service.RESTART_SECONDS == 2


def test_a_worker_the_oom_killer_ends_does_not_stop_the_server() -> None:
    """systemd's default OOMPolicy=stop stops the whole unit when any process in it is
    OOM-killed, and the workers are in it: the job fails, the server stays."""
    assert "OOMPolicy=continue\n" in EXPECTED_UNIT


def test_the_launchd_agent_is_deliberately_not_changed_with_it() -> None:
    assert "<key>SuccessfulExit</key>" in EXPECTED_PLIST
    assert "<key>KeepAlive</key>\n  <true/>" not in EXPECTED_PLIST


EXPECTED_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.crucible.serve</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/telltale/miniforge3/envs/crucible/bin/crucible</string>
    <string>serve</string>
    <string>--host</string>
    <string>127.0.0.1</string>
    <string>--port</string>
    <string>7100</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>CRUCIBLE_HOME</key>
    <string>/Users/telltale/.crucible</string>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/bin:/bin</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>WorkingDirectory</key>
  <string>/Users/telltale/.crucible</string>
  <key>StandardOutPath</key>
  <string>/Users/telltale/.crucible/logs/serve.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/telltale/.crucible/logs/serve.log</string>
  <key>ProcessType</key>
  <string>Interactive</string>
</dict>
</plist>
"""


def test_the_launchd_plist_is_exactly_this() -> None:
    assert (
        service.launchd_plist_text(
            program="/Users/telltale/miniforge3/envs/crucible/bin/crucible",
            crucible_home=Path("/Users/telltale/.crucible"),
            host="127.0.0.1",
            port=7100,
            path_value="/opt/homebrew/bin:/usr/bin:/bin",
            log_path=Path("/Users/telltale/.crucible/logs/serve.log"),
        )
        == EXPECTED_PLIST
    )


def test_a_percent_in_a_systemd_value_is_doubled() -> None:
    text = service.systemd_unit_text(
        server_name="crucible@box",
        program="/opt/env/bin/crucible",
        crucible_home=Path("/home/o/100%/.crucible"),
        host="127.0.0.1",
        port=7100,
        path_value="/usr/bin",
    )
    assert 'Environment="CRUCIBLE_HOME=/home/o/100%%/.crucible"' in text


def test_a_plist_value_is_xml_escaped() -> None:
    text = service.launchd_plist_text(
        program="/opt/env/bin/crucible",
        crucible_home=Path("/Users/o/a&b"),
        host="127.0.0.1",
        port=7100,
        path_value="/usr/bin",
        log_path=Path("/Users/o/a&b/logs/serve.log"),
    )
    assert "<string>/Users/o/a&amp;b</string>" in text
    assert "a&b<" not in text


@pytest.mark.parametrize("value", ["/a\nb", "/a\rb"])
def test_a_line_break_in_a_value_is_refused_not_stripped(value: str) -> None:
    with pytest.raises(service.ServiceError) as caught:
        service.systemd_unit_text(
            server_name="crucible@box",
            program="/opt/env/bin/crucible",
            crucible_home=Path("/home/o/.crucible"),
            host="127.0.0.1",
            port=7100,
            path_value=value,
        )
    assert "line break" in str(caught.value)


def test_each_backend_has_exactly_one_mechanism() -> None:
    assert service.mechanism_for("cuda-linux") == service.SYSTEMD
    assert service.mechanism_for("mlx-darwin") == service.LAUNCHD


def test_a_backend_with_no_mechanism_is_refused_by_name() -> None:
    with pytest.raises(service.ServiceError) as caught:
        service.mechanism_for("rocm-linux")
    message = str(caught.value)
    assert "rocm-linux" in message
    assert "cuda-linux" in message and "mlx-darwin" in message


def install_systemd(home: Path, runner: Runner, **overrides: Any) -> list[str]:
    options: dict[str, Any] = {
        "home": home,
        "server_name": "crucible@owens-pc",
        "executable": str(env_bin(home) / "python"),
        "crucible_home": home / ".crucible",
        "host": "127.0.0.1",
        "port": 7100,
        "runner": runner,
        "path_value": PATH_VALUE,
        "user": "telltale",
    }
    options.update(overrides)
    return service.install(service.SYSTEMD, **options)


def test_install_writes_the_unit_reloads_and_enables(user_home: Path) -> None:
    runner = Runner(LINGER_ON)
    lines = install_systemd(user_home, runner)
    unit = service.unit_path(user_home)
    assert unit.is_file()
    script = env_bin(user_home) / "crucible"
    assert f"ExecStart={script} serve --host 127.0.0.1 --port 7100" in (
        unit.read_text(encoding="utf-8")
    )
    assert runner.calls[0] == ("systemctl", "--user", "daemon-reload")
    assert runner.calls[1] == (
        "systemctl", "--user", "enable", "--now", "crucible.service",
    )
    assert any("linger: on" in line for line in lines)


def test_install_records_the_installing_shells_path(
    user_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hosttools, "search_path", lambda: "/opt/homebrew/bin:/usr/bin")
    runner = Runner(LINGER_ON)
    install_systemd(user_home, runner, path_value=None)
    unit = service.unit_path(user_home).read_text(encoding="utf-8")
    assert 'Environment="PATH=/opt/homebrew/bin:/usr/bin' in unit


def test_install_appends_the_servers_own_bin_to_the_recorded_path(
    user_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hosttools, "search_path", lambda: "/usr/bin:/bin")
    install_systemd(user_home, Runner(LINGER_ON), path_value=None)
    unit = service.unit_path(user_home).read_text(encoding="utf-8")
    assert f'Environment="PATH=/usr/bin:/bin:{env_bin(user_home)}"\n' in unit


def test_the_servers_bin_is_appended_and_never_prepended(user_home: Path) -> None:
    recorded = service.path_including_program_dir(
        "/opt/homebrew/bin:/usr/bin", str(env_bin(user_home) / "crucible")
    )
    assert recorded.startswith("/opt/homebrew/bin:/usr/bin:")
    assert recorded.endswith(str(env_bin(user_home)))


def test_a_bin_already_on_the_path_is_not_moved_to_the_back(
    user_home: Path,
) -> None:
    directory = str(env_bin(user_home))
    before = f"{directory}:/usr/bin"
    assert (
        service.path_including_program_dir(before, f"{directory}/crucible") == before
    )


def test_install_records_crucible_home_so_the_service_serves_one_config(
    user_home: Path,
) -> None:
    runner = Runner(LINGER_ON)
    install_systemd(user_home, runner, crucible_home=Path("/tmp/crucible-a1"))
    unit = service.unit_path(user_home).read_text(encoding="utf-8")
    assert 'Environment="CRUCIBLE_HOME=/tmp/crucible-a1' in unit
    assert "WorkingDirectory=/tmp/crucible-a1" in unit


def test_install_refuses_when_there_is_no_console_script(user_home: Path) -> None:
    bin_dir = env_bin(user_home, with_script=True)
    (bin_dir / "crucible").unlink()
    with pytest.raises(service.ServiceError) as caught:
        service.install(
            service.SYSTEMD,
            home=user_home,
            server_name="crucible@owens-pc",
            executable=str(bin_dir / "python"),
            crucible_home=user_home / ".crucible",
            host="127.0.0.1",
            port=7100,
            runner=Runner(LINGER_ON),
            path_value=PATH_VALUE,
        )
    message = str(caught.value)
    assert "console script" in message
    assert "sys.path" in message


def test_install_prints_the_exact_linger_command_when_it_is_off(
    user_home: Path,
) -> None:
    lines = install_systemd(user_home, Runner(LINGER_OFF))
    assert any("linger: OFF" in line for line in lines)
    assert any("sudo loginctl enable-linger telltale" in line for line in lines)


def test_linger_that_cannot_be_asked_is_unknown_and_never_read_as_off(
    user_home: Path,
) -> None:
    runner = Runner({("loginctl",): answer(code=1, err="no loginctl here")})
    lines = install_systemd(user_home, runner)
    assert any("linger: UNKNOWN" in line for line in lines)


def test_a_changed_definition_is_restarted_onto(user_home: Path) -> None:
    runner = Runner(LINGER_ON)
    install_systemd(user_home, runner)
    assert not any("restart" in " ".join(call) for call in runner.calls), (
        "a first install writes the unit and starts it; there is nothing stale "
        "to restart, and a needless restart is a needless outage"
    )

    again = Runner(LINGER_ON)
    install_systemd(user_home, again)
    assert not any("restart" in " ".join(call) for call in again.calls)

    moved = Runner(LINGER_ON)
    lines = install_systemd(user_home, moved, port=7101)
    assert ("systemctl", "--user", "restart", "crucible.service") in moved.calls
    assert any("restarted crucible.service" in line for line in lines)


def test_install_is_idempotent(user_home: Path) -> None:
    runner = Runner(LINGER_ON)
    install_systemd(user_home, runner)
    first = service.unit_path(user_home).read_text(encoding="utf-8")
    install_systemd(user_home, runner)
    assert service.unit_path(user_home).read_text(encoding="utf-8") == first


def test_a_failing_systemctl_is_a_named_refusal(user_home: Path) -> None:
    runner = Runner(
        {("systemctl", "--user", "enable"): answer(code=1, err="Unit not found.")}
    )
    with pytest.raises(service.ServiceError) as caught:
        install_systemd(user_home, runner)
    assert "Unit not found." in str(caught.value)
    assert "systemctl --user enable --now crucible.service" in str(caught.value)


def install_launchd(home: Path, runner: Runner, **overrides: Any) -> list[str]:
    options: dict[str, Any] = {
        "home": home,
        "server_name": "crucible@studio",
        "executable": str(env_bin(home) / "python"),
        "crucible_home": home / ".crucible",
        "host": "127.0.0.1",
        "port": 7100,
        "runner": runner,
        "path_value": PATH_VALUE,
    }
    options.update(overrides)
    return service.install(service.LAUNCHD, **options)


def test_launchd_install_writes_the_plist_bootstraps_and_kickstarts(
    user_home: Path,
) -> None:
    runner = Runner()
    install_launchd(user_home, runner)
    plist = service.plist_path(user_home)
    assert plist.is_file()
    assert "<key>Label</key>" in plist.read_text(encoding="utf-8")
    assert runner.calls[0] == ("launchctl", "list")
    assert runner.ran("launchctl", "bootstrap")
    assert runner.ran("launchctl", "kickstart", "-k")
    assert not runner.ran("launchctl", "bootout")


def test_launchd_install_makes_the_log_directory_launchd_needs(
    user_home: Path,
) -> None:
    install_launchd(user_home, Runner())
    assert (user_home / ".crucible" / "logs").is_dir()


def test_launchd_reinstall_boots_the_old_agent_out_first(user_home: Path) -> None:
    loaded = Runner({("launchctl", "list"): answer(out="1234\t0\tcom.crucible.serve\n")})
    install_launchd(user_home, loaded)
    assert loaded.ran("launchctl", "bootout")
    order = [call for call in loaded.calls if call[0] == "launchctl"]
    assert order[1][1] == "bootout"
    assert order[2][1] == "bootstrap"


def test_uninstall_with_nothing_installed_says_so_and_runs_nothing(
    user_home: Path,
) -> None:
    runner = Runner()
    lines = service.uninstall(service.SYSTEMD, home=user_home, runner=runner)
    assert "nothing to remove" in lines[0]
    assert runner.calls == []


def test_uninstall_disables_removes_and_reloads(user_home: Path) -> None:
    runner = Runner(LINGER_ON)
    install_systemd(user_home, runner)
    runner.calls.clear()
    service.uninstall(service.SYSTEMD, home=user_home, runner=runner)
    assert runner.calls[0] == (
        "systemctl", "--user", "disable", "--now", "crucible.service",
    )
    assert not service.unit_path(user_home).exists()
    assert runner.calls[-1] == ("systemctl", "--user", "daemon-reload")


def test_uninstall_is_idempotent(user_home: Path) -> None:
    runner = Runner(LINGER_ON)
    install_systemd(user_home, runner)
    service.uninstall(service.SYSTEMD, home=user_home, runner=runner)
    lines = service.uninstall(service.SYSTEMD, home=user_home, runner=runner)
    assert "nothing to remove" in lines[0]


def test_launchd_uninstall_unloads_and_removes(user_home: Path) -> None:
    install_launchd(user_home, Runner())
    loaded = Runner({("launchctl", "list"): answer(out="- 0 com.crucible.serve\n")})
    lines = service.uninstall(service.LAUNCHD, home=user_home, runner=loaded)
    assert loaded.ran("launchctl", "bootout")
    assert not service.plist_path(user_home).exists()
    assert any("removed" in line for line in lines)


def test_start_refuses_when_nothing_is_installed(user_home: Path) -> None:
    with pytest.raises(service.ServiceError) as caught:
        service.start(service.SYSTEMD, home=user_home, runner=Runner())
    assert "crucible service install" in str(caught.value)


def test_start_starts(user_home: Path) -> None:
    runner = Runner(LINGER_ON)
    install_systemd(user_home, runner)
    runner.calls.clear()
    service.start(service.SYSTEMD, home=user_home, runner=runner)
    assert runner.calls == [("systemctl", "--user", "start", "crucible.service")]


def test_stop_stops(user_home: Path) -> None:
    install_systemd(user_home, Runner(LINGER_ON))
    runner = Runner()
    service.stop(service.SYSTEMD, home=user_home, runner=runner)
    assert runner.calls == [("systemctl", "--user", "stop", "crucible.service")]


def test_stopping_a_machine_with_no_unit_is_quiet_and_runs_nothing(
    user_home: Path,
) -> None:
    runner = Runner()
    lines = service.stop(service.SYSTEMD, home=user_home, runner=runner)
    assert runner.calls == []
    assert "nothing to stop" in lines[0]


def test_launchd_stop_is_a_bootout_because_keepalive_would_restart_a_kill(
    user_home: Path,
) -> None:
    loaded = Runner({("launchctl", "list"): answer(out="1234 0 com.crucible.serve\n")})
    service.stop(service.LAUNCHD, home=user_home, runner=loaded)
    assert loaded.ran("launchctl", "bootout")
    assert not loaded.ran("launchctl", "kill")


def test_launchd_stop_on_an_unloaded_agent_does_nothing(user_home: Path) -> None:
    runner = Runner()
    lines = service.stop(service.LAUNCHD, home=user_home, runner=runner)
    assert "not loaded" in lines[0]
    assert not runner.ran("launchctl", "bootout")


def test_systemctl_show_is_parsed_as_properties_not_scraped() -> None:
    parsed = service.parse_systemctl_show(
        "ActiveState=active\nSubState=running\nMainPID=4321\nUnitFileState=enabled\n"
    )
    assert parsed == {
        "ActiveState": "active",
        "SubState": "running",
        "MainPID": "4321",
        "UnitFileState": "enabled",
    }


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1234\t0\tcom.crucible.serve\n", (True, 1234)),
        ("-\t0\tcom.crucible.serve\n", (True, None)),
        ("1234\t0\tcom.apple.something\n", (False, None)),
        ("", (False, None)),
    ],
)
def test_launchctl_list_columns(text: str, expected: tuple[bool, int | None]) -> None:
    assert service.parse_launchctl_list(text, service.LAUNCHD_LABEL) == expected


@pytest.mark.parametrize(
    "ran,expected",
    [
        (answer(out="Linger=yes\n"), True),
        (answer(out="Linger=no\n"), False),
        (answer(code=1, err="Failed to get user: no such user"), None),
        (answer(out="Something=else\n"), None),
    ],
)
def test_linger_is_reported_including_that_it_could_not_be_asked(
    ran: service.Ran, expected: bool | None
) -> None:
    runner = Runner({("loginctl",): ran})
    assert service.read_linger(runner, "telltale") is expected


def test_status_reports_the_pid_and_the_unit_path(user_home: Path) -> None:
    runner = Runner(
        {
            ("systemctl", "--user", "show"): answer(
                out="ActiveState=active\nSubState=running\nMainPID=4321\n"
                "UnitFileState=enabled\n"
            ),
            ("loginctl",): answer(out="Linger=yes\n"),
        }
    )
    install_systemd(user_home, Runner(LINGER_ON))
    state = service.status(
        service.SYSTEMD, user_home, runner=runner, user="telltale"
    )
    assert state.running is True
    assert state.pid == 4321
    assert state.installed is True
    assert state.definition == service.unit_path(user_home)
    assert state.linger is True


def test_status_of_a_stopped_service_is_not_running_with_no_pid(
    user_home: Path,
) -> None:
    runner = Runner(
        {
            ("systemctl", "--user", "show"): answer(
                out="ActiveState=inactive\nSubState=dead\nMainPID=0\n"
                "UnitFileState=enabled\n"
            ),
            ("loginctl",): answer(out="Linger=no\n"),
        }
    )
    state = service.status(
        service.SYSTEMD, user_home, runner=runner, user="telltale"
    )
    assert state.running is False
    assert state.pid is None
    assert state.installed is False
    assert state.linger is False


def test_launchd_status_reads_the_agent(user_home: Path) -> None:
    runner = Runner(
        {("launchctl", "list"): answer(out="987\t0\tcom.crucible.serve\n")}
    )
    state = service.status(service.LAUNCHD, user_home, runner=runner)
    assert (state.running, state.pid) == (True, 987)
    assert state.definition == service.plist_path(user_home)
    assert state.linger is None


@pytest.fixture
def installed_config(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[], None]:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)

    def init() -> None:
        assert cli.main(["init", "--enable-echo"]) == 0

    return init


def test_cli_service_install_writes_the_unit_for_this_config(
    installed_config: Callable[[], None],
    user_home: Path,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    installed_config()
    capsys.readouterr()
    runner = Runner(LINGER_ON)
    monkeypatch.setattr(service, "subprocess_runner", runner)
    script = str(env_bin(user_home) / "crucible")
    monkeypatch.setattr(service, "console_script", lambda executable: script)
    assert cli.main(["service", "install"]) == 0
    unit = service.unit_path(user_home).read_text(encoding="utf-8")
    config = load_config(home)
    assert f"ExecStart={script} serve" in unit
    assert f"--host {config.host} --port {config.port}" in unit
    assert f'Environment="CRUCIBLE_HOME={config.home}"' in unit
    assert f"WorkingDirectory={config.home}" in unit
    assert f"({config.name})" in unit
    assert "mechanism: systemd" in capsys.readouterr().out


def test_cli_service_status_json_exits_nonzero_when_it_is_not_running(
    installed_config: Callable[[], None],
    user_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    installed_config()
    capsys.readouterr()
    runner = Runner(
        {
            ("systemctl", "--user", "show"): answer(
                out="ActiveState=inactive\nSubState=dead\nMainPID=0\n"
                "UnitFileState=disabled\n"
            ),
            ("loginctl",): answer(out="Linger=no\n"),
        }
    )
    monkeypatch.setattr(service, "subprocess_runner", runner)
    assert cli.main(["service", "status", "--json"]) == 1
    state = json.loads(capsys.readouterr().out)
    assert state["running"] is False
    assert state["mechanism"] == "systemd"
    assert state["definition"].endswith("crucible.service")


def test_cli_service_on_a_mac_uses_launchd(
    home: Path,
    user_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_MAC_BACKEND)
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    monkeypatch.setattr(service, "subprocess_runner", Runner())
    monkeypatch.setattr(
        service, "console_script", lambda executable: str(env_bin(user_home) / "crucible")
    )
    assert cli.main(["service", "install"]) == 0
    assert service.plist_path(user_home).is_file()
    assert "mechanism: launchd" in capsys.readouterr().out


def test_doctor_names_the_services_path_beside_the_shells(
    home: Path,
    user_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_MAC_BACKEND)
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    assert cli.main(["init"]) == 0
    capsys.readouterr()

    path = service.plist_path(user_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        service.launchd_plist_text(
            program="/opt/crucible/bin/crucible",
            crucible_home=home,
            host="127.0.0.1",
            port=7100,
            path_value="/opt/homebrew/bin:/usr/bin:/bin",
            log_path=home / "logs" / "serve.log",
        ),
        encoding="utf-8",
    )

    cli.main(["doctor", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["path"] == {
        "shell": "/usr/bin:/bin:/usr/sbin:/sbin",
        "service": "/opt/homebrew/bin:/usr/bin:/bin",
        "mechanism": "launchd",
        "definition": str(path),
        "agree": False,
    }

    cli.main(["doctor"])
    printed = capsys.readouterr().out
    assert "PATH (this shell):   /usr/bin:/bin:/usr/sbin:/sbin" in printed
    assert "PATH (the service):  /opt/homebrew/bin:/usr/bin:/bin" in printed
    assert "the two differ, which is normal" in printed


def test_doctor_says_none_recorded_rather_than_nothing(
    home: Path,
    user_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_MAC_BACKEND)
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    cli.main(["doctor", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["path"]["service"] is None
    assert report["path"]["agree"] is None
    assert report["path"]["mechanism"] == "launchd"
    cli.main(["doctor"])
    assert "PATH (the service):  none recorded" in capsys.readouterr().out


def test_the_recorded_path_survives_the_escaping_that_wrote_it(
    user_home: Path, tmp_path: Path
) -> None:
    awkward = "/opt/homebrew/bin:/Users/a&b/100%/bin"

    unit = service.unit_path(user_home)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(
        service.systemd_unit_text(
            server_name="forge",
            program="/opt/crucible/bin/crucible",
            crucible_home=user_home / ".crucible",
            host="127.0.0.1",
            port=7100,
            path_value=awkward,
        ),
        encoding="utf-8",
    )
    assert service.read_recorded_path("systemd", user_home) == awkward

    plist = service.plist_path(user_home)
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text(
        service.launchd_plist_text(
            program="/opt/crucible/bin/crucible",
            crucible_home=user_home / ".crucible",
            host="127.0.0.1",
            port=7100,
            path_value=awkward,
            log_path=user_home / "serve.log",
        ),
        encoding="utf-8",
    )
    assert service.read_recorded_path("launchd", user_home) == awkward


def test_a_unit_written_before_the_path_was_recorded_reads_as_none(
    user_home: Path,
) -> None:
    unit = service.unit_path(user_home)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(
        "[Service]\nExecStart=/opt/crucible/bin/crucible serve\n", encoding="utf-8"
    )
    assert service.read_recorded_path("systemd", user_home) is None


def test_cli_service_refuses_without_a_config(
    home: Path, user_home: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["service", "status"]) == 1
    assert "crucible init" in capsys.readouterr().err


def test_a_space_in_the_path_stays_one_environment_assignment() -> None:
    windows_path = "/usr/bin:/mnt/c/Program Files/Git/usr/bin:/mnt/c/Windows"
    text = service.systemd_unit_text(
        server_name="crucible@owens-pc-wsl",
        program="/home/telltale/anaconda3/envs/crucible/bin/crucible",
        crucible_home=Path("/home/telltale/.crucible"),
        host="127.0.0.1",
        port=7100,
        path_value=windows_path,
    )
    assert f'Environment="PATH={windows_path}"\n' in text
    line = next(l for l in text.splitlines() if l.startswith('Environment="PATH'))
    assert line.count('"') == 2 and line.endswith('"')


def test_a_quote_in_an_environment_value_is_refused_rather_than_escaped() -> None:
    for awkward in ['/usr/bin:/opt/a"b', "/usr/bin:/opt/a" + chr(92) + "b"]:
        with pytest.raises(service.ServiceError) as caught:
            service.systemd_unit_text(
                server_name="crucible@box",
                program="/opt/env/bin/crucible",
                crucible_home=Path("/home/o/.crucible"),
                host="127.0.0.1",
                port=7100,
                path_value=awkward,
            )
        assert "double quote or a backslash" in str(caught.value)


def test_in_wsl_reads_the_kernel_and_not_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.undo()
    release = tmp_path / "osrelease"
    monkeypatch.setattr(service, "OSRELEASE", release)
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)

    release.write_text("6.6.87.1-microsoft-standard-WSL2\n", encoding="utf-8")
    assert service.in_wsl() is True
    assert service.systemd_scope() == service.SYSTEM_SCOPE

    release.write_text("6.8.0-generic\n", encoding="utf-8")
    assert service.in_wsl() is False
    assert service.systemd_scope() == service.USER_SCOPE


def test_a_kernel_that_cannot_be_read_is_not_wsl(monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path) -> None:
    monkeypatch.undo()
    monkeypatch.setattr(service, "OSRELEASE", tmp_path / "does-not-exist")
    assert service.in_wsl() is False


def test_in_wsl_the_unit_is_the_machines_and_systemctl_drops_user(
    monkeypatch: pytest.MonkeyPatch, user_home: Path
) -> None:
    monkeypatch.setattr(service, "in_wsl", lambda: True)
    monkeypatch.setattr(service.os, "geteuid", lambda: 1000, raising=False)
    monkeypatch.setenv(service.WSL_DISTRO_ENV, "Ubuntu")
    system_dir = user_home / "etc-systemd-system"
    monkeypatch.setattr(service, "SYSTEM_UNIT_DIR", system_dir)
    runner = CopyingRunner(LINGER_ON)
    install_systemd(user_home, runner)

    assert service.unit_path(user_home) == system_dir / "crucible.service"
    assert (system_dir / "crucible.service").is_file()
    verbs = [call for call in runner.calls if "systemctl" in call]
    assert verbs[0][-1:] == ("daemon-reload",)
    assert verbs[1][-4:] == ("systemctl", "enable", "--now", "crucible.service")
    assert not any("--user" in call for call in runner.calls), (
        "the whole point is that the guest's server is not a user unit"
    )


def test_a_system_unit_says_whose_server_it_is(
    monkeypatch: pytest.MonkeyPatch, user_home: Path
) -> None:
    monkeypatch.setattr(service, "in_wsl", lambda: True)
    text = service.systemd_unit_text(
        server_name="crucible@owens-pc",
        program=str(env_bin(user_home) / "crucible"),
        crucible_home=user_home / ".crucible",
        host="127.0.0.1",
        port=7100,
        path_value=PATH_VALUE,
        run_as="telltale",
    )
    assert "User=telltale" in text
    assert "WantedBy=multi-user.target" in text
    assert "WantedBy=default.target" not in text


class CopyingRunner(Runner):

    def __call__(self, argv: Sequence[str]) -> service.Ran:
        words = list(argv)
        if "install" in words and "-D" in words:
            source, target = words[-2], words[-1]
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            Path(target).write_text(Path(source).read_text(encoding="utf-8"), encoding="utf-8")
        return super().__call__(argv)


@pytest.fixture
def wsl_guest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(service, "in_wsl", lambda: True)
    monkeypatch.setattr(service.os, "geteuid", lambda: 1000, raising=False)
    monkeypatch.setenv(service.WSL_DISTRO_ENV, "Ubuntu")
    etc = tmp_path / "etc-systemd-system"
    monkeypatch.setattr(service, "SYSTEM_UNIT_DIR", etc)
    return etc


ROOT_DOOR = ("wsl.exe", "-d", "Ubuntu", "-u", "root", "--exec")


def test_root_prefix_is_empty_for_a_process_that_is_already_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service.os, "geteuid", lambda: 0, raising=False)
    assert service.root_prefix({"WSL_DISTRO_NAME": "Ubuntu"}) == []


def test_root_prefix_refuses_when_the_distro_cannot_be_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service.os, "geteuid", lambda: 1000, raising=False)
    with pytest.raises(service.ServiceError) as caught:
        service.root_prefix({})
    assert "WSL_DISTRO_NAME" in str(caught.value)


def test_in_wsl_the_unit_is_written_to_etc_through_the_root_door(
    user_home: Path, wsl_guest: Path
) -> None:
    runner = CopyingRunner()
    install_systemd(user_home, runner)
    unit = wsl_guest / "crucible.service"
    assert unit.is_file(), "the system unit was never written"
    written = [call for call in runner.calls if "install" in call]
    assert len(written) == 1
    assert written[0][: len(ROOT_DOOR)] == ROOT_DOOR
    assert written[0][-1] == str(unit)


def test_the_system_managers_verbs_go_through_root_and_never_say_user(
    user_home: Path, wsl_guest: Path
) -> None:
    runner = CopyingRunner()
    install_systemd(user_home, runner)
    verbs = [call for call in runner.calls if "systemctl" in call]
    assert verbs, "nothing drove systemd"
    for call in verbs:
        assert call[: len(ROOT_DOOR)] == ROOT_DOOR, call
        assert "--user" not in call, call


def test_the_system_unit_names_the_installing_user_and_multi_user_target(
    user_home: Path, wsl_guest: Path
) -> None:
    install_systemd(user_home, CopyingRunner(), user="telltale")
    text = (wsl_guest / "crucible.service").read_text(encoding="utf-8")
    assert "User=telltale\n" in text
    assert "WantedBy=multi-user.target\n" in text
    assert "WantedBy=default.target" not in text


def test_a_system_install_says_nothing_about_linger(
    user_home: Path, wsl_guest: Path
) -> None:
    lines = install_systemd(user_home, CopyingRunner(LINGER_OFF))
    said = "\n".join(lines)
    assert "loginctl enable-linger" not in said, said
    assert "die with your shell" not in said, said
    assert "system unit, running as telltale" in said, said


def test_a_second_system_install_changes_nothing_and_restarts_nothing(
    user_home: Path, wsl_guest: Path
) -> None:
    install_systemd(user_home, CopyingRunner())
    second = CopyingRunner()
    install_systemd(user_home, second)
    assert not any("restart" in call for call in second.calls), second.calls


def test_stopping_a_system_unit_goes_through_root(
    user_home: Path, wsl_guest: Path
) -> None:
    runner = CopyingRunner()
    install_systemd(user_home, runner)
    stopper = CopyingRunner()
    service.stop(service.SYSTEMD, home=user_home, runner=stopper)
    assert stopper.calls[0][: len(ROOT_DOOR)] == ROOT_DOOR
    assert "--user" not in stopper.calls[0]


def test_removing_a_system_unit_removes_the_file_as_root(
    user_home: Path, wsl_guest: Path
) -> None:
    install_systemd(user_home, CopyingRunner())
    remover = CopyingRunner()
    service.uninstall(service.SYSTEMD, home=user_home, runner=remover)
    removals = [call for call in remover.calls if "rm" in call]
    assert len(removals) == 1, remover.calls
    assert removals[0] == (*ROOT_DOOR, "rm", "-f", str(wsl_guest / "crucible.service"))


def test_reading_the_status_of_a_system_unit_needs_no_root(
    user_home: Path, wsl_guest: Path
) -> None:
    install_systemd(user_home, CopyingRunner())
    reader = CopyingRunner()
    service.status(service.SYSTEMD, user_home, runner=reader, user="telltale")
    shows = [call for call in reader.calls if "show" in call]
    assert shows, reader.calls
    assert all("wsl.exe" not in call for call in shows), shows


def test_a_fresh_system_install_stops_nobodys_user_manager(
    user_home: Path, wsl_guest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service.os, "getuid", lambda: 1000, raising=False)
    runner = CopyingRunner()
    install_systemd(user_home, runner)
    assert not any("user@1000.service" in call for call in runner.calls), runner.calls


def test_a_user_unit_outside_wsl_is_still_stopped_by_its_own_manager(
    user_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_systemd(user_home, Runner(LINGER_ON))
    monkeypatch.setattr(service, "in_wsl", lambda: False)
    runner = Runner()
    service.stop(service.SYSTEMD, home=user_home, runner=runner)
    assert runner.calls == [("systemctl", "--user", "stop", "crucible.service")]


def test_when_neither_manager_answers_the_unit_state_is_unknown(
    user_home: Path, wsl_guest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service.os, "getuid", lambda: 1000, raising=False)
    (user_home / ".config/systemd/user").mkdir(parents=True, exist_ok=True)
    (user_home / ".config/systemd/user/crucible.service").write_text("[Unit]\n")
    runner = Runner(
        {
            ("systemctl",): answer(code=1, err="Failed to connect to bus\n"),
            ("loginctl",): answer(out="Linger=yes\n"),
        }
    )
    state = service.status(
        service.SYSTEMD, user_home, runner=runner, user="telltale"
    )
    assert state.running is None

