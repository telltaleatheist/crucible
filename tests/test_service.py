"""`crucible service` — the unit and the plist, byte for byte, and the verbs.

Nothing here runs `systemctl` or `launchctl`. Every subprocess this module makes
goes through one injectable runner (`crucible/service.py`), so the tests assert
on the **argv that would have been run** and on the **text that would have been
written** — which is the only way to check a service definition, because nobody
sees one until the machine reboots.

`user_home` is replaced in every test that writes, so no run of this suite can
touch a real `~/.config/systemd` or `~/Library/LaunchAgents`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest

from crucible import cli, hosttools, service
from crucible.config import load_config

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND

PATH_VALUE = "/usr/local/bin:/usr/bin:/bin"


class Runner:
    """A recording stand-in for `subprocess_runner`.

    `answers` maps a leading argv slice (as a tuple) to the `Ran` to return for
    any command that starts with it. Anything unmatched exits 0 with no output,
    because the ordinary case in these tests is a command that works and the
    interesting case is the one that does not.
    """

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
    """A throwaway home. No test may write into the real one."""
    root = tmp_path / "operator-home"
    root.mkdir()
    monkeypatch.setattr(service, "user_home", lambda: root)
    return root


def env_bin(home: Path, *, with_script: bool = True) -> Path:
    """An env's `bin/` holding an interpreter and the `crucible` console script.

    Both files, because `install` resolves the script from the interpreter's
    directory and refuses by name when it is not there.
    """
    directory = home / "env" / "bin"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    if with_script:
        (directory / "crucible").write_text("#!/bin/sh\n", encoding="utf-8")
    return directory


# ------------------------------------------------------- the generated text


EXPECTED_UNIT = """[Unit]
Description=Crucible inference server (crucible@owens-pc)
Documentation=https://github.com/telltaleatheist/crucible
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/telltale/.crucible
ExecStart=/home/telltale/anaconda3/envs/crucible/bin/crucible serve --host 127.0.0.1 --port 7100
Environment=CRUCIBLE_HOME=/home/telltale/.crucible
Environment=PATH=/usr/local/bin:/usr/bin:/bin
Restart=on-failure
RestartSec=5

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
    """MEASURED on the PC: `python -m crucible` from a user unit started in
    $HOME, which holds a checkout directory named `crucible`, and `-m` put the
    cwd on sys.path — so `crucible.voices` resolved to the manifest DIRECTORY
    and the unit crash-looped on an ImportError."""
    assert "-m crucible" not in EXPECTED_UNIT
    assert "WorkingDirectory=/home/telltale/.crucible" in EXPECTED_UNIT


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
    """systemd expands `%x` specifiers; an undoubled one reaches the service as
    something else entirely."""
    text = service.systemd_unit_text(
        server_name="crucible@box",
        program="/opt/env/bin/crucible",
        crucible_home=Path("/home/o/100%/.crucible"),
        host="127.0.0.1",
        port=7100,
        path_value="/usr/bin",
    )
    assert "Environment=CRUCIBLE_HOME=/home/o/100%%/.crucible" in text


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


# --------------------------------------------------------------- mechanism


def test_each_backend_has_exactly_one_mechanism() -> None:
    assert service.mechanism_for("cuda-linux") == service.SYSTEMD
    assert service.mechanism_for("mlx-darwin") == service.LAUNCHD


def test_a_backend_with_no_mechanism_is_refused_by_name() -> None:
    with pytest.raises(service.ServiceError) as caught:
        service.mechanism_for("rocm-linux")
    message = str(caught.value)
    assert "rocm-linux" in message
    assert "cuda-linux" in message and "mlx-darwin" in message


# ----------------------------------------------------------------- install


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
    """The bug this exists for: a unit with no PATH gets the bare service-manager
    PATH, and a fully installed host then reports `ffmpeg` missing."""
    monkeypatch.setattr(hosttools, "search_path", lambda: "/opt/homebrew/bin:/usr/bin")
    runner = Runner(LINGER_ON)
    install_systemd(user_home, runner, path_value=None)
    unit = service.unit_path(user_home).read_text(encoding="utf-8")
    assert "Environment=PATH=/opt/homebrew/bin:/usr/bin" in unit


def test_install_appends_the_servers_own_bin_to_the_recorded_path(
    user_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Since 0.6.0 the server can arrive as an env pack, and then the shell
    that runs `service install` is a `wsl.exe --exec` shell whose PATH cannot
    contain a directory created a minute earlier. Recording it is what makes
    "this is the PATH the service has" true of the process."""
    monkeypatch.setattr(hosttools, "search_path", lambda: "/usr/bin:/bin")
    install_systemd(user_home, Runner(LINGER_ON), path_value=None)
    unit = service.unit_path(user_home).read_text(encoding="utf-8")
    assert f"Environment=PATH=/usr/bin:/bin:{env_bin(user_home)}\n" in unit


def test_the_servers_bin_is_appended_and_never_prepended(user_home: Path) -> None:
    """It also holds `python3`, `uvicorn` and half a dozen dependency scripts.
    In FRONT of a host's own, those would silently change what every bare name
    means in order to fix nothing."""
    recorded = service.path_including_program_dir(
        "/opt/homebrew/bin:/usr/bin", str(env_bin(user_home) / "crucible")
    )
    assert recorded.startswith("/opt/homebrew/bin:/usr/bin:")
    assert recorded.endswith(str(env_bin(user_home)))


def test_a_bin_already_on_the_path_is_not_moved_to_the_back(
    user_home: Path,
) -> None:
    """Idempotent: re-installing from a shell that HAS the pack on its PATH
    must not demote it behind everything else."""
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
    assert "Environment=CRUCIBLE_HOME=/tmp/crucible-a1" in unit
    # …and the cwd is that same directory, not the operator's $HOME, which is
    # where the PC's ImportError came from.
    assert "WorkingDirectory=/tmp/crucible-a1" in unit


def test_install_refuses_when_there_is_no_console_script(user_home: Path) -> None:
    """No fallback to `python -m crucible`: that is exactly the bug."""
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
    # Nothing was loaded, so nothing was booted out.
    assert not runner.ran("launchctl", "bootout")


def test_launchd_install_makes_the_log_directory_launchd_needs(
    user_home: Path,
) -> None:
    """launchd refuses to load an agent whose StandardOutPath directory does not
    exist, and says nothing about the directory when it does."""
    install_launchd(user_home, Runner())
    assert (user_home / ".crucible" / "logs").is_dir()


def test_launchd_reinstall_boots_the_old_agent_out_first(user_home: Path) -> None:
    """`bootstrap` is what reads the plist; `kickstart` is not. A rewritten plist
    that is not booted out is a rewritten plist nothing is running."""
    loaded = Runner({("launchctl", "list"): answer(out="1234\t0\tcom.crucible.serve\n")})
    install_launchd(user_home, loaded)
    assert loaded.ran("launchctl", "bootout")
    order = [call for call in loaded.calls if call[0] == "launchctl"]
    assert order[1][1] == "bootout"
    assert order[2][1] == "bootstrap"


# --------------------------------------------------------------- uninstall


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


# ------------------------------------------------------------ start / stop


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
    runner = Runner()
    service.stop(service.SYSTEMD, home=user_home, runner=runner)
    assert runner.calls == [("systemctl", "--user", "stop", "crucible.service")]


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


# --------------------------------------------------------------- the parsers


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


# ------------------------------------------------------------------ status


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
    # The question does not exist on launchd, and None is how it says so.
    assert state.linger is None


# --------------------------------------------------------------------- CLI


@pytest.fixture
def installed_config(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[], None]:
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_BACKEND)

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
    # The console script is resolved from `sys.executable`, and whether the
    # interpreter running the suite has one beside it is a fact about the
    # machine rather than about this wiring. Pinned so the test asserts what
    # the CLI PASSES, not what pip happened to install.
    script = str(env_bin(user_home) / "crucible")
    monkeypatch.setattr(service, "console_script", lambda executable: script)
    assert cli.main(["service", "install"]) == 0
    unit = service.unit_path(user_home).read_text(encoding="utf-8")
    config = load_config(home)
    assert f"ExecStart={script} serve" in unit
    assert f"--host {config.host} --port {config.port}" in unit
    assert f"Environment=CRUCIBLE_HOME={config.home}" in unit
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
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_MAC_BACKEND)
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    monkeypatch.setattr(service, "subprocess_runner", Runner())
    monkeypatch.setattr(
        service, "console_script", lambda executable: str(env_bin(user_home) / "crucible")
    )
    assert cli.main(["service", "install"]) == 0
    assert service.plist_path(user_home).is_file()
    assert "mechanism: launchd" in capsys.readouterr().out


def test_cli_service_refuses_without_a_config(
    home: Path, user_home: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["service", "status"]) == 1
    assert "crucible init" in capsys.readouterr().err
