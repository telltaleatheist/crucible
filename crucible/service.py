from __future__ import annotations

import getpass
import os
import plistlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from xml.sax.saxutils import escape as xml_escape

from . import hosttools
from .backend import CUDA_LINUX, MLX_DARWIN
from .errors import CrucibleError
from .wsl import root_argv

SERVICE_MECHANISM: dict[str, str] = {
    CUDA_LINUX: "systemd",
    MLX_DARWIN: "launchd",
}

SYSTEMD = "systemd"
LAUNCHD = "launchd"

UNIT_NAME = "crucible.service"
LAUNCHD_LABEL = "com.crucible.serve"

RESTART_SECONDS = 2

CONSOLE_SCRIPT = "crucible"


class ServiceError(CrucibleError):
    ...


def user_home() -> Path:
    return Path.home()


USER_SCOPE = "user"
SYSTEM_SCOPE = "system"

SYSTEM_UNIT_DIR = Path("/etc/systemd/system")

OSRELEASE = Path("/proc/sys/kernel/osrelease")


def in_wsl() -> bool:
    try:
        release = OSRELEASE.read_text(encoding="utf-8")
    except OSError:
        return False
    return "microsoft" in release.lower()


def systemd_scope() -> str:
    return SYSTEM_SCOPE if in_wsl() else USER_SCOPE


def systemctl_argv(scope: str, *verbs: str) -> list[str]:
    prefix = ("--user",) if scope == USER_SCOPE else ()
    return ["systemctl", *prefix, *verbs]


WSL_DISTRO_ENV = "WSL_DISTRO_NAME"


def _passwordless_sudo() -> bool:
    try:
        return (
            subprocess.run(
                ["sudo", "-n", "true"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=10,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def root_prefix(environ: Mapping[str, str] | None = None) -> list[str]:
    if not hasattr(os, "geteuid"):
        raise ServiceError(
            "a system unit is a POSIX thing and this platform has no euid; "
            "nothing here should have asked for one"
        )
    if os.geteuid() == 0:
        return []
    if _passwordless_sudo():
        return ["sudo", "-n"]
    distro = (os.environ if environ is None else environ).get(WSL_DISTRO_ENV)
    if not distro:
        raise ServiceError(
            f"the system unit needs root and ${WSL_DISTRO_ENV} is unset, so the "
            "`wsl.exe -u root` door cannot be named. Install from a WSL shell, "
            "or run this as root"
        )
    return root_argv(distro)


def installed_scope(home: Path) -> str | None:
    scope = systemd_scope()
    return scope if unit_path(home, scope).is_file() else None


def acting_scope(home: Path, verb: str) -> str:
    scope = installed_scope(home)
    if scope is None:
        raise ServiceError(
            f"there is no {UNIT_NAME} on this machine to {verb}: "
            f"{unit_path(home)} does not exist. Run `crucible service install` first"
        )
    return scope


def writing_door(scope: str) -> list[str]:
    return root_prefix() if scope == SYSTEM_SCOPE else []


def unit_path(home: Path, scope: str | None = None) -> Path:
    if (scope if scope is not None else systemd_scope()) == SYSTEM_SCOPE:
        return SYSTEM_UNIT_DIR / UNIT_NAME
    return home / ".config" / "systemd" / "user" / UNIT_NAME


def plist_path(home: Path) -> Path:
    return home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def serve_log_path(crucible_home: Path) -> Path:
    return crucible_home / "logs" / "serve.log"


def definition_path(mechanism: str, home: Path) -> Path:
    if mechanism == SYSTEMD:
        return unit_path(home)
    if mechanism == LAUNCHD:
        return plist_path(home)
    raise ServiceError(f"there is no service mechanism called {mechanism!r}")


def console_script(executable: str) -> str:
    path = Path(executable).parent / CONSOLE_SCRIPT
    if not path.is_file():
        raise ServiceError(
            f"there is no `{CONSOLE_SCRIPT}` console script at {path}. A service "
            f"must run it rather than `{executable} -m crucible`, because `-m` "
            "puts the working directory on sys.path and a user unit starts in "
            "$HOME — where a directory named `crucible` (the checkout) shadows "
            "the installed package and the server dies on an import. Install the "
            f"package into the env that owns {executable} (`pip install -e .`) "
            "and run this again"
        )
    return str(path)


def path_including_program_dir(path_value: str, program: str) -> str:
    directory = str(Path(program).resolve().parent)
    entries = [entry for entry in path_value.split(os.pathsep) if entry != ""]
    if directory in entries:
        return path_value
    return os.pathsep.join([*entries, directory])


def mechanism_for(backend_kind: str) -> str:
    found = SERVICE_MECHANISM.get(backend_kind)
    if found is None:
        raise ServiceError(
            f"there is no service mechanism for backend {backend_kind!r}; "
            f"crucible supervises {sorted(SERVICE_MECHANISM)} "
            f"({CUDA_LINUX} with a systemd user unit, {MLX_DARWIN} with a launchd "
            "agent) and has no third way to keep a server up on this host. Run "
            "`crucible serve` in the foreground, or under whatever this host's "
            "supervisor is"
        )
    return found


@dataclass(frozen=True)
class Ran:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def text(self) -> str:
        return (self.stderr.strip() or self.stdout.strip()) or "no output"


Runner = Callable[[Sequence[str]], Ran]


def subprocess_runner(argv: Sequence[str]) -> Ran:
    completed = subprocess.run(
        list(argv), capture_output=True, text=True, timeout=120
    )
    return Ran(
        argv=tuple(argv),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _require(runner: Runner, argv: Sequence[str], what: str) -> Ran:
    ran = runner(argv)
    if not ran.ok:
        raise ServiceError(
            f"{what}: `{' '.join(ran.argv)}` exited {ran.returncode}: {ran.text()}"
        )
    return ran


def _one_line(where: str, value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ServiceError(
            f"{where} contains a line break ({value!r}); a unit file and a plist "
            "are line-oriented and a value that breaks a line is a value that "
            "means something else by the time the service manager reads it"
        )
    return value


def systemd_unit_text(
    *,
    server_name: str,
    program: str,
    crucible_home: Path,
    host: str,
    port: int,
    path_value: str,
    run_as: str | None = None,
) -> str:
    def escape(where: str, value: str) -> str:
        return _one_line(where, value).replace("%", "%%")

    def environment(name: str, where: str, value: str) -> str:
        if '"' in value or "\\" in value:
            raise ServiceError(
                f"{where} contains a double quote or a backslash ({value!r}); a "
                "systemd `Environment=` value is quoted so that a space in it "
                "stays part of one assignment, and those two characters have "
                "their own meaning inside those quotes"
            )
        return f'Environment="{name}={escape(where, value)}"\n'

    return (
        "[Unit]\n"
        f"Description=Crucible inference server ({escape('server name', server_name)})\n"
        "Documentation=https://github.com/telltaleatheist/crucible\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"WorkingDirectory={escape('CRUCIBLE_HOME', str(crucible_home))}\n"
        f"ExecStart={escape('the crucible console script', program)} serve"
        f" --host {escape('the bind host', host)} --port {int(port)}\n"
        + environment("CRUCIBLE_HOME", "CRUCIBLE_HOME", str(crucible_home))
        + environment("PATH", "PATH", path_value)
        + "Restart=always\n"
        f"RestartSec={RESTART_SECONDS}\n"
        + (f"User={_one_line('run_as', run_as)}\n" if run_as else "")
        + "\n"
        + "[Install]\n"
        + ("WantedBy=multi-user.target\n" if run_as
           else "WantedBy=default.target\n")
    )


def launchd_plist_text(
    *,
    program: str,
    crucible_home: Path,
    host: str,
    port: int,
    path_value: str,
    log_path: Path,
) -> str:
    arguments = [
        _one_line("the crucible console script", program),
        "serve",
        "--host",
        _one_line("the bind host", host),
        "--port",
        str(int(port)),
    ]
    argument_lines = "".join(
        f"    <string>{xml_escape(value)}</string>\n" for value in arguments
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        "<dict>\n"
        "  <key>Label</key>\n"
        f"  <string>{LAUNCHD_LABEL}</string>\n"
        "  <key>ProgramArguments</key>\n"
        "  <array>\n"
        f"{argument_lines}"
        "  </array>\n"
        "  <key>EnvironmentVariables</key>\n"
        "  <dict>\n"
        "    <key>CRUCIBLE_HOME</key>\n"
        f"    <string>{xml_escape(_one_line('CRUCIBLE_HOME', str(crucible_home)))}</string>\n"
        "    <key>PATH</key>\n"
        f"    <string>{xml_escape(_one_line('PATH', path_value))}</string>\n"
        "  </dict>\n"
        "  <key>RunAtLoad</key>\n"
        "  <true/>\n"
        "  <key>KeepAlive</key>\n"
        "  <dict>\n"
        "    <key>SuccessfulExit</key>\n"
        "    <false/>\n"
        "  </dict>\n"
        "  <key>WorkingDirectory</key>\n"
        f"  <string>{xml_escape(str(crucible_home))}</string>\n"
        "  <key>StandardOutPath</key>\n"
        f"  <string>{xml_escape(_one_line('the log path', str(log_path)))}</string>\n"
        "  <key>StandardErrorPath</key>\n"
        f"  <string>{xml_escape(str(log_path))}</string>\n"
        "  <key>ProcessType</key>\n"
        "  <string>Interactive</string>\n"
        "</dict>\n"
        "</plist>\n"
    )


@dataclass(frozen=True)
class Status:
    mechanism: str
    definition: Path
    installed: bool
    running: bool | None
    pid: int | None
    detail: str
    linger: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "definition": str(self.definition),
            "installed": self.installed,
            "running": self.running,
            "pid": self.pid,
            "detail": self.detail,
            "linger": self.linger,
        }


def read_recorded_path(mechanism: str, home: Path) -> str | None:
    path = definition_path(mechanism, home)
    if not path.is_file():
        return None
    try:
        if mechanism == LAUNCHD:
            with path.open("rb") as handle:
                document = plistlib.load(handle)
            variables = document.get("EnvironmentVariables")
            if not isinstance(variables, dict):
                return None
            value = variables.get("PATH")
            return value if isinstance(value, str) else None
        for line in path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator != "=" or name.strip() != "Environment":
                continue
            value = value.strip()
            if len(value) < 2 or not value.startswith('"') or not value.endswith('"'):
                continue
            value = value[1:-1]
            key, is_pair, recorded = value.partition("=")
            if is_pair == "=" and key == "PATH":
                return recorded.replace("%%", "%")
        return None
    except (OSError, ValueError, plistlib.InvalidFileException):
        return None


def parse_systemctl_show(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator == "=":
            values[key.strip()] = value.strip()
    return values


def parse_launchctl_list(text: str, label: str) -> tuple[bool, int | None]:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1] == label:
            raw = parts[0]
            return True, (int(raw) if raw.lstrip("-").isdigit() and raw != "-" else None)
    return False, None


def read_linger(runner: Runner, user: str) -> bool | None:
    ran = runner(["loginctl", "show-user", user, "--property=Linger"])
    if not ran.ok:
        return None
    value = parse_systemctl_show(ran.stdout).get("Linger")
    if value is None:
        return None
    return value.lower() == "yes"


ServiceState = tuple[bool | None, int | None, str, bool | None]


def _not_asked(tool: str, ran: Ran) -> str:
    return (
        f"{tool} could not be asked: `{' '.join(ran.argv)}` exited "
        f"{ran.returncode}: {ran.text()}"
    )


def _systemd_state(runner: Runner, user: str | None) -> ServiceState:
    ran = runner(
        systemctl_argv(
            systemd_scope(),
            "show",
            UNIT_NAME,
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
            "--property=UnitFileState",
        )
    )
    linger = read_linger(runner, user if user is not None else getpass.getuser())
    if not ran.ok:
        return None, None, _not_asked("systemctl", ran), linger
    properties = parse_systemctl_show(ran.stdout)
    active = properties.get("ActiveState", "unknown")
    raw_pid = properties.get("MainPID", "0")
    pid = int(raw_pid) if raw_pid.isdigit() and raw_pid != "0" else None
    detail = (
        f"{active}/{properties.get('SubState', 'unknown')}, unit file "
        f"{properties.get('UnitFileState', 'unknown')}"
    )
    return active == "active", pid, detail, linger


def _launchd_state(runner: Runner) -> ServiceState:
    ran = runner(["launchctl", "list"])
    if not ran.ok:
        return None, None, _not_asked("launchctl", ran), None
    loaded, pid = parse_launchctl_list(ran.stdout, LAUNCHD_LABEL)
    detail = (
        f"agent {LAUNCHD_LABEL} is "
        + ("loaded" if loaded else "not loaded")
        + (f" and running as pid {pid}" if pid is not None else "")
    )
    return pid is not None, pid, detail, None


def status(
    mechanism: str, home: Path, *, runner: Runner, user: str | None = None
) -> Status:
    definition = definition_path(mechanism, home)
    installed = definition.is_file()
    if mechanism == SYSTEMD:
        running, pid, detail, linger = _systemd_state(runner, user)
    elif mechanism == LAUNCHD:
        running, pid, detail, linger = _launchd_state(runner)
    else:
        raise ServiceError(f"there is no service mechanism called {mechanism!r}")
    return Status(
        mechanism=mechanism,
        definition=definition,
        installed=installed,
        running=running,
        pid=pid,
        detail=detail,
        linger=linger,
    )


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _agent_target() -> str:
    return f"{_domain()}/{LAUNCHD_LABEL}"


def _agent_is_loaded(runner: Runner) -> bool:
    ran = runner(["launchctl", "list"])
    if not ran.ok:
        raise ServiceError(
            f"launchctl could not be asked what is loaded: "
            f"`{' '.join(ran.argv)}` exited {ran.returncode}: {ran.text()}"
        )
    loaded, _ = parse_launchctl_list(ran.stdout, LAUNCHD_LABEL)
    return loaded


def write_definition(
    path: Path,
    text: str,
    *,
    elevate: Sequence[str] = (),
    runner: Runner | None = None,
) -> tuple[Path, bool]:
    before = path.read_text(encoding="utf-8") if path.is_file() else None
    if elevate:
        if runner is None:
            raise ServiceError(
                "an elevated write needs a runner to elevate through; this is a "
                "caller bug, not a host problem"
            )
        handle, staged_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".staged")
        os.close(handle)
        staged = Path(staged_name)
        staged.write_text(text, encoding="utf-8")
        try:
            _require(
                runner,
                [*elevate, "install", "-D", "-m", "0644", str(staged), str(path)],
                f"{path} could not be written as root",
            )
        finally:
            staged.unlink(missing_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return path, before is not None and before != text


def install(
    mechanism: str,
    *,
    home: Path,
    server_name: str,
    executable: str,
    crucible_home: Path,
    host: str,
    port: int,
    runner: Runner,
    path_value: str | None = None,
    user: str | None = None,
) -> list[str]:
    recorded = hosttools.search_path() if path_value is None else path_value
    program = console_script(executable)
    recorded = path_including_program_dir(recorded, program)
    if mechanism == SYSTEMD:
        return _install_systemd(
            home=home, server_name=server_name, program=program, crucible_home=crucible_home,
            host=host, port=port, runner=runner, recorded=recorded, user=user,
        )
    if mechanism == LAUNCHD:
        return _install_launchd(
            home=home, program=program, crucible_home=crucible_home,
            host=host, port=port, runner=runner, recorded=recorded,
        )
    raise ServiceError(f"there is no service mechanism called {mechanism!r}")


def _install_systemd(
    *, home: Path, server_name: str, program: str, crucible_home: Path, host: str,
    port: int, runner: Runner, recorded: str, user: str | None,
) -> list[str]:
    scope = systemd_scope()
    who = user if user is not None else getpass.getuser()
    elevate = writing_door(scope)
    path, changed = write_definition(
        unit_path(home, scope),
        systemd_unit_text(
            server_name=server_name,
            program=program,
            crucible_home=crucible_home,
            host=host,
            port=port,
            path_value=recorded,
            run_as=(who if scope == SYSTEM_SCOPE else None),
        ),
        elevate=elevate,
        runner=runner,
    )
    lines = [f"wrote {path}"]
    lines += _enable_systemd_unit(runner, scope, elevate, changed=changed)
    lines.append(f"runs: {program} serve")
    lines.append(f"PATH recorded: {recorded}")
    if scope == SYSTEM_SCOPE:
        lines.append(
            f"scope: system unit, running as {who} — it starts with the "
            "distro and needs no linger"
        )
        return lines
    return lines + _linger_lines(read_linger(runner, who), who)


def _enable_systemd_unit(runner: Runner, scope: str, elevate: list[str], *, changed: bool) -> list[str]:
    _require(
        runner,
        [*elevate, *systemctl_argv(scope, "daemon-reload")],
        "systemd would not reload its units",
    )
    _require(
        runner,
        [*elevate, *systemctl_argv(scope, "enable", "--now", UNIT_NAME)],
        f"systemd would not enable and start {UNIT_NAME}",
    )
    lines = [f"enabled and started {UNIT_NAME}"]
    if changed:
        _require(
            runner,
            [*elevate, *systemctl_argv(scope, "restart", UNIT_NAME)],
            f"systemd would not restart {UNIT_NAME} onto its new definition",
        )
        lines.append(f"restarted {UNIT_NAME} onto its new definition")
    return lines


def _linger_lines(linger: bool | None, who: str) -> list[str]:
    if linger is True:
        return [
            f"linger: on for {who} — this server survives a logout and starts "
            "at boot"
        ]
    if linger is False:
        return [
            f"linger: OFF for {who}. A user service stops when that user's "
            "last session ends, so this Crucible will die with your shell and "
            "will not come back at boot. Granting it is yours to do:",
            f"    sudo loginctl enable-linger {who}",
        ]
    return [
        f"linger: UNKNOWN — loginctl could not be asked about {who}, so "
        "nothing here knows whether this server survives a logout. On a "
        f"host that has loginctl: sudo loginctl enable-linger {who}"
    ]


def _install_launchd(
    *, home: Path, program: str, crucible_home: Path, host: str, port: int,
    runner: Runner, recorded: str,
) -> list[str]:
    log_path = serve_log_path(crucible_home)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    path, _changed = write_definition(
        plist_path(home),
        launchd_plist_text(
            program=program,
            crucible_home=crucible_home,
            host=host,
            port=port,
            path_value=recorded,
            log_path=log_path,
        ),
    )
    lines = [f"wrote {path}"]
    if _agent_is_loaded(runner):
        _require(
            runner,
            ["launchctl", "bootout", _agent_target()],
            f"launchd would not unload the {LAUNCHD_LABEL} agent it already has",
        )
        lines.append(f"unloaded the previous {LAUNCHD_LABEL}")
    _require(
        runner,
        ["launchctl", "bootstrap", _domain(), str(path)],
        f"launchd would not load {path}",
    )
    _require(
        runner,
        ["launchctl", "kickstart", "-k", _agent_target()],
        f"launchd loaded {LAUNCHD_LABEL} but would not start it",
    )
    lines.append(f"loaded and started {LAUNCHD_LABEL} in {_domain()}")
    lines.append(f"runs: {program} serve")
    lines.append(f"PATH recorded: {recorded}")
    lines.append(f"stdout and stderr: {log_path}")
    return lines


def uninstall(mechanism: str, *, home: Path, runner: Runner) -> list[str]:
    lines: list[str] = []
    if mechanism == SYSTEMD:
        scope = installed_scope(home)
        if scope is None:
            return [f"nothing to remove: there is no unit at {unit_path(home, systemd_scope())}"]
        path = unit_path(home, scope)
        elevate = writing_door(scope)
        _require(
            runner,
            [*elevate, *systemctl_argv(scope, "disable", "--now", UNIT_NAME)],
            f"systemd would not stop and disable {UNIT_NAME}",
        )
        lines.append(f"stopped and disabled {UNIT_NAME}")
        if elevate:
            _require(runner, [*elevate, "rm", "-f", str(path)], f"{path} could not be removed as root")
        else:
            path.unlink()
        lines.append(f"removed {path}")
        _require(
            runner,
            [*elevate, *systemctl_argv(scope, "daemon-reload")],
            "systemd would not reload its units",
        )
        return lines

    if mechanism == LAUNCHD:
        path = plist_path(home)
        if _agent_is_loaded(runner):
            _require(
                runner,
                ["launchctl", "bootout", _agent_target()],
                f"launchd would not unload {LAUNCHD_LABEL}",
            )
            lines.append(f"unloaded {LAUNCHD_LABEL}")
        if path.is_file():
            path.unlink()
            lines.append(f"removed {path}")
        if not lines:
            return [f"nothing to remove: there is no agent and no plist at {path}"]
        return lines

    raise ServiceError(f"there is no service mechanism called {mechanism!r}")


def start(mechanism: str, *, home: Path, runner: Runner) -> list[str]:
    path = definition_path(mechanism, home)
    if not path.is_file():
        raise ServiceError(
            f"there is no Crucible service on this host: {path} does not exist. "
            "Run `crucible service install` first — starting a service nobody has "
            "defined is not something this can guess at"
        )
    if mechanism == SYSTEMD:
        scope = acting_scope(home, "start")
        _require(
            runner,
            [*writing_door(scope), *systemctl_argv(scope, "start", UNIT_NAME)],
            f"systemd would not start {UNIT_NAME}",
        )
        return [f"started {UNIT_NAME}"]

    if mechanism == LAUNCHD:
        lines: list[str] = []
        if not _agent_is_loaded(runner):
            _require(
                runner,
                ["launchctl", "bootstrap", _domain(), str(path)],
                f"launchd would not load {path}",
            )
            lines.append(f"loaded {LAUNCHD_LABEL}")
        _require(
            runner,
            ["launchctl", "kickstart", _agent_target()],
            f"launchd would not start {LAUNCHD_LABEL}",
        )
        lines.append(f"started {LAUNCHD_LABEL}")
        return lines

    raise ServiceError(f"there is no service mechanism called {mechanism!r}")


def stop(mechanism: str, *, home: Path, runner: Runner) -> list[str]:
    if mechanism == SYSTEMD:
        scope = installed_scope(home)
        if scope is None:
            return [f"nothing to stop: there is no {UNIT_NAME} on this machine"]
        _require(
            runner,
            [*writing_door(scope), *systemctl_argv(scope, "stop", UNIT_NAME)],
            f"systemd would not stop {UNIT_NAME}",
        )
        return [f"stopped {UNIT_NAME}"]

    if mechanism == LAUNCHD:
        if not _agent_is_loaded(runner):
            return [f"{LAUNCHD_LABEL} is not loaded; nothing to stop"]
        _require(
            runner,
            ["launchctl", "bootout", _agent_target()],
            f"launchd would not unload {LAUNCHD_LABEL}",
        )
        return [f"unloaded {LAUNCHD_LABEL} (the plist stays; `start` brings it back)"]

    raise ServiceError(f"there is no service mechanism called {mechanism!r}")


__all__ = [
    "CONSOLE_SCRIPT",
    "LAUNCHD",
    "LAUNCHD_LABEL",
    "Ran",
    "Runner",
    "SERVICE_MECHANISM",
    "SYSTEMD",
    "ServiceError",
    "Status",
    "UNIT_NAME",
    "console_script",
    "definition_path",
    "install",
    "launchd_plist_text",
    "read_recorded_path",
    "mechanism_for",
    "parse_launchctl_list",
    "parse_systemctl_show",
    "plist_path",
    "read_linger",
    "serve_log_path",
    "start",
    "status",
    "stop",
    "subprocess_runner",
    "systemd_unit_text",
    "uninstall",
    "unit_path",
    "user_home",
    "write_definition",
]
