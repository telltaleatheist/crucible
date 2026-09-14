"""`crucible service` — a local Crucible is a SERVICE, and no app owns it.

Owen's ruling, 2026-09-13, asked whether a local Crucible is a machine service or
an app's child process: **"service"** (PHASE5-APPS.md section 6.0). Everything in
this module follows from that one word:

- **Nobody owns it, so nobody has to be chosen as the owner.** When Foundry is
  hosted inside BookForge, the question of which of them installs and starts the
  server simply stops existing. Both call an idempotent "make sure this machine
  has one" and connect.
- **It survives the app.** A Crucible holding a 19 GB model must not die because
  somebody closed a window, and a render must not die with it.
- **Nothing can be orphaned that was never anyone's child.** Foundry's
  `mount.ts:988` names the hazard this removes: a crash that leaves a guest
  process holding the card with nothing to SIGTERM it.

Two mechanisms, one per backend, and no third
---------------------------------------------
`cuda-linux` gets a **systemd user unit** and `mlx-darwin` gets a **launchd
agent**. A backend this module has no mechanism for is refused by name rather
than served with a guess — DESIGN.md section 2's list of backends is short and
explicit, and so is this one.

They are user-level in both cases, deliberately. A system unit would need root to
install, would run as a user with no HuggingFace cache and no conda env, and
would put a server that holds one operator's models outside that operator's
control. The cost is stated below, in `linger`.

THE PATH IS RECORDED, AND THIS IS THE BUG THAT MADE IT NECESSARY
----------------------------------------------------------------
A systemd user unit and a launchd agent are started with a **bare PATH** —
measured as `/usr/bin:/bin:/usr/sbin:/sbin` on Owen's Mac — and a `crucible
doctor` run over a non-login shell on that same Mac reported `job tts: NOT READY
— there is no ffmpeg on PATH` while ffmpeg sat at `/opt/homebrew/bin/ffmpeg` the
whole time. A service installed without a PATH would fail in exactly that way, on
a fully installed host, and the failure would arrive as a refused job rather than
as anything an operator could see at install time.

So `install` writes the **installing shell's** PATH into the unit or the plist
(`crucible/hosttools.py` is its one owner), and `crucible doctor` names the PATH
it searched whenever it reports a tool missing. Hardcoding a Homebrew prefix
would fix one Mac, and would be a second owner of a fact the environment already
holds.

**And `doctor` reads the recorded PATH back, beside its own.** The Mac audit of
2026-09-14 found the same message a second time — `job tts: NOT READY — there
is no ffmpeg on PATH` over `ssh mac '<cmd>'` — and this time the service was
perfectly healthy: the plist carried `/opt/homebrew/bin`, `launchctl print`
confirmed the running process had it, and the only thing missing a PATH was the
non-login shell the operator happened to be typing in. Naming one PATH was
therefore not enough; the reader's real question is *which of the two*, and
Crucible is the thing that wrote the other one, so `read_recorded_path()` below
answers it and `crucible doctor` prints both lines. `None` is a real answer —
no service is installed — and is reported as that rather than as an empty
PATH.

**The server's own `bin/` is APPENDED to that PATH, and only appended.** Since
0.6.0 the server can arrive as an env pack (PHASE14-ENVPACKS.md), and then the
shell that runs `crucible service install` is a `wsl.exe --exec` shell whose
PATH is the guest's default — which cannot contain a directory that was created
a minute earlier. Recording it is what makes "this is the PATH the service has"
true of the process rather than of the installer. **Appended and never
prepended**, because the pack's `bin/` also holds `python3`, `uvicorn` and half
a dozen of its dependencies' scripts, and putting those in front of a host's
own would silently change what every bare name means in order to fix nothing.
This closes no live defect — `tasks.install_command()` resolves the script
beside `sys.executable` before it ever looks at PATH — and is here so that a
bare name inside the server resolves to the server's own neighbour instead of
to nothing.

`CRUCIBLE_HOME` is recorded for the same reason and it is not optional: a service
started without it serves `~/.crucible`, which on a host where the operator set
`CRUCIBLE_HOME` is a **different server with a different token**. The value
written is the one this process resolved, so the service and the shell that
installed it can never be looking at two configs.

Every subprocess goes through one injectable runner
---------------------------------------------------
`systemctl` and `launchctl` are reached only through `Runner`, so the tests never
call either — they assert on the argv that would have been run and on the text
that would have been written, byte for byte. The unit and the plist are produced
by pure functions for the same reason: a service definition is the one artefact
here that nobody sees until the machine reboots.
"""

from __future__ import annotations

import getpass
import os
import plistlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from xml.sax.saxutils import escape as xml_escape

from . import hosttools
from .backend import CUDA_LINUX, MLX_DARWIN
from .errors import CrucibleError

#: Which mechanism supervises a Crucible on each backend. A backend that is not
#: in here is refused by name — see `mechanism_for`.
SERVICE_MECHANISM: dict[str, str] = {
    CUDA_LINUX: "systemd",
    MLX_DARWIN: "launchd",
}

SYSTEMD = "systemd"
LAUNCHD = "launchd"

#: The systemd unit's name, and the launchd agent's label. Both are fixed: a
#: machine runs one Crucible, and a second one on the same host is a second
#: `CRUCIBLE_HOME`, a second port and a second checkout, which is a thing an
#: operator does by hand and not a thing this verb parameterises.
UNIT_NAME = "crucible.service"
LAUNCHD_LABEL = "com.crucible.serve"

#: How long systemd waits before restarting a crashed server.
RESTART_SECONDS = 5

#: The console script `pip install -e .` puts beside the interpreter. **The unit
#: runs THIS and never `python -m crucible`**, and that is not a style choice —
#: see `console_script`.
CONSOLE_SCRIPT = "crucible"


class ServiceError(CrucibleError):
    """A service could not be installed, started, stopped or read. Names which."""


# ------------------------------------------------------------------- locating


def user_home() -> Path:
    """The operator's home directory.

    A module-level probe so a test can replace it and never write into a real
    `~/.config` or `~/Library` — the same reason `crucible/jobs/asr` keeps
    `ffmpeg_path` at module level.
    """
    return Path.home()


def unit_path(home: Path) -> Path:
    """`~/.config/systemd/user/crucible.service`."""
    return home / ".config" / "systemd" / "user" / UNIT_NAME


def plist_path(home: Path) -> Path:
    """`~/Library/LaunchAgents/com.crucible.serve.plist`."""
    return home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def serve_log_path(crucible_home: Path) -> Path:
    """Where launchd points the server's stdout and stderr.

    Inside `CRUCIBLE_HOME`, beside every other log this server writes, rather
    than in `~/Library/Logs`: a reader looking for what a Crucible did should
    find all of it in one tree, and `crucible doctor` already prints that tree.
    """
    return crucible_home / "logs" / "serve.log"


def definition_path(mechanism: str, home: Path) -> Path:
    """The unit or the plist, whichever this host uses."""
    if mechanism == SYSTEMD:
        return unit_path(home)
    if mechanism == LAUNCHD:
        return plist_path(home)
    raise ServiceError(f"there is no service mechanism called {mechanism!r}")


def console_script(executable: str) -> str:
    """The `crucible` console script beside this interpreter, or a refusal.

    **MEASURED, on Owen's PC, 2026-09-13, by installing the unit this module's
    first version generated.** It ran `python -m crucible serve`, and it
    crash-looped:

        ImportError: cannot import name 'load_all_voices' from
        'crucible.voices' (unknown location)

    A systemd user unit with no `WorkingDirectory` starts in `$HOME`, and
    `$HOME` on that box holds the Linux checkout — a directory called
    `crucible`. `python -m` puts the cwd on `sys.path`, so `crucible.voices`
    resolved to `~/crucible/voices/`, the manifest DIRECTORY, as a namespace
    package with `__file__ is None`, instead of to `crucible/voices.py`.
    Reproduced exactly: from `$HOME`, `import crucible.voices` gives `__file__
    None`; from `/` it gives the real module.

    A console script cannot do that. Its `sys.path[0]` is the script's own
    directory, never the cwd, so what it imports does not depend on where it was
    started. `WorkingDirectory` is set as well (see the generators) — belt and
    braces, because the two failures are different: one is about imports, the
    other about where a relative path in a log line lands.

    Refused by name when it is not there, rather than falling back to `python
    -m`: a fallback here reinstates precisely the bug above, on the machine
    where the checkout is in the home directory, which is the machine that has
    it.
    """
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
    """`path_value` with the server's own `bin/` at the END, once.

    See the module docstring for why it is there and why it is appended rather
    than prepended. Idempotent: a directory already on the PATH stays where it
    is, so re-running `service install` from a shell that HAS the pack on its
    PATH does not move it behind the rest.
    """
    directory = str(Path(program).resolve().parent)
    entries = [entry for entry in path_value.split(os.pathsep) if entry != ""]
    if directory in entries:
        return path_value
    return os.pathsep.join([*entries, directory])


def mechanism_for(backend_kind: str) -> str:
    """This backend's service mechanism, or a refusal naming the backend.

    Never a default. A backend with no mechanism is a backend nobody has decided
    how to supervise, and inventing one here would install a unit that does not
    start anything.
    """
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


# --------------------------------------------------------------- the runner


@dataclass(frozen=True)
class Ran:
    """One external command and what it said. The whole of this module's I/O."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def text(self) -> str:
        """stderr if there is any, else stdout, else a placeholder. For messages."""
        return (self.stderr.strip() or self.stdout.strip()) or "no output"


#: What every verb below is handed instead of `subprocess`. A test passes one
#: that records and answers; nothing in the suite runs systemctl or launchctl.
Runner = Callable[[Sequence[str]], Ran]


def subprocess_runner(argv: Sequence[str]) -> Ran:
    """The real one. Captures both streams; never raises on a non-zero exit.

    A non-zero exit is data here, not an exception: `launchctl print` returning
    non-zero is how this module asks *"is the agent loaded?"*, and a runner that
    threw would make a question into a failure.
    """
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
    """Run it, or refuse by name quoting what it said."""
    ran = runner(argv)
    if not ran.ok:
        raise ServiceError(
            f"{what}: `{' '.join(ran.argv)}` exited {ran.returncode}: {ran.text()}"
        )
    return ran


# ----------------------------------------------------- the generated text


def _one_line(where: str, value: str) -> str:
    """A value that is about to be interpolated into a unit or a plist.

    A newline in any of these would end the directive it sits in and start
    something else, so it is refused rather than stripped: a stripped newline
    changes the value silently, and the value is a path to somebody's weights.
    """
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
) -> str:
    """`~/.config/systemd/user/crucible.service`, exactly.

    `program` is the **console script**, never `<python> -m crucible`. The
    difference is an ImportError on Owen's PC and `console_script` is where the
    whole finding is written down.

    `WorkingDirectory` is `CRUCIBLE_HOME` and not the operator's `$HOME`, which
    is where a user unit otherwise starts. A server's cwd should be its own
    state directory: it is the only directory it owns, it is where every
    relative path it writes belongs, and `$HOME` is a place whose contents
    change with what the operator happens to have checked out — which is exactly
    how the import bug above happened.

    `Restart=on-failure` and not `always`: a server that exited 0 was stopped on
    purpose, and restarting it would make `crucible service stop` a thing that
    does not work. `WantedBy=default.target` is the user-session equivalent of
    multi-user; `loginctl enable-linger` is what makes that survive a logout, and
    it is the operator's to grant — see `read_linger`.

    `%` is doubled because systemd expands `%x` specifiers in a unit file, and a
    PATH or a home directory with a percent sign in it would otherwise reach the
    service as something else entirely.
    """
    def escape(where: str, value: str) -> str:
        return _one_line(where, value).replace("%", "%%")

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
        f"Environment=CRUCIBLE_HOME={escape('CRUCIBLE_HOME', str(crucible_home))}\n"
        f"Environment=PATH={escape('PATH', path_value)}\n"
        "Restart=on-failure\n"
        f"RestartSec={RESTART_SECONDS}\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
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
    """`~/Library/LaunchAgents/com.crucible.serve.plist`, exactly.

    `program` is the console script and `WorkingDirectory` is `CRUCIBLE_HOME`,
    both for `systemd_unit_text`'s reasons — a launchd agent's default cwd is
    `/`, which does not have the PC's import problem today but is not a promise
    anybody made, and a server's cwd is its own state directory either way.

    `KeepAlive` is a dict with `SuccessfulExit` false rather than a bare `<true/>`
    for `systemd_unit_text`'s reason: restart a crash, leave a deliberate stop
    alone. `RunAtLoad` is what starts it at login.

    Both streams go to one file, because they are one narrative: uvicorn logs
    requests on one and a traceback arrives on the other, and reading them
    interleaved is the only way to see which request produced it.
    """
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


# ------------------------------------------------------------------- reading


@dataclass(frozen=True)
class Status:
    """What `crucible service status` reports, and what `--json` prints."""

    mechanism: str
    #: The unit or the plist. Named whether or not it exists, because "it is not
    #: there" is only useful next to where it would have been.
    definition: Path
    installed: bool
    running: bool
    pid: int | None
    detail: str
    #: systemd only. `True`/`False` when `loginctl` answered, `None` when it
    #: could not be asked — which is REPORTED and never read as "off". On
    #: launchd it is `None` because the question does not exist there.
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
    """The PATH the SERVICE was installed with, read back out of what we wrote.

    Not what this process has — `crucible/hosttools.py` owns that — and not a
    guess. It is parsed out of the unit or the plist because Crucible is what
    put it there, so the two can never disagree about what the service will
    see.

    Three different `None`s are deliberately one `None`: no definition file, a
    definition with no PATH in it (a unit written before 0.6.0), and a plist
    this build cannot read. All three mean "there is no recorded PATH to
    compare against", which is what a caller needs; the file's own path is
    already on `Status.definition` for anybody who wants to look.

    The plist is read with `plistlib` rather than a regex: `launchd_plist_text`
    XML-escapes the value, and a PATH with an `&` in it would come back wrong
    from anything that did not decode it.
    """
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
            if separator == "=" and name.strip() == "Environment":
                key, is_pair, recorded = value.partition("=")
                if is_pair == "=" and key == "PATH":
                    # `systemd_unit_text` doubles every `%` because systemd
                    # expands `%x` specifiers, so reading it back has to undo
                    # exactly that and nothing else.
                    return recorded.replace("%%", "%")
        return None
    except (OSError, ValueError, plistlib.InvalidFileException):
        return None


def parse_systemctl_show(text: str) -> dict[str, str]:
    """`systemctl show`'s `Key=Value` lines as a mapping.

    `show` rather than `status`, and this is the difference between a fact and a
    log scrape (ARCHITECTURE.md R4): `status` prints a human page whose wording
    changes with the version, `show` prints properties and has for a decade.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator == "=":
            values[key.strip()] = value.strip()
    return values


def parse_launchctl_list(text: str, label: str) -> tuple[bool, int | None]:
    """`launchctl list`'s three columns, for one label: (loaded, pid).

    The columns are `PID Status Label`, and a job that is loaded but not running
    prints `-` for the PID. A label that is not in the listing at all is not
    loaded, which is a different answer from "loaded with no pid" and is why this
    returns both.
    """
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1] == label:
            raw = parts[0]
            return True, (int(raw) if raw.lstrip("-").isdigit() and raw != "-" else None)
    return False, None


def read_linger(runner: Runner, user: str) -> bool | None:
    """Whether this user's session lingers, or None because it could not be asked.

    **Reported, never assumed.** `loginctl enable-linger` is the operator's to
    run — it is what makes a user service survive a logout and start at boot, and
    it is exactly the kind of thing a tool must not do to somebody's machine on
    its own. Without it a Crucible installed on a headless box dies the moment
    the installing SSH session ends, which is a surprise worth naming rather than
    a default worth setting.

    `None` is an honest third answer: a container with no `loginctl`, or a host
    where the call fails, has not said no.
    """
    ran = runner(["loginctl", "show-user", user, "--property=Linger"])
    if not ran.ok:
        return None
    value = parse_systemctl_show(ran.stdout).get("Linger")
    if value is None:
        return None
    return value.lower() == "yes"


def status(
    mechanism: str, home: Path, *, runner: Runner, user: str | None = None
) -> Status:
    """Is there a Crucible service on this host, and is it up?"""
    definition = definition_path(mechanism, home)
    installed = definition.is_file()
    if mechanism == SYSTEMD:
        ran = runner(
            [
                "systemctl",
                "--user",
                "show",
                UNIT_NAME,
                "--property=ActiveState",
                "--property=SubState",
                "--property=MainPID",
                "--property=UnitFileState",
            ]
        )
        linger = read_linger(runner, user if user is not None else getpass.getuser())
        if not ran.ok:
            return Status(
                mechanism=mechanism,
                definition=definition,
                installed=installed,
                running=False,
                pid=None,
                detail=(
                    f"systemctl could not be asked: `{' '.join(ran.argv)}` exited "
                    f"{ran.returncode}: {ran.text()}"
                ),
                linger=linger,
            )
        properties = parse_systemctl_show(ran.stdout)
        active = properties.get("ActiveState", "unknown")
        raw_pid = properties.get("MainPID", "0")
        pid = int(raw_pid) if raw_pid.isdigit() and raw_pid != "0" else None
        running = active == "active"
        return Status(
            mechanism=mechanism,
            definition=definition,
            installed=installed,
            running=running,
            pid=pid,
            detail=(
                f"{active}/{properties.get('SubState', 'unknown')}, unit file "
                f"{properties.get('UnitFileState', 'unknown')}"
            ),
            linger=linger,
        )

    if mechanism == LAUNCHD:
        ran = runner(["launchctl", "list"])
        if not ran.ok:
            return Status(
                mechanism=mechanism,
                definition=definition,
                installed=installed,
                running=False,
                pid=None,
                detail=(
                    f"launchctl could not be asked: `{' '.join(ran.argv)}` exited "
                    f"{ran.returncode}: {ran.text()}"
                ),
                linger=None,
            )
        loaded, pid = parse_launchctl_list(ran.stdout, LAUNCHD_LABEL)
        return Status(
            mechanism=mechanism,
            definition=definition,
            installed=installed,
            running=pid is not None,
            pid=pid,
            detail=(
                f"agent {LAUNCHD_LABEL} is "
                + ("loaded" if loaded else "not loaded")
                + (f" and running as pid {pid}" if pid is not None else "")
            ),
            linger=None,
        )

    raise ServiceError(f"there is no service mechanism called {mechanism!r}")


def _domain() -> str:
    """`gui/<uid>` — the launchd domain a per-user agent lives in."""
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


# ------------------------------------------------------------------- writing


def write_definition(path: Path, text: str) -> Path:
    """Write the unit or the plist, creating its directory. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


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
    """Write the definition and make the service run. Idempotent.

    `executable` is the interpreter this process is running under, and what goes
    into the definition is the **`crucible` console script beside it**, resolved
    and refused by name here rather than assembled in the text generators — see
    `console_script` for the ImportError that made that the rule.

    `path_value` defaults to the **installing shell's** PATH, which is the whole
    point of recording one — see the module docstring. It is a parameter so a
    test can pin it, not so a caller can invent one.

    The definition is rewritten every time, and that is deliberate: an install
    against a config whose port changed must produce a unit that serves the new
    port, and a unit left over from an older build is exactly the sort of stale
    second copy ARCHITECTURE.md R1 is about.
    """
    # Read through the module and not off a bound name: `hosttools.search_path`
    # is the one owner of this fact, and a `from … import` here would make a
    # second one that nothing could replace or correct.
    recorded = hosttools.search_path() if path_value is None else path_value
    program = console_script(executable)
    recorded = path_including_program_dir(recorded, program)
    lines: list[str] = []

    if mechanism == SYSTEMD:
        path = write_definition(
            unit_path(home),
            systemd_unit_text(
                server_name=server_name,
                program=program,
                crucible_home=crucible_home,
                host=host,
                port=port,
                path_value=recorded,
            ),
        )
        lines.append(f"wrote {path}")
        _require(
            runner,
            ["systemctl", "--user", "daemon-reload"],
            "systemd would not reload its user units",
        )
        _require(
            runner,
            ["systemctl", "--user", "enable", "--now", UNIT_NAME],
            f"systemd would not enable and start {UNIT_NAME}",
        )
        lines.append(f"enabled and started {UNIT_NAME}")
        lines.append(f"runs: {program} serve")
        lines.append(f"PATH recorded: {recorded}")
        linger = read_linger(runner, user if user is not None else getpass.getuser())
        who = user if user is not None else getpass.getuser()
        if linger is True:
            lines.append(
                f"linger: on for {who} — this server survives a logout and starts "
                "at boot"
            )
        elif linger is False:
            lines.append(
                f"linger: OFF for {who}. A user service stops when that user's "
                "last session ends, so this Crucible will die with your shell and "
                "will not come back at boot. Granting it is yours to do:"
            )
            lines.append(f"    sudo loginctl enable-linger {who}")
        else:
            lines.append(
                f"linger: UNKNOWN — loginctl could not be asked about {who}, so "
                "nothing here knows whether this server survives a logout. On a "
                f"host that has loginctl: sudo loginctl enable-linger {who}"
            )
        return lines

    if mechanism == LAUNCHD:
        log_path = serve_log_path(crucible_home)
        # launchd refuses to load an agent whose StandardOutPath directory does
        # not exist, and the failure it gives says nothing about the directory.
        log_path.parent.mkdir(parents=True, exist_ok=True)
        path = write_definition(
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
        lines.append(f"wrote {path}")
        # ASK, then act. `bootstrap` on an already-loaded agent fails, and the
        # reinstall case has to work, but swallowing the failure would also
        # swallow a real one. So the load state is a question with an answer
        # before anything is attempted — and the old definition is booted out,
        # because `bootstrap` is what reads the plist and `kickstart` is not.
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

    raise ServiceError(f"there is no service mechanism called {mechanism!r}")


def uninstall(mechanism: str, *, home: Path, runner: Runner) -> list[str]:
    """Stop the service, forget it, and remove its definition. Idempotent."""
    lines: list[str] = []
    if mechanism == SYSTEMD:
        path = unit_path(home)
        if not path.is_file():
            return [f"nothing to remove: there is no unit at {path}"]
        _require(
            runner,
            ["systemctl", "--user", "disable", "--now", UNIT_NAME],
            f"systemd would not stop and disable {UNIT_NAME}",
        )
        lines.append(f"stopped and disabled {UNIT_NAME}")
        path.unlink()
        lines.append(f"removed {path}")
        _require(
            runner,
            ["systemctl", "--user", "daemon-reload"],
            "systemd would not reload its user units",
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
    """Make sure the service is running. Idempotent; refuses if none is installed."""
    path = definition_path(mechanism, home)
    if not path.is_file():
        raise ServiceError(
            f"there is no Crucible service on this host: {path} does not exist. "
            "Run `crucible service install` first — starting a service nobody has "
            "defined is not something this can guess at"
        )
    if mechanism == SYSTEMD:
        _require(
            runner,
            ["systemctl", "--user", "start", UNIT_NAME],
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
    """Stop the service without forgetting it. Idempotent.

    On launchd that is `bootout` rather than a signal, and the reason is
    `KeepAlive`: an agent killed with SIGTERM exited unsuccessfully, so launchd
    would start it straight back up. Unloading it is the only stop that stops.
    The plist stays, so `RunAtLoad` brings it back at the next login and
    `crucible service start` brings it back now.
    """
    if mechanism == SYSTEMD:
        _require(
            runner,
            ["systemctl", "--user", "stop", UNIT_NAME],
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
