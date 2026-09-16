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

They are user-level by default, deliberately: a user unit needs no privileges,
and a system unit would run as a user with no HuggingFace cache and no conda env
unless told whose server it is. The cost is stated below, in `linger`.

**WSL is the one exception, and it is not a preference.** WSLg overmounts
`/run/user/<uid>` and hides the user manager's D-Bus socket, so a user unit
there is one an orchestrator cannot reach (`systemd_scope`). In WSL the unit is
therefore a SYSTEM unit with `User=` naming the installing account — same
account, same caches, same conda env, a manager that answers. It needs root to
install, and `root_prefix` is the one place that says how root is reached.

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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
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

#: How long systemd waits before restarting a stopped server. Two seconds and
#: not five: with `Restart=always` (see `systemd_unit_text`) this is also the
#: gap a person waits after `Restart engine` on the host's menu, and five
#: seconds of a tray saying "starting…" for a restart that takes one is a
#: number chosen for a crash loop being read as a number chosen for a person.
RESTART_SECONDS = 2

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


#: The two systemd scopes this server can be installed into.
USER_SCOPE = "user"
SYSTEM_SCOPE = "system"

#: Where a SYSTEM unit lives. Not under `home`: a system unit is the machine's,
#: not a user's, and that is the whole point of using one.
SYSTEM_UNIT_DIR = Path("/etc/systemd/system")

#: The kernel's own answer to "is this WSL". A module constant so a test can
#: point it somewhere instead of monkeypatching `Path.read_text` globally.
OSRELEASE = Path("/proc/sys/kernel/osrelease")


def in_wsl() -> bool:
    """True inside WSL, read from the KERNEL and not from the environment.

    `WSL_DISTRO_NAME` and `WSL_INTEROP` are set for a login shell and are
    ABSENT from a unit's environment — measured 2026-09-16 — so a service that
    asked them would decide it was not in WSL precisely when it is running as
    the service. `/proc/sys/kernel/osrelease` is the kernel's own answer
    (`6.6.87.1-microsoft-standard-WSL2`) and is there for every process.
    """
    try:
        release = OSRELEASE.read_text(encoding="utf-8")
    except OSError:
        return False
    return "microsoft" in release.lower()


def systemd_scope() -> str:
    """SYSTEM inside WSL, USER everywhere else.

    WSLg mounts its own tmpfs over `/run/user/<uid>`, which HIDES the D-Bus
    socket systemd's user manager is listening on. `systemctl --user` then
    fails "Failed to connect to bus" for every caller, the Windows
    orchestrator's unit probe fails with it, and the engine silently drops to
    `owner=found` — at which point Windows can no longer restart or upgrade the
    engine it exists to manage. WSLg is ON BY DEFAULT, so this is the ordinary
    state of a stock WSL2, not a local misconfiguration.

    The system manager has no such problem: `/run/dbus/system_bus_socket` is
    not overmounted (measured — `findmnt /run/dbus` returns nothing), and the
    Windows side reaches it through `wsl.exe -u root`, which needs no password.

    Outside WSL this changes nothing: a user unit needs no privileges and
    works, and asking a Linux operator for root to install their own server
    would be a cost paid for somebody else's bug.
    """
    return SYSTEM_SCOPE if in_wsl() else USER_SCOPE


def systemctl_argv(scope: str, *verbs: str) -> list[str]:
    """`systemctl [--user] <verbs…>` for this scope. One place decides."""
    prefix = ("--user",) if scope == USER_SCOPE else ()
    return ["systemctl", *prefix, *verbs]


#: What WSL calls the distro this process is inside. Set for a login shell and
#: for `install.sh`; ABSENT from a unit's environment, which costs nothing —
#: a running unit installs nothing.
WSL_DISTRO_ENV = "WSL_DISTRO_NAME"


def root_prefix(environ: Mapping[str, str] | None = None) -> list[str]:
    """The argv prefix that runs a command as root, or `[]` when already root.

    A SYSTEM unit needs root to install, and inside WSL there is no password to
    give: `sudo -n true` on a stock Ubuntu answers "a password is required"
    (measured 2026-09-16), and an install driven by the Windows orchestrator has
    no terminal to type one into. The door that needs no password is the
    WINDOWS one — `wsl.exe -u root` grants root to any distro without asking —
    and interop makes it reachable from INSIDE the distro too.

    So this is the same door `crucible/host/presence.py` opens to restart the
    unit, approached from the other side, and the system has one answer to "how
    does Crucible get root in WSL" instead of two.

    Outside WSL nothing calls this: `systemd_scope` returns the user scope,
    which needs no privileges at all.
    """
    if not hasattr(os, "geteuid"):
        raise ServiceError(
            "a system unit is a POSIX thing and this platform has no euid; "
            "nothing here should have asked for one"
        )
    if os.geteuid() == 0:
        return []
    distro = (os.environ if environ is None else environ).get(WSL_DISTRO_ENV)
    if not distro:
        raise ServiceError(
            f"the system unit needs root and ${WSL_DISTRO_ENV} is unset, so the "
            "`wsl.exe -u root` door cannot be named. Install from a WSL shell, "
            "or run this as root"
        )
    return ["wsl.exe", "-d", distro, "-u", "root", "--exec"]


def writing_door(scope: str) -> list[str]:
    """The prefix a systemd verb that CHANGES something needs in this scope.

    Reading is free: `systemctl show` against the system manager answers any
    user, and elevating it would buy a `wsl.exe` round trip per status poll for
    nothing. Starting, stopping, enabling and writing the unit file are the
    verbs that need root, and they all come through here.
    """
    return root_prefix() if scope == SYSTEM_SCOPE else []


def unit_path(home: Path, scope: str | None = None) -> Path:
    """`~/.config/systemd/user/crucible.service`, or the system unit in WSL."""
    if (scope if scope is not None else systemd_scope()) == SYSTEM_SCOPE:
        return SYSTEM_UNIT_DIR / UNIT_NAME
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
        return unit_path(home)  # scope-aware: the system unit inside WSL
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
    run_as: str | None = None,
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

    **`Restart=always`, and this reverses an earlier reading of the same
    question** (RULING, PHASE15-HOST.md 4.1, 2026-09-14). The old comment here
    said `on-failure` "because a server that exited 0 was stopped on purpose,
    and restarting it would make `crucible service stop` a thing that does not
    work". The first half of that is not true of systemd and the second half
    does not follow from it. `systemctl --user stop` puts the unit in the
    STOPPED state, and `Restart=` is not consulted for a unit systemd itself
    stopped — so `always` and `stop` coexist. What `on-failure` actually bought
    was the defect of 2026-09-14: a clean `SIGTERM` (a `wsl --terminate`, an
    OOM killer's polite half, a shutdown that raced the guest) exits 0, and the
    engine then stayed down at 16:10 with nothing noticing, because on Windows
    nothing was watching. `crucible host` is now the thing that watches, and the
    unit's own `Restart=` is what it deliberately does NOT reimplement (4.1:
    "It never loops on restart; the systemd unit's own `Restart=` handles
    crashes") — so the unit has to be the half that is total.

    The launchd agent below keeps `SuccessfulExit: false` and is NOT changed
    with it. There is no host on the Mac (4.4) and therefore no second watcher
    to divide the work with; `launchctl stop` there really is the only way a
    person stops one, and `KeepAlive: true` would undo it.

    `WantedBy=default.target` is the user-session equivalent of multi-user;
    `loginctl enable-linger` is what makes that survive a logout, and it is the
    operator's to grant — see `read_linger`.

    `%` is doubled because systemd expands `%x` specifiers in a unit file, and a
    PATH or a home directory with a percent sign in it would otherwise reach the
    service as something else entirely.

    **`Environment=` VALUES ARE QUOTED, AND THAT IS NOT COSMETIC.** `Environment=`
    is the one directive here that takes a SPACE-SEPARATED LIST of assignments,
    so an unquoted value containing a space is not one value with a space in it
    — it is the first word, and then a second assignment systemd cannot parse.
    On WSL the installing shell's PATH carries the Windows PATH through interop
    (`/mnt/c/Program Files/Git/usr/bin`), so owens-pc's unit was rejected word by
    word, MEASURED 2026-09-14 23:27:02 in the journal:

        crucible.service:12: Invalid environment assignment, ignoring:
        Files/Git/mingw64/bin:/mnt/c/Program

    and the service ran with the bare PATH the whole recorded-PATH mechanism
    above exists to prevent — silently, because the line that survives the
    splitting is a valid `PATH=` and the unit starts. A double quote or a
    backslash inside the value is refused rather than escaped: systemd's quoting
    has its own backslash rules, and a value that needs them is a value whose
    meaning has already stopped being obvious.
    """
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
        # A SYSTEM unit runs as root unless told whose server this is, and
        # this one owns a home, a token and a pairing file under ONE user.
        # The user scope needs no such line: it is already that user's manager.
        + (f"User={_one_line('run_as', run_as)}\n" if run_as else "")
        + "\n"
        + "[Install]\n"
        # `default.target` is the USER manager's "logged in". The system
        # manager's equivalent is `multi-user.target`, and a system unit
        # wanted by the former is enabled into a target that never runs.
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
    """`~/Library/LaunchAgents/com.crucible.serve.plist`, exactly.

    `program` is the console script and `WorkingDirectory` is `CRUCIBLE_HOME`,
    both for `systemd_unit_text`'s reasons — a launchd agent's default cwd is
    `/`, which does not have the PC's import problem today but is not a promise
    anybody made, and a server's cwd is its own state directory either way.

    `KeepAlive` is a dict with `SuccessfulExit` false rather than a bare
    `<true/>`: restart a crash, leave a deliberate stop alone. This is NO
    LONGER what the systemd unit does — see the ruling in `systemd_unit_text` —
    and the difference is deliberate rather than an oversight: that unit is
    watched by `crucible host`, which distinguishes "systemd stopped it" from
    "it died", and this agent is watched by nobody, so `launchctl stop` on the
    Mac is the only stop there is and `KeepAlive: true` would undo it.
    `RunAtLoad` is what starts it at login.

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
            if separator != "=" or name.strip() != "Environment":
                continue
            value = value.strip()
            # QUOTED SINCE 2026-09-15, and both shapes are read. A unit written
            # before that carries `Environment=PATH=...` bare, and it is on
            # disk until the operator reinstalls the service — a reader that
            # knew only the new shape would answer "no recorded PATH" for a
            # service that has one, which is the third `None` above pretending
            # to be the second.
            if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
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
        scope = systemd_scope()
        ran = runner(
            systemctl_argv(
                scope,
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


def write_definition(
    path: Path,
    text: str,
    *,
    elevate: Sequence[str] = (),
    runner: Runner | None = None,
) -> tuple[Path, bool]:
    """Write the unit or the plist. Returns the path and whether it CHANGED.

    `elevate` is `root_prefix`'s answer: empty when this process can write the
    path itself, and the `wsl.exe -u root` door when it cannot. The text is
    staged beside the temp directory and moved into place by `install(1)`
    rather than piped, because the runner speaks argv and not stdin — and the
    unit carries no secret (the token lives in `config.toml`), so a staged copy
    exposes nothing.

    The second half is not bookkeeping. `systemctl enable --now` starts a unit
    that is STOPPED and does nothing at all to one already running, which is
    exactly the state an upgrade finds. So rewriting ExecStart and calling it
    leaves the OLD executable serving while every line this function prints
    says the new one was installed. Measured 2026-09-16: a guest upgraded to
    0.6.3 went on answering /v1/info with 0.6.0 out of the previous release's
    conda path, and nothing in the install said so. The caller restarts when
    this says True.
    """
    before = path.read_text(encoding="utf-8") if path.is_file() else None
    if elevate:
        if runner is None:
            raise ServiceError(
                "an elevated write needs a runner to elevate through; this is a "
                "caller bug, not a host problem"
            )
        # A UNIQUE name: `/tmp` is shared and sticky, so a fixed one could
        # collide with another account's file and be unwritable.
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
    # REPLACED, not merely "differs". On a first install `before` is None and
    # the unit is about to be started by `enable --now` with nothing stale
    # behind it, so a restart there would be an outage bought for nothing.
    # What matters is a definition that MOVED under a running process.
    return path, before is not None and before != text


def retire_user_unit(home: Path, runner: Runner, elevate: Sequence[str]) -> list[str]:
    """Take down a pre-7b.9 USER unit before a SYSTEM unit takes its port.

    A guest installed before the scope moved has `crucible.service` under
    `~/.config/systemd/user`, enabled, running, and holding 7100. The new
    system unit's `enable --now` would then fail to bind, and the failure would
    read as a port conflict rather than as the upgrade it is.

    It cannot be retired through its own manager. The reason the scope moved at
    all is that WSLg overmounts `/run/user/<uid>` and HIDES that manager's bus
    socket (`systemd_scope`), so `systemctl --user` there fails for root and
    user alike — setting `XDG_RUNTIME_DIR` does not help, because the socket is
    not missing, it is covered.

    What always answers is the SYSTEM manager, which owns `user@<uid>.service`.
    Stopping that stops every unit the user manager was running, this one
    included. The file is removed FIRST, so a manager that comes back — linger
    brings it back on demand — comes back without it. One mechanism, no
    second-guessing about which door happens to be open.

    The cost is honest: any other service that user was running in this distro
    restarts. A WSL distro that exists to hold an inference server is the case
    this is for, and a Linux host never reaches here at all.
    """
    stale = unit_path(home, USER_SCOPE)
    if not stale.is_file():
        return []
    # STOP FIRST, then remove. A stop that fails must leave the file where it
    # is: the next install then sees a stale unit and retires it again, instead
    # of finding nothing to retire and meeting the old server at the port.
    _require(
        runner,
        [*elevate, "systemctl", "stop", f"user@{os.getuid()}.service"],
        "the old user manager would not stop, so its Crucible still holds the port",
    )
    stale.unlink()
    return [
        f"retired the user unit at {stale} and stopped the user manager that "
        "was running it"
    ]


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
        scope = systemd_scope()
        who = user if user is not None else getpass.getuser()
        # A system unit lives in `/etc` and is driven through the system
        # manager, both of which need root. `root_prefix` is the ONE place that
        # says how root is reached, and it answers `[]` for the user scope and
        # for a process that already is root.
        elevate = writing_door(scope)
        if scope == SYSTEM_SCOPE:
            lines.extend(retire_user_unit(home, runner, elevate))
        path, changed = write_definition(
            unit_path(home, scope),
            systemd_unit_text(
                server_name=server_name,
                program=program,
                crucible_home=crucible_home,
                host=host,
                port=port,
                path_value=recorded,
                # A system unit must be told whose server it is; a user unit
                # already is. `user` is what the caller states, falling back
                # to the account doing the installing.
                run_as=(who if scope == SYSTEM_SCOPE else None),
            ),
            elevate=elevate,
            runner=runner,
        )
        lines.append(f"wrote {path}")
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
        lines.append(f"enabled and started {UNIT_NAME}")
        if changed:
            # THE DEFINITION MOVED, SO THE RUNNING PROCESS IS STALE. The
            # `enable --now` above does nothing to a unit that is already
            # active. Without this the old executable keeps serving and the
            # install still reports success — the one failure here that says
            # nothing at all.
            _require(
                runner,
                [*elevate, *systemctl_argv(scope, "restart", UNIT_NAME)],
                f"systemd would not restart {UNIT_NAME} onto its new definition",
            )
            lines.append(f"restarted {UNIT_NAME} onto its new definition")
        lines.append(f"runs: {program} serve")
        lines.append(f"PATH recorded: {recorded}")
        if scope == SYSTEM_SCOPE:
            # LINGER IS A USER-MANAGER FACT and says nothing about a system
            # unit. Printing "this will die with your shell" over a unit
            # `multi-user.target` starts at boot would be a true sentence about
            # the wrong thing — the shape docs/ARCHITECTURE.md R1 is about.
            lines.append(
                f"scope: system unit, running as {who} — it starts with the "
                "distro and needs no linger"
            )
            return lines

        linger = read_linger(runner, who)
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
        scope = systemd_scope()
        path = unit_path(home, scope)
        if not path.is_file():
            return [f"nothing to remove: there is no unit at {path}"]
        elevate = writing_door(scope)
        _require(
            runner,
            [*elevate, *systemctl_argv(scope, "disable", "--now", UNIT_NAME)],
            f"systemd would not stop and disable {UNIT_NAME}",
        )
        lines.append(f"stopped and disabled {UNIT_NAME}")
        if elevate:
            # `/etc/systemd/system` is root's, so the unlink is too.
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
    """Make sure the service is running. Idempotent; refuses if none is installed."""
    path = definition_path(mechanism, home)
    if not path.is_file():
        raise ServiceError(
            f"there is no Crucible service on this host: {path} does not exist. "
            "Run `crucible service install` first — starting a service nobody has "
            "defined is not something this can guess at"
        )
    if mechanism == SYSTEMD:
        scope = systemd_scope()
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
    """Stop the service without forgetting it. Idempotent.

    On launchd that is `bootout` rather than a signal, and the reason is
    `KeepAlive`: an agent killed with SIGTERM exited unsuccessfully, so launchd
    would start it straight back up. Unloading it is the only stop that stops.
    The plist stays, so `RunAtLoad` brings it back at the next login and
    `crucible service start` brings it back now.
    """
    if mechanism == SYSTEMD:
        scope = systemd_scope()
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
