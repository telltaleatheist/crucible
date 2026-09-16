"""The Startup shortcut — PHASE15-HOST.md 4.1.

`%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\Startup\\Crucible.lnk`, and
nothing else: no admin, no Task Scheduler, no Windows service. A per-user login
item is what a tray program IS, and each of the alternatives costs something
this does not — Task Scheduler needs elevation to register and shows up in a
place nobody looks; a service cannot own a notification-area icon at all.

It is also what makes 4.7's reboot states true. "Reboot, then Crucible
continues" is only a sentence a person can trust if something starts after the
reboot, and this is that something.

THE TARGET IS `pythonw.exe`, NOT `crucible.cmd`
------------------------------------------------
A `.cmd` opens a console window. A login item that flashes a black box on every
boot is a login item people disable. The shortcut runs the installed pythonw
with a small Python entry point that binds CRUCIBLE_HOME before dispatching
`crucible local tray`. This preserves a custom install location after login,
when Explorer does not inherit the installer's environment. The pack's own
`Lib\\site-packages` is on its interpreter's path and WorkingDirectory is the
pack, so the caller's directory cannot shadow the installed module.

WHY POWERSHELL AND NOT pywin32
-------------------------------
A `.lnk` is a COM object (`WScript.Shell`), and the only ways to write one are
COM or a shell32 P/Invoke. `pywin32` would be a second Windows-only dependency
in a pack whose whole point is to be small, and it would have to be pinned,
built and shipped for one file that is written once. PowerShell ships with
Windows and speaks COM in four lines. The cost is a subprocess; the alternative
is a dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PureWindowsPath
import subprocess
from typing import Mapping

from .errors import HostError
from .paths import crucible_root, host_pack_dir, pythonw_path
from .runner import Runner

#: The file, exactly. One name, so the install verb, the remove verb and the
#: test all mean the same file.
SHORTCUT_NAME = "Crucible.lnk"

#: The relative path under `%APPDATA%`. Spelled as its four segments rather
#: than as one string, because "Start Menu" has a space in it and a joined
#: literal is a string somebody eventually quotes wrongly.
STARTUP_SEGMENTS = ("Microsoft", "Windows", "Start Menu", "Programs", "Startup")

SHORTCUT_TIMEOUT_SECONDS = 60.0

#: Description of the separately owned desktop presence.
SHORTCUT_DESCRIPTION = "Crucible — the engine's presence on this machine"


def startup_python(env: Mapping[str, str]) -> str:
    """Bind login to the installed home, independent of Explorer's environment."""
    return (
        "import os,runpy,sys;"
        f"os.environ['CRUCIBLE_HOME']={str(crucible_root(env))!r};"
        "sys.argv=['crucible','local','tray'];"
        "runpy.run_module('crucible.cli',run_name='__main__')"
    )


@dataclass(frozen=True)
class StartupOutcome:
    path: str
    #: Did THIS call change anything?
    changed: bool
    detail: str


def startup_dir(env: Mapping[str, str]) -> PureWindowsPath:
    """`%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\Startup`.

    From `APPDATA` and never assembled from a username, for `paths.py`'s
    reason: a roaming profile, a redirected AppData or a domain account all
    make `C:\\Users\\<name>\\AppData\\Roaming` a guess. Unset is refused.
    """
    roaming = env.get("APPDATA")
    if roaming is None or roaming.strip() == "":
        raise HostError(
            "host_no_localappdata",
            "APPDATA is not set, so this user has no Startup folder to put the "
            "login item in. It is read from the environment and never assembled "
            "from a username.",
        )
    path = PureWindowsPath(roaming)
    for segment in STARTUP_SEGMENTS:
        path = path / segment
    return path


def shortcut_path(env: Mapping[str, str]) -> PureWindowsPath:
    return startup_dir(env) / SHORTCUT_NAME


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def write_script(target: str, arguments: str, working_dir: str, lnk: str) -> str:
    """The PowerShell that writes the `.lnk`, as one `-Command` string.

    `WScript.Shell`'s `CreateShortcut` both creates and opens, so this is also
    how an existing shortcut is REWRITTEN in place — which is what makes
    `--install-startup` idempotent rather than a thing that has to check first.
    """
    return (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut("
        + _ps_quote(lnk)
        + "); "
        + "$s.TargetPath = "
        + _ps_quote(target)
        + "; $s.Arguments = "
        + _ps_quote(arguments)
        + "; $s.WorkingDirectory = "
        + _ps_quote(working_dir)
        + "; $s.Description = "
        + _ps_quote(SHORTCUT_DESCRIPTION)
        + "; $s.Save()"
    )


def install_argv(env: Mapping[str, str]) -> list[str]:
    """The whole command that writes the shortcut. Data, so a test can read it."""
    pack = host_pack_dir(env)
    lnk = shortcut_path(env)
    script = (
        f"New-Item -ItemType Directory -Force -Path {_ps_quote(str(lnk.parent))} | Out-Null; "
        + write_script(str(pythonw_path(env)), subprocess.list2cmdline(["-c", startup_python(env)]),
                       str(pack), str(lnk))
    )
    return ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script]


def remove_argv(env: Mapping[str, str]) -> list[str]:
    lnk = shortcut_path(env)
    return [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        # ONE f-string for the whole script, and every literal brace doubled.
        # Written as two adjacent strings first, with only the first an
        # f-string, it emitted `}} else {` — a PowerShell parse error, found
        # by running the verb on a real machine rather than by reading it.
        f"if (Test-Path {_ps_quote(str(lnk))}) {{ "
        f"Remove-Item -Force {_ps_quote(str(lnk))}; Write-Output 'removed' "
        f"}} else {{ Write-Output 'absent' }}",
    ]


def install(runner: Runner) -> StartupOutcome:
    """`crucible host --install-startup`. Idempotent: it rewrites in place."""
    lnk = shortcut_path(runner.env)
    result = runner.run(install_argv(runner.env), timeout_s=SHORTCUT_TIMEOUT_SECONDS)
    if not result.ok:
        raise HostError(
            "host_no_localappdata" if "APPDATA" in result.said() else "host_no_pack",
            f"the Startup item {lnk} could not be written: {result.said()}",
        )
    return StartupOutcome(
        path=str(lnk),
        changed=True,
        detail=f"{lnk} now starts Crucible's tray at login, with no console window",
    )


def remove(runner: Runner) -> StartupOutcome:
    """`crucible host --remove-startup`. Says whether there was one."""
    lnk = shortcut_path(runner.env)
    result = runner.run(remove_argv(runner.env), timeout_s=SHORTCUT_TIMEOUT_SECONDS)
    if not result.ok:
        raise HostError(
            "host_no_pack",
            f"the Startup item {lnk} could not be removed: {result.said()}",
        )
    removed = "removed" in result.stdout
    return StartupOutcome(
        path=str(lnk),
        changed=removed,
        detail=(
            f"{lnk} is gone; Crucible will not start at login"
            if removed
            else f"there was no {lnk}; nothing to remove"
        ),
    )
