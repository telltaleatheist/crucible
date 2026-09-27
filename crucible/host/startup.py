from __future__ import annotations

from dataclasses import dataclass
from pathlib import PureWindowsPath
import subprocess
from typing import Mapping

from .errors import HostError
from .paths import crucible_root, host_pack_dir, pythonw_path
from .runner import Runner

SHORTCUT_NAME = "Crucible.lnk"

STARTUP_SEGMENTS = ("Microsoft", "Windows", "Start Menu", "Programs", "Startup")

SHORTCUT_TIMEOUT_SECONDS = 60.0

SHORTCUT_DESCRIPTION = "Crucible — the engine's presence on this machine"


def startup_python(env: Mapping[str, str]) -> str:
    return (
        "import os,runpy,sys;"
        f"os.environ['CRUCIBLE_HOME']={str(crucible_root(env))!r};"
        "sys.argv=['crucible','local','tray'];"
        "runpy.run_module('crucible.cli',run_name='__main__')"
    )


@dataclass(frozen=True)
class StartupOutcome:
    path: str
    changed: bool
    detail: str


def startup_dir(env: Mapping[str, str]) -> PureWindowsPath:
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
        f"if (Test-Path {_ps_quote(str(lnk))}) {{ "
        f"Remove-Item -Force {_ps_quote(str(lnk))}; Write-Output 'removed' "
        f"}} else {{ Write-Output 'absent' }}",
    ]


def install(runner: Runner) -> StartupOutcome:
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
