"""What starts Crucible's tray when this Windows user logs in.

A value under the user's own Run key (`HKCU\\...\\CurrentVersion\\Run`, no elevation) runs
`pythonw.exe <host>\\launchers\\crucible-tray.pyw`. Python writes both itself: the value with
`winreg`, the launcher as a plain file.

Until 2.0.21 this was a `Startup\\Crucible.lnk` written by an inline `powershell -Command`
through WScript.Shell COM, pointing a hidden `pythonw.exe -c "<code>"` at the Startup
folder. On Victoria's laptop (2026-10-24) Microsoft Defender blocked that process as
`Trojan:Win32/Commando.A!ml` - the shape of a persistence dropper - so the install stopped
at `install-desktop` and the tray, which rewrote the item at every start, was blocked at
every login too. Now no shell is started to register anything, and nothing runs code from
a command line: the launcher is a file that can be read. An install removes the old
`Startup\\Crucible.lnk` (it would start a second tray beside the Run value).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Mapping, Protocol

from .errors import HostError
from .paths import crucible_root, host_pack_dir, pythonw_path
from .runner import Runner

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = "Crucible"

LAUNCHERS_SUBDIR = "launchers"
TRAY_LAUNCHER = "crucible-tray.pyw"

LEGACY_SHORTCUT_NAME = "Crucible.lnk"

STARTUP_SEGMENTS = ("Microsoft", "Windows", "Start Menu", "Programs", "Startup")


class LoginSystem(Protocol):
    """The three things a login entry touches: a file, the user's Run key, and the old
    Startup shortcut. One seam, so a test never writes this machine's registry."""

    def write_text(self, path: str, text: str) -> None: ...

    def remove_file(self, path: str) -> bool: ...

    def set_run_value(self, name: str, command: str) -> None: ...

    def delete_run_value(self, name: str) -> bool: ...


class WindowsLogin:
    def write_text(self, path: str, text: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(target.name + ".tmp")
        staged.write_text(text, encoding="utf-8")
        os.replace(staged, target)

    def remove_file(self, path: str) -> bool:
        try:
            Path(path).unlink()
        except FileNotFoundError:
            return False
        return True

    def set_run_value(self, name: str, command: str) -> None:
        import winreg

        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, command)

    def delete_run_value(self, name: str) -> bool:
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
                winreg.DeleteValue(key, name)
        except FileNotFoundError:
            return False
        return True


SYSTEM: LoginSystem = WindowsLogin()


def launcher_source(env: Mapping[str, str], words: tuple[str, ...]) -> str:
    """A launcher's text: binds CRUCIBLE_HOME (a login does not inherit the installer's
    environment) and runs `crucible <words>`."""
    argv = ", ".join(repr(word) for word in ("crucible", *words))
    return (
        f"# Written by Crucible: runs `crucible {' '.join(words)}` with this install's home.\n"
        "import os\nimport runpy\nimport sys\n\n"
        f"os.environ[\"CRUCIBLE_HOME\"] = {str(crucible_root(env))!r}\n"
        f"sys.argv = [{argv}]\n"
        "runpy.run_module(\"crucible.cli\", run_name=\"__main__\")\n"
    )


def launcher_path(env: Mapping[str, str], name: str) -> PureWindowsPath:
    return host_pack_dir(env) / LAUNCHERS_SUBDIR / name


def launch_command(env: Mapping[str, str], name: str) -> str:
    """`pythonw.exe <launcher>`, quoted as Windows reads a command line."""
    return subprocess.list2cmdline([str(pythonw_path(env)), str(launcher_path(env, name))])


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
            "APPDATA is not set, so this user has no Start Menu folder. It is read from "
            "the environment and never assembled from a username.",
        )
    path = PureWindowsPath(roaming)
    for segment in STARTUP_SEGMENTS:
        path = path / segment
    return path


def legacy_shortcut_path(env: Mapping[str, str]) -> PureWindowsPath:
    """The Startup shortcut releases before 2.0.21 wrote; an install removes it."""
    return startup_dir(env) / LEGACY_SHORTCUT_NAME


def run_entry() -> str:
    return f"HKCU\\{RUN_KEY}\\{RUN_VALUE}"


def install(runner: Runner) -> StartupOutcome:
    env = runner.env
    legacy = legacy_shortcut_path(env)
    launcher = launcher_path(env, TRAY_LAUNCHER)
    command = launch_command(env, TRAY_LAUNCHER)
    try:
        SYSTEM.write_text(str(launcher), launcher_source(env, ("local", "tray")))
        SYSTEM.set_run_value(RUN_VALUE, command)
        SYSTEM.remove_file(str(legacy))
    except OSError as exc:
        raise HostError(
            "host_no_pack",
            f"the login entry {run_entry()} (running {launcher}) could not be written: {exc}",
        ) from exc
    return StartupOutcome(
        path=run_entry(),
        changed=True,
        detail=f"{run_entry()} now starts Crucible's tray at login ({command}), with no console window",
    )


def remove(runner: Runner) -> StartupOutcome:
    env = runner.env
    legacy = legacy_shortcut_path(env)
    try:
        removed_value = SYSTEM.delete_run_value(RUN_VALUE)
        removed_legacy = SYSTEM.remove_file(str(legacy))
    except OSError as exc:
        raise HostError(
            "host_no_pack",
            f"the login entry {run_entry()} could not be removed: {exc}",
        ) from exc
    removed = [what for what, gone in ((run_entry(), removed_value), (str(legacy), removed_legacy)) if gone]
    return StartupOutcome(
        path=run_entry(),
        changed=bool(removed),
        detail=(
            f"{' and '.join(removed)} removed; Crucible will not start at login"
            if removed
            else f"there was no {run_entry()} and no {legacy}; nothing to remove"
        ),
    )


__all__ = [
    "LoginSystem", "RUN_KEY", "RUN_VALUE", "SYSTEM", "StartupOutcome", "TRAY_LAUNCHER",
    "WindowsLogin", "install", "launch_command", "launcher_path", "launcher_source",
    "legacy_shortcut_path", "remove", "run_entry", "startup_dir",
]
