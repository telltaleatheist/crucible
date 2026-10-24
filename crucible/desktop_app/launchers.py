from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from typing import Mapping

from .. import VERSION
from ..platform import startup
from ..platform.paths import host_pack_dir, pythonw_path
from ..platform.powershell import quote as ps_quote
from ..platform.powershell import script_argv
from ..platform.startup import launcher_path, launcher_source, startup_dir

APP_WORDS = ("app",)
APP_LAUNCHER = "crucible-app.pyw"

APP_ID = "com.crucible.app"
OWNED_IDS = frozenset({APP_ID, "com.crucible.tray"})

START_MENU_NAME = "Crucible.lnk"
START_MENU_DESCRIPTION = "Crucible: models, voices and packages on this computer"

MAC_EXECUTABLE = "Crucible"
MAC_TRAY_SCRIPT = "crucible-tray"
MAC_ICON = "crucible"

CODESIGN_SECONDS = 60

WINDOWS_DETACHED = 0x00000008 | 0x00000200


def assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


def start_menu_path(env: Mapping[str, str]) -> PureWindowsPath:
    return startup_dir(env).parent / START_MENU_NAME


def write_app_launcher(env: Mapping[str, str]) -> PureWindowsPath:
    """The file the Start Menu item runs (`crucible app`); written by Python, so the
    shortcut carries a path and never code (startup.py says why)."""
    path = launcher_path(env, APP_LAUNCHER)
    startup.SYSTEM.write_text(str(path), launcher_source(env, APP_WORDS))
    return path


def start_menu_install_argv(env: Mapping[str, str], icon: str) -> list[str]:
    """The Start Menu item: `pythonw.exe <app launcher>`. Not a login item, so it is
    still made through WScript.Shell, the one writer of a .lnk without pywin32."""
    lnk = start_menu_path(env)
    arguments = subprocess.list2cmdline([str(launcher_path(env, APP_LAUNCHER))])
    return script_argv(
        f"New-Item -ItemType Directory -Force -Path {ps_quote(str(lnk.parent))} | Out-Null; "
        + _shortcut_script(str(pythonw_path(env)), arguments, str(host_pack_dir(env)), str(lnk),
                           START_MENU_DESCRIPTION, icon)
    )


def _shortcut_script(target: str, arguments: str, working_dir: str, lnk: str,
                     description: str, icon: str) -> str:
    return (
        f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut({ps_quote(lnk)}); "
        f"$s.TargetPath = {ps_quote(target)}; $s.Arguments = {ps_quote(arguments)}; "
        f"$s.WorkingDirectory = {ps_quote(working_dir)}; $s.Description = {ps_quote(description)}; "
        f"$s.IconLocation = {ps_quote(icon)}; $s.Save()"
    )


def start_menu_remove_argv(env: Mapping[str, str]) -> list[str]:
    lnk = ps_quote(str(start_menu_path(env)))
    return script_argv(
        f"if (Test-Path {lnk}) {{ Remove-Item -Force {lnk}; Write-Output 'removed' }} "
        "else { Write-Output 'absent' }"
    )


def mac_bundle_path(user_home: Path | None = None) -> Path:
    return (user_home if user_home is not None else Path.home()) / "Applications" / "Crucible.app"


def _script(home: Path, cwd: Path, python: str, words: tuple[str, ...]) -> str:
    command = shlex.join([python, "-m", "crucible.cli", *words])
    return (f"#!/bin/sh\nexport CRUCIBLE_HOME={shlex.quote(home.as_posix())}\n"
            f"cd {shlex.quote(cwd.as_posix())} || exit 1\nexec {command}\n")


def mac_info_plist() -> dict:
    return {
        "CFBundleIdentifier": APP_ID,
        "CFBundleName": "Crucible",
        "CFBundleDisplayName": "Crucible",
        "CFBundleExecutable": MAC_EXECUTABLE,
        "CFBundleIconFile": MAC_ICON,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "NSHighResolutionCapable": True,
    }


def write_mac_bundle(bundle: Path, home: Path, python: str, cwd: Path) -> Path:
    contents = bundle / "Contents"
    executable = contents / "MacOS" / MAC_EXECUTABLE
    tray = contents / "Resources" / MAC_TRAY_SCRIPT
    executable.parent.mkdir(parents=True, exist_ok=True)
    tray.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text(_script(home, cwd, python, APP_WORDS), encoding="utf-8")
    tray.write_text(_script(home, cwd, python, ("local", "tray")), encoding="utf-8")
    executable.chmod(0o755)
    tray.chmod(0o755)
    shutil.copyfile(assets_dir() / f"{MAC_ICON}.icns", contents / "Resources" / f"{MAC_ICON}.icns")
    with (contents / "Info.plist").open("wb") as f:
        plistlib.dump(mac_info_plist(), f)
    return tray


def codesign_argv(bundle: Path) -> list[str]:
    return ["codesign", "--sign", "-", "--force", str(bundle)]


def sign_mac_bundle(bundle: Path) -> str:
    try:
        done = subprocess.run(codesign_argv(bundle), capture_output=True, text=True,
                              timeout=CODESIGN_SECONDS)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"not signed ({exc}); macOS still opens it, since it holds only scripts"
    if done.returncode != 0:
        return f"not signed ({done.stderr.strip()}); macOS still opens it, since it holds only scripts"
    return "signed ad hoc"


def app_argv(platform: str = sys.platform, executable: str = sys.executable,
             bundle: Path | None = None) -> list[str]:
    if platform == "darwin":
        bundle = bundle if bundle is not None else mac_bundle_path()
        if (bundle / "Contents" / "Info.plist").is_file():
            return ["open", str(bundle)]
    if platform == "win32":
        windowless = Path(executable).with_name("pythonw.exe")
        if windowless.is_file():
            executable = str(windowless)
    return [executable, "-m", "crucible.cli", *APP_WORDS]


def spawn_app() -> None:
    options: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                     "stderr": subprocess.DEVNULL, "env": dict(os.environ), "close_fds": True}
    if sys.platform == "win32":
        options["creationflags"] = WINDOWS_DETACHED
    else:
        options["start_new_session"] = True
    subprocess.Popen(app_argv(), **options)
