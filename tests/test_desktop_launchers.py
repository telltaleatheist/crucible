from __future__ import annotations

import plistlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from crucible import desktop, local, uninstall
from crucible.cli import build_parser
from crucible.desktop_app import app as app_main
from crucible.desktop_app import instance, launchers
from crucible.platform.runner import RunResult

REPO = Path(__file__).resolve().parent.parent

WINDOWS_ENV = {
    "LOCALAPPDATA": r"C:\Users\tellt\AppData\Local",
    "APPDATA": r"C:\Users\tellt\AppData\Roaming",
}


@dataclass
class Recorder:
    env: Mapping[str, str] = field(default_factory=lambda: dict(WINDOWS_ENV))
    platform: str = "win32"
    calls: list[list[str]] = field(default_factory=list)
    result: RunResult = RunResult(code=0, stdout="removed", stderr="", failure=None)

    def run(self, argv: Sequence[str], *, timeout_s: float, env: Mapping[str, str] | None = None) -> RunResult:
        self.calls.append(list(argv))
        return self.result


def test_the_start_menu_item_sits_in_programs_not_in_startup() -> None:
    assert str(launchers.start_menu_path(WINDOWS_ENV)) == (
        r"C:\Users\tellt\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Crucible.lnk")


def test_the_start_menu_item_opens_the_window_windowless_with_the_icon() -> None:
    script = launchers.start_menu_install_argv(WINDOWS_ENV, r"C:\x\crucible.ico")[-1]
    assert r"\Crucible\host\pythonw.exe" in script and "crucible.cmd" not in script
    assert "sys.argv=[''crucible'',''app'']" in script
    assert "os.environ[''CRUCIBLE_HOME'']=''C:\\\\Users\\\\tellt\\\\AppData\\\\Local\\\\Crucible''" in script
    assert "$s.IconLocation = 'C:\\x\\crucible.ico'" in script
    assert r"Start Menu\Programs\Crucible.lnk" in script and "Startup" not in script


def test_removing_the_start_menu_item_names_only_that_item() -> None:
    script = launchers.start_menu_remove_argv(WINDOWS_ENV)[-1]
    assert script.count("Programs\\Crucible.lnk") == 2 and "Startup" not in script


def test_windows_install_desktop_writes_the_login_item_and_the_start_menu_item(monkeypatch, capsys) -> None:
    runner = Recorder()
    monkeypatch.setattr(desktop.sys, "platform", "win32")
    monkeypatch.setattr(desktop, "ProcessRunner", lambda platform, env: runner)
    desktop.install_desktop()
    joined = [" ".join(call) for call in runner.calls]
    assert "Startup\\Crucible.lnk" in joined[0] and "Programs\\Crucible.lnk" in joined[1]
    assert '"app": "C:\\\\Users\\\\tellt\\\\AppData\\\\Roaming' in capsys.readouterr().out


def test_a_start_menu_item_that_cannot_be_written_says_what_to_run(monkeypatch) -> None:
    runner = Recorder(result=RunResult(code=1, stdout="", stderr="access denied", failure=None))
    with pytest.raises(local.LocalError, match=r"start_menu_failed: .*access denied. Run `crucible local install-desktop` again"):
        desktop._install_start_menu(runner)


def test_windows_remove_desktop_closes_the_window_then_removes_both_items(monkeypatch, tmp_path: Path) -> None:
    runner = Recorder()
    closed: list[str] = []
    monkeypatch.setattr(desktop.sys, "platform", "win32")
    monkeypatch.setattr(desktop, "crucible_home", lambda: tmp_path)
    monkeypatch.setattr(desktop, "ProcessRunner", lambda platform, env: runner)
    monkeypatch.setattr(local, "close_app", lambda home: closed.append("app"))
    monkeypatch.setattr(desktop, "close_tray", lambda: closed.append("tray"))
    desktop.remove_desktop()
    joined = [" ".join(call) for call in runner.calls]
    assert closed == ["app", "tray"]
    assert "Startup\\Crucible.lnk" in joined[0] and "Programs\\Crucible.lnk" in joined[1]


def test_a_window_that_will_not_close_stops_the_removal_and_says_what_to_do(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(instance, "close_running", lambda home: False)
    with pytest.raises(local.LocalError, match="app_close_failed: .*Close the Crucible window, then run this again"):
        local.close_app(tmp_path)


def test_the_mac_bundle_opens_the_app_and_carries_the_tray_for_launchd(tmp_path: Path) -> None:
    bundle = tmp_path / "Applications" / "Crucible.app"
    tray = launchers.write_mac_bundle(bundle, Path("/Users/o/.crucible"), "/Users/o/.crucible/server/bin/python3",
                                      Path("/Users/o/.crucible/server/lib/python3.11/site-packages"))
    contents = bundle / "Contents"
    with (contents / "Info.plist").open("rb") as f:
        info = plistlib.load(f)
    assert info["CFBundleIdentifier"] == "com.crucible.app" and info["CFBundleExecutable"] == "Crucible"
    assert info["CFBundleIconFile"] == "crucible" and "LSUIElement" not in info
    app = (contents / "MacOS" / "Crucible").read_text(encoding="utf-8")
    assert app.startswith("#!/bin/sh\nexport CRUCIBLE_HOME=/Users/o/.crucible\n")
    assert app.rstrip().endswith("exec /Users/o/.crucible/server/bin/python3 -m crucible.cli app")
    assert tray == contents / "Resources" / "crucible-tray"
    assert tray.read_text(encoding="utf-8").rstrip().endswith("-m crucible.cli local tray")
    assert (contents / "Resources" / "crucible.icns").read_bytes()[:4] == b"icns"
    assert launchers.codesign_argv(bundle) == ["codesign", "--sign", "-", "--force", str(bundle)]


def test_a_bundle_the_tray_installed_before_the_app_existed_is_still_ours(tmp_path: Path) -> None:
    bundle = tmp_path / "Crucible.app"
    (bundle / "Contents").mkdir(parents=True)
    with (bundle / "Contents" / "Info.plist").open("wb") as f:
        plistlib.dump({"CFBundleIdentifier": "com.crucible.tray"}, f)
    desktop._refuse_a_foreign_bundle(bundle)
    with (bundle / "Contents" / "Info.plist").open("wb") as f:
        plistlib.dump({"CFBundleIdentifier": "com.other.app"}, f)
    with pytest.raises(local.LocalError, match="desktop_not_owned"):
        desktop._refuse_a_foreign_bundle(bundle)


def test_signing_that_fails_still_leaves_an_openable_bundle(monkeypatch, tmp_path: Path) -> None:
    def no_codesign(*args, **kwargs):
        raise FileNotFoundError("codesign")
    monkeypatch.setattr(launchers.subprocess, "run", no_codesign)
    assert launchers.sign_mac_bundle(tmp_path).startswith("not signed (codesign)")


def test_the_tray_opens_the_bundle_on_a_mac_and_pythonw_on_windows(tmp_path: Path) -> None:
    bundle = tmp_path / "Crucible.app"
    (bundle / "Contents").mkdir(parents=True)
    (bundle / "Contents" / "Info.plist").write_bytes(b"x")
    assert launchers.app_argv("darwin", "/py/python3", bundle) == ["open", str(bundle)]
    assert launchers.app_argv("darwin", "/py/python3", tmp_path / "none.app") == [
        "/py/python3", "-m", "crucible.cli", "app"]
    python = tmp_path / "python.exe"
    (tmp_path / "pythonw.exe").write_bytes(b"")
    assert launchers.app_argv("win32", str(python)) == [str(tmp_path / "pythonw.exe"), "-m", "crucible.cli", "app"]


def test_uninstall_removes_the_launcher_through_remove_desktop(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "installation.json").write_text("{}", encoding="utf-8")
    steps = uninstall._registration_steps(home, "win32", WINDOWS_ENV, Recorder(), uninstall.STARTUP,
                                          tmp_path, tmp_path / "python.exe")
    removal = next(step for step in steps if step.name == "remove-desktop")
    assert "Start Menu item" in removal.what and "Crucible.app" in removal.what
    assert {"app.lock", "app.door", "app.log"} <= set(uninstall.STATE_FILES)


def test_crucible_app_is_a_command_and_a_windowless_entry_point() -> None:
    args = build_parser().parse_args(["app"])
    assert args.func.__module__ == "crucible.cli.app_cmd"
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert '[project.gui-scripts]\ncrucible-app = "crucible.desktop_app:main"' in text
    assert '"desktop_app/assets/*"' in text


def test_a_second_launch_focuses_the_first_and_never_draws(monkeypatch, tmp_path: Path) -> None:
    first = instance.Instance(tmp_path)
    first.claim()
    heard: list[str] = []
    first.serve(heard.append)
    monkeypatch.setitem(sys.modules, "crucible.desktop_app.window", None)
    try:
        assert app_main.launch(tmp_path) == 0
    finally:
        first.close()
    assert heard == ["focus"]


def test_a_python_without_tk_names_the_installer(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setitem(sys.modules, "crucible.desktop_app.window", None)
    assert app_main.launch(tmp_path) == 1
    said = capsys.readouterr().err
    assert "app_no_tk" in said and "install" in said
    assert not instance.still_open(tmp_path)


def test_the_icon_is_drawn_for_every_size_a_launcher_asks_for() -> None:
    assets = REPO / "crucible" / "desktop_app" / "assets"
    for size in (16, 32, 64, 256, 512):
        assert (assets / f"icon-{size}.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (assets / "crucible.ico").read_bytes()[:4] == b"\x00\x00\x01\x00"
    assert (REPO / "installer" / "windows" / "welcome.bmp").read_bytes()[:2] == b"BM"


NSI = REPO / "installer" / "windows" / "crucible.nsi"
BUILD = REPO / "scripts" / "build-installer.sh"


def test_the_setup_installs_per_user_into_the_same_home_as_the_one_liner() -> None:
    text = NSI.read_text(encoding="utf-8")
    assert "RequestExecutionLevel user" in text
    assert 'InstallDir "$LOCALAPPDATA\\Crucible"' in text
    assert "!insertmacro MUI_PAGE_DIRECTORY" in text and "!define MUI_FINISHPAGE_RUN_TEXT \"Open Crucible\"" in text


def test_the_setup_checks_each_download_before_the_one_install_procedure_runs() -> None:
    text = NSI.read_text(encoding="utf-8")
    assert text.count("Call Fetch") == 2
    assert 'NScurl::http GET "$Url" "$Target" /INSIST /CANCEL' in text
    assert "NScurl::sha256 -file \"$Target\"" in text and "${If} $0 != $Pinned" in text
    install = re.search(r"nsExec::ExecToLog '(.*-File \"\$PLUGINSDIR\\\$\{SCRIPT\}\" -Release.*)'", text)
    assert install, "the setup no longer runs the generated install.ps1"
    assert '-PythonArchive "$PLUGINSDIR\\downloads\\${PY_ASSET}"' in install.group(1)
    assert "-WheelSha ${WHEEL_SHA}" in install.group(1)
    assert '!define SCRIPT "crucible-install.ps1"' in text


def test_the_setup_uninstalls_with_crucibles_own_uninstall() -> None:
    text = NSI.read_text(encoding="utf-8")
    uninstall_section = text[text.index('Section "Uninstall"'):]
    assert "-Uninstall -Root \"$INSTDIR\"" in uninstall_section
    assert "RMDir /r" not in text and 'RMDir "$INSTDIR"' in uninstall_section
    assert 'DeleteRegKey HKCU "${UNINSTALL_KEY}"' in uninstall_section
    assert "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Crucible" in text


def test_the_build_pins_nsis_and_nscurl_and_reads_python_from_install_ps1() -> None:
    text = BUILD.read_text(encoding="utf-8")
    for name in ("NSIS_SHA", "NSCURL_SHA"):
        assert re.search(rf'^{name}="[0-9a-f]{{64}}"$', text, re.MULTILINE), name
    assert 'NSIS_URL="https://downloads.sourceforge.net/project/nsis/' in text
    assert 'NSCURL_URL="https://github.com/negrutiu/nsis-nscurl/releases/download/' in text
    assert "sdk/bootstrap/scripts/install.ps1" in text and "pin PySha" in text
    ps1 = (REPO / "sdk" / "bootstrap" / "scripts" / "install.ps1").read_text(encoding="utf-8")
    for pinned in ("PyUrl", "PySha", "PyAsset", "PyVersion"):
        assert re.search(rf"^\${pinned} = '[^']+'$", ps1, re.MULTILINE), pinned
    for parameter in ("PythonArchive", "WheelFile", "WheelSha"):
        assert f"[string]${parameter} = ''" in ps1


def test_the_release_builds_the_setup_and_uploads_it() -> None:
    text = (REPO / "scripts" / "release.sh").read_text(encoding="utf-8")
    assert './scripts/build-installer.sh --version "$VERSION" --wheel "$WHEEL" --out "$OUT"' in text
    assert 'SETUP_EXE="$OUT/crucible-setup-$VERSION.exe"' in text
