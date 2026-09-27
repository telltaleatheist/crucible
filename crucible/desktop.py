from __future__ import annotations

import json
import os
import plistlib
import shlex
import subprocess
import sys
import threading
import time as time
from pathlib import Path
from typing import Any, Callable

from . import local, traylife
from .config import crucible_home
from .controller_client import LocalError
from .errors import CrucibleError
from .platform.runner import ProcessRunner

LABEL = "com.crucible.tray"

REFRESH_SECONDS = 5

LAUNCHCTL_SECONDS = 15

SHARING_UNOWNED = "sharing_unowned"

TRAY_ERRORS = (OSError, ValueError, RuntimeError, CrucibleError)


def close_tray() -> None:
    traylife.close_tray(crucible_home())


def _mac_bundle() -> Path:
    return Path.home() / "Applications" / "Crucible.app"


def _mac_agent() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / (LABEL + ".plist")


def _refuse_a_foreign_bundle(bundle: Path) -> None:
    info = bundle / "Contents" / "Info.plist"
    if not info.is_file():
        raise LocalError(
            f"desktop_not_owned: the existing {bundle} has no ownership record, so it was "
            f"left alone. Move it out of {bundle.parent}, then run `crucible local install-desktop` again"
        )
    with info.open("rb") as f:
        if plistlib.load(f).get("CFBundleIdentifier") != LABEL:
            raise LocalError(
                f"desktop_not_owned: {bundle} belongs to another installation, so it was "
                f"left alone. Move it out of {bundle.parent}, then run this again"
            )


def _write_mac_bundle(bundle: Path, home: Path) -> Path:
    contents = bundle / "Contents"
    binary = contents / "MacOS" / "Crucible"
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "crucible.cli", "local", "tray"]
    binary.write_text("#!/bin/sh\nexport CRUCIBLE_HOME=" + shlex.quote(str(home)) +
                      "\ncd " + shlex.quote(str(Path(__file__).resolve().parent.parent)) +
                      " || exit 1\nexec " + shlex.join(command) + "\n", encoding="utf-8")
    binary.chmod(0o755)
    with (contents / "Info.plist").open("wb") as f:
        plistlib.dump({"CFBundleIdentifier": LABEL, "CFBundleName": "Crucible",
                       "CFBundleExecutable": "Crucible", "CFBundlePackageType": "APPL",
                       "LSUIElement": True}, f)
    return binary


def _launchctl(*words: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *words], check=check, capture_output=True, timeout=LAUNCHCTL_SECONDS)


def _load_mac_agent(binary: Path) -> Path:
    agent = _mac_agent()
    agent.parent.mkdir(parents=True, exist_ok=True)
    with agent.open("wb") as f:
        plistlib.dump({"Label": LABEL, "ProgramArguments": [str(binary)],
                       "RunAtLoad": True, "ProcessType": "Interactive"}, f)
    domain = f"gui/{os.getuid()}"
    if _launchctl("print", f"{domain}/{LABEL}", check=False).returncode == 0:
        _launchctl("bootout", f"{domain}/{LABEL}")
    _launchctl("bootstrap", domain, str(agent))
    return agent


def install_desktop() -> None:
    if sys.platform == "win32":
        from .platform.startup import install
        install(ProcessRunner(sys.platform, os.environ))
        return
    if sys.platform != "darwin":
        print(json.dumps({"desktop": "skipped", "detail": "Linux uses its service manager; no desktop component installed"}))
        return
    bundle = _mac_bundle()
    if bundle.exists():
        _refuse_a_foreign_bundle(bundle)
    binary = _write_mac_bundle(bundle, crucible_home().resolve())
    agent = _load_mac_agent(binary)
    print(json.dumps({"app": str(bundle), "login_item": str(agent)}))


def _remove_mac_desktop() -> None:
    import shutil

    target = f"gui/{os.getuid()}/{LABEL}"
    if _launchctl("print", target, check=False).returncode == 0:
        _launchctl("bootout", target)
    bundle = _mac_bundle()
    if (bundle / "Contents" / "Info.plist").exists():
        _refuse_a_foreign_bundle(bundle)
        shutil.rmtree(bundle)
    _mac_agent().unlink(missing_ok=True)


def remove_desktop() -> None:
    close_tray()
    if sys.platform == "win32":
        from .platform.startup import remove
        remove(ProcessRunner(sys.platform, os.environ))
    elif sys.platform == "darwin":
        _remove_mac_desktop()


def _move_items(home: Path, retrying: bool, try_again: Callable[[], None]) -> list:
    if sys.platform != "win32":
        return []
    import pystray

    from .host import menu as host_menu
    from .host import outcome
    from .host.errors import HostError

    try:
        recorded = outcome.read(home)
    except HostError:
        recorded = None
    items = []
    for entry in host_menu.outcome_items(None if recorded is None else recorded.state, busy=retrying):
        handler = (lambda *_: try_again()) if entry.item_id == host_menu.TRY_AGAIN else None
        items.append(pystray.MenuItem(entry.label, handler, enabled=entry.enabled))
    return items


def tray() -> None:
    from .processlock import ProcessLock
    home = crucible_home()
    home.mkdir(parents=True, exist_ok=True)
    guard = ProcessLock(home / traylife.LOCK_NAME)
    if not guard.acquire():
        return
    try:
        _run_tray(home)
    finally:
        traylife.pid_path(home).unlink(missing_ok=True)
        traylife.close_request_path(home).unlink(missing_ok=True)
        guard.close()


class TrayIcon:
    def __init__(self, home: Path, pystray: Any, icon: Any) -> None:
        self.home = home
        self.pystray = pystray
        self.icon = icon
        self.close_request = traylife.close_request_path(home)
        self.stopped = threading.Event()
        self.retrying = threading.Event()
        self.busy = threading.Lock()
        self.state: dict[str, Any] = {"state": "starting", "detail": "Starting Crucible"}
        self.notice: dict[str, Any] = {"message": "", "adopt": False}

    def _sharing_label(self) -> str:
        from .sharing import read

        try:
            shared = read(self.home) is not None
        except TRAY_ERRORS as exc:
            shared = False
            self.notice["message"] = str(exc)
        if shared:
            return "Stop Tailscale sharing"
        return "Use existing Tailscale sharing" if self.notice["adopt"] else "Enable Tailscale sharing"

    def refresh(self) -> None:
        item = self.pystray.MenuItem
        self.icon.title = "Crucible — " + self.state["state"]
        sharing_label = self._sharing_label()
        running = self.state["state"] == "running"
        self.icon.menu = self.pystray.Menu(
            item(self.notice["message"] or self.state["detail"], None, enabled=False),
            *_move_items(self.home, self.retrying.is_set(), self.try_again),
            item("Open Crucible", lambda *_: self.action("open-console")),
            item("Connect an app…", lambda *_: self.action("connect")),
            item("Start Crucible", lambda *_: self.action("start"), enabled=not running),
            item("Stop Crucible", lambda *_: self.action("stop"), enabled=running),
            item(sharing_label, lambda *_: self.action("sharing")),
            item("Close tray icon", lambda *_: self.close()),
        )
        self.icon.update_menu()

    def _toggle_sharing(self) -> None:
        from . import sharing

        runner = ProcessRunner(sys.platform, os.environ)
        engine = sharing.PairedEngine(self.home)
        if sharing.read(self.home) is not None:
            sharing.disable(self.home, runner, engine)
            self.notice["message"] = "Tailscale sharing stopped"
        else:
            result = sharing.enable(self.home, runner, engine, adopt=self.notice["adopt"])
            self.notice["message"] = "Tailscale sharing enabled: " + result["authority"]
        self.notice["adopt"] = False

    def _run_engine_verb(self, verb: str) -> None:
        self.state.update(local.act(verb))
        if self.state.get("sharing", {}).get("state") == "degraded":
            self.notice["message"] = "Crucible is running; sharing needs attention: " + self.state["sharing"]["detail"]

    def _said_no(self, exc: BaseException) -> None:
        self.notice["message"] = str(exc)
        if SHARING_UNOWNED in str(exc):
            self.notice["adopt"] = True
            self.notice["message"] = "A matching Tailscale forward exists. Select Use existing Tailscale sharing to manage it here."

    def _act(self, verb: str) -> None:
        if not self.busy.acquire(blocking=False):
            return
        self.notice["message"] = ""
        try:
            if verb == "sharing":
                self._toggle_sharing()
            else:
                self._run_engine_verb(verb)
        except TRAY_ERRORS as exc:
            self._said_no(exc)
        finally:
            self.busy.release()
            self.refresh()

    def action(self, verb: str) -> None:
        threading.Thread(target=lambda: self._act(verb), daemon=True).start()

    def _say(self, line: str) -> None:
        self.notice["message"] = line
        self.refresh()

    def _try_again_now(self) -> None:
        from .host.errors import HostError
        from .host.retry import try_again as run_try_again

        try:
            ended = run_try_again(self.home, self._say)
            if ended is not None and ended.sentence:
                self.notice["message"] = ended.sentence
        except (HostError, OSError) as exc:
            self.notice["message"] = str(exc)
        finally:
            self.retrying.clear()
            self.refresh()

    def try_again(self) -> None:
        if self.retrying.is_set():
            return
        self.retrying.set()
        threading.Thread(target=self._try_again_now, name="crucible-try-again", daemon=True).start()

    def close(self) -> None:
        self.stopped.set()
        self.icon.stop()

    def _observe(self) -> None:
        if not self.busy.acquire(blocking=False):
            return
        try:
            self.state.update(local.status(self.home))
        except TRAY_ERRORS as exc:
            self.state.update(state="failed", detail=str(exc))
        finally:
            self.busy.release()
        self.refresh()

    def watch(self) -> None:
        while not self.stopped.is_set():
            if self.close_request.exists():
                self.close()
                return
            self._observe()
            self.stopped.wait(REFRESH_SECONDS)


def _run_tray(home: Path) -> None:
    import pystray

    from .host.tray import ICON_FOREGROUND, icon_image
    pid_file = traylife.pid_path(home)
    close_request = traylife.close_request_path(home)
    close_request.unlink(missing_ok=True)
    pid_file.write_text(str(os.getpid()))
    if sys.platform == "win32":
        local.ensure_controller(home)
    tray_icon = TrayIcon(home, pystray, pystray.Icon("crucible", icon_image(ICON_FOREGROUND), "Crucible"))
    tray_icon.refresh()
    threading.Thread(target=tray_icon.watch, daemon=True).start()
    try:
        tray_icon.icon.run()
    finally:
        tray_icon.stopped.set()
        pid_file.unlink(missing_ok=True)
        close_request.unlink(missing_ok=True)


def run_tray_verb(verb: str) -> None:
    verbs: dict[str, Callable[[], None]] = {
        "tray": tray,
        "install-desktop": install_desktop,
        "remove-desktop": remove_desktop,
    }
    verbs[verb]()
