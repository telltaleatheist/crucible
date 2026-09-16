"""Optional desktop presence. Closing it never stops the controller or engine."""
from __future__ import annotations

import json
import os
from pathlib import Path
import plistlib
import shlex
import subprocess
import sys
import threading
import time

from .config import crucible_home
from . import local
from .errors import CrucibleError

LABEL = "com.crucible.tray"


def close_tray() -> None:
    home = crucible_home()
    pid_file = home / "tray.pid"
    if not pid_file.exists():
        return
    from .host.app import _alive
    raw = pid_file.read_text().strip()
    if not raw.isdigit() or not _alive(int(raw)):
        pid_file.unlink(missing_ok=True)
        return
    (home / "tray.close").write_text("close\n")
    deadline = time.monotonic() + 15
    # The tray removes its PID in finally, before Python has unloaded its
    # runtime. A Windows pack cannot be replaced until that process exits.
    while True:
        if not _alive(int(raw)):
            pid_file.unlink(missing_ok=True)
            return
        if time.monotonic() >= deadline:
            raise local.LocalError("tray_close_failed: the tray did not close; installation is unchanged")
        time.sleep(0.1)


def install_desktop() -> None:
    if sys.platform == "win32":
        from .host.startup import install
        from .host.runner import ProcessRunner
        install(ProcessRunner(sys.platform, os.environ))
        return
    if sys.platform != "darwin":
        print(json.dumps({"desktop": "skipped", "detail": "Linux uses its service manager; no desktop component installed"}))
        return
    home = crucible_home().resolve()
    bundle = Path.home() / "Applications" / "Crucible.app"
    contents = bundle / "Contents"
    binary = contents / "MacOS" / "Crucible"
    info = contents / "Info.plist"
    if bundle.exists():
        if not info.is_file():
            raise local.LocalError("desktop_not_owned: the existing Crucible.app has no ownership record")
        with info.open("rb") as f:
            if plistlib.load(f).get("CFBundleIdentifier") != LABEL:
                raise local.LocalError("desktop_not_owned: Crucible.app belongs to another installation")
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
    agent = Path.home() / "Library" / "LaunchAgents" / (LABEL + ".plist")
    agent.parent.mkdir(parents=True, exist_ok=True)
    with agent.open("wb") as f:
        plistlib.dump({"Label": LABEL, "ProgramArguments": [str(binary)],
                       "RunAtLoad": True, "ProcessType": "Interactive"}, f)
    domain = f"gui/{os.getuid()}"
    loaded = subprocess.run(["launchctl", "print", f"{domain}/{LABEL}"], capture_output=True, timeout=15)
    if loaded.returncode == 0:
        subprocess.run(["launchctl", "bootout", f"{domain}/{LABEL}"], check=True, capture_output=True, timeout=15)
    subprocess.run(["launchctl", "bootstrap", domain, str(agent)], check=True, capture_output=True, timeout=15)
    print(json.dumps({"app": str(bundle), "login_item": str(agent)}))


def remove_desktop() -> None:
    close_tray()
    if sys.platform == "win32":
        from .host.startup import remove
        from .host.runner import ProcessRunner
        remove(ProcessRunner(sys.platform, os.environ))
    elif sys.platform == "darwin":
        target = f"gui/{os.getuid()}/{LABEL}"
        loaded = subprocess.run(["launchctl", "print", target], capture_output=True, timeout=15)
        if loaded.returncode == 0:
            subprocess.run(["launchctl", "bootout", target], check=True, capture_output=True, timeout=15)
        # Remove only our own named bundle, never the user's Applications tree.
        import shutil
        bundle = Path.home() / "Applications" / "Crucible.app"
        info = bundle / "Contents" / "Info.plist"
        if info.exists():
            with info.open("rb") as f:
                if plistlib.load(f).get("CFBundleIdentifier") != LABEL:
                    raise local.LocalError("desktop_not_owned: Crucible.app belongs to another installation")
            shutil.rmtree(bundle)
        (Path.home() / "Library" / "LaunchAgents" / (LABEL + ".plist")).unlink(missing_ok=True)


def tray() -> None:
    from .processlock import ProcessLock
    home = crucible_home()
    home.mkdir(parents=True, exist_ok=True)
    pid_file = home / "tray.pid"
    guard = ProcessLock(home / "tray.lock")
    if not guard.acquire():
        return
    try:
        _run_tray(home)
    finally:
        pid_file.unlink(missing_ok=True)
        (home / "tray.close").unlink(missing_ok=True)
        guard.close()


def _run_tray(home: Path) -> None:
    import pystray
    from .host.tray import icon_image, ICON_FOREGROUND
    pid_file = home / "tray.pid"
    (home / "tray.close").unlink(missing_ok=True)
    pid_file.write_text(str(os.getpid()))
    if sys.platform == "win32":
        try:
            observed = local.controller_ping()
        except OSError:
            local._spawn_controller(home)
            deadline = time.monotonic() + 90
            while True:
                try:
                    observed = local.controller_ping()
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise local.LocalError("controller_start_failed: inspect host.log")
                    time.sleep(0.5)
        if observed.get("role") != "orchestrator":
            raise local.LocalError("wrong_controller: port 7101 is occupied")
    stopped = threading.Event()
    busy = threading.Lock()
    state = {"state": "starting", "detail": "Starting Crucible"}
    notice = {"message": "", "adopt": False}
    icon = pystray.Icon("crucible", icon_image(ICON_FOREGROUND), "Crucible")

    def refresh() -> None:
        from .sharing import read
        icon.title = "Crucible — " + state["state"]
        try:
            shared = read(home) is not None
        except (OSError, ValueError, CrucibleError) as exc:
            shared = False
            notice["message"] = str(exc)
        sharing_label = ("Stop Tailscale sharing" if shared else
                         "Use existing Tailscale sharing" if notice["adopt"] else
                         "Enable Tailscale sharing")
        icon.menu = pystray.Menu(
            pystray.MenuItem(notice["message"] or state["detail"], None, enabled=False),
            pystray.MenuItem("Open Crucible", lambda *_: action("open-console")),
            pystray.MenuItem("Connect an app…", lambda *_: action("connect")),
            pystray.MenuItem("Start Crucible", lambda *_: action("start"), enabled=state["state"] != "running"),
            pystray.MenuItem("Stop Crucible", lambda *_: action("stop"), enabled=state["state"] == "running"),
            pystray.MenuItem(sharing_label, lambda *_: action("sharing")),
            pystray.MenuItem("Close tray icon", lambda *_: close()),
        )
        icon.update_menu()

    def action(verb: str) -> None:
        def run() -> None:
            if not busy.acquire(blocking=False):
                return
            notice["message"] = ""
            try:
                if verb == "sharing":
                    from . import sharing
                    from .host.runner import ProcessRunner
                    runner = ProcessRunner(sys.platform, os.environ)
                    engine = sharing.Engine(home)
                    if sharing.read(home) is not None:
                        sharing.disable(home, runner, engine)
                        notice["message"] = "Tailscale sharing stopped"
                    else:
                        result = sharing.enable(home, runner, engine, adopt=notice["adopt"])
                        notice["message"] = "Tailscale sharing enabled: " + result["authority"]
                    notice["adopt"] = False
                else:
                    state.update(local.act(verb))
                    if state.get("sharing", {}).get("state") == "degraded":
                        notice["message"] = "Crucible is running; sharing needs attention: " + state["sharing"]["detail"]
            except (OSError, ValueError, RuntimeError, CrucibleError) as exc:
                notice["message"] = str(exc)
                if "sharing_unowned" in str(exc):
                    notice["adopt"] = True
                    notice["message"] = "A matching Tailscale forward exists. Select Use existing Tailscale sharing to manage it here."
            finally:
                busy.release()
                refresh()
        threading.Thread(target=run, daemon=True).start()

    def close() -> None:
        stopped.set()
        icon.stop()

    def watch() -> None:
        while not stopped.is_set():
            if (home / "tray.close").exists():
                close()
                return
            if busy.acquire(blocking=False):
                try:
                    state.update(local.status(home))
                except (OSError, ValueError, RuntimeError, CrucibleError) as exc:
                    state.update(state="failed", detail=str(exc))
                finally:
                    busy.release()
                refresh()
            stopped.wait(5)

    refresh()
    threading.Thread(target=watch, daemon=True).start()
    try:
        icon.run()
    finally:
        stopped.set()
        pid_file.unlink(missing_ok=True)
        (home / "tray.close").unlink(missing_ok=True)
