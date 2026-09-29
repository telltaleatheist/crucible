from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..config import crucible_home
from ..errors import CrucibleError

Ask = Callable[[str], bool]

SERVICE_INSTALL_SECONDS = 120

NOT_INSTALLED = "crucible_not_installed"


class HostRefusal(CrucibleError):
    pass


def installed_here(home: Path, platform: str = sys.platform) -> bool:
    if platform == "win32":
        return (home / "installation.json").is_file() or (home / "host" / "pythonw.exe").is_file()
    return (home / "config.toml").is_file()


def public_question(names: str) -> str:
    return (
        f"This PC's network {names} is marked Public, so Windows keeps other computers out.\n\n"
        "If it is your home or office network, Crucible can mark it Private so they can use "
        "this PC. Say No on a cafe, hotel, airport or any other shared network.\n\n"
        f"Mark {names} as Private?"
    )


def logs_folder(home: Path) -> Path:
    logs = home / "logs"
    return logs if logs.is_dir() else home


def open_argv(path: Path, platform: str = sys.platform) -> list[str] | None:
    if platform == "win32":
        return None
    return ["open" if platform == "darwin" else "xdg-open", str(path)]


class LocalHost:
    def __init__(self, home: Path | None = None, platform: str = sys.platform) -> None:
        self.home = home if home is not None else crucible_home()
        self.platform = platform

    def status(self) -> Mapping[str, Any]:
        from .. import local

        if not installed_here(self.home, self.platform):
            raise HostRefusal(f"{NOT_INSTALLED}: there is no Crucible installation at {self.home}")
        if self.platform == "win32" and not (self.home / "pairing").exists():
            return {"state": "stopped", "detail": "Crucible's background helper is not running"}
        return local.status(self.home)

    def act(self, action: str) -> Mapping[str, Any]:
        from .. import local

        if action == "restart":
            local.run_engine_verb("stop", self.home)
        elif action == "repair":
            self._service_install()
        return local.run_engine_verb("start", self.home)

    def _service_install(self) -> None:
        argv = [sys.executable, "-m", "crucible.cli", "service", "install"]
        done = subprocess.run(argv, capture_output=True, text=True, timeout=SERVICE_INSTALL_SECONDS,
                              env=dict(os.environ, CRUCIBLE_HOME=str(self.home)))
        if done.returncode != 0:
            said = (done.stderr or done.stdout).strip().splitlines()[-3:]
            raise HostRefusal(
                "service_install_failed: " + " ".join(said)
                + f". Run `crucible doctor` in a terminal to see what is wrong; its log is in {logs_folder(self.home)}"
            )

    def lan_supported(self) -> bool:
        return self.platform == "win32"

    def lan_record(self) -> Mapping[str, Any] | None:
        if not self.lan_supported():
            return None
        from .. import lan

        return lan.read(self.home)

    def set_lan(self, on: bool, ask: Ask) -> Mapping[str, Any]:
        from .. import lan
        from ..platform.runner import ProcessRunner
        from ..sharing import PairedEngine

        runner = ProcessRunner(self.platform, os.environ)
        engine = PairedEngine(self.home, "lan")
        if not on:
            return lan.disable(self.home, runner, engine)

        def ask_private(public: Sequence[Any]) -> bool:
            return ask(public_question(", ".join(interface.label for interface in public)))

        return lan.enable(self.home, runner, engine, ask_private=ask_private)

    def open_logs(self) -> Path:
        folder = logs_folder(self.home)
        argv = open_argv(folder, self.platform)
        if argv is None:
            os.startfile(str(folder))
        else:
            subprocess.Popen(argv)
        return folder
