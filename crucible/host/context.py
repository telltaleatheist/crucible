from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import VERSION
from ..platform.runner import Runner
from . import installer
from .log import HostLog
from .presence import Presence, PresenceWatcher

DEFAULT_RELEASE = VERSION

INSTALL_SH_URL = (
    "https://github.com/telltaleatheist/crucible/releases/download/"
    "v{release}/install.sh"
)


@dataclass
class HostContext:
    runner: Runner
    log: HostLog
    home: Path
    watcher: PresenceWatcher
    presence: Presence
    release: str = DEFAULT_RELEASE
    name: str = ""

    def install_walk(self, emit: installer.Emit, **options: Any) -> installer.EngineInstall:
        return installer.EngineInstall(
            self.runner,
            emit,
            release=self.release,
            home=self.home,
            install_sh_url=INSTALL_SH_URL.format(release=self.release),
            **options,
        )

    def logged_as(self, prefix: str) -> installer.Emit:
        return lambda event: self.log.write(f"{prefix}: {event.event}: {event.data}")
