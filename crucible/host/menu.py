from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Distro(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class Engine(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    INSTALLING = "installing"


class Owner(str, Enum):
    NONE = "none"
    WSL_UNIT = "wsl-unit"
    HOST_CHILD = "host-child"
    FOUND = "found"


OPEN_CONSOLE = "open-console"
INSTALL_ENGINE = "install-engine"
RESTART_ENGINE = "restart-engine"
STOP_ENGINE = "stop-engine"
OPEN_LOG = "open-log"
QUIT = "quit"
TRY_AGAIN = "try-again"
RESTART_OWED = "restart-owed"

ITEM_IDS = (
    OPEN_CONSOLE, INSTALL_ENGINE, RESTART_OWED, TRY_AGAIN, RESTART_ENGINE, STOP_ENGINE, OPEN_LOG, QUIT,
)

TRY_AGAIN_STATES = frozenset({"cannot", "failed"})

TRY_AGAIN_LABEL = "Try again: set up the Linux engine"
RESTART_OWED_LABEL = "Needs Update and restart (to install WSL), then sign in"

INSTALL_ENGINE_LABEL = "Install the WSL2 engine (faster pages and text; TTS, ASR…)…"


@dataclass(frozen=True)
class MenuItem:
    item_id: str
    label: str
    enabled: bool


@dataclass(frozen=True)
class MenuModel:
    title: str
    items: tuple[MenuItem, ...]

    def item(self, item_id: str) -> MenuItem | None:
        for entry in self.items:
            if entry.item_id == item_id:
                return entry
        return None


def title_for(distro: Distro, engine: Engine, owner: Owner) -> str:
    if engine is Engine.INSTALLING:
        return "Crucible — installing…"
    if engine is Engine.STARTING:
        return "Crucible — starting…"
    if engine is Engine.FAILED:
        return "Crucible — engine did not start — open the log"
    if engine is Engine.STOPPED:
        return "Crucible — stopped"
    if owner is Owner.FOUND:
        return "Crucible — running (found on this machine)"
    if distro is Distro.PRESENT:
        return "Crucible — running (WSL)"
    if distro is Distro.ABSENT:
        return "Crucible — running (llama-windows)"
    return "Crucible — running (WSL unreadable)"


def quit_label(distro: Distro, owner: Owner) -> str:
    if owner is Owner.FOUND:
        return "Quit (the engine keeps running)"
    if distro is Distro.ABSENT:
        return "Quit (stops the engine)"
    if distro is Distro.PRESENT:
        return "Quit (the engine keeps running)"
    return "Quit"


def outcome_items(outcome_state: str | None, *, busy: bool) -> list[MenuItem]:
    if outcome_state == "reboot-pending":
        return [MenuItem(RESTART_OWED, RESTART_OWED_LABEL, False)]
    if outcome_state in TRY_AGAIN_STATES:
        return [MenuItem(TRY_AGAIN, TRY_AGAIN_LABEL, not busy)]
    return []


def menu_model(
    distro: Distro, engine: Engine, owner: Owner, outcome_state: str | None = None
) -> MenuModel:
    busy = engine is Engine.INSTALLING
    running = engine is Engine.RUNNING
    found = owner is Owner.FOUND
    items: list[MenuItem] = [
        MenuItem(OPEN_CONSOLE, "Open console", running),
    ]
    if distro in (Distro.ABSENT, Distro.UNKNOWN) and not found:
        items.append(
            MenuItem(
                INSTALL_ENGINE,
                INSTALL_ENGINE_LABEL,
                not busy,
            )
        )
    if not found:
        items.extend(outcome_items(outcome_state, busy=busy))
    items.extend(
        [
            MenuItem(RESTART_ENGINE, "Restart engine", not busy and not found),
            MenuItem(STOP_ENGINE, "Stop engine", running and not found),
            MenuItem(OPEN_LOG, "Open log", True),
            MenuItem(QUIT, quit_label(distro, owner), True),
        ]
    )
    return MenuModel(title=title_for(distro, engine, owner), items=tuple(items))
