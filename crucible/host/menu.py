"""The tray menu, as a PURE FUNCTION of (distro, engine). PHASE15-HOST.md 4.2.

The one thing a tray program has that can be tested is what it would draw, and
it can only be tested if drawing it is a function rather than a sequence of
calls into somebody's GUI toolkit. So `menu_model()` takes two enums and returns
data; `tray.py` turns that data into pystray objects and does nothing else.

Every cell of the 3 x 5 table is exercised by `tests/test_host_menu.py` — all of
it, not a sample — because "what does the menu say when WSL cannot be asked and
the engine is starting" is exactly the question a screenshot from Owen will be
about.

THE TWO DECISIONS IN HERE THAT ARE NOT COSMETIC
-----------------------------------------------
- **`install-engine` is ABSENT, not disabled**, unless the distro is missing or
  unknown. 4.2 says "only when the distro is absent". A greyed "Install the
  WSL2 engine" on a machine that already has one is an invitation to wonder
  whether it worked.
- **`distro = unknown` offers the install anyway.** `wsl.exe` failing to answer
  is not the same fact as "there is no distro", and reading it as `absent`
  would import a second one; reading it as `present` would hide the only item
  that can fix a machine with no WSL. So it offers, and the title says the
  state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Distro(str, Enum):
    """Whether this machine runs the WSL server or the host-mode child (4.1)."""

    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class Engine(str, Enum):
    """What the last `GET /v1/ping` and the last recovery say (4.1)."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    INSTALLING = "installing"


#: The ids every item carries. A click handler and a test name an item by its
#: id, never by its label — labels are prose and change.
OPEN_CONSOLE = "open-console"
INSTALL_ENGINE = "install-engine"
RESTART_ENGINE = "restart-engine"
STOP_ENGINE = "stop-engine"
OPEN_LOG = "open-log"
QUIT = "quit"

ITEM_IDS = (OPEN_CONSOLE, INSTALL_ENGINE, RESTART_ENGINE, STOP_ENGINE, OPEN_LOG, QUIT)

#: 4.2's label, after section 0's amendment. It is an UPGRADE and says so: the
#: `llama-windows` server already works, and what WSL adds is vLLM/SGLang
#: (parallel page reading, the faster text path) and the five Python job types.
#: A label that said "install the engine" would read as "you have none".
INSTALL_ENGINE_LABEL = "Install the WSL2 engine (faster pages and text; TTS, ASR…)…"


@dataclass(frozen=True)
class MenuItem:
    item_id: str
    label: str
    enabled: bool


@dataclass(frozen=True)
class MenuModel:
    """The whole menu: the title line (4.2) and the items, in order."""

    title: str
    items: tuple[MenuItem, ...]

    def item(self, item_id: str) -> MenuItem | None:
        """The item with this id, or None when this state does not offer it."""
        for entry in self.items:
            if entry.item_id == item_id:
                return entry
        return None


def title_for(distro: Distro, engine: Engine) -> str:
    """The title line, exactly as 4.1's table spells it."""
    if engine is Engine.INSTALLING:
        return "Crucible — installing…"
    if engine is Engine.STARTING:
        return "Crucible — starting…"
    if engine is Engine.FAILED:
        return "Crucible — engine did not start — open the log"
    if engine is Engine.STOPPED:
        return "Crucible — stopped"
    # RUNNING, and WHICH server it is, is the fact 4.1 says the menu must say.
    # `unknown` cannot claim either: the distro probe is what would have told
    # us, and it did not answer.
    if distro is Distro.PRESENT:
        return "Crucible — running (WSL)"
    if distro is Distro.ABSENT:
        # NOT "host mode": section 0's amendment made Windows a BACKEND, and
        # the title names the backend, the way the WSL line names WSL. A
        # person reading "running (llama-windows)" beside "Install the WSL2
        # engine" can see what the upgrade would change.
        return "Crucible — running (llama-windows)"
    return "Crucible — running (WSL unreadable)"


def quit_label(distro: Distro) -> str:
    """What quitting costs, in the label, because it differs by which server."""
    if distro is Distro.ABSENT:
        # The host-mode server is this process's CHILD (4.1), so it goes too.
        return "Quit (stops the engine)"
    if distro is Distro.PRESENT:
        return "Quit (the engine keeps running)"
    return "Quit"


def menu_model(distro: Distro, engine: Engine) -> MenuModel:
    """4.2's menu for this state. Pure: no clock, no environment, no I/O."""
    busy = engine is Engine.INSTALLING
    running = engine is Engine.RUNNING
    items: list[MenuItem] = [
        # Nothing to open when nothing answers: the URL comes from the pairing
        # file (3.6) and points at a server that is up.
        MenuItem(OPEN_CONSOLE, "Open console", running),
    ]
    if distro in (Distro.ABSENT, Distro.UNKNOWN):
        items.append(
            MenuItem(
                INSTALL_ENGINE,
                INSTALL_ENGINE_LABEL,
                not busy,
            )
        )
    items.extend(
        [
            # Restart is offered in every state except while an install holds
            # the machine — including FAILED, which is precisely the state a
            # person wants to retry from after fixing whatever the log said.
            MenuItem(RESTART_ENGINE, "Restart engine", not busy),
            MenuItem(STOP_ENGINE, "Stop engine", running),
            # Always: the log is the one thing that is useful when everything
            # else is not.
            MenuItem(OPEN_LOG, "Open log", True),
            MenuItem(QUIT, quit_label(distro), True),
        ]
    )
    return MenuModel(title=title_for(distro, engine), items=tuple(items))
