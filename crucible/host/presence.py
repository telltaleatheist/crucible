"""Presence — PHASE15-HOST.md 4.1. Which server this machine runs, and is it up.

This is the whole reason `crucible host` exists. WSL has no boot: nothing starts
a distro at login, so before this the engine was down after every reboot until
an app happened to poke it, and a clean stop on 2026-09-14 left it down at 16:10
with nobody noticing. Windows is the only place a process can run that is able
to notice.

WHAT IT DOES NOT DO, AND THAT IS THE DESIGN
--------------------------------------------
It does not restart in a loop. 4.1: "It never loops on restart; the systemd
unit's own `Restart=` handles crashes" — and that unit is now `Restart=always`
precisely so that this half does not have to be (the ruling is in
`crucible/service.py`). One recovery per down-edge, then a state with a name and
a menu item, because a tray that silently retries for an hour is a tray that
tells a person nothing while their machine does something.

THE RECIPES ARE NAMED BECAUSE THEY WERE FOUND, NOT DESIGNED
------------------------------------------------------------
`user-unit-start` and `user-bus-restart` are the two things that actually
brought the unit up on 2026-09-14 (4.1 records them in that order). The second
exists because a distro booted by `wsl.exe --exec` sometimes has no user D-Bus
at all, and `systemctl --user` then fails with "Failed to connect to bus" — at
which point restarting `user@1000` as root is what creates the session the user
unit needs. Doing the second WITHOUT trying the first would restart a working
session for nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

from .log import HostLog
from .menu import Distro, Engine
from .paths import engine_url
from .runner import Child, RunResult, Runner
from .wsl_states import CRUCIBLE_DISTRO

#: 4.1's two numbers, with 4.1's names.
BOOT_WAIT_SECONDS = 30
WATCH_SECONDS = 15

#: One ping must not hang the tray for longer than the watch interval.
PING_TIMEOUT_SECONDS = 4.0
#: `wsl.exe -l -v` on a machine whose WSL service is starting is slow, not stuck.
WSL_LIST_TIMEOUT_SECONDS = 20.0
#: `wsl -d crucible --exec true` BOOTS a distro, which is a VM starting.
WSL_BOOT_TIMEOUT_SECONDS = 120.0
#: A systemctl call inside a booted distro.
RECIPE_TIMEOUT_SECONDS = 60.0

#: The recipes, by the names 4.1 gives them. The VALUE is the argv, so a test
#: asserts the command rather than trusting the prose next to it.
RECIPE_USER_UNIT_START = "user-unit-start"
RECIPE_USER_BUS_RESTART = "user-bus-restart"
RECIPE_HOST_MODE_RESPAWN = "host-mode-respawn"


def wsl_boot_argv(distro: str = CRUCIBLE_DISTRO) -> list[str]:
    """`wsl -d crucible --exec true` — 4.1's boot.

    `--exec` and not a bash string: wsl.exe pre-expands `$var` in the implicit
    shell form, and `--exec` is the spelling every other wsl call in this
    system uses for that reason.
    """
    return ["wsl.exe", "-d", distro, "--exec", "true"]


def wsl_list_argv() -> list[str]:
    return ["wsl.exe", "-l", "-v"]


def recipe_argv(name: str, distro: str = CRUCIBLE_DISTRO) -> list[str]:
    """The argv for a named recipe. Unknown names are a programming error."""
    if name == RECIPE_USER_UNIT_START:
        return ["wsl.exe", "-d", distro, "--exec", "systemctl", "--user", "start", "crucible"]
    if name == RECIPE_USER_BUS_RESTART:
        # As ROOT: this restarts the user manager that owns the bus the user
        # unit needs. uid 1000 is the rootfs's `crucible` user (4b creates
        # exactly one non-root user), and 4.1 names this command literally.
        return ["wsl.exe", "-d", distro, "-u", "root", "--exec", "systemctl", "restart", "user@1000"]
    raise ValueError(f"no recipe called {name!r}; the recipes are {RECIPES}")


RECIPES: tuple[str, ...] = (RECIPE_USER_UNIT_START, RECIPE_USER_BUS_RESTART)


def parse_wsl_list(text: str) -> list[str]:
    """The distro names out of `wsl -l -v`.

    wsl.exe writes UTF-16LE; the runner has already decoded it, but a NUL can
    survive a lossy decode, so they are stripped here rather than trusted away.
    The header row is skipped by its own words rather than by position, because
    it is localised and its position is the only thing that is not.
    """
    names: list[str] = []
    for raw in text.replace("\x00", "").splitlines():
        line = raw.strip()
        if line == "":
            continue
        if line.startswith("*"):
            line = line[1:].strip()
        first = line.split()[0] if line.split() else ""
        if first.upper() in ("NAME", "NOM", "NAAM"):
            continue
        # A row is `<name> <state> <version>`; a header in a language we do not
        # know has no integer in its last column, which is what tells them
        # apart without a table of translations.
        parts = line.split()
        if len(parts) >= 3 and parts[-1].isdigit():
            names.append(" ".join(parts[:-2]))
    return names


@dataclass(frozen=True)
class Presence:
    """The pair 4.1 names, plus the sentence the log and the tray show."""

    distro: Distro
    engine: Engine
    detail: str


class PresenceWatcher:
    """Boots, pings, recovers, and answers with a {@link Presence}.

    Everything it does to the machine goes through `runner`, so the whole of
    4.1 is exercised in `tests/test_host_presence.py` on a machine with no WSL.
    """

    def __init__(
        self,
        runner: Runner,
        log: HostLog,
        *,
        distro: str = CRUCIBLE_DISTRO,
        boot_wait_s: float = BOOT_WAIT_SECONDS,
        watch_s: float = WATCH_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._runner = runner
        self._log = log
        self._distro = distro
        self._boot_wait_s = boot_wait_s
        self.watch_s = watch_s
        self._monotonic = monotonic
        self._sleep = sleep
        #: The host-mode child, when this machine has no distro. The host owns
        #: it; 4.2's Quit label says so.
        self.child: Child | None = None
        #: One recovery per down-edge (4.1). Cleared when a ping succeeds.
        self._recovery_spent = False

    # ------------------------------------------------------------- probes

    def probe_distro(self) -> tuple[Distro, str]:
        """`present` / `absent` / `unknown`, with what wsl.exe said.

        `unknown` is a state and not an error: 4.1 says the menu offers the
        install from it, and never reads it as `absent`, because importing a
        second distro onto a machine whose WSL merely failed to answer is the
        one mistake here that cannot be undone by pressing the button again.
        """
        result = self._runner.run(wsl_list_argv(), timeout_s=WSL_LIST_TIMEOUT_SECONDS)
        if not result.ok:
            return Distro.UNKNOWN, f"wsl -l -v: {result.said()}"
        names = parse_wsl_list(result.stdout)
        if self._distro in names:
            return Distro.PRESENT, f'wsl -l -v lists "{self._distro}"'
        listed = ", ".join(names) if names else "nothing"
        return Distro.ABSENT, f'wsl -l -v lists {listed} and no "{self._distro}"'

    def ping(self) -> bool:
        """Did `GET /v1/ping` answer at all?

        ANY status counts, including a 4xx. Something answering on 7100 with a
        refusal is a server that is up; treating that as "down" would have the
        host boot a distro because a token was wrong.
        """
        return self._runner.get(engine_url("/v1/ping"), timeout_s=PING_TIMEOUT_SECONDS) is not None

    # -------------------------------------------------------------- boot

    def boot(self) -> Presence:
        """4.1's boot: start the distro, wait, then the recipes in order."""
        distro, detail = self.probe_distro()
        if distro is not Distro.PRESENT:
            # Nothing to boot. The host-mode server is `app.py`'s to start,
            # because it is a child this object does not own until it is told.
            return Presence(distro, Engine.STOPPED, detail)
        self._log.write(f"boot: {detail}")
        started = self._runner.run(
            wsl_boot_argv(self._distro), timeout_s=WSL_BOOT_TIMEOUT_SECONDS
        )
        if not started.ok:
            self._log.write(f"boot: wsl --exec true failed: {started.said()}")
        if self._wait_for_ping(self._boot_wait_s):
            self._recovery_spent = False
            return Presence(distro, Engine.RUNNING, "the engine answered /v1/ping")
        self._log.write(
            f"boot: nothing on {engine_url('/v1/ping')} after {self._boot_wait_s:.0f}s; "
            "running the recovery recipes"
        )
        if self.recover(all_recipes=True):
            self._recovery_spent = False
            return Presence(distro, Engine.RUNNING, "a recovery recipe brought it up")
        return Presence(
            distro,
            Engine.FAILED,
            "the distro booted and the engine did not start; both recipes were spent",
        )

    def _wait_for_ping(self, seconds: float) -> bool:
        deadline = self._monotonic() + seconds
        while True:
            if self.ping():
                return True
            if self._monotonic() >= deadline:
                return False
            self._sleep(1.0)

    # ---------------------------------------------------------- recovery

    def recover(self, *, all_recipes: bool) -> bool:
        """Run the recipes, in order, until one leaves the engine answering.

        `all_recipes=False` is the WATCH's budget: 4.1 gives a down-edge ONE
        attempt, because the unit restarts itself and a tray that keeps trying
        hides that it is not working.
        """
        for name in RECIPES:
            argv = recipe_argv(name, self._distro)
            result = self._runner.run(argv, timeout_s=RECIPE_TIMEOUT_SECONDS)
            self._log.write(
                f"recovery {name}: {'ok' if result.ok else result.said()}"
            )
            if self._wait_for_ping(10.0):
                return True
            if not all_recipes:
                return False
        return False

    def respawn_host_mode(self, argv: Sequence[str], env: dict[str, str]) -> Child:
        """`host-mode-respawn` — the one recipe on a machine with no distro."""
        self._log.write(f"recovery {RECIPE_HOST_MODE_RESPAWN}: {' '.join(argv)}")
        self.child = self._runner.spawn(argv, env=env)
        return self.child

    # ------------------------------------------------------------- watch

    def poll(self, distro: Distro) -> Presence:
        """One watch tick. 4.1: ping; down → one recovery; then `stopped`."""
        if self.ping():
            self._recovery_spent = False
            return Presence(distro, Engine.RUNNING, "the engine answered /v1/ping")
        if self._recovery_spent:
            return Presence(distro, Engine.STOPPED, "the engine is not answering")
        self._recovery_spent = True
        if distro is Distro.PRESENT:
            if self.recover(all_recipes=False):
                return Presence(distro, Engine.RUNNING, "a recovery brought it back")
            return Presence(
                distro,
                Engine.STOPPED,
                f"{RECIPE_USER_UNIT_START} did not bring it back; use Restart engine",
            )
        # No distro: the child is ours, and whether it is alive is a question
        # with an answer rather than a probe.
        alive = self.child is not None and self.child.poll() is None
        return Presence(
            distro,
            Engine.STOPPED,
            "the host-mode server is running and not answering"
            if alive
            else "the host-mode server is not running",
        )

    def stop_child(self, timeout_s: float = 20.0) -> RunResult | None:
        """Stop the host-mode child, if there is one. Used by Quit and by 4.3."""
        if self.child is None:
            return None
        if self.child.poll() is not None:
            self.child = None
            return None
        self.child.terminate()
        code = self.child.wait(timeout_s)
        self._log.write(f"host-mode server stopped (exit {code})")
        self.child = None
        return RunResult(code=code, stdout="", stderr="", failure=None)
