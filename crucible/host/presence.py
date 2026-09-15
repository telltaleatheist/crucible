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
from urllib.parse import urlsplit

from .log import HostLog
from .menu import Distro, Engine, Owner
from .paths import ENGINE_HOST, ENGINE_PORT, engine_url
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
#: `cat` of one small file inside a distro that is ALREADY running.
GUEST_READ_TIMEOUT_SECONDS = 30.0

#: The recipes, by the names 4.1 gives them. The VALUE is the argv, so a test
#: asserts the command rather than trusting the prose next to it.
RECIPE_USER_UNIT_START = "user-unit-start"
RECIPE_USER_BUS_RESTART = "user-bus-restart"
RECIPE_HOST_MODE_RESPAWN = "host-mode-respawn"

#: The guest unit every recipe and the restart name. Spelled once.
UNIT_NAME = "crucible.service"

#: PHASE17 4.2's `wsl-unit` restart, and it is NOT in `RECIPES`.
#:
#: The two in `RECIPES` are RECOVERIES — things to try when an engine that
#: should be up is not. This is the working door into a unit that IS up, which
#: is what a restart asks for: `boot()` on a running engine pings, succeeds
#: immediately and changes nothing, so a restart built out of it would be a
#: button that does nothing whenever it is most obviously pressed. The
#: escalation from here IS `RECIPES`, in order, exactly as `boot()` escalates.
RECIPE_USER_UNIT_RESTART = "user-unit-restart"


def wsl_boot_argv(distro: str = CRUCIBLE_DISTRO) -> list[str]:
    """`wsl -d crucible --exec true` — 4.1's boot.

    `--exec` and not a bash string: wsl.exe pre-expands `$var` in the implicit
    shell form, and `--exec` is the spelling every other wsl call in this
    system uses for that reason.
    """
    return ["wsl.exe", "-d", distro, "--exec", "true"]


def wsl_list_argv() -> list[str]:
    return ["wsl.exe", "-l", "-v"]


def wsl_running_argv() -> list[str]:
    """`wsl -l -v --running` — the distros that are ALREADY up.

    The list the engine hunt (`find_engine`) is allowed to walk. Asking a
    STOPPED distro anything boots it, and booting somebody else's distro to
    find out whether it holds a Crucible is a side effect the host has no
    business having. `-l -v` and not `-l`, because `--running` alone prints a
    different shape and `parse_wsl_list` would then need a second parser.
    """
    return ["wsl.exe", "-l", "-v", "--running"]


def guest_pairing_argv(distro: str) -> list[str]:
    """`cat` the guest's own pairing file (3.6), from Windows.

    `--exec bash -lc` and not `wsl.exe -d X bash -c`: the implicit-shell form
    lets wsl.exe pre-expand `$CRUCIBLE_HOME` on the WINDOWS side, where it is
    empty (BookForge's `wsl-exe-implicit-shell-trap.md`). `installer.py` reads
    the guest's home with the same line for the same reason.
    """
    return [
        "wsl.exe",
        "-d",
        distro,
        "--exec",
        "bash",
        "-lc",
        'cat "${CRUCIBLE_HOME:-$HOME/.crucible}/pairing"',
    ]


def keepalive_argv(distro: str) -> list[str]:
    """Hold a distro open — 7b.4c, measured on the button's night.

    **A WSL distro terminates seconds after the last `wsl.exe` session ends,
    even with systemd units running and linger enabled.** A `Restart=always`
    unit does not keep the VM alive, because the VM is not something the guest
    can hold; only a process on the WINDOWS side can. 4.1 had the host run
    `wsl -d crucible --exec true` and then let go, which boots the distro and
    then lets the thing it is watching disappear on its own.

    So the host holds one session open for as long as a WSL engine is meant to
    be the engine. `sleep infinity` because it is the one command that costs
    nothing and cannot exit: the process the host really wants is the wsl.exe
    on ITS side, and the guest-side sleep is only what gives that something to
    wait for.
    """
    return ["wsl.exe", "-d", distro, "--exec", "sleep", "infinity"]


def pairing_line_authority(line: str) -> str | None:
    """The `host:port` a `crucible://` line points at, or None if it is not one.

    Only the authority, and only after the LAST `@`: the name in the userinfo
    is percent-encoded precisely so that a name containing `@` cannot make
    this ambiguous (`crucible/pairing.py`), and `rsplit` is the half of that
    contract the reader owes.
    """
    parts = urlsplit(line.strip())
    if parts.scheme != "crucible":
        return None
    authority = parts.netloc.rsplit("@", 1)[-1]
    return authority or None


def recipe_argv(name: str, distro: str = CRUCIBLE_DISTRO) -> list[str]:
    """The argv for a named recipe. Unknown names are a programming error."""
    if name == RECIPE_USER_UNIT_START:
        return ["wsl.exe", "-d", distro, "--exec", "systemctl", "--user", "start", "crucible"]
    if name == RECIPE_USER_UNIT_RESTART:
        return [
            "wsl.exe", "-d", distro, "--exec", "systemctl", "--user", "restart", "crucible"
        ]
    if name == RECIPE_USER_BUS_RESTART:
        # As ROOT: this restarts the user manager that owns the bus the user
        # unit needs. uid 1000 is the rootfs's `crucible` user (4b creates
        # exactly one non-root user), and 4.1 names this command literally.
        return ["wsl.exe", "-d", distro, "-u", "root", "--exec", "systemctl", "restart", "user@1000"]
    raise ValueError(
        f"no recipe called {name!r}; the recovery recipes are {RECIPES} and the "
        f"restart is {RECIPE_USER_UNIT_RESTART!r}"
    )


RECIPES: tuple[str, ...] = (RECIPE_USER_UNIT_START, RECIPE_USER_BUS_RESTART)

#: Recipes that may run ONLY in the distro Crucible IMPORTED — PHASE15 4.1a's
#: rule, and CONSENT (PHASE17 2.5) does not widen it.
#:
#: `systemctl restart user@1000` kills every process uid 1000 owns in that
#: distro. In the `crucible` rootfs that is Crucible's own processes and the
#: cost is the restart. In a distro a person also uses it is everything they
#: are running — on the night this rule was written, a five-thousand-step LoRA
#: trainer. Consent says "you may watch, claim and restart the UNIT in this
#: distro"; it does not and cannot say "you may restart everything I am
#: running in it", because the person granting it is naming a distro, not
#: enumerating what is inside it at the moment the recipe fires.
#:
#: There is nothing else to list. `--terminate` and `--unregister` appear on no
#: branch the orchestrator can reach with a distro name it was GIVEN: the one
#: `wsl --terminate` in `wsl_states.py` is the 4c `distro_not_systemd` row and
#: it is hardcoded to `crucible`, `foreign_distro_not_systemd` is `instruct`
#: with no argv at all, and `crucible/uninstall.py` refuses `--unregister` by
#: ruling. Checked 2026-09-15, when consent was built.
DESTRUCTIVE_RECIPES: frozenset[str] = frozenset({RECIPE_USER_BUS_RESTART})


def recipe_permitted(name: str, distro: str) -> bool:
    """May this recipe run in this distro? `False` is refused, never skipped.

    The predicate is the distro's NAME and not the consent flag, deliberately:
    the question a destructive recipe asks is *"did Crucible create this
    rootfs"*, which consent never changes. A caller that gets `False` refuses
    by name (`orchestrator_recipe_not_ours`) and says so in the log — a recipe
    that was quietly not run is a recovery a person believes happened.
    """
    return distro == CRUCIBLE_DISTRO or name not in DESTRUCTIVE_RECIPES


def unit_enabled_argv(distro: str) -> list[str]:
    """`systemctl --user is-enabled crucible.service`, inside a distro.

    PHASE17 2.5's probe: consent names a distro, and this is what turns that
    name into the fact the owner needs — *is there a unit here this
    orchestrator can restart*. It is asked as the ORDINARY user and not
    through `-u root`, because the thing that makes it fail on a distro in the
    state 7b.8 measured is a missing user D-Bus, and the root door onto the
    same manager (`systemctl --user -M <user>@`) was measured failing for the
    same cause on the same machine the same night. A second probe that cannot
    succeed when the first failed is a second round trip for nothing; what the
    first one SAID is carried into the log instead, because "Failed to connect
    to bus" is the sentence that tells a person what to repair.
    """
    return [
        "wsl.exe", "-d", distro, "--exec",
        "systemctl", "--user", "is-enabled", UNIT_NAME,
    ]


#: What `is-enabled` prints when the unit EXISTS. `disabled` is in here and
#: that is the point of reading stdout rather than the exit code: a disabled
#: unit exits non-zero and is still a unit `systemctl --user restart` starts.
#: `not-found` is the one answer that means there is nothing to manage.
UNIT_STATES: frozenset[str] = frozenset(
    {
        "enabled",
        "enabled-runtime",
        "disabled",
        "static",
        "indirect",
        "generated",
        "transient",
        "linked",
        "linked-runtime",
        "masked",
        "masked-runtime",
        "alias",
    }
)


@dataclass(frozen=True)
class UnitProbe:
    """What {@link PresenceWatcher.probe_unit} found. PHASE17 2.5."""

    #: Is there a unit here this orchestrator could restart?
    readable: bool
    #: The state `is-enabled` printed, when it printed one.
    state: str
    #: The sentence for the log — what systemctl said, when it said no.
    detail: str


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
class FoundEngine:
    """An engine that was already answering, and the distro it turned out to be in."""

    distro: str
    #: The guest's OWN pairing line, read out of its home. 3.6: the Windows
    #: file is the host's COPY of the guest's line, never a second composition
    #: of one — a line composed here would carry the HOST's token, and the
    #: engine would refuse every app that read it.
    line: str


@dataclass(frozen=True)
class Presence:
    """The pair 4.1 names, plus who owns it and the sentence the log shows."""

    distro: Distro
    engine: Engine
    detail: str
    owner: Owner


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
        consented: bool = False,
        boot_wait_s: float = BOOT_WAIT_SECONDS,
        watch_s: float = WATCH_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._runner = runner
        self._log = log
        self._distro = distro
        #: PHASE17 2.5: was this distro NAMED by a person in the config, as one
        #: this orchestrator may manage? It changes two things and no others --
        #: the owner a running engine here gets (after the unit probe), and the
        #: sentences the log writes about why. It never widens
        #: `DESTRUCTIVE_RECIPES`, which ask about the rootfs and not about
        #: permission.
        self.consented = consented
        self._boot_wait_s = boot_wait_s
        self.watch_s = watch_s
        self._monotonic = monotonic
        self._sleep = sleep
        #: The host-mode child, when this machine has no distro. The host owns
        #: it; 4.2's Quit label says so.
        self.child: Child | None = None
        #: The engine that was already answering when the host started, if
        #: there was one. Set by `adopt`, read by the pairing write and by the
        #: hold — a `FOUND` engine is still in a distro, and that distro still
        #: has to be held open (7b.4c).
        self.found: FoundEngine | None = None
        #: The held `wsl.exe` session (7b.4c) and the distro it holds.
        self.held: Child | None = None
        self.held_distro: str | None = None
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

    def read_guest_pairing(self, distro: str) -> str | None:
        """The guest's pairing line, or None when that distro has no engine.

        3.6's rule made real: *"the Windows file is the host's COPY of the
        guest's line, because the guest's own home is inside the distro where
        no Windows app looks."* The line is checked against the address the
        host watches before it is believed — a distro holding a Crucible that
        answers somewhere else is not this machine's engine.
        """
        read = self._runner.run(
            guest_pairing_argv(distro), timeout_s=GUEST_READ_TIMEOUT_SECONDS
        )
        if not read.ok:
            return None
        line = read.stdout.strip()
        if pairing_line_authority(line) != f"{ENGINE_HOST}:{ENGINE_PORT}":
            return None
        return line

    def find_engine(self) -> FoundEngine | None:
        """Which RUNNING distro holds the engine that is answering on 7100.

        Asked only of distros that are already up (`wsl_running_argv`), so the
        hunt never boots one. A machine where the answer is None still HAS an
        engine — something answered the ping — it is just not one the host can
        read a pairing line out of, and that is a sentence rather than a
        guess.
        """
        listed = self._runner.run(
            wsl_running_argv(), timeout_s=WSL_LIST_TIMEOUT_SECONDS
        )
        if not listed.ok:
            self._log.write(f"find-engine: wsl -l -v --running: {listed.said()}")
            return None
        for name in parse_wsl_list(listed.stdout):
            line = self.read_guest_pairing(name)
            if line is not None:
                self._log.write(
                    f'find-engine: the engine on {engine_url()} is the "{name}" '
                    "distro's, and this host did not start it"
                )
                return FoundEngine(distro=name, line=line)
        return None

    def probe_unit(self) -> UnitProbe:
        """Is there a `crucible.service` in this distro that could be restarted?

        PHASE17 2.5's gate. Consent names a distro; this is the fact that turns
        that name into an OWNER, and it is asked rather than assumed because
        the two things consent promises — a claim that is true and an
        `engine-restart` that works — both rest on a unit existing. A distro
        whose user bus is unreachable (7b.8 measured exactly that) answers
        nothing here, and the orchestrator then keeps `found` and says why.

        The exit code is NOT the answer: `is-enabled` exits non-zero for a
        unit that is merely `disabled`, and a disabled unit is still a unit
        `systemctl --user restart` starts. What it PRINTED is the answer.
        """
        result = self._runner.run(
            unit_enabled_argv(self._distro), timeout_s=RECIPE_TIMEOUT_SECONDS
        )
        state = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
        if state in UNIT_STATES:
            return UnitProbe(True, state, f"{UNIT_NAME} is {state}")
        return UnitProbe(False, state, result.said())

    def running_owner(self, distro: Distro, detail: str) -> Presence:
        """The owner a RUNNING WSL engine gets — PHASE15 4.1a with PHASE17 2.5.

        Without consent this is `wsl-unit` and always was: the distro is the
        one Crucible imported, and the orchestrator booted the unit in it.

        With consent the distro is somebody else's and the name alone is not
        enough, so the unit is PROBED. A unit that answers makes the owner
        `wsl-unit` — the claim is then a true statement and `engine-restart`
        has a door. A unit that cannot be read leaves the owner `found`, which
        is what the machine was before consent was written, with the reason in
        the log: consent is permission, not a fact about the guest, and an
        orchestrator that took the permission as the fact would claim an
        engine it cannot restart.
        """
        if not self.consented:
            return Presence(distro, Engine.RUNNING, detail, Owner.WSL_UNIT)
        probe = self.probe_unit()
        if probe.readable:
            self._log.write(
                f'consent: "{self._distro}" is named in config.toml and its '
                f"{probe.detail} — owner=wsl-unit (PHASE17 2.5)"
            )
            return Presence(
                distro,
                Engine.RUNNING,
                f'{detail}; "{self._distro}" is consented and its {probe.detail}',
                Owner.WSL_UNIT,
            )
        self._log.write(
            f'consent: "{self._distro}" is named in config.toml, but '
            f"{UNIT_NAME} could not be read there ({probe.detail}), so the "
            "engine stays owner=found — consent is permission to manage a "
            "unit, not evidence that there is one (PHASE17 2.5)"
        )
        return self.adopt(distro)

    def adopt(self, distro: Distro) -> Presence:
        """An engine was already answering. Watch it; never replace it.

        Section 0 is one server per machine, and the host starting a second
        one would make that false in the most expensive way — two claimants on
        7100, and a ping that cannot tell which of them answered.
        """
        self.found = self.find_engine()
        self._recovery_spent = False
        if self.found is None:
            return Presence(
                distro,
                Engine.RUNNING,
                f"something already answers on {engine_url('/v1/ping')} and this "
                "host did not start it; no running distro holds a pairing line "
                "for that address, so there is no line to copy",
                Owner.FOUND,
            )
        return Presence(
            distro,
            Engine.RUNNING,
            f'the engine on {engine_url()} is the "{self.found.distro}" distro\'s '
            "and this host did not start it",
            Owner.FOUND,
        )

    # -------------------------------------------------------------- boot

    def boot(self) -> Presence:
        """4.1's boot: start the distro, wait, then the recipes in order."""
        distro, detail = self.probe_distro()
        if distro is not Distro.PRESENT:
            # Nothing to boot. The host-mode server is `app.py`'s to start,
            # because it is a child this object does not own until it is told.
            return Presence(distro, Engine.STOPPED, detail, Owner.NONE)
        self._log.write(f"boot: {detail}")
        started = self._runner.run(
            wsl_boot_argv(self._distro), timeout_s=WSL_BOOT_TIMEOUT_SECONDS
        )
        if not started.ok:
            self._log.write(f"boot: wsl --exec true failed: {started.said()}")
        if self._wait_for_ping(self._boot_wait_s):
            self._recovery_spent = False
            return self.running_owner(distro, "the engine answered /v1/ping")
        self._log.write(
            f"boot: nothing on {engine_url('/v1/ping')} after {self._boot_wait_s:.0f}s; "
            "running the recovery recipes"
        )
        if self.recover(all_recipes=True):
            self._recovery_spent = False
            return self.running_owner(distro, "a recovery recipe brought it up")
        return Presence(
            distro,
            Engine.FAILED,
            "the distro booted and the engine did not start; both recipes were spent",
            Owner.NONE,
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
            if not recipe_permitted(name, self._distro):
                # BY NAME, and in the log, because a recipe silently not run
                # is a recovery a person believes happened. PHASE15 4.1a's
                # rule survives consent unchanged (PHASE17 2.5).
                self._log.write(
                    f"recovery {name}: REFUSED orchestrator_recipe_not_ours — "
                    f'it restarts every process uid 1000 owns in "{self._distro}", '
                    f'and Crucible imported "{CRUCIBLE_DISTRO}", not that one. '
                    "Consent widens watching, claiming and the unit restart; it "
                    "does not widen this."
                )
                continue
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

    def restart_wsl_unit(self) -> bool:
        """PHASE17 4.2's `wsl-unit` restart: the working door, then `RECIPES`.

        `systemctl --user restart crucible` is one command and it is the
        whole of a restart on a distro whose user bus works. When it does not
        bring the engine back, the escalation is 4.1's two recovery recipes in
        the order 4.1 names them — the SAME order `boot()` uses, because they
        are the same two facts about the same user manager.

        On a distro whose bus is unreachable (7b.8 measured exactly that on
        Owen's Ubuntu) none of the three can work, and this returns False
        rather than pretending. That distro's engine is a `found` one anyway,
        and a `found` engine never reaches this method.
        """
        result = self._runner.run(
            recipe_argv(RECIPE_USER_UNIT_RESTART, self._distro),
            timeout_s=RECIPE_TIMEOUT_SECONDS,
        )
        self._log.write(
            f"restart {RECIPE_USER_UNIT_RESTART}: {'ok' if result.ok else result.said()}"
        )
        if self._wait_for_ping(self._boot_wait_s):
            self._recovery_spent = False
            return True
        self._log.write(
            f"restart: nothing on {engine_url('/v1/ping')} after "
            f"{self._boot_wait_s:.0f}s; escalating to the recovery recipes"
        )
        if self.recover(all_recipes=True):
            self._recovery_spent = False
            return True
        return False

    def respawn_host_mode(self, argv: Sequence[str], env: dict[str, str]) -> Child:
        """`host-mode-respawn` — the one recipe on a machine with no distro."""
        self._log.write(f"recovery {RECIPE_HOST_MODE_RESPAWN}: {' '.join(argv)}")
        self.child = self._runner.spawn(argv, env=env)
        return self.child

    # ------------------------------------------------------------- watch

    def poll(self, distro: Distro, owner: Owner) -> Presence:
        """One watch tick. 4.1: ping; down → one recovery; then `stopped`."""
        if self.ping():
            self._recovery_spent = False
            return Presence(
                distro, Engine.RUNNING, "the engine answered /v1/ping", owner
            )
        if owner is Owner.FOUND:
            # The host did not start it, so it has no recipe for it: there is
            # no unit it may call by name and no child it may respawn. Saying
            # so is the whole of what it can do, and it is more than running
            # somebody else's systemctl would be.
            return Presence(
                distro,
                Engine.STOPPED,
                "the engine this host found is no longer answering; it was not "
                "started here, so there is nothing here to restart",
                Owner.FOUND,
            )
        if self._recovery_spent:
            return Presence(
                distro, Engine.STOPPED, "the engine is not answering", owner
            )
        self._recovery_spent = True
        if distro is Distro.PRESENT:
            if self.recover(all_recipes=False):
                return Presence(
                    distro, Engine.RUNNING, "a recovery brought it back", owner
                )
            return Presence(
                distro,
                Engine.STOPPED,
                f"{RECIPE_USER_UNIT_START} did not bring it back; use Restart engine",
                owner,
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
            owner,
        )

    # -------------------------------------------------------- holding it open

    def hold(self, distro: str) -> Child:
        """Hold a distro open, per 7b.4c. Idempotent while the session lives."""
        if self.held is not None and self.held.poll() is None:
            if self.held_distro == distro:
                return self.held
            self.release()
        self.held = self._runner.spawn(keepalive_argv(distro))
        self.held_distro = distro
        self._log.write(
            f'hold: "{distro}" is held open by pid {self.held.pid} — a distro '
            "terminates seconds after the last wsl.exe session ends, whatever "
            "its units say (7b.4c)"
        )
        return self.held

    def rehold(self) -> Child | None:
        """Take the hold again if it died. Called from the watch, every tick."""
        if self.held_distro is None:
            return None
        if self.held is not None and self.held.poll() is None:
            return self.held
        self._log.write(f'hold: the session on "{self.held_distro}" ended; taking it again')
        distro, self.held, self.held_distro = self.held_distro, None, None
        return self.hold(distro)

    def release(self) -> None:
        """Let the distro go. Quit's job, and the hold's own re-take."""
        if self.held is None:
            return
        if self.held.poll() is None:
            self.held.terminate()
            self.held.wait(10.0)
        self._log.write(f'hold: released "{self.held_distro}"')
        self.held = None
        self.held_distro = None

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
