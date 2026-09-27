"""Presence — PHASE15-HOST.md 4.1. Which server this machine runs, and is it up.

This is the whole reason `crucible orchestrator` exists. WSL has no boot: nothing starts
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

#: The guest unit every recipe and the restart name. Spelled once.
UNIT_NAME = "crucible.service"

#: The recovery recipe: start the guest's system unit. Touches nothing but
#: Crucible's own unit, and so is permitted in ANY distro.
RECIPE_SYSTEM_UNIT_START = "system-unit-start"

#: The restart of the guest's system unit, PHASE17 4.2's `wsl-unit` restart.
RECIPE_SYSTEM_UNIT_RESTART = "system-unit-restart"

#: The host-mode child's respawn, the one recipe on a machine with no distro.
RECIPE_HOST_MODE_RESPAWN = "host-mode-respawn"


def wsl_boot_argv(distro: str = CRUCIBLE_DISTRO) -> list[str]:
    """`wsl -d crucible --exec true` — 4.1's boot.

    `--exec` and not a bash string: wsl.exe pre-expands `$var` in the implicit
    shell form, and `--exec` is the spelling every other wsl call in this
    system uses for that reason.
    """
    return guest_argv(distro, ["true"])


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

    AS THE USER THE ENGINE RUNS AS (`guest_argv`), fresh-install #27,
    2026-09-26. This spelled `wsl.exe -d <distro> --exec` itself, so it ran as
    whatever the distro's default user was at that moment: root straight after
    the import, `crucible` after a restart. `$HOME/.crucible` then named
    whichever home that user had, and on kylies-pc the tray read neither the
    guest's pairing nor its token after the engine moved from /root to
    /home/crucible, wrote no pairing file and made no claim. `guest_argv` names
    the user in Crucible's own distro and keeps a foreign distro's default.
    """
    return guest_argv(
        distro,
        ["bash", "-lc", 'cat "${CRUCIBLE_HOME:-$HOME/.crucible}/pairing"'],
    )


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
    return guest_argv(distro, ["sleep", "infinity"])


#: How long an upgrading tray's handed-over hold lasts (#35, 2026-09-26). The
#: 1.0.48 host upgrade took 55 s from the old tray's quit to the new tray's
#: start; ten minutes covers a slow download of the interpreter as well, and
#: costs nothing but a guest kept up a little longer if no new tray comes.
HANDOVER_SECONDS = 600


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


def _printed_state(result: RunResult) -> str:
    """The first line systemctl PRINTED, which is the answer — not its code.

    `is-enabled` exits non-zero for a unit that is merely `disabled`, and a
    disabled unit is still a unit that can be restarted.
    """
    text = result.stdout.strip()
    return text.splitlines()[0].strip() if text else ""


def system_systemctl_argv(distro: str, verb: str) -> list[str]:
    """One `systemctl <verb> crucible.service` against the guest's SYSTEM manager.

    No `--user`, no `XDG_RUNTIME_DIR`, no uid to read first — and that is the
    point. WSLg mounts its own tmpfs over `/run/user/<uid>`, which HIDES the
    socket the user manager is listening on (measured 2026-09-16 with
    /proc/self/mountinfo: two mounts on one path, `ss` sees the socket, `ls`
    cannot). `/run/dbus` is not overmounted, so the system manager is
    reachable.

    `-u root` because a system unit is the machine's. It needs no password:
    wsl.exe grants root from the Windows side, which is where this runs.
    """
    return [
        "wsl.exe", "-d", distro, "-u", "root", "--exec",
        "systemctl", verb, UNIT_NAME,
    ]


#: What `is-enabled` prints when the unit EXISTS. `disabled` is in here and
#: that is the point of reading stdout rather than the exit code: a disabled
#: unit exits non-zero and is still a unit `systemctl restart` starts.
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


#: Where WSL registers each distribution for the signed-in user: one subkey per
#: distro, its name in `DistributionName`. The authority `wsl -l` itself reads.
LXSS_KEY = r"Software\Microsoft\Windows\CurrentVersion\Lxss"


def registered_wsl_distros() -> list[str] | None:
    """Distribution names WSL has registered for this user, or None off Windows.

    `[]` when the Lxss key is absent or has no distro under it. That's a fresh
    machine, and it's a FACT here rather than a reading of wsl.exe's prose,
    which is localised.
    """
    try:
        import winreg
    except ImportError:
        return None
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, LXSS_KEY)
    except FileNotFoundError:
        return []
    names: list[str] = []
    with key:
        index = 0
        while True:
            try:
                sub = winreg.EnumKey(key, index)
            except OSError:
                break
            index += 1
            try:
                with winreg.OpenKey(key, sub) as distro:
                    names.append(str(winreg.QueryValueEx(distro, "DistributionName")[0]))
            except OSError:
                continue
    return names


def read_wsl_distros(result: object) -> list[str] | None:
    """The distros `wsl -l -v` reported, `[]` for none, or None when unreadable.

    ONE READER, for the state probe and the installer both (kylies-pc,
    2026-09-26). On a machine with WSL live and NO distribution, `wsl -l -v`
    prints "has no installed distributions" and exits non-zero. The state probe
    read that as an empty list only because it ignored every failure (which
    would also hide a real one), and the installer's import step read it as
    `wsl_read_failed` and stopped. So every first install on a fresh machine
    died at step 2 of 11. A failure is an empty list exactly when WSL has
    nothing registered (`registered_wsl_distros`); any other failure is
    unreadable and says so.
    """
    if getattr(result, "ok"):
        return parse_wsl_list(getattr(result, "stdout"))
    if registered_wsl_distros() == []:
        return []
    return None


#: The user Crucible's OWN distro runs its engine as: `finishImportScript` creates
#: it and writes it into wsl.conf as the default.
GUEST_USER = "crucible"


def guest_argv(distro: str, argv: list[str] | tuple[str, ...]) -> list[str]:
    """`wsl.exe -d <distro> [-u crucible] --exec <argv>`: the one spelling.

    THE USER IS NAMED, NOT DEFAULTED, IN CRUCIBLE'S DISTRO (2026-09-26,
    kylies-pc). `[user] default=crucible` in wsl.conf takes effect only at the
    distro's NEXT start, and the move ran its first guest commands straight after
    the import, so they ran as root: the engine landed in /root/.crucible. After a
    restart the default was `crucible`, the next carry found no ~/.crucible, and
    install.sh minted a new home WITH A NEW TOKEN. The door then refused the
    Windows token, and every later host upgrade failed with 401. A distro somebody
    else made (an app's `Ubuntu`) keeps its own default user: that's the user
    whose engine it is.
    """
    user = ["-u", GUEST_USER] if distro == _crucible_distro() else []
    return ["wsl.exe", "-d", distro, *user, "--exec", *argv]


def _crucible_distro() -> str:
    from .wsl_states import CRUCIBLE_DISTRO

    return CRUCIBLE_DISTRO


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
        #: sentences the log writes about why.
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

    @property
    def distro(self) -> str:
        """THE name of the distro this orchestrator manages. One owner.

        Written for PHASE17 2.5's consequence rather than for tidiness: consent
        makes this "Ubuntu" on Owen's PC while `CRUCIBLE_DISTRO` stays
        "crucible", so every part of the host that acts on the guest has to ask
        the SAME object which one it means. On 1.0.4 the hold and the claim read
        this attribute and the guest carry read `EngineInstall`'s default
        instead, so the host claimed the engine in "Ubuntu" and ran `install.sh`
        in "crucible" — "There is no distribution with the supplied name.",
        measured 2026-09-19 03:10.
        """
        return self._distro

    # ------------------------------------------------------------- probes

    def probe_distro(self) -> tuple[Distro, str]:
        """`present` / `absent` / `unknown`, with what wsl.exe said.

        `unknown` is a state and not an error: 4.1 says the menu offers the
        install from it, and never reads it as `absent`, because importing a
        second distro onto a machine whose WSL merely failed to answer is the
        one mistake here that cannot be undone by pressing the button again.
        """
        result = self._runner.run(wsl_list_argv(), timeout_s=WSL_LIST_TIMEOUT_SECONDS)
        # ONE LINE, KEYED ON THE CODE (FRESH-INSTALL #10/#17, kylies-pc
        # 2026-09-26). This logged `result.said()`, which before WSL was live
        # was wsl.exe's whole usage screen, then its multi-line
        # WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED text, then "has no installed
        # distributions". What the reply MEANS goes in the log; the reader
        # that decides "no distros" is the installer's (`read_wsl_distros`).
        from .wslstate import wsl_answer_line

        names = read_wsl_distros(result)
        if names is None:
            return Distro.UNKNOWN, f"wsl -l -v: {wsl_answer_line(result)}"
        if not result.ok:
            # A fact from the registry WSL itself reads: nothing registered.
            return Distro.ABSENT, f"wsl -l -v: {wsl_answer_line(result)}; no distributions"
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
        `engine-restart` that works — both rest on a unit existing.

        The exit code is NOT the answer: `is-enabled` exits non-zero for a
        unit that is merely `disabled`, and a disabled unit is still a unit
        `systemctl restart` starts. What it PRINTED is the answer.
        """
        result = self._runner.run(
            system_systemctl_argv(self._distro, "is-enabled"),
            timeout_s=RECIPE_TIMEOUT_SECONDS,
        )
        state = _printed_state(result)
        if state in UNIT_STATES:
            return UnitProbe(True, state, f"{UNIT_NAME} is {state}")
        return UnitProbe(False, state, f'no system {UNIT_NAME} in "{self._distro}" ({result.said()})')

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
        """4.1's boot: start the distro, wait, then the recovery."""
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
            "running the recovery"
        )
        if self.recover():
            self._recovery_spent = False
            return self.running_owner(distro, "a recovery recipe brought it up")
        return Presence(
            distro,
            Engine.FAILED,
            "the distro booted and the engine did not start; the recovery was spent",
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

    def recover(self) -> bool:
        """Start the guest's system unit, and say whether the engine answered.

        4.1 gives a down-edge ONE attempt, because the unit restarts itself
        and a tray that keeps trying hides that it is not working.
        """
        probe = self.probe_unit()
        if not probe.readable:
            self._log.write(f"recovery {RECIPE_SYSTEM_UNIT_START}: NOT RUN — {probe.detail}")
            return False
        result = self._runner.run(
            system_systemctl_argv(self._distro, "start"),
            timeout_s=RECIPE_TIMEOUT_SECONDS,
        )
        self._log.write(
            f"recovery {RECIPE_SYSTEM_UNIT_START}: "
            f"{'ok' if result.ok else result.said()}"
        )
        return self._wait_for_ping(10.0)

    def restart_wsl_unit(self) -> bool:
        """PHASE17 4.2's `wsl-unit` restart: `systemctl restart` on the system unit."""
        probe = self.probe_unit()
        if not probe.readable:
            self._log.write(f"restart {RECIPE_SYSTEM_UNIT_RESTART}: NOT RUN — {probe.detail}")
            return False
        result = self._runner.run(
            system_systemctl_argv(self._distro, "restart"),
            timeout_s=RECIPE_TIMEOUT_SECONDS,
        )
        self._log.write(
            f"restart {RECIPE_SYSTEM_UNIT_RESTART}: "
            f"{'ok' if result.ok else result.said()}"
        )
        if self._wait_for_ping(self._boot_wait_s):
            self._recovery_spent = False
            return True
        self._log.write(
            f"restart: nothing on {engine_url('/v1/ping')} after the system "
            f"unit was restarted"
        )
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
            if owner is Owner.NONE:
                # AN ENGINE THAT IS ANSWERING HAS AN OWNER. This used to
                # hand `owner` straight back, so NONE was permanent: `boot`
                # DECIDES an owner (`running_owner`) and the watch tick
                # never did, while `boot`'s own `the recovery was spent`
                # branch returns exactly Owner.NONE.
                #
                # It is not a cosmetic field. `engine_token` reads None for
                # an ownerless host and every authenticated door then
                # answers 503 host_no_token - `/quit` included, so the tray
                # could not even be asked to stop and had to be killed.
                # Measured 2026-09-17, after a guest was reinstalled under a
                # running host.
                #
                # ONLY when nobody owns it: re-deciding an owner already
                # settled would buy a wsl.exe round trip every 15 seconds to
                # re-learn what is known.
                if distro is Distro.PRESENT:
                    return self.running_owner(distro, "the engine answered /v1/ping")
                # No distro to have started it in, and it is answering: it
                # is somebody else's, which is what `found` means.
                return self.adopt(distro)
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
            if self.recover():
                return Presence(
                    distro, Engine.RUNNING, "a recovery brought it back", owner
                )
            return Presence(
                distro,
                Engine.STOPPED,
                f"{RECIPE_SYSTEM_UNIT_START} did not bring it back; use Restart engine",
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

    def hand_over(self, seconds: int = HANDOVER_SECONDS) -> Child | None:
        """Leave a BOUNDED hold behind for the next tray, then let ours go.

        Fresh-install #35 (2026-09-26, kylies-pc). An upgrade quits this tray
        and starts the new one about a minute later, and a distro terminates
        seconds after its last `wsl.exe` session ends (7b.4c): the old tray
        let go, WSL idled the guest, and the new tray found nothing on :7100
        for 30 s before a recovery recipe started the unit again. So the old
        tray starts a `sleep <seconds>` session it does NOT wait for, before
        it ends its own. It is a child nobody reaps, which is why it is
        bounded: it ends by itself whether or not a new tray ever arrives,
        and a new tray that does arrive takes its own hold regardless.

        The new session is started BEFORE ours is released, so there is no
        instant with no session on the distro. None when nothing was held.
        """
        if self.held_distro is None:
            return None
        distro = self.held_distro
        bridge = self._runner.spawn(guest_argv(distro, ["sleep", str(int(seconds))]))
        self._log.write(
            f'hold: "{distro}" is handed over to pid {bridge.pid} for {int(seconds)} s, '
            "so the next tray finds its engine still up (#35)"
        )
        self.release()
        return bridge

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

    def stop_child(self, timeout_s: float = 60.0) -> RunResult | None:
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
