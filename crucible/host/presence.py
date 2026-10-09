from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence
from urllib.parse import urlsplit

from .. import wsl
from ..platform.paths import ENGINE_HOST, ENGINE_PORT, engine_url
from ..platform.runner import Child, Runner, RunResult
from ..service import UNIT_NAME
from ..wsl import CRUCIBLE_DISTRO
from .log import HostLog
from .state import Distro, Engine, Owner
from .wslstate import read_wsl_distros, wsl_answer_line

BOOT_WAIT_SECONDS = 30
WATCH_SECONDS = 15

UNIT_START_BUDGET_SECONDS = 120

STARTING_UNIT_STATES: frozenset[str] = frozenset({"active", "activating", "reloading"})

PING_TIMEOUT_SECONDS = 4.0
WSL_LIST_TIMEOUT_SECONDS = 20.0
WSL_BOOT_TIMEOUT_SECONDS = 120.0
RECIPE_TIMEOUT_SECONDS = 60.0
GUEST_READ_TIMEOUT_SECONDS = 30.0

RECIPE_SYSTEM_UNIT_START = "system-unit-start"

RECIPE_SYSTEM_UNIT_RESTART = "system-unit-restart"

RECIPE_HOST_MODE_RESPAWN = "host-mode-respawn"


def wsl_boot_argv(distro: str = CRUCIBLE_DISTRO) -> list[str]:
    return wsl.guest_argv(distro, ["true"])


def wsl_list_argv() -> list[str]:
    return wsl.list_argv()


def wsl_running_argv() -> list[str]:
    return wsl.list_argv(running=True)


def guest_pairing_argv(distro: str) -> list[str]:
    return wsl.pairing_argv(distro)


def keepalive_argv(distro: str) -> list[str]:
    return wsl.guest_argv(distro, ["sleep", "infinity"])


HANDOVER_SECONDS = 600


def pairing_line_authority(line: str) -> str | None:
    parts = urlsplit(line.strip())
    if parts.scheme != "crucible":
        return None
    authority = parts.netloc.rsplit("@", 1)[-1]
    return authority or None


def _printed_state(result: RunResult) -> str:
    text = result.stdout.strip()
    return text.splitlines()[0].strip() if text else ""


def system_systemctl_argv(distro: str, verb: str) -> list[str]:
    return wsl.root_argv(distro, ["systemctl", verb, UNIT_NAME])


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
    readable: bool
    state: str
    detail: str


@dataclass(frozen=True)
class FoundEngine:
    distro: str
    line: str


@dataclass(frozen=True)
class Presence:
    distro: Distro
    engine: Engine
    detail: str
    owner: Owner


class PresenceWatcher:
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
        self.consented = consented
        self._boot_wait_s = boot_wait_s
        self.watch_s = watch_s
        self._monotonic = monotonic
        self._sleep = sleep
        self.child: Child | None = None
        self.found: FoundEngine | None = None
        self.held: Child | None = None
        self.held_distro: str | None = None
        self._recovery_spent = False

    @property
    def distro(self) -> str:
        return self._distro


    def probe_distro(self) -> tuple[Distro, str]:
        result = self._runner.run(wsl_list_argv(), timeout_s=WSL_LIST_TIMEOUT_SECONDS)

        names = read_wsl_distros(result)
        if names is None:
            return Distro.UNKNOWN, f"wsl -l -v: {wsl_answer_line(result)}"
        if not result.ok:
            return Distro.ABSENT, f"wsl -l -v: {wsl_answer_line(result)}; no distributions"
        if self._distro in names:
            return Distro.PRESENT, f'wsl -l -v lists "{self._distro}"'
        listed = ", ".join(names) if names else "nothing"
        return Distro.ABSENT, f'wsl -l -v lists {listed} and no "{self._distro}"'

    def ping(self) -> bool:
        return self._runner.get(engine_url("/v1/ping"), timeout_s=PING_TIMEOUT_SECONDS) is not None

    def read_guest_pairing(self, distro: str) -> str | None:
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
        listed = self._runner.run(
            wsl_running_argv(), timeout_s=WSL_LIST_TIMEOUT_SECONDS
        )
        if not listed.ok:
            self._log.write(f"find-engine: wsl -l -v --running: {listed.output_tail()}")
            return None
        for name in wsl.parse_distro_list(listed.stdout):
            line = self.read_guest_pairing(name)
            if line is not None:
                self._log.write(
                    f'find-engine: the engine on {engine_url()} is the "{name}" '
                    "distro's, and this controller did not start it"
                )
                return FoundEngine(distro=name, line=line)
        return None

    def probe_unit(self) -> UnitProbe:
        result = self._runner.run(
            system_systemctl_argv(self._distro, "is-enabled"),
            timeout_s=RECIPE_TIMEOUT_SECONDS,
        )
        state = _printed_state(result)
        if state in UNIT_STATES:
            return UnitProbe(True, state, f"{UNIT_NAME} is {state}")
        return UnitProbe(False, state, f'no system {UNIT_NAME} in "{self._distro}" ({result.output_tail()})')

    def running_owner(self, distro: Distro, detail: str) -> Presence:
        if not self.consented:
            return Presence(distro, Engine.RUNNING, detail, Owner.WSL_UNIT)
        probe = self.probe_unit()
        if probe.readable:
            self._log.write(
                f'consent: "{self._distro}" is named in config.toml and its '
                f"{probe.detail} — owner=wsl-unit (docs/internals/host-and-platform.md, \"Ownership\")"
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
            "unit, not evidence that there is one (docs/internals/host-and-platform.md, \"Ownership\")"
        )
        return self.adopt(distro)

    def adopt(self, distro: Distro) -> Presence:
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
            "and this controller did not start it",
            Owner.FOUND,
        )


    def boot(self) -> Presence:
        distro, detail = self.probe_distro()
        if distro is not Distro.PRESENT:
            return Presence(distro, Engine.STOPPED, detail, Owner.NONE)
        self._log.write(f"boot: {detail}")
        started = self._runner.run(
            wsl_boot_argv(self._distro), timeout_s=WSL_BOOT_TIMEOUT_SECONDS
        )
        if not started.ok:
            self._log.write(f"boot: wsl --exec true failed: {started.output_tail()}")
        if self.wait_for_ping(self._boot_wait_s):
            self._recovery_spent = False
            return self.running_owner(distro, "the engine answered /v1/ping")
        if self._wait_for_a_starting_unit():
            self._recovery_spent = False
            return self.running_owner(distro, f"the engine answered /v1/ping once {UNIT_NAME} had started")
        if self.recover():
            self._recovery_spent = False
            return self.running_owner(distro, "a recovery recipe brought it up")
        return Presence(
            distro,
            Engine.FAILED,
            "the distro booted and the engine did not start; the recovery was spent",
            Owner.NONE,
        )

    def wait_for_ping(self, seconds: float) -> bool:
        deadline = self._monotonic() + seconds
        while True:
            if self.ping():
                return True
            if self._monotonic() >= deadline:
                return False
            self._sleep(1.0)

    def unit_active_state(self) -> str:
        result = self._runner.run(
            system_systemctl_argv(self._distro, "is-active"),
            timeout_s=RECIPE_TIMEOUT_SECONDS,
        )
        return _printed_state(result)

    def _wait_for_a_starting_unit(self) -> bool:
        state = self.unit_active_state()
        if state not in STARTING_UNIT_STATES:
            self._log.write(
                f"boot: nothing on {engine_url('/v1/ping')} after {self._boot_wait_s:.0f}s "
                f"and {UNIT_NAME} is {state or 'unreadable'}, so it is not starting; "
                "running the recovery"
            )
            return False
        self._log.write(
            f"boot: nothing on {engine_url('/v1/ping')} yet after {self._boot_wait_s:.0f}s, "
            f"and {UNIT_NAME} is {state}: the server is still starting (the first start "
            "after an install imports and compiles everything it serves with before it "
            f"binds the port), so it gets up to {UNIT_START_BUDGET_SECONDS}s more before "
            "anything is called a failure"
        )
        if self.wait_for_ping(UNIT_START_BUDGET_SECONDS):
            return True
        self._log.write(
            f"boot: {UNIT_NAME} was {state} and nothing answered on "
            f"{engine_url('/v1/ping')} within "
            f"{self._boot_wait_s + UNIT_START_BUDGET_SECONDS:.0f}s; running the recovery"
        )
        return False

    def recover(self) -> bool:
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
            f"{'ok' if result.ok else result.output_tail()}"
        )
        return self.wait_for_ping(10.0)

    def restart_wsl_unit(self) -> bool:
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
            f"{'ok' if result.ok else result.output_tail()}"
        )
        if self.wait_for_ping(self._boot_wait_s):
            self._recovery_spent = False
            return True
        self._log.write(
            f"restart: nothing on {engine_url('/v1/ping')} after the system "
            f"unit was restarted"
        )
        return False

    def respawn_host_mode(self, argv: Sequence[str], env: dict[str, str]) -> Child:
        self._log.write(f"recovery {RECIPE_HOST_MODE_RESPAWN}: {' '.join(argv)}")
        self.child = self._runner.spawn(argv, env=env)
        return self.child


    def poll(self, distro: Distro, owner: Owner) -> Presence:
        if self.ping():
            self._recovery_spent = False
            if owner is Owner.NONE:
                if distro is Distro.PRESENT:
                    return self.running_owner(distro, "the engine answered /v1/ping")
                return self.adopt(distro)
            return Presence(
                distro, Engine.RUNNING, "the engine answered /v1/ping", owner
            )
        if owner is Owner.FOUND:
            return Presence(
                distro,
                Engine.STOPPED,
                "the engine this controller found is no longer answering; it was not "
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
        alive = self.child is not None and self.child.poll() is None
        return Presence(
            distro,
            Engine.STOPPED,
            "the host-mode server is running and not answering"
            if alive
            else "the host-mode server is not running",
            owner,
        )


    def hold(self, distro: str) -> Child:
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
        if self.held_distro is None:
            return None
        if self.held is not None and self.held.poll() is None:
            return self.held
        self._log.write(f'hold: the session on "{self.held_distro}" ended; taking it again')
        distro, self.held, self.held_distro = self.held_distro, None, None
        return self.hold(distro)

    def hand_over(self, seconds: int = HANDOVER_SECONDS) -> Child | None:
        if self.held_distro is None:
            return None
        distro = self.held_distro
        bridge = self._runner.spawn(wsl.guest_argv(distro, ["sleep", str(int(seconds))]))
        self._log.write(
            f'hold: "{distro}" is handed over to pid {bridge.pid} for {int(seconds)} s, '
            "so the next tray finds its engine still up (#35)"
        )
        self.release()
        return bridge

    def release(self) -> None:
        if self.held is None:
            return
        if self.held.poll() is None:
            self.held.terminate()
            self.held.wait(10.0)
        self._log.write(f'hold: released "{self.held_distro}"')
        self.held = None
        self.held_distro = None

    def stop_child(self, timeout_s: float = 60.0) -> RunResult | None:
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
