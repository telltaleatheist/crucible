from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Protocol

from ..platform.hostconfig import CONSENT_TABLE, WSL_KEY, WSL_NEVER, declined_wsl, read_token
from ..platform.paths import ENGINE_PORT, engine_url
from ..wsl import CRUCIBLE_DISTRO
from . import installer, outcome, wslstate
from .catalog import CatalogPort, GuestCatalog, HttpCatalog
from .context import HostContext
from .controller_door import OrchestratorDoor
from .errors import HostError
from .state import EngineDecision, Owner

Emit = installer.Emit

MoveSequence = Callable[[Emit], None]


@dataclass(frozen=True)
class ResumeRule:
    codes: frozenset[str]
    rechecked: str
    resumes: Callable[[wslstate.LiveWsl], bool]
    why: str


RESUME_RULES: tuple[ResumeRule, ...] = (
    ResumeRule(
        outcome.TRANSIENT_CANNOT_CODES, "WSL",
        lambda live: live.live, "resumed: WSL is live now",
    ),
    ResumeRule(
        outcome.FIRMWARE_CANNOT_CODES, "virtualization",
        lambda live: live.answer.kind != "no_hypervisor", "resumed: virtualization is on now",
    ),
)


def decide_engine(context: HostContext, move: Callable[[str], EngineDecision]) -> EngineDecision:
    log = context.log.write
    if context.presence.owner is Owner.FOUND:
        log(
            "engine: this machine's engine is one this controller did not "
            "start (owner=found), so nothing is moved (PHASE17 4.1a)"
        )
        return EngineDecision.FOUND
    try:
        declined = declined_wsl(context.home)
    except HostError as exc:
        log(f"engine: {exc.code}: {exc.message}")
        return EngineDecision.UNREADABLE
    previous = outcome.read_or_quarantine(context.home, log)
    if declined:
        return _declined(context, previous)
    if previous is None:
        return move("started")
    if previous.state == outcome.CANNOT:
        return _after_cannot(context, previous, move)
    return _after(context, previous, move)


def _declined(context: HostContext, previous: outcome.Outcome | None) -> EngineDecision:
    if previous is None or previous.state != outcome.DECLINED:
        outcome.write(
            context.home,
            state=outcome.DECLINED,
            release=context.release,
            attempts=0 if previous is None else previous.attempts,
        )
    context.log.write(
        'engine: this machine declined the Linux engine '
        f'([{CONSENT_TABLE}] {WSL_KEY} = "{WSL_NEVER}"); it stays native'
    )
    return EngineDecision.DECLINED


def _after_cannot(
    context: HostContext, previous: outcome.Outcome, move: Callable[[str], EngineDecision]
) -> EngineDecision:
    log = context.log.write
    rule = next((rule for rule in RESUME_RULES if previous.code in rule.codes), None)
    if rule is None:
        log(
            f"engine: this machine cannot run the Linux engine "
            f"({previous.code}), recorded {previous.at}. Nothing is retried "
            "on its own; the apps offer Try again (2.5)"
        )
        return EngineDecision.CANNOT
    try:
        live = wslstate.probe_live(context.runner)
    except Exception as exc:
        log(f"engine: the {rule.rechecked} re-check crashed: {type(exc).__name__}: {exc}")
        return EngineDecision.CANNOT
    log(f"engine: {previous.code} was recorded {previous.at}; checked again at this start: {live.line()}")
    return move(rule.why) if rule.resumes(live) else EngineDecision.CANNOT


def _after(
    context: HostContext, previous: outcome.Outcome, move: Callable[[str], EngineDecision]
) -> EngineDecision:
    if previous.state == outcome.FAILED and previous.attempts >= outcome.FAILED_ATTEMPT_CEILING:
        context.log.write(
            f"engine: the move has failed {previous.attempts} times in a row "
            f"({previous.code}); it stays failed until somebody presses Try "
            "again (2.2)"
        )
        return EngineDecision.FAILED
    if previous.state == outcome.DONE:
        context.log.write(
            f"engine: the last move finished at {previous.at} and this "
            "machine's engine is not the guest's now; walking the sequence "
            "again"
        )
    return move("resumed" if previous.state == outcome.REBOOT_PENDING else "started")


def run_move(context: HostContext, door: OrchestratorDoor | None, why: str) -> EngineDecision:
    log = context.log.write
    if door is None:
        log(
            "engine: NOT moved — this controller has no door yet, and the "
            "move runs under the door's claim so that a POST /install can "
            "be refused and attached rather than queued"
        )
        return EngineDecision.NO_DOOR
    if not door.claim():
        log(
            "engine: a move is already running on this machine; this one is "
            "not a second walk over the same distro"
        )
        return EngineDecision.ALREADY_RUNNING
    log(f"engine: the move is {why}")
    try:
        door.run_recorded()
    except HostError as exc:
        log(f"engine: {exc.code}: {exc.message}")
        return EngineDecision.of(outcome.classify(exc.code))
    except Exception as exc:
        log(f"engine: the move crashed: {type(exc).__name__}: {exc}")
        return EngineDecision.FAILED
    finally:
        door.release()
    return EngineDecision.DONE


class MoveHost(Protocol):
    operation: ContextManager[Any]

    def check_restartable(self) -> None: ...

    def resume_model_cleanup(self, *, raise_errors: bool = False) -> None: ...

    def verify_active_guest(self) -> None: ...

    def stop_windows_for_move(self) -> None: ...

    def finish_wsl_move(self) -> None: ...

    def stopped_windows_catalog(self) -> CatalogPort: ...


def booted_since(previous: outcome.Outcome) -> bool:
    boot = wslstate.booted_at()
    at = previous.at_epoch()
    if boot is None or at is None:
        return True
    return boot > at


@dataclass
class MoveAttempt:
    number: int
    restarts_before: int
    rebooted: bool
    walk: installer.EngineInstall | None = None

    @classmethod
    def after(cls, previous: outcome.Outcome | None) -> "MoveAttempt":
        failed_before = previous is not None and previous.state == outcome.FAILED
        owed = previous is not None and previous.state == outcome.REBOOT_PENDING
        return cls(
            number=previous.attempts + 1 if failed_before else 1,
            restarts_before=previous.restarts if owed else 0,
            rebooted=booted_since(previous) if owed else True,
        )

    @property
    def restarts(self) -> int:
        return self.restarts_before if self.walk is None else self.walk.restarts


class MoveRecorder:
    def __init__(self, context: HostContext, attempt: MoveAttempt) -> None:
        self._context = context
        self._attempt = attempt
        self._recorded = False

    def record(self, state: outcome.MoveState, code: str | None, sentence: str | None) -> None:
        if self._recorded:
            return
        self._recorded = True
        context = self._context
        written = outcome.write(
            context.home,
            state=state,
            code=code,
            sentence=sentence,
            release=context.release,
            attempts=self._attempt.number,
            restarts=self._attempt.restarts,
        )
        context.log.write(
            f"outcome: {outcome.path(context.home)} says {written.state}"
            + (f" ({written.code})" if written.code else "")
            + f", written {written.at}"
        )

    def recording(self, emit: Emit) -> Emit:
        def emit_recorded(event: installer.Event) -> None:
            if event.event == "failed":
                code = event.data.get("code")
                message = event.data.get("message")
                code = code if isinstance(code, str) else "task_failed"
                self.record(outcome.classify(code), code, message if isinstance(message, str) else None)
            elif event.event == "done":
                self.record(outcome.DONE, None, None)
            emit(event)

        return emit_recorded


def _complete_the_active_guest(context: HostContext, host: MoveHost, emit: Emit) -> None:
    record = context.home / installer.CLEANUP_RECORD
    if record.exists() and installer.quarantine_bad_cleanup_record(context.home, context.log.write) is None:
        host.resume_model_cleanup(raise_errors=True)
    else:
        host.verify_active_guest()
    context.install_walk(emit).complete()


def _move_catalogs(context: HostContext) -> tuple[CatalogPort | None, CatalogPort | None]:
    token = read_token(context.home)
    if token is None:
        return None, None
    windows = HttpCatalog(engine_url(), token, where="the Windows engine")
    guest = GuestCatalog(
        context.runner, CRUCIBLE_DISTRO, token, ENGINE_PORT, where=f'the "{CRUCIBLE_DISTRO}" engine'
    )
    return windows, guest


def _walk_to_the_guest(context: HostContext, host: MoveHost, emit: Emit, attempt: MoveAttempt) -> None:
    windows, guest = _move_catalogs(context)
    attempt.walk = context.install_walk(
        emit,
        restarts=attempt.restarts_before,
        rebooted=attempt.rebooted,
        log=context.log.write,
        windows_catalog=windows,
        guest_catalog=guest,
        stop_windows_server=host.stop_windows_for_move,
        switch_pairing=host.finish_wsl_move,
        windows_after_switch=host.stopped_windows_catalog,
    )
    attempt.walk.run()


def _run_attempt(context: HostContext, host: MoveHost, emit: Emit) -> None:
    attempt = MoveAttempt.after(outcome.read_or_quarantine(context.home, context.log.write))
    recorder = MoveRecorder(context, attempt)
    try:
        if context.presence.owner is Owner.WSL_UNIT:
            _complete_the_active_guest(context, host, recorder.recording(emit))
        else:
            _walk_to_the_guest(context, host, recorder.recording(emit), attempt)
    except HostError as exc:
        recorder.record(outcome.classify(exc.code), exc.code, exc.message)
        raise
    except Exception as exc:
        recorder.record(outcome.FAILED, "task_failed", f"{type(exc).__name__}: {exc}")
        raise
    recorder.record(outcome.DONE, None, None)


def move_sequence(context: HostContext, host: MoveHost) -> MoveSequence:
    def run_sequence(emit: Emit) -> None:
        with host.operation:
            host.check_restartable()
            _run_attempt(context, host, emit)

    return run_sequence
