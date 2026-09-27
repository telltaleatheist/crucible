from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, ClassVar, Iterator

from .accelerator import (
    ProcessIdentity,
    ask_pid_to_stop,
    forget_leftover,
    leftovers,
    note_leftover,
    process_alive,
    process_identity,
)
from .cardkinds import KIND_ALIGN, KIND_DENOISE, KIND_LLM, KIND_NOUNS, KIND_TTS
from .clock import utcnow as _now
from .config import Config
from .engines import (
    STOP_TIMEOUT_SECONDS,
    EngineError,
    NarratorEngine,
    SubprocessEngine,
    build_engine,
    build_voice_engine,
    engine_load_args,
    engine_log_path,
    engine_model_name,
    start_engine,
)
from .errors import ApiError, JobError
from .manifests import NO_DEFAULTS, ModelDefaults, fingerprint
from .workers import WorkerSession

__all__ = [
    "CLEARANCE_MARGIN_SECONDS",
    "CLEARANCE_TIMEOUT_SECONDS",
    "DEFAULT_READY_TIMEOUT_SECONDS",
    "KIND_ALIGN",
    "KIND_DENOISE",
    "KIND_LLM",
    "KIND_NOUNS",
    "KIND_TTS",
    "DyingResident",
    "Occupant",
    "Residency",
    "Resident",
    "ResidentAligner",
    "ResidentModel",
    "ResidentSeparator",
    "ResidentVoice",
    "build_engine",
    "build_voice_engine",
    "describe_resident",
    "engine_model_name",
    "resident_record_path",
    "say_to",
    "stop_budget_of",
]

DEFAULT_READY_TIMEOUT_SECONDS = 900.0

CLEARANCE_MARGIN_SECONDS = 30.0

CLEARANCE_TIMEOUT_SECONDS = STOP_TIMEOUT_SECONDS + CLEARANCE_MARGIN_SECONDS

RESIDENT_RECORD_NAME = "resident.json"

LEFTOVER_POLL_SECONDS = 1.0


def resident_record_path(home: Path) -> Path:
    return Path(home) / "run" / RESIDENT_RECORD_NAME


def stop_budget_of(engine: object) -> float:
    return float(getattr(engine, "stop_budget_seconds", STOP_TIMEOUT_SECONDS))


@dataclass(frozen=True)
class ResidentModel:
    kind = KIND_LLM

    model_id: str
    backend: str
    engine: str
    engine_model_name: str
    base_url: str
    port: int
    revision: str
    max_model_len: int
    memory_bytes_estimate: int
    log_path: Path
    loaded_at: str
    engine_args: tuple[str, ...]
    defaults: ModelDefaults = NO_DEFAULTS

    @property
    def id(self) -> str:
        return self.model_id

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_id, self.revision)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "backend": self.backend,
            "engine": self.engine,
            "engine_model_name": self.engine_model_name,
            "base_url": self.base_url,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
            "max_model_len": self.max_model_len,
            "defaults": self.defaults.to_dict(),
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "log_path": str(self.log_path),
            "loaded_at": self.loaded_at,
        }


@dataclass(frozen=True)
class ResidentVoice:
    kind = KIND_TTS

    voice_id: str
    backend: str
    narrator_engine: str
    revision: str
    fingerprint: str
    sample_rate: int
    max_chars: int | None
    memory_bytes_estimate: int
    log_path: Path
    loaded_at: str
    reference: dict[str, Any] | None = None

    @property
    def id(self) -> str:
        return self.voice_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "voice": self.voice_id,
            "backend": self.backend,
            "narrator_engine": self.narrator_engine,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
            "sample_rate": self.sample_rate,
            "max_chars": self.max_chars,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "log_path": str(self.log_path),
            "loaded_at": self.loaded_at,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class ResidentAligner:
    kind = KIND_ALIGN

    aligner_id: str
    backend: str
    revision: str
    fingerprint: str
    device: str
    dtype: str
    max_audio_s: float
    memory_bytes_estimate: int
    log_path: Path
    loaded_at: str

    @property
    def id(self) -> str:
        return self.aligner_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "aligner": self.aligner_id,
            "backend": self.backend,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
            "device": self.device,
            "dtype": self.dtype,
            "max_audio_s": self.max_audio_s,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "log_path": str(self.log_path),
            "loaded_at": self.loaded_at,
        }


@dataclass(frozen=True)
class ResidentSeparator:
    kind = KIND_DENOISE

    separator_id: str
    backend: str
    model_filename: str
    revision: str
    fingerprint: str
    sample_rate: int
    use_autocast: bool
    memory_bytes_estimate: int
    log_path: Path
    loaded_at: str

    @property
    def id(self) -> str:
        return self.separator_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "separator": self.separator_id,
            "backend": self.backend,
            "model_filename": self.model_filename,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
            "sample_rate": self.sample_rate,
            "use_autocast": self.use_autocast,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "log_path": str(self.log_path),
            "loaded_at": self.loaded_at,
        }


Resident = ResidentModel | ResidentVoice | ResidentAligner | ResidentSeparator


@dataclass(frozen=True)
class DyingResident:
    subject_id: str
    kind: str
    engine: SubprocessEngine | None
    session: WorkerSession | None
    pids: frozenset[int]
    since: str
    log_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.subject_id,
            "since": self.since,
            "pids": sorted(self.pids),
        }


@dataclass(frozen=True)
class Occupant:
    resident: Resident
    engine: SubprocessEngine | None = None
    session: WorkerSession | None = None
    base_url: str | None = None

    @property
    def pids(self) -> frozenset[int]:
        pids: frozenset[int] = frozenset()
        if self.engine is not None:
            pids |= self.engine.pids
        if self.session is not None:
            pids |= self.session.pids
        return pids

    def stop(self) -> None:
        if self.engine is not None:
            self.engine.stop()
        if self.session is not None:
            self.session.stop()


def say_to(on_progress: Callable[[str], None] | None) -> Callable[[str], None]:
    def say(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    return say


def describe_resident(residency: "Residency", kind: str, absent: str) -> str:
    resident = residency.resident
    if resident is None:
        return absent
    if resident.kind == kind:
        return f"{resident.id!r} is"
    return f"the resident {KIND_NOUNS[resident.kind]} is {resident.id!r}"


class Residency:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._resident: Resident | None = None
        self._engine: SubprocessEngine | None = None
        self._session: WorkerSession | None = None
        self._dying: DyingResident | None = None
        self._warming: str | None = None
        self._claim: str | None = None
        self._claim_thread: int | None = None
        self._claim_clears = False
        self._cleared: str | None = None
        self._claim_lock = threading.Condition()
        self._asking_again = threading.Lock()
        self._record_lock = threading.Lock()


    @property
    def claimed_by(self) -> str | None:
        return self._claim

    @contextmanager
    def claimed(self, holder: str, *, may_mutate: bool) -> Iterator[None]:
        self.claim(holder, may_mutate=may_mutate)
        try:
            yield
        finally:
            self.release(holder)

    def claim(self, holder: str, *, may_mutate: bool, clears: bool = False) -> None:
        self.refuse_if_stopping(f"give the card to {holder!r}")
        with self._claim_lock:
            if self._claim is not None:
                raise JobError(
                    "engine_in_use",
                    f"the resident engine is held by {self._claim!r} and "
                    f"{holder!r} cannot have it at the same time. narrator has "
                    "one stdin and one stdout, so two conversations on it read "
                    "each other's replies",
                )
            self._claim = holder
            self._claim_thread = threading.get_ident() if may_mutate else None
            self._claim_clears = clears

    def release(self, holder: str) -> None:
        with self._claim_lock:
            if self._claim != holder:
                raise JobError(
                    "engine_in_use",
                    f"{holder!r} tried to release the card, which is held by "
                    f"{self._claim!r}",
                )
            self._claim = None
            self._claim_thread = None
            self._claim_clears = False
            self._claim_lock.notify_all()


    def being_cleared(self, subject_id: str) -> bool:
        with self._claim_lock:
            return self._being_cleared(subject_id)

    def _being_cleared(self, subject_id: str) -> bool:
        resident = self._resident
        if resident is None:
            return self._cleared == subject_id
        return self._claim_clears and resident.id == subject_id

    def clearance_timeout(self) -> float:
        dying = self._dying
        engine = self._engine or (None if dying is None else dying.engine)
        budget = getattr(engine, "stop_budget_seconds", None)
        if budget is None:
            return CLEARANCE_TIMEOUT_SECONDS
        return float(budget) + CLEARANCE_MARGIN_SECONDS

    def await_clearance(
        self, subject_id: str, *, timeout: float | None = None
    ) -> bool:
        if timeout is None:
            timeout = self.clearance_timeout()
        with self._claim_lock:
            if not self._being_cleared(subject_id):
                return False

            def wedged(holder: str | None) -> Exception:
                return JobError(
                    "engine_in_use",
                    f"waited {timeout:.0f}s for the card to be cleared of "
                    f"{subject_id!r} and it has not been. It is still held "
                    f"by {holder!r}, which is longer than the engine's "
                    "own SIGTERM deadline: something is wedged, and "
                    "unloading on top of it would make it worse",
                )

            self._wait_out_clearance(timeout, wedged)
            return self._resident is None and self._cleared == subject_id


    def _clearing_elsewhere(self) -> bool:
        return self._claim_clears and self._claim_thread != threading.get_ident()

    def _wait_out_clearance(
        self, timeout: float, wedged: Callable[[str | None], Exception]
    ) -> None:
        deadline = time.monotonic() + timeout
        while self._clearing_elsewhere():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise wedged(self._claim)
            self._claim_lock.wait(remaining)

    def await_settled(self, what: str, *, timeout: float) -> None:
        with self._claim_lock:

            def wedged(holder: str | None) -> Exception:
                return ApiError(
                    409,
                    "engine_in_use",
                    f"{what} waited {timeout:.0f}s for the settlement to finish "
                    f"clearing the card and it has not. The card is still held "
                    f"by {holder!r}, which is longer than the engine's own "
                    "SIGTERM deadline: something is wedged, and starting work on "
                    "top of an engine that will not stop would make it worse",
                    {"held_by": holder},
                )

            self._wait_out_clearance(timeout, wedged)

    @asynccontextmanager
    async def settled_for(
        self,
        what: str,
        *,
        same_intent: Callable[[], bool] | None = None,
    ) -> AsyncIterator[None]:
        timeout = self.clearance_timeout()
        deadline = time.monotonic() + timeout
        while True:
            self._claim_lock.acquire()
            if not self._clearing_elsewhere():
                break
            if same_intent is not None and same_intent():
                break
            self._claim_lock.release()
            await asyncio.to_thread(
                self.await_settled,
                what,
                timeout=max(0.0, deadline - time.monotonic()),
            )
        try:
            yield
        finally:
            self._claim_lock.release()

    def claim_to_clear(self, holder: str, *, held: Callable[[], object]) -> bool:
        with self._claim_lock:
            if held() is not None:
                return False
            if self._resident is None or self._claim is not None:
                return False
            self.claim(holder, may_mutate=True, clears=True)
            return True

    def refuse_if_claimed(self, what: str) -> None:
        holder = self._claim
        if holder is None:
            return
        raise ApiError(
            409,
            "engine_in_use",
            f"{what} needs the card, which is held by {holder!r}. The card has "
            "one holder at a time, and both ways past that are damage: narrator "
            "has one stdin and one stdout, so a second conversation reads the "
            "first one's replies, and a job that loads or unloads would take the "
            "engine off the card mid-sentence",
            {"held_by": holder},
        )

    def _refuse_mutation_if_claimed(self, what: str) -> None:
        holder = self._claim
        if holder is not None and self._claim_thread != threading.get_ident():
            raise JobError(
                "engine_in_use",
                f"cannot {what}: the resident engine is held by {holder!r}. "
                "Taking it off the card now would end that conversation "
                "mid-sentence",
            )

    def refuse_if_stopping(self, what: str) -> None:
        dying = self._dying
        if dying is None:
            return
        if not self._still_running(dying):
            self._let_go_of(dying)
            return
        second_stop: Exception | None = None
        if self._asking_again.acquire(blocking=False):
            try:
                self._stop_the_dying()
            except Exception as exc:
                second_stop = exc
            finally:
                self._asking_again.release()
                self._record_residents()
        dying = self._dying
        if dying is None:
            return
        running = self._still_running(dying)
        if not running:
            self._let_go_of(dying)
            return
        pids = " ".join(str(pid) for pid in running)
        log = "its log" if dying.log_path is None else str(dying.log_path)
        said = "" if second_stop is None else f" The second stop said: {second_stop}"
        raise JobError(
            "engine_still_stopping",
            f"cannot {what}: {dying.subject_id} (the {KIND_NOUNS[dying.kind]} "
            f"that was resident) was asked to stop at {dying.since}, and asked "
            f"again just now, and pid(s) {pids} are still running on the card. "
            "Crucible does not SIGKILL a process holding CUDA — that wedges "
            "WSL2 until Windows reboots — and loading now would put a second "
            f"engine on a card that is already full. Run `kill {pids}` (never "
            "-9) and try again once it has exited; Crucible notices the exit "
            f"by itself, no restart needed. Why it hangs is in {log}.{said}",
        )

    @staticmethod
    def _still_running(dying: DyingResident) -> list[int]:
        return sorted(pid for pid in dying.pids if process_alive(pid))

    def _let_go_of(self, dying: DyingResident) -> None:
        if self._dying is not dying:
            return
        self._dying = None
        print(
            f"crucible: {dying.subject_id}'s pid(s) {sorted(dying.pids)} have "
            f"exited since the stop at {dying.since}; the card is free again",
            file=sys.stderr,
        )
        self._record_residents()


    @property
    def resident(self) -> Resident | None:
        return self._resident

    @property
    def resident_id(self) -> str | None:
        return None if self._resident is None else self._resident.id

    @property
    def resident_kind(self) -> str | None:
        return None if self._resident is None else self._resident.kind

    @property
    def engine_exit_code(self) -> int | None:
        engine = self._engine
        if engine is None or not isinstance(self._resident, ResidentModel):
            return None
        return engine.exit_code

    @property
    def resident_model(self) -> ResidentModel | None:
        return self._resident if isinstance(self._resident, ResidentModel) else None

    @property
    def resident_voice(self) -> ResidentVoice | None:
        return self._resident if isinstance(self._resident, ResidentVoice) else None

    @property
    def voice_engine(self) -> NarratorEngine | None:
        if not isinstance(self._resident, ResidentVoice):
            return None
        engine = self._engine
        if not isinstance(engine, NarratorEngine):
            raise EngineError(
                f"a voice is resident but the engine holding the card is "
                f"{type(engine).__name__}, not a narrator"
            )
        return engine

    @property
    def resident_aligner(self) -> ResidentAligner | None:
        return self._resident if isinstance(self._resident, ResidentAligner) else None

    @property
    def aligner_session(self) -> "WorkerSession | None":
        return None if self.resident_aligner is None else self._session

    @property
    def resident_separator(self) -> ResidentSeparator | None:
        return (
            self._resident if isinstance(self._resident, ResidentSeparator) else None
        )

    @property
    def separator_session(self) -> "WorkerSession | None":
        return None if self.resident_separator is None else self._session

    @property
    def warming(self) -> str | None:
        return self._warming

    def ids(self) -> list[str]:
        return [] if self._resident is None else [self._resident.id]

    def is_resident(self, kind: str, subject_id: str) -> bool:
        return (
            self._resident is not None
            and self._resident.kind == kind
            and self._resident.id == subject_id
        )

    def begin_warming(self, subject_id: str) -> None:
        self._warming = subject_id

    def end_warming(self) -> None:
        self._warming = None

    @property
    def stopping(self) -> DyingResident | None:
        return self._dying

    def owned_pids(self) -> frozenset[int]:
        pids: frozenset[int] = frozenset()
        if self._engine is not None:
            pids |= self._engine.pids
        if self._session is not None:
            pids |= self._session.pids
        if self._dying is not None:
            pids |= self._dying.pids
        return pids

    def reclaimable_bytes(self, excluding: str | None = None) -> int:
        if self._resident is None or self._resident.id == excluding:
            return 0
        return self._resident.memory_bytes_estimate


    def _evict(self, say: Callable[[str], None], incoming: str) -> None:
        if self._resident is None:
            return
        previous = self._resident
        say(
            f"unloading {previous.id} (the resident {previous.kind}) to make room "
            f"for {incoming} — one resident engine at a time, of either kind"
        )
        self.unload(previous.id)

    @property
    def home(self) -> Path:
        return self._config.home

    def log_path_for(self, subject_id: str) -> Path:
        return engine_log_path(self._config.home, subject_id)

    def occupy(
        self,
        kind: str,
        subject_id: str,
        start: Callable[[], Occupant],
        *,
        say: Callable[[str], None],
    ) -> Resident:
        self._refuse_mutation_if_claimed(f"load {subject_id}")
        self.refuse_if_stopping(f"load {subject_id}")
        self._evict(say, subject_id)
        self.begin_warming(subject_id)
        try:
            occupant = start()
        finally:
            self.end_warming()
        resident = occupant.resident
        if resident.kind != kind or resident.id != subject_id:
            occupant.stop()
            raise EngineError(
                f"loading the {kind} {subject_id!r} started the {resident.kind} "
                f"{resident.id!r} instead; it was stopped. That is a bug in the "
                f"job package that built it: report it with {resident.log_path}"
            )
        self._engine = occupant.engine
        self._session = occupant.session
        self._resident = resident
        at = "" if occupant.base_url is None else f" at {occupant.base_url}"
        say(f"{subject_id} is resident{at}")
        self._record_residents()
        return resident

    _start = staticmethod(start_engine)

    _engine_args = staticmethod(engine_load_args)

    load_voice: ClassVar[Callable[..., ResidentVoice]]

    def unload(self, subject_id: str) -> Resident:
        self._refuse_mutation_if_claimed(f"unload {subject_id}")
        resident = self._resident
        if resident is None or resident.id != subject_id:
            raise KeyError(subject_id)
        leaving = Occupant(resident, self._engine, self._session)
        self._resident = None
        self._engine = None
        self._session = None
        self._dying = DyingResident(
            subject_id=resident.id,
            kind=resident.kind,
            engine=leaving.engine,
            session=leaving.session,
            pids=leaving.pids,
            since=_now(),
            log_path=resident.log_path,
        )
        with self._claim_lock:
            self._cleared = subject_id if self._claim_clears else None
        try:
            self._stop_the_dying()
        finally:
            self._record_residents()
        return resident

    def _stop_the_dying(self) -> None:
        dying = self._dying
        if dying is None:
            return
        if dying.engine is not None:
            dying.engine.stop()
        if dying.session is not None:
            dying.session.stop()
        self._dying = None

    def shutdown(self) -> None:
        try:
            self._stop_the_dying()
        finally:
            self._record_residents()
        if self._resident is not None:
            self.unload(self._resident.id)

    @property
    def record_path(self) -> Path:
        return resident_record_path(self._config.home)

    def _stop_budget(self) -> float:
        dying = self._dying
        engine = self._engine or (None if dying is None else dying.engine)
        return stop_budget_of(engine)

    def _record_residents(self) -> None:
        with self._record_lock:
            self._write_record()

    def _write_record(self) -> None:
        path = self.record_path
        identities: dict[int, ProcessIdentity] = {}
        for pid in sorted(self.owned_pids()):
            identity = process_identity(pid)
            if identity is not None:
                identities[pid] = identity
        for left in leftovers():
            identities.setdefault(left.identity.pid, left.identity)
        try:
            if not identities:
                path.unlink(missing_ok=True)
                return
            dying = self._dying
            document = {
                "crucible_pid": os.getpid(),
                "recorded_at": _now(),
                "subject": self.resident_id
                or (None if dying is None else dying.subject_id),
                "stop_budget_seconds": self._stop_budget(),
                "processes": [identity.to_dict() for identity in identities.values()],
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            writing = path.with_name(path.name + ".writing")
            writing.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            os.replace(writing, path)
        except OSError as exc:
            print(
                f"crucible: could not record the resident engine's pids in "
                f"{path}: {type(exc).__name__}: {exc}. If Crucible crashes now, "
                "the next start cannot ask that engine to stop",
                file=sys.stderr,
            )

    def start_reclaiming(self) -> threading.Thread:
        worker = threading.Thread(
            target=self.reclaim_leftovers, name="crucible-reclaim", daemon=True
        )
        worker.start()
        return worker

    def reclaim_leftovers(self) -> list[int]:
        path = self.record_path
        recorded = self._read_record(path)
        if recorded is None:
            return []
        document, processes = recorded
        crucible_pid = document.get("crucible_pid")
        if isinstance(crucible_pid, int) and process_alive(crucible_pid):
            print(
                f"crucible: {path} belongs to Crucible pid {crucible_pid}, which "
                "is still running; its engines are its own and were left alone",
                file=sys.stderr,
            )
            return []
        asked: list[ProcessIdentity] = []
        for wanted in processes:
            found = process_identity(wanted.pid)
            if found is None or not found.same_process_as(wanted):
                continue
            asked_at = _now()
            if ask_pid_to_stop(found.pid):
                print(
                    f"crucible: a previous Crucible left pid {found.pid} "
                    f"({found.command}) running; asked it to stop (SIGTERM) at "
                    f"{asked_at}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"crucible: a previous Crucible left pid {found.pid} "
                    f"({found.command}) running and it could not be signalled; "
                    f"stop it with `kill {found.pid}` (never -9)",
                    file=sys.stderr,
                )
            note_leftover(found, asked_at)
            asked.append(found)
        budget = document.get("stop_budget_seconds")
        deadline = time.monotonic() + (
            float(budget) if isinstance(budget, (int, float)) else STOP_TIMEOUT_SECONDS
        )
        while any(process_alive(each.pid) for each in asked):
            if time.monotonic() >= deadline:
                break
            time.sleep(LEFTOVER_POLL_SECONDS)
        reclaimed = []
        for each in asked:
            if process_alive(each.pid):
                print(
                    f"crucible: pid {each.pid} ({each.command}) from a previous "
                    "Crucible did not exit after SIGTERM and was left running; "
                    f"stop it with `kill {each.pid}` (never -9). Loads are "
                    "refused until it is gone",
                    file=sys.stderr,
                )
                continue
            forget_leftover(each.pid)
            reclaimed.append(each.pid)
        if reclaimed:
            print(
                f"crucible: reclaimed the card from a previous Crucible's "
                f"engine(s): pid(s) {reclaimed} exited on SIGTERM",
                file=sys.stderr,
            )
        self._record_residents()
        return reclaimed

    @staticmethod
    def _read_record(
        path: Path,
    ) -> tuple[dict[str, Any], list[ProcessIdentity]] | None:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            print(
                f"crucible: could not read {path}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return None
        try:
            document = json.loads(text)
            processes = [
                ProcessIdentity(
                    pid=int(entry["pid"]),
                    command=str(entry["command"]),
                    started=str(entry["started"]),
                )
                for entry in document["processes"]
            ]
        except (ValueError, KeyError, TypeError) as exc:
            quarantine = path.with_name(
                f"{path.name}.bad-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
            )
            try:
                os.replace(path, quarantine)
            except OSError:
                quarantine = path
            print(
                f"crucible: {path} was not a resident record ({exc}); moved it "
                f"to {quarantine} and went on. If an engine from a previous "
                "Crucible is still on the card, the load refusal will name it",
                file=sys.stderr,
            )
            return None
        return document, processes
