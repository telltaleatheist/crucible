from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

from .accelerator import probe_unified_memory
from .alignmodels import AlignBackendSpec, AlignManifest
from .denoisemodels import DenoiseBackendSpec, DenoiseManifest
from .backend import MLX_DARWIN
from .config import Config
from .engines import (
    STOP_TIMEOUT_SECONDS,
    EngineError,
    NarratorEngine,
    SubprocessEngine,
    build_engine,
    build_voice_engine,
    engine_log_path,
    engine_model_name,
    find_free_port,
)
from .engines.vllm import DECIDE_ARGS as VLLM_DECIDE_ARGS
from .errors import ApiError, JobError
from .jobenv import tts_env
from .manifests import (
    NO_DEFAULTS,
    BackendSpec,
    ModelDefaults,
    ModelManifest,
    fingerprint,
)
from .narratorvoices import DOCUMENT_READERS, write_document
from .voicereference import VoiceReference
from .voices import VoiceBackendSpec, VoiceManifest
from .vram import KvPlan
from .workers import WorkerError, WorkerSession
from .workers import torch_allocator_environment, torch_memory_cap

KIND_LLM = "llm"
KIND_TTS = "tts"
KIND_ALIGN = "align"
KIND_DENOISE = "denoise"

DEFAULT_READY_TIMEOUT_SECONDS = 900.0

CLEARANCE_TIMEOUT_SECONDS = STOP_TIMEOUT_SECONDS + 30.0


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.subject_id,
            "since": self.since,
            "pids": sorted(self.pids),
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


KIND_NOUNS: dict[str, str] = {
    KIND_LLM: "model",
    KIND_TTS: "voice",
    KIND_ALIGN: "aligner",
    KIND_DENOISE: "separator",
}


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

    def await_clearance(
        self, subject_id: str, *, timeout: float = CLEARANCE_TIMEOUT_SECONDS
    ) -> bool:
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
        timeout = CLEARANCE_TIMEOUT_SECONDS
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
        raise JobError(
            "engine_still_stopping",
            f"cannot {what}: {dying.subject_id} (the {KIND_NOUNS[dying.kind]} "
            f"that was resident) was asked to stop at {dying.since} and has not "
            f"confirmed it. Its pid(s) {sorted(dying.pids)} still hold the card, "
            "and Crucible does not SIGKILL a process holding CUDA — that wedges "
            "WSL2 until Windows reboots. Loading now would put a second engine "
            "on a card that is already full. Stop that process by hand, then "
            "restart Crucible",
        )


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

    def load(
        self,
        manifest: ModelManifest,
        spec: BackendSpec,
        weights_dir: Path,
        python: Path,
        *,
        plan: "KvPlan | None",
        context: int,
        timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
        on_progress: Callable[[str], None] | None = None,
        card_args: tuple[str, ...] = (),
    ) -> ResidentModel:
        self._refuse_mutation_if_claimed(f"load {manifest.id}")
        self.refuse_if_stopping(f"load {manifest.id}")

        def say(message: str) -> None:
            if on_progress is not None:
                on_progress(message)

        self._evict(say, manifest.id)

        log_path = engine_log_path(self._config.home, manifest.id)
        engine = build_engine(spec.engine, python, log_path)
        served = engine_model_name(spec.engine, weights_dir, manifest.id)
        port = find_free_port()

        self.begin_warming(manifest.id)
        say(
            f"starting {spec.engine} for {manifest.id} on 127.0.0.1:{port} "
            f"(context {context}); log {log_path}"
        )
        args = self._engine_args(
            manifest, spec, weights_dir, plan, context=context, card_args=card_args
        )
        try:
            self._start(
                engine,
                weights_dir,
                served,
                port,
                args,
                say,
                timeout,
            )
        finally:
            self.end_warming()

        self._engine = engine
        self._resident = ResidentModel(
            model_id=manifest.id,
            backend=spec.backend,
            engine=spec.engine,
            engine_model_name=served,
            base_url=engine.base_url,
            port=port,
            revision=spec.revision,
            max_model_len=context,
            defaults=manifest.defaults,
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=_now(),
            engine_args=tuple(args),
        )
        say(f"{manifest.id} is resident at {engine.base_url}")
        return self._resident

    def load_voice(
        self,
        manifest: VoiceManifest,
        spec: VoiceBackendSpec,
        weights_dir: Path,
        python: Path,
        *,
        reference: VoiceReference | None = None,
        timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
        on_progress: Callable[[str], None] | None = None,
        serving_width: int | None = None,
    ) -> ResidentVoice:
        self._refuse_mutation_if_claimed(f"load {manifest.id}")
        self.refuse_if_stopping(f"load {manifest.id}")

        def say(message: str) -> None:
            if on_progress is not None:
                on_progress(message)

        self._evict(say, manifest.id)

        log_path = engine_log_path(self._config.home, manifest.id)
        env_spec = tts_env(manifest.narrator_engine, spec.backend)
        voices = (
            write_document(
                self._config.home, manifest, spec, weights_dir, reference
            )
            if manifest.narrator_engine in DOCUMENT_READERS
            else None
        )
        engine = build_voice_engine(
            manifest.narrator_engine,
            python,
            log_path,
            serving_stack=env_spec.serving_stack,
            max_num_seqs=(
                None if manifest.serving is None
                else (
                    manifest.serving.max_num_seqs
                    if serving_width is None
                    else min(serving_width, manifest.serving.max_num_seqs)
                )
            ),
            mem_fraction=(
                None if manifest.serving is None
                else manifest.serving.mem_fraction
            ),
            context_length=(
                None if manifest.serving is None
                else manifest.serving.context_length
            ),
            voices=voices,
            mlx_total_bytes=(
                probe_unified_memory()[1] if spec.backend == MLX_DARWIN else None
            ),
        )
        port = find_free_port()

        self.begin_warming(manifest.id)
        say(
            f"starting narrator ({manifest.narrator_engine}) for {manifest.id} "
            f"on {spec.backend}; log {log_path}"
        )
        try:
            self._start(
                engine,
                weights_dir,
                manifest.id,
                port,
                [],
                say,
                timeout,
                confirm=lambda: self._load_the_voice(
                    engine, manifest, weights_dir, say
                ),
            )
        finally:
            self.end_warming()

        self._engine = engine
        self._resident = ResidentVoice(
            voice_id=manifest.id,
            backend=spec.backend,
            narrator_engine=manifest.narrator_engine,
            revision=spec.weights_identity,
            fingerprint=manifest.fingerprint(spec.backend),
            sample_rate=manifest.sample_rate,
            max_chars=spec.max_chars,
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=_now(),
            reference=None if reference is None else reference.to_report(),
        )
        say(f"{manifest.id} is resident")
        return self._resident

    def load_aligner(
        self,
        manifest: AlignManifest,
        spec: AlignBackendSpec,
        weights_dir: Path,
        python: Path,
        *,
        max_audio_s: float,
        timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
        on_progress: Callable[[str], None] | None = None,
    ) -> ResidentAligner:
        from .jobs.align import device_for, start_aligner_session

        self._refuse_mutation_if_claimed(f"load {manifest.id}")
        self.refuse_if_stopping(f"load {manifest.id}")

        def say(message: str) -> None:
            if on_progress is not None:
                on_progress(message)

        self._evict(say, manifest.id)

        log_path = engine_log_path(self._config.home, manifest.id)
        device = device_for(spec.backend)
        dtype = spec.dtype
        self.begin_warming(manifest.id)
        say(
            f"loading {manifest.id} ({spec.engine}) on {device} at {dtype}; "
            f"log {log_path}"
        )
        try:
            session = start_aligner_session(
                python,
                weights_dir,
                spec,
                log_path,
                ready_silence_timeout=timeout,
                on_ready=lambda message: say(
                    f"{manifest.id} loaded in {message['seconds']:.1f}s on "
                    f"{message['device']} at {message['dtype']}"
                ),
                on_progress=lambda message: say(str(message["message"])),
            )
        finally:
            self.end_warming()

        self._session = session
        self._resident = ResidentAligner(
            aligner_id=manifest.id,
            backend=spec.backend,
            revision=spec.revision,
            fingerprint=fingerprint(manifest.id, spec.revision),
            device=device,
            dtype=dtype,
            max_audio_s=max_audio_s,
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=_now(),
        )
        say(f"{manifest.id} is resident")
        return self._resident

    def load_separator(
        self,
        manifest: DenoiseManifest,
        spec: DenoiseBackendSpec,
        model_file_dir: Path,
        python: Path,
        script: Path,
        *,
        use_autocast: bool,
        environment: dict[str, str],
        timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
        on_progress: Callable[[str], None] | None = None,
    ) -> ResidentSeparator:
        self._refuse_mutation_if_claimed(f"load {manifest.id}")
        self.refuse_if_stopping(f"load {manifest.id}")

        def say(message: str) -> None:
            if on_progress is not None:
                on_progress(message)

        self._evict(say, manifest.id)

        log_path = engine_log_path(self._config.home, manifest.id)
        session = WorkerSession(
            python=python,
            script=script,
            log_path=log_path,
            environment={**environment, **torch_allocator_environment(spec.backend)},
        )

        self.begin_warming(manifest.id)
        say(
            f"loading {manifest.id} ({manifest.model_filename}) with "
            f"use_autocast={use_autocast}; log {log_path}"
        )
        try:
            outcome = session.start(
                {
                    "op": "load",
                    "model_file_dir": str(model_file_dir),
                    "model_filename": manifest.model_filename,
                    "use_autocast": use_autocast,
                    "memory_cap_bytes": torch_memory_cap(
                        spec.backend, spec.memory_bytes_estimate
                    ),
                },
                ready_silence_timeout=timeout,
                on_ready=lambda message: say(
                    f"{manifest.id} loaded in {message['seconds']:.1f}s"
                ),
                on_progress=lambda message: say(str(message["message"])),
            )
        finally:
            self.end_warming()
        if outcome.results:
            session.stop()
            raise WorkerError(
                f"{script.name} answered a load request with "
                f"{len(outcome.results)} result(s); a load produces none"
            )

        self._session = session
        self._resident = ResidentSeparator(
            separator_id=manifest.id,
            backend=spec.backend,
            model_filename=manifest.model_filename,
            revision=spec.revision,
            fingerprint=fingerprint(manifest.id, spec.revision),
            sample_rate=manifest.sample_rate,
            use_autocast=use_autocast,
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=_now(),
        )
        say(f"{manifest.id} is resident")
        return self._resident

    @staticmethod
    def _load_the_voice(
        engine: NarratorEngine,
        manifest: VoiceManifest,
        weights_dir: Path,
        say: Callable[[str], None],
    ) -> dict[str, Any]:
        say(f"loading {manifest.id} into narrator from {weights_dir}")
        loaded = engine.load(
            voice=manifest.id, weights_dir=weights_dir, warm=True, on_progress=say
        )
        reported = loaded.get("sampleRate")
        if not isinstance(reported, int) or isinstance(reported, bool):
            raise EngineError(
                f"{engine.name} loaded {manifest.id} and reported sampleRate "
                f"{reported!r}, which is not a sample rate. Every duration and "
                "every byte count downstream is derived from it"
            )
        if reported != manifest.sample_rate:
            raise EngineError(
                f"{engine.name} renders {manifest.id} at {reported} Hz, but "
                f"{manifest.path.name} declares {manifest.sample_rate}. Crucible "
                "refuses rather than resampling: a FLAC written at the manifest's "
                "rate from bytes generated at the engine's is a chunk of the "
                "wrong length, and nothing in the file would say so. Fix the "
                "manifest, or find out why the engine changed"
            )
        say(
            f"narrator loaded {manifest.id}: engine {loaded.get('engine')!r}, "
            f"backend {loaded.get('backend')!r}, {reported} Hz"
        )
        return loaded

    @staticmethod
    def _start(
        engine: SubprocessEngine,
        weights_dir: Path,
        served: str,
        port: int,
        args: list[str],
        say: Callable[[str], None],
        timeout: float,
        confirm: Callable[[], Any] | None = None,
    ) -> None:
        try:
            engine.start(weights_dir, served, port, args)
            engine.ready(timeout, on_progress=say)
            if confirm is not None:
                confirm()
        except BaseException as start_failure:
            try:
                engine.stop()
            except EngineError as stop_failure:
                raise EngineError(
                    f"{start_failure}\n...and stopping it also failed: {stop_failure}"
                ) from start_failure
            raise

    @staticmethod
    def _engine_args(
        manifest: ModelManifest,
        spec: BackendSpec,
        weights_dir: Path,
        plan: "KvPlan | None",
        *,
        context: int,
        card_args: tuple[str, ...] = (),
    ) -> list[str]:
        args = list(spec.engine_args)
        if spec.engine == "vllm":
            args += ["--max-model-len", str(context)]
            args += list(VLLM_DECIDE_ARGS)
            args += list(card_args)
        if spec.engine == "llama-server":
            if spec.file is None:
                raise EngineError(
                    f"{manifest.path.name}'s {spec.backend} block names no "
                    "`file`, and llama-server serves one GGUF. A block for "
                    "this backend without a file is a block for nothing"
                )
            args = ["-m", str(weights_dir / spec.file)] + args
            if spec.mmproj is not None:
                args += ["--mmproj", str(weights_dir / spec.mmproj)]
            args += ["-c", str(context)]
        if plan is not None:
            args += plan.flags()
        return args

    def unload(self, subject_id: str) -> Resident:
        self._refuse_mutation_if_claimed(f"unload {subject_id}")
        resident = self._resident
        if resident is None or resident.id != subject_id:
            raise KeyError(subject_id)
        engine, session = self._engine, self._session
        self._resident = None
        self._engine = None
        self._session = None
        self._dying = DyingResident(
            subject_id=resident.id,
            kind=resident.kind,
            engine=engine,
            session=session,
            pids=(frozenset() if engine is None else engine.pids)
            | (frozenset() if session is None else session.pids),
            since=_now(),
        )
        with self._claim_lock:
            self._cleared = subject_id if self._claim_clears else None
        self._stop_the_dying()
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
        self._stop_the_dying()
        if self._resident is not None:
            self.unload(self._resident.id)
