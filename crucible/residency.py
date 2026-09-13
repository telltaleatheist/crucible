"""What is on the accelerator right now, whatever kind of thing it is.

PHASE2-LLM.md section 3 said **one resident model at a time**. PHASE3-TTS.md
section 5 generalises it, and the reason is arithmetic rather than architecture:
the accelerator does not care what kind of thing is on it, and a card holding a
Higgs checkpoint has no room for a 9B. So this holds **at most one resident
engine, of either kind**, and loading a voice unloads a model exactly as loading
a model unloads a voice.

This file used to be `crucible/jobs/llm/residency.py`. It moved out from under
`jobs/llm/` because it is no longer the llm's: `jobs/tts/` mutates it too, and a
`tts` job reaching into another job type's package for the thing that owns the
card would make the one-at-a-time rule look like a courtesy between two modules
rather than a property of the server.

Only the exclusive job lane mutates this (the `load-model` / `unload-model` and
`load-voice` / `unload-voice` jobs), so the proxy, `/v1/health` and `/v1/voices`
read a value that is never half-written: an engine is published as resident only
once it has proved it is up, and it is unpublished before it is signalled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .engines import (
    EngineError,
    SubprocessEngine,
    build_engine,
    build_voice_engine,
    engine_log_path,
    engine_model_name,
    find_free_port,
)
from .manifests import BackendSpec, ModelManifest
from .voices import VoiceBackendSpec, VoiceManifest

#: The two kinds of thing that can hold the card, and what `/v1/health` reports
#: as `resident_kind` so a client can tell which door to knock on.
KIND_LLM = "llm"
KIND_TTS = "tts"

#: How long a load waits for the engine to prove it is up. vLLM on a 19 GB model
#: spends most of it reading weights and capturing CUDA graphs; narrator on
#: `cuda-linux` spends it starting SGLang-Omni, measured at about 110 s.
DEFAULT_READY_TIMEOUT_SECONDS = 900.0


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
    context_default: int
    memory_bytes_estimate: int
    log_path: Path
    loaded_at: str

    @property
    def id(self) -> str:
        return self.model_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "backend": self.backend,
            "engine": self.engine,
            "engine_model_name": self.engine_model_name,
            "base_url": self.base_url,
            "revision": self.revision,
            "context_default": self.context_default,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "log_path": str(self.log_path),
            "loaded_at": self.loaded_at,
        }


@dataclass(frozen=True)
class ResidentVoice:
    """A voice narrator is currently serving.

    No `base_url`: narrator's wire is newline-delimited JSON over stdin and
    stdout, not HTTP, so there is nothing for a client to be proxied to and the
    field would be a lie if it were here to make the two shapes symmetrical.
    `fingerprint` is what a render records as the voice it used, and it is bound
    to the revision rather than to the id, because two merges of one run are two
    sets of weights under one name.
    """

    kind = KIND_TTS

    voice_id: str
    backend: str
    narrator_engine: str
    revision: str
    fingerprint: str
    sample_rate: int
    max_chars: int
    memory_bytes_estimate: int
    log_path: Path
    loaded_at: str

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
        }


Resident = ResidentModel | ResidentVoice


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def describe_resident(residency: "Residency", kind: str, absent: str) -> str:
    """What holds the card, as the tail of a `*_not_resident` refusal.

    One wording for both doors. `absent` is what to say when nothing is resident,
    in the vocabulary of the door the reader came through — "no model is" for
    `unload-model`, "no voice is" for `unload-voice` — and when something of the
    OTHER kind is resident the message says so by name, because "no model is
    resident" while narrator holds the whole card is true and useless.
    """
    resident = residency.resident
    if resident is None:
        return absent
    if resident.kind == kind:
        return f"{resident.id!r} is"
    other = "voice" if resident.kind == KIND_TTS else "model"
    return f"the resident {other} is {resident.id!r}"


class Residency:
    """The one-resident-engine holder for a server instance."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._resident: Resident | None = None
        self._engine: SubprocessEngine | None = None
        self._warming: str | None = None

    # -------------------------------------------------------------- reading

    @property
    def resident(self) -> Resident | None:
        return self._resident

    @property
    def resident_id(self) -> str | None:
        return None if self._resident is None else self._resident.id

    @property
    def resident_kind(self) -> str | None:
        """`"llm"`, `"tts"`, or None — `/v1/health`'s `resident_kind`."""
        return None if self._resident is None else self._resident.kind

    @property
    def resident_model(self) -> ResidentModel | None:
        """The resident, if it is a model. None when a voice holds the card.

        The OpenAI proxy asks for this rather than for `resident`: with a voice
        resident there is no `base_url` to forward a chat request to, and the
        proxy's `model_not_resident` is the honest answer.
        """
        return self._resident if isinstance(self._resident, ResidentModel) else None

    @property
    def resident_voice(self) -> ResidentVoice | None:
        """The resident, if it is a voice. None when a model holds the card."""
        return self._resident if isinstance(self._resident, ResidentVoice) else None

    @property
    def warming(self) -> str | None:
        """The id a load job is currently warming, or None."""
        return self._warming

    def ids(self) -> list[str]:
        return [] if self._resident is None else [self._resident.id]

    def is_resident(self, kind: str, subject_id: str) -> bool:
        """Is exactly this thing on the card?

        Kind **and** id. Model ids and voice ids are separate namespaces, and
        nothing stops a voice being called `qwen3.5-9b`; asking on the id alone
        would let a resident voice light up a model's `/v1/models` row.
        """
        return (
            self._resident is not None
            and self._resident.kind == kind
            and self._resident.id == subject_id
        )

    def begin_warming(self, subject_id: str) -> None:
        """Mark a load as in progress, so `/v1/health` says `warming`.

        Set for the whole load job, not just the engine's readiness poll: from
        the client's side, "this server is warming something" is true from the
        moment the lane picks the job up.
        """
        self._warming = subject_id

    def end_warming(self) -> None:
        self._warming = None

    def owned_pids(self) -> frozenset[int]:
        return frozenset() if self._engine is None else self._engine.pids

    def reclaimable_bytes(self, excluding: str | None = None) -> int:
        """What unloading the current resident would give back.

        Zero when the resident *is* `excluding` — reloading a model does not free
        its own memory before it needs it again. Across kinds it is never zero:
        a voice's bytes are as reclaimable as a model's, which is the whole point
        of one holder for both.
        """
        if self._resident is None or self._resident.id == excluding:
            return 0
        return self._resident.memory_bytes_estimate

    # -------------------------------------------------------------- writing

    def _evict(self, say: Callable[[str], None], incoming: str) -> None:
        """Unload whatever is there, naming what it was, before `incoming` loads."""
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
        timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
        on_progress: Callable[[str], None] | None = None,
    ) -> ResidentModel:
        """Make this model the resident one, unloading whatever was there."""

        def say(message: str) -> None:
            if on_progress is not None:
                on_progress(message)

        self._evict(say, manifest.id)

        log_path = engine_log_path(self._config.home, manifest.id)
        engine = build_engine(spec.engine, python, log_path)
        served = engine_model_name(spec.engine, weights_dir, manifest.id)
        port = find_free_port()

        context = manifest.context_for(spec.backend)
        self.begin_warming(manifest.id)
        say(
            f"starting {spec.engine} for {manifest.id} on 127.0.0.1:{port} "
            f"(context {context}); log {log_path}"
        )
        try:
            self._start(
                engine,
                weights_dir,
                served,
                port,
                self._engine_args(manifest, spec),
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
            context_default=context,
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=_now(),
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
        timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
        on_progress: Callable[[str], None] | None = None,
    ) -> ResidentVoice:
        """Make this voice the resident one, unloading whatever was there.

        A Higgs v3 voice change IS a full worker restart — the voice is the merged
        checkpoint the engine was started on — so there is no cheaper path here
        than the one a model takes, and none is pretended at.
        """

        def say(message: str) -> None:
            if on_progress is not None:
                on_progress(message)

        self._evict(say, manifest.id)

        log_path = engine_log_path(self._config.home, manifest.id)
        engine = build_voice_engine(manifest.narrator_engine, python, log_path)
        # narrator answers no HTTP route, so this port is not a proxy target; it
        # is found and passed for the same reason every other engine's is, so
        # that an engine which does decide to bind something has a free one.
        port = find_free_port()

        self.begin_warming(manifest.id)
        say(
            f"starting narrator ({manifest.narrator_engine}) for {manifest.id} "
            f"on {spec.backend}; log {log_path}"
        )
        try:
            self._start(engine, weights_dir, manifest.id, port, [], say, timeout)
        finally:
            self.end_warming()

        self._engine = engine
        self._resident = ResidentVoice(
            voice_id=manifest.id,
            backend=spec.backend,
            narrator_engine=manifest.narrator_engine,
            revision=spec.revision,
            fingerprint=manifest.fingerprint(spec.backend),
            sample_rate=manifest.sample_rate,
            max_chars=spec.max_chars,
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=_now(),
        )
        say(f"{manifest.id} is resident")
        return self._resident

    @staticmethod
    def _start(
        engine: SubprocessEngine,
        weights_dir: Path,
        served: str,
        port: int,
        args: list[str],
        say: Callable[[str], None],
        timeout: float,
    ) -> None:
        """Spawn and wait, tidying up a half-started engine without hiding why."""
        try:
            engine.start(weights_dir, served, port, args)
            engine.ready(timeout, on_progress=say)
        except EngineError as start_failure:
            # Tidy up the half-started engine, but report the *start* failure —
            # that is the one that explains the load. A stop failure on top of it
            # is appended, never substituted.
            try:
                engine.stop()
            except EngineError as stop_failure:
                raise EngineError(
                    f"{start_failure}\n...and stopping it also failed: {stop_failure}"
                ) from start_failure
            raise

    @staticmethod
    def _engine_args(manifest: ModelManifest, spec: BackendSpec) -> list[str]:
        """The manifest's args plus what Crucible always sets.

        `--max-model-len` only goes to vLLM; mlx-lm takes the context from the
        model's own config and has no such flag (see engines/mlx_lm.py).
        """
        args = list(spec.engine_args)
        if spec.engine == "vllm":
            args += ["--max-model-len", str(manifest.context_for(spec.backend))]
        return args

    def unload(self, subject_id: str) -> Resident:
        """Stop the engine serving `subject_id`. Raises KeyError if not resident."""
        resident = self._resident
        if resident is None or resident.id != subject_id:
            raise KeyError(subject_id)
        engine = self._engine
        # Unpublish first: from here on nothing new is proxied to a dying engine.
        self._resident = None
        self._engine = None
        if engine is not None:
            engine.stop()
        return resident

    def shutdown(self) -> None:
        """Stop whatever is resident. Called when the server exits."""
        if self._resident is not None:
            self.unload(self._resident.id)
