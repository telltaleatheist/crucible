"""The `tts` job type's lifecycle half: `load-voice` and `unload-voice`.

PHASE3-TTS.md section 5, mirroring `jobs/llm/`'s pair exactly. What is here is
putting a voice on the accelerator and taking it off, which is the thing the
exclusive lane serialises and the thing every other door depends on having got
right. The render door (`{"type": "tts"}`, section 6) is `render.py` beside this
file and the helpers both halves share are `common.py`; the streaming door
(section 7) is still unbuilt.

Every refusal happens **before the job is queued**, in `preflight()`, so the
client gets an HTTP error naming the thing rather than a job that fails a minute
later:

    unknown_voice        no manifest with that id (raised by jobs.resolve_model)
    backend_unsupported  the manifest has no block for this host's backend
    env_missing          ~/.crucible/envs/tts-<engine> is not installed
    voice_not_installed  no weights at the manifest's pinned revision
    accelerator_busy     somebody else's process is on the card
    insufficient_memory  free memory is below the manifest's estimate
    voice_not_resident   (unload) that voice is not the one that is loaded

**`model` on the wire is the voice id.** DESIGN.md's word for the thing that
produces the bytes is `model`, and for `tts` that thing is the voice — which for
Higgs is the literal truth rather than a pun: a v3 voice *is* the merged
checkpoint the engine was started on, so a voice change is a full worker restart.
No new vocabulary is invented for it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ... import accelerator
from ...config import Config
from ...engines import EngineError
from ...errors import ApiError, JobError
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    KIND_TTS,
    Residency,
    describe_resident,
)
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from .common import (
    describe_voices,
    known_voice,
    require_loadable,
    validated_params,
    voice_provenance,
    voice_rows,
)
from .render import TtsJobType, TtsParams

__all__ = [
    "LoadVoiceJobType",
    "LoadVoiceParams",
    "TtsJobType",
    "TtsParams",
    "UnloadVoiceJobType",
    "voice_rows",
]


class LoadVoiceParams(BaseModel):
    """`params` for a load-voice job. Unknown keys are refused, not ignored."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: float = Field(default=DEFAULT_READY_TIMEOUT_SECONDS, ge=30, le=7200)


class UnloadVoiceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------- load job


class LoadVoiceJobType:
    """`POST /v1/jobs {"type": "load-voice", "model": "<voice id>"}`."""

    name = "load-voice"

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return describe_voices(self._config.backend_kind, self._residency)

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return voice_provenance(self._config.backend_kind, model)

    def vram_estimate(self, model: str | None) -> int:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a voice")
        manifest = known_voice(model)
        if not manifest.supports(self._config.backend_kind):
            return 0
        return manifest.spec(self._config.backend_kind).memory_bytes_estimate

    def check(self, backend: Any) -> JobTypeStatus:
        rows = voice_rows(self._config, backend, self._residency)
        ready = [row["id"] for row in rows if row["loadable"]]
        if not ready:
            blocked = sorted(
                {
                    row["reason"]
                    for row in rows
                    if row["reason"] is not None and row["backend_supported"]
                }
            )
            return JobTypeStatus(
                ready=False,
                detail=(
                    "no voice is loadable here: "
                    + ("; ".join(blocked) if blocked else "this build ships none")
                ),
            )
        return JobTypeStatus(ready=True, detail=f"loadable: {ready}")

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a voice")
        validated_params(LoadVoiceParams, params, self.name)
        # A streaming session holds the resident engine, and loading over it
        # would SIGTERM narrator mid-sentence (PHASE3-TTS.md section 7).
        self._residency.refuse_if_claimed(f"loading {model!r}")
        _, spec, _ = require_loadable(self._config, self._backend, model)
        accelerator.guard(
            self._config.backend_kind,
            model_id=model,
            need_bytes=spec.memory_bytes_estimate,
            owned_pids=self._residency.owned_pids(),
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
        )

    def run(self, job: Job, ctx: JobContext) -> None:
        params = LoadVoiceParams.model_validate(job.params)
        model = job.model
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a voice")
        # `/v1/health` says `warming` for the whole job, not only while the
        # engine's readiness is being polled: from the client's side this server
        # is warming a voice from the moment the lane picks the job up.
        self._residency.begin_warming(model)
        try:
            self._load(ctx, model, params)
        finally:
            self._residency.end_warming()

    def _load(self, ctx: JobContext, model: str, params: LoadVoiceParams) -> None:
        try:
            manifest, spec, (python, installed) = require_loadable(
                self._config, self._backend, model
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None

        ctx.warming(f"checking the accelerator for {model}")
        try:
            # The card can change between the queue and the lane, so the guard
            # runs again here, against the same rules.
            state = accelerator.guard(
                self._config.backend_kind,
                model_id=model,
                need_bytes=spec.memory_bytes_estimate,
                owned_pids=self._residency.owned_pids(),
                desktop_allowance_bytes=self._config.desktop_allowance_bytes,
                reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None
        ctx.warming(state.detail)

        ctx.progress(0.0, f"loading {model}")
        try:
            resident = self._residency.load_voice(
                manifest,
                spec,
                installed.path,
                python,
                timeout=params.timeout_s,
                on_progress=ctx.warming,
            )
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        ctx.progress(1.0, f"{model} is resident")
        ctx.done_extra(resident=resident.voice_id, fingerprint=resident.fingerprint)


# --------------------------------------------------------------- unload job


class UnloadVoiceJobType:
    """`POST /v1/jobs {"type": "unload-voice", "model": "<voice id>"}`."""

    name = "unload-voice"

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return describe_voices(self._config.backend_kind, self._residency)

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return voice_provenance(self._config.backend_kind, model)

    def vram_estimate(self, model: str | None) -> int:
        return 0

    def check(self, backend: Any) -> JobTypeStatus:
        voice = self._residency.resident_voice
        return JobTypeStatus(
            ready=True,
            detail=(
                f"resident: {voice.voice_id}" if voice else "no voice is resident"
            ),
        )

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a voice")
        validated_params(UnloadVoiceParams, params, self.name)
        self._residency.refuse_if_claimed(f"unloading {model!r}")
        if not self._residency.is_resident(KIND_TTS, model):
            raise ApiError(
                409,
                "voice_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_TTS, "no voice is"),
                {"requested": model, "resident": self._residency.resident_id},
            )

    def run(self, job: Job, ctx: JobContext) -> None:
        UnloadVoiceParams.model_validate(job.params)
        model = job.model
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a voice")
        if not self._residency.is_resident(KIND_TTS, model):
            # Checked before `unload()` rather than caught from it: the holder
            # unloads by id alone, and a model sharing a voice's id would be
            # taken off the card by `unload-voice`.
            raise JobError(
                "voice_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_TTS, "no voice is"),
            )
        ctx.progress(0.0, f"unloading {model}")
        try:
            self._residency.unload(model)
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        ctx.progress(1.0, f"{model} is unloaded")
        ctx.done_extra(resident=self._residency.resident_id)
