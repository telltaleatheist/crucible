from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ... import accelerator
from ...config import Config
from ...engines import EngineError
from ...errors import ApiError, JobError
from ..llm import LeaseOnLoad, _open_lease_for_load
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    KIND_TTS,
    Residency,
    describe_resident,
)
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from .common import (
    voice_load_plan,
    describe_voices,
    known_voice,
    require_loadable,
    require_reference,
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


class ReferenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: str
    transcript: str
    name: str | None = None


class LoadVoiceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_s: float = Field(default=DEFAULT_READY_TIMEOUT_SECONDS, ge=30, le=7200)
    reference: ReferenceInput | None = None
    lease: LeaseOnLoad | None = None


class UnloadVoiceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoadVoiceJobType:
    name = "load-voice"

    def __init__(
        self,
        config: Config,
        backend: Any,
        residency: Residency,
        leases: Any | None = None,
    ) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency
        self._leases = leases

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return describe_voices(self._config, self._residency)

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
        if model is None:
            raise ApiError(400, "model_required", f"{self.name} needs a voice")
        validated = validated_params(LoadVoiceParams, params, self.name)
        self._residency.refuse_if_claimed(f"loading {model!r}")
        manifest, spec, _ = require_loadable(self._config, self._backend, model)
        require_reference(manifest, validated.reference)
        accelerator.guard(
            self._config.backend_kind,
            model_id=model,
            need_bytes=voice_load_plan(
                self._config, self._backend, manifest, spec
            ).need_bytes,
            owned_pids=self._residency.owned_pids(),
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
        )

    def run(self, job: Job, ctx: JobContext) -> None:
        params = LoadVoiceParams.model_validate(job.params)
        model = job.model
        if model is None:
            raise JobError("model_required", f"{self.name} needs a voice")
        self._residency.begin_warming(model)
        try:
            self._load(ctx, model, params, job.client)
        finally:
            self._residency.end_warming()

    def _load(
        self,
        ctx: JobContext,
        model: str,
        params: LoadVoiceParams,
        client: str | None,
    ) -> None:
        try:
            manifest, spec, (python, installed) = require_loadable(
                self._config, self._backend, model
            )
            reference = require_reference(manifest, params.reference)
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None

        ctx.warming(f"checking the accelerator for {model}")
        try:
            plan = voice_load_plan(self._config, self._backend, manifest, spec)
            state = accelerator.guard(
                self._config.backend_kind,
                model_id=model,
                need_bytes=plan.need_bytes,
                owned_pids=self._residency.owned_pids(),
                desktop_allowance_bytes=self._config.desktop_allowance_bytes,
                reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None
        ctx.warming(state.detail)

        ctx.raise_if_cancelled()
        ctx.progress(0.0, f"loading {model}")
        try:
            resident = self._residency.load_voice(
                manifest,
                spec,
                installed.path,
                python,
                reference=reference,
                timeout=params.timeout_s,
                on_progress=ctx.warming,
                serving_width=plan.width,
            )
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        ctx.raise_if_cancelled()
        ctx.progress(1.0, f"{model} is resident")
        extra: dict[str, Any] = {
            "resident": resident.voice_id,
            "fingerprint": resident.fingerprint,
            "reference": resident.reference,
        }
        extra["lease_id"] = None
        if params.lease is not None:
            extra["lease_id"] = _open_lease_for_load(
                self._leases,
                kind=resident.kind,
                subject=resident.voice_id,
                request=params.lease,
                client=client,
            )
        ctx.done_extra(**extra)


class UnloadVoiceJobType:
    name = "unload-voice"

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return describe_voices(self._config, self._residency)

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
        if model is None:
            raise ApiError(400, "model_required", f"{self.name} needs a voice")
        validated_params(UnloadVoiceParams, params, self.name)
        if self._residency.being_cleared(model):
            return
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
        if model is None:
            raise JobError("model_required", f"{self.name} needs a voice")
        if self._residency.await_clearance(model):
            ctx.progress(0.0, f"unloading {model}")
            ctx.progress(1.0, f"{model} is unloaded — the card was cleared of it")
            ctx.done_extra(resident=self._residency.resident_id)
            return
        if not self._residency.is_resident(KIND_TTS, model):
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
