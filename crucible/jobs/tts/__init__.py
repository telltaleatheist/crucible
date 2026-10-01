from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ...cardkinds import KIND_TTS
from ...config import Config
from ...engines import EngineError
from ...errors import JobError
from ...jobtypes import LOAD_VOICE, TTS_JOB, UNLOAD_VOICE
from ...residency import DEFAULT_READY_TIMEOUT_SECONDS, Residency
from ...voicereference import VoiceReference
from ...voices import VoiceManifest
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from ..binding import JobTypeBinding
from ..leaseonload import LeaseOnLoad, open_lease_for_load
from ..template import as_job_error, card_guard, parse_params, require_model, run_model
from ..unload import UnloadJobType
from .common import (
    describe_voices,
    known_voice,
    occupy_voice,
    require_loadable,
    require_reference,
    voice_load_plan,
    voice_provenance,
    voice_rows,
)
from .render import TtsJobType, TtsParams

__all__ = [
    "JOB_TYPES",
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


@dataclass(frozen=True)
class VoiceLoadNeeds:
    manifest: VoiceManifest
    spec: Any
    python: Any
    installed: Any
    reference: VoiceReference | None
    plan: Any
    state: Any


class LoadVoiceJobType:
    name = LOAD_VOICE.name

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
        manifest = known_voice(run_model(model, self.name, "a voice"))
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

    def requirements(self, model: str, reference: Any) -> VoiceLoadNeeds:
        manifest, spec, (python, installed) = require_loadable(
            self._config, self._backend, model
        )
        parsed = require_reference(manifest, reference)
        plan = voice_load_plan(self._config, self._backend, manifest, spec)
        state = card_guard(
            self._config,
            model=model,
            need_bytes=plan.need_bytes,
            owned_pids=self._residency.owned_pids(),
            reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
        )
        return VoiceLoadNeeds(manifest, spec, python, installed, parsed, plan, state)

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name, "a voice")
        validated = parse_params(LoadVoiceParams, params, self.name)
        self._residency.refuse_if_claimed(f"loading {model!r}")
        self.requirements(model, validated.reference)

    def run(self, job: Job, ctx: JobContext) -> None:
        params = LoadVoiceParams.model_validate(job.params)
        model = run_model(job.model, self.name, "a voice")
        self._residency.begin_warming(model, KIND_TTS)
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
        ctx.warming(f"checking the accelerator for {model}")
        needs = as_job_error(self.requirements, model, params.reference)
        ctx.warming(needs.state.detail)

        ctx.raise_if_cancelled()
        ctx.progress(0.0, f"loading {model}")
        try:
            resident = occupy_voice(
                self._residency,
                needs.manifest,
                needs.spec,
                needs.installed.path,
                needs.python,
                reference=needs.reference,
                timeout=params.timeout_s,
                on_progress=ctx.warming,
                serving_width=needs.plan.width,
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
            extra["lease_id"] = open_lease_for_load(
                self._leases,
                kind=resident.kind,
                subject=resident.voice_id,
                request=params.lease,
                client=client,
            )
        ctx.done_extra(**extra)


class UnloadVoiceJobType(UnloadJobType):
    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        super().__init__(
            UNLOAD_VOICE,
            residency,
            describe=lambda: describe_voices(config, residency),
            provenance=lambda model: voice_provenance(config.backend_kind, model),
        )


JOB_TYPES: tuple[JobTypeBinding, ...] = (
    JobTypeBinding(
        LOAD_VOICE,
        lambda wiring: LoadVoiceJobType(
            wiring.config, wiring.backend, wiring.residency, wiring.leases
        ),
    ),
    JobTypeBinding(
        UNLOAD_VOICE,
        lambda wiring: UnloadVoiceJobType(
            wiring.config, wiring.backend, wiring.residency
        ),
    ),
    JobTypeBinding(
        TTS_JOB,
        lambda wiring: TtsJobType(wiring.config, wiring.backend, wiring.residency),
    ),
)
