from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ...errors import JobError
from ...jobtypes import ECHO_JOB
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from ..binding import JobTypeBinding

_SLICE_SECONDS = 0.02


class EchoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delay_ms: int = Field(
        default=25,
        ge=0,
        le=60_000,
        description="Milliseconds to wait before copying the inputs to artifacts, "
        "0 to 60000; a cancel is honoured during the wait.",
    )


class EchoJobType:
    name = ECHO_JOB.name

    def describe_models(self) -> list[ModelDescriptor]:
        return []

    def vram_estimate(self, model: str | None) -> int:
        if model is not None:
            raise JobError("unknown_model", f"echo serves no models, got {model!r}")
        return 0

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        if model is not None:
            raise JobError("unknown_model", f"echo serves no models, got {model!r}")
        return None

    def check(self, backend: Any) -> JobTypeStatus:
        return JobTypeStatus(
            ready=True,
            detail="enabled; copies inputs to artifacts, uses no accelerator",
        )

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        return None

    def run(self, job: Job, ctx: JobContext) -> None:
        params = EchoParams.model_validate(job.params)
        inputs = ctx.inputs()
        if not inputs:
            raise JobError("no_inputs", "echo needs at least one input")

        total = len(inputs)
        for index, (name, path) in enumerate(inputs.items()):
            ctx.raise_if_cancelled()
            ctx.progress(index / total, f"copying {name}")
            self._sleep(ctx, params.delay_ms / 1000.0)
            ctx.raise_if_cancelled()
            ctx.artifact(name, path)
        ctx.progress(1.0, f"echoed {total} input(s)")

    @staticmethod
    def _sleep(ctx: JobContext, seconds: float) -> None:
        remaining = seconds
        while remaining > 0:
            ctx.raise_if_cancelled()
            slice_seconds = min(_SLICE_SECONDS, remaining)
            time.sleep(slice_seconds)
            remaining -= slice_seconds


JOB_TYPES: tuple[JobTypeBinding, ...] = (
    JobTypeBinding(ECHO_JOB, lambda wiring: EchoJobType()),
)
