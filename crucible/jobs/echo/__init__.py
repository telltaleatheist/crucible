"""`echo` — the test job type.

Copies each input to an artifact of the same name, emitting progress between the
copies so the SSE stream has something ordered to assert on. It touches no
accelerator and serves no model. Registered only when `[jobs] enable_echo = true`.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ...errors import JobError
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor

# Cancellation is cooperative: the delay is slept in slices this long so that a
# DELETE /jobs/{id} lands promptly instead of after the whole delay.
_SLICE_SECONDS = 0.02


class EchoParams(BaseModel):
    """`params` for an echo job. Unknown keys are refused, not ignored."""

    model_config = ConfigDict(extra="forbid")

    delay_ms: int = Field(default=25, ge=0, le=60_000)


class EchoJobType:
    name = "echo"

    def describe_models(self) -> list[ModelDescriptor]:
        return []

    def vram_estimate(self, model: str | None) -> int:
        if model is not None:
            raise JobError("unknown_model", f"echo serves no models, got {model!r}")
        return 0

    def check(self, backend: Any) -> JobTypeStatus:
        return JobTypeStatus(
            ready=True,
            detail="enabled; copies inputs to artifacts, uses no accelerator",
        )

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
