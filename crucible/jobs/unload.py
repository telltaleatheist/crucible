from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from ..cardkinds import KIND_NOUNS
from ..engines import EngineError
from ..errors import ApiError, JobError
from ..jobtypes import JobTypeSpec
from ..residency import Residency, describe_resident
from ..workers import WorkerError
from .base import Job, JobContext, JobTypeStatus, ModelDescriptor
from .template import parse_params, require_model, run_model


class UnloadParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _with_article(noun: str) -> str:
    return f"{'an' if noun[0] in 'aeiou' else 'a'} {noun}"


class UnloadJobType:
    def __init__(
        self,
        spec: JobTypeSpec,
        residency: Residency,
        *,
        describe: Callable[[], list[ModelDescriptor]],
        provenance: Callable[[str | None], dict[str, Any] | None],
    ) -> None:
        if spec.unloads is None:
            raise TypeError(
                f"{spec.name!r} takes nothing off the card by its spec in "
                "crucible/jobtypes.py, so it is not an unload job type"
            )
        self.name = spec.name
        self._kind = spec.unloads
        self._noun = KIND_NOUNS[spec.unloads]
        self._residency = residency
        self._describe = describe
        self._provenance = provenance

    @property
    def residency(self) -> Residency:
        return self._residency

    @property
    def not_resident_code(self) -> str:
        return f"{self._noun}_not_resident"

    def describe_models(self) -> list[ModelDescriptor]:
        return self._describe()

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return self._provenance(model)

    def vram_estimate(self, model: str | None) -> int:
        return 0

    def check(self, backend: Any) -> JobTypeStatus:
        resident = self._residency.resident
        if resident is not None and resident.kind == self._kind:
            return JobTypeStatus(ready=True, detail=f"resident: {resident.id}")
        return JobTypeStatus(ready=True, detail=f"no {self._noun} is resident")

    def _not_resident(self, model: str) -> str:
        return f"{model!r} is not resident on this server; " + describe_resident(
            self._residency, self._kind, f"no {self._noun} is"
        )

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name, _with_article(self._noun))
        parse_params(UnloadParams, params, self.name)
        if self._residency.being_cleared(model):
            return
        self._residency.refuse_if_claimed(f"unloading {model!r}")
        if not self._residency.is_resident(self._kind, model):
            raise ApiError(
                409,
                self.not_resident_code,
                self._not_resident(model),
                {"requested": model, "resident": self._residency.resident_id},
            )

    def run(self, job: Job, ctx: JobContext) -> None:
        UnloadParams.model_validate(job.params)
        model = run_model(job.model, self.name, _with_article(self._noun))
        if self._residency.await_clearance(model):
            ctx.progress(0.0, f"unloading {model}")
            ctx.progress(1.0, f"{model} is unloaded — the card was cleared of it")
            ctx.done_extra(resident=self._residency.resident_id)
            return
        if not self._residency.is_resident(self._kind, model):
            raise JobError(self.not_resident_code, self._not_resident(model))
        ctx.progress(0.0, f"unloading {model}")
        try:
            self._residency.unload(model)
        except KeyError:
            raise JobError(self.not_resident_code, self._not_resident(model)) from None
        except EngineError as exc:
            raise JobError("engine_failed", str(exc)) from None
        except WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None
        ctx.progress(1.0, f"{model} is unloaded")
        ctx.done_extra(resident=self._residency.resident_id)


__all__ = ["UnloadJobType", "UnloadParams"]
