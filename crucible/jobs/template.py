from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from .. import accelerator, workers
from ..errors import ApiError, JobError
from ..manifests import fingerprint
from .base import JobContext, ModelDescriptor

if TYPE_CHECKING:
    from ..config import Config
    from ..residency import Residency

M = TypeVar("M")
P = TypeVar("P", bound=BaseModel)
R = TypeVar("R")


def _problems(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in problem['loc']) or '<root>'}: {problem['msg']}"
        for problem in exc.errors()
    )


def parse_params(
    model: type[P], params: dict[str, Any], job_type: str, *, lead: str | None = None
) -> P:
    try:
        return model.model_validate(params)
    except ValidationError as exc:
        raise ApiError(
            400,
            "invalid_params",
            f"{lead or f'{job_type} params are not valid'}: {_problems(exc)}",
        ) from None


def as_job_error(step: Callable[..., R], *args: Any, **kwargs: Any) -> R:
    try:
        return step(*args, **kwargs)
    except ApiError as exc:
        raise JobError(exc.code, exc.message) from None


def require_model(model: str | None, job_type: str, noun: str = "a model") -> str:
    if model is None:
        raise ApiError(400, "model_required", f"{job_type} needs {noun}")
    return model


def run_model(model: str | None, job_type: str, noun: str = "a model") -> str:
    if model is None:
        raise JobError("model_required", f"{job_type} needs {noun}")
    return model


def card_guard(
    config: "Config",
    *,
    model: str,
    need_bytes: int,
    owned_pids: frozenset[int],
    reclaimable_bytes: int = 0,
) -> accelerator.AcceleratorState:
    return accelerator.guard(
        config.backend_kind,
        model_id=model,
        need_bytes=need_bytes,
        owned_pids=owned_pids,
        desktop_allowance_bytes=config.desktop_allowance_bytes,
        reclaimable_bytes=reclaimable_bytes,
    )


class ManifestCatalog(Generic[M]):
    def __init__(
        self,
        load: Callable[[], dict[str, M]],
        error: type[Exception],
        *,
        unreadable_code: str,
        what: str,
        unknown: str,
        unknown_code: str = "unknown_model",
        offer_in_details: bool = False,
    ) -> None:
        self._load = load
        self._error = error
        self._unreadable_code = unreadable_code
        self._what = what
        self._unknown = unknown
        self._unknown_code = unknown_code
        self._offer_in_details = offer_in_details

    def all(self) -> dict[str, M]:
        try:
            return self._load()
        except self._error as exc:
            raise ApiError(
                500,
                self._unreadable_code,
                f"this server cannot read its {self._what}: {exc}",
            ) from None

    def known(self, model_id: str) -> M:
        manifests = self.all()
        manifest = manifests.get(model_id)
        if manifest is not None:
            return manifest
        raise ApiError(
            400,
            self._unknown_code,
            f"no {self._unknown} {model_id!r}; this build ships {sorted(manifests)}",
            {"model": model_id, "offered": sorted(manifests)}
            if self._offer_in_details
            else None,
        )

    def memory_estimate(self, model_id: str, backend_kind: str) -> int:
        manifest: Any = self.known(model_id)
        if not manifest.supports(backend_kind):
            return 0
        return manifest.spec(backend_kind).memory_bytes_estimate

    def provenance(self, backend_kind: str, model: str | None) -> dict[str, Any] | None:
        if model is None:
            return None
        manifest: Any = self.known(model)
        spec = manifest.backends.get(backend_kind)
        if spec is None:
            return {"id": model, "revision": None, "fingerprint": None}
        return {
            "id": model,
            "revision": spec.revision,
            "fingerprint": fingerprint(model, spec.revision),
        }

    def descriptors(
        self,
        backend_kind: str,
        *,
        installed: Callable[[Any, Any], bool],
        resident: Callable[[str], bool],
        source: Callable[[Any], str] = lambda spec: spec.hf_repo,
        estimate: Callable[[Any], int] = lambda spec: spec.memory_bytes_estimate,
    ) -> list[ModelDescriptor]:
        rows: list[ModelDescriptor] = []
        for manifest in self.all().values():
            entry: Any = manifest
            if entry.supports(backend_kind):
                spec = entry.spec(backend_kind)
                revision, origin, need = spec.revision, source(spec), estimate(spec)
                present = installed(entry, spec)
            else:
                revision, origin, need, present = "", "", 0, False
            rows.append(
                ModelDescriptor(
                    id=entry.id,
                    revision=revision,
                    source=origin,
                    installed=present,
                    resident=resident(entry.id),
                    vram_bytes=need,
                )
            )
        return rows


class ResidentWorker:
    resident_kind: ClassVar[str]
    _config: "Config"
    _residency: "Residency"

    def _resident_session(self) -> workers.WorkerSession | None:
        raise NotImplementedError

    def _guard(self, model: str, need_bytes: int) -> accelerator.AcceleratorState:
        return card_guard(
            self._config,
            model=model,
            need_bytes=need_bytes,
            owned_pids=self._residency.owned_pids(),
            reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
        )

    def _session(
        self,
        ctx: JobContext,
        model: str,
        need_bytes: int,
        load: Callable[[], None],
    ) -> workers.WorkerSession:
        session = self._resident_session()
        if session is not None and self._residency.is_resident(self.resident_kind, model):
            if session.alive:
                return session
            ctx.warming(
                f"the resident {model} worker is gone (its log is "
                f"{session.log_path}); loading it again"
            )
            self._forget(ctx, model)
        state = as_job_error(self._guard, model, need_bytes)
        ctx.warming(state.detail)
        try:
            load()
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None
        loaded = self._resident_session()
        if loaded is None:
            raise JobError(
                "worker_failed",
                f"{model} loaded but no session was published; this is a bug in "
                f"{type(self).__module__}, whose occupant carried no session",
            )
        return loaded

    def _forget(self, ctx: JobContext, model: str) -> None:
        try:
            self._residency.unload(model)
        except (KeyError, workers.WorkerError) as exc:
            line = f"could not take {model} off the card: {type(exc).__name__}: {exc}"
            print(f"crucible: {line}", file=sys.stderr)
            ctx.note(line)


__all__ = [
    "ManifestCatalog",
    "ResidentWorker",
    "as_job_error",
    "card_guard",
    "parse_params",
    "require_model",
    "run_model",
]
