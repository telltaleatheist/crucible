from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from pydantic import BaseModel, ConfigDict

from ... import weights, workers
from ...backend import MLX_DARWIN
from ...cardkinds import KIND_SEGMENT
from ...clock import utcnow
from ...config import Config
from ...errors import ApiError, JobCancelled, JobError
from ...jobtypes import LOAD_SEGMENT, SEGMENT_JOB, UNLOAD_SEGMENT
from ...manifests import fingerprint
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    Occupant,
    Residency,
    ResidentSegmenter,
    say_to,
)
from ...segmentmodels import (
    SegmentBackendSpec,
    SegmentManifest,
    SegmentManifestError,
    load_all_segment_manifests,
)
from .. import worker_type
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from ..binding import JobTypeBinding
from ..image import IMAGE_MAGIC
from ..leaseonload import (
    HeldForLoad,
    LeaseOnLoad,
    hold_for_load,
    let_go_of,
    require_lease_request,
)
from ..template import (
    ManifestCatalog,
    ResidentWorker,
    as_job_error,
    parse_params,
    require_model,
    run_model,
)
from ..unload import UnloadJobType
from .params import SegmentParams, refuse_outside, refuse_what_the_model_cannot_take
from .picture import picture_size

__all__ = [
    "JOB_TYPES",
    "LoadSegmentJobType",
    "LoadSegmentParams",
    "SegmentJobType",
    "SegmentParams",
    "UnloadSegmentJobType",
    "occupy_segmenter",
]

JOB_TYPE = SEGMENT_JOB.name

MASK_ARTIFACT = "mask.png"

CUTOUT_ARTIFACT = "cutout.png"

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

READY_SILENCE_TIMEOUT_SECONDS = 300.0

WORKER_SCRIPT = Path(__file__).resolve().parent / "worker.py"

WORKER_ENVIRONMENT = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

MPS_FALLBACK = {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}


def worker_environment(python: Path, backend_kind: str) -> dict[str, str]:
    return {
        **workers.worker_environment(python.parent.parent),
        **workers.torch_allocator_environment(backend_kind),
        **WORKER_ENVIRONMENT,
        **(MPS_FALLBACK if backend_kind == MLX_DARWIN else {}),
    }


def start_segment_session(
    python: Path,
    needs: "Needs",
    log_path: Path,
    *,
    ready_silence_timeout: float,
    on_ready: Callable[[dict[str, Any]], None] | None = None,
) -> workers.WorkerSession:
    spec = needs.spec
    session = workers.WorkerSession(
        python=python,
        script=WORKER_SCRIPT,
        log_path=log_path,
        environment=worker_environment(python, spec.backend),
    )
    outcome = session.start(
        {
            "op": "load",
            "engine": spec.engine,
            "model_dir": str(needs.weights_dir),
            "device": spec.device,
            "dtype": spec.dtype,
            "working_side": spec.working_side,
            "memory_cap_bytes": workers.torch_memory_cap(
                spec.backend, spec.memory_bytes_estimate
            ),
        },
        ready_silence_timeout=ready_silence_timeout,
        on_ready=on_ready,
    )
    if outcome.results:
        session.stop()
        raise workers.WorkerError(
            f"{WORKER_SCRIPT.name} answered a load request with "
            f"{len(outcome.results)} result(s); a load produces none"
        )
    return session


def occupy_segmenter(
    residency: Residency,
    needs: "Needs",
    *,
    timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
    on_progress: Callable[[str], None] | None = None,
) -> ResidentSegmenter:
    say = say_to(on_progress)
    manifest, spec = needs.manifest, needs.spec
    loaded: dict[str, Any] = {}

    def start() -> Occupant:
        log_path = residency.log_path_for(manifest.id)
        say(f"starting the {spec.engine} worker for {manifest.id}; log {log_path}")
        session = start_segment_session(
            needs.python, needs, log_path, ready_silence_timeout=timeout, on_ready=loaded.update
        )
        resident = ResidentSegmenter(
            model_id=manifest.id,
            backend=spec.backend,
            engine=spec.engine,
            revision=spec.revision,
            fingerprint=fingerprint(manifest.id, spec.revision),
            device=spec.device,
            dtype=spec.dtype,
            versions=dict(loaded.get("versions") or {}),
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=utcnow(),
        )
        return Occupant(resident, session=session)

    return cast(ResidentSegmenter, residency.occupy(KIND_SEGMENT, manifest.id, start, say=say))


MANIFESTS: ManifestCatalog[SegmentManifest] = ManifestCatalog(
    lambda: load_all_segment_manifests(),
    SegmentManifestError,
    unreadable_code="segment_manifests_unreadable",
    what="segment manifests",
    unknown="segment model",
)


def _descriptors(config: Config, residency: Residency) -> list[ModelDescriptor]:
    return MANIFESTS.descriptors(
        config.backend_kind,
        installed=lambda manifest, spec: weights.installed(config, manifest, spec)
        is not None,
        resident=lambda model_id: residency.is_resident(KIND_SEGMENT, model_id),
    )


@dataclass(frozen=True)
class Needs:
    manifest: SegmentManifest
    spec: SegmentBackendSpec
    python: Path
    weights_dir: Path


@dataclass(frozen=True)
class Picture:
    path: Path
    width: int
    height: int


class _Segmenting:
    def __init__(self, ctx: JobContext) -> None:
        self._ctx = ctx

    def progress(self, message: dict[str, Any]) -> None:
        stage = str(message.get("stage"))
        if stage == "cancelled":
            return
        fraction = min(1.0, max(0.0, float(message.get("fraction") or 0.0)))
        self._ctx.progress(fraction, stage, stage=stage)


def input_picture(ctx: JobContext, params: SegmentParams, spec: SegmentBackendSpec) -> Picture:
    inputs = ctx.inputs()
    if len(inputs) != 1:
        raise JobError(
            "invalid_inputs",
            f"a segment job reads exactly one picture and this one carries "
            f"{len(inputs)} input(s) ({sorted(inputs)}); send the PNG, JPEG or WebP "
            "to cut from as the job's one input",
        )
    path = next(iter(inputs.values()))
    with path.open("rb") as handle:
        head = handle.read(16)
    if not any(head[at : at + len(magic)] == magic for magic, at, _ in IMAGE_MAGIC):
        raise JobError(
            "invalid_inputs",
            f"input {path.name!r} is not a "
            f"{', '.join(name for _, _, name in IMAGE_MAGIC)} image (its first bytes "
            f"are {head[:8].hex()}); send the picture itself",
        )
    size = picture_size(path)
    if size is None:
        raise JobError(
            "invalid_inputs",
            f"input {path.name!r} begins like a picture but its header does not say "
            "how big it is, so it is damaged or cut short; send the whole file",
        )
    width, height = size
    if width * height > spec.max_pixels:
        raise JobError(
            "image_too_large",
            f"{path.name!r} is {width}x{height}, {width * height:,} pixels; this "
            f"server segments at most {spec.max_pixels:,} pixels (the mask and the "
            "cutout are made at the input's full size). Send a smaller copy and scale "
            "the mask up",
        )
    refuse_outside(params, width, height, repr(path.name))
    return Picture(path, width, height)


def check_png(path: Path, what: str) -> None:
    head = path.read_bytes()[: len(PNG_MAGIC)] if path.is_file() else b""
    if head != PNG_MAGIC:
        raise JobError(
            "worker_failed",
            f"the segment worker was to write the {what} at {path} and it is not a PNG "
            f"(it begins {head.hex() or 'with nothing'}); the worker log says why",
        )


class SegmentJobType(ResidentWorker):

    name = JOB_TYPE
    resident_kind = KIND_SEGMENT

    def __init__(
        self, config: Config, backend: Any, residency: Residency, leases: Any | None = None
    ) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency
        self._leases = leases

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return _descriptors(self._config, self._residency)

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return MANIFESTS.provenance(self._config.backend_kind, run_model(model, self.name))

    def vram_estimate(self, model: str | None) -> int:
        return MANIFESTS.memory_estimate(run_model(model, self.name), self._config.backend_kind)

    def check(self, backend: Any) -> JobTypeStatus:
        env = worker_type.env_or_status(self._config, JOB_TYPE, backend.kind)
        if isinstance(env, JobTypeStatus):
            return env
        try:
            manifests = MANIFESTS.all()
        except ApiError as exc:
            return JobTypeStatus(ready=False, detail=exc.message)
        installed = worker_type.installed_ids(self._config, manifests.values(), backend.kind)
        if not installed:
            offered = sorted(m.id for m in manifests.values() if m.supports(backend.kind))
            pulls = " or ".join(f"`crucible models pull {model}`" for model in offered)
            return JobTypeStatus(
                ready=False,
                detail=f"{env.detail}; no segment model is installed — {pulls or 'none is declared for this backend'}",
            )
        return JobTypeStatus(ready=True, detail=f"{env.detail}; installed: {installed}")

    def _resident_session(self) -> workers.WorkerSession | None:
        return self._residency.segment_session

    def loadable(self, model_id: str) -> Needs:
        backend_kind = self._backend.kind
        manifest = MANIFESTS.known(model_id)
        spec = _require_block(manifest, model_id, backend_kind)
        worker_type.refuse_if_larger_than_host(self._backend, model_id, spec.memory_bytes_estimate)
        python = worker_type.require_worker_python(self._config, JOB_TYPE, backend_kind, model_id)
        return Needs(
            manifest,
            spec,
            python,
            worker_type.require_weights(self._config, manifest, spec, model_id),
        )

    def requirements(self, model_id: str, params: SegmentParams) -> Needs:
        catalog = MANIFESTS.all()
        manifest = MANIFESTS.known(model_id)
        refuse_what_the_model_cannot_take(
            params, manifest, _require_block(manifest, model_id, self._backend.kind), catalog
        )
        return self.loadable(model_id)

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        parsed = parse_params(SegmentParams, params, self.name)
        self._admit(model, self.requirements(model, parsed), parsed.lease, "segmenting with")

    def _admit(self, model: str, needs: Needs, lease: LeaseOnLoad | None, doing: str) -> None:
        if lease is not None:
            require_lease_request(lease, act=needs.manifest.kind)
        self._residency.refuse_if_claimed(f"{doing} {model!r}")
        if self._residency.is_resident(KIND_SEGMENT, model):
            return
        self._guard(model, needs.spec.memory_bytes_estimate)

    def _hold(self, job: Job, model: str, lease: LeaseOnLoad | None) -> HeldForLoad | None:
        if lease is None:
            return None
        return hold_for_load(
            self._leases, kind=KIND_SEGMENT, subject=model, request=lease, client=job.client
        )

    def _worker(self, ctx: JobContext, model: str, needs: Needs) -> workers.WorkerSession:
        return self._session(
            ctx,
            model,
            needs.spec.memory_bytes_estimate,
            lambda: occupy_segmenter(self._residency, needs, on_progress=ctx.warming),
        )

    def _segment(
        self, ctx: JobContext, model: str, session: workers.WorkerSession, request: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            outcome = session.send(
                request,
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
                on_progress=_Segmenting(ctx).progress,
                cancelled=lambda: ctx.cancelled,
                cancel_request={"op": "cancel", "request_id": request["request_id"]},
            )
            (result,) = workers.require_positional_results(outcome, 1, "mask")
            return result
        except workers.WorkerError as exc:
            self._forget(ctx, model)
            raise JobError("worker_failed", str(exc)) from None
        except JobCancelled:
            if not session.alive:
                self._forget(ctx, model)
            raise

    def run(self, job: Job, ctx: JobContext) -> None:
        params = SegmentParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        needs = as_job_error(self.requirements, model, params)
        picture = input_picture(ctx, params, needs.spec)
        session = self._worker(ctx, model, needs)
        held = self._hold(job, model, params.lease)
        try:
            self._make(ctx, model, session, params, needs, picture)
        except BaseException:
            let_go_of(self._leases, held)
            raise
        ctx.done_extra(lease_id=None if held is None else held.lease_id)

    def _make(
        self,
        ctx: JobContext,
        model: str,
        session: workers.WorkerSession,
        params: SegmentParams,
        needs: Needs,
        picture: Picture,
    ) -> None:
        mask = ctx.scratch / MASK_ARTIFACT
        cutout = ctx.scratch / CUTOUT_ARTIFACT
        request = segment_request(params, needs, picture, mask, cutout)
        result = self._segment(ctx, model, session, request)
        check_png(mask, "mask")
        check_png(cutout, "cutout")
        ctx.artifact(MASK_ARTIFACT, mask)
        ctx.artifact(CUTOUT_ARTIFACT, cutout)
        ctx.progress(1.0, f"{result.get('width')}x{result.get('height')} mask made", stage="done")
        ctx.done_extra(
            segment=effective_params(params, needs, picture, result),
            resident=self._residency.resident_id,
        )


def _require_block(manifest: SegmentManifest, model_id: str, backend_kind: str) -> SegmentBackendSpec:
    return worker_type.require_block(manifest, model_id, backend_kind, "segment model")


def _points(params: SegmentParams) -> list[dict[str, Any]] | None:
    if params.points is None:
        return None
    return [{"x": p.x, "y": p.y, "label": p.label} for p in params.points]


def segment_request(
    params: SegmentParams, needs: Needs, picture: Picture, mask: Path, cutout: Path
) -> dict[str, Any]:
    return {
        "op": "segment",
        "request_id": uuid.uuid4().hex,
        "kind": needs.manifest.kind,
        "image_path": str(picture.path),
        "width": picture.width,
        "height": picture.height,
        "points": _points(params),
        "box": None if params.box is None else list(params.box),
        "mask_path": str(mask),
        "cutout_path": str(cutout),
        "revision": needs.spec.revision,
        "backend": needs.spec.backend,
    }


def effective_params(
    params: SegmentParams, needs: Needs, picture: Picture, result: dict[str, Any]
) -> dict[str, Any]:
    spec = needs.spec
    return {
        "model": needs.manifest.id,
        "kind": needs.manifest.kind,
        "hf_repo": spec.hf_repo,
        "revision": spec.revision,
        "backend": spec.backend,
        "engine": spec.engine,
        "dtype": spec.dtype,
        "input": picture.path.name,
        "width": result.get("width", picture.width),
        "height": result.get("height", picture.height),
        "points": _points(params),
        "box": None if params.box is None else list(params.box),
        "mask": MASK_ARTIFACT,
        "cutout": CUTOUT_ARTIFACT,
        "score": result.get("score"),
        "multimask": result.get("multimask"),
        "coverage": result.get("coverage"),
        "seconds": result.get("seconds"),
        "stage_seconds": result.get("stage_seconds"),
        "peak_bytes": result.get("peak_bytes"),
        "stage_peak_bytes": result.get("stage_peak_bytes"),
        "memory_bytes_estimate": spec.memory_bytes_estimate,
        "memory_basis": spec.memory_basis,
        "versions": result.get("versions"),
    }


class LoadSegmentParams(BaseModel):
    """`params` for a load-segment job: warm a segment model up, optionally leased."""

    model_config = ConfigDict(extra="forbid")

    lease: LeaseOnLoad | None = None


class LoadSegmentJobType(SegmentJobType):

    name = LOAD_SEGMENT.name

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        parsed = parse_params(LoadSegmentParams, params, self.name)
        self._admit(model, self.loadable(model), parsed.lease, "loading")

    def run(self, job: Job, ctx: JobContext) -> None:
        params = LoadSegmentParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        needs = as_job_error(self.loadable, model)
        ctx.progress(0.0, f"loading {model}")
        self._worker(ctx, model, needs)
        held = self._hold(job, model, params.lease)
        ctx.progress(1.0, f"{model} is resident")
        ctx.done_extra(
            resident=self._residency.resident_id,
            lease_id=None if held is None else held.lease_id,
        )


class UnloadSegmentJobType(UnloadJobType):
    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        super().__init__(
            UNLOAD_SEGMENT,
            residency,
            describe=lambda: _descriptors(config, residency),
            provenance=lambda model: MANIFESTS.provenance(config.backend_kind, model),
        )


JOB_TYPES: tuple[JobTypeBinding, ...] = (
    JobTypeBinding(
        SEGMENT_JOB,
        lambda wiring: SegmentJobType(wiring.config, wiring.backend, wiring.residency, wiring.leases),
    ),
    JobTypeBinding(
        UNLOAD_SEGMENT,
        lambda wiring: UnloadSegmentJobType(wiring.config, wiring.backend, wiring.residency),
    ),
    JobTypeBinding(
        LOAD_SEGMENT,
        lambda wiring: LoadSegmentJobType(wiring.config, wiring.backend, wiring.residency, wiring.leases),
    ),
)
