from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ... import weights, workers
from ...backend import CUDA_LINUX, MLX_DARWIN
from ...cardkinds import KIND_IMAGE
from ...clock import utcnow
from ...config import Config
from ...errors import ApiError, JobCancelled, JobError
from ...imagemodels import (
    ImageBackendSpec,
    ImageManifest,
    ImageManifestError,
    load_all_image_manifests,
)
from ...jobtypes import IMAGE_JOB, LOAD_IMAGE, UNLOAD_IMAGE
from ...manifests import fingerprint
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    Occupant,
    Residency,
    ResidentImage,
    say_to,
)
from .. import worker_type
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from ..binding import JobTypeBinding
from ..template import (
    ManifestCatalog,
    ResidentWorker,
    as_job_error,
    parse_params,
    require_model,
    run_model,
)
from ..unload import UnloadJobType
from . import inpaint

__all__ = [
    "JOB_TYPES",
    "ImageJobType",
    "ImageParams",
    "LoadImageJobType",
    "LoadImageParams",
    "UnloadImageJobType",
    "occupy_image",
]

JOB_TYPE = IMAGE_JOB.name


SIDE_MULTIPLE = 16

MIN_SIDE = 256

MAX_SIDE = 2048

MAX_STEPS = 100

DEFAULT_STEPS = 40

DEFAULT_SIDE = 1024

MAX_SEED = 2**32 - 1

MAX_GUIDANCE = 10.0

ARTIFACT_NAME = "image.png"

GENERATED_NAME = "generated.png"

READY_SILENCE_TIMEOUT_SECONDS = 900.0

WORKER_SCRIPT = Path(__file__).resolve().parent / "worker.py"

DEVICE_FOR_BACKEND: dict[str, str] = {
    CUDA_LINUX: "cuda",
    MLX_DARWIN: "metal",
}

IMAGE_MAGIC: tuple[tuple[bytes, int, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", 0, "PNG"),
    (b"\xff\xd8\xff", 0, "JPEG"),
    (b"WEBP", 8, "WebP"),
)


def device_for(backend_kind: str) -> str:
    found = DEVICE_FOR_BACKEND.get(backend_kind)
    if found is None:
        raise JobError(
            "backend_unsupported",
            f"there is no image device for backend {backend_kind!r}; this build "
            f"generates images on {sorted(DEVICE_FOR_BACKEND)}",
        )
    return found


class ImageParams(BaseModel):
    """`params` for an image job. Unknown keys are refused, not ignored."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    negative_prompt: str | None = None
    width: int = Field(default=DEFAULT_SIDE, ge=MIN_SIDE, le=MAX_SIDE)
    height: int = Field(default=DEFAULT_SIDE, ge=MIN_SIDE, le=MAX_SIDE)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    steps: int = Field(default=DEFAULT_STEPS, ge=1, le=MAX_STEPS)
    guidance: float = Field(default=1.0, ge=1.0, le=MAX_GUIDANCE)
    image_strength: float | None = Field(default=None, gt=0.0, lt=1.0)
    mask: str | None = None
    mask_blur: int | None = Field(default=None, ge=0, le=inpaint.MAX_MASK_BLUR)

    @field_validator("prompt")
    @classmethod
    def says_something(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError("the prompt is empty; describe the picture")
        return value

    @field_validator("width", "height")
    @classmethod
    def whole_tiles(cls, value: int) -> int:
        if value % SIDE_MULTIPLE:
            below = value - value % SIDE_MULTIPLE
            raise ValueError(
                f"{value} is not a multiple of {SIDE_MULTIPLE}; the model works in "
                f"{SIDE_MULTIPLE}-pixel tiles. Send {below} or {below + SIDE_MULTIPLE}"
            )
        return value

    @field_validator("mask")
    @classmethod
    def names_an_input(cls, value: str | None) -> str | None:
        if value is not None and value.strip() == "":
            raise ValueError(
                "mask is empty; send the name of the input that carries the mask, "
                'for example "mask.png"'
            )
        return value

    @model_validator(mode="after")
    def blur_needs_a_mask(self) -> "ImageParams":
        if self.mask_blur is not None and self.mask is None:
            raise ValueError(
                "mask_blur softens the edge of a mask and this job has none; send mask "
                "(the name of the mask input) or drop mask_blur"
            )
        return self

    @property
    def effective_mask_blur(self) -> int | None:
        if self.mask is None:
            return None
        return inpaint.DEFAULT_MASK_BLUR if self.mask_blur is None else self.mask_blur

    @model_validator(mode="after")
    def guidance_needs_a_negative(self) -> "ImageParams":
        if self.negative_prompt is not None and self.guidance <= 1.0:
            raise ValueError(
                "negative_prompt is only read when guidance is above 1.0 (true "
                "classifier-free guidance, which runs the model twice per step); "
                "send guidance, for example 4.0, or drop negative_prompt"
            )
        if self.negative_prompt is None and self.guidance > 1.0:
            raise ValueError(
                "guidance above 1.0 needs a negative_prompt to guide away from; "
                "without one the engines ignore guidance. Send negative_prompt or "
                "guidance 1.0"
            )
        return self


def refuse_what_the_arm_cannot_make(params: ImageParams, spec: ImageBackendSpec, model: str) -> None:
    details = {"model": model, "backend": spec.backend, "width": params.width, "height": params.height}
    if params.width % spec.size_multiple or params.height % spec.size_multiple:
        raise ApiError(
            400,
            "image_size_not_supported",
            f"{model} on {spec.backend} needs width and height in multiples of "
            f"{spec.size_multiple} ({spec.engine} works in {spec.size_multiple}-pixel "
            f"blocks); {params.width}x{params.height} is not. Round each side to a "
            f"multiple of {spec.size_multiple}",
            {**details, "size_multiple": spec.size_multiple},
        )
    pixels = params.width * params.height
    if max(params.width, params.height) > spec.max_side or pixels > spec.max_pixels:
        raise ApiError(
            400,
            "image_too_large",
            f"{params.width}x{params.height} is {pixels:,} pixels; {model} on "
            f"{spec.backend} makes at most {spec.max_pixels:,} pixels with no side "
            f"over {spec.max_side}, because its memory ({spec.memory_basis}: "
            f"{spec.memory_note}) was sized at that limit. Ask for a smaller image",
            {**details, "max_pixels": spec.max_pixels, "max_side": spec.max_side},
        )
    if params.image_strength is not None and not spec.image_to_image:
        raise ApiError(
            400,
            "image_to_image_unsupported",
            f"{model} on {spec.backend} ({spec.engine}) does not start from an input "
            "image yet; drop image_strength and the input, or send the job to a "
            "server whose capability row says it does",
            details,
        )
    if params.mask is not None and not spec.inpaint:
        raise ApiError(
            400,
            "inpaint_unsupported",
            f"{model} on {spec.backend} ({spec.engine}) does not regenerate a masked "
            "region; drop mask and mask_blur, or send the job to a server that does",
            details,
        )


@dataclass(frozen=True)
class Pictures:
    image: Path | None = None
    mask: Path | None = None


def _require_picture(path: Path, role: str) -> None:
    with path.open("rb") as handle:
        head = handle.read(16)
    if not any(head[at : at + len(magic)] == magic for magic, at, _ in IMAGE_MAGIC):
        raise JobError(
            "invalid_inputs",
            f"input {path.name!r} ({role}) is not a "
            f"{', '.join(name for _, _, name in IMAGE_MAGIC)} image (its first bytes "
            f"are {head[:8].hex()}); send the picture itself",
        )


def input_pictures(ctx: JobContext, params: ImageParams) -> Pictures:
    inputs = ctx.inputs()
    if params.mask is not None:
        return _masked(inputs, params.mask)
    if params.image_strength is None:
        if inputs:
            raise JobError(
                "invalid_inputs",
                f"this job carries input(s) {sorted(inputs)} and no image_strength or "
                "mask; an input image is only read for image-to-image (image_strength, "
                "0 to 1, how much of the input survives) or inpainting (mask, the name "
                "of the mask input)",
            )
        return Pictures()
    if len(inputs) != 1:
        raise JobError(
            "invalid_inputs",
            f"image_strength was sent with {len(inputs)} input(s) "
            f"({sorted(inputs)}); image-to-image starts from exactly one image",
        )
    path = next(iter(inputs.values()))
    _require_picture(path, "the image")
    return Pictures(image=path)


def _masked(inputs: dict[str, Path], mask_name: str) -> Pictures:
    if mask_name not in inputs:
        raise JobError(
            "invalid_inputs",
            f"mask names the input {mask_name!r} and this job carries {sorted(inputs)}; "
            "send the mask as an input under that name, beside the image",
        )
    others = sorted(name for name in inputs if name != mask_name)
    if len(others) != 1:
        raise JobError(
            "invalid_inputs",
            f"a masked job carries exactly two inputs, the image and the mask "
            f"{mask_name!r}; this one carries {sorted(inputs)}",
        )
    image, mask = inputs[others[0]], inputs[mask_name]
    _require_picture(image, "the image")
    _require_picture(mask, "the mask")
    image_size, mask_size = inpaint.picture_size(image), inpaint.picture_size(mask)
    if image_size is not None and mask_size is not None and image_size != mask_size:
        raise JobError(
            "mask_size_mismatch",
            f"the mask {mask_name!r} is {mask_size[0]}x{mask_size[1]} and the image "
            f"{image.name!r} is {image_size[0]}x{image_size[1]}; draw the mask on the "
            "image's own canvas, the same size",
        )
    return Pictures(image=image, mask=mask)


WORKER_ENVIRONMENT = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def start_image_session(
    python: Path,
    weights_dir: Path,
    spec: ImageBackendSpec,
    log_path: Path,
    *,
    ready_silence_timeout: float,
    on_ready: Callable[[dict[str, Any]], None] | None = None,
) -> workers.WorkerSession:
    session = workers.WorkerSession(
        python=python,
        script=WORKER_SCRIPT,
        log_path=log_path,
        environment={
            **workers.worker_environment(python.parent.parent),
            **workers.torch_allocator_environment(spec.backend),
            **WORKER_ENVIRONMENT,
        },
    )
    outcome = session.start(
        {
            "op": "load",
            "engine": spec.engine,
            "model_dir": str(weights_dir),
            "device": device_for(spec.backend),
            "dtype": spec.dtype,
            "mlx_cache_limit_bytes": spec.mlx_cache_limit_bytes,
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


def occupy_image(
    residency: Residency,
    manifest: ImageManifest,
    spec: ImageBackendSpec,
    weights_dir: Path,
    python: Path,
    *,
    timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
    on_progress: Callable[[str], None] | None = None,
) -> ResidentImage:
    say = say_to(on_progress)
    loaded: dict[str, Any] = {}

    def start() -> Occupant:
        log_path = residency.log_path_for(manifest.id)
        say(f"starting the {spec.engine} worker for {manifest.id}; log {log_path}")
        session = start_image_session(
            python,
            weights_dir,
            spec,
            log_path,
            ready_silence_timeout=timeout,
            on_ready=loaded.update,
        )
        resident = ResidentImage(
            model_id=manifest.id,
            backend=spec.backend,
            engine=spec.engine,
            revision=spec.revision,
            fingerprint=fingerprint(manifest.id, spec.revision),
            device=device_for(spec.backend),
            dtype=spec.dtype,
            versions=dict(loaded.get("versions") or {}),
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=utcnow(),
        )
        return Occupant(resident, session=session)

    return cast(ResidentImage, residency.occupy(KIND_IMAGE, manifest.id, start, say=say))


MANIFESTS: ManifestCatalog[ImageManifest] = ManifestCatalog(
    lambda: load_all_image_manifests(),
    ImageManifestError,
    unreadable_code="image_manifests_unreadable",
    what="image manifests",
    unknown="image model",
)


def _descriptors(config: Config, residency: Residency) -> list[ModelDescriptor]:
    return MANIFESTS.descriptors(
        config.backend_kind,
        installed=lambda manifest, spec: weights.installed(config, manifest, spec)
        is not None,
        resident=lambda model_id: residency.is_resident(KIND_IMAGE, model_id),
    )


@dataclass(frozen=True)
class Needs:
    manifest: ImageManifest
    spec: ImageBackendSpec
    python: Path
    weights_dir: Path


class _Generation:
    def __init__(self, ctx: JobContext, steps: int) -> None:
        self._ctx = ctx
        self._steps = steps

    def progress(self, message: dict[str, Any]) -> None:
        stage = str(message.get("stage"))
        step = int(message.get("step") or 0)
        if stage == "cancelled":
            return
        fraction = min(1.0, step / self._steps) if stage == "denoising" else (
            1.0 if stage in ("decoding", "saving") else 0.0
        )
        words = (
            f"step {step} of {self._steps}" if stage == "denoising" else stage
        )
        self._ctx.progress(fraction, words, stage=stage, step=step, steps=self._steps)


class ImageJobType(ResidentWorker):

    name = JOB_TYPE
    resident_kind = KIND_IMAGE

    def __init__(
        self, config: Config, backend: Any, residency: Residency
    ) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return _descriptors(self._config, self._residency)

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return MANIFESTS.provenance(self._config.backend_kind, run_model(model, self.name))

    def vram_estimate(self, model: str | None) -> int:
        return MANIFESTS.memory_estimate(
            run_model(model, self.name), self._config.backend_kind
        )

    def check(self, backend: Any) -> JobTypeStatus:
        env = worker_type.env_or_status(self._config, JOB_TYPE, backend.kind)
        if isinstance(env, JobTypeStatus):
            return env
        try:
            manifests = MANIFESTS.all()
        except ApiError as exc:
            return JobTypeStatus(ready=False, detail=exc.message)
        installed = worker_type.installed_ids(
            self._config, manifests.values(), backend.kind
        )
        if not installed:
            offered = sorted(
                manifest.id for manifest in manifests.values() if manifest.supports(backend.kind)
            )
            pulls = " or ".join(f"`crucible models pull {model}`" for model in offered)
            return JobTypeStatus(
                ready=False,
                detail=f"{env.detail}; no image model is installed — {pulls or 'none is declared for this backend'}",
            )
        return JobTypeStatus(ready=True, detail=f"{env.detail}; installed: {installed}")

    def _resident_session(self) -> workers.WorkerSession | None:
        return self._residency.image_session

    def requirements(self, model_id: str, params: ImageParams) -> Needs:
        needs = self.loadable(model_id)
        refuse_what_the_arm_cannot_make(params, needs.spec, model_id)
        return needs

    def loadable(self, model_id: str) -> Needs:
        backend_kind = self._backend.kind
        manifest = MANIFESTS.known(model_id)
        spec = worker_type.require_block(manifest, model_id, backend_kind, "image model")
        worker_type.refuse_if_larger_than_host(
            self._backend, model_id, spec.memory_bytes_estimate
        )
        python = worker_type.require_worker_python(
            self._config, JOB_TYPE, backend_kind, model_id
        )
        return Needs(
            manifest,
            spec,
            python,
            worker_type.require_weights(self._config, manifest, spec, model_id),
        )

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        parsed = parse_params(ImageParams, params, self.name)
        self._admit(model, self.requirements(model, parsed), "generating an image with")

    def _admit(self, model: str, needs: Needs, doing: str) -> None:
        self._residency.refuse_if_claimed(f"{doing} {model!r}")
        if self._residency.is_resident(KIND_IMAGE, model):
            return
        self._guard(model, needs.spec.memory_bytes_estimate)

    def _worker(self, ctx: JobContext, model: str, needs: Needs) -> workers.WorkerSession:
        return self._session(
            ctx,
            model,
            needs.spec.memory_bytes_estimate,
            lambda: occupy_image(
                self._residency,
                needs.manifest,
                needs.spec,
                needs.weights_dir,
                needs.python,
                on_progress=ctx.warming,
            ),
        )

    def _generate(
        self, ctx: JobContext, model: str, session: workers.WorkerSession, request: dict[str, Any], steps: int
    ) -> workers.WorkerOutcome:
        generation = _Generation(ctx, steps)
        try:
            return session.send(
                request,
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
                on_progress=generation.progress,
                cancelled=lambda: ctx.cancelled,
                cancel_request={"op": "cancel", "request_id": request["request_id"]},
            )
        except workers.WorkerError as exc:
            self._forget(ctx, model)
            raise JobError("worker_failed", str(exc)) from None
        except JobCancelled:
            if not session.alive:
                self._forget(ctx, model)
            raise

    def run(self, job: Job, ctx: JobContext) -> None:
        params = ImageParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        needs = as_job_error(self.requirements, model, params)
        pictures = input_pictures(ctx, params)
        seed = params.seed if params.seed is not None else secrets.randbelow(MAX_SEED + 1)
        output = ctx.scratch / ARTIFACT_NAME
        session = self._worker(ctx, model, needs)
        self._make(ctx, model, session, params, needs, pictures, seed, output)

    def _make(
        self,
        ctx: JobContext,
        model: str,
        session: workers.WorkerSession,
        params: ImageParams,
        needs: Needs,
        pictures: Pictures,
        seed: int,
        output: Path,
    ) -> None:
        request = {
            "op": "generate",
            "request_id": uuid.uuid4().hex,
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "width": params.width,
            "height": params.height,
            "seed": seed,
            "steps": params.steps,
            "guidance": params.guidance,
            "image_path": None if pictures.image is None else str(pictures.image),
            "image_strength": params.image_strength,
            "mask_path": None if pictures.mask is None else str(pictures.mask),
            "mask_blur": params.effective_mask_blur,
            "output_path": str(output),
            "revision": needs.spec.revision,
            "backend": needs.spec.backend,
        }
        outcome = self._generate(ctx, model, session, request, params.steps)
        try:
            (result,) = workers.require_positional_results(outcome, 1, "image")
        except workers.WorkerError as exc:
            self._forget(ctx, model)
            raise JobError("worker_failed", str(exc)) from None
        refused = result.get("refused")
        if refused:
            raise JobError(str(refused["code"]), str(refused["message"]))
        ctx.artifact(ARTIFACT_NAME, output)
        generated = output.parent / GENERATED_NAME
        if pictures.mask is not None and generated.is_file():
            ctx.artifact(GENERATED_NAME, generated)
        ctx.progress(1.0, f"{result['width']}x{result['height']} image made", stage="done")
        ctx.done_extra(
            image=effective_params(params, seed, needs, result, pictures),
            resident=self._residency.resident_id,
        )


def effective_params(
    params: ImageParams, seed: int, needs: Needs, result: dict[str, Any], pictures: Pictures
) -> dict[str, Any]:
    return {
        "model": needs.manifest.id,
        "hf_repo": needs.spec.hf_repo,
        "revision": needs.spec.revision,
        "backend": needs.spec.backend,
        "engine": needs.spec.engine,
        "dtype": needs.spec.dtype,
        "prompt": params.prompt,
        "negative_prompt": params.negative_prompt,
        "width": result["width"],
        "height": result["height"],
        "seed": seed,
        "steps": params.steps,
        "guidance": params.guidance,
        "image_strength": params.image_strength,
        "input": None if pictures.image is None else pictures.image.name,
        "mask": params.mask,
        "mask_blur": params.effective_mask_blur,
        "mask_coverage": result.get("mask_coverage"),
        "mask_outside_drift": result.get("mask_outside_drift"),
        "mask_blend_steps": result.get("mask_blend_steps"),
        "seconds": result.get("seconds"),
        "stage_seconds": result.get("stage_seconds"),
        "peak_bytes": result.get("peak_bytes"),
        "stage_peak_bytes": result.get("stage_peak_bytes"),
        "memory_bytes_estimate": needs.spec.memory_bytes_estimate,
        "memory_basis": needs.spec.memory_basis,
        "prompt_cache": result.get("prompt_cache"),
    }


class LoadImageParams(BaseModel):
    """`params` for a load-image job: warm the image model up."""

    model_config = ConfigDict(extra="forbid")



class LoadImageJobType(ImageJobType):

    name = LOAD_IMAGE.name

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        parse_params(LoadImageParams, params, self.name)
        self._admit(model, self.loadable(model), "loading")

    def run(self, job: Job, ctx: JobContext) -> None:
        LoadImageParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        needs = as_job_error(self.loadable, model)
        ctx.progress(0.0, f"loading {model}")
        self._worker(ctx, model, needs)
        ctx.progress(1.0, f"{model} is resident")
        ctx.done_extra(resident=self._residency.resident_id)


class UnloadImageJobType(UnloadJobType):
    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        super().__init__(
            UNLOAD_IMAGE,
            residency,
            describe=lambda: _descriptors(config, residency),
            provenance=lambda model: MANIFESTS.provenance(config.backend_kind, model),
        )


JOB_TYPES: tuple[JobTypeBinding, ...] = (
    JobTypeBinding(
        IMAGE_JOB,
        lambda wiring: ImageJobType(
            wiring.config, wiring.backend, wiring.residency
        ),
    ),
    JobTypeBinding(
        UNLOAD_IMAGE,
        lambda wiring: UnloadImageJobType(wiring.config, wiring.backend, wiring.residency),
    ),
    JobTypeBinding(
        LOAD_IMAGE,
        lambda wiring: LoadImageJobType(
            wiring.config, wiring.backend, wiring.residency
        ),
    ),
)
