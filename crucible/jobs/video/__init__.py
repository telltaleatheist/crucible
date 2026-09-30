from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from pydantic import BaseModel, ConfigDict

from ... import hosttools, jobenv, videoweights, weights, workers
from ...cardkinds import KIND_VIDEO
from ...clock import utcnow
from ...config import Config
from ...errors import ApiError, JobCancelled, JobError
from ...jobtypes import LOAD_VIDEO, UNLOAD_VIDEO, VIDEO_JOB
from ...manifests import fingerprint
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    Occupant,
    Residency,
    ResidentVideo,
    say_to,
)
from ...videomodels import (
    IMAGE_TO_VIDEO,
    LTX,
    TEXT_TO_VIDEO,
    VideoBackendSpec,
    VideoManifest,
    VideoManifestError,
    load_all_video_manifests,
)
from .. import worker_type
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from ..binding import JobTypeBinding
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
from .params import LEASE_ACT, MAX_SEED, Settled, VideoParams, settle

__all__ = [
    "JOB_TYPES",
    "LoadVideoJobType",
    "LoadVideoParams",
    "UnloadVideoJobType",
    "VideoJobType",
    "VideoParams",
    "occupy_video",
]

JOB_TYPE = VIDEO_JOB.name

ARTIFACT_NAME = "video.mp4"

READY_SILENCE_TIMEOUT_SECONDS = 1800.0

FFMPEG_WHY = "muxes every clip's frames and sound into video.mp4 through it"

HERE = Path(__file__).resolve().parent

WORKER_SCRIPTS: dict[str, Path] = {
    LTX: HERE / "ltx_worker.py",
}

WORKER_ENVIRONMENT = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

IMAGE_MAGIC: tuple[tuple[bytes, int, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", 0, "PNG"),
    (b"\xff\xd8\xff", 0, "JPEG"),
    (b"WEBP", 8, "WebP"),
)

MP4_MAGIC = (b"ftyp", 4)


def worker_script(engine: str) -> Path:
    found = WORKER_SCRIPTS.get(engine)
    if found is None:
        raise JobError(
            "backend_unsupported",
            f"there is no video worker for engine {engine!r}; this build runs "
            f"{sorted(WORKER_SCRIPTS)}",
        )
    return found


def require_video_python(config: Config, spec: VideoBackendSpec, model_id: str) -> Path:
    try:
        env = jobenv.video_env(spec.engine, spec.backend)
        return jobenv.require_env(config.home, env, spec.backend)
    except jobenv.EnvError as exc:
        raise ApiError(
            409,
            "env_missing",
            f"cannot run {model_id!r}: {exc}. `crucible install video` builds it",
            {"model": model_id, "env": str(config.home / "envs" / f"video-{spec.engine}")},
        ) from None


def require_video_weights(
    config: Config, manifest: VideoManifest, spec: VideoBackendSpec
) -> Path:
    try:
        return videoweights.require_installed(config, manifest, spec).path
    except weights.WeightsError as exc:
        details = {"model": manifest.id, "hf_repo": spec.hf_repo, "revision": spec.revision}
        if spec.gated and weights.hf_token(config) is None:
            raise ApiError(
                409,
                "model_gated",
                weights.gated_message(spec.hf_repo, config, manifest.pull_command),
                {
                    **details,
                    "accept_url": weights.HF_ACCEPT_URL.format(repo=spec.hf_repo),
                    "retry": manifest.pull_command,
                },
            ) from None
        raise ApiError(409, "model_not_installed", str(exc), details) from None


def start_image(ctx: JobContext) -> Path | None:
    """The one start image of an image-to-video job, or None for text-to-video."""
    inputs = ctx.inputs()
    if not inputs:
        return None
    if len(inputs) != 1:
        raise JobError(
            "invalid_inputs",
            f"this job carries {len(inputs)} inputs ({sorted(inputs)}); a video job "
            "reads at most one, the picture its first frame starts from",
        )
    path = next(iter(inputs.values()))
    with path.open("rb") as handle:
        head = handle.read(16)
    if not any(head[at : at + len(magic)] == magic for magic, at, _ in IMAGE_MAGIC):
        raise JobError(
            "invalid_inputs",
            f"input {path.name!r} is not a "
            f"{', '.join(name for _, _, name in IMAGE_MAGIC)} image (its first bytes "
            f"are {head[:8].hex()}); send the picture the clip starts from",
        )
    return path


def check_video_file(path: Path) -> None:
    magic, at = MP4_MAGIC
    head = path.read_bytes()[: at + len(magic)] if path.is_file() else b""
    if head[at : at + len(magic)] != magic:
        raise JobError(
            "worker_failed",
            f"the video worker wrote {path} but it is not an MP4 (it begins "
            f"{head[:12].hex() or 'with nothing'}); the worker log says why",
        )


def start_video_session(
    python: Path,
    needs: "Needs",
    log_path: Path,
    *,
    ready_silence_timeout: float,
    on_ready: Callable[[dict[str, Any]], None] | None = None,
) -> workers.WorkerSession:
    spec = needs.spec
    script = worker_script(spec.engine)
    session = workers.WorkerSession(
        python=python,
        script=script,
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
            "model_dir": str(needs.weights_dir),
            "transformer_path": str(spec.transformer_path(needs.weights_dir)),
            "hf_repo": spec.hf_repo,
            "device": spec.device,
            "dtype": spec.dtype,
            "memory_cap_bytes": workers.torch_memory_cap(
                spec.backend, spec.memory_bytes_estimate
            ),
            "memory_budget_bytes": spec.memory_bytes_estimate,
        },
        ready_silence_timeout=ready_silence_timeout,
        on_ready=on_ready,
    )
    if outcome.results:
        session.stop()
        raise workers.WorkerError(
            f"{script.name} answered a load request with "
            f"{len(outcome.results)} result(s); a load produces none"
        )
    return session


def occupy_video(
    residency: Residency,
    needs: "Needs",
    *,
    timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
    on_progress: Callable[[str], None] | None = None,
) -> ResidentVideo:
    say = say_to(on_progress)
    manifest, spec = needs.manifest, needs.spec
    loaded: dict[str, Any] = {}

    def start() -> Occupant:
        log_path = residency.log_path_for(manifest.id)
        say(f"starting the {spec.engine} worker for {manifest.id}; log {log_path}")
        session = start_video_session(
            needs.python, needs, log_path, ready_silence_timeout=timeout, on_ready=loaded.update
        )
        resident = ResidentVideo(
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

    return cast(ResidentVideo, residency.occupy(KIND_VIDEO, manifest.id, start, say=say))


MANIFESTS: ManifestCatalog[VideoManifest] = ManifestCatalog(
    lambda: load_all_video_manifests(),
    VideoManifestError,
    unreadable_code="video_manifests_unreadable",
    what="video manifests",
    unknown="video model",
)


def _descriptors(config: Config, residency: Residency) -> list[ModelDescriptor]:
    return MANIFESTS.descriptors(
        config.backend_kind,
        installed=lambda manifest, spec: videoweights.installed(config, manifest, spec)
        is not None,
        resident=lambda model_id: residency.is_resident(KIND_VIDEO, model_id),
    )


@dataclass(frozen=True)
class Needs:
    manifest: VideoManifest
    spec: VideoBackendSpec
    python: Path
    weights_dir: Path


class _Generation:
    def __init__(self, ctx: JobContext) -> None:
        self._ctx = ctx

    def progress(self, message: dict[str, Any]) -> None:
        stage = str(message.get("stage"))
        if stage == "cancelled":
            return
        step, steps = message.get("step"), message.get("steps")
        fraction = min(1.0, max(0.0, float(message.get("fraction") or 0.0)))
        words = f"{stage}: step {step} of {steps}" if steps else stage.replace("_", " ")
        self._ctx.progress(fraction, words, stage=stage, step=step, steps=steps)


def _env_status(config: Config, backend_kind: str) -> JobTypeStatus | None:
    try:
        specs = jobenv.video_envs(backend_kind)
    except jobenv.EnvError as exc:
        return JobTypeStatus(ready=False, detail=str(exc))
    for spec in specs:
        status = jobenv.env_status(config.home, spec, backend_kind)
        if not status.installed:
            return JobTypeStatus(
                ready=False, detail=f"{status.detail}; `crucible install video` builds it"
            )
    return None


def _require_block(manifest: VideoManifest, model_id: str, backend_kind: str) -> VideoBackendSpec:
    return worker_type.require_block(manifest, model_id, backend_kind, "video model")


class VideoJobType(ResidentWorker):

    name = JOB_TYPE
    resident_kind = KIND_VIDEO

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
        broken = _env_status(self._config, backend.kind)
        if broken is not None:
            return broken
        if hosttools.ffmpeg_path() is None:
            return JobTypeStatus(
                ready=False,
                detail="video env built; but there is no ffmpeg on PATH, and video "
                + FFMPEG_WHY + ". " + hosttools.searched_note(),
            )
        try:
            manifests = MANIFESTS.all()
        except ApiError as exc:
            return JobTypeStatus(ready=False, detail=exc.message)
        offered = sorted(m.id for m in manifests.values() if m.supports(backend.kind))
        installed = [
            model_id
            for model_id in offered
            if videoweights.installed(
                self._config, manifests[model_id], manifests[model_id].spec(backend.kind)
            )
            is not None
        ]
        if not installed:
            pulls = " or ".join(f"`crucible models pull {model}`" for model in offered)
            return JobTypeStatus(
                ready=False,
                detail=f"video env built; no video model is installed — {pulls or 'none is declared for this backend'}",
            )
        return JobTypeStatus(ready=True, detail=f"video env built; installed: {installed}")

    def _resident_session(self) -> workers.WorkerSession | None:
        return self._residency.video_session

    def loadable(self, model_id: str) -> Needs:
        backend_kind = self._backend.kind
        manifest = MANIFESTS.known(model_id)
        spec = _require_block(manifest, model_id, backend_kind)
        worker_type.refuse_if_larger_than_host(self._backend, model_id, spec.memory_bytes_estimate)
        python = require_video_python(self._config, spec, model_id)
        return Needs(manifest, spec, python, require_video_weights(self._config, manifest, spec))

    def requirements(self, model_id: str, params: VideoParams) -> Needs:
        manifest = MANIFESTS.known(model_id)
        spec = _require_block(manifest, model_id, self._backend.kind)
        settle(params, model_id, spec, 0)
        needs = self.loadable(model_id)
        hosttools.require_ffmpeg(JOB_TYPE, FFMPEG_WHY)
        return needs

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        parsed = parse_params(VideoParams, params, self.name)
        self._admit(model, self.requirements(model, parsed), parsed.lease, "making video with")

    def _admit(self, model: str, needs: Needs, lease: LeaseOnLoad | None, doing: str) -> None:
        if lease is not None:
            require_lease_request(lease, act=LEASE_ACT)
        self._residency.refuse_if_claimed(f"{doing} {model!r}")
        if self._residency.is_resident(KIND_VIDEO, model):
            return
        self._guard(model, needs.spec.memory_bytes_estimate)

    def _hold(self, job: Job, model: str, lease: LeaseOnLoad | None) -> HeldForLoad | None:
        if lease is None:
            return None
        return hold_for_load(
            self._leases, kind=KIND_VIDEO, subject=model, request=lease, client=job.client
        )

    def _worker(self, ctx: JobContext, model: str, needs: Needs) -> workers.WorkerSession:
        return self._session(
            ctx,
            model,
            needs.spec.memory_bytes_estimate,
            lambda: occupy_video(self._residency, needs, on_progress=ctx.warming),
        )

    def _generate(
        self, ctx: JobContext, model: str, session: workers.WorkerSession, request: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            outcome = session.send(
                request,
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
                on_progress=_Generation(ctx).progress,
                cancelled=lambda: ctx.cancelled,
                cancel_request={"op": "cancel", "request_id": request["request_id"]},
            )
            (result,) = workers.require_positional_results(outcome, 1, "video")
            return result
        except workers.WorkerError as exc:
            self._forget(ctx, model)
            raise JobError("worker_failed", str(exc)) from None
        except JobCancelled:
            if not session.alive:
                self._forget(ctx, model)
            raise

    def run(self, job: Job, ctx: JobContext) -> None:
        params = VideoParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        needs = as_job_error(self.requirements, model, params)
        source = start_image(ctx)
        mode = TEXT_TO_VIDEO if source is None else IMAGE_TO_VIDEO
        seed = params.seed if params.seed is not None else secrets.randbelow(MAX_SEED + 1)
        settled = as_job_error(settle, params, model, needs.spec, seed, mode)
        ffmpeg = as_job_error(hosttools.require_ffmpeg, JOB_TYPE, FFMPEG_WHY)
        session = self._worker(ctx, model, needs)
        held = self._hold(job, model, params.lease)
        try:
            self._make(ctx, model, session, params, needs, settled, source, ffmpeg)
        except BaseException:
            let_go_of(self._leases, held)
            raise
        ctx.done_extra(lease_id=None if held is None else held.lease_id)

    def _make(
        self,
        ctx: JobContext,
        model: str,
        session: workers.WorkerSession,
        params: VideoParams,
        needs: Needs,
        settled: Settled,
        source: Path | None,
        ffmpeg: str,
    ) -> None:
        output = ctx.scratch / ARTIFACT_NAME
        request = generate_request(params, needs, settled, source, output, ffmpeg)
        result = self._generate(ctx, model, session, request)
        check_video_file(output)
        ctx.artifact(ARTIFACT_NAME, output)
        ctx.progress(
            1.0,
            f"{result.get('duration_s')} s of {result.get('width')}x{result.get('height')} video made",
            stage="done",
        )
        ctx.done_extra(
            video=effective_params(params, needs, settled, result, source),
            resident=self._residency.resident_id,
        )


def generate_request(
    params: VideoParams,
    needs: Needs,
    settled: Settled,
    source: Path | None,
    output: Path,
    ffmpeg: str,
) -> dict[str, Any]:
    return {
        "op": "generate",
        "request_id": uuid.uuid4().hex,
        "mode": settled.mode,
        "prompt": params.prompt,
        "width": settled.width,
        "height": settled.height,
        "num_frames": settled.num_frames,
        "fps": settled.fps,
        "seed": settled.seed,
        "steps": settled.steps,
        "audio": params.audio,
        "image_path": None if source is None else str(source),
        "output_path": str(output),
        "ffmpeg": ffmpeg,
        "revision": needs.spec.revision,
        "backend": needs.spec.backend,
    }


def effective_params(
    params: VideoParams,
    needs: Needs,
    settled: Settled,
    result: dict[str, Any],
    source: Path | None,
) -> dict[str, Any]:
    spec = needs.spec
    transformer = spec.transformer_companion
    return {
        "model": needs.manifest.id,
        "hf_repo": spec.hf_repo,
        "revision": spec.revision,
        "transformer": {
            "hf_repo": transformer.hf_repo,
            "revision": transformer.revision,
            "file": transformer.files[0].target,
            "sha256": transformer.files[0].sha256,
        },
        "backend": spec.backend,
        "engine": spec.engine,
        "dtype": spec.dtype,
        "quantization": result.get("quantization"),
        "mode": settled.mode,
        "prompt": params.prompt,
        "input": None if source is None else source.name,
        "width": result.get("width", settled.width),
        "height": result.get("height", settled.height),
        "num_frames": result.get("num_frames", settled.num_frames),
        "fps": settled.fps,
        "duration_s": result.get("duration_s", settled.duration_s),
        "video_tokens": settled.video_tokens,
        "seed": settled.seed,
        "steps": settled.steps,
        "audio": params.audio,
        "audio_seconds": result.get("audio_seconds"),
        "audio_sample_rate": result.get("audio_sample_rate"),
        "audio_channels": result.get("audio_channels"),
        "artifact": ARTIFACT_NAME,
        "bytes": result.get("bytes"),
        "encoder": result.get("encoder"),
        "seconds": result.get("seconds"),
        "stage_seconds": result.get("stage_seconds"),
        "peak_bytes": result.get("peak_bytes"),
        "stage_peak_bytes": result.get("stage_peak_bytes"),
        "memory_bytes_estimate": spec.memory_bytes_estimate,
        "memory_basis": spec.memory_basis,
        "stage_memory_bytes": dict(spec.stage_memory_bytes),
        "prompt_cache": result.get("prompt_cache"),
        "versions": result.get("versions"),
    }


class LoadVideoParams(BaseModel):
    """`params` for a load-video job: warm the video model up, optionally leased."""

    model_config = ConfigDict(extra="forbid")

    lease: LeaseOnLoad | None = None


class LoadVideoJobType(VideoJobType):

    name = LOAD_VIDEO.name

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        parsed = parse_params(LoadVideoParams, params, self.name)
        self._admit(model, self.loadable(model), parsed.lease, "loading")

    def run(self, job: Job, ctx: JobContext) -> None:
        params = LoadVideoParams.model_validate(job.params)
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


class UnloadVideoJobType(UnloadJobType):
    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        super().__init__(
            UNLOAD_VIDEO,
            residency,
            describe=lambda: _descriptors(config, residency),
            provenance=lambda model: MANIFESTS.provenance(config.backend_kind, model),
        )


JOB_TYPES: tuple[JobTypeBinding, ...] = (
    JobTypeBinding(
        VIDEO_JOB,
        lambda wiring: VideoJobType(wiring.config, wiring.backend, wiring.residency, wiring.leases),
    ),
    JobTypeBinding(
        UNLOAD_VIDEO,
        lambda wiring: UnloadVideoJobType(wiring.config, wiring.backend, wiring.residency),
    ),
    JobTypeBinding(
        LOAD_VIDEO,
        lambda wiring: LoadVideoJobType(wiring.config, wiring.backend, wiring.residency, wiring.leases),
    ),
)
