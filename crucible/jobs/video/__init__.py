from __future__ import annotations

import secrets
import subprocess
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, cast

from pydantic import BaseModel, ConfigDict

from ... import gpubusy, hosttools, jobenv, videodesktop, videoweights, weights, workers
from ...backend import MLX_DARWIN
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
    AUDIO_TO_VIDEO,
    IMAGE_AUDIO_TO_VIDEO,
    IMAGE_TO_VIDEO,
    LTX,
    LTX_2_MLX,
    TEXT_TO_VIDEO,
    VideoBackendSpec,
    VideoManifest,
    VideoManifestError,
    load_all_video_manifests,
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
from .params import MAX_SEED, Settled, VideoParams, refuse_mode, settle

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
    LTX_2_MLX: HERE / "ltx2mlx_worker.py",
}

WORKER_ENVIRONMENT = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

IMAGE_MAGIC: tuple[tuple[bytes, int, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", 0, "PNG"),
    (b"\xff\xd8\xff", 0, "JPEG"),
    (b"WEBP", 8, "WebP"),
)

AUDIO_MAGIC: tuple[tuple[bytes, int, str], ...] = (
    (b"WAVE", 8, "WAV"),
    (b"fLaC", 0, "FLAC"),
    (b"ID3", 0, "MP3"),
    (b"OggS", 0, "Ogg"),
)

AUDIO_FORMATS = ("WAV", "FLAC", "MP3", "Ogg")

SNIFF_BYTES = 16

MP4_MAGIC = (b"ftyp", 4)

FFPROBE_WHY = "measures a soundtrack's length with ffprobe, which ships beside ffmpeg"

SOUNDTRACK_PROBE_TIMEOUT_SECONDS = 120


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


@dataclass(frozen=True)
class Soundtrack:
    """A supplied soundtrack: the clip's sound, held fixed while the picture is made to it.
    `seconds` is None until it is measured, which happens only once the arm has said it
    takes one."""

    name: str
    path: Path
    format: str
    seconds: float | None = None


@dataclass(frozen=True)
class ClipInputs:
    picture: Path | None
    soundtrack: Soundtrack | None

    @property
    def mode(self) -> str:
        if self.soundtrack is None:
            return TEXT_TO_VIDEO if self.picture is None else IMAGE_TO_VIDEO
        return AUDIO_TO_VIDEO if self.picture is None else IMAGE_AUDIO_TO_VIDEO


def sniff(head: bytes) -> tuple[str, str] | None:
    """("picture" or "soundtrack", the format) for an input's first bytes, or None.

    By content, never by name: a WAV is RIFF....WAVE (a WebP is RIFF....WEBP), and an MP3
    without an ID3 tag starts at a frame sync (eleven set bits) with a layer other than the
    reserved 00, which a JPEG's FF D8 and an AAC ADTS header's layer 00 are not.
    """
    for magic, at, name in IMAGE_MAGIC:
        if head[at : at + len(magic)] == magic:
            return "picture", name
    for magic, at, name in AUDIO_MAGIC:
        if head[at : at + len(magic)] == magic and (name != "WAV" or head[:4] == b"RIFF"):
            return "soundtrack", name
    if len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0 and head[1] & 0x06:
        return "soundtrack", "MP3"
    return None


def clip_inputs(ctx: JobContext) -> ClipInputs:
    """What the job's inputs are: at most one picture (the first frame) and at most one
    soundtrack (the clip's sound), told apart by their bytes; input names are free."""
    found: dict[str, list[tuple[str, str]]] = {"picture": [], "soundtrack": []}
    paths: dict[str, Path] = {}
    for name, path in sorted(ctx.inputs().items()):
        with path.open("rb") as handle:
            head = handle.read(SNIFF_BYTES)
        kind = sniff(head)
        if kind is None:
            raise JobError(
                "invalid_inputs",
                f"input {name!r} is neither a {', '.join(n for _, _, n in IMAGE_MAGIC)} "
                f"picture nor a {', '.join(AUDIO_FORMATS)} soundtrack (its first bytes are "
                f"{head[:8].hex()}); a video job reads the picture its first frame starts "
                "from and the soundtrack it is made to",
            )
        found[kind[0]].append((name, kind[1]))
        paths[name] = path
    doubled = {kind: entries for kind, entries in found.items() if len(entries) > 1}
    if doubled:
        said = "; ".join(
            f"{len(entries)} {kind}s ({', '.join(f'{n!r}, {f}' for n, f in entries)})"
            for kind, entries in doubled.items()
        )
        raise JobError(
            "invalid_inputs",
            f"this job carries {said}; a video job reads at most one picture (its first "
            "frame) and at most one soundtrack (its sound)",
        )
    picture = paths[found["picture"][0][0]] if found["picture"] else None
    soundtrack = None
    if found["soundtrack"]:
        name, kind = found["soundtrack"][0]
        soundtrack = Soundtrack(name, paths[name], kind)
    return ClipInputs(picture, soundtrack)


def measured(soundtrack: Soundtrack) -> Soundtrack:
    return replace(
        soundtrack,
        seconds=soundtrack_seconds(soundtrack.path, soundtrack.name, soundtrack.format),
    )


def soundtrack_seconds(path: Path, name: str, kind: str) -> float:
    """The soundtrack's length as ffprobe reads it from the container."""
    ffprobe = hosttools.ffprobe_path()
    if ffprobe is None:
        raise ApiError(
            409,
            "ffmpeg_missing",
            f"there is no ffprobe on this server's PATH, and {JOB_TYPE} {FFPROBE_WHY} "
            + hosttools.searched_note(),
            {"path": hosttools.search_path(), "missing": ["ffprobe"]},
        )
    try:
        completed = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
             str(path)],
            capture_output=True,
            text=True,
            timeout=SOUNDTRACK_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise JobError(
            "invalid_inputs",
            f"ffprobe took more than {SOUNDTRACK_PROBE_TIMEOUT_SECONDS} s to read the length "
            f"of input {name!r} ({kind}); send a soundtrack it can read",
        ) from None
    said = completed.stdout.strip()
    try:
        seconds = float(said)
    except ValueError:
        seconds = float("nan")
    if completed.returncode != 0 or not seconds > 0:
        tail = (completed.stderr.strip() or said or "nothing")[-300:]
        raise JobError(
            "invalid_inputs",
            f"input {name!r} begins like a {kind} file but ffprobe cannot read its length "
            f"(exit {completed.returncode}: {tail}); send a soundtrack that plays",
        )
    return seconds


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
    transformer = spec.transformer_path(needs.weights_dir)
    session = workers.WorkerSession(
        python=python,
        script=script,
        log_path=log_path,
        environment={
            **workers.worker_environment(python.parent.parent),
            **workers.torch_allocator_environment(spec.backend),
            **WORKER_ENVIRONMENT,
            **desktop_environment(needs.desktop),
        },
    )
    outcome = session.start(
        {
            "op": "load",
            "engine": spec.engine,
            "model_dir": str(needs.weights_dir),
            "transformer_path": None if transformer is None else str(transformer),
            "hf_repo": spec.hf_repo,
            "device": spec.device,
            "dtype": spec.dtype,
            "mlx_cache_limit_bytes": spec.mlx_cache_limit_bytes,
            "desktop": needs.desktop,
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
    desktop: dict[str, Any] | None = None


class _Generation:
    def __init__(self, ctx: JobContext, sampler: gpubusy.GpuBusySampler | None = None) -> None:
        self._ctx = ctx
        self._sampler = sampler

    def progress(self, message: dict[str, Any]) -> None:
        stage = str(message.get("stage"))
        if stage == "cancelled":
            return
        if self._sampler is not None:
            self._sampler.mark(stage)
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


TRIAL_TABLE = "video_trial"

TRIAL_LIMITS = (
    "max_side",
    "max_pixels",
    "max_frames",
    "max_video_tokens",
    "max_video_tokens_image_to_video",
)


def trial_settings(config: Config) -> dict[str, Any]:
    """The operator's [video_trial] table, or {} when it is absent (the default).

    A measurement knob, not a setting: it lifts this machine's declared clip limits so
    an operator can step up and measure what the hardware really holds, the way the
    shipped limits were meant to be replaced by measured ones. Admission still books the
    declared estimate, so the operator owns the memory risk while it is set; remove the
    table when the measuring is done.
    """
    import tomllib

    try:
        document = tomllib.loads(Path(config.path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    table = document.get(TRIAL_TABLE)
    if not isinstance(table, dict):
        return {}
    found: dict[str, Any] = {}
    for key in TRIAL_LIMITS:
        value = table.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            found[key] = value
    silence = table.get("silence_timeout_s")
    if isinstance(silence, (int, float)) and not isinstance(silence, bool) and silence > 0:
        found["silence_timeout_s"] = float(silence)
    return found


def with_trial(spec: VideoBackendSpec, trial: dict[str, Any]) -> VideoBackendSpec:
    limits = {key: value for key, value in trial.items() if key in TRIAL_LIMITS}
    return replace(spec, **limits) if limits else spec


def tiled(spec: VideoBackendSpec, desktop: dict[str, Any] | None) -> VideoBackendSpec:
    """The longer clip limits a block declares for when the full-size pass is tiled.

    Tiling bounds the refine pass's memory by the tile, not the clip, so the clip can run
    to the length that was measured with it; with tiling off (`max_tile_tokens = 0` or the
    table disabled) the untiled limits stand, which the declared memory covers.
    """
    if spec.tiled_max_frames is None or spec.tiled_max_video_tokens is None:
        return spec
    if desktop is None or int(desktop.get("max_tile_tokens", 0)) <= 0:
        return spec
    return replace(
        spec, max_frames=spec.tiled_max_frames, max_video_tokens=spec.tiled_max_video_tokens
    )


def machine_spec(config: Config, spec: VideoBackendSpec) -> VideoBackendSpec:
    """The block as this machine runs it: tiled limits when its engine tiles, then any
    [video_trial] lift."""
    desktop = desktop_settings(config) if spec.engine == LTX_2_MLX else None
    return with_trial(tiled(spec, desktop), trial_settings(config))


# The table's keys, defaults and ranges belong to crucible/videodesktop.py, which the
# Settings door reads too; these names are the ones this module has always used.
DESKTOP_TABLE = videodesktop.TABLE
DESKTOP_DEFAULTS = videodesktop.DEFAULTS
DESKTOP_MAXIMUM = videodesktop.MAXIMUM
DESKTOP_MINIMUM = videodesktop.MINIMUM

DESKTOP_ENVIRONMENT: dict[str, str] = {
    "mlx_max_ops_per_buffer": "MLX_MAX_OPS_PER_BUFFER",
    "mlx_max_mb_per_buffer": "MLX_MAX_MB_PER_BUFFER",
    "dit_eval_every": "LTX2_DIT_EVAL_EVERY",
}


def desktop_settings(config: Config) -> dict[str, Any] | None:
    """The machine's [video_desktop] table with its defaults filled in, or None when absent.

    These knobs trade speed for a desktop that stays responsive while the Mac renders: they
    cut the GPU work into smaller pieces (smaller Metal command buffers, a sync after every
    transformer block, the full-size pass tiled so no one attention or feed-forward kernel
    spans the whole clip) and stream the transformer's blocks from disk instead of holding
    all 20.6 GB. They are not clip limits (that is [video_trial]). Only the Mac's engine
    reads them; a value of the wrong type or out of range is ignored, not guessed at.
    """
    import tomllib

    try:
        document = tomllib.loads(Path(config.path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return dict(DESKTOP_DEFAULTS)
    table = document.get(DESKTOP_TABLE)
    if not isinstance(table, dict):
        return dict(DESKTOP_DEFAULTS)
    if table.get("enabled") is False:
        return None
    found = dict(DESKTOP_DEFAULTS)
    for key in DESKTOP_MINIMUM:
        value = table.get(key)
        if videodesktop.int_in_range(key, value):
            found[key] = value
    if isinstance(table.get("low_ram"), bool):
        found["low_ram"] = table["low_ram"]
    return found


GPU_BUSY_TARGET_DEFAULTS: dict[str, float] = {
    MLX_DARWIN: videodesktop.GPU_BUSY_TARGET_DEFAULT,
}


def gpu_busy_target(config: Config, backend_kind: str) -> float | None:
    """The mean GPU busy (percent) a render on this machine should stay at or under.

    `[video_desktop] gpu_busy_target_pct` when set (1 to 100), else the backend's default:
    90 on the Mac, where the person's desktop draws on the same GPU, and none on the PC, which
    renders headless in WSL. It only judges a run (done.video.gpu_busy_target_met); what
    brings the number down is the table's `gpu_duty_pct`.
    """
    import tomllib

    try:
        document = tomllib.loads(Path(config.path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        document = {}
    table = document.get(DESKTOP_TABLE)
    if isinstance(table, dict):
        value = table.get(videodesktop.GPU_BUSY_TARGET)
        if videodesktop.busy_target_in_range(value):
            return float(value)
    return GPU_BUSY_TARGET_DEFAULTS.get(backend_kind)


def desktop_environment(settings: dict[str, Any] | None) -> dict[str, str]:
    """The env vars MLX (mlx/utils.h) and ltx-2-mlx (model/transformer/model.py) read once,
    at import, so they are set on the worker before it starts."""
    if settings is None:
        return {}
    return {name: str(settings[key]) for key, name in DESKTOP_ENVIRONMENT.items()}


class VideoJobType(ResidentWorker):

    name = JOB_TYPE
    resident_kind = KIND_VIDEO

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
        spec = machine_spec(self._config, _require_block(manifest, model_id, backend_kind))
        worker_type.refuse_if_larger_than_host(self._backend, model_id, spec.memory_bytes_estimate)
        python = require_video_python(self._config, spec, model_id)
        desktop = desktop_settings(self._config) if spec.engine == LTX_2_MLX else None
        return Needs(
            manifest, spec, python, require_video_weights(self._config, manifest, spec), desktop
        )

    def requirements(self, model_id: str, params: VideoParams) -> Needs:
        manifest = MANIFESTS.known(model_id)
        spec = machine_spec(
            self._config, _require_block(manifest, model_id, self._backend.kind)
        )
        settle(params, model_id, spec, 0)
        needs = self.loadable(model_id)
        hosttools.require_ffmpeg(JOB_TYPE, FFMPEG_WHY)
        return needs

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        parsed = parse_params(VideoParams, params, self.name)
        self._admit(model, self.requirements(model, parsed), "making video with")

    def _admit(self, model: str, needs: Needs, doing: str) -> None:
        self._residency.refuse_if_claimed(f"{doing} {model!r}")
        if self._residency.is_resident(KIND_VIDEO, model):
            return
        self._guard(model, needs.spec.memory_bytes_estimate)

    def _worker(self, ctx: JobContext, model: str, needs: Needs) -> workers.WorkerSession:
        return self._session(
            ctx,
            model,
            needs.spec.memory_bytes_estimate,
            lambda: occupy_video(self._residency, needs, on_progress=ctx.warming),
        )

    def _generate(
        self,
        ctx: JobContext,
        model: str,
        session: workers.WorkerSession,
        request: dict[str, Any],
        sampler: gpubusy.GpuBusySampler | None = None,
    ) -> dict[str, Any]:
        try:
            outcome = session.send(
                request,
                ready_silence_timeout=trial_settings(self._config).get(
                    "silence_timeout_s", READY_SILENCE_TIMEOUT_SECONDS
                ),
                on_progress=_Generation(ctx, sampler).progress,
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
        inputs = clip_inputs(ctx)
        as_job_error(refuse_mode, params, inputs.mode, model, needs.spec)
        if inputs.soundtrack is not None:
            inputs = replace(inputs, soundtrack=as_job_error(measured, inputs.soundtrack))
        seed = params.seed if params.seed is not None else secrets.randbelow(MAX_SEED + 1)
        settled = as_job_error(
            settle, params, model, needs.spec, seed, inputs.mode,
            None if inputs.soundtrack is None else inputs.soundtrack.seconds,
        )
        ffmpeg = as_job_error(hosttools.require_ffmpeg, JOB_TYPE, FFMPEG_WHY)
        session = self._worker(ctx, model, needs)
        self._make(ctx, model, session, params, needs, settled, inputs, ffmpeg)

    def _make(
        self,
        ctx: JobContext,
        model: str,
        session: workers.WorkerSession,
        params: VideoParams,
        needs: Needs,
        settled: Settled,
        inputs: ClipInputs,
        ffmpeg: str,
    ) -> None:
        output = ctx.scratch / ARTIFACT_NAME
        request = generate_request(params, needs, settled, inputs, output, ffmpeg)
        sampler = gpubusy.start(self._backend.kind)
        try:
            result = self._generate(ctx, model, session, request, sampler)
        finally:
            busy = sampler.stop()
        target = gpu_busy_target(self._config, self._backend.kind)
        busy = {
            **busy,
            "gpu_busy_target_pct": target,
            "gpu_busy_target_met": gpubusy.verdict(busy, target),
        }
        check_video_file(output)
        ctx.artifact(ARTIFACT_NAME, output)
        ctx.progress(
            1.0,
            f"{result.get('duration_s')} s of {result.get('width')}x{result.get('height')} video made",
            stage="done",
        )
        ctx.done_extra(
            video={**effective_params(params, needs, settled, result, inputs), **busy},
            resident=self._residency.resident_id,
        )


def generate_request(
    params: VideoParams,
    needs: Needs,
    settled: Settled,
    inputs: ClipInputs,
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
        "refine_steps": needs.spec.refine_steps,
        "audio": params.audio,
        "image_path": None if inputs.picture is None else str(inputs.picture),
        "soundtrack_path": None if inputs.soundtrack is None else str(inputs.soundtrack.path),
        "output_path": str(output),
        "ffmpeg": ffmpeg,
        "revision": needs.spec.revision,
        "backend": needs.spec.backend,
    }


def soundtrack_report(
    inputs: ClipInputs, settled: Settled, result: dict[str, Any]
) -> dict[str, Any] | None:
    """What the clip did with a supplied soundtrack: all of it, or its first `used_s`
    seconds when it is longer than the clip (said here, never dropped silently)."""
    soundtrack = inputs.soundtrack
    if soundtrack is None:
        return None
    total = round(cast(float, soundtrack.seconds), 3)
    used = settled.duration_s
    return {
        "input": soundtrack.name,
        "format": soundtrack.format,
        "total_s": total,
        "used_s": used,
        "unused_s": round(max(0.0, total - used), 3),
        "covers": "the whole soundtrack" if used >= total else f"its first {used} s",
        "sound": "supplied",
        "sample_rate": result.get("audio_sample_rate"),
        "channels": result.get("audio_channels"),
        "audio_tokens": result["soundtrack"]["audio_tokens"],
    }


def effective_params(
    params: VideoParams,
    needs: Needs,
    settled: Settled,
    result: dict[str, Any],
    inputs: ClipInputs,
) -> dict[str, Any]:
    spec = needs.spec
    transformer = spec.transformer_companion
    return {
        "model": needs.manifest.id,
        "hf_repo": spec.hf_repo,
        "revision": spec.revision,
        "transformer": None if transformer is None else {
            "hf_repo": transformer.hf_repo,
            "revision": transformer.revision,
            "file": transformer.files[0].target,
            "sha256": transformer.files[0].sha256,
        },
        "backend": spec.backend,
        "engine": spec.engine,
        "dtype": spec.dtype,
        "quantization": result.get("quantization"),
        "sampling": result.get("sampling"),
        "desktop": result.get("desktop"),
        "mode": settled.mode,
        "prompt": params.prompt,
        "input": None if inputs.picture is None else inputs.picture.name,
        "width": result.get("width", settled.width),
        "height": result.get("height", settled.height),
        "num_frames": result.get("num_frames", settled.num_frames),
        "fps": settled.fps,
        "duration_s": result.get("duration_s", settled.duration_s),
        "video_tokens": settled.video_tokens,
        "seed": settled.seed,
        "steps": settled.steps,
        "refine_steps": spec.refine_steps,
        "audio": params.audio,
        "audio_seconds": result.get("audio_seconds"),
        "audio_sample_rate": result.get("audio_sample_rate"),
        "audio_channels": result.get("audio_channels"),
        "soundtrack": soundtrack_report(inputs, settled, result),
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
    """`params` for a load-video job: warm the video model up."""

    model_config = ConfigDict(extra="forbid")



class LoadVideoJobType(VideoJobType):

    name = LOAD_VIDEO.name

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        parse_params(LoadVideoParams, params, self.name)
        self._admit(model, self.loadable(model), "loading")

    def run(self, job: Job, ctx: JobContext) -> None:
        LoadVideoParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        needs = as_job_error(self.loadable, model)
        ctx.progress(0.0, f"loading {model}")
        self._worker(ctx, model, needs)
        ctx.progress(1.0, f"{model} is resident")
        ctx.done_extra(resident=self._residency.resident_id)


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
        lambda wiring: VideoJobType(wiring.config, wiring.backend, wiring.residency),
    ),
    JobTypeBinding(
        UNLOAD_VIDEO,
        lambda wiring: UnloadVideoJobType(wiring.config, wiring.backend, wiring.residency),
    ),
    JobTypeBinding(
        LOAD_VIDEO,
        lambda wiring: LoadVideoJobType(wiring.config, wiring.backend, wiring.residency),
    ),
)
