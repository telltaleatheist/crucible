from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, cast

from pydantic import BaseModel, ConfigDict

from ... import workers
from ...backend import CUDA_LINUX
from ...clock import utcnow
from ...config import Config
from ...denoisemodels import (
    PULL_COMMAND,
    DenoiseBackendSpec,
    DenoiseManifest,
    DenoiseManifestError,
    denoise_models_root,
    load_all_denoise_manifests,
)
from ...denoisemodels import installed as model_installed
from ...denoisemodels import missing as missing_model_files
from ...errors import ApiError, JobError
from ...jobtypes import DENOISE_JOB, RVC_ENV, UNLOAD_DENOISER
from ...manifests import fingerprint
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    KIND_DENOISE,
    Occupant,
    Residency,
    ResidentSeparator,
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
from ..unload import UnloadJobType, UnloadParams

__all__ = [
    "JOB_TYPES",
    "DenoiseJobType",
    "DenoiseParams",
    "UnloadDenoiserJobType",
    "denoise_models_dir",
    "denoise_models_dir_for",
    "occupy_separator",
]

JOB_TYPE = DENOISE_JOB.name

ENV_JOB_TYPE = RVC_ENV.name

UnloadDenoiserParams = UnloadParams

NO_PARAMS = (
    "denoise takes no params — every separation knob is an engine default this "
    "server does not put on the wire (docs/internals/asr-and-align.md)"
)

OUTPUT_FORMAT = "WAV"

READY_SILENCE_TIMEOUT_SECONDS = 900.0

ENGINE_ENVIRONMENT: dict[str, str] = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}

WORKER_SCRIPT = Path(__file__).resolve().parent / "worker.py"


def denoise_models_dir_for(home: Path) -> Path:
    return denoise_models_root(home)


def denoise_models_dir(config: Config) -> Path:
    return denoise_models_dir_for(config.home)


def occupy_separator(
    residency: Residency,
    manifest: DenoiseManifest,
    spec: DenoiseBackendSpec,
    model_file_dir: Path,
    python: Path,
    script: Path,
    *,
    use_autocast: bool,
    environment: dict[str, str],
    timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
    on_progress: Callable[[str], None] | None = None,
) -> ResidentSeparator:
    say = say_to(on_progress)

    def start() -> Occupant:
        log_path = residency.log_path_for(manifest.id)
        session = workers.WorkerSession(
            python=python,
            script=script,
            log_path=log_path,
            environment={
                **environment,
                **workers.torch_allocator_environment(spec.backend),
            },
        )
        say(
            f"loading {manifest.id} ({manifest.model_filename}) with "
            f"use_autocast={use_autocast}; log {log_path}"
        )
        outcome = session.start(
            {
                "op": "load",
                "model_file_dir": str(model_file_dir),
                "model_filename": manifest.model_filename,
                "use_autocast": use_autocast,
                "memory_cap_bytes": workers.torch_memory_cap(
                    spec.backend, spec.memory_bytes_estimate
                ),
            },
            ready_silence_timeout=timeout,
            on_ready=lambda message: say(
                f"{manifest.id} loaded in {message['seconds']:.1f}s"
            ),
            on_progress=lambda message: say(str(message["message"])),
        )
        if outcome.results:
            session.stop()
            raise workers.WorkerError(
                f"{script.name} answered a load request with "
                f"{len(outcome.results)} result(s); a load produces none"
            )
        resident = ResidentSeparator(
            separator_id=manifest.id,
            backend=spec.backend,
            model_filename=manifest.model_filename,
            revision=spec.revision,
            fingerprint=fingerprint(manifest.id, spec.revision),
            sample_rate=manifest.sample_rate,
            use_autocast=use_autocast,
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=utcnow(),
        )
        return Occupant(resident, session=session)

    return cast(
        ResidentSeparator,
        residency.occupy(KIND_DENOISE, manifest.id, start, say=say),
    )


class DenoiseParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


MANIFESTS: ManifestCatalog[DenoiseManifest] = ManifestCatalog(
    lambda: load_all_denoise_manifests(),
    DenoiseManifestError,
    unreadable_code="denoise_manifests_unreadable",
    what="denoise manifests",
    unknown="denoise manifest for",
)


def _descriptors(config: Config, residency: Residency) -> list[ModelDescriptor]:
    return MANIFESTS.descriptors(
        config.backend_kind,
        installed=lambda manifest, spec: model_installed(config.home, manifest, spec)
        is not None,
        resident=lambda model_id: residency.is_resident(KIND_DENOISE, model_id),
        source=lambda spec: f"{spec.hf_repo}:{spec.model_path}",
    )


def _missing_files(config: Config, manifest: DenoiseManifest) -> list[str]:
    return missing_model_files(config.home, manifest)


def _require_model_files(config: Config, manifest: DenoiseManifest) -> Path:
    root = denoise_models_dir(config)
    missing = _missing_files(config, manifest)
    if not missing:
        return root
    spec = manifest.backends.get(config.backend_kind)
    where = (
        f"{spec.hf_repo}@{spec.revision[:12]}" if spec is not None else "its upstream"
    )
    paths = (
        [spec.model_path, spec.config_path] if spec is not None else []
    )
    command = f"{PULL_COMMAND} {manifest.id}"
    raise ApiError(
        409,
        "denoise_model_missing",
        f"audio-separator needs {manifest.model_filename!r} and "
        f"{manifest.config_filename!r} in {root}, and {missing} are not there. "
        f"Crucible does not let the library fetch them: its own downloader "
        f"pulls from a GitHub release, which is not a source this server takes "
        f"weights from. The bytes are {where}, at {paths} — run "
        f"`{command}` to place them and try again",
        {
            "root": str(root),
            "missing": sorted(missing),
            "command": command,
            "hf_repo": None if spec is None else spec.hf_repo,
            "revision": None if spec is None else spec.revision,
            "paths": paths,
        },
    )


class DenoiseJobType(ResidentWorker):
    name = JOB_TYPE
    resident_kind = KIND_DENOISE

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency
        self._loaded_seconds: dict[str, float] = {}

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
        env = worker_type.env_or_status(
            self._config,
            ENV_JOB_TYPE,
            backend.kind,
            missing_note=" (denoise shares the rvc env — `crucible install rvc`)",
        )
        if isinstance(env, JobTypeStatus):
            return env
        try:
            manifests = MANIFESTS.all()
        except ApiError as exc:
            return JobTypeStatus(ready=False, detail=exc.message)
        installed = [
            manifest.id
            for manifest in manifests.values()
            if manifest.supports(backend.kind) and not _missing_files(
                self._config, manifest
            )
        ]
        if not installed:
            root = denoise_models_dir(self._config)
            pullable = sorted(
                manifest.id
                for manifest in manifests.values()
                if manifest.supports(backend.kind)
            )
            return JobTypeStatus(
                ready=False,
                detail=(
                    f"{env.detail}; but no separator checkpoint is in {root} — "
                    f"`{PULL_COMMAND} <id>` fetches one, and this build ships "
                    f"{pullable}"
                ),
                awaiting_weights=True,
            )
        return JobTypeStatus(
            ready=True, detail=f"{env.detail}; installed: {installed}"
        )


    def _resident_session(self) -> workers.WorkerSession | None:
        return self._residency.separator_session

    def requirements(
        self, model_id: str
    ) -> tuple[DenoiseManifest, DenoiseBackendSpec, Path, Path]:
        backend_kind = self._backend.kind
        manifest = MANIFESTS.known(model_id)
        spec = worker_type.require_block(manifest, model_id, backend_kind, "denoise model")
        worker_type.refuse_if_larger_than_host(
            self._backend, model_id, spec.memory_bytes_estimate
        )
        python = worker_type.require_worker_python(
            self._config, ENV_JOB_TYPE, backend_kind, model_id, note=" (denoise shares the rvc env)"
        )
        root = _require_model_files(self._config, manifest)
        return manifest, spec, python, root

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        parse_params(DenoiseParams, params, self.name, lead=NO_PARAMS)
        _manifest, spec, _python, _root = self.requirements(model)
        self._residency.refuse_if_claimed(f"denoising with {model!r}")
        if self._residency.is_resident(KIND_DENOISE, model):
            return
        self._guard(model, spec.memory_bytes_estimate)


    def run(self, job: Job, ctx: JobContext) -> None:
        DenoiseParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        manifest, spec, python, root = as_job_error(self.requirements, model)

        source = self._input(ctx)
        output_dir = ctx.scratch / "stems"
        session = self._session(
            ctx,
            model,
            spec.memory_bytes_estimate,
            lambda: self._load(ctx, manifest, spec, root, python),
        )

        request = {
            "op": "separate",
            "input": str(source),
            "output_dir": str(output_dir),
            "output_format": OUTPUT_FORMAT,
            "sample_rate": manifest.sample_rate,
        }

        def on_ready(message: dict[str, Any]) -> None:
            ctx.warming(
                f"{source.name}: {message['seconds']}s of "
                f"{message['channels']}-channel audio at {message['sample_rate']} Hz, "
                f"through {manifest.display}"
            )

        def on_progress(message: dict[str, Any]) -> None:
            elapsed = message.get("elapsed_s")
            ctx.progress(
                0.0,
                f"separating {source.name} through {manifest.model_filename}"
                + (f", {float(elapsed):.0f}s so far" if elapsed is not None else ""),
                stage=message["stage"],
            )

        try:
            outcome = session.send(
                request,
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
                on_ready=on_ready,
                on_progress=on_progress,
                cancelled=lambda: ctx.cancelled,
            )
            results = workers.require_positional_results(outcome, 1, "input")
        except workers.WorkerError as exc:
            self._forget(ctx, model)
            raise JobError("worker_failed", str(exc)) from None

        stems = results[0]["stems"]
        primary = self._check(manifest, outcome.ready, stems)
        ctx.artifact(primary["name"], output_dir / primary["name"])
        ctx.progress(
            1.0,
            f"{len(stems)} stem(s) from {source.name} through {manifest.display}",
            stage="separating",
        )
        ctx.done_extra(
            primary_stem=primary["name"],
            stems=[stem["name"] for stem in stems],
            sample_rate=primary["sample_rate"],
            frames=primary["frames"],
            separate_seconds=results[0]["separate_seconds"],
            load_seconds=self._loaded_seconds.pop(job.id, 0.0),
            resident=self._residency.resident_id,
        )


    def _load(
        self,
        ctx: JobContext,
        manifest: DenoiseManifest,
        spec: DenoiseBackendSpec,
        model_file_dir: Path,
        python: Path,
    ) -> None:
        began = time.perf_counter()
        occupy_separator(
            self._residency,
            manifest,
            spec,
            model_file_dir,
            python,
            WORKER_SCRIPT,
            use_autocast=self._backend.kind == CUDA_LINUX,
            environment=dict(ENGINE_ENVIRONMENT),
            timeout=DEFAULT_READY_TIMEOUT_SECONDS,
            on_progress=ctx.warming,
        )
        self._loaded_seconds[ctx.job.id] = round(time.perf_counter() - began, 2)

    @staticmethod
    def _input(ctx: JobContext) -> Path:
        inputs = ctx.inputs()
        if not inputs:
            raise JobError(
                "invalid_inputs", "a denoise job with no audio to denoise is not a job"
            )
        if len(inputs) != 1:
            raise JobError(
                "invalid_inputs",
                f"this job carries {len(inputs)} inputs ({sorted(inputs)}); "
                "denoise takes exactly one audio file. The blocking is the "
                "client's, and a block is one file by the time it is sent",
            )
        return next(iter(inputs.values()))

    @staticmethod
    def _check(
        manifest: DenoiseManifest,
        ready: dict[str, Any],
        stems: list[dict[str, Any]],
    ) -> dict[str, Any]:
        marker = f"({manifest.primary_stem})"
        hits = [stem for stem in stems if marker in stem["name"].lower()]
        if len(hits) != 1:
            raise JobError(
                "denoise_primary_stem_missing",
                f"{manifest.model_filename} produced {len(hits)} output(s) naming "
                f"{marker!r} and exactly one is the answer; it wrote "
                f"{[stem['name'] for stem in stems]}. Zero means the model "
                "produced something other than what this manifest says it "
                "produces; two means nothing here can say which one is the "
                "denoised audio",
            )
        primary = hits[0]
        if primary["sample_rate"] != manifest.sample_rate:
            raise JobError(
                "denoise_resampled",
                f"{primary['name']} came back at {primary['sample_rate']} Hz and "
                f"the input was {manifest.sample_rate} Hz — the model resampled "
                "it, which invalidates every sample offset the caller sliced by",
            )
        if primary["frames"] != ready["frames"]:
            raise JobError(
                "denoise_length_changed",
                f"{primary['name']} is {primary['frames']} frames and the input "
                f"was {ready['frames']} — the model changed the length. Slicing "
                "a stem back at the input's offsets is only safe because it does "
                "not",
            )
        return primary


class UnloadDenoiserJobType(UnloadJobType):
    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        super().__init__(
            UNLOAD_DENOISER,
            residency,
            describe=lambda: _descriptors(config, residency),
            provenance=lambda model: MANIFESTS.provenance(config.backend_kind, model),
        )


JOB_TYPES: tuple[JobTypeBinding, ...] = (
    JobTypeBinding(
        DENOISE_JOB,
        lambda wiring: DenoiseJobType(wiring.config, wiring.backend, wiring.residency),
    ),
    JobTypeBinding(
        UNLOAD_DENOISER,
        lambda wiring: UnloadDenoiserJobType(
            wiring.config, wiring.backend, wiring.residency
        ),
    ),
)
