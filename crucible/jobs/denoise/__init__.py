from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from ... import accelerator, workers
from ...backend import CUDA_LINUX
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
from ...manifests import fingerprint
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    KIND_DENOISE,
    Residency,
    describe_resident,
)
from .. import worker_type
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor

__all__ = [
    "DenoiseJobType",
    "DenoiseParams",
    "UnloadDenoiserJobType",
    "denoise_models_dir",
    "denoise_models_dir_for",
]

JOB_TYPE = "denoise"

ENV_JOB_TYPE = "rvc"

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


class DenoiseParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _manifests() -> dict[str, DenoiseManifest]:
    try:
        return load_all_denoise_manifests()
    except DenoiseManifestError as exc:
        raise ApiError(
            500,
            "denoise_manifests_unreadable",
            f"this server cannot read its denoise manifests: {exc}",
        ) from None


def _known(model_id: str) -> DenoiseManifest:
    manifests = _manifests()
    manifest = manifests.get(model_id)
    if manifest is None:
        raise ApiError(
            400,
            "unknown_model",
            f"no denoise manifest for {model_id!r}; this build ships "
            f"{sorted(manifests)}",
        )
    return manifest


def _params(params: dict[str, Any]) -> DenoiseParams:
    try:
        return DenoiseParams.model_validate(params)
    except ValidationError as exc:
        raise ApiError(
            400,
            "invalid_params",
            "denoise takes no params — every separation knob is an engine "
            "default this server does not put on the wire (PHASE4-AUDIO.md "
            "section 4.2): "
            + "; ".join(
                f"{'.'.join(str(p) for p in problem['loc']) or '<root>'}: "
                f"{problem['msg']}"
                for problem in exc.errors()
            ),
        ) from None


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


def _denoise_provenance(
    backend_kind: str, model: str | None
) -> dict[str, Any] | None:
    if model is None:
        return None
    manifest = _known(model)
    spec = manifest.backends.get(backend_kind)
    if spec is None:
        return {"id": model, "revision": None, "fingerprint": None}
    return {
        "id": model,
        "revision": spec.revision,
        "fingerprint": fingerprint(model, spec.revision),
    }


class DenoiseJobType:
    name = JOB_TYPE

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency
        self._loaded_seconds: dict[str, float] = {}

    @property
    def residency(self) -> Residency:
        return self._residency

    def _owned_pids(self) -> frozenset[int]:
        return self._residency.owned_pids()


    def describe_models(self) -> list[ModelDescriptor]:
        backend_kind = self._config.backend_kind
        rows: list[ModelDescriptor] = []
        for manifest in _manifests().values():
            if manifest.supports(backend_kind):
                spec = manifest.spec(backend_kind)
                revision, source, estimate = (
                    spec.revision,
                    f"{spec.hf_repo}:{spec.model_path}",
                    spec.memory_bytes_estimate,
                )
                installed = (
                    model_installed(self._config.home, manifest, spec) is not None
                )
            else:
                revision, source, estimate, installed = "", "", 0, False
            rows.append(
                ModelDescriptor(
                    id=manifest.id,
                    revision=revision,
                    source=source,
                    installed=installed,
                    resident=self._residency.is_resident(KIND_DENOISE, manifest.id),
                    vram_bytes=estimate,
                )
            )
        return rows

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        return _denoise_provenance(self._config.backend_kind, model)

    def vram_estimate(self, model: str | None) -> int:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        manifest = _known(model)
        if not manifest.supports(self._config.backend_kind):
            return 0
        return manifest.spec(self._config.backend_kind).memory_bytes_estimate

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
            manifests = _manifests()
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


    def _require_runnable(
        self, model_id: str
    ) -> tuple[DenoiseManifest, DenoiseBackendSpec, Path, Path]:
        backend_kind = self._backend.kind
        manifest = _known(model_id)
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
        if model is None:
            raise ApiError(400, "model_required", f"{self.name} needs a model")
        _params(params)
        _manifest, spec, _python, _root = self._require_runnable(model)
        self._residency.refuse_if_claimed(f"denoising with {model!r}")
        if self._residency.is_resident(KIND_DENOISE, model):
            return
        accelerator.guard(
            self._config.backend_kind,
            model_id=model,
            need_bytes=spec.memory_bytes_estimate,
            owned_pids=self._owned_pids(),
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
        )


    def run(self, job: Job, ctx: JobContext) -> None:
        DenoiseParams.model_validate(job.params)
        model = job.model
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")

        try:
            manifest, spec, python, root = self._require_runnable(model)
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None

        source = self._input(ctx)
        output_dir = ctx.scratch / "stems"
        session = self._session(ctx, manifest, spec, root, python, model)

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


    def _session(
        self,
        ctx: JobContext,
        manifest: DenoiseManifest,
        spec: DenoiseBackendSpec,
        model_file_dir: Path,
        python: Path,
        model: str,
    ) -> workers.WorkerSession:
        session = self._residency.separator_session
        if session is not None and self._residency.is_resident(KIND_DENOISE, model):
            if session.alive:
                return session
            ctx.warming(
                f"the resident {model} worker is gone (its log is "
                f"{session.log_path}); loading it again"
            )
            self._forget(ctx, model)

        try:
            state = accelerator.guard(
                self._config.backend_kind,
                model_id=model,
                need_bytes=spec.memory_bytes_estimate,
                owned_pids=self._owned_pids(),
                desktop_allowance_bytes=self._config.desktop_allowance_bytes,
                reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None
        ctx.warming(state.detail)

        began = time.perf_counter()
        try:
            self._residency.load_separator(
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
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None
        self._loaded_seconds[ctx.job.id] = round(time.perf_counter() - began, 2)
        loaded = self._residency.separator_session
        if loaded is None:
            raise JobError(
                "worker_failed",
                f"{model} loaded but no session was published; this is a bug in "
                "crucible/residency.py",
            )
        return loaded

    def _forget(self, ctx: JobContext, model: str) -> None:
        try:
            self._residency.unload(model)
        except (KeyError, workers.WorkerError) as exc:
            line = f"could not take {model} off the card: {type(exc).__name__}: {exc}"
            print(f"crucible: {line}", file=sys.stderr)
            ctx.note(line)

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


class UnloadDenoiserParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UnloadDenoiserJobType:
    name = "unload-denoiser"

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        rows: list[ModelDescriptor] = []
        for manifest in _manifests().values():
            backend_kind = self._config.backend_kind
            supported = manifest.supports(backend_kind)
            spec = manifest.spec(backend_kind) if supported else None
            rows.append(
                ModelDescriptor(
                    id=manifest.id,
                    revision=spec.revision if spec else "",
                    source=f"{spec.hf_repo}:{spec.model_path}" if spec else "",
                    installed=(
                        model_installed(self._config.home, manifest, spec) is not None
                        if spec
                        else False
                    ),
                    resident=self._residency.is_resident(KIND_DENOISE, manifest.id),
                    vram_bytes=spec.memory_bytes_estimate if spec else 0,
                )
            )
        return rows

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return _denoise_provenance(self._config.backend_kind, model)

    def vram_estimate(self, model: str | None) -> int:
        return 0

    def check(self, backend: Any) -> JobTypeStatus:
        separator = self._residency.resident_separator
        return JobTypeStatus(
            ready=True,
            detail=(
                f"resident: {separator.separator_id}"
                if separator
                else "no separator is resident"
            ),
        )

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:
            raise ApiError(400, "model_required", f"{self.name} needs a separator")
        try:
            UnloadDenoiserParams.model_validate(params)
        except ValidationError as exc:
            raise ApiError(
                400, "invalid_params", f"{self.name} takes no params: {exc}"
            ) from None
        if self._residency.being_cleared(model):
            return
        self._residency.refuse_if_claimed(f"unloading {model!r}")
        if not self._residency.is_resident(KIND_DENOISE, model):
            raise ApiError(
                409,
                "separator_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_DENOISE, "no separator is"),
                {"requested": model, "resident": self._residency.resident_id},
            )

    def run(self, job: Job, ctx: JobContext) -> None:
        UnloadDenoiserParams.model_validate(job.params)
        model = job.model
        if model is None:
            raise JobError("model_required", f"{self.name} needs a separator")
        if self._residency.await_clearance(model):
            ctx.progress(0.0, f"unloading {model}")
            ctx.progress(1.0, f"{model} is unloaded — the card was cleared of it")
            ctx.done_extra(resident=self._residency.resident_id)
            return
        if not self._residency.is_resident(KIND_DENOISE, model):
            raise JobError(
                "separator_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_DENOISE, "no separator is"),
            )
        ctx.progress(0.0, f"unloading {model}")
        try:
            self._residency.unload(model)
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None
        ctx.progress(1.0, f"{model} is unloaded")
        ctx.done_extra(resident=self._residency.resident_id)
