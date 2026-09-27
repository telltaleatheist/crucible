from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ... import accelerator, hosttools, weights, workers
from ...alignmodels import (
    AlignBackendSpec,
    AlignManifest,
    AlignManifestError,
    load_all_align_manifests,
)
from ...backend import CUDA_LINUX, MLX_DARWIN
from ...config import Config
from ...errors import ApiError, JobCancelled, JobError
from ...manifests import fingerprint
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    KIND_ALIGN,
    Residency,
    describe_resident,
)
from .. import worker_type
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor

__all__ = ["AlignJobType", "AlignParams", "UnloadAlignerJobType"]

JOB_TYPE = "align"

QWEN3_MAX_AUDIO_S = 300.0

QWEN3_LANGUAGES: dict[str, str] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "yue": "Cantonese",
}

DEVICE_FOR_BACKEND: dict[str, str] = {
    CUDA_LINUX: "cuda",
    MLX_DARWIN: "mps",
}


def device_for(backend_kind: str) -> str:
    found = DEVICE_FOR_BACKEND.get(backend_kind)
    if found is None:
        raise JobError(
            "backend_unsupported",
            f"there is no align device for backend {backend_kind!r}; this build "
            f"aligns on {sorted(DEVICE_FOR_BACKEND)}",
        )
    return found


READY_SILENCE_TIMEOUT_SECONDS = 900.0

WORKER_SCRIPT = Path(__file__).resolve().parent / "worker.py"


def start_aligner_session(
    python: Path,
    weights_dir: Path,
    spec: AlignBackendSpec,
    log_path: Path,
    *,
    ready_silence_timeout: float,
    on_ready: Callable[[dict[str, Any]], None] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> workers.WorkerSession:
    session = workers.WorkerSession(
        python=python,
        script=WORKER_SCRIPT,
        log_path=log_path,
        environment={
            **workers.worker_environment(python.parent.parent),
            **workers.torch_allocator_environment(spec.backend),
        },
    )
    outcome = session.start(
        {
            "op": "load",
            "model_dir": str(weights_dir),
            "device": device_for(spec.backend),
            "dtype": spec.dtype,
            "memory_cap_bytes": workers.torch_memory_cap(
                spec.backend, spec.memory_bytes_estimate
            ),
        },
        ready_silence_timeout=ready_silence_timeout,
        on_ready=on_ready,
        on_progress=on_progress,
    )
    if outcome.results:
        session.stop()
        raise workers.WorkerError(
            f"{WORKER_SCRIPT.name} answered a load request with "
            f"{len(outcome.results)} result(s); a load produces none"
        )
    return session


class AlignChunk(BaseModel):
    """One chunk: the index its audio is named after, and its spoken text."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError(
                "a chunk's text is empty; a forced aligner places the text it is "
                "given and there is nothing here to place"
            )
        return value


class AlignParams(BaseModel):
    """`params` for an align job. Unknown keys are refused, not ignored."""

    model_config = ConfigDict(extra="forbid")

    language: str
    chunks: list[AlignChunk] = Field(min_length=1)

    @field_validator("language")
    @classmethod
    def known_language(cls, value: str) -> str:
        if value in QWEN3_LANGUAGES:
            return value
        raise ValueError(
            f"{value!r} is not a language Qwen3-ForcedAligner supports; it takes "
            f"one of {sorted(QWEN3_LANGUAGES)}. It does not fall back to English "
            "for a language it was not trained on — it places words badly, and a "
            "silently mis-aligned book is worse than a refused one"
        )

    @field_validator("chunks")
    @classmethod
    def unique_indexes(cls, value: list[AlignChunk]) -> list[AlignChunk]:
        seen = [chunk.index for chunk in value]
        duplicates = sorted({index for index in seen if seen.count(index) > 1})
        if duplicates:
            raise ValueError(
                f"chunk index {duplicates} appears more than once; an index names "
                "one chunk and one input file"
            )
        return value

    def model_language(self) -> str:
        return QWEN3_LANGUAGES[self.language]


def _manifests() -> dict[str, AlignManifest]:
    try:
        return load_all_align_manifests()
    except AlignManifestError as exc:
        raise ApiError(
            500,
            "align_manifests_unreadable",
            f"this server cannot read its align manifests: {exc}",
        ) from None


def _known(model_id: str) -> AlignManifest:
    manifests = _manifests()
    manifest = manifests.get(model_id)
    if manifest is None:
        raise ApiError(
            400,
            "unknown_model",
            f"no align manifest for {model_id!r}; this build ships {sorted(manifests)}",
        )
    return manifest


def _params(model: type[BaseModel], params: dict[str, Any], job_type: str) -> Any:
    try:
        return model.model_validate(params)
    except ValidationError as exc:
        raise ApiError(
            400,
            "invalid_params",
            f"{job_type} params are not valid: "
            + "; ".join(
                f"{'.'.join(str(p) for p in problem['loc']) or '<root>'}: "
                f"{problem['msg']}"
                for problem in exc.errors()
            ),
        ) from None


def _require_ffmpeg() -> str:
    return hosttools.require_ffmpeg(
        "align",
        "decodes every chunk through it to 16 kHz mono float32 — the rate the "
        "model's feature extractor was trained at, which is why it is not "
        "something a client is asked to do.",
    )


def _align_provenance(backend_kind: str, model: str | None) -> dict[str, Any] | None:
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


def _descriptors(config: Config, residency: Residency) -> list[ModelDescriptor]:
    backend_kind = config.backend_kind
    rows: list[ModelDescriptor] = []
    for manifest in _manifests().values():
        if manifest.supports(backend_kind):
            spec = manifest.spec(backend_kind)
            revision, source, estimate = (
                spec.revision,
                spec.hf_repo,
                spec.memory_bytes_estimate,
            )
            installed = weights.installed(config, manifest, spec) is not None
        else:
            revision, source, estimate, installed = "", "", 0, False
        rows.append(
            ModelDescriptor(
                id=manifest.id,
                revision=revision,
                source=source,
                installed=installed,
                resident=residency.is_resident(KIND_ALIGN, manifest.id),
                vram_bytes=estimate,
            )
        )
    return rows


class AlignJobType:

    name = JOB_TYPE

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency


    def describe_models(self) -> list[ModelDescriptor]:
        return _descriptors(self._config, self._residency)

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        return _align_provenance(self._config.backend_kind, model)

    def vram_estimate(self, model: str | None) -> int:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        manifest = _known(model)
        if not manifest.supports(self._config.backend_kind):
            return 0
        return manifest.spec(self._config.backend_kind).memory_bytes_estimate

    def check(self, backend: Any) -> JobTypeStatus:
        env = worker_type.env_or_status(self._config, JOB_TYPE, backend.kind)
        if isinstance(env, JobTypeStatus):
            return env
        if hosttools.ffmpeg_path() is None:
            return JobTypeStatus(
                ready=False,
                detail=f"{env.detail}; but there is no ffmpeg on PATH, and align "
                "decodes every chunk through it. " + hosttools.searched_note(),
            )
        try:
            manifests = _manifests()
        except ApiError as exc:
            return JobTypeStatus(ready=False, detail=exc.message)
        installed = worker_type.installed_ids(
            self._config, manifests.values(), backend.kind
        )
        if not installed:
            return JobTypeStatus(
                ready=False,
                detail=(
                    f"{env.detail}; no aligner is installed — "
                    "`crucible models pull qwen3-aligner`"
                ),
            )
        return JobTypeStatus(ready=True, detail=f"{env.detail}; installed: {installed}")


    def _require_runnable(
        self, model_id: str
    ) -> tuple[AlignManifest, AlignBackendSpec, Path, Path]:
        backend_kind = self._backend.kind
        manifest = _known(model_id)
        spec = worker_type.require_block(manifest, model_id, backend_kind, "aligner")
        worker_type.refuse_if_larger_than_host(
            self._backend, model_id, spec.memory_bytes_estimate
        )
        python = worker_type.require_worker_python(
            self._config, JOB_TYPE, backend_kind, model_id
        )
        return manifest, spec, python, worker_type.require_weights(
            self._config, manifest, spec, model_id
        )

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:
            raise ApiError(400, "model_required", f"{self.name} needs a model")
        _params(AlignParams, params, self.name)
        _require_ffmpeg()
        _, spec, _, _ = self._require_runnable(model)
        self._residency.refuse_if_claimed(f"aligning with {model!r}")
        if self._residency.is_resident(KIND_ALIGN, model):
            return
        accelerator.guard(
            self._config.backend_kind,
            model_id=model,
            need_bytes=spec.memory_bytes_estimate,
            owned_pids=self._residency.owned_pids(),
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
        )


    def run(self, job: Job, ctx: JobContext) -> None:
        params = AlignParams.model_validate(job.params)
        model = job.model
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")

        try:
            ffmpeg = _require_ffmpeg()
            manifest, spec, python, weights_dir = self._require_runnable(model)
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None

        audio = self._chunk_inputs(ctx, params)
        session = self._session(ctx, manifest, spec, weights_dir, python, model)

        request = {
            "op": "align",
            "language": params.model_language(),
            "max_audio_s": QWEN3_MAX_AUDIO_S,
            "ffmpeg": ffmpeg,
            "chunks": [
                {"audio": str(audio[chunk.index]), "text": chunk.text}
                for chunk in params.chunks
            ],
        }

        total = len(params.chunks)
        landed: list[dict[str, Any]] = []

        def on_result(result: dict[str, Any]) -> None:
            position = len(landed)
            if position >= total:
                return
            row: dict[str, Any] = {"index": params.chunks[position].index}
            if "error" in result:
                row["error"] = result["error"]
            else:
                row["items"] = result["items"]
            landed.append(row)
            ctx.cue(row)

        def on_progress(message: dict[str, Any]) -> None:
            processed = int(message["processed"])
            ctx.progress(
                min(1.0, processed / total),
                f"aligned {processed} of {total} chunk(s)",
                stage=message["stage"],
                processed=processed,
                total=total,
            )

        try:
            outcome = session.send(
                request,
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
                on_progress=on_progress,
                cancelled=lambda: ctx.cancelled,
                on_result=on_result,
            )
        except workers.WorkerError as exc:
            self._forget(ctx, model)
            raise JobError("worker_failed", str(exc)) from None
        except JobCancelled:
            self._forget(ctx, model)
            raise

        try:
            workers.require_positional_results(outcome, total, "chunk")
        except workers.WorkerError as exc:
            self._forget(ctx, model)
            raise JobError("worker_failed", str(exc)) from None

        document = {
            "model": model,
            "revision": spec.revision,
            "hf_repo": spec.hf_repo,
            "engine": spec.engine,
            "dtype": spec.dtype,
            "device": device_for(self._config.backend_kind),
            "language": params.language,
            "language_name": params.model_language(),
            "max_audio_s": QWEN3_MAX_AUDIO_S,
            "sample_rate": 16_000,
            "items_are": "the model's own tokenization, not the caller's words",
            "chunks": landed,
        }
        path = ctx.scratch / "alignment.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        ctx.artifact("alignment.json", path)

        failed = [row["index"] for row in landed if "error" in row]
        ctx.progress(
            1.0,
            f"{total - len(failed)} of {total} chunk(s) aligned"
            + (f", {len(failed)} failed: {failed}" if failed else ""),
            stage="aligning",
            processed=total,
            total=total,
        )
        ctx.done_extra(
            chunks=total, failed=failed, resident=self._residency.resident_id
        )


    @staticmethod
    def _chunk_inputs(ctx: JobContext, params: AlignParams) -> dict[int, Path]:
        inputs = ctx.inputs()
        by_index: dict[int, Path] = {}
        unnamed: list[str] = []
        for name, path in inputs.items():
            stem = Path(name).stem
            try:
                by_index[int(stem)] = path
            except ValueError:
                unnamed.append(name)
        if unnamed:
            raise JobError(
                "invalid_inputs",
                f"input(s) {sorted(unnamed)} are not named <index>.<ext>; an align "
                "input is matched to its text by the index in its filename",
            )
        wanted = {chunk.index for chunk in params.chunks}
        missing = sorted(wanted - set(by_index))
        extra = sorted(set(by_index) - wanted)
        if missing or extra:
            raise JobError(
                "invalid_inputs",
                "the chunks and the inputs do not line up: "
                + "; ".join(
                    part
                    for part in (
                        f"chunk(s) {missing} have no audio" if missing else "",
                        f"input(s) {extra} have no chunk" if extra else "",
                    )
                    if part
                ),
            )
        return by_index

    def _session(
        self,
        ctx: JobContext,
        manifest: AlignManifest,
        spec: AlignBackendSpec,
        weights_dir: Path,
        python: Path,
        model: str,
    ) -> workers.WorkerSession:
        session = self._residency.aligner_session
        if session is not None and self._residency.is_resident(KIND_ALIGN, model):
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
                owned_pids=self._residency.owned_pids(),
                desktop_allowance_bytes=self._config.desktop_allowance_bytes,
                reclaimable_bytes=self._residency.reclaimable_bytes(excluding=model),
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None
        ctx.warming(state.detail)

        try:
            self._residency.load_aligner(
                manifest,
                spec,
                weights_dir,
                python,
                max_audio_s=QWEN3_MAX_AUDIO_S,
                timeout=DEFAULT_READY_TIMEOUT_SECONDS,
                on_progress=ctx.warming,
            )
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None
        loaded = self._residency.aligner_session
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


class UnloadAlignerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UnloadAlignerJobType:

    name = "unload-aligner"

    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency

    @property
    def residency(self) -> Residency:
        return self._residency

    def describe_models(self) -> list[ModelDescriptor]:
        return _descriptors(self._config, self._residency)

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        return _align_provenance(self._config.backend_kind, model)

    def vram_estimate(self, model: str | None) -> int:
        return 0

    def check(self, backend: Any) -> JobTypeStatus:
        aligner = self._residency.resident_aligner
        return JobTypeStatus(
            ready=True,
            detail=(
                f"resident: {aligner.aligner_id}"
                if aligner
                else "no aligner is resident"
            ),
        )

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:
            raise ApiError(400, "model_required", f"{self.name} needs an aligner")
        _params(UnloadAlignerParams, params, self.name)
        if self._residency.being_cleared(model):
            return
        self._residency.refuse_if_claimed(f"unloading {model!r}")
        if not self._residency.is_resident(KIND_ALIGN, model):
            raise ApiError(
                409,
                "aligner_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_ALIGN, "no aligner is"),
                {"requested": model, "resident": self._residency.resident_id},
            )

    def run(self, job: Job, ctx: JobContext) -> None:
        UnloadAlignerParams.model_validate(job.params)
        model = job.model
        if model is None:
            raise JobError("model_required", f"{self.name} needs an aligner")
        if self._residency.await_clearance(model):
            ctx.progress(0.0, f"unloading {model}")
            ctx.progress(1.0, f"{model} is unloaded — the card was cleared of it")
            ctx.done_extra(resident=self._residency.resident_id)
            return
        if not self._residency.is_resident(KIND_ALIGN, model):
            raise JobError(
                "aligner_not_resident",
                f"{model!r} is not resident on this server; "
                + describe_resident(self._residency, KIND_ALIGN, "no aligner is"),
            )
        ctx.progress(0.0, f"unloading {model}")
        try:
            self._residency.unload(model)
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None
        ctx.progress(1.0, f"{model} is unloaded")
        ctx.done_extra(resident=self._residency.resident_id)
