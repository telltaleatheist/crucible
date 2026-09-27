from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ... import hosttools, weights, workers
from ...alignmodels import (
    AlignBackendSpec,
    AlignManifest,
    AlignManifestError,
    load_all_align_manifests,
)
from ...backend import CUDA_LINUX, MLX_DARWIN
from ...clock import utcnow
from ...config import Config
from ...errors import ApiError, JobCancelled, JobError
from ...jobtypes import ALIGN_JOB, UNLOAD_ALIGNER
from ...manifests import fingerprint
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    KIND_ALIGN,
    Occupant,
    Residency,
    ResidentAligner,
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
    "AlignJobType",
    "AlignParams",
    "UnloadAlignerJobType",
    "occupy_aligner",
]

JOB_TYPE = ALIGN_JOB.name

UnloadAlignerParams = UnloadParams

FFMPEG_WHY = (
    "decodes every chunk through it to 16 kHz mono float32 — the rate the "
    "model's feature extractor was trained at, which is why it is not "
    "something a client is asked to do."
)

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


def occupy_aligner(
    residency: Residency,
    manifest: AlignManifest,
    spec: AlignBackendSpec,
    weights_dir: Path,
    python: Path,
    *,
    max_audio_s: float,
    timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
    on_progress: Callable[[str], None] | None = None,
) -> ResidentAligner:
    say = say_to(on_progress)

    def start() -> Occupant:
        log_path = residency.log_path_for(manifest.id)
        device = device_for(spec.backend)
        say(
            f"loading {manifest.id} ({spec.engine}) on {device} at {spec.dtype}; "
            f"log {log_path}"
        )
        session = start_aligner_session(
            python,
            weights_dir,
            spec,
            log_path,
            ready_silence_timeout=timeout,
            on_ready=lambda message: say(
                f"{manifest.id} loaded in {message['seconds']:.1f}s on "
                f"{message['device']} at {message['dtype']}"
            ),
            on_progress=lambda message: say(str(message["message"])),
        )
        resident = ResidentAligner(
            aligner_id=manifest.id,
            backend=spec.backend,
            revision=spec.revision,
            fingerprint=fingerprint(manifest.id, spec.revision),
            device=device,
            dtype=spec.dtype,
            max_audio_s=max_audio_s,
            memory_bytes_estimate=spec.memory_bytes_estimate,
            log_path=log_path,
            loaded_at=utcnow(),
        )
        return Occupant(resident, session=session)

    return cast(
        ResidentAligner, residency.occupy(KIND_ALIGN, manifest.id, start, say=say)
    )


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


MANIFESTS: ManifestCatalog[AlignManifest] = ManifestCatalog(
    lambda: load_all_align_manifests(),
    AlignManifestError,
    unreadable_code="align_manifests_unreadable",
    what="align manifests",
    unknown="align manifest for",
)


def _descriptors(config: Config, residency: Residency) -> list[ModelDescriptor]:
    return MANIFESTS.descriptors(
        config.backend_kind,
        installed=lambda manifest, spec: weights.installed(config, manifest, spec)
        is not None,
        resident=lambda model_id: residency.is_resident(KIND_ALIGN, model_id),
    )


class AlignJobType(ResidentWorker):

    name = JOB_TYPE
    resident_kind = KIND_ALIGN

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
        return MANIFESTS.provenance(self._config.backend_kind, run_model(model, self.name))

    def vram_estimate(self, model: str | None) -> int:
        return MANIFESTS.memory_estimate(
            run_model(model, self.name), self._config.backend_kind
        )

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
            manifests = MANIFESTS.all()
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


    def _resident_session(self) -> workers.WorkerSession | None:
        return self._residency.aligner_session

    def requirements(
        self, model_id: str
    ) -> tuple[str, AlignManifest, AlignBackendSpec, Path, Path]:
        ffmpeg = hosttools.require_ffmpeg(JOB_TYPE, FFMPEG_WHY)
        backend_kind = self._backend.kind
        manifest = MANIFESTS.known(model_id)
        spec = worker_type.require_block(manifest, model_id, backend_kind, "aligner")
        worker_type.refuse_if_larger_than_host(
            self._backend, model_id, spec.memory_bytes_estimate
        )
        python = worker_type.require_worker_python(
            self._config, JOB_TYPE, backend_kind, model_id
        )
        return ffmpeg, manifest, spec, python, worker_type.require_weights(
            self._config, manifest, spec, model_id
        )

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        parse_params(AlignParams, params, self.name)
        _, _, spec, _, _ = self.requirements(model)
        self._residency.refuse_if_claimed(f"aligning with {model!r}")
        if self._residency.is_resident(KIND_ALIGN, model):
            return
        self._guard(model, spec.memory_bytes_estimate)


    def run(self, job: Job, ctx: JobContext) -> None:
        params = AlignParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        ffmpeg, manifest, spec, python, weights_dir = as_job_error(self.requirements, model)

        audio = self._chunk_inputs(ctx, params)
        session = self._session(
            ctx,
            model,
            spec.memory_bytes_estimate,
            lambda: occupy_aligner(
                self._residency,
                manifest,
                spec,
                weights_dir,
                python,
                max_audio_s=QWEN3_MAX_AUDIO_S,
                timeout=DEFAULT_READY_TIMEOUT_SECONDS,
                on_progress=ctx.warming,
            ),
        )

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


class UnloadAlignerJobType(UnloadJobType):
    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        super().__init__(
            UNLOAD_ALIGNER,
            residency,
            describe=lambda: _descriptors(config, residency),
            provenance=lambda model: MANIFESTS.provenance(config.backend_kind, model),
        )


JOB_TYPES: tuple[JobTypeBinding, ...] = (
    JobTypeBinding(
        ALIGN_JOB,
        lambda wiring: AlignJobType(wiring.config, wiring.backend, wiring.residency),
    ),
    JobTypeBinding(
        UNLOAD_ALIGNER,
        lambda wiring: UnloadAlignerJobType(
            wiring.config, wiring.backend, wiring.residency
        ),
    ),
)
