from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ... import hosttools, rvcbase, weights, workers
from ...config import Config
from ...errors import ApiError, JobError
from ...jobtypes import RVC_JOB
from ...rvcmodels import (
    RvcBackendSpec,
    RvcManifest,
    RvcManifestError,
    load_all_rvc_manifests,
)
from .. import worker_type
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from ..binding import JobTypeBinding
from ..template import (
    ManifestCatalog,
    as_job_error,
    card_guard,
    parse_params,
    require_model,
    run_model,
)

__all__ = ["JOB_TYPES", "RvcJobType", "RvcParams", "rvc_base_dir"]

JOB_TYPE = RVC_JOB.name

BATCH_SIZE = 96

PROTECT_RATE_OFF_INVERTED_SCALE = 0.5

LEAK_BYTES_PER_AUDIO_SECOND = 3.4e9 / 600.0

MEMORY_FRACTION = 0.5

MAX_BATCH_AUDIO_SECONDS = 1800.0

FALLBACK_MEMORY_BUDGET_BYTES = 3 * 1024**3

DEFAULT_PIECE_S = 60.0
MIN_PIECE_S = 10.0
MAX_PIECE_S = 600.0
DEFAULT_OVERLAP_S = 0.5
MAX_OVERLAP_S = 5.0
DEFAULT_CROSSFADE_S = 0.02
MAX_CROSSFADE_S = 1.0

READY_SILENCE_TIMEOUT_SECONDS = 900.0

ENGINE_ENVIRONMENT: dict[str, str] = {
    "URVC_SKIP_INIT": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}

WORKER_SCRIPT = Path(__file__).resolve().parent / "worker.py"


def rvc_base_dir(config: Config) -> Path:
    return rvcbase.base_root(config)


def ffmpeg_paths() -> dict[str, str | None]:
    return {"ffmpeg": hosttools.ffmpeg_path(), "ffprobe": hosttools.ffprobe_path()}


def _require_ffmpeg() -> dict[str, str]:
    found = ffmpeg_paths()
    absent = [tool for tool, path in found.items() if path is None]
    if absent:
        raise ApiError(
            409,
            "ffmpeg_missing",
            f"this server has no {' and no '.join(absent)}; ultimate-rvc decodes "
            "every piece with ffmpeg and probes it with ffprobe. `crucible install rvc` "
            "places Crucible's pinned build. " + hosttools.searched_note(),
            {"missing": absent, "path": hosttools.search_path()},
        )
    return {tool: str(path) for tool, path in found.items()}


def _base_assets() -> "rvcbase.RvcBaseAssets":
    try:
        return rvcbase.load_rvc_base()
    except rvcbase.RvcBaseError as exc:
        raise ApiError(
            500,
            "rvc_base_declaration_unreadable",
            f"this server cannot read its base-asset declaration: {exc}",
        ) from None


class RvcParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_rate: float = Field(ge=0.0, le=1.0)
    protect_rate: float = Field(ge=0.0, le=PROTECT_RATE_OFF_INVERTED_SCALE)

    n_semitones: int = Field(ge=-24, le=24)

    f0_method: str | None = None

    hop_length: int | None = Field(default=None, ge=1, le=512)

    piece_s: float | None = None
    overlap_s: float | None = None
    crossfade_s: float | None = None

    output_rate: Literal["native", "input"] = "native"

    output_channels: Literal["input", "mono"] = "input"

    @field_validator("piece_s")
    @classmethod
    def piece_in_range(cls, value: float | None) -> float | None:
        if value is not None and not (MIN_PIECE_S <= value <= MAX_PIECE_S):
            raise ValueError(
                f"piece_s is {value}; a piece is {MIN_PIECE_S:g} to {MAX_PIECE_S:g} "
                "seconds — ten minutes is the longest whose memory cost is "
                "measured. Send null for this server's default "
                f"({DEFAULT_PIECE_S:g}); the input itself may be any length"
            )
        return value

    @field_validator("overlap_s")
    @classmethod
    def overlap_in_range(cls, value: float | None) -> float | None:
        if value is not None and not (0.0 <= value <= MAX_OVERLAP_S):
            raise ValueError(
                f"overlap_s is {value}; it is 0 to {MAX_OVERLAP_S:g} seconds of real "
                "audio converted on each side of a piece and then dropped. Send "
                f"null for this server's default ({DEFAULT_OVERLAP_S:g})"
            )
        return value

    @field_validator("crossfade_s")
    @classmethod
    def crossfade_in_range(cls, value: float | None) -> float | None:
        if value is not None and not (0.0 <= value <= MAX_CROSSFADE_S):
            raise ValueError(
                f"crossfade_s is {value}; it is 0 to {MAX_CROSSFADE_S:g} seconds. "
                "Longer fades blend two conversions whose pitch agrees and whose "
                "phase does not. Send null for this server's default "
                f"({DEFAULT_CROSSFADE_S:g})"
            )
        return value

    @model_validator(mode="after")
    def pieces_fit_together(self) -> "RvcParams":
        piece, overlap = self.piece_seconds(), self.overlap_seconds()
        if overlap * 2 >= piece:
            raise ValueError(
                f"overlap_s {overlap:g} on both sides of a {piece:g} s piece "
                "converts more of its neighbours than of itself; keep the overlap "
                "under half the piece"
            )
        if self.crossfade_s is not None and self.crossfade_s > 2 * overlap:
            raise ValueError(
                f"crossfade_s {self.crossfade_s:g} is longer than the "
                f"{2 * overlap:g} s of converted audio both pieces share at a seam "
                f"(twice overlap_s {overlap:g}); raise overlap_s to at least "
                f"{self.crossfade_s / 2:g}, or shorten the fade"
            )
        return self

    def piece_seconds(self) -> float:
        return self.piece_s if self.piece_s is not None else DEFAULT_PIECE_S

    def overlap_seconds(self) -> float:
        return self.overlap_s if self.overlap_s is not None else DEFAULT_OVERLAP_S

    def crossfade_seconds(self) -> float:
        if self.crossfade_s is not None:
            return self.crossfade_s
        return min(DEFAULT_CROSSFADE_S, 2 * self.overlap_seconds())


MANIFESTS: ManifestCatalog[RvcManifest] = ManifestCatalog(
    lambda: load_all_rvc_manifests(),
    RvcManifestError,
    unreadable_code="rvc_manifests_unreadable",
    what="RVC manifests",
    unknown="RVC manifest for",
)


def _require_base_assets(config: Config) -> Path:
    assets = _base_assets()
    root = rvc_base_dir(config)
    absent = rvcbase.missing(config, assets)
    if absent:
        raise ApiError(
            409,
            "rvc_base_models_missing",
            f"ultimate-rvc needs its shared base assets and this host is missing "
            f"{[str(root / name) for name in absent]}. They are the engine's, not "
            f"any model's — the same files urvc's own first-run downloader would "
            f"have fetched, which URVC_SKIP_INIT turns off. Run "
            f"`{rvcbase.PULL_COMMAND}` to place them "
            f"({assets.total_bytes / 1e9:.2f} GB from "
            f"{assets.hf_repo}@{assets.revision[:12]}, verified file by file)",
            {
                "root": str(root),
                "missing": sorted(absent),
                "hf_repo": assets.hf_repo,
                "revision": assets.revision,
                "command": rvcbase.PULL_COMMAND,
            },
        )
    return root


class RvcJobType:
    name = JOB_TYPE

    def __init__(
        self,
        config: Config,
        backend: Any,
        owned_pids: Callable[[], frozenset[int]],
    ) -> None:
        self._config = config
        self._backend = backend
        self._owned_pids = owned_pids


    def describe_models(self) -> list[ModelDescriptor]:
        return MANIFESTS.descriptors(
            self._config.backend_kind,
            installed=lambda manifest, spec: weights.installed(
                self._config, manifest, spec
            )
            is not None,
            resident=lambda model_id: False,
            source=lambda spec: f"{spec.hf_repo}:{spec.archive}",
        )

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
        absent_tools = [tool for tool, path in ffmpeg_paths().items() if path is None]
        if absent_tools:
            return JobTypeStatus(
                ready=False,
                detail=(
                    f"{env.detail}; but there is no {' and no '.join(absent_tools)}: "
                    "urvc decodes every piece with ffmpeg and probes it with "
                    "ffprobe — `crucible install rvc` places Crucible's pinned build"
                ),
            )
        root = rvc_base_dir(self._config)
        try:
            absent = rvcbase.missing(self._config, _base_assets())
        except ApiError as exc:
            return JobTypeStatus(ready=False, detail=exc.message)
        if absent:
            return JobTypeStatus(
                ready=False,
                detail=(
                    f"{env.detail}; but urvc's base assets are not at {root}: "
                    f"{sorted(absent)} missing — `{rvcbase.PULL_COMMAND}`"
                ),
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
                    f"{env.detail}; no RVC model is installed — "
                    "`crucible rvc pull <id>`"
                ),
                awaiting_weights=True,
            )
        return JobTypeStatus(ready=True, detail=f"{env.detail}; installed: {installed}")


    def requirements(self, model_id: str, params: RvcParams) -> tuple[
        RvcManifest, RvcBackendSpec, Path, Path, dict[str, str], Path, Any
    ]:
        backend_kind = self._backend.kind
        manifest = MANIFESTS.known(model_id)
        spec = worker_type.require_block(manifest, model_id, backend_kind, "RVC model")
        worker_type.refuse_if_larger_than_host(
            self._backend, model_id, spec.memory_bytes_estimate
        )
        python = worker_type.require_worker_python(
            self._config, JOB_TYPE, backend_kind, model_id
        )
        weights_dir = worker_type.require_weights(self._config, manifest, spec, model_id)
        tools = _require_ffmpeg()
        self._require_index(manifest, params)
        base = _require_base_assets(self._config)
        state = card_guard(
            self._config,
            model=model_id,
            need_bytes=spec.memory_bytes_estimate,
            owned_pids=self._owned_pids(),
        )
        return manifest, spec, python, weights_dir, tools, base, state

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        self.requirements(model, parse_params(RvcParams, params, self.name))

    @staticmethod
    def _require_index(manifest: RvcManifest, params: RvcParams) -> None:
        if manifest.has_index or params.index_rate == 0:
            return
        raise ApiError(
            400,
            "model_has_no_index",
            f"RVC model {manifest.id!r} ships no .index, so an index_rate of "
            f"{params.index_rate} asks for feature retrieval that cannot happen. "
            "Send index_rate 0. Note that protect_rate is then a no-op too — it "
            "blends back features that are only cloned when retrieval runs",
            {"model": manifest.id, "index_rate": params.index_rate},
        )


    def run(self, job: Job, ctx: JobContext) -> None:
        params = RvcParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        manifest, spec, python, weights_dir, tools, base, state = as_job_error(
            self.requirements, model, params
        )
        ctx.warming(state.detail)

        names = self._inputs(ctx)
        models_dir = self._stage_models(ctx, base, weights_dir, manifest)
        output_dir = ctx.scratch / "converted"

        request = {
            "models_dir": str(models_dir),
            "model_name": manifest.model_name,
            "input_dir": str(ctx.job.inputs_dir),
            "output_dir": str(output_dir),
            "inputs": names,
            "index_rate": params.index_rate,
            "protect_rate": params.protect_rate,
            "n_semitones": params.n_semitones,
            "batch_size": BATCH_SIZE,
            "memory_fraction": MEMORY_FRACTION,
            "leak_bytes_per_audio_s": LEAK_BYTES_PER_AUDIO_SECOND,
            "max_batch_audio_s": MAX_BATCH_AUDIO_SECONDS,
            "fallback_budget_bytes": FALLBACK_MEMORY_BUDGET_BYTES,
            "piece_s": params.piece_seconds(),
            "overlap_s": params.overlap_seconds(),
            "crossfade_s": params.crossfade_seconds(),
            "output_rate": params.output_rate,
            "output_channels": params.output_channels,
            "staging_dir": str(ctx.scratch / "staging"),
            "ffmpeg": tools["ffmpeg"],
            "ffprobe": tools["ffprobe"],
        }
        if params.f0_method is not None:
            request["f0_method"] = params.f0_method
        if params.hop_length is not None:
            request["hop_length"] = params.hop_length

        total = len(names)
        pieces = [total]

        def on_ready(message: dict[str, Any]) -> None:
            pieces[0] = int(message.get("pieces", message["files"]))
            budget = message.get("memory_budget_bytes")
            bound = (
                ""
                if budget is None
                else f", each urvc process recycled by {budget / 1e9:.1f} GB "
                f"({message.get('memory_basis')}) or "
                f"{float(message.get('batch_audio_s', 0)) / 60:.0f} min of audio"
            )
            ctx.warming(
                f"{message['files']} file(s), cut into {pieces[0]} piece(s), "
                f"through {manifest.model_name} in at least {message['batches']} "
                f"batch(es) of {message['batch_size']} piece(s) at most{bound} — "
                "the batch is a memory bound, not a throughput choice"
            )

        published: list[str] = []

        def on_result(message: dict[str, Any]) -> None:
            name = names[len(published)] if len(published) < len(names) else None
            published.append("" if name is None else name)
            if name is None or "error" in message:
                return
            ctx.artifact(name, output_dir / name)
            (output_dir / name).unlink(missing_ok=True)

        def on_progress(message: dict[str, Any]) -> None:
            processed = int(message["processed"])
            if message["stage"] == "cutting":
                ctx.progress(
                    0.0,
                    f"finding quiet points: {processed} of {total} file(s) read",
                    stage="cutting",
                    processed=processed,
                    total=total,
                )
                return
            ctx.progress(
                min(1.0, processed / pieces[0]),
                f"converted {processed} of {pieces[0]} piece(s) of {total} file(s) "
                f"(batch {message['batch']} of {message['batches']})",
                stage=message["stage"],
                processed=processed,
                total=pieces[0],
            )

        try:
            outcome = workers.run_worker(
                python=python,
                script=WORKER_SCRIPT,
                request=request,
                log_path=self._config.logs_dir / f"rvc-{job.id}.log",
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
                on_ready=on_ready,
                on_progress=on_progress,
                on_result=on_result,
                cancelled=lambda: ctx.cancelled,
                environment=dict(ENGINE_ENVIRONMENT),
            )
            results = workers.require_positional_results(outcome, total, "file")
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None

        missing = [
            f"{name}: {result['error']}"
            for name, result in zip(names, results)
            if "error" in result
        ]
        if missing:
            raise JobError(
                "rvc_output_missing",
                f"{len(missing)} of {total} input(s) produced no output, so this job "
                "is not a conversion of its inputs: the others are kept as "
                "artifacts, and a book assembled from them would have sentences in "
                "two voices. "
                + "; ".join(missing[:20])
                + (f" (and {len(missing) - 20} more)" if len(missing) > 20 else ""),
            )

        ctx.progress(
            1.0,
            f"{total} file(s) converted through {manifest.model_name}",
            stage="converting",
            processed=total,
            total=total,
        )
        ctx.done_extra(
            files=total,
            pieces=pieces[0],
            model_name=manifest.model_name,
            outputs={
                name: {
                    key: result[key]
                    for key in ("frames", "sample_rate", "channels", "format", "subtype")
                    if key in result
                }
                for name, result in zip(names, results)
            },
        )


    @staticmethod
    def _inputs(ctx: JobContext) -> list[str]:
        inputs = ctx.inputs()
        if not inputs:
            raise JobError(
                "invalid_inputs", "an rvc job with no audio to convert is not a job"
            )
        return sorted(inputs)

    @staticmethod
    def _stage_models(
        ctx: JobContext, base: Path, weights_dir: Path, manifest: RvcManifest
    ) -> Path:
        root = ctx.scratch / "urvc-models" / "rvc"
        root.mkdir(parents=True, exist_ok=True)
        for name in ("embedders", "predictors", "pretraineds"):
            source = base / "rvc" / name
            if source.is_dir():
                (root / name).symlink_to(source, target_is_directory=True)
        voices = root / "voice_models"
        voices.mkdir()
        model_dir = weights_dir / "rvc" / "voice_models" / manifest.model_name
        if not model_dir.is_dir():
            raise JobError(
                "model_not_installed",
                f"{manifest.id!r} is stamped as pulled but {model_dir} is not there; "
                f"{manifest.path.name} says the archive holds "
                f"rvc/voice_models/{manifest.model_name}. Re-pull it with "
                f"`crucible rvc pull {manifest.id} --force`",
            )
        (voices / manifest.model_name).symlink_to(model_dir, target_is_directory=True)
        return root.parent


JOB_TYPES: tuple[JobTypeBinding, ...] = (
    JobTypeBinding(
        RVC_JOB,
        lambda wiring: RvcJobType(
            wiring.config, wiring.backend, wiring.residency.owned_pids
        ),
    ),
)
