"""The `score` job: a recorded song in, a YuE2-native ABC score out (docs/SCORE.md).

SheetSage2 transcribes the recording (crucible/jobs/score/worker.py, in the score env);
the worker then makes its ABC YuE2-native (yue2native.py: section labels mapped onto
YuE2's, the result held to YuE2's own validator). What is done with the score - a cover's
lyrics, its style, the song YuE2 makes from it - is the app's.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from pydantic import BaseModel, ConfigDict

from ... import hosttools, scoreweights, weights, workers
from ...backend import MLX_DARWIN
from ...cardkinds import KIND_SCORE
from ...clock import utcnow
from ...config import Config
from ...errors import ApiError, JobCancelled, JobError
from ...jobtypes import LOAD_SCORE, SCORE_JOB, UNLOAD_SCORE
from ...manifests import fingerprint
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    Occupant,
    Residency,
    ResidentScorer,
    say_to,
)
from ...scoremodels import (
    ENCODER_COMPANION,
    ScoreBackendSpec,
    ScoreManifest,
    ScoreManifestError,
    load_all_score_manifests,
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
from . import recording
from .params import ScoreParams, refuse_past_the_ceiling
from .recording import Recording

__all__ = [
    "JOB_TYPES",
    "LoadScoreJobType",
    "LoadScoreParams",
    "ScoreJobType",
    "ScoreParams",
    "UnloadScoreJobType",
    "occupy_scorer",
]

JOB_TYPE = SCORE_JOB.name

# The YuE2-native score: SheetSage2's, relabelled, the Vocal voice still singing.
SCORE_ARTIFACT = "score.abc"

# The same score with the sung melody moved to the instrument voice, as YuE's cover skill
# hands an instrumental cover to YuE2; this is the form YuE2's validator passed.
INSTRUMENTAL_ARTIFACT = "instrumental.abc"

# SheetSage2's score exactly as it came out: the evidence, kept on success and on failure.
RAW_ARTIFACT = "sheetsage2.abc"

# SheetSage2's playback MIDI: the melodies, and the chords when they were kept.
MIDI_ARTIFACT = "transcription.mid"

REPORT_ARTIFACT = "score.json"

SHEETSAGE2_DIR = "sheetsage2"

FFMPEG_WHY = "decodes every recording through it before the model loads"

READY_SILENCE_TIMEOUT_SECONDS = 600.0

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


@dataclass(frozen=True)
class Needs:
    manifest: ScoreManifest
    spec: ScoreBackendSpec
    python: Path
    weights_dir: Path

    @property
    def encoder_dir(self) -> Path:
        return scoreweights.companion_dir(self.weights_dir, self.spec.encoder)


def start_score_session(
    python: Path,
    needs: Needs,
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
            "encoder_dir": str(needs.encoder_dir),
            "device": spec.device,
            "autocast": spec.autocast,
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


def occupy_scorer(
    residency: Residency,
    needs: Needs,
    *,
    timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
    on_progress: Callable[[str], None] | None = None,
) -> ResidentScorer:
    """Put the score model on the card. Residency.occupy takes off whatever is resident
    first - a YuE2 kept for a cover's song included: one resident engine at a time."""
    say = say_to(on_progress)
    manifest, spec = needs.manifest, needs.spec
    loaded: dict[str, Any] = {}

    def start() -> Occupant:
        log_path = residency.log_path_for(manifest.id)
        say(f"starting the {spec.engine} worker for {manifest.id}; log {log_path}")
        session = start_score_session(
            needs.python, needs, log_path, ready_silence_timeout=timeout, on_ready=loaded.update
        )
        resident = ResidentScorer(
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

    return cast(ResidentScorer, residency.occupy(KIND_SCORE, manifest.id, start, say=say))


MANIFESTS: ManifestCatalog[ScoreManifest] = ManifestCatalog(
    lambda: load_all_score_manifests(),
    ScoreManifestError,
    unreadable_code="score_manifests_unreadable",
    what="score manifests",
    unknown="score model",
)


def _installed(config: Config, manifest: ScoreManifest, spec: ScoreBackendSpec) -> bool:
    return scoreweights.installed(config, manifest, spec) is not None


def _descriptors(config: Config, residency: Residency) -> list[ModelDescriptor]:
    return MANIFESTS.descriptors(
        config.backend_kind,
        installed=lambda manifest, spec: _installed(config, manifest, spec),
        resident=lambda model_id: residency.is_resident(KIND_SCORE, model_id),
    )


def _require_block(manifest: ScoreManifest, model_id: str, backend_kind: str) -> ScoreBackendSpec:
    return worker_type.require_block(manifest, model_id, backend_kind, "score model")


def _require_weights(config: Config, manifest: ScoreManifest, spec: ScoreBackendSpec, model_id: str) -> Path:
    try:
        return scoreweights.require_installed(config, manifest, spec).path
    except weights.WeightsError as exc:
        raise ApiError(
            409,
            "model_not_installed",
            str(exc),
            {"model": model_id, "hf_repo": spec.hf_repo, "revision": spec.revision},
        ) from None


class _Scoring:
    def __init__(self, ctx: JobContext) -> None:
        self._ctx = ctx

    def progress(self, message: dict[str, Any]) -> None:
        stage = str(message.get("stage"))
        if stage == "cancelled":
            return
        fraction = min(1.0, max(0.0, float(message.get("fraction") or 0.0)))
        said = stage
        if message.get("window") is not None:
            said = f"{stage} window {message['window']} of {message.get('windows')}"
            if message.get("tokens") is not None:
                said += f", {message['tokens']} tokens"
        extra = {
            key: message[key] for key in ("window", "windows", "tokens") if key in message
        }
        self._ctx.progress(fraction, said, stage=stage, **extra)


class ScoreJobType(ResidentWorker):

    name = JOB_TYPE
    resident_kind = KIND_SCORE

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
        return MANIFESTS.memory_estimate(run_model(model, self.name), self._config.backend_kind)

    def check(self, backend: Any) -> JobTypeStatus:
        env = worker_type.env_or_status(self._config, JOB_TYPE, backend.kind)
        if isinstance(env, JobTypeStatus):
            return env
        if hosttools.ffmpeg_path() is None:
            return JobTypeStatus(
                ready=False,
                detail=f"{env.detail}; there is no ffmpeg on PATH, and score "
                f"{FFMPEG_WHY}",
            )
        try:
            manifests = MANIFESTS.all()
        except ApiError as exc:
            return JobTypeStatus(ready=False, detail=exc.message)
        installed = [
            manifest.id
            for manifest in manifests.values()
            if manifest.supports(backend.kind)
            and _installed(self._config, manifest, manifest.spec(backend.kind))
        ]
        if not installed:
            offered = sorted(m.id for m in manifests.values() if m.supports(backend.kind))
            pulls = " or ".join(f"`crucible models pull {model}`" for model in offered)
            return JobTypeStatus(
                ready=False,
                detail=f"{env.detail}; no score model is installed — {pulls or 'none is declared for this backend'}",
            )
        return JobTypeStatus(ready=True, detail=f"{env.detail}; installed: {installed}")

    def _resident_session(self) -> workers.WorkerSession | None:
        return self._residency.score_session

    def loadable(self, model_id: str) -> Needs:
        backend_kind = self._backend.kind
        manifest = MANIFESTS.known(model_id)
        spec = _require_block(manifest, model_id, backend_kind)
        worker_type.refuse_if_larger_than_host(self._backend, model_id, spec.memory_bytes_estimate)
        python = worker_type.require_worker_python(self._config, JOB_TYPE, backend_kind, model_id)
        return Needs(manifest, spec, python, _require_weights(self._config, manifest, spec, model_id))

    def requirements(self, model_id: str, params: ScoreParams) -> tuple[str, Needs]:
        manifest = MANIFESTS.known(model_id)
        refuse_past_the_ceiling(params, model_id, _require_block(manifest, model_id, self._backend.kind))
        ffmpeg = hosttools.require_ffmpeg(JOB_TYPE, FFMPEG_WHY)
        return ffmpeg, self.loadable(model_id)

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        parsed = parse_params(ScoreParams, params, self.name)
        _, needs = self.requirements(model, parsed)
        self._admit(model, needs, "transcribing with")

    def _admit(self, model: str, needs: Needs, doing: str) -> None:
        self._residency.refuse_if_claimed(f"{doing} {model!r}")
        if self._residency.is_resident(KIND_SCORE, model):
            return
        self._guard(model, needs.spec.memory_bytes_estimate)

    def _worker(self, ctx: JobContext, model: str, needs: Needs) -> workers.WorkerSession:
        return self._session(
            ctx,
            model,
            needs.spec.memory_bytes_estimate,
            lambda: occupy_scorer(self._residency, needs, on_progress=ctx.warming),
        )

    def _transcribe(
        self, ctx: JobContext, model: str, session: workers.WorkerSession, request: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            outcome = session.send(
                request,
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
                on_progress=_Scoring(ctx).progress,
                cancelled=lambda: ctx.cancelled,
                cancel_request={"op": "cancel", "request_id": request["request_id"]},
            )
            (result,) = workers.require_positional_results(outcome, 1, "score")
            return result
        except workers.WorkerError as exc:
            self._forget(ctx, model)
            raise JobError("worker_failed", str(exc)) from None
        except JobCancelled:
            if not session.alive:
                self._forget(ctx, model)
            raise

    def run(self, job: Job, ctx: JobContext) -> None:
        params = ScoreParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        ffmpeg, needs = as_job_error(self.requirements, model, params)
        source = recording.the_one_input(ctx.inputs())
        ctx.progress(0.0, f"decoding {source.name}", stage="audio")
        heard = recording.decode(
            ffmpeg, source, ctx.scratch, params,
            ceiling_s=needs.spec.max_audio_s, model_id=model,
        )
        session = self._worker(ctx, model, needs)
        self._make(ctx, model, session, params, needs, heard)

    def _make(
        self,
        ctx: JobContext,
        model: str,
        session: workers.WorkerSession,
        params: ScoreParams,
        needs: Needs,
        heard: Recording,
    ) -> None:
        paths = {
            name: ctx.scratch / name
            for name in (SCORE_ARTIFACT, INSTRUMENTAL_ARTIFACT, RAW_ARTIFACT, MIDI_ARTIFACT, REPORT_ARTIFACT)
        }
        request = score_request(params, heard, ctx.scratch, paths)
        result = self._transcribe(ctx, model, session, request)
        report = build_report(params, needs, heard, result)
        paths[REPORT_ARTIFACT].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        wrote = result.get("wrote") or {}
        native = result.get("yue2_native") or {}
        if wrote.get("raw") and native.get("ok"):
            ctx.artifact(SCORE_ARTIFACT, _written(paths[SCORE_ARTIFACT], "score"))
            ctx.artifact(INSTRUMENTAL_ARTIFACT, _written(paths[INSTRUMENTAL_ARTIFACT], "instrumental score"))
        if wrote.get("raw"):
            ctx.artifact(RAW_ARTIFACT, _written(paths[RAW_ARTIFACT], "SheetSage2 score"))
        if wrote.get("midi"):
            ctx.artifact(MIDI_ARTIFACT, _written(paths[MIDI_ARTIFACT], "MIDI"))
        ctx.artifact(REPORT_ARTIFACT, paths[REPORT_ARTIFACT])
        if not wrote.get("raw"):
            reason = report["sheetsage2"].get("abc_error") or "it returned no ABC and said nothing"
            raise JobError(
                "score_not_transcribed",
                f"SheetSage2 transcribed {heard.source.name!r} and made no score: {reason}. "
                f"Its MIDI (if any) and {REPORT_ARTIFACT} are kept with the job",
            )
        if not native.get("ok"):
            error = native.get("error") or {}
            raise JobError(
                error.get("code") or "score_not_yue2_native",
                f"{error.get('message') or 'the score is not YuE2-native'}. SheetSage2's "
                f"score is kept as {RAW_ARTIFACT}, and {REPORT_ARTIFACT} records every "
                "label decision and check",
            )
        lengths = report["lengths"]["score"] or {}
        ctx.progress(
            1.0,
            f"score made: {lengths.get('bars')} bars, {lengths.get('seconds')} s",
            stage="done",
        )
        ctx.done_extra(score=report, resident=self._residency.resident_id)


def _written(path: Path, what: str) -> Path:
    if not path.is_file():
        raise JobError(
            "worker_failed",
            f"the score worker said it wrote the {what} at {path} and there is no file "
            "there; the worker log says why",
        )
    return path


def score_request(
    params: ScoreParams, heard: Recording, scratch: Path, paths: dict[str, Path]
) -> dict[str, Any]:
    return {
        "op": "score",
        "request_id": uuid.uuid4().hex,
        "audio_path": str(heard.path),
        "samples": heard.samples,
        "keep_harmony": params.keep_harmony,
        "prompts": params.prompts,
        "output_dir": str(scratch / SHEETSAGE2_DIR),
        "score_path": str(paths[SCORE_ARTIFACT]),
        "instrumental_path": str(paths[INSTRUMENTAL_ARTIFACT]),
        "raw_path": str(paths[RAW_ARTIFACT]),
        "midi_path": str(paths[MIDI_ARTIFACT]),
    }


def build_report(
    params: ScoreParams, needs: Needs, heard: Recording, result: dict[str, Any]
) -> dict[str, Any]:
    """`score.json` and the `done` event's `score`: what ran, what SheetSage2 said, what
    was done to its score and what came of it. Every field is a fact from the run."""
    spec = needs.spec
    windows = list(result.get("windows") or [])
    wrote = result.get("wrote") or {}
    sheetsage2 = dict(result.get("sheetsage2") or {})
    return {
        "model": needs.manifest.id,
        "hf_repo": spec.hf_repo,
        "revision": spec.revision,
        "encoder": {
            "hf_repo": spec.encoder.hf_repo,
            "revision": spec.encoder.revision,
            "companion": ENCODER_COMPANION,
        },
        "backend": spec.backend,
        "engine": spec.engine,
        "device": spec.device,
        "dtype": result.get("dtype"),
        "task": params.task,
        "keep_harmony": params.keep_harmony,
        "prompts": params.prompts,
        "recording": heard.to_dict(),
        "windows": windows,
        "token_cap": result.get("token_cap"),
        "windows_at_token_limit": [w["window"] for w in windows if w.get("reached_token_limit")],
        "warnings": sheetsage2.get("warnings", []),
        "sheetsage2": sheetsage2,
        "yue2_native": result.get("yue2_native"),
        "lengths": result.get("lengths") or {"sheetsage2": None, "score": None, "instrumental": None},
        "artifacts": {
            "score": SCORE_ARTIFACT if wrote.get("score") else None,
            "instrumental": INSTRUMENTAL_ARTIFACT if wrote.get("instrumental") else None,
            "sheetsage2": RAW_ARTIFACT if wrote.get("raw") else None,
            "midi": MIDI_ARTIFACT if wrote.get("midi") else None,
            "report": REPORT_ARTIFACT,
        },
        "peak_gpu_mib": sheetsage2.get("peak_gpu_mib"),
        "elapsed_seconds": sheetsage2.get("elapsed_seconds"),
        "seconds": result.get("seconds"),
        "stage_seconds": result.get("stage_seconds"),
        "peak_bytes": result.get("peak_bytes"),
        "stage_peak_bytes": result.get("stage_peak_bytes"),
        "memory_bytes_estimate": spec.memory_bytes_estimate,
        "memory_basis": spec.memory_basis,
        "verified": spec.verified,
        "versions": result.get("versions"),
        "licence": needs.manifest.licence,
    }


class LoadScoreParams(BaseModel):
    """`params` for a load-score job: put a score model on the card."""

    model_config = ConfigDict(extra="forbid")


class LoadScoreJobType(ScoreJobType):

    name = LOAD_SCORE.name

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        parse_params(LoadScoreParams, params, self.name)
        self._admit(model, self.loadable(model), "loading")

    def run(self, job: Job, ctx: JobContext) -> None:
        LoadScoreParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        needs = as_job_error(self.loadable, model)
        ctx.progress(0.0, f"loading {model}")
        self._worker(ctx, model, needs)
        ctx.progress(1.0, f"{model} is resident")
        ctx.done_extra(resident=self._residency.resident_id)


class UnloadScoreJobType(UnloadJobType):
    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        super().__init__(
            UNLOAD_SCORE,
            residency,
            describe=lambda: _descriptors(config, residency),
            provenance=lambda model: MANIFESTS.provenance(config.backend_kind, model),
        )


JOB_TYPES: tuple[JobTypeBinding, ...] = (
    JobTypeBinding(
        SCORE_JOB,
        lambda wiring: ScoreJobType(wiring.config, wiring.backend, wiring.residency),
    ),
    JobTypeBinding(
        UNLOAD_SCORE,
        lambda wiring: UnloadScoreJobType(wiring.config, wiring.backend, wiring.residency),
    ),
    JobTypeBinding(
        LOAD_SCORE,
        lambda wiring: LoadScoreJobType(wiring.config, wiring.backend, wiring.residency),
    ),
)
