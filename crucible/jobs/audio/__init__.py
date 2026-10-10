from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from pydantic import BaseModel, ConfigDict

from ... import audioweights, jobenv, weights, workers
from ...audiomodels import (
    STABLE_AUDIO_3,
    YUE2,
    AudioBackendSpec,
    AudioManifest,
    AudioManifestError,
    HeldNeed,
    load_all_audio_manifests,
)
from ...cardkinds import KIND_AUDIO
from ...clock import utcnow
from ...config import Config
from ...errors import ApiError, JobCancelled, JobError
from ...jobtypes import AUDIO_JOB, LOAD_AUDIO, UNLOAD_AUDIO
from ...manifests import fingerprint
from ...residency import (
    DEFAULT_READY_TIMEOUT_SECONDS,
    Occupant,
    Residency,
    ResidentAudio,
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
from .params import MAX_SEED, AudioParams, Settled, refuse_what_the_model_cannot_take, settle

__all__ = [
    "JOB_TYPES",
    "AudioJobType",
    "AudioParams",
    "LoadAudioJobType",
    "LoadAudioParams",
    "UnloadAudioJobType",
    "occupy_audio",
]

JOB_TYPE = AUDIO_JOB.name

ARTIFACT_STEM = "audio"

SCORE_ARTIFACT = "score.abc"

READY_SILENCE_TIMEOUT_SECONDS = 900.0

HERE = Path(__file__).resolve().parent

WORKER_SCRIPTS: dict[str, Path] = {
    STABLE_AUDIO_3: HERE / "stable_audio_worker.py",
    YUE2: HERE / "yue2_worker.py",
}

AUDIO_MAGIC: dict[str, tuple[bytes, int]] = {
    "flac": (b"fLaC", 0),
    "wav": (b"WAVE", 8),
    # An MPEG-1 Layer III frame header with no CRC: what libsndfile's LAME writer
    # begins with (it writes no ID3 tag).
    "mp3": (b"\xff\xfb", 0),
}

WORKER_ENVIRONMENT = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def artifact_name(audio_format: str) -> str:
    return f"{ARTIFACT_STEM}.{audio_format}"


def worker_script(engine: str) -> Path:
    found = WORKER_SCRIPTS.get(engine)
    if found is None:
        raise JobError(
            "backend_unsupported",
            f"there is no audio worker for engine {engine!r}; this build runs "
            f"{sorted(WORKER_SCRIPTS)}",
        )
    return found


def require_audio_python(config: Config, spec: AudioBackendSpec, model_id: str) -> Path:
    try:
        env = jobenv.audio_env(spec.engine, spec.backend)
        return jobenv.require_env(config.home, env, spec.backend)
    except jobenv.EnvError as exc:
        raise ApiError(
            409,
            "env_missing",
            f"cannot run {model_id!r}: {exc}. `crucible install audio` builds it",
            {"model": model_id, "env": str(config.home / "envs" / f"audio-{spec.engine}")},
        ) from None


def require_audio_weights(
    config: Config, manifest: AudioManifest, spec: AudioBackendSpec
) -> Path:
    try:
        return audioweights.require_installed(config, manifest, spec).path
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


def refuse_input_files(ctx: JobContext) -> None:
    inputs = ctx.inputs()
    if inputs:
        raise JobError(
            "invalid_inputs",
            f"this job carries input(s) {sorted(inputs)}; an audio job makes sound "
            "from words alone and reads no files. Send it without inputs",
        )


def check_audio_file(path: Path, audio_format: str) -> None:
    magic, at = AUDIO_MAGIC[audio_format]
    head = path.read_bytes()[: at + len(magic)] if path.is_file() else b""
    if head[at : at + len(magic)] != magic:
        raise JobError(
            "worker_failed",
            f"the audio worker wrote {path} but it is not a {audio_format.upper()} "
            f"file (it begins {head[:12].hex() or 'with nothing'}); the worker log "
            "says why",
        )


def start_audio_session(
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
            "parts": audioweights.part_dirs(needs.weights_dir, spec),
            "hf_repo": spec.hf_repo,
            "device": spec.device,
            "dtype": spec.dtype,
            "memory_cap_bytes": workers.torch_memory_cap(spec.backend, needs.need_bytes),
            "memory_budget_bytes": needs.need_bytes,
            "low_vram": needs.low_vram,
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


def occupy_audio(
    residency: Residency,
    needs: "Needs",
    *,
    timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
    on_progress: Callable[[str], None] | None = None,
) -> ResidentAudio:
    say = say_to(on_progress)
    manifest, spec = needs.manifest, needs.spec
    loaded: dict[str, Any] = {}

    def start() -> Occupant:
        log_path = residency.log_path_for(manifest.id)
        say(f"starting the {spec.engine} worker for {manifest.id}; log {log_path}")
        session = start_audio_session(
            needs.python, needs, log_path, ready_silence_timeout=timeout, on_ready=loaded.update
        )
        resident = ResidentAudio(
            model_id=manifest.id,
            backend=spec.backend,
            engine=spec.engine,
            revision=spec.revision,
            fingerprint=fingerprint(manifest.id, spec.revision),
            device=spec.device,
            dtype=spec.dtype,
            versions=dict(loaded.get("versions") or {}),
            memory_bytes_estimate=needs.need_bytes,
            log_path=log_path,
            loaded_at=utcnow(),
        )
        return Occupant(resident, session=session)

    return cast(ResidentAudio, residency.occupy(KIND_AUDIO, manifest.id, start, say=say))


MANIFESTS: ManifestCatalog[AudioManifest] = ManifestCatalog(
    lambda: load_all_audio_manifests(),
    AudioManifestError,
    unreadable_code="audio_manifests_unreadable",
    what="audio manifests",
    unknown="audio model",
)


def _descriptors(config: Config, residency: Residency) -> list[ModelDescriptor]:
    return MANIFESTS.descriptors(
        config.backend_kind,
        installed=lambda manifest, spec: audioweights.installed(config, manifest, spec)
        is not None,
        resident=lambda model_id: residency.is_resident(KIND_AUDIO, model_id),
        estimate=lambda spec: spec.need_on(config.audio_low_vram).bytes,
    )


@dataclass(frozen=True)
class Needs:
    manifest: AudioManifest
    spec: AudioBackendSpec
    python: Path
    weights_dir: Path
    # What this host holds the model against: AudioBackendSpec.need_on, the one rule the
    # capability verdict weighs too.
    need: HeldNeed

    @property
    def need_bytes(self) -> int:
        return self.need.bytes

    @property
    def low_vram(self) -> bool:
        return self.need.low_vram


class _Generation:
    def __init__(self, ctx: JobContext) -> None:
        self._ctx = ctx

    def progress(self, message: dict[str, Any]) -> None:
        stage = str(message.get("stage"))
        if stage == "cancelled":
            return
        step, steps = message.get("step"), message.get("steps")
        fraction = min(1.0, max(0.0, float(message.get("fraction") or 0.0)))
        words = f"{stage}: {step} of {steps}" if steps else stage
        self._ctx.progress(fraction, words, stage=stage, step=step, steps=steps)


def _installed_here(config: Config, manifest: AudioManifest, backend_kind: str) -> bool:
    return audioweights.installed(config, manifest, manifest.spec(backend_kind)) is not None


def _env_status(config: Config, backend_kind: str) -> JobTypeStatus | None:
    try:
        specs = jobenv.audio_envs(backend_kind)
    except jobenv.EnvError as exc:
        return JobTypeStatus(ready=False, detail=str(exc))
    for spec in specs:
        status = jobenv.env_status(config.home, spec, backend_kind)
        if not status.installed:
            return JobTypeStatus(
                ready=False, detail=f"{status.detail}; `crucible install audio` builds it"
            )
    return None


class AudioJobType(ResidentWorker):

    name = JOB_TYPE
    resident_kind = KIND_AUDIO

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
            run_model(model, self.name),
            self._config.backend_kind,
            estimate=lambda spec: spec.need_on(self._config.audio_low_vram).bytes,
        )

    def check(self, backend: Any) -> JobTypeStatus:
        broken = _env_status(self._config, backend.kind)
        if broken is not None:
            return broken
        try:
            manifests = MANIFESTS.all()
        except ApiError as exc:
            return JobTypeStatus(ready=False, detail=exc.message)
        offered = sorted(m.id for m in manifests.values() if m.supports(backend.kind))
        installed = [
            model_id
            for model_id in offered
            if _installed_here(self._config, manifests[model_id], backend.kind)
        ]
        if not installed:
            pulls = " or ".join(f"`crucible models pull {model}`" for model in offered)
            return JobTypeStatus(
                ready=False,
                detail=f"audio envs built; no audio model is installed — {pulls or 'none is declared for this backend'}",
            )
        return JobTypeStatus(ready=True, detail=f"audio envs built; installed: {installed}")

    def _resident_session(self) -> workers.WorkerSession | None:
        return self._residency.audio_session

    def loadable(self, model_id: str) -> Needs:
        backend_kind = self._backend.kind
        manifest = MANIFESTS.known(model_id)
        spec = _require_block(manifest, model_id, backend_kind)
        need = spec.need_on(self._config.audio_low_vram)
        worker_type.refuse_if_larger_than_host(self._backend, model_id, need.bytes)
        python = require_audio_python(self._config, spec, model_id)
        return Needs(
            manifest, spec, python, require_audio_weights(self._config, manifest, spec), need
        )

    def requirements(self, model_id: str, params: AudioParams) -> Needs:
        manifest = MANIFESTS.known(model_id)
        refuse_what_the_model_cannot_take(
            params, manifest, _require_block(manifest, model_id, self._backend.kind)
        )
        return self.loadable(model_id)

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        parsed = parse_params(AudioParams, params, self.name)
        self._admit(model, self.requirements(model, parsed), "making audio with")

    def _admit(self, model: str, needs: Needs, doing: str) -> None:
        self._residency.refuse_if_claimed(f"{doing} {model!r}")
        if self._residency.is_resident(KIND_AUDIO, model):
            return
        self._guard(model, needs.need_bytes)

    def _worker(self, ctx: JobContext, model: str, needs: Needs) -> workers.WorkerSession:
        return self._session(
            ctx,
            model,
            needs.need_bytes,
            lambda: occupy_audio(self._residency, needs, on_progress=ctx.warming),
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
            (result,) = workers.require_positional_results(outcome, 1, "audio")
            return result
        except workers.WorkerError as exc:
            self._forget(ctx, model)
            raise JobError("worker_failed", str(exc)) from None
        except JobCancelled:
            if not session.alive:
                self._forget(ctx, model)
            raise

    def run(self, job: Job, ctx: JobContext) -> None:
        params = AudioParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        needs = as_job_error(self.requirements, model, params)
        refuse_input_files(ctx)
        seed = params.seed if params.seed is not None else secrets.randbelow(MAX_SEED + 1)
        session = self._worker(ctx, model, needs)
        self._make(ctx, model, session, params, needs, settle(params, needs.spec, seed))

    def _make(
        self,
        ctx: JobContext,
        model: str,
        session: workers.WorkerSession,
        params: AudioParams,
        needs: Needs,
        settled: Settled,
    ) -> None:
        output = ctx.scratch / artifact_name(params.format)
        score = ctx.scratch / SCORE_ARTIFACT
        request = generate_request(params, needs, settled, output, score)
        result = self._generate(ctx, model, session, request)
        check_audio_file(output, params.format)
        ctx.artifact(output.name, output)
        wrote_score = bool(result.get("score_path")) and score.is_file()
        if wrote_score:
            ctx.artifact(SCORE_ARTIFACT, score)
        ctx.progress(1.0, made_words(result.get("audio_seconds"), result["decode_stages"]), stage="done")
        ctx.done_extra(
            audio=effective_params(params, needs, settled, result, output.name, wrote_score),
            resident=self._residency.resident_id,
        )


def _require_block(manifest: AudioManifest, model_id: str, backend_kind: str) -> AudioBackendSpec:
    return worker_type.require_block(manifest, model_id, backend_kind, "audio model")


def stages_at_cap(decode_stages: dict[str, Any] | None) -> list[str] | None:
    """The stages that ran to their token cap without the model ending them, in order; []
    when every stage ended itself, None for an engine that decodes no tokens. The job
    still succeeds - the audio is real, only longer than the model meant (Victoria's
    6-minute song, 2026-10-09) - and nothing re-runs it: this is how a client sees it."""
    if decode_stages is None:
        return None
    return [stage for stage, facts in decode_stages.items() if facts["ended"] == "cap"]


def made_words(audio_seconds: Any, decode_stages: dict[str, Any] | None) -> str:
    words = f"{audio_seconds} s of audio made"
    if decode_stages is None:
        return words
    capped = [stage for stage in decode_stages if decode_stages[stage]["ended"] == "cap"]
    if not capped:
        return words
    reached = ", ".join(f"{stage} at its {decode_stages[stage]['cap']}-token cap" for stage in capped)
    return f"{words}; {reached} without ending (see audio.decode_stages)"


def generate_request(
    params: AudioParams, needs: Needs, settled: Settled, output: Path, score: Path
) -> dict[str, Any]:
    return {
        "op": "generate",
        "request_id": uuid.uuid4().hex,
        "kind": needs.manifest.kind,
        "prompt": params.prompt,
        "tags": params.tags,
        "lyrics": params.lyrics,
        "negative_prompt": params.negative_prompt,
        "duration_s": settled.duration_s,
        "seed": settled.seed,
        "steps": settled.steps,
        "cfg": settled.cfg,
        "instrumental": settled.instrumental,
        "sample_rate": needs.spec.sample_rate,
        "channels": needs.spec.channels,
        "format": params.format,
        "output_path": str(output),
        "score_path": str(score) if needs.manifest.takes_lyrics else None,
        "revision": needs.spec.revision,
        "backend": needs.spec.backend,
    }


def effective_params(
    params: AudioParams,
    needs: Needs,
    settled: Settled,
    result: dict[str, Any],
    artifact: str,
    wrote_score: bool,
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
        "prompt": params.prompt,
        "tags": params.tags,
        "lyrics": params.lyrics,
        "duration_s": settled.duration_s,
        "seed": settled.seed,
        "steps": settled.steps,
        "cfg": settled.cfg,
        "instrumental": settled.instrumental,
        "format": params.format,
        "artifact": artifact,
        "score": SCORE_ARTIFACT if wrote_score else None,
        "audio_seconds": result.get("audio_seconds"),
        "sample_rate": result.get("sample_rate"),
        "channels": result.get("channels"),
        "seconds": result.get("seconds"),
        "stage_seconds": result.get("stage_seconds"),
        "peak_bytes": result.get("peak_bytes"),
        "stage_peak_bytes": result.get("stage_peak_bytes"),
        "memory_bytes_estimate": needs.need_bytes,
        "memory_basis": spec.memory_basis,
        "low_vram": needs.low_vram,
        "versions": result.get("versions"),
        "notes": result.get("notes"),
        "decode_stages": result["decode_stages"],
        "stages_at_cap": stages_at_cap(result["decode_stages"]),
    }


class LoadAudioParams(BaseModel):
    """`params` for a load-audio job: warm the audio model up."""

    model_config = ConfigDict(extra="forbid")



class LoadAudioJobType(AudioJobType):

    name = LOAD_AUDIO.name

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        parse_params(LoadAudioParams, params, self.name)
        self._admit(model, self.loadable(model), "loading")

    def run(self, job: Job, ctx: JobContext) -> None:
        LoadAudioParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        needs = as_job_error(self.loadable, model)
        ctx.progress(0.0, f"loading {model}")
        self._worker(ctx, model, needs)
        ctx.progress(1.0, f"{model} is resident")
        ctx.done_extra(resident=self._residency.resident_id)


class UnloadAudioJobType(UnloadJobType):
    def __init__(self, config: Config, backend: Any, residency: Residency) -> None:
        super().__init__(
            UNLOAD_AUDIO,
            residency,
            describe=lambda: _descriptors(config, residency),
            provenance=lambda model: MANIFESTS.provenance(config.backend_kind, model),
        )


JOB_TYPES: tuple[JobTypeBinding, ...] = (
    JobTypeBinding(
        AUDIO_JOB,
        lambda wiring: AudioJobType(wiring.config, wiring.backend, wiring.residency),
    ),
    JobTypeBinding(
        UNLOAD_AUDIO,
        lambda wiring: UnloadAudioJobType(wiring.config, wiring.backend, wiring.residency),
    ),
    JobTypeBinding(
        LOAD_AUDIO,
        lambda wiring: LoadAudioJobType(wiring.config, wiring.backend, wiring.residency),
    ),
)
