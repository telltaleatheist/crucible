from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from ... import hosttools, workers
from ...backend import CUDA_LINUX
from ...cardkinds import KIND_DENOISE
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
from ..unload import UnloadJobType

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

NO_PARAMS = (
    "denoise takes three params, `stems` (\"primary\" or \"all\"), `format` "
    "(\"wav\", \"flac\" or \"mp3\") and `hop_s` (seconds between the starts of "
    "two windows, 0.5 up to the separator's own default) — every other separation knob "
    "is an engine default this server does not put on the wire "
    "(docs/internals/asr-and-align.md)"
)

# The smallest hop a job may ask for. The cost is about inversely proportional to the hop
# (8 s separated a 224 s song in about 8 s on the 3090 Ti), so 0.5 s is about 16 times the
# default's work: every sample under 22 windows of an 11 s model. The largest is the
# manifest's own hop_s, which is never past its window, so no sample goes uncovered.
HOP_FLOOR_S = 0.5

# Which stems a run publishes. "primary" is the manifest's answer (the dry voice, the
# vocals) and is what a book's denoise wants; "all" also returns what was separated
# from it (the instrumental, the noise), so a caller can keep either side - talking
# over a music video with its music taken out, or the backing track without the
# singer (Owen, 2026-10-04).
PRIMARY_STEM_ONLY = "primary"
EVERY_STEM = "all"

# What the separator itself writes, always: lossless, so the one encode a client's
# `format` asks for happens once, after separation (stemaudio.finish).
SEPARATOR_FORMAT = "WAV"

# A stem's container: the same three an audio job offers (jobs/audio/params.FORMATS),
# written with the same encodings (workerio.write_audio: 24-bit FLAC, 192 kbps CBR mp3).
# wav is the default and is the separator's own file, untouched, as before formats.
STEM_FORMATS: tuple[str, ...] = ("wav", "flac", "mp3")

FFMPEG_WHY = (
    "decodes every input that is not WAV or FLAC at the separator's own rate with ffmpeg "
    "and probes it with ffprobe — `crucible install rvc` places Crucible's pinned build"
)

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

    stems: Literal["primary", "all"] = Field(
        default=PRIMARY_STEM_ONLY,
        description='"primary" returns only the separator\'s answer (the vocals, the dry '
        'voice); "all" also returns what was separated from it (the instrumental, the '
        "noise), the primary first.",
    )
    format: Literal["wav", "flac", "mp3"] = Field(
        default="wav",
        description='Every stem\'s container: "wav" (the default, the separator\'s own '
        '16-bit WAV), "flac" (24-bit) or "mp3" (192 kbps CBR), as an audio job writes '
        "them. The separator always works on lossless audio; this is the one encode.",
    )
    hop_s: float | None = Field(
        default=None,
        description="Seconds between the starts of two of the separator's windows (11 s "
        "for most separators, 8 s for `denoise-roformer` and `bleed-suppressor-roformer`). "
        "Smaller means more windows over every sample, so window seams and bleed are "
        "averaged out, and proportionally more time. From 0.5 up to the separator's own "
        "default, which is what a job that sends none runs at (8 for every separator "
        "here); outside that is refused `denoise_hop_out_of_range`.",
    )


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


def ffmpeg_tools() -> dict[str, str | None]:
    return {"ffmpeg": hosttools.ffmpeg_path(), "ffprobe": hosttools.ffprobe_path()}


def _require_ffmpeg() -> dict[str, str]:
    found = ffmpeg_tools()
    absent = [tool for tool, path in found.items() if path is None]
    if absent:
        raise ApiError(
            409,
            "ffmpeg_missing",
            f"this server has no {' and no '.join(absent)}; denoise {FFMPEG_WHY}. "
            + hosttools.searched_note(),
            {"missing": absent, "path": hosttools.search_path()},
        )
    return {tool: str(path) for tool, path in found.items()}


def resolve_hop(params: DenoiseParams, manifest: DenoiseManifest) -> float:
    """The hop a job runs at: the one it asked for, or the separator's own default."""
    if params.hop_s is None:
        return float(manifest.hop_s)
    if not HOP_FLOOR_S <= params.hop_s <= manifest.hop_s:
        raise ApiError(
            400,
            "denoise_hop_out_of_range",
            f"hop_s {params.hop_s} is outside what {manifest.id} takes: from "
            f"{HOP_FLOOR_S} s up to its own default of {manifest.hop_s} s. A smaller "
            "hop puts more windows over every sample and costs proportionally more "
            f"time ({HOP_FLOOR_S} s is about {manifest.hop_s / HOP_FLOOR_S:.0f}x the "
            "default's); a larger one than the default buys nothing this server offers",
            {
                "hop_s": params.hop_s,
                "min": HOP_FLOOR_S,
                "max": manifest.hop_s,
                "model": manifest.id,
            },
        )
    return float(params.hop_s)


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
        absent_tools = [tool for tool, path in ffmpeg_tools().items() if path is None]
        if absent_tools:
            return JobTypeStatus(
                ready=False,
                detail=(
                    f"{env.detail}; but there is no {' and no '.join(absent_tools)}: "
                    f"denoise {FFMPEG_WHY}"
                ),
            )
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
    ) -> tuple[DenoiseManifest, DenoiseBackendSpec, Path, Path, dict[str, str]]:
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
        return manifest, spec, python, root, _require_ffmpeg()

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        parsed = parse_params(DenoiseParams, params, self.name, lead=NO_PARAMS)
        manifest, spec, _python, _root, _tools = self.requirements(model)
        resolve_hop(parsed, manifest)
        self._residency.refuse_if_claimed(f"denoising with {model!r}")
        if self._residency.is_resident(KIND_DENOISE, model):
            return
        self._guard(model, spec.memory_bytes_estimate)


    def run(self, job: Job, ctx: JobContext) -> None:
        params = DenoiseParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        manifest, spec, python, root, tools = as_job_error(self.requirements, model)
        hop_s = as_job_error(resolve_hop, params, manifest)

        source = self._input(ctx)
        output_dir = ctx.scratch / "stems"
        work_dir = ctx.scratch / "separating"
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
            "work_dir": str(work_dir),
            "separator_format": SEPARATOR_FORMAT,
            "output_format": params.format,
            "sample_rate": manifest.sample_rate,
            "hop_s": hop_s,
            "ffmpeg": tools["ffmpeg"],
            "ffprobe": tools["ffprobe"],
        }

        def on_ready(message: dict[str, Any]) -> None:
            ctx.warming(
                f"{source.name}: {message['seconds']}s of "
                f"{message['channels']}-channel {message['codec']} at "
                f"{message['sample_rate']} Hz, through {manifest.display}, a window "
                f"every {hop_s:g} s; {_conversion(message, manifest.sample_rate)}"
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
        silent = results[0]["silent_stems"]
        primary = self._check(manifest, outcome.ready, stems)
        published = [primary]
        if params.stems == EVERY_STEM:
            published += [stem for stem in stems if stem is not primary]
        for stem in published:
            self._same_timeline(manifest, outcome.ready, stem)
            ctx.artifact(stem["name"], output_dir / stem["name"])
        ctx.progress(
            1.0,
            f"{len(stems)} stem(s) from {source.name} through {manifest.display}",
            stage="separating",
        )
        ready = outcome.ready
        result = results[0]
        ctx.done_extra(
            primary_stem=primary["name"],
            primary_is=manifest.primary_is,
            secondary_is=manifest.secondary_is,
            stems=[stem["name"] for stem in stems],
            sample_rate=primary["sample_rate"],
            frames=primary["frames"],
            format=params.format,
            input={
                key: ready[key]
                for key in ("codec", "container", "sample_rate", "frames", "channels")
            },
            separated={
                "sample_rate": primary["separated_rate"],
                "frames": primary["separated_frames"],
                "decoded": ready["decoded"],
                "resampled": ready["resampled"],
                "said": _conversion(ready, manifest.sample_rate),
                "silent_stems": silent,
                "silent_said": _silence(silent),
                "hop_s": hop_s,
                "hop_s_default": manifest.hop_s,
                "window_s": result["window_s"],
                "windows_said": _windows(hop_s, result["window_s"]),
            },
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
        wanted = manifest.primary_stem.lower()
        marker = f"({wanted})"
        hits = [stem for stem in stems if _stem_label(stem["name"]) == wanted]
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
        return hits[0]

    @staticmethod
    def _same_timeline(
        manifest: DenoiseManifest, ready: dict[str, Any], stem: dict[str, Any]
    ) -> None:
        """A published stem lines up sample for sample with the input it came from: the
        separator kept the rate and length it was given, and the stem went back to the
        client's own rate and length after it."""
        if stem["separated_rate"] != manifest.sample_rate:
            raise JobError(
                "denoise_resampled",
                f"{stem['name']} came out of the separator at "
                f"{stem['separated_rate']} Hz and it was given "
                f"{manifest.sample_rate} Hz — the model resampled it, which "
                "invalidates every sample offset the caller sliced by",
            )
        if stem["separated_frames"] != ready["separator_frames"]:
            raise JobError(
                "denoise_length_changed",
                f"{stem['name']} came out of the separator {stem['separated_frames']} "
                f"frames long and it was given {ready['separator_frames']} — the model "
                "changed the length. Slicing a stem back at the input's offsets is only "
                "safe because it does not",
            )
        if stem["sample_rate"] != ready["sample_rate"]:
            raise JobError(
                "denoise_resampled",
                f"{stem['name']} is {stem['sample_rate']} Hz and the input was "
                f"{ready['sample_rate']} Hz — it was not returned to the input's rate",
            )
        if stem["frames"] != ready["frames"]:
            raise JobError(
                "denoise_length_changed",
                f"{stem['name']} is {stem['frames']} frames and the input was "
                f"{ready['frames']} — it was not returned to the input's length",
            )


# audio-separator names a stem `<input name>_(<stem>)_<model name>.<ext>`, and no model name
# here has a parenthesis in it, so the stem is the LAST parenthesised label. The input's own
# name can carry one too: an instrumental chained into the bleed suppressor is
# `song_(Instrumental)_mel_band_roformer_instrumental_becruily.wav`, and both of ITS stems
# then contain "(instrumental)".
_STEM_LABEL = re.compile(r"_\(([^()]*)\)_[^()]*$")


def _stem_label(name: str) -> str | None:
    found = _STEM_LABEL.search(name)
    return None if found is None else found.group(1).lower()


def _windows(hop_s: float, window_s: float) -> str:
    """How the windows lay over the input, said for the record."""
    return (
        f"{window_s:g} s windows, one every {hop_s:g} s: every sample under about "
        f"{window_s / hop_s:.1f} windows, averaged"
    )


def _conversion(ready: dict[str, Any], separator_rate: int) -> str:
    """What was done to the input on its way to the separator, said for the record."""
    if not ready["decoded"]:
        return "the separator read the input as sent"
    if not ready["resampled"]:
        return f"decoded with ffmpeg to {separator_rate} Hz WAV for the separator"
    return (
        f"decoded with ffmpeg and resampled {ready['sample_rate']} Hz -> "
        f"{separator_rate} Hz for the separator, every stem resampled back to "
        f"{ready['sample_rate']} Hz and {ready['frames']} frames"
    )


def _silence(silent: list[str]) -> str | None:
    """The stems the separator found silent and did not write, said for the record."""
    if not silent:
        return None
    return (
        f"the separator found {', '.join(silent)} silent and wrote no file for "
        f"{'it' if len(silent) == 1 else 'them'} (audio-separator skips a stem whose peak "
        "is under 1e-6); returned as digital silence on the input's timeline, as a clean "
        "instrumental's vocals are"
    )


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
