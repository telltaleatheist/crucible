from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

from ... import accelerator, hosttools, jobenv, weights, workers
from ...asrmodels import (
    QWEN_ASR_ENGINES,
    QWEN_CONTEXT_MAX_TOKENS,
    QWEN_PIECE_MAX_SECONDS,
    AsrManifest,
    AsrManifestError,
    load_all_asr_manifests,
)
from ...cardfacts import card_for
from ...config import Config
from ...errors import ApiError, JobError
from ...jobtypes import ASR_JOB
from ...journal import Identity
from ...manifests import fingerprint
from .. import worker_type
from ..align import QWEN3_LANGUAGES
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
from . import qwen, speechonly
from .document import progress_decoding, transcript_document, worker_failed

__all__ = ["JOB_TYPES", "AsrJobType", "AsrParams"]

JOB_TYPE = ASR_JOB.name

FFMPEG_WHY = (
    "decodes every input through it — faster-whisper's own PyAV decoder "
    "silently truncates some m4b files, which ends a transcript hours early "
    "with no error."
)

WINDOW_SECONDS = 900
OVERLAP_SECONDS = 15

READY_SILENCE_TIMEOUT_SECONDS = 900.0

OVERLAP_TOLERANCE_SECONDS = 0.1

COMPUTE_TYPE_FOR_ENGINE: dict[str, str] = {
    "faster-whisper": "float16",
    "mlx-whisper": "float16",
}
DEVICE_FOR_ENGINE: dict[str, str] = {
    "faster-whisper": "cuda",
    "mlx-whisper": "metal",
}

WORKER_SCRIPT_FOR_ENGINE: dict[str, Path] = {
    "faster-whisper": Path(__file__).resolve().parent / "worker.py",
    "mlx-whisper": Path(__file__).resolve().parent / "mlx_worker.py",
}

ENGINES_WITHOUT_VAD: frozenset[str] = frozenset({"mlx-whisper", *QWEN_ASR_ENGINES})

ENV_FOR_ENGINE: dict[str, str] = {
    "faster-whisper": "asr",
    "mlx-whisper": "asr",
    "vllm": "llm",
    "mlx-audio": "llm",
    "qwen-asr": "align",
}

CONTEXT_MAX_CHARS = 8192

_CHAT_CONTROL = re.compile(r"<\|[^|]*\|>|<asr_text>")


def _for_engine(table: dict[str, Any], engine: str, what: str) -> Any:
    found = table.get(engine)
    if found is None:
        raise JobError(
            "engine_unsupported",
            f"there is no {what} for asr engine {engine!r}; this build runs "
            f"{sorted(table)}",
        )
    return found

WHISPER_LANGUAGES = frozenset(
    """af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo
    fr gl gu ha haw he hi hr ht hu hy id is it ja jw ka kk km kn ko la lb ln lo
    lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si
    sk sl sn so sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi yi yo zh
    yue""".split()
)

AUTO_LANGUAGE = "auto"

class AsrParams(BaseModel):
    """`params` for an asr job. Unknown keys are refused, not ignored."""

    model_config = ConfigDict(extra="forbid")

    language: str = Field(
        description="The spoken language as an ISO code, e.g. `en`. `auto` asks "
        "Whisper to detect it; Qwen3-ASR takes only en, de, fr, es, it, pt, ru, ja, "
        "ko, zh or yue."
    )
    vad_filter: bool = Field(
        description="true runs faster-whisper's own voice-activity filter; refused "
        "on engines without one (mlx-whisper, Qwen3-ASR) and together with "
        "`speech_only: true`."
    )
    word_timestamps: bool = Field(
        description="true adds `words` with their own start and end seconds to each "
        "segment; on Qwen3-ASR it runs the forced aligner after transcription."
    )
    initial_prompt: StrictStr | None = Field(
        default=None,
        description="Whisper only: text the model is primed with as if it were the "
        "transcript so far (a title, the names in it); not blank.",
    )
    context: StrictStr | None = Field(
        default=None,
        description="Qwen3-ASR only: the instruction and vocabulary it reads in its "
        "system turn before every piece; at most 8192 characters, not blank.",
    )
    piece_s: float | None = Field(
        default=None,
        description="Qwen3-ASR only: the longest piece in seconds the audio is cut "
        "into, 5 to 180; null is 30.",
    )
    overlap_s: float | None = Field(
        default=None,
        description="Qwen3-ASR only: seconds of audio each piece also hears on each "
        "side, 0 to 5 and under half a piece; null is 0.4 with word timestamps, "
        "else 0. Above 0 needs `word_timestamps`.",
    )
    speech_only: StrictBool | None = Field(
        default=None,
        description="true takes stretches without speech out before transcribing "
        "and lists them in the transcript's `removed`, keeping the original "
        "timeline. Null follows `not vad_filter`, so it is on unless `vad_filter` "
        "is true.",
    )
    speech_threshold: float | None = Field(
        default=None,
        description="With `speech_only`: the detector score at which a frame counts "
        "as speech, 0.1 to 0.7 (lower keeps more); null is 0.3.",
    )
    speech_pad_s: float | None = Field(
        default=None,
        description="With `speech_only`: seconds of audio kept either side of "
        "speech, 0.1 to 2; null is 0.3.",
    )
    speech_min_gap_s: float | None = Field(
        default=None,
        description="With `speech_only`: the shortest stretch without speech that "
        "is taken out, in seconds, 1 to 60; null is 2.",
    )
    resume: StrictStr | None = Field(
        default=None,
        description="The `resume_id` of an earlier run of this same job, to read its "
        "finished pieces back instead of redoing them. Qwen3-ASR only "
        "(`resume_unsupported` on Whisper).",
    )

    @field_validator("speech_threshold")
    @classmethod
    def speech_threshold_in_range(cls, value: float | None) -> float | None:
        if value is not None and not (
            speechonly.MIN_THRESHOLD <= value <= speechonly.MAX_THRESHOLD
        ):
            raise ValueError(
                f"speech_threshold is {value}; it is {speechonly.MIN_THRESHOLD:g} to "
                f"{speechonly.MAX_THRESHOLD:g}, the detector score at which a frame "
                "counts as speech (lower keeps more). Send null for this server's "
                f"default ({speechonly.DEFAULT_THRESHOLD:g})"
            )
        return value

    @field_validator("speech_pad_s")
    @classmethod
    def speech_pad_in_range(cls, value: float | None) -> float | None:
        if value is not None and not (
            speechonly.MIN_PAD_S <= value <= speechonly.MAX_PAD_S
        ):
            raise ValueError(
                f"speech_pad_s is {value}; it is {speechonly.MIN_PAD_S:g} to "
                f"{speechonly.MAX_PAD_S:g} seconds of audio kept either side of "
                "speech, and less than 0.1 cuts against a sentence's first word. "
                f"Send null for this server's default ({speechonly.DEFAULT_PAD_S:g})"
            )
        return value

    @field_validator("speech_min_gap_s")
    @classmethod
    def speech_min_gap_in_range(cls, value: float | None) -> float | None:
        if value is not None and not (
            speechonly.MIN_MIN_GAP_S <= value <= speechonly.MAX_MIN_GAP_S
        ):
            raise ValueError(
                f"speech_min_gap_s is {value}; it is {speechonly.MIN_MIN_GAP_S:g} to "
                f"{speechonly.MAX_MIN_GAP_S:g} seconds, the shortest stretch without "
                "speech that is taken out. Send null for this server's default "
                f"({speechonly.DEFAULT_MIN_GAP_S:g})"
            )
        return value

    @model_validator(mode="after")
    def speech_knobs_need_speech_only(self) -> "AsrParams":
        if self.speech_only is None:
            self.speech_only = not self.vad_filter
            return self
        if self.speech_only:
            if self.vad_filter:
                raise ValueError(
                    "speech_only and vad_filter are two speech detectors; send one. "
                    "speech_only is Crucible's, keeps the original timeline and "
                    "lists what it removed; vad_filter is faster-whisper's own"
                )
            return self
        stray = [
            name
            for name in ("speech_threshold", "speech_pad_s", "speech_min_gap_s")
            if getattr(self, name) is not None
        ]
        if stray:
            raise ValueError(
                f"{', '.join(stray)} tune(s) speech_only, and speech_only is false, "
                "so they would change nothing. Send speech_only: true, or leave "
                f"{'them' if len(stray) > 1 else 'it'} out"
            )
        return self

    def speech_settings(self, weights: Path) -> dict[str, Any] | None:
        if not self.speech_only:
            return None
        return {
            "weights": str(weights),
            "sha256": hosttools.SILERO_VAD.sha256,
            "threshold": (
                self.speech_threshold
                if self.speech_threshold is not None
                else speechonly.DEFAULT_THRESHOLD
            ),
            "pad_s": (
                self.speech_pad_s if self.speech_pad_s is not None else speechonly.DEFAULT_PAD_S
            ),
            "min_gap_s": (
                self.speech_min_gap_s
                if self.speech_min_gap_s is not None
                else speechonly.DEFAULT_MIN_GAP_S
            ),
        }

    @field_validator("piece_s")
    @classmethod
    def piece_in_range(cls, value: float | None) -> float | None:
        if value is not None and not (
            qwen.MIN_PIECE_S <= value <= QWEN_PIECE_MAX_SECONDS
        ):
            raise ValueError(
                f"piece_s is {value}; a piece is {qwen.MIN_PIECE_S:g} to "
                f"{QWEN_PIECE_MAX_SECONDS} seconds, the longest the Qwen engines "
                "were sized for. Send null for this server's default "
                f"({qwen.DEFAULT_PIECE_S:g})"
            )
        return value

    @field_validator("overlap_s")
    @classmethod
    def overlap_in_range(cls, value: float | None) -> float | None:
        if value is not None and not (0.0 <= value <= qwen.MAX_OVERLAP_S):
            raise ValueError(
                f"overlap_s is {value}; it is 0 to {qwen.MAX_OVERLAP_S:g} seconds of "
                "real audio on each side of a piece. Send null for this server's "
                "default"
            )
        return value

    @model_validator(mode="after")
    def overlap_under_half_a_piece(self) -> "AsrParams":
        piece = self.piece_s if self.piece_s is not None else qwen.DEFAULT_PIECE_S
        overlap = self.overlap_s if self.overlap_s is not None else 0.0
        if overlap * 2 >= piece:
            raise ValueError(
                f"overlap_s {overlap:g} on both sides of a {piece:g} s piece hears "
                "more of its neighbours than of itself; keep the overlap under half "
                "the piece"
            )
        return self

    def piece_seconds(self) -> float:
        return self.piece_s if self.piece_s is not None else qwen.DEFAULT_PIECE_S

    def overlap_seconds(self) -> float:
        if self.overlap_s is not None:
            return self.overlap_s
        return qwen.DEFAULT_OVERLAP_S if self.word_timestamps else 0.0

    @field_validator("initial_prompt")
    @classmethod
    def prompt_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and value.strip() == "":
            raise ValueError(
                "initial_prompt is blank; send null for no prompt, or the text "
                "whisper should be primed with (a title, the names in it)"
            )
        return value

    @field_validator("context")
    @classmethod
    def context_is_plain_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value.strip() == "":
            raise ValueError(
                "context is blank; send null for no context, or the instruction "
                "and vocabulary Qwen3-ASR should read before the audio"
            )
        if len(value) > CONTEXT_MAX_CHARS:
            raise ValueError(
                f"context is {len(value)} characters; Qwen3-ASR is given at most "
                f"{QWEN_CONTEXT_MAX_TOKENS} tokens of it, which is well under "
                f"{CONTEXT_MAX_CHARS} characters. Send the instruction and the "
                "names, not the document"
            )
        found = _CHAT_CONTROL.search(value)
        if found is not None:
            raise ValueError(
                f"context contains {found.group(0)!r}, one of the chat "
                "template's own control tokens; the context is placed verbatim "
                "inside the system turn, where that would end the turn"
            )
        return value

    @field_validator("language")
    @classmethod
    def known_language(cls, value: str) -> str:
        if value == AUTO_LANGUAGE or value in WHISPER_LANGUAGES:
            return value
        raise ValueError(
            f"{value!r} is not a language faster-whisper knows; send a code from "
            f"{sorted(WHISPER_LANGUAGES)} or {AUTO_LANGUAGE!r} to have it detected"
        )

    def whisper_language(self) -> str | None:
        return None if self.language == AUTO_LANGUAGE else self.language


MANIFESTS: ManifestCatalog[AsrManifest] = ManifestCatalog(
    lambda: load_all_asr_manifests(),
    AsrManifestError,
    unreadable_code="asr_manifests_unreadable",
    what="ASR model manifests",
    unknown="ASR manifest for model",
    offer_in_details=True,
)


def _most_a_job_needs(spec: Any, backend_kind: str) -> int:
    if spec.engine in QWEN_ASR_ENGINES:
        return qwen.need_bytes(spec, backend_kind, with_aligner=True)
    return spec.memory_bytes_estimate


def _env_spec(env: str, backend_kind: str) -> jobenv.EnvSpec:
    if env == "llm":
        return jobenv.llm_env(backend_kind)
    return jobenv.worker_env(env, backend_kind)


def _env_ready(config: Config, env: str, backend_kind: str) -> tuple[bool, str]:
    try:
        status = jobenv.env_status(
            config.home, _env_spec(env, backend_kind), backend_kind
        )
    except jobenv.EnvError as exc:
        return False, str(exc)
    return status.installed, status.detail


def _python_for(config: Config, engine: str, backend_kind: str, model_id: str) -> Path:
    env = _for_engine(ENV_FOR_ENGINE, engine, "env")
    try:
        spec = _env_spec(env, backend_kind)
        return jobenv.require_env(config.home, spec, backend_kind)
    except jobenv.EnvError as exc:
        directory = config.home / "envs" / env
        raise ApiError(
            409,
            "env_missing",
            f"cannot run {model_id!r}: its engine {engine!r} runs in the {env} "
            f"env, and {exc}",
            {"model": model_id, "env": str(directory)},
        ) from None


class AsrJobType:

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
        backend_kind = self._config.backend_kind
        return MANIFESTS.descriptors(
            backend_kind,
            installed=lambda manifest, spec: weights.installed(
                self._config, manifest, spec
            )
            is not None,
            resident=lambda model_id: False,
            estimate=lambda spec: _most_a_job_needs(spec, backend_kind),
        )

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        manifest = MANIFESTS.known(run_model(model, self.name))
        spec = manifest.backends.get(self._config.backend_kind)
        if spec is None:
            return {
                "id": model,
                "revision": None,
                "fingerprint": None,
                "engine": None,
                "hf_repo": None,
            }
        return {
            "id": model,
            "revision": spec.revision,
            "fingerprint": fingerprint(model, spec.revision),
            "engine": spec.engine,
            "hf_repo": spec.hf_repo,
        }

    def vram_estimate(self, model: str | None) -> int:
        manifest = MANIFESTS.known(run_model(model, self.name))
        if not manifest.supports(self._config.backend_kind):
            return 0
        return _most_a_job_needs(
            manifest.spec(self._config.backend_kind), self._config.backend_kind
        )

    def check(self, backend: Any) -> JobTypeStatus:
        if hosttools.ffmpeg_path() is None:
            return JobTypeStatus(
                ready=False,
                detail="there is no ffmpeg on PATH, and asr decodes every input "
                "through it",
            )
        try:
            manifests = MANIFESTS.all()
        except ApiError as exc:
            return JobTypeStatus(ready=False, detail=exc.message)
        envs: dict[str, tuple[bool, str]] = {}
        for env in sorted(set(ENV_FOR_ENGINE.values())):
            envs[env] = _env_ready(self._config, env, backend.kind)
        runnable: list[str] = []
        pulled_without_env: list[str] = []
        for manifest in manifests.values():
            if not manifest.supports(backend.kind):
                continue
            spec = manifest.spec(backend.kind)
            if weights.installed(self._config, manifest, spec) is None:
                continue
            if envs[_for_engine(ENV_FOR_ENGINE, spec.engine, "env")][0]:
                runnable.append(manifest.id)
            else:
                pulled_without_env.append(manifest.id)
        detail = "; ".join(f"{env}: {text}" for env, (_, text) in envs.items())
        if not runnable:
            missing = (
                f"; installed but their env is not: {pulled_without_env}"
                if pulled_without_env
                else "; no ASR model is installed — `crucible models pull <id>`"
            )
            return JobTypeStatus(ready=False, detail=detail + missing)
        return JobTypeStatus(ready=True, detail=f"{detail}; runnable: {runnable}")


    def _spec_for(self, model_id: str) -> tuple[AsrManifest, Any]:
        manifest = MANIFESTS.known(model_id)
        return manifest, worker_type.require_block(
            manifest, model_id, self._backend.kind, "ASR model"
        )

    def _require_runnable(
        self, model_id: str, params: AsrParams
    ) -> tuple[AsrManifest, Any, Path, Path, "qwen.AlignerPlan | None"]:
        backend_kind = self._backend.kind
        manifest, spec = self._spec_for(model_id)
        worker_type.refuse_if_larger_than_host(
            self._backend,
            model_id,
            (
                qwen.floor_bytes(
                    manifest, spec, backend_kind, with_aligner=params.word_timestamps
                )
                if spec.engine in QWEN_ASR_ENGINES
                else spec.memory_bytes_estimate
            ),
        )
        accelerator.refuse_if_card_lacks(
            model_id=model_id,
            spec=spec,
            card=card_for(self._config.home, self._backend.gpu),
        )
        python = _python_for(self._config, spec.engine, backend_kind, model_id)
        weights_dir = worker_type.require_weights(self._config, manifest, spec, model_id)
        aligner = None
        if spec.engine in QWEN_ASR_ENGINES and params.word_timestamps:
            aligner = qwen.plan_aligner(self._config, spec, backend_kind)
        return manifest, spec, python, weights_dir, aligner

    def _need_bytes(self, manifest: Any, spec: Any, params: AsrParams) -> int:
        if spec.engine in QWEN_ASR_ENGINES:
            return qwen.need_bytes(
                spec,
                self._backend.kind,
                with_aligner=params.word_timestamps,
                width=self._width(manifest, spec, params),
            )
        return spec.memory_bytes_estimate

    def _width(self, manifest: Any, spec: Any, params: AsrParams) -> int | None:
        if spec.engine not in QWEN_ASR_ENGINES:
            return None
        return qwen.serving_width(
            manifest,
            spec,
            self._backend.kind,
            total_bytes=self._backend.gpu.vram_bytes,
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            with_aligner=params.word_timestamps,
        )

    def _refuse_what_this_engine_has_not_got(
        self, model_id: str, params: AsrParams
    ) -> None:
        _, spec = self._spec_for(model_id)
        engine = spec.engine
        if params.resume is not None and engine not in QWEN_ASR_ENGINES:
            raise ApiError(
                400,
                "resume_unsupported",
                f"{model_id!r} is whisper ({engine!r}), which keeps no resume "
                "journal yet, so there is nothing to resume; send the job without "
                "resume to transcribe from the start",
                {"type": JOB_TYPE, "engine": engine, "resume": params.resume},
            )
        if engine in ENGINES_WITHOUT_VAD and params.vad_filter:
            raise ApiError(
                400,
                "vad_unsupported_by_engine",
                f"{model_id!r} transcribes with {engine!r}, and that engine has no "
                "voice-activity detector at all — faster-whisper's is Silero, "
                "and mlx-whisper and the Qwen3-ASR engines ship nothing of the "
                "kind. Send vad_filter: false and get a transcript this server "
                "can describe, rather than one produced under rules nothing in "
                "the file records",
                {"backend": self._backend.kind, "engine": engine, "vad_filter": True},
            )
        qwen_engine = engine in QWEN_ASR_ENGINES
        if qwen_engine and params.initial_prompt is not None:
            raise ApiError(
                400,
                "initial_prompt_unsupported_by_engine",
                f"{model_id!r} is Qwen3-ASR, which has no initial_prompt: that is "
                "whisper's primed transcript, 223 tokens that scroll out. Send "
                "`context`, the instruction and vocabulary Qwen reads in its "
                "system turn before every piece",
                {"engine": engine, "field": "initial_prompt"},
            )
        if not qwen_engine and (params.piece_s is not None or params.overlap_s is not None):
            raise ApiError(
                400,
                "pieces_unsupported_by_engine",
                f"{model_id!r} is whisper ({engine!r}), which decodes in its own "
                "30-second windows; piece_s and overlap_s are how a Qwen3-ASR job "
                "cuts its input. Send neither",
                {"engine": engine, "piece_s": params.piece_s, "overlap_s": params.overlap_s},
            )
        if qwen_engine and not params.word_timestamps and (params.overlap_s or 0.0) > 0:
            raise ApiError(
                400,
                "overlap_needs_word_timestamps",
                f"overlap_s {params.overlap_s:g} makes neighbouring pieces hear the "
                "same words, and only word times can say which piece keeps each "
                "one. Send word_timestamps: true, or overlap_s: 0",
                {"engine": engine, "overlap_s": params.overlap_s},
            )
        if not qwen_engine and params.context is not None:
            raise ApiError(
                400,
                "context_unsupported_by_engine",
                f"{model_id!r} is whisper ({engine!r}), which has no context: "
                "that is Qwen3-ASR's system-turn instruction. Send "
                "`initial_prompt`, the text whisper is primed with as if it "
                "were the transcript so far",
                {"engine": engine, "field": "context"},
            )
        if qwen_engine and params.language not in QWEN3_LANGUAGES:
            auto = (
                "; `auto` is refused because detection costs the 1.7B time and "
                "the aligner has none"
                if params.language == AUTO_LANGUAGE
                else ""
            )
            raise ApiError(
                400,
                "language_unsupported_by_engine",
                f"{model_id!r} is always told its language, and it must be one "
                f"the aligner places words in: {sorted(QWEN3_LANGUAGES)}. "
                f"{params.language!r} is not{auto}",
                {"engine": engine, "language": params.language},
            )

    def requirements(
        self, model_id: str, params: AsrParams
    ) -> tuple[str, AsrManifest, Any, Path, Path, "qwen.AlignerPlan | None", Any]:
        self._refuse_what_this_engine_has_not_got(model_id, params)
        ffmpeg = hosttools.require_ffmpeg(JOB_TYPE, FFMPEG_WHY)
        manifest, spec, python, weights_dir, aligner = self._require_runnable(
            model_id, params
        )
        state = card_guard(
            self._config,
            model=model_id,
            need_bytes=self._need_bytes(manifest, spec, params),
            owned_pids=self._owned_pids(),
        )
        return ffmpeg, manifest, spec, python, weights_dir, aligner, state

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        model = require_model(model, self.name)
        self.requirements(model, parse_params(AsrParams, params, self.name))

    def journal_identity(self, model: str | None, params: dict[str, Any]) -> Identity | None:
        if model is None:
            return None
        checked = parse_params(AsrParams, params, self.name)
        _, spec = self._spec_for(model)
        if spec.engine not in QWEN_ASR_ENGINES:
            return None
        backend_kind = self._backend.kind
        aligner: dict[str, Any] | None = None
        if checked.word_timestamps:
            aligner_manifest, aligner_spec = qwen.aligner_spec(spec, backend_kind)
            aligner = {"id": aligner_manifest.id, "revision": aligner_spec.revision}
        speech = checked.speech_settings(Path("silero_vad"))
        if speech is not None:
            speech = {key: value for key, value in speech.items() if key != "weights"}
        return Identity(
            job_type=JOB_TYPE,
            model=model,
            revision=spec.revision,
            format_version=qwen.JOURNAL_FORMAT_VERSION,
            params={
                "engine": spec.engine,
                "dtype": qwen.run_dtype_on(self._config, self._backend, spec),
                "language": checked.language,
                "context": checked.context,
                "word_timestamps": checked.word_timestamps,
                "vad_filter": checked.vad_filter,
                "piece_s": checked.piece_seconds(),
                "overlap_s": checked.overlap_seconds(),
                "speech_only": checked.speech_only,
                "speech": speech,
                "aligner": aligner,
            },
        )


    def run(self, job: Job, ctx: JobContext) -> None:
        params = AsrParams.model_validate(job.params)
        model = run_model(job.model, self.name)
        audio = self._one_input(ctx)
        ffmpeg, manifest, spec, python, weights_dir, aligner, state = as_job_error(
            self.requirements, model, params
        )
        ctx.warming(state.detail)
        speech = params.speech_settings(self._speech_detector(ctx, params))

        progress_decoding(ctx, f"decoding {audio.name}")

        if spec.engine in QWEN_ASR_ENGINES:
            document = qwen.QwenAsrRun(
                config=self._config,
                backend=self._backend,
                ctx=ctx,
                job=job,
                model=model,
                spec=spec,
                python=python,
                weights_dir=weights_dir,
                aligner=aligner,
                ffmpeg=ffmpeg,
                audio=audio,
                language=params.language,
                context=params.context,
                word_timestamps=params.word_timestamps,
                piece_s=params.piece_seconds(),
                overlap_s=params.overlap_seconds(),
                width=self._width(manifest, spec, params),
                speech=speech,
                journal=ctx.journal,
                resumed=ctx.resumed,
            ).run()
        else:
            document = self._whisper(
                ctx, job, model, spec, python, weights_dir, ffmpeg, audio, params, speech
            )

        path = ctx.scratch / "transcript.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        ctx.artifact("transcript.json", path)
        ctx.progress(
            1.0,
            f"{len(document['segments'])} segments over "
            f"{document['duration_s']:.0f}s of audio",
            stage="transcribing",
            processed_s=document["duration_s"],
            total_s=document["duration_s"],
            cues=len(document["segments"]),
        )

    def _whisper(
        self,
        ctx: JobContext,
        job: Job,
        model: str,
        spec: Any,
        python: Path,
        weights_dir: Path,
        ffmpeg: str,
        audio: Path,
        params: AsrParams,
        speech: dict[str, Any] | None,
    ) -> dict[str, Any]:
        outcome = self._transcribe(
            ctx, job, spec.engine, python, weights_dir, ffmpeg, audio, params, speech
        )

        windows = outcome.ready["windows"]
        with worker_failed():
            results = workers.require_positional_results(outcome, windows, "window")

        failures = [
            f"window {index} ({index * WINDOW_SECONDS}s): {result['error']}"
            for index, result in enumerate(results)
            if "error" in result
        ]
        if failures:
            raise JobError(
                "asr_window_failed",
                f"{len(failures)} of {windows} window(s) failed, so the transcript "
                "would have holes in it and nothing in the file would say where: "
                + "; ".join(failures),
            )

        return self._transcript(model, spec, params, outcome, results, speech)

    def _speech_detector(self, ctx: JobContext, params: AsrParams) -> Path:
        path = hosttools.silero_vad_path(self._config.home)
        if not params.speech_only:
            return path
        if not hosttools.silero_vad_placed(self._config.home):
            ctx.warming("fetching the speech detector (silero-vad, 2.8 MB)")
            try:
                hosttools.ensure_silero_vad(self._config.home)
            except hosttools.HostToolError as exc:
                raise JobError(
                    "speech_detector_unavailable",
                    f"speech_only needs the speech detector, and it could not be "
                    f"fetched: {exc.message}. Send speech_only: false to transcribe "
                    "everything, or run `crucible install asr` on this server",
                ) from None
        return path

    @staticmethod
    def _one_input(ctx: JobContext) -> Path:
        inputs = ctx.inputs()
        if len(inputs) != 1:
            raise JobError(
                "invalid_inputs",
                f"an asr job takes exactly one audio file; this one has "
                f"{len(inputs)}: {sorted(inputs)}",
            )
        return next(iter(inputs.values()))

    def _transcribe(
        self,
        ctx: JobContext,
        job: Job,
        engine: str,
        python: Path,
        weights_dir: Path,
        ffmpeg: str,
        audio: Path,
        params: AsrParams,
        speech: dict[str, Any] | None,
    ) -> workers.WorkerOutcome:
        request = {
            "model_dir": str(weights_dir),
            "ffmpeg": ffmpeg,
            "audio": str(audio),
            "language": params.whisper_language(),
            "vad_filter": params.vad_filter,
            "word_timestamps": params.word_timestamps,
            "initial_prompt": params.initial_prompt,
            "device": _for_engine(DEVICE_FOR_ENGINE, engine, "device"),
            "compute_type": _for_engine(
                COMPUTE_TYPE_FOR_ENGINE, engine, "compute_type"
            ),
            "window_s": WINDOW_SECONDS,
            "overlap_s": OVERLAP_SECONDS,
            "speech": speech,
        }

        def on_ready(message: dict[str, Any]) -> None:
            kept = (
                ""
                if message.get("speech_s") is None
                else f", {message['speech_s']:.0f}s of it kept as speech"
            )
            ctx.warming(
                f"{message['duration_s']:.0f}s of audio decoded{kept}, "
                f"{message['windows']} window(s) of {WINDOW_SECONDS}s to transcribe "
                f"on {message['device']} at {message['compute_type']}"
            )

        def on_progress(message: dict[str, Any]) -> None:
            processed = float(message["processed_s"])
            total = float(message["total_s"])
            stage = message["stage"]
            fraction = 0.0 if stage == "decoding" else (
                min(1.0, processed / total) if total > 0 else 0.0
            )
            ctx.progress(
                fraction,
                f"{stage} {processed:.0f}s of {total:.0f}s, "
                f"{message['cues']} segments",
                stage=stage,
                processed_s=processed,
                total_s=total,
                cues=message["cues"],
            )

        with worker_failed():
            return workers.run_worker(
                python=python,
                script=_for_engine(
                    WORKER_SCRIPT_FOR_ENGINE, engine, "worker script"
                ),
                request=request,
                environment=workers.worker_environment(python.parent.parent),
                log_path=self._config.logs_dir / f"asr-{job.id}.log",
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
                on_ready=on_ready,
                on_progress=on_progress,
                cancelled=lambda: ctx.cancelled,
            )


    @staticmethod
    def _transcript(
        model: str,
        spec: Any,
        params: AsrParams,
        outcome: workers.WorkerOutcome,
        results: tuple[dict[str, Any], ...],
        speech: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with worker_failed(ValueError):
            timeline = speechonly.timeline_for(outcome.ready, speech)
        segments: list[dict[str, Any]] = []
        for index, result in enumerate(results):
            offset = index * float(WINDOW_SECONDS)
            for segment in result["segments"]:
                shifted = dict(segment)
                shifted["start"] = segment["start"] + offset
                shifted["end"] = segment["end"] + offset
                if "words" in segment:
                    shifted["words"] = [
                        {
                            **word,
                            "start": word["start"] + offset,
                            "end": word["end"] + offset,
                        }
                        for word in segment["words"]
                    ]
                segments.append(shifted)

        segments.sort(key=lambda row: row["start"])
        deduplicated: list[dict[str, Any]] = []
        for segment in segments:
            if (
                deduplicated
                and segment["start"]
                < deduplicated[-1]["end"] - OVERLAP_TOLERANCE_SECONDS
            ):
                continue
            deduplicated.append(segment)
        deduplicated = [speechonly.to_source(segment, timeline) for segment in deduplicated]

        first = results[0]
        return transcript_document(
            model=model,
            spec=spec,
            engine={},
            language=first["language"],
            language_probability=first["language_probability"],
            language_requested=params.language,
            vad_filter=params.vad_filter,
            word_timestamps=params.word_timestamps,
            initial_prompt=params.initial_prompt,
            prompt={},
            duration_s=outcome.ready["duration_s"],
            layout={
                "window_s": WINDOW_SECONDS,
                "overlap_s": OVERLAP_SECONDS,
                "windows": outcome.ready["windows"],
            },
            speech=speech,
            timeline=timeline,
            segments=deduplicated,
        )


JOB_TYPES: tuple[JobTypeBinding, ...] = (
    JobTypeBinding(
        ASR_JOB,
        lambda wiring: AsrJobType(
            wiring.config, wiring.backend, wiring.residency.owned_pids
        ),
    ),
)
