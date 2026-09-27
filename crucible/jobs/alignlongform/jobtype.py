from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ... import hosttools, jobenv, weights
from ...alignmodels import load_all_align_manifests
from ...asrmodels import load_all_asr_manifests
from ...config import Config
from ...errors import ApiError, JobError
from ...jobtypes import ALIGN_LONGFORM
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from ..binding import JobTypeBinding
from ..template import card_guard, require_model
from . import ALIGNER_MODEL, JOB_TYPE_NAME, STAGES, coarse, cues, plan, stages, validate

STAGE_END = {"transcribe": 0.70, "coarse-align": 0.75, "align": 0.95, "write": 1.0}


class AlignLongformJobType:

    name = JOB_TYPE_NAME

    def __init__(self, config: Config, backend: Any) -> None:
        self._config = config
        self._backend = backend


    def describe_models(self) -> list[ModelDescriptor]:
        out: list[ModelDescriptor] = []
        for manifest in load_all_align_manifests().values():
            if manifest.id != ALIGNER_MODEL:
                continue
            spec = manifest.backends.get(self._config.backend_kind)
            if spec is None:
                continue
            out.append(
                ModelDescriptor(
                    id=manifest.id,
                    revision=spec.revision,
                    source=spec.hf_repo,
                    installed=weights.installed(self._config, manifest, spec) is not None,
                    resident=False,
                    vram_bytes=spec.memory_bytes_estimate,
                )
            )
        return out

    def vram_estimate(self, model: str | None) -> int:
        manifests = load_all_align_manifests()
        manifest = manifests.get(model or ALIGNER_MODEL)
        if manifest is None:
            return 0
        spec = manifest.backends.get(self._config.backend_kind)
        return spec.memory_bytes_estimate if spec else 0

    def model_provenance(self, model: str | None) -> dict[str, Any] | None:
        manifest = load_all_align_manifests().get(model or ALIGNER_MODEL)
        if manifest is None:
            return None
        spec = manifest.backends.get(self._config.backend_kind)
        if spec is None:
            return None
        return {
            "id": manifest.id,
            "revision": spec.revision,
            "fingerprint": f"{manifest.id}@{spec.revision}",
        }


    def check(self, backend: Any) -> JobTypeStatus:
        missing: list[str] = []
        for job_type in ("asr", "align"):
            status = jobenv.env_status(
                self._config.home,
                jobenv.worker_env(job_type, self._config.backend_kind),
                self._config.backend_kind,
            )
            if not status.installed:
                missing.append(f"{job_type} ({status.detail})")
        if missing:
            return JobTypeStatus(
                ready=False,
                detail=(
                    "align-longform needs the envs of both stages it drives and is missing "
                    + "; ".join(missing)
                ),
            )
        return JobTypeStatus(
            ready=True,
            detail="drives the asr and align workers; one slot for the whole book",
        )


    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        try:
            parsed = validate(params)
        except ValueError as exc:
            raise ApiError(400, "invalid_params", f"{self.name} params are not valid: {exc}")
        model = require_model(model, self.name, "an aligner")

        asr_manifests = load_all_asr_manifests()
        rough = asr_manifests.get(parsed.rough_model)
        if rough is None:
            raise ApiError(
                400, "unknown_rough_model",
                f"{parsed.rough_model!r} is not an ASR model this server has a manifest for; "
                f"it knows {sorted(asr_manifests)}.",
            )
        rough_spec = rough.backends.get(self._config.backend_kind)
        if rough_spec is None:
            raise ApiError(
                409, "rough_model_not_on_this_backend",
                f"{parsed.rough_model!r} has no {self._config.backend_kind} block, so the rough "
                "pass cannot run on this host.",
            )
        try:
            weights.require_installed(self._config, rough, rough_spec)
        except weights.WeightsError as exc:
            raise ApiError(409, "rough_model_not_installed", str(exc)) from None

        card_guard(
            self._config,
            model=model,
            need_bytes=self.vram_estimate(model),
            owned_pids=frozenset(),
        )


    def run(self, job: Job, ctx: JobContext) -> None:
        params = validate(job.params)
        audio = _the_one_audio(self.name, ctx)
        ffmpeg = hosttools.ffmpeg_path()
        if ffmpeg is None:
            raise JobError(
                "ffmpeg_missing",
                "align-longform decodes the audiobook and cuts its windows with ffmpeg, "
                "and there is none on this host's PATH.",
            )
        aligner = load_all_align_manifests()[job.model or ALIGNER_MODEL]
        aligner_spec = aligner.backends[self._config.backend_kind]
        aligner_weights = weights.require_installed(self._config, aligner, aligner_spec).path
        rough = load_all_asr_manifests()[params.rough_model]
        rough_spec = rough.backends[self._config.backend_kind]
        rough_weights = weights.require_installed(self._config, rough, rough_spec).path
        book = _Book(
            config=self._config, job=job, ctx=ctx, params=params, audio=audio,
            ffmpeg=ffmpeg, aligner=aligner, aligner_spec=aligner_spec,
            aligner_weights=aligner_weights, rough_weights=rough_weights,
        )
        try:
            _run_stages(book)
        except stages.StageFailed as exc:
            raise JobError(exc.code, str(exc)) from None
        except cues.NoCues as exc:
            raise JobError("no_cues", str(exc)) from None


@dataclass
class _Book:

    config: Config
    job: Job
    ctx: JobContext
    params: Any
    audio: Path
    ffmpeg: str
    aligner: Any
    aligner_spec: Any
    aligner_weights: Path
    rough_weights: Path
    duration: float = 0.0

    def worker_python(self, job_type: str) -> Path:
        return jobenv.env_python(
            self.config.home, jobenv.worker_env(job_type, self.config.backend_kind)
        )


def _the_one_audio(name: str, ctx: JobContext) -> Path:
    inputs = ctx.inputs()
    if len(inputs) != 1:
        raise JobError(
            "one_audio_input",
            f"{name} takes exactly one input — the audiobook — and got "
            f"{len(inputs)} ({sorted(inputs)}). The EPUB never crosses: the sentences "
            "are in params, as text.",
        )
    return next(iter(inputs.values()))


def _run_stages(book: _Book) -> None:
    ctx = book.ctx
    book.duration = stages.probe_duration(book.ffmpeg, book.audio)
    words = _transcribe(book)
    ctx.raise_if_cancelled()
    rough_times = _place_sentences(book, words)
    ctx.raise_if_cancelled()
    chunk_plan = _plan_windows(book, rough_times)
    aligned = _align_windows(book, chunk_plan)
    _write_transcript(book, rough_times, chunk_plan, aligned)


def _transcribe(book: _Book) -> list[tuple[str, float]]:
    ctx = book.ctx
    ctx.progress(0.0, "transcribing the audiobook", stage=STAGES[0])
    return stages.transcribe(
        python=book.worker_python("asr"),
        weights_dir=book.rough_weights,
        ffmpeg=book.ffmpeg,
        audio=book.audio,
        language=book.params.language,
        log_path=book.config.logs_dir / f"alf-asr-{book.job.id}.log",
        on_progress=lambda m: ctx.progress(
            STAGE_END["transcribe"] * min(1.0, float(m.get("processed_s", 0))
                                          / max(1.0, book.duration)),
            "transcribing the audiobook", stage=STAGES[0],
        ),
        cancelled=lambda: ctx.cancelled,
    )


def _place_sentences(book: _Book, words: list[tuple[str, float]]) -> coarse.CoarseResult:
    book.ctx.progress(STAGE_END["transcribe"], "placing the book against the transcript",
                      stage=STAGES[1])
    return coarse.coarse_align(
        [s.text for s in book.params.sentences],
        [(coarse._norm(w), t) for w, t in words],
    )


def _plan_windows(book: _Book, rough_times: coarse.CoarseResult) -> Any:
    chunk_s = book.params.chunk_s
    chunk_plan = plan.plan_chunks(
        rough_times.rough, rough_times.first_index, rough_times.last_index,
        book.duration, chunk_s,
    )
    warning = plan.capped_warning(chunk_plan, chunk_s)
    if warning:
        book.ctx.progress(STAGE_END["coarse-align"], warning, stage=STAGES[1])
    if not chunk_plan.chunks:
        raise JobError(
            "nothing_narrated",
            "no sentence could be placed in the audio, so there is nothing to align. "
            "Either the book and the audiobook are different works, or the language is "
            "wrong for this narration.",
        )
    return chunk_plan


def _align_windows(book: _Book, chunk_plan: Any) -> list[dict[str, Any]]:
    ctx = book.ctx
    ctx.progress(STAGE_END["coarse-align"], f"aligning {len(chunk_plan.chunks)} window(s)",
                 stage=STAGES[2])
    chunk_dir = ctx.scratch / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    texts: list[str] = []
    for chunk in chunk_plan.chunks:
        out = chunk_dir / f"{chunk.index}.wav"
        stages.slice_chunk(book.ffmpeg, book.audio, chunk.start, chunk.end, out)
        files.append(out)
        texts.append(" ".join(book.params.sentences[i].text for i in chunk.sentences))
    ctx.raise_if_cancelled()
    return stages.align_chunks(
        python=book.worker_python("align"),
        weights_dir=book.aligner_weights,
        ffmpeg=book.ffmpeg,
        language_name=book.params.language_name,
        chunk_files=files,
        chunk_texts=texts,
        max_audio_s=book.params.chunk_s * 2,
        log_path=book.config.logs_dir / f"alf-align-{book.job.id}.log",
        spec=book.aligner_spec,
        cancelled=lambda: ctx.cancelled,
    )


def _write_transcript(
    book: _Book, rough_times: coarse.CoarseResult, chunk_plan: Any,
    aligned: list[dict[str, Any]],
) -> None:
    ctx = book.ctx
    params = book.params
    ctx.progress(STAGE_END["align"], "writing the transcript", stage=STAGES[3])
    written = _build_cues(params, chunk_plan, aligned)
    if not written:
        raise JobError(
            "no_cues",
            "the aligner placed no item, so there is nothing to write. A bare WEBVTT "
            "is not a transcript.",
        )
    vtt_path = ctx.scratch / "alignment.vtt"
    vtt_path.write_text(cues.write_vtt(written), encoding="utf-8")
    ctx.artifact("alignment.vtt", vtt_path)
    report_path = ctx.scratch / "align-report.json"
    stages.write_report(report_path, {
        "sentences": len(params.sentences),
        "placed": len(written),
        "dropped": rough_times.dropped,
        "rate_tokens_per_second": round(rough_times.rate, 3),
        "chunks": len(chunk_plan.chunks),
        "capped": chunk_plan.capped,
        "duration_s": round(book.duration, 3),
        "rough_model": params.rough_model,
        "aligner": f"{book.aligner.id}@{book.aligner_spec.revision}",
    })
    ctx.artifact("align-report.json", report_path)
    ctx.progress(1.0, f"placed {len(written)} of {len(params.sentences)} sentence(s)",
                 stage=STAGES[3])


def _build_cues(params: Any, chunk_plan: Any, aligned: list[dict[str, Any]]) -> list[cues.Cue]:
    out: list[cues.Cue] = []
    for chunk, result in zip(chunk_plan.chunks, aligned):
        items = result.get("items") or []
        if result.get("error") or not items:
            continue
        cursor = 0
        for sentence_index in chunk.sentences:
            sentence = params.sentences[sentence_index]
            width = len(coarse.toks(sentence.text))
            span = items[cursor:cursor + width]
            cursor += width
            if not span:
                break
            out.append(cues.Cue(
                start=chunk.start + float(span[0]["start"]),
                end=chunk.start + float(span[-1]["end"]),
                text=sentence.text,
                kind=sentence.kind,
            ))
    return out


JOB_TYPES: tuple[JobTypeBinding, ...] = (
    JobTypeBinding(
        ALIGN_LONGFORM,
        lambda wiring: AlignLongformJobType(wiring.config, wiring.backend),
    ),
)
