"""The `align-longform` job type — four stages, two existing workers, one lane.

The contract and its refusals are in `__init__.py`; the stage drivers are in
`stages.py`; the two ported algorithms are `coarse.py` and `cues.py`. This file
is the thing that runs them in order and reports.

WHAT IT CHARGES
---------------
One slot, its own, for the whole duration — Owen's TTS ruling applied here by the
same reasoning: *"the entire tts step goes to the other system. That includes
anything the step needs to do even if it's cpu."* `transcribe` and `coarse-align`
are the CPU stages and are most of the wall clock, and they run here rather than
being sliced back to the client.

TWO MODELS, ONE `model` FIELD
-----------------------------
A job carries one model id and this act needs two. `model` is the ALIGNER —
`qwen3-aligner`, the thing that holds the card and the thing a failure is usually
about — and the rough pass's model is a PARAM (`rough_model`). That asymmetry is
deliberate rather than a wire limitation: the aligner decides what this job IS,
while the whisper size is a speed/quality dial the caller turns per book.

PROGRESS IS PER STAGE AND THE NAMES ARE THE CONTRACT
-----------------------------------------------------
`STAGES` in `__init__.py` are matched verbatim by BookForge's generate-sentences
row to fill its stacked bars, so they are part of the wire rather than log prose.
The fractions below are the local aligner's own measured shape: the rough pass
dominates, the card stage is fast, and coarse-align and write are near-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ... import accelerator, weights, workerenv
from ...alignmodels import load_all_align_manifests
from ...asrmodels import load_all_asr_manifests
from ...config import Config
from ...errors import ApiError, JobError
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor
from . import ALIGNER_MODEL, JOB_TYPE_NAME, STAGES, validate
from . import coarse, cues, plan, stages

#: Where each stage ends, as a fraction of the whole. The rough pass is ~40
#: minutes on a 16 h book and the card stage is minutes, so the bar spends most
#: of its life in `transcribe` — which is the truth, and a bar that raced to 90%
#: and sat there would be a worse lie than a slow one.
STAGE_END = {"transcribe": 0.70, "coarse-align": 0.75, "align": 0.95, "write": 1.0}


class AlignLongformJobType:
    """`align-longform`: a whole audiobook in, a VTT and a report out."""

    name = JOB_TYPE_NAME

    def __init__(self, config: Config, backend: Any) -> None:
        self._config = config
        self._backend = backend

    # ------------------------------------------------------------- models

    def describe_models(self) -> list[ModelDescriptor]:
        """The ALIGNER only. The rough model is a param, not this job's subject."""
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
        """The card stage's need.

        The ROUGH pass also uses the card and is not added: the two are never
        resident together (`stages.py`), so the peak is whichever is larger, and
        on every shipped pair that is the aligner. Stated rather than summed,
        because summing would refuse a job that fits.
        """
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

    # ------------------------------------------------------------- health

    def check(self, backend: Any) -> JobTypeStatus:
        """BOTH envs, because this job is nothing without either.

        Reported as one status naming which half is missing, rather than as
        "ready" on the strength of the aligner alone — a job type that says ready
        and then fails in its first stage is the shape this server spends its
        refusals avoiding.
        """
        missing: list[str] = []
        for job_type in ("asr", "align"):
            status = workerenv.env_status(
                self._config.home, job_type, self._config.backend_kind
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

    # ---------------------------------------------------------- preflight

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        try:
            parsed = validate(params)
        except ValueError as exc:
            raise ApiError(400, "invalid_params", f"{self.name} params are not valid: {exc}")
        if model is None:
            raise ApiError(400, "model_required", f"{self.name} needs an aligner")

        # The ROUGH model must be installed too, and refusing here is the point:
        # discovering it after a 40-minute transcribe would be the same failure
        # an hour later.
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

        accelerator.guard(
            self._config.backend_kind,
            model_id=model,
            need_bytes=self.vram_estimate(model),
            owned_pids=frozenset(),
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
        )

    # --------------------------------------------------------------- run

    def run(self, job: Job, ctx: JobContext) -> None:
        params = validate(job.params)
        inputs = ctx.inputs()
        if len(inputs) != 1:
            raise JobError(
                "one_audio_input",
                f"{self.name} takes exactly one input — the audiobook — and got "
                f"{len(inputs)} ({sorted(inputs)}). The EPUB never crosses: the sentences "
                "are in params, as text.",
            )
        audio = next(iter(inputs.values()))
        ffmpeg = _require_ffmpeg()

        aligner = load_all_align_manifests()[job.model or ALIGNER_MODEL]
        aligner_spec = aligner.backends[self._config.backend_kind]
        aligner_weights = weights.require_installed(self._config, aligner, aligner_spec).path
        rough = load_all_asr_manifests()[params.rough_model]
        rough_spec = rough.backends[self._config.backend_kind]
        rough_weights = weights.require_installed(self._config, rough, rough_spec).path

        try:
            duration = stages.probe_duration(ffmpeg, audio)

            # ── 1. transcribe ────────────────────────────────────────────
            ctx.progress(0.0, "transcribing the audiobook", stage=STAGES[0])
            words = stages.transcribe(
                home=self._config.home,
                python=workerenv.worker_env_python(self._config.home, "asr"),
                weights_dir=rough_weights,
                ffmpeg=ffmpeg,
                audio=audio,
                language=params.language,
                log_path=self._config.logs_dir / f"alf-asr-{job.id}.log",
                on_progress=lambda m: ctx.progress(
                    STAGE_END["transcribe"] * min(1.0, float(m.get("processed_s", 0))
                                                  / max(1.0, duration)),
                    "transcribing the audiobook", stage=STAGES[0],
                ),
                cancelled=lambda: ctx.cancelled,
            )
            ctx.raise_if_cancelled()

            # ── 2. coarse-align ──────────────────────────────────────────
            ctx.progress(STAGE_END["transcribe"], "placing the book against the transcript",
                         stage=STAGES[1])
            rough_times = coarse.coarse_align(
                [s.text for s in params.sentences],
                [(coarse._norm(w), t) for w, t in words],
            )
            ctx.raise_if_cancelled()

            # ── 3. align ─────────────────────────────────────────────────
            chunk_plan = plan.plan_chunks(
                rough_times.rough, rough_times.first_index, rough_times.last_index,
                duration, params.chunk_s,
            )
            warning = plan.capped_warning(chunk_plan, params.chunk_s)
            if warning:
                ctx.progress(STAGE_END["coarse-align"], warning, stage=STAGES[1])
            if not chunk_plan.chunks:
                raise JobError(
                    "nothing_narrated",
                    "no sentence could be placed in the audio, so there is nothing to align. "
                    "Either the book and the audiobook are different works, or the language is "
                    "wrong for this narration.",
                )

            ctx.progress(STAGE_END["coarse-align"], f"aligning {len(chunk_plan.chunks)} window(s)",
                         stage=STAGES[2])
            chunk_dir = ctx.scratch / "chunks"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            files: list[Path] = []
            texts: list[str] = []
            for chunk in chunk_plan.chunks:
                out = chunk_dir / f"{chunk.index}.wav"
                stages.slice_chunk(ffmpeg, audio, chunk.start, chunk.end, out)
                files.append(out)
                texts.append(" ".join(params.sentences[i].text for i in chunk.sentences))
            ctx.raise_if_cancelled()

            aligned = stages.align_chunks(
                home=self._config.home,
                python=workerenv.worker_env_python(self._config.home, "align"),
                weights_dir=aligner_weights,
                ffmpeg=ffmpeg,
                language_name=params.language_name,
                chunk_files=files,
                chunk_texts=texts,
                max_audio_s=params.chunk_s * 2,
                log_path=self._config.logs_dir / f"alf-align-{job.id}.log",
                cancelled=lambda: ctx.cancelled,
            )

            # ── 4. write ─────────────────────────────────────────────────
            ctx.progress(STAGE_END["align"], "writing the transcript", stage=STAGES[3])
            written = _build_cues(params, chunk_plan, aligned)
            if not written:
                raise JobError(
                    "no_cues",
                    "the aligner placed no item, so there is nothing to write. A bare WEBVTT "
                    "is not a transcript.",
                )
            vtt = cues.write_vtt(written)
            # WRITTEN AND THEN REGISTERED. `ctx.scratch` is a working directory,
            # not the artifact store — a file left there is a job that reported
            # `done` with `artifacts: []`, which is success delivering nothing.
            # Measured on the first green run, 2026-09-15.
            vtt_path = ctx.scratch / "alignment.vtt"
            vtt_path.write_text(vtt, encoding="utf-8")
            ctx.artifact("alignment.vtt", vtt_path)
            report_path = ctx.scratch / "align-report.json"
            stages.write_report(report_path, {
                "sentences": len(params.sentences),
                "placed": len(written),
                "dropped": rough_times.dropped,
                "rate_tokens_per_second": round(rough_times.rate, 3),
                "chunks": len(chunk_plan.chunks),
                "capped": chunk_plan.capped,
                "duration_s": round(duration, 3),
                "rough_model": params.rough_model,
                "aligner": f"{aligner.id}@{aligner_spec.revision}",
            })
            ctx.artifact("align-report.json", report_path)
            ctx.progress(1.0, f"placed {len(written)} of {len(params.sentences)} sentence(s)",
                         stage=STAGES[3])
        except stages.StageFailed as exc:
            raise JobError(exc.code, str(exc)) from None
        except cues.NoCues as exc:
            raise JobError("no_cues", str(exc)) from None


def _build_cues(params: Any, chunk_plan: Any, aligned: list[dict[str, Any]]) -> list[cues.Cue]:
    """Aligner items back onto sentences, BY TOKEN COUNT and by POSITION.

    Crucible's aligner asserts nothing about words — it returns one timestamped
    item per its OWN tokenisation — so the mapping is the caller's, exactly as it
    is for the `align` job type. Position is the chunk's whole identity: no index
    travels either way.
    """
    out: list[cues.Cue] = []
    for chunk, result in zip(chunk_plan.chunks, aligned):
        items = result.get("items") or []
        if result.get("error") or not items:
            # A failed window keeps its coarse timing rather than vanishing: the
            # sentences are still in the book and still in the audio.
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


def _require_ffmpeg() -> str:
    from ... import hosttools

    found = hosttools.which("ffmpeg")
    if found is None:
        raise JobError(
            "ffmpeg_missing",
            "align-longform decodes the audiobook and cuts its windows with ffmpeg, and there "
            "is none on this host's PATH.",
        )
    return found
