"""The `asr` job type: one audio file in, one transcript out.

PHASE4-AUDIO.md section 3. This is a real gap rather than an oversight in the
app: `generate-sentences` is a GPU queue step, it is the only ASR site in
BookForge that takes the arbiter lease, and the whole-m4b align door needs a
rough transcript before the aligner runs.

**There is no default model.** A job names one or it is refused, because an ASR
pass at the wrong size is a transcript that looks fine, is worse, and has nothing
in it to say so. `jobs.resolve_model` does that refusal for free, since this type
advertises six models and none of them is preferred.

What is the client's and what is the server's
---------------------------------------------
The client says *what to transcribe* and *how to read it*: the model, the
language, and the two switches that change what whisper is asked for
(`vad_filter`, `word_timestamps`). Both switches are required — they default to
true in BookForge, and a default here would mean a transcript silently produced
under different rules than the caller assumed.

Everything about *how it is run* is the server's and appears nowhere on the wire:

- **`compute_type`.** `float16` on an accelerator, `int8` on CPU. Crucible has no
  CPU backend, so it is `float16`, always. BookForge also has a **one-shot CPU
  fallback** here (`transcribe-bridge.ts`: a CUDA load that fails is retried once
  on CPU at int8). That does not come across, and the reason is that it is not a
  fallback at all, it is a silent substitution: the run still produces a
  transcript, the transcript is a different transcript, and nothing in the output
  says which one you got. Crucible refuses instead.
- **Windowing.** 900-second windows, each extended 15 seconds past its own
  boundary so a sentence straddling the cut is spoken in full inside it. Those
  are BookForge's measured numbers: an 18-hour file handed to
  `model.transcribe()` in one piece frames the whole signal into about 19 GiB of
  float64 and OOMs, and a 900-second window keeps the peak independent of book
  length. A client sends one file and never learns any of this.
- **Decoding.** 16 kHz mono float32 through ffmpeg, once, in the worker.

What comes back
---------------
`transcript.json`, holding whisper's own segments in absolute book time with the
window overlaps removed, plus what produced them. Sentence-cue grouping and the
WebVTT stay in BookForge, for the reason section 2 gives about `align`: Crucible
returns what the model said and asserts nothing about the client's units.

The one deviation from the app worth knowing about is where the overlap
duplicates are dropped. BookForge groups words into sentence cues first and
dedupes the *cues*; here the grouping does not exist yet, so the same rule — sort
by start, drop anything that begins inside a kept span, with the same 0.1 s
tolerance — is applied to the *segments*. The boundary behaviour is therefore
close but not identical, and it is written down here rather than discovered later.

A failed window fails the job
-----------------------------
The worker keeps going after a window fails, so one run finds every bad stretch
instead of one per re-run. But the job then fails, naming the windows, and
publishes no artifact. A hole in the middle of a transcript is invisible in the
output — which is the same argument as the one against a default model, and it
gets the same answer.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from ... import accelerator, weights, workerenv, workers
from ...asrmodels import AsrManifest, AsrManifestError, load_all_asr_manifests
from ...config import Config
from ...errors import ApiError, JobError
from ..base import Job, JobContext, JobTypeStatus, ModelDescriptor

__all__ = ["AsrJobType", "AsrParams"]

JOB_TYPE = "asr"

#: The audio window, and how far each window reaches past its own end. Engine
#: knowledge, measured by BookForge against real books
#: (`electron/scripts/transcribe_audiobook.py`), not a wire parameter.
WINDOW_SECONDS = 900
OVERLAP_SECONDS = 15

#: How long the server waits on a worker that has said *nothing at all* before it
#: gives up on it. Not a run deadline: every message resets it, and the worker
#: reports decode progress from its first seconds, so this only fires on a worker
#: that is genuinely wedged. 900 s covers loading a 3 GB model from a cold disk.
READY_SILENCE_TIMEOUT_SECONDS = 900.0

#: Two segments whose spans overlap by more than this are the same speech heard
#: twice, once in each of two consecutive windows. BookForge's number.
OVERLAP_TOLERANCE_SECONDS = 0.1

#: `compute_type` on an accelerator. There is no CPU entry because there is no
#: CPU backend; see the module docstring on why the app's CPU fallback is not
#: carried across.
COMPUTE_TYPE = "float16"
DEVICE = "cuda"

#: The language codes faster-whisper accepts, read from
#: `faster_whisper/tokenizer.py`'s `_LANGUAGE_CODES` at master on 2026-09-13.
#: Checked here so a typo is a 400 naming the code rather than a job that dies
#: mid-stream inside the worker — whisper raises on an unknown code only when the
#: tokenizer is built, which is after the model is on the card.
WHISPER_LANGUAGES = frozenset(
    """af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo
    fr gl gu ha haw he hi hr ht hu hy id is it ja jw ka kk km kn ko la lb ln lo
    lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si
    sk sl sn so sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi yi yo zh
    yue""".split()
)

#: What a client sends instead of a code to ask whisper to detect the language.
#: It is a value, not an absence: "detect it" is a decision, and a job that did
#: not make it is a job that did not say what it wanted.
AUTO_LANGUAGE = "auto"

WORKER_SCRIPT = Path(__file__).resolve().parent / "worker.py"


class AsrParams(BaseModel):
    """`params` for an asr job. Unknown keys are refused, not ignored."""

    model_config = ConfigDict(extra="forbid")

    language: str
    vad_filter: bool
    word_timestamps: bool

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
        """What the worker passes to `model.transcribe`. None means detect."""
        return None if self.language == AUTO_LANGUAGE else self.language


# ------------------------------------------------------------------ helpers


def _manifests() -> dict[str, AsrManifest]:
    try:
        return load_all_asr_manifests()
    except AsrManifestError as exc:
        raise ApiError(
            500,
            "asr_manifests_unreadable",
            f"this server cannot read its ASR model manifests: {exc}",
        ) from None


def _known(model_id: str) -> AsrManifest:
    manifests = _manifests()
    manifest = manifests.get(model_id)
    if manifest is None:
        raise ApiError(
            400,
            "unknown_model",
            f"no ASR manifest for model {model_id!r}; this build ships "
            f"{sorted(manifests)}",
        )
    return manifest


def _params(params: dict[str, Any]) -> AsrParams:
    try:
        return AsrParams.model_validate(params)
    except ValidationError as exc:
        raise ApiError(
            400,
            "invalid_params",
            "asr params are not valid: "
            + "; ".join(
                f"{'.'.join(str(p) for p in problem['loc']) or '<root>'}: "
                f"{problem['msg']}"
                for problem in exc.errors()
            ),
        ) from None


def ffmpeg_path() -> str | None:
    """Where ffmpeg is on this host, or None.

    A module-level probe, for the reason `crucible/accelerator.py` gives about
    its own: a test replaces it and asserts on the refusal, instead of asserting
    on whatever happens to be installed on the machine running the suite.
    """
    return shutil.which("ffmpeg")


def _require_ffmpeg() -> str:
    """ffmpeg's path, or a refusal by name before the job is queued.

    The worker decodes through ffmpeg rather than through faster-whisper's own
    PyAV decoder, which silently truncates some assembled m4b files (see
    `worker.py`). So ffmpeg is not optional, and its absence is a fact about the
    host that should be a 409 at submit time rather than a job that dies a minute
    in.
    """
    found = ffmpeg_path()
    if found is None:
        raise ApiError(
            409,
            "ffmpeg_missing",
            "there is no ffmpeg on this server's PATH, and asr decodes every input "
            "through it — faster-whisper's own PyAV decoder silently truncates some "
            "m4b files, which ends a transcript hours early with no error",
        )
    return found


# ------------------------------------------------------------------ job type


class AsrJobType:
    """`POST /v1/jobs {"type": "asr", "model": "<id>", "inputs": {...}}`."""

    name = JOB_TYPE

    def __init__(
        self,
        config: Config,
        backend: Any,
        owned_pids: Callable[[], frozenset[int]],
    ) -> None:
        self._config = config
        self._backend = backend
        # The accelerator guard must not report Crucible's own resident engine as
        # somebody else's process holding the card, so this type is handed the
        # same owned-pid set the `llm` types use. It is a callable and not a set
        # because the answer changes every time a model is loaded or unloaded.
        self._owned_pids = owned_pids

    # ----------------------------------------------------------- describing

    def describe_models(self) -> list[ModelDescriptor]:
        backend_kind = self._config.backend_kind
        rows: list[ModelDescriptor] = []
        for manifest in _manifests().values():
            if manifest.supports(backend_kind):
                spec = manifest.spec(backend_kind)
                revision, source, estimate = (
                    spec.revision,
                    spec.hf_repo,
                    spec.memory_bytes_estimate,
                )
            else:
                revision, source, estimate = "", "", 0
            rows.append(
                ModelDescriptor(
                    id=manifest.id,
                    revision=revision,
                    source=source,
                    # Nothing is ever resident for `asr`: the worker loads the
                    # model, transcribes one file and exits. The aligner is the
                    # job type that stays resident across a book, and it is
                    # section 2's, not this one's.
                    resident=False,
                    vram_bytes=estimate,
                )
            )
        return rows

    def vram_estimate(self, model: str | None) -> int:
        if model is None:
            raise JobError("model_required", f"{self.name} needs a model")
        manifest = _known(model)
        if not manifest.supports(self._config.backend_kind):
            return 0
        return manifest.spec(self._config.backend_kind).memory_bytes_estimate

    def check(self, backend: Any) -> JobTypeStatus:
        try:
            env = workerenv.env_status(self._config.home, JOB_TYPE, backend.kind)
        except workerenv.WorkerEnvError as exc:
            return JobTypeStatus(ready=False, detail=str(exc))
        if not env.installed:
            return JobTypeStatus(ready=False, detail=env.detail)
        if ffmpeg_path() is None:
            return JobTypeStatus(
                ready=False,
                detail=f"{env.detail}; but there is no ffmpeg on PATH, and asr "
                "decodes every input through it",
            )
        try:
            manifests = _manifests()
        except ApiError as exc:
            return JobTypeStatus(ready=False, detail=exc.message)
        installed = [
            manifest.id
            for manifest in manifests.values()
            if manifest.supports(backend.kind)
            and weights.installed(
                self._config, manifest, manifest.spec(backend.kind)
            )
            is not None
        ]
        if not installed:
            return JobTypeStatus(
                ready=False,
                detail=(
                    f"{env.detail}; no ASR model is installed — "
                    "`crucible models pull <id>`"
                ),
            )
        return JobTypeStatus(ready=True, detail=f"{env.detail}; installed: {installed}")

    # ------------------------------------------------------------ preflight

    def _require_runnable(self, model_id: str) -> tuple[AsrManifest, Any, Path, Path]:
        """Manifest, spec, env python and weights dir, or the named refusal.

        The order is `llm`'s and for `llm`'s reason: what no amount of installing
        can fix first, then what an install or a pull would fix, then the live
        accelerator. Nobody is told to download 3 GB of weights for a model that
        will never fit.
        """
        backend_kind = self._backend.kind
        manifest = _known(model_id)
        if not manifest.supports(backend_kind):
            raise ApiError(
                400,
                "backend_unsupported",
                f"ASR model {model_id!r} has no {backend_kind} block; "
                f"{manifest.path.name} declares {sorted(manifest.backends)}. "
                "faster-whisper is CTranslate2, which has no Metal backend, so "
                "there is no mlx-darwin block for any of them",
                {
                    "model": model_id,
                    "backend": backend_kind,
                    "declared": sorted(manifest.backends),
                },
            )
        spec = manifest.spec(backend_kind)
        accelerator.refuse_if_larger_than_host(
            model_id=model_id,
            need_bytes=spec.memory_bytes_estimate,
            host_total_bytes=self._backend.gpu.vram_bytes,
            host_name=self._backend.gpu.name,
        )
        try:
            python = workerenv.require_env(self._config.home, JOB_TYPE, backend_kind)
        except workerenv.WorkerEnvError as exc:
            raise ApiError(
                409,
                "env_missing",
                f"cannot run {model_id!r}: {exc}",
                {
                    "model": model_id,
                    "env": str(workerenv.worker_env_dir(self._config.home, JOB_TYPE)),
                },
            ) from None
        try:
            installed = weights.require_installed(self._config, manifest, spec)
        except weights.WeightsError as exc:
            raise ApiError(
                409,
                "model_not_installed",
                str(exc),
                {"model": model_id, "hf_repo": spec.hf_repo, "revision": spec.revision},
            ) from None
        return manifest, spec, python, installed.path

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        if model is None:  # unreachable: resolve_model requires one
            raise ApiError(400, "model_required", f"{self.name} needs a model")
        _params(params)
        _require_ffmpeg()
        _, spec, _, _ = self._require_runnable(model)
        accelerator.guard(
            self._config.backend_kind,
            model_id=model,
            need_bytes=spec.memory_bytes_estimate,
            owned_pids=self._owned_pids(),
            desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            # Deliberately no `reclaimable_bytes`. An `llm` load may unload the
            # previous resident to make room for itself; an asr job never unloads
            # somebody's model to run a transcript, so the memory a resident
            # engine holds is not memory this job can have.
        )

    # ------------------------------------------------------------------ run

    def run(self, job: Job, ctx: JobContext) -> None:
        params = AsrParams.model_validate(job.params)
        model = job.model
        if model is None:  # unreachable: resolve_model requires one
            raise JobError("model_required", f"{self.name} needs a model")
        audio = self._one_input(ctx)

        try:
            ffmpeg = _require_ffmpeg()
            manifest, spec, python, weights_dir = self._require_runnable(model)
            # The card can change between the queue and the lane, so the guard
            # runs again here against the same rules.
            state = accelerator.guard(
                self._config.backend_kind,
                model_id=model,
                need_bytes=spec.memory_bytes_estimate,
                owned_pids=self._owned_pids(),
                desktop_allowance_bytes=self._config.desktop_allowance_bytes,
            )
        except ApiError as exc:
            raise JobError(exc.code, exc.message) from None
        ctx.warming(state.detail)

        # Zeros rather than absent fields: every `stage` line carries the same
        # three numbers, so a consumer reads one shape and never has to ask
        # whether this particular event happens to have them. A `total_s` of 0
        # is what "the container has not been probed yet" looks like, and it is
        # what BookForge's own decode line reports before ffprobe answers.
        ctx.progress(
            0.0,
            f"decoding {audio.name}",
            stage="decoding",
            processed_s=0.0,
            total_s=0.0,
            cues=0,
        )
        outcome = self._transcribe(ctx, job, python, weights_dir, ffmpeg, audio, params)

        windows = outcome.ready["windows"]
        try:
            results = workers.require_positional_results(outcome, windows, "window")
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None

        failures = [
            f"window {index} ({index * WINDOW_SECONDS}s): {result['error']}"
            for index, result in enumerate(results)
            if "error" in result
        ]
        if failures:
            # No artifact. A transcript with a fifteen-minute hole in the middle
            # looks exactly like a transcript without one.
            raise JobError(
                "asr_window_failed",
                f"{len(failures)} of {windows} window(s) failed, so the transcript "
                "would have holes in it and nothing in the file would say where: "
                + "; ".join(failures),
            )

        document = self._transcript(model, spec, params, outcome, results)
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

    @staticmethod
    def _one_input(ctx: JobContext) -> Path:
        """The single audio file, or a refusal naming what arrived instead."""
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
        python: Path,
        weights_dir: Path,
        ffmpeg: str,
        audio: Path,
        params: AsrParams,
    ) -> workers.WorkerOutcome:
        request = {
            "model_dir": str(weights_dir),
            "ffmpeg": ffmpeg,
            "audio": str(audio),
            "language": params.whisper_language(),
            "vad_filter": params.vad_filter,
            "word_timestamps": params.word_timestamps,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "window_s": WINDOW_SECONDS,
            "overlap_s": OVERLAP_SECONDS,
        }

        def on_ready(message: dict[str, Any]) -> None:
            ctx.warming(
                f"{message['duration_s']:.0f}s of audio decoded, "
                f"{message['windows']} window(s) of {WINDOW_SECONDS}s to transcribe "
                f"on {message['device']} at {message['compute_type']}"
            )

        def on_progress(message: dict[str, Any]) -> None:
            processed = float(message["processed_s"])
            total = float(message["total_s"])
            stage = message["stage"]
            # The decode phase drives no fraction. It is real work with a real
            # position, but none of the transcript exists yet, and a bar that
            # counts the decode as progress towards the transcript is a bar that
            # lies. `stage` plus the two second counts is exactly what
            # BookForge's own parser reads off its DECODE and PROGRESS lines.
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

        try:
            return workers.run_worker(
                python=python,
                script=WORKER_SCRIPT,
                request=request,
                log_path=self._config.logs_dir / f"asr-{job.id}.log",
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
                on_ready=on_ready,
                on_progress=on_progress,
                cancelled=lambda: ctx.cancelled,
            )
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None

    # ----------------------------------------------------------- transcript

    @staticmethod
    def _transcript(
        model: str,
        spec: Any,
        params: AsrParams,
        outcome: workers.WorkerOutcome,
        results: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        """Window-relative results into one absolute, deduplicated document.

        A result's window is its **position** in the stream and nothing else —
        the worker reports no index, because an index a worker reports is an
        index a worker can get wrong. Window `n` starts at `n * WINDOW_SECONDS`
        by construction, so the shift is arithmetic the server does.
        """
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

        # Each window reaches OVERLAP_SECONDS past its own boundary, so the last
        # seconds of every window are spoken again at the start of the next one.
        # Sort by start and drop anything that begins inside a span already kept.
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

        # Every window detects the language independently when none was given.
        # The first window's answer is the document's, because that is the one
        # BookForge shows and the one a re-run reproduces; the rest are the same
        # answer on any real book and the disagreement is not something Crucible
        # is in a position to adjudicate.
        first = results[0]
        return {
            "model": model,
            "revision": spec.revision,
            "hf_repo": spec.hf_repo,
            "language": first["language"],
            "language_probability": first["language_probability"],
            "language_requested": params.language,
            "vad_filter": params.vad_filter,
            "word_timestamps": params.word_timestamps,
            "duration_s": outcome.ready["duration_s"],
            "window_s": WINDOW_SECONDS,
            "overlap_s": OVERLAP_SECONDS,
            "windows": outcome.ready["windows"],
            "segments": deduplicated,
        }
