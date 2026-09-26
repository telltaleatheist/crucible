"""`asr` on Qwen3-ASR: pieces, a serving engine, the aligner, and the loop guard.

docs/PHASE25-QWEN-ASR.md. The whisper engines are one worker, one request, one
exit (`crucible/jobs/asr/__init__.py`). This engine is two models and a policy,
so the job drives them itself:

1. **The ASR session** — `qwen_worker.py` in the llm env: vLLM 0.29.0 in-process
   on cuda-linux, mlx-audio 0.5.5 on mlx-darwin. It decodes the input once,
   cuts it at quiet points into pieces of at most 180 s (the aligner's limit),
   writes each as a wav, and decodes pieces in batches of the manifest's
   `max_batch`.
2. **The aligner session** — the `align` job type's own worker
   (`crucible/jobs/align/worker.py`) in the align env, running the manifest's
   `aligner` (`qwen3-aligner`, Qwen3-ForcedAligner-0.6B) exactly as `align`
   runs it. Only when the job asked for `word_timestamps`.
3. **The loop guard** (`loopguard.py`) between them: every decoded piece is
   read for a loop, every aligned piece for a collapse, and a piece that fails
   either is re-cut smaller and decoded again, up to the ladder's end, and then
   the job fails as `asr_decode_loop` naming where.

PER-JOB, NOT RESIDENT, AND WHY (docs/PHASE25 section 3)
-------------------------------------------------------
Both sessions belong to this job: started by it, stopped in its `finally`, and
held by nothing else — the owner of the hold is the job, and there is no hold
for a reconciler to find afterwards. That is `asr`'s existing shape, and it is
kept deliberately rather than by default:

- **The job needs TWO models on the card at once.** A word-timestamped piece is
  decoded, aligned, and — when the aligner collapses — decoded again, so the
  ASR model and the aligner are both live for the whole run. `Residency` holds
  ONE thing (a model, a voice or an aligner session); a resident ASR engine
  would be evicted by the aligner the job itself needs, every job.
- **The unit of work is hours of audio.** ContentStudio's stream was 3.9 h. A
  cold vLLM start (weights, CUDA graphs) is a minute or two against the
  twenty-odd minutes of work behind it.
- **It never unloads anybody.** Like every `asr` job it is refused by name,
  before it is queued, when the card will not hold it beside what is resident
  — it does not evict a cleanup model to transcribe.

What would change this is written down (PHASE25 section 9): residency that can
hold an ASR engine and its aligner as one resident thing, which is what a
streaming-transcription door would need anyway.

Why vLLM runs IN the worker process, not as `vllm serve`
---------------------------------------------------------
The resident `llm` engines are `vllm serve` behind a proxy. Here the engine is
a library call inside the job's own worker, for the reasons above plus one:
the HTTP transcription door decodes uploaded audio with vLLM's `[audio]`
extras (av, soundfile, soxr), which the pinned llm env does not carry, while
the in-process call takes the samples this worker already decoded, at 16 kHz,
where vLLM resamples nothing. Same engine, same paged KV cache, same CUDA
graphs and continuous batching; no env change.

`VLLM_ENABLE_V1_MULTIPROCESSING=0` keeps vLLM's engine core in the worker's
own process (`vllm/envs.py` L1386 at v0.29.0, default 1). With it on, the core
is a grandchild holding the card; a stop that reaches the worker's group
reaches it too (`crucible/procgroup.py`), but one process holding CUDA is one
process to account for, and `workers.py`'s never-SIGKILL rule is simplest to
keep when the thing that must exit cleanly is the thing that was asked to.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ... import weights, workerenv, workers
from ...alignmodels import AlignBackendSpec, AlignManifest, AlignManifestError
from ...alignmodels import load_align_manifest
from ...asrmodels import (
    MLX_AUDIO_ENGINE,
    QWEN_ASR_TORCH_ENGINE,
    QWEN_CONTEXT_MAX_TOKENS,
    QWEN_PIECE_MAX_SECONDS,
    VLLM_ENGINE,
    AsrBackendSpec,
)
from ...config import Config
from ...engines.vllm import ENVIRONMENT as VLLM_ENVIRONMENT
from ...errors import ApiError, JobError
from ..align import QWEN3_LANGUAGES, QWEN3_MAX_AUDIO_S
from ..align import WORKER_SCRIPT as ALIGN_WORKER_SCRIPT
from ..align import device_for as align_device_for
from ..base import Job, JobContext
from . import loopguard

QWEN_WORKER_SCRIPT = Path(__file__).resolve().parent / "qwen_worker.py"

#: How long either session may say nothing at all before it becomes ready.
#: `asr`'s own figure, and the longest quiet stretch is the ASR load: 4.7 GB of
#: weights from a cold disk, then vLLM's CUDA-graph capture for each batch size
#: up to `max_batch`. After `ready` there is no timeout (`workers.py`).
READY_SILENCE_TIMEOUT_SECONDS = 900.0

#: The environment of the ASR worker, per engine. vLLM's is the resident
#: engine's own (`crucible/engines/vllm.py`, one owner) plus the in-process
#: engine core (this module's docstring). mlx-audio needs nothing set.
WORKER_ENVIRONMENT_FOR_ENGINE: dict[str, dict[str, str]] = {
    VLLM_ENGINE: {**VLLM_ENVIRONMENT, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
    MLX_AUDIO_ENGINE: {},
    # Qwen's own package on torch: the same environment the aligner worker,
    # which imports the same package from the same env, has always run with.
    QWEN_ASR_TORCH_ENGINE: {},
}


# ------------------------------------------------------------------ planning


@dataclass(frozen=True)
class AlignerPlan:
    """The aligner a word-timestamped job runs, resolved and runnable."""

    manifest: AlignManifest
    spec: AlignBackendSpec
    python: Path
    weights_dir: Path


def aligner_spec(asr: AsrBackendSpec, backend_kind: str) -> tuple[AlignManifest, AlignBackendSpec]:
    """The manifest's `aligner`, on this backend, or a refusal naming it.

    A 500 and not a 400: the ASR manifest is the server's own file, and an
    `aligner` it names that does not exist is this build's misconfiguration,
    not the caller's.
    """
    aligner_id = asr.require("aligner")
    try:
        manifest = load_align_manifest(aligner_id)
    except AlignManifestError as exc:
        raise ApiError(
            500,
            "asr_manifests_unreadable",
            f"the {asr.backend} block names aligner {aligner_id!r}, which this "
            f"build cannot load: {exc}",
        ) from None
    if not manifest.supports(backend_kind):
        raise ApiError(
            500,
            "asr_manifests_unreadable",
            f"the {asr.backend} block names aligner {aligner_id!r}, which has no "
            f"{backend_kind} block ({manifest.path.name} declares "
            f"{sorted(manifest.backends)})",
        )
    return manifest, manifest.spec(backend_kind)


def need_bytes(asr: AsrBackendSpec, backend_kind: str, with_aligner: bool) -> int:
    """What the card must hold for this job: the ASR engine, and its aligner.

    Two manifests, two figures, added — never a copy of the aligner's number in
    the ASR manifest, which would go stale the day the aligner is re-measured.
    """
    total = asr.memory_bytes_estimate
    if with_aligner:
        _, spec = aligner_spec(asr, backend_kind)
        total += spec.memory_bytes_estimate
    return total


def plan_aligner(config: Config, asr: AsrBackendSpec, backend_kind: str) -> AlignerPlan:
    """The aligner's env and weights, or the same refusals `align` makes."""
    manifest, spec = aligner_spec(asr, backend_kind)
    try:
        python = workerenv.require_env(config.home, "align", backend_kind)
    except workerenv.WorkerEnvError as exc:
        raise ApiError(
            409,
            "env_missing",
            f"word timestamps on {asr.hf_repo} need the aligner {manifest.id!r}, "
            f"and {exc}",
            {"model": manifest.id, "env": str(workerenv.worker_env_dir(config.home, "align"))},
        ) from None
    try:
        installed = weights.require_installed(config, manifest, spec)
    except weights.WeightsError as exc:
        raise ApiError(
            409,
            "model_not_installed",
            f"word timestamps need the aligner {manifest.id!r}: {exc}",
            {"model": manifest.id, "hf_repo": spec.hf_repo, "revision": spec.revision},
        ) from None
    return AlignerPlan(manifest=manifest, spec=spec, python=python, weights_dir=installed.path)


def gpu_memory_utilization(estimate: int, card_bytes: int) -> float:
    """vLLM's startup gate, as the share of the card the estimate is.

    With `kv_cache_memory_bytes` stated, vLLM uses `gpu_memory_utilization` only
    to refuse a start when less than that share of the card is free
    (`v1/worker/utils.py` `request_memory` at v0.29.0). Rounded UP to the
    hundredth, so the gate is never looser than the guard that admitted the job.
    """
    if card_bytes <= 0:
        raise JobError(
            "accelerator_unreadable",
            f"this host reports a card of {card_bytes} bytes; vLLM's start gate is "
            "a share of the card and cannot be computed from that",
        )
    share = math.ceil(estimate / card_bytes * 100) / 100
    if share > 1.0:
        raise JobError(
            "model_too_large_for_host",
            f"the ASR engine's estimate {estimate} B is more than this card's "
            f"{card_bytes} B",
        )
    return share


# -------------------------------------------------------------------- pieces


@dataclass
class Piece:
    """One stretch of the input, in absolute seconds, and what became of it."""

    start_s: float
    duration_s: float
    wav: str
    level: int
    budget: int = 0
    text: str = ""
    tokens: int = 0
    hit_token_limit: bool = False
    items: list[dict[str, Any]] = field(default_factory=list)

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def where(self) -> str:
        return (
            f"{self.start_s:.1f}-{self.end_s:.1f}s "
            f"({loopguard.clock(self.start_s)}-{loopguard.clock(self.end_s)})"
        )


# ----------------------------------------------------------------------- run


class QwenAsrRun:
    """One job's transcription on a Qwen3-ASR engine. `run()` returns the document."""

    def __init__(
        self,
        *,
        config: Config,
        backend: Any,
        ctx: JobContext,
        job: Job,
        model: str,
        spec: AsrBackendSpec,
        python: Path,
        weights_dir: Path,
        aligner: AlignerPlan | None,
        ffmpeg: str,
        audio: Path,
        language: str,
        context: str | None,
        word_timestamps: bool,
    ) -> None:
        self._config = config
        self._backend = backend
        self._ctx = ctx
        self._job = job
        self._model = model
        self._spec = spec
        self._python = python
        self._weights_dir = weights_dir
        self._aligner = aligner
        self._ffmpeg = ffmpeg
        self._audio = audio
        self._language = language
        self._context = context
        self._word_timestamps = word_timestamps
        self._asr: workers.WorkerSession | None = None
        self._align: workers.WorkerSession | None = None
        self._duration_s = 0.0
        self._finished: list[Piece] = []
        self._redecoded: list[dict[str, Any]] = []
        self._silent = 0

    # -------------------------------------------------------------- driving

    def run(self) -> dict[str, Any]:
        try:
            self._start_asr()
            pending = self._split(str(self._audio), level=0, base_s=0.0)
            if self._word_timestamps:
                self._start_aligner()
            while pending:
                pending = self._round(pending)
        finally:
            self._stop_all()
        return self._document()

    def _round(self, pending: list[Piece]) -> list[Piece]:
        """Decode, check, align, check. Returns what must be decoded again."""
        again: list[Piece] = []
        self._transcribe(pending)
        to_align: list[Piece] = []
        for piece in pending:
            signal = loopguard.text_signal(
                piece.text, piece.duration_s, piece.hit_token_limit, piece.budget
            )
            if signal is not None:
                again += self._redecode(piece, signal)
            elif not loopguard.words_of(piece.text):
                # The model heard nothing to write down — an empty string, or
                # punctuation with no word in it. Not a hole: the same answer
                # whisper gives a silent stretch, which is no segment. (And not
                # one for the aligner, which has no word to place and says so
                # by failing the chunk.)
                self._silent += 1
            elif self._word_timestamps:
                to_align.append(piece)
            else:
                self._land(piece)
        if to_align:
            self._align_pieces(to_align)
            for piece in to_align:
                signal = loopguard.alignment_signal(piece.items)
                if signal is not None:
                    again += self._redecode(piece, signal)
                else:
                    self._land(piece)
        return again

    def _redecode(self, piece: Piece, signal: loopguard.LoopSignal) -> list[Piece]:
        """The next rung of the budget for one piece, or `asr_decode_loop`."""
        window = loopguard.next_window(piece.level)
        rungs = ", ".join(f"{s} s" for s in loopguard.WINDOW_LADDER_SECONDS)
        if window is None:
            raise JobError(
                "asr_decode_loop",
                f"the piece at {piece.where()} still loops after re-decoding at "
                f"every window in the budget ({rungs}): {signal.detail}. Nothing "
                "was published — a transcript with this stretch missing would "
                "look exactly like one without it",
            )
        self._redecoded.append(
            {
                "start": piece.start_s,
                "end": piece.end_s,
                "window_s": loopguard.WINDOW_LADDER_SECONDS[piece.level],
                "next_window_s": window,
                "signal": signal.kind,
                "detail": signal.detail,
            }
        )
        self._ctx.note(
            f"re-decoding {piece.where()} in pieces of at most {window} s: "
            f"{signal.detail}"
        )
        return self._split(piece.wav, level=piece.level + 1, base_s=piece.start_s)

    def _land(self, piece: Piece) -> None:
        self._finished.append(piece)
        self._ctx.progress(
            min(1.0, self._landed_s() / self._duration_s),
            f"{len(self._finished)} piece(s), {self._landed_s():.0f}s of "
            f"{self._duration_s:.0f}s transcribed",
            stage="aligning" if self._word_timestamps else "transcribing",
            processed_s=self._landed_s(),
            total_s=self._duration_s,
            cues=len(self._finished),
        )

    def _landed_s(self) -> float:
        return sum(piece.duration_s for piece in self._finished)

    # ------------------------------------------------------------- sessions

    def _start_asr(self) -> None:
        engine = self._spec.engine
        vllm = engine == VLLM_ENGINE
        request = {
            "op": "load",
            "engine": engine,
            "model_dir": str(self._weights_dir),
            "dtype": self._spec.require("dtype"),
            "max_batch": self._spec.require("max_batch"),
            "max_new_tokens": self._spec.require("max_new_tokens"),
            # vLLM's three; null on mlx-audio, which has no such knobs. Sent
            # either way so the wire has no optional keys.
            "max_model_len": self._spec.require("max_model_len") if vllm else None,
            "kv_cache_memory_bytes": (
                self._spec.require("kv_cache_memory_bytes") if vllm else None
            ),
            "gpu_memory_utilization": (
                gpu_memory_utilization(
                    self._spec.memory_bytes_estimate, self._backend.gpu.vram_bytes
                )
                if vllm
                else None
            ),
            "language": QWEN3_LANGUAGES[self._language],
            "context": self._context,
            "context_max_tokens": QWEN_CONTEXT_MAX_TOKENS,
            # The torch device for Qwen's own package: the aligner's own answer
            # for this backend (`mps` on the Mac), one owner. Null on the two
            # engines that choose their device themselves.
            "device": (
                align_device_for(self._config.backend_kind)
                if engine == QWEN_ASR_TORCH_ENGINE
                else None
            ),
        }
        environment = WORKER_ENVIRONMENT_FOR_ENGINE.get(engine)
        if environment is None:
            raise JobError(
                "engine_unsupported",
                f"there is no worker environment for asr engine {engine!r}",
            )
        session = workers.WorkerSession(
            python=self._python,
            script=QWEN_WORKER_SCRIPT,
            log_path=self._config.logs_dir / f"asr-{self._job.id}.log",
            environment=environment,
        )
        self._ctx.warming(
            f"loading {self._model} on {engine} at {request['dtype']}; log "
            f"{session.log_path}"
        )
        try:
            outcome = session.start(
                request, ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS
            )
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None
        self._asr = session
        ready = outcome.ready
        self._ctx.warming(
            f"{self._model} ready on {ready['device']} at {ready['dtype']} in "
            f"{float(ready['seconds']):.0f}s; context {ready['context_tokens']} "
            "token(s)"
        )

    def _start_aligner(self) -> None:
        plan = self._aligner
        if plan is None:  # unreachable: `run` plans it whenever timestamps are on
            raise JobError("worker_failed", "word timestamps without an aligner plan")
        device = align_device_for(self._config.backend_kind)
        session = workers.WorkerSession(
            python=plan.python,
            script=ALIGN_WORKER_SCRIPT,
            log_path=self._config.logs_dir / f"asr-{self._job.id}-aligner.log",
            # Beside vLLM on the PC, so capped at its OWN admitted share, not the
            # card (`workerenv.torch_memory_cap`).
            environment=workerenv.torch_allocator_environment(
                self._config.backend_kind
            ),
        )
        self._ctx.warming(
            f"loading {plan.manifest.id} on {device} at {plan.spec.dtype} for the "
            f"word times; log {session.log_path}"
        )
        try:
            session.start(
                {
                    "op": "load",
                    "model_dir": str(plan.weights_dir),
                    "device": device,
                    "dtype": plan.spec.dtype,
                    "memory_cap_bytes": workerenv.torch_memory_cap(
                        self._config.backend_kind, plan.spec.memory_bytes_estimate
                    ),
                },
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
            )
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None
        self._align = session

    def _send(
        self,
        session: workers.WorkerSession | None,
        request: dict[str, Any],
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> workers.WorkerOutcome:
        if session is None:  # unreachable: every caller runs after its start
            raise JobError("worker_failed", f"no session for {request['op']!r}")
        try:
            return session.send(
                request,
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
                on_progress=on_progress,
                cancelled=lambda: self._ctx.cancelled,
            )
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None

    def _stop_all(self) -> None:
        """Both sessions off the card, whatever happened. Never replaces the error.

        A stop that fails is a worker that did not go on SIGTERM (`workers.py`
        never escalates): the card is held by a process no row points at. That
        is said — on the job's stream and in the server's log — and it does not
        replace the failure or the success being reported, because the cleanup
        is not the operation (`AlignJobType._forget`'s rule).
        """
        for name, session in (("aligner", self._align), ("asr", self._asr)):
            if session is None:
                continue
            try:
                session.stop()
            except workers.WorkerError as exc:
                line = f"the {name} worker would not stop: {exc}"
                print(f"crucible: {line}", file=sys.stderr)
                self._ctx.note(line)
        self._align = None
        self._asr = None

    # ------------------------------------------------------------------ ops

    def _split(self, source: str, *, level: int, base_s: float) -> list[Piece]:
        window = loopguard.WINDOW_LADDER_SECONDS[level]
        out_dir = self._ctx.scratch / "pieces" / f"w{window}"

        def on_progress(message: dict[str, Any]) -> None:
            if level == 0:
                # Decode drives no fraction, for `asr`'s reason: none of the
                # transcript exists yet.
                self._ctx.progress(
                    0.0,
                    f"decoding {self._audio.name}: {float(message['processed_s']):.0f}s",
                    stage="decoding",
                    processed_s=float(message["processed_s"]),
                    total_s=0.0,
                    cues=0,
                )

        outcome = self._send(
            self._asr,
            {
                "op": "split",
                "ffmpeg": self._ffmpeg,
                "source": source,
                "max_piece_s": window,
                "out_dir": str(out_dir),
            },
            on_progress,
        )
        count = int(outcome.ready["pieces"])
        try:
            results = workers.require_positional_results(outcome, count, "piece")
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None
        if level == 0:
            self._duration_s = float(outcome.ready["duration_s"])
            self._ctx.warming(
                f"{self._duration_s:.0f}s of audio in {count} piece(s) of at most "
                f"{window}s"
            )
        max_new = int(self._spec.require("max_new_tokens"))
        return [
            Piece(
                start_s=base_s + float(result["offset_s"]),
                duration_s=float(result["duration_s"]),
                wav=str(result["wav"]),
                level=level,
                budget=loopguard.token_budget(float(result["duration_s"]), max_new),
            )
            for result in results
        ]

    def _transcribe(self, pieces: list[Piece]) -> None:
        def on_progress(message: dict[str, Any]) -> None:
            done = int(message["processed"])
            heard = self._landed_s() + sum(p.duration_s for p in pieces[:done])
            self._ctx.progress(
                min(1.0, heard / self._duration_s),
                f"transcribed {done} of {len(pieces)} piece(s)",
                stage="transcribing",
                processed_s=heard,
                total_s=self._duration_s,
                cues=len(self._finished),
            )

        outcome = self._send(
            self._asr,
            {
                "op": "transcribe",
                "pieces": [{"wav": p.wav, "max_tokens": p.budget} for p in pieces],
            },
            on_progress,
        )
        try:
            results = workers.require_positional_results(outcome, len(pieces), "piece")
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None
        for piece, result in zip(pieces, results):
            piece.text = str(result["text"])
            piece.tokens = int(result["tokens"])
            piece.hit_token_limit = bool(result["hit_token_limit"])

    def _align_pieces(self, pieces: list[Piece]) -> None:
        outcome = self._send(
            self._align,
            {
                "op": "align",
                "language": QWEN3_LANGUAGES[self._language],
                "max_audio_s": QWEN3_MAX_AUDIO_S,
                "ffmpeg": self._ffmpeg,
                "chunks": [{"audio": p.wav, "text": p.text} for p in pieces],
            },
        )
        try:
            results = workers.require_positional_results(outcome, len(pieces), "piece")
        except workers.WorkerError as exc:
            raise JobError("worker_failed", str(exc)) from None
        failures = [
            f"{piece.where()}: {result['error']}"
            for piece, result in zip(pieces, results)
            if "error" in result
        ]
        if failures:
            # No artifact, for the reason a failed whisper window gets none: a
            # stretch with no word times in a word-timestamped transcript is a
            # hole nothing in the file would point at.
            raise JobError(
                "asr_align_failed",
                f"the aligner failed {len(failures)} piece(s), so their words "
                "would have no times: " + "; ".join(failures),
            )
        for piece, result in zip(pieces, results):
            piece.items = list(result["items"])

    # ------------------------------------------------------------- document

    def _document(self) -> dict[str, Any]:
        segments = []
        for piece in sorted(self._finished, key=lambda p: p.start_s):
            row: dict[str, Any] = {
                "start": piece.start_s,
                "end": piece.end_s,
                "text": piece.text,
            }
            if self._word_timestamps:
                row["words"] = [
                    {
                        "start": piece.start_s + float(item["start"]),
                        "end": piece.start_s + float(item["end"]),
                        "word": str(item["text"]),
                        # whisper's four keys, and the fourth is null on purpose:
                        # the aligner places words, it does not score them, and
                        # an invented confidence is worse than none.
                        "probability": None,
                    }
                    for item in piece.items
                ]
            segments.append(row)
        aligner = self._aligner
        return {
            "model": self._model,
            "revision": self._spec.revision,
            "hf_repo": self._spec.hf_repo,
            "engine": self._spec.engine,
            "dtype": self._spec.require("dtype"),
            "aligner": (
                {
                    "model": aligner.manifest.id,
                    "revision": aligner.spec.revision,
                    "hf_repo": aligner.spec.hf_repo,
                }
                if aligner is not None
                else None
            ),
            "language": self._language,
            # Asserted by the caller, not detected: this engine is always told
            # the language (docs/PHASE25 section 2), so 1.0 is the assertion —
            # the mlx-whisper worker's rule for a named language.
            "language_probability": 1.0,
            "language_requested": self._language,
            "vad_filter": False,
            "word_timestamps": self._word_timestamps,
            "initial_prompt": None,
            "context": self._context,
            "duration_s": self._duration_s,
            "piece_max_s": QWEN_PIECE_MAX_SECONDS,
            "pieces": len(self._finished) + self._silent,
            "silent_pieces": self._silent,
            "redecoded": self._redecoded,
            "segments": segments,
        }

