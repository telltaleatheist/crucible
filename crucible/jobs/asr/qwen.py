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

SPEECH ONLY (Owen, 2026-09-27; off by default)
----------------------------------------------
*"Sending silences through an asr model produces hallucination and
nonsense."* With `speech_only`, the worker's first `split` runs Silero VAD on
the CPU over the decoded source and cuts the SHORTENED signal
(`speechonly.py`); the cutter prefers the joins where a stretch was taken out.
Pieces, ownership (`own_words`), the loop guard and every re-cut then work on
that one timeline, exactly as they do on the source without it; `_document`
moves every time back to the source through the worker's `kept` table, and
the transcript lists what was removed.

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

from ... import asrplan, weights, workerenv, workers
from ...alignmodels import AlignBackendSpec, AlignManifest, AlignManifestError
from ...alignmodels import load_align_manifest
from ...asrmodels import (
    MLX_AUDIO_ENGINE,
    QWEN_ASR_TORCH_ENGINE,
    QWEN_CONTEXT_MAX_TOKENS,
    VLLM_ENGINE,
    AsrBackendSpec,
)
from ...config import Config
from ...engines.vllm import ENVIRONMENT as VLLM_ENVIRONMENT
from ...engines.vllm import run_dtype
from ...ladder import card_for
from ...errors import ApiError, JobError
from ..align import QWEN3_LANGUAGES, QWEN3_MAX_AUDIO_S
from ..align import WORKER_SCRIPT as ALIGN_WORKER_SCRIPT
from ..align import device_for as align_device_for
from ..base import Job, JobContext
from . import loopguard, speechonly

QWEN_WORKER_SCRIPT = Path(__file__).resolve().parent / "qwen_worker.py"

#: HOW BIG THE PIECES ARE AND HOW MUCH REAL AUDIO EACH CARRIES PAST ITS EDGES,
#: the caller's to set (`piece_s`, `overlap_s`; Owen, 2026-09-26) and these when
#: it does not.
#:
#: 30 s, not 180. training-pc-1 measured the 180 s cut dropping sentence
#: openings: a cut at a pause lands a piece's first word at sample zero, and the
#: model skips it (327 of 6,532 cues on The Coming of the Third Reich started
#: with a dropped word; the same audio with 1.5 s of lead-in heard it). 30 s is
#: the length WhisperX and faster-whisper settle on, and antirez's Qwen3-ASR port
#: measured 120 s pieces repeating ~20% of their text and 180 s ones looping.
#:
#: 0.4 s of real audio each side is faster-whisper's `speech_pad_ms` default;
#: Silero's own 30 ms is what Qwen's toolkit cuts with, and it has our problem.
#: Overlap needs word timestamps: a word heard twice is kept only by the piece
#: whose core holds its midpoint (`own_words`), and without word times nothing
#: can say where a word is. So a plain-text job's default overlap is 0, and a
#: plain-text job ASKING for overlap is refused (`asr/__init__.py`).
DEFAULT_PIECE_S = 30.0
DEFAULT_OVERLAP_S = 0.4
MIN_PIECE_S = 5.0
MAX_OVERLAP_S = 5.0

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


def need_bytes(
    asr: AsrBackendSpec, backend_kind: str, with_aligner: bool, width: int | None = None
) -> int:
    """What the card must hold for this job: the ASR engine, and its aligner.

    Two manifests, two figures, added — never a copy of the aligner's number in
    the ASR manifest, which would go stale the day the aligner is re-measured.
    `width` is the pieces at once the engine is started with (`serving_width`);
    None is the manifest's own `max_batch`, and its estimate unchanged.
    """
    total = asr.memory_bytes_estimate
    if width is not None and asr.engine == VLLM_ENGINE:
        total = asrplan.need(asr, width)
    if with_aligner:
        total += aligner_bytes(asr, backend_kind)
    return total


def aligner_bytes(asr: AsrBackendSpec, backend_kind: str) -> int:
    """The aligner's own manifest figure on this backend."""
    _, spec = aligner_spec(asr, backend_kind)
    return spec.memory_bytes_estimate


def serving_width(
    manifest: Any,
    asr: AsrBackendSpec,
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
    with_aligner: bool,
) -> int | None:
    """How many pieces at once the engine starts with on this card, or None.

    Owen, 2026-09-26: *"yes, fewer at once before quantizing for asr too"*
    (`crucible/asrplan.py`). On a card that cannot hold the manifest's
    `max_batch` pieces at once at full precision, the widest width that fits
    is taken, down to one, rather than refusing. The budget is the capability
    walk's (the card less the desktop allowance, as `ttsplan.load_plan` reads
    it), and a word-timestamped job counts its aligner too, so it may narrow
    further than the capability verdict (which is about the transcriber
    alone) said.

    None keeps `max_batch`: a model with no ladder (the Mac's engines take one
    piece per call already), or a card that holds the full width. On a card
    where not even one piece fits, the narrowest width is returned, so the
    guard refuses by name at the LEAST the job could run in and not at the
    declared width's figure (`ttsplan.load_plan` does the same for a voice).
    """
    ladder = asrplan.ladder_for(manifest, asr, backend_kind)
    if ladder is None:
        return None
    budget = max(0, total_bytes - desktop_allowance_bytes)
    extra = aligner_bytes(asr, backend_kind) if with_aligner else 0
    width = asrplan.width_for(asr, backend_kind, manifest, budget, extra)
    if width is None:
        width = ladder[-1].width
    return None if width == asr.max_batch else width


def floor_bytes(
    manifest: Any, asr: AsrBackendSpec, backend_kind: str, with_aligner: bool
) -> int:
    """The least this job can run in: one piece at a time, if there is a ladder.

    What `refuse_if_larger_than_host` checks, so "never on this host" is only
    said below it, and not about a card that fewer pieces at once would fit.
    """
    ladder = asrplan.ladder_for(manifest, asr, backend_kind)
    width = None if ladder is None else ladder[-1].width
    return need_bytes(asr, backend_kind, with_aligner, width)


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
    """One stretch of the input, in absolute seconds, and what became of it.

    `start_s`/`duration_s` are the CORE, the stretch this piece owns and
    reports; `audio_start_s`/`audio_duration_s` are what its wav holds, the core
    plus the overlap on each side. The aligner's times are relative to the
    audio start.
    """

    start_s: float
    duration_s: float
    audio_start_s: float
    audio_duration_s: float
    wav: str
    level: int
    budget: int = 0
    text: str = ""
    tokens: int = 0
    hit_token_limit: bool = False
    items: list[dict[str, Any]] = field(default_factory=list)
    #: With `speech_only`, every second above is on the SHORTENED signal's
    #: timeline and this maps it back (2026-09-27); None is the source's own.
    timeline: speechonly.Timeline | None = None

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def where(self) -> str:
        """Where this piece is, in the SOURCE's seconds: what a reader can find."""
        start, end = self.start_s, self.end_s
        if self.timeline is not None:
            start, end = self.timeline.span(start, end)
        return (
            f"{start:.1f}-{end:.1f}s "
            f"({loopguard.clock(start)}-{loopguard.clock(end)})"
        )


def own_words(piece: Piece, *, is_last: bool) -> None:
    """Keep the words whose midpoint lies in this piece's core, and their text.

    THE RULE THAT MAKES OVERLAP SAFE. With `overlap_s` of real audio on both
    sides, a word near a cut is heard by two pieces. Each keeps only the words
    whose midpoint falls in its own core, [start, end), so every word is kept
    exactly once and none is lost: the cores tile the source with no gap. The
    last piece's core is closed at its end so the final word has an owner.

    The TEXT is sliced to the kept words: from the first kept word's place in
    the decoded text to the last one's, widened over punctuation and quotes
    attached to them. The aligner's items are the text's own words in order,
    so each is found by a forward search; one that is not found is a defect in
    that assumption, and it fails the job by name rather than guessing a slice.
    """
    items = piece.items
    if not items:
        return
    core_start = piece.start_s
    core_end = piece.end_s
    keep: list[int] = []
    for position, item in enumerate(items):
        middle = piece.audio_start_s + (float(item["start"]) + float(item["end"])) / 2
        if middle >= core_start and (middle < core_end or (is_last and middle <= core_end)):
            keep.append(position)
    if len(keep) == len(items):
        return
    if not keep:
        piece.items = []
        piece.text = ""
        return
    spans = _word_spans(piece.text, items, piece.where())
    first = spans[keep[0]][0]
    last = spans[keep[-1]][1]
    text = piece.text
    while first > 0 and not text[first - 1].isspace():
        first -= 1
    while last < len(text) and not text[last].isspace():
        last += 1
    piece.items = [items[position] for position in keep]
    piece.text = text[first:last]


def _fold(character: str) -> str:
    """One character, lower-cased when that keeps it one character."""
    lowered = character.lower()
    return lowered if len(lowered) == 1 else character


def _letters(text: str) -> tuple[str, list[int]]:
    """`text`'s letters and digits, folded, and each one's index in `text`."""
    kept: list[str] = []
    where: list[int] = []
    for index, character in enumerate(text):
        if character.isalnum():
            kept.append(_fold(character))
            where.append(index)
    return "".join(kept), where


def _word_spans(
    text: str, items: list[dict[str, Any]], where: str
) -> list[tuple[int, int]]:
    """Each aligner item's [start, end) character span in `text`, in order.

    MATCHED ON LETTERS AND DIGITS ONLY, both sides (2026-09-26). The aligner
    returns its own normalisation of the text it was given, not the text: 
    "life-changing" comes back as `lifechanging`, and punctuation and case go.
    1.0.41 searched for each item verbatim and failed two whole jobs on the
    first hyphen (training-pc-1's tc.wav runs). Reduced to letters and digits,
    an item IS a run of the text's own letters, in order, whatever the aligner
    did to the spelling around them. An item with no letter or digit at all
    gets an empty span where the search stands.
    """
    letters, positions = _letters(text)
    cursor = 0
    spans: list[tuple[int, int]] = []
    for item in items:
        word, _ = _letters(str(item["text"]))
        here = positions[cursor] if cursor < len(positions) else len(text)
        if not word:
            spans.append((here, here))
            continue
        found = letters.find(word, cursor)
        if found < 0:
            raise JobError(
                "asr_overlap_unmapped",
                f"the piece at {where}: the aligner's word {item['text']!r} is not "
                "a run of the decoded text's letters after character "
                f"{here}, so the overlap cannot be trimmed to this piece's own "
                "words. Send overlap_s: 0 to run without overlap",
            )
        spans.append((positions[found], positions[found + len(word) - 1] + 1))
        cursor = found + len(word)
    return spans


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
        piece_s: float,
        overlap_s: float,
        width: int | None = None,
        speech: dict[str, Any] | None = None,
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
        self._piece_s = piece_s
        self._overlap_s = overlap_s
        #: Pieces at once (`serving_width`); None is the manifest's `max_batch`.
        self._width = width
        self._ladder = loopguard.window_ladder(piece_s)
        self._asr: workers.WorkerSession | None = None
        self._align: workers.WorkerSession | None = None
        #: `speech_only` (2026-09-27): the worker's settings object, or None.
        #: With it every piece lives on the shortened signal's timeline
        #: (`_duration_s` is ITS length), and `_timeline` maps back to the
        #: source, whose length is `_source_s`.
        self._speech = speech
        self._timeline: speechonly.Timeline | None = None
        self._source_s = 0.0
        self._duration_s = 0.0
        self._finished: list[Piece] = []
        self._redecoded: list[dict[str, Any]] = []
        self._silent = 0

    # -------------------------------------------------------------- driving

    def run(self) -> dict[str, Any]:
        try:
            self._start_asr()
            pending = self._split(level=0, region=None)
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
            # Over the audio the model HEARD, overlap included, since that is
            # what its text covers.
            signal = loopguard.text_signal(
                piece.text, piece.audio_duration_s, piece.hit_token_limit, piece.budget
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
                    continue
                own_words(piece, is_last=piece.end_s >= self._duration_s)
                if piece.items:
                    self._land(piece)
                else:
                    # Every word it heard was in its overlap, owned by a
                    # neighbour: this piece's own stretch was silence.
                    self._silent += 1
        return again

    def _redecode(self, piece: Piece, signal: loopguard.LoopSignal) -> list[Piece]:
        """The next rung of the budget for one piece, or `asr_decode_loop`."""
        window = loopguard.next_window(self._ladder, piece.level)
        rungs = ", ".join(f"{s:g} s" for s in self._ladder)
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
                "window_s": self._ladder[piece.level],
                "next_window_s": window,
                "signal": signal.kind,
                "detail": signal.detail,
            }
        )
        self._ctx.note(
            f"re-decoding {piece.where()} in pieces of at most {window:g} s: "
            f"{signal.detail}"
        )
        # The CORE is re-cut, out of the original source, so the smaller pieces'
        # overlap is the real audio either side and not the looping piece's own.
        return self._split(level=piece.level + 1, region=(piece.start_s, piece.end_s))

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

    def _run_dtype(self) -> str:
        """The dtype the ASR engine is STARTED in on this card, and the one the
        result records. The manifest's `bfloat16` (Owen's full-precision ruling
        of 2026-09-24), except under vLLM on a card without bf16, which runs it
        in float16: Owen, 2026-09-26, *"we can quantize if we need to. no less
        than 4"* (`engines.vllm.run_dtype`, fresh-install #48). Same two bytes a
        parameter, so the manifest's memory figures hold."""
        stated = self._spec.require("dtype")
        if self._spec.engine != VLLM_ENGINE:
            return stated
        return run_dtype(self._spec, card_for(self._config.home, self._backend.gpu))

    def _start_asr(self) -> None:
        engine = self._spec.engine
        vllm = engine == VLLM_ENGINE
        # FEWER AT ONCE BEFORE A SMALLER MODEL (Owen, 2026-09-26, `asrplan`):
        # a narrower width is a smaller KV pool and a smaller start gate, and
        # the rest of the manifest's figures are unchanged.
        narrowed = vllm and self._width is not None
        max_batch = self._width if narrowed else self._spec.require("max_batch")
        request = {
            "op": "load",
            "engine": engine,
            "model_dir": str(self._weights_dir),
            "dtype": self._run_dtype(),
            "max_batch": max_batch,
            "max_new_tokens": self._spec.require("max_new_tokens"),
            # vLLM's three; null on mlx-audio, which has no such knobs. Sent
            # either way so the wire has no optional keys.
            "max_model_len": self._spec.require("max_model_len") if vllm else None,
            "kv_cache_memory_bytes": (
                (
                    asrplan.kv_pool(self._spec, max_batch)
                    if narrowed
                    else self._spec.require("kv_cache_memory_bytes")
                )
                if vllm
                else None
            ),
            "gpu_memory_utilization": (
                gpu_memory_utilization(
                    asrplan.need(self._spec, max_batch)
                    if narrowed
                    else self._spec.memory_bytes_estimate,
                    self._backend.gpu.vram_bytes,
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
        pace = (
            f", {max_batch} piece(s) at a time instead of "
            f"{self._spec.require('max_batch')}: this card cannot hold more at "
            "once at full precision (fewer at once is tried before a smaller "
            "or quantized model)"
            if narrowed
            else ""
        )
        self._ctx.warming(
            f"loading {self._model} on {engine} at {request['dtype']}{pace}; log "
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

    def _split(
        self, *, level: int, region: tuple[float, float] | None
    ) -> list[Piece]:
        window = self._ladder[level]
        out_dir = self._ctx.scratch / "pieces" / f"level{level}"

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
                "source": str(self._audio),
                "max_piece_s": window,
                "out_dir": str(out_dir),
                "region_s": None if region is None else list(region),
                "overlap_s": self._overlap_s,
                "speech": self._speech,
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
            self._source_s = self._duration_s
            kept = ""
            if self._speech is not None:
                try:
                    self._timeline = speechonly.Timeline.from_ready(outcome.ready)
                except ValueError as exc:
                    raise JobError("worker_failed", str(exc)) from None
                self._source_s = self._timeline.total_samples / float(
                    speechonly.SAMPLE_RATE
                )
                kept = (
                    f" of speech (of {self._source_s:.0f}s; "
                    f"{len(self._timeline.removed())} stretch(es) without speech "
                    "taken out)"
                )
            self._ctx.warming(
                f"{self._duration_s:.0f}s of audio{kept} in {count} piece(s) of at "
                f"most {window:g}s, {self._overlap_s:g}s of overlap each side"
            )
        max_new = int(self._spec.require("max_new_tokens"))
        return [
            Piece(
                start_s=float(result["offset_s"]),
                duration_s=float(result["duration_s"]),
                audio_start_s=float(result["audio_offset_s"]),
                audio_duration_s=float(result["audio_duration_s"]),
                wav=str(result["wav"]),
                level=level,
                budget=loopguard.token_budget(
                    float(result["audio_duration_s"]), max_new
                ),
                timeline=self._timeline,
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

    def _span(self, start: float, end: float) -> tuple[float, float]:
        """A stretch on the pieces' timeline, in the source's seconds."""
        if self._timeline is None:
            return start, end
        return self._timeline.span(start, end)

    def _word(self, start: float, end: float) -> tuple[float, float]:
        """A word on the pieces' timeline, in the source's seconds: both ends in
        the kept region holding its middle, so the aligner can never stretch a
        word across a removed stretch (`speechonly.Timeline.word`)."""
        if self._timeline is None:
            return start, end
        return self._timeline.word(start, end)

    def _document(self) -> dict[str, Any]:
        """The transcript. Everything was done on the pieces' timeline (the
        shortened signal's, with `speech_only`); every time is moved to the
        SOURCE's here, and nowhere earlier, so ownership and the loop guard
        never saw two timelines."""
        segments = []
        for piece in sorted(self._finished, key=lambda p: p.start_s):
            start, end = self._span(piece.start_s, piece.end_s)
            row: dict[str, Any] = {
                "start": start,
                "end": end,
                "text": piece.text,
            }
            if self._word_timestamps:
                words = []
                for item in piece.items:
                    word_start, word_end = self._word(
                        piece.audio_start_s + float(item["start"]),
                        piece.audio_start_s + float(item["end"]),
                    )
                    words.append(
                        {
                            "start": word_start,
                            "end": word_end,
                            "word": str(item["text"]),
                            # whisper's four keys, and the fourth is null on
                            # purpose: the aligner places words, it does not
                            # score them, and an invented confidence is worse
                            # than none.
                            "probability": None,
                        }
                    )
                row["words"] = words
            segments.append(row)
        redecoded = []
        for entry in self._redecoded:
            start, end = self._span(entry["start"], entry["end"])
            redecoded.append({**entry, "start": start, "end": end})
        aligner = self._aligner
        return {
            "model": self._model,
            "revision": self._spec.revision,
            "hf_repo": self._spec.hf_repo,
            "engine": self._spec.engine,
            "dtype": self._run_dtype(),
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
            "duration_s": self._source_s,
            "piece_max_s": self._piece_s,
            "overlap_s": self._overlap_s,
            "pieces": len(self._finished) + self._silent,
            "silent_pieces": self._silent,
            "redecoded": redecoded,
            **speechonly.report(self._speech, self._timeline),
            "segments": segments,
        }

