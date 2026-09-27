from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ... import asrplan, jobenv, weights, workers
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
from ...cardfacts import card_for
from ...errors import ApiError, JobError
from ..align import QWEN3_LANGUAGES, QWEN3_MAX_AUDIO_S
from ..align import device_for as align_device_for
from ..align import start_aligner_session
from ..base import Job, JobContext
from . import loopguard, speechonly
from .document import progress_decoding, transcript_document, worker_failed

QWEN_WORKER_SCRIPT = Path(__file__).resolve().parent / "qwen_worker.py"

DEFAULT_PIECE_S = 30.0
DEFAULT_OVERLAP_S = 0.4
MIN_PIECE_S = 5.0
MAX_OVERLAP_S = 5.0

READY_SILENCE_TIMEOUT_SECONDS = 900.0

ALIGN_BATCH = 16

JOURNAL_FORMAT_VERSION = 1

WORKER_ENVIRONMENT_FOR_ENGINE: dict[str, dict[str, str]] = {
    VLLM_ENGINE: {**VLLM_ENVIRONMENT, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
    MLX_AUDIO_ENGINE: {},
    QWEN_ASR_TORCH_ENGINE: {},
}


@dataclass(frozen=True)
class AlignerPlan:

    manifest: AlignManifest
    spec: AlignBackendSpec
    python: Path
    weights_dir: Path


def aligner_spec(asr: AsrBackendSpec, backend_kind: str) -> tuple[AlignManifest, AlignBackendSpec]:
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
    total = asr.memory_bytes_estimate
    if width is not None and asr.engine == VLLM_ENGINE:
        total = asrplan.need(asr, width)
    if with_aligner:
        total += aligner_bytes(asr, backend_kind)
    return total


def aligner_bytes(asr: AsrBackendSpec, backend_kind: str) -> int:
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
    ladder = asrplan.ladder_for(manifest, asr, backend_kind)
    width = None if ladder is None else ladder[-1].width
    return need_bytes(asr, backend_kind, with_aligner, width)


def plan_aligner(config: Config, asr: AsrBackendSpec, backend_kind: str) -> AlignerPlan:
    manifest, spec = aligner_spec(asr, backend_kind)
    try:
        python = jobenv.require_env(config.home, jobenv.worker_env("align", backend_kind), backend_kind)
    except jobenv.EnvError as exc:
        raise ApiError(
            409,
            "env_missing",
            f"word timestamps on {asr.hf_repo} need the aligner {manifest.id!r}, "
            f"and {exc}",
            {"model": manifest.id, "env": str(config.home / "envs" / "align")},
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


def run_dtype_on(config: Config, backend: Any, spec: AsrBackendSpec) -> str:
    stated = spec.require("dtype")
    if spec.engine != VLLM_ENGINE:
        return stated
    return run_dtype(spec, card_for(config.home, backend.gpu))


def gpu_memory_utilization(estimate: int, card_bytes: int) -> float:
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


@dataclass
class Piece:

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
    timeline: speechonly.Timeline | None = None

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def where(self) -> str:
        start, end = speechonly.span(self.timeline, self.start_s, self.end_s)
        return (
            f"{start:.1f}-{end:.1f}s "
            f"({loopguard.clock(start)}-{loopguard.clock(end)})"
        )


def _sample(seconds: float) -> int:
    return int(round(float(seconds) * speechonly.SAMPLE_RATE))


def piece_key(piece: Piece) -> str:
    return f"L{piece.level}.{_sample(piece.start_s):011d}-{_sample(piece.end_s):011d}"


def _roundtrip(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _plan_difference(recorded: dict[str, Any], plan: dict[str, Any]) -> str:
    if recorded.get("duration_s") != plan.get("duration_s"):
        return (
            f"the audio decoded to {plan.get('duration_s')} s and the journal's "
            f"to {recorded.get('duration_s')} s"
        )
    if recorded.get("kept") != plan.get("kept"):
        return "speech_only kept different stretches of the audio"
    before, now = recorded.get("pieces") or [], plan.get("pieces") or []
    if len(before) != len(now):
        return f"{len(now)} piece(s) now and {len(before)} in the journal"
    for index, (old, new) in enumerate(zip(before, now)):
        if old != new:
            return (
                f"piece {index} is {new[0]:.3f}+{new[1]:.3f} s now and "
                f"{old[0]:.3f}+{old[1]:.3f} s in the journal"
            )
    return "the recorded plan differs in a field this build does not name"


def own_words(piece: Piece, *, is_last: bool) -> None:
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
    lowered = character.lower()
    return lowered if len(lowered) == 1 else character


def _letters(text: str) -> tuple[str, list[int]]:
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


class QwenAsrRun:

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
        journal: Any | None = None,
        resumed: bool = False,
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
        self._width = width
        self._ladder = loopguard.window_ladder(piece_s)
        self._asr: workers.WorkerSession | None = None
        self._align: workers.WorkerSession | None = None
        self._speech = speech
        self._timeline: speechonly.Timeline | None = None
        self._source_s = 0.0
        self._duration_s = 0.0
        self._finished: list[Piece] = []
        self._redecoded: list[dict[str, Any]] = []
        self._silent = 0
        self._text_published = False
        self._journal = journal
        self._resumed = resumed and journal is not None
        self._total = 0
        self._decoded = 0
        self._aligned = 0


    def run(self) -> dict[str, Any]:
        try:
            self._start_asr()
            pending = self._split(level=0, region=None)
            self._total = len(pending)
            if self._word_timestamps and not self._resumed:
                self._ensure_aligner()
            while pending:
                pending = self._round(pending)
        finally:
            self._stop_all()
            self._save_progress(force=True)
        return self._document()

    def _round(self, pending: list[Piece]) -> list[Piece]:
        again: list[Piece] = []
        self._transcribe(pending)
        to_align: list[Piece] = []
        for piece in pending:
            signal = loopguard.text_signal(
                piece.text, piece.audio_duration_s, piece.hit_token_limit, piece.budget
            )
            if signal is not None:
                again += self._redecode(piece, signal)
            elif not loopguard.words_of(piece.text):
                self._silent += 1
                self._verdict(piece, {"outcome": "silent"})
            elif self._word_timestamps:
                to_align.append(piece)
            else:
                self._land(piece)
        if to_align:
            if not self._text_published:
                self._publish_text(to_align)
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
                    self._silent += 1
                    self._verdict(piece, {"outcome": "silent"})
        return again

    def _redecode(self, piece: Piece, signal: loopguard.LoopSignal) -> list[Piece]:
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
        entry = {
            "start": piece.start_s,
            "end": piece.end_s,
            "window_s": self._ladder[piece.level],
            "next_window_s": window,
            "signal": signal.kind,
            "detail": signal.detail,
        }
        self._redecoded.append(entry)
        self._verdict(piece, {"outcome": "redecode", **entry})
        self._ctx.note(
            f"re-decoding {piece.where()} in pieces of at most {window:g} s: "
            f"{signal.detail}"
        )
        children = self._split(
            level=piece.level + 1, region=(piece.start_s, piece.end_s)
        )
        self._total += len(children) - 1
        return children

    def _land(self, piece: Piece) -> None:
        self._finished.append(piece)
        self._verdict(
            piece,
            {"outcome": "landed", "text": piece.text, "words": len(piece.items)},
        )
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


    def _unit(self, kind: str, piece: Piece) -> Any | None:
        if not self._resumed or self._journal is None:
            return None
        return self._journal.get(f"{kind}.{piece_key(piece)}")

    def _put(self, kind: str, piece: Piece, data: dict[str, Any]) -> None:
        if self._journal is None:
            return
        try:
            self._journal.put(f"{kind}.{piece_key(piece)}", data)
        except OSError as exc:
            raise JobError(
                "journal_unwritable",
                f"the piece at {piece.where()}: its {kind} could not be written to "
                f"journal {self._journal.id}: {type(exc).__name__}: {exc}",
            ) from None

    def _verdict(self, piece: Piece, verdict: dict[str, Any]) -> None:
        if self._journal is None:
            return
        verdict = _roundtrip(verdict)
        recorded = self._unit("verdict", piece)
        if recorded is None:
            self._put("verdict", piece, verdict)
        elif recorded != verdict:
            raise JobError(
                "resume_mismatch",
                f"the piece at {piece.where()}: journal {self._journal.id} says it "
                f"was {recorded.get('outcome')!r} and this run finds it "
                f"{verdict.get('outcome')!r} from the same text and word times, so "
                "the loop guard or the ownership rule changed since the journal "
                "was written. Send the job without resume to start fresh",
            )
        self._save_progress()

    def _check_plan(self, key: str, plan: dict[str, Any], where: str) -> None:
        if self._journal is None:
            return
        plan = _roundtrip(plan)
        recorded = self._journal.get(key) if self._resumed else None
        if recorded is None:
            try:
                self._journal.put(key, plan)
            except OSError as exc:
                raise JobError(
                    "journal_unwritable",
                    f"the piece plan for {where} could not be written to journal "
                    f"{self._journal.id}: {type(exc).__name__}: {exc}",
                ) from None
            return
        if recorded == plan:
            return
        raise JobError(
            "resume_plan_mismatch",
            f"the pieces cut from {where} are not the ones journal "
            f"{self._journal.id} recorded: {_plan_difference(recorded, plan)}. "
            "Its text belongs to other pieces of audio and cannot be stitched to "
            "these; send the job without resume to start fresh",
        )

    def _save_progress(self, *, force: bool = False) -> None:
        if self._journal is None or self._total <= 0:
            return
        done = len(self._finished) + self._silent
        detail = f"{self._decoded:,} decoded"
        if self._word_timestamps:
            detail += f", {self._aligned:,} aligned"
        try:
            self._journal.progress(
                done,
                self._total,
                f"{done:,} of {self._total:,} pieces done ({detail})",
                force=force,
            )
        except OSError as exc:
            print(
                f"crucible: journal {self._journal.id} progress not saved: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )


    def _run_dtype(self) -> str:
        return run_dtype_on(self._config, self._backend, self._spec)

    def _start_asr(self) -> None:
        engine = self._spec.engine
        vllm = engine == VLLM_ENGINE
        narrowed = vllm and self._width is not None
        max_batch = self._width if narrowed else self._spec.require("max_batch")
        request = {
            "op": "load",
            "engine": engine,
            "model_dir": str(self._weights_dir),
            "dtype": self._run_dtype(),
            "max_batch": max_batch,
            "max_new_tokens": self._spec.require("max_new_tokens"),
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
        with worker_failed():
            outcome = session.start(
                request, ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS
            )
        self._asr = session
        ready = outcome.ready
        self._ctx.warming(
            f"{self._model} ready on {ready['device']} at {ready['dtype']} in "
            f"{float(ready['seconds']):.0f}s; context {ready['context_tokens']} "
            "token(s)"
        )

    def _ensure_aligner(self) -> None:
        if self._align is None:
            self._start_aligner()

    def _start_aligner(self) -> None:
        plan = self._aligner
        if plan is None:
            raise JobError("worker_failed", "word timestamps without an aligner plan")
        device = align_device_for(self._config.backend_kind)
        log_path = self._config.logs_dir / f"asr-{self._job.id}-aligner.log"
        self._ctx.warming(
            f"loading {plan.manifest.id} on {device} at {plan.spec.dtype} for the "
            f"word times; log {log_path}"
        )
        with worker_failed():
            self._align = start_aligner_session(
                plan.python,
                plan.weights_dir,
                plan.spec,
                log_path,
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
            )

    def _send(
        self,
        session: workers.WorkerSession | None,
        request: dict[str, Any],
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        on_result: Callable[[dict[str, Any]], None] | None = None,
    ) -> workers.WorkerOutcome:
        if session is None:
            raise JobError("worker_failed", f"no session for {request['op']!r}")
        with worker_failed():
            return session.send(
                request,
                ready_silence_timeout=READY_SILENCE_TIMEOUT_SECONDS,
                on_progress=on_progress,
                cancelled=lambda: self._ctx.cancelled,
                on_result=on_result,
            )

    def _stop_all(self) -> None:
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


    def _split(
        self, *, level: int, region: tuple[float, float] | None
    ) -> list[Piece]:
        window = self._ladder[level]
        out_dir = self._ctx.scratch / "pieces" / f"level{level}"

        def on_progress(message: dict[str, Any]) -> None:
            if level == 0:
                progress_decoding(
                    self._ctx,
                    f"decoding {self._audio.name}: {float(message['processed_s']):.0f}s",
                    float(message["processed_s"]),
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
        with worker_failed():
            results = workers.require_positional_results(outcome, count, "piece")
        if level == 0:
            self._duration_s = float(outcome.ready["duration_s"])
            self._source_s = self._duration_s
            kept = ""
            with worker_failed(ValueError):
                self._timeline = speechonly.timeline_for(outcome.ready, self._speech)
            if self._timeline is not None:
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
        region_key = (
            "all" if region is None else f"{_sample(region[0]):011d}-{_sample(region[1]):011d}"
        )
        self._check_plan(
            f"plan.L{level}.{region_key}",
            {
                "window_s": window,
                "overlap_s": self._overlap_s,
                "duration_s": float(outcome.ready["duration_s"]),
                "samples": outcome.ready.get("samples"),
                "kept": outcome.ready.get("kept"),
                "pieces": [
                    [
                        float(result["offset_s"]),
                        float(result["duration_s"]),
                        float(result["audio_offset_s"]),
                        float(result["audio_duration_s"]),
                    ]
                    for result in results
                ],
            },
            "the whole input" if region is None else f"the re-cut of {region_key}",
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
        todo: list[Piece] = []
        for piece in pieces:
            unit = self._unit("text", piece)
            if unit is None:
                todo.append(piece)
                continue
            piece.text = str(unit["text"])
            piece.tokens = int(unit["tokens"])
            piece.hit_token_limit = bool(unit["hit_token_limit"])
            self._decoded += 1
        if len(todo) < len(pieces):
            self._ctx.note(
                f"{len(pieces) - len(todo):,} of {len(pieces):,} piece(s) already "
                f"decoded in journal {self._journal.id if self._journal else '?'}; "
                f"decoding {len(todo):,}"
            )
        if not todo:
            return

        def on_progress(message: dict[str, Any]) -> None:
            done = int(message["processed"])
            heard = self._landed_s() + sum(p.duration_s for p in todo[:done])
            self._ctx.progress(
                min(1.0, heard / self._duration_s),
                f"transcribed {done} of {len(todo)} piece(s)",
                stage="transcribing",
                processed_s=heard,
                total_s=self._duration_s,
                cues=len(self._finished),
            )

        landed = [0]

        def on_result(message: dict[str, Any]) -> None:
            position = landed[0]
            landed[0] += 1
            if position >= len(todo):
                return
            try:
                row = {
                    "text": str(message["text"]),
                    "tokens": int(message["tokens"]),
                    "hit_token_limit": bool(message["hit_token_limit"]),
                }
            except (KeyError, TypeError, ValueError):
                return
            piece = todo[position]
            piece.text = row["text"]
            piece.tokens = row["tokens"]
            piece.hit_token_limit = row["hit_token_limit"]
            self._put("text", piece, row)
            self._decoded += 1
            self._save_progress()

        outcome = self._send(
            self._asr,
            {
                "op": "transcribe",
                "pieces": [{"wav": p.wav, "max_tokens": p.budget} for p in todo],
            },
            on_progress,
            on_result,
        )
        with worker_failed():
            results = workers.require_positional_results(outcome, len(todo), "piece")
        for piece, result in zip(todo, results):
            piece.text = str(result["text"])
            piece.tokens = int(result["tokens"])
            piece.hit_token_limit = bool(result["hit_token_limit"])
        self._save_progress()

    def _publish_text(self, pieces: list[Piece]) -> None:
        rows = []
        for piece in sorted(pieces, key=lambda p: p.start_s):
            start, end = speechonly.span(self._timeline, piece.start_s, piece.end_s)
            audio_start, audio_end = speechonly.span(self._timeline, 
                piece.audio_start_s, piece.audio_start_s + piece.audio_duration_s
            )
            rows.append(
                {
                    "start": start,
                    "end": end,
                    "audio_start": audio_start,
                    "audio_end": audio_end,
                    "text": piece.text,
                }
            )
        document = {
            "model": self._model,
            "revision": self._spec.revision,
            "language": self._language,
            "duration_s": self._source_s,
            "piece_max_s": self._piece_s,
            "overlap_s": self._overlap_s,
            "word_times": False,
            "note": (
                "each piece's text as decoded, before alignment and before the "
                "overlap's words were given to their owner; rows overlap by up to "
                "overlap_s. The finished transcript is transcript.json"
            ),
            "pieces": rows,
        }
        path = self._ctx.scratch / "transcript.text.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        self._ctx.artifact("transcript.text.json", path)
        self._text_published = True
        self._ctx.note(
            f"published transcript.text.json ({len(rows)} piece(s)) before aligning"
        )

    def _align_pieces(self, pieces: list[Piece]) -> None:
        todo: list[Piece] = []
        for piece in pieces:
            unit = self._unit("words", piece)
            if unit is None:
                todo.append(piece)
                continue
            piece.items = list(unit["items"])
            self._aligned += 1
        if len(todo) < len(pieces):
            self._ctx.note(
                f"{len(pieces) - len(todo):,} of {len(pieces):,} piece(s) already "
                f"aligned in journal {self._journal.id if self._journal else '?'}; "
                f"aligning {len(todo):,}"
            )
        if not todo:
            return
        self._ensure_aligner()
        pieces = todo
        results: list[dict[str, Any]] = []
        aligned_s = 0.0
        for first in range(0, len(pieces), ALIGN_BATCH):
            batch = pieces[first:first + ALIGN_BATCH]
            outcome = self._send(
                self._align,
                {
                    "op": "align",
                    "language": QWEN3_LANGUAGES[self._language],
                    "max_audio_s": QWEN3_MAX_AUDIO_S,
                    "ffmpeg": self._ffmpeg,
                    "chunks": [{"audio": p.wav, "text": p.text} for p in batch],
                },
            )
            with worker_failed():
                landed = workers.require_positional_results(outcome, len(batch), "piece")
            results += landed
            for piece, result in zip(batch, landed):
                if "error" not in result:
                    self._put("words", piece, {"items": list(result["items"])})
                    self._aligned += 1
            self._save_progress()
            aligned_s += sum(p.end_s - p.start_s for p in batch)
            done = min(first + ALIGN_BATCH, len(pieces))
            self._ctx.progress(
                min(1.0, (self._landed_s() + aligned_s) / self._duration_s),
                f"word times for {done} of {len(pieces)} piece(s)",
                stage="aligning",
                processed_s=self._landed_s() + aligned_s,
                total_s=self._duration_s,
                cues=len(self._finished),
            )
        failures = [
            f"{piece.where()}: {result['error']}"
            for piece, result in zip(pieces, results)
            if "error" in result
        ]
        if failures:
            raise JobError(
                "asr_align_failed",
                f"the aligner failed {len(failures)} piece(s), so their words "
                "would have no times: " + "; ".join(failures),
            )
        for piece, result in zip(pieces, results):
            piece.items = list(result["items"])


    def _document(self) -> dict[str, Any]:
        segments = []
        for piece in sorted(self._finished, key=lambda p: p.start_s):
            row: dict[str, Any] = {
                "start": piece.start_s,
                "end": piece.end_s,
                "text": piece.text,
            }
            if self._word_timestamps:
                words = []
                for item in piece.items:
                    words.append(
                        {
                            "start": piece.audio_start_s + float(item["start"]),
                            "end": piece.audio_start_s + float(item["end"]),
                            "word": str(item["text"]),
                            "probability": None,
                        }
                    )
                row["words"] = words
            segments.append(speechonly.to_source(row, self._timeline))
        redecoded = []
        for entry in self._redecoded:
            start, end = speechonly.span(self._timeline, entry["start"], entry["end"])
            redecoded.append({**entry, "start": start, "end": end})
        aligner = self._aligner
        return transcript_document(
            model=self._model,
            spec=self._spec,
            engine={
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
            },
            language=self._language,
            language_probability=1.0,
            language_requested=self._language,
            vad_filter=False,
            word_timestamps=self._word_timestamps,
            initial_prompt=None,
            prompt={"context": self._context},
            duration_s=self._source_s,
            layout={
                "piece_max_s": self._piece_s,
                "overlap_s": self._overlap_s,
                "pieces": len(self._finished) + self._silent,
                "silent_pieces": self._silent,
                "redecoded": redecoded,
            },
            speech=self._speech,
            timeline=self._timeline,
            segments=segments,
        )

