from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import workerio

sys.path.pop(0)
workerio.claim_stdout()

import gc
import json
import shutil
import threading
import time

from workerio import cap_memory, send

yue2native = workerio.load_sibling("yue2native", __file__)
# The YuE2 score-length reader the audio job uses (standard library only).
scorelength = workerio.load_sibling(
    "scorelength",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio", "scorelength.py"),
)

LABEL = "score"

WHY_REQUIRED = "every parameter is required because each one changes the score"

SAMPLE_RATE = 24_000

# SheetSage2 writes its own outputs (score.abc, transcription.mid, result.json, the
# annotation files) into the job's directory; the MIDI is the one taken from them. Its
# score is taken from the result itself, the same text it writes.
SHEETSAGE2_MIDI = "transcription.mid"

# The share of the bar each stage fills: the windows are the work.
READING, WINDOWS, NOTATION = 0.02, 0.9, 0.08

ENCODE_SHARE = 0.1

# Torch threads, as SheetSage2's own entry points set them (infer.py, transcribe_cover.py).
MAX_THREADS = 4

YUE2_TOKENS_NOTE = (
    "not counted: the YuE2 tokenizer is not in the score env, and YuE2 tokenizes a "
    "score inside its own pipeline. `chars` is the score's length in characters; YuE2 "
    "writes a score of up to 4,096 of its own tokens (docs/SCORE.md)"
)

_STATE: dict = {"engine": None}


class Cancelled(Exception):
    def __init__(self, stage: str) -> None:
        super().__init__(f"cancelled while {stage}")
        self.stage = stage


class CancelBox:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_id = None

    def ask(self, request_id) -> None:
        with self._lock:
            self._request_id = request_id

    def asked(self, request_id: str) -> bool:
        with self._lock:
            return self._request_id == request_id

    def clear(self, request_id: str) -> None:
        with self._lock:
            if self._request_id == request_id:
                self._request_id = None


CANCEL = CancelBox()


def require(request: dict, key: str, kind):
    return workerio.require(request, key, kind, LABEL, WHY_REQUIRED)


def nullable(request: dict, key: str, kind):
    if key not in request:
        raise KeyError(f"the {LABEL} request has no {key!r}; it is required and null when unset")
    return None if request[key] is None else require(request, key, kind)


class Job:
    def __init__(self, request: dict) -> None:
        self.request_id = require(request, "request_id", str)
        self.audio_path = require(request, "audio_path", str)
        self.samples = require(request, "samples", int)
        self.keep_harmony = require(request, "keep_harmony", bool)
        self.prompts = require(request, "prompts", list)
        self.output_dir = require(request, "output_dir", str)
        self.score_path = require(request, "score_path", str)
        self.instrumental_path = require(request, "instrumental_path", str)
        self.raw_path = require(request, "raw_path", str)
        self.midi_path = require(request, "midi_path", str)


class Progress:
    """SheetSage2's own progress callback, as fractions and stages, and the cancel check:
    it is called every 64 decoded tokens, so a cancel lands within a few of them."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.stage = None
        self.stage_seconds: dict = {}
        self._started = time.time()
        self.windows = 1

    def check(self) -> None:
        if CANCEL.asked(self.request_id):
            raise Cancelled(self.stage)

    def enter(self, stage: str, fraction: float, **fields) -> None:
        self.check()
        now = time.time()
        if stage != self.stage:
            if self.stage is not None:
                self.stage_seconds[self.stage] = round(
                    self.stage_seconds.get(self.stage, 0.0) + now - self._started, 2
                )
            self.stage, self._started = stage, now
        send("progress", stage=stage, fraction=round(min(1.0, fraction), 4), **fields)

    def sheetsage2(self, cap: int):
        def report(value: dict) -> None:
            stage = value.get("stage")
            windows = int(value.get("windows") or self.windows)
            self.windows = max(1, windows)
            share = WINDOWS / self.windows
            done = READING + share * (int(value.get("window") or 1) - 1)
            if stage == "encoding":
                self.enter("encoding", done, window=value.get("window"), windows=windows)
            elif stage == "decoding":
                tokens = int(value.get("tokens") or 0)
                within = ENCODE_SHARE + (1 - ENCODE_SHARE) * min(1.0, tokens / cap)
                self.enter("decoding", done + share * within, window=value.get("window"),
                           windows=windows, tokens=tokens)
            elif stage == "notation":
                self.enter("notation", READING + WINDOWS)
            else:
                self.check()

        return report

    def finish(self) -> dict:
        if self.stage is not None:
            self.stage_seconds[self.stage] = round(
                self.stage_seconds.get(self.stage, 0.0) + time.time() - self._started, 2
            )
        return self.stage_seconds


def _version_of(distribution: str):
    from importlib import metadata

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


class SheetSage2Engine:
    """SheetSage2 on its MERT-v2 encoder, loaded offline from the pulled files.

    SheetSage2's own loader (modeling_sheetsage2.from_pretrained) checks the encoder's
    hashes, loads both in float32 on the CPU and merges its LoRA adapters there before
    anything moves; only the merged model goes to the device. On CUDA it then computes
    under a bfloat16 autocast; everywhere else in float32 (its own rule,
    generation_sheetsage2.inference_autocast)."""

    name = "sheetsage2"

    def __init__(self, request: dict) -> None:
        import torch
        import transformers
        from transformers import AutoModel

        self._torch = torch
        self.model_dir = require(request, "model_dir", str)
        self.encoder_dir = require(request, "encoder_dir", str)
        self.device = require(request, "device", str)
        self.autocast = nullable(request, "autocast", str)
        torch.set_num_threads(min(MAX_THREADS, torch.get_num_threads()))
        if self.device == "cuda":
            cap_memory(torch, nullable(request, "memory_cap_bytes", int))
        model = AutoModel.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            local_files_only=True,
            base_model_path=self.encoder_dir,
        )
        self._model = model.eval().to(self.device)
        self.cap = int(self._model.max_output_seq_len)
        self.versions = {
            "torch": torch.__version__,
            "torchaudio": _version_of("torchaudio"),
            "transformers": transformers.__version__,
            "numpy": _version_of("numpy"),
        }

    @property
    def dtype(self) -> str:
        """SheetSage2's own `dtype` argument: bf16 is an autocast, CUDA's alone."""
        return "bf16" if self.device == "cuda" and self.autocast == "bfloat16" else "fp32"

    def transcribe(self, waveform, job: Job, progress: Progress) -> dict:
        try:
            return self._model.transcribe(
                waveform,
                output_dir=job.output_dir,
                sampling_rate=SAMPLE_RATE,
                prompts=list(job.prompts),
                dtype=self.dtype,
                melody_only=not job.keep_harmony,
                progress=progress.sheetsage2(self.cap),
            )
        except RuntimeError as exc:
            # A melody-only run with no ABC raises with the finished transcription on it
            # (pipeline_sheetsage2.transcribe); the job reports SheetSage2's reason.
            result = getattr(exc, "result", None)
            if result is None:
                raise
            return result

    def peak_bytes(self) -> int:
        torch = self._torch
        if self.device == "cuda":
            return int(torch.cuda.max_memory_reserved(0))
        if self.device == "mps":
            return int(torch.mps.driver_allocated_memory())
        return 0

    def release(self) -> None:
        gc.collect()
        torch = self._torch
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
        elif self.device == "mps":
            torch.mps.empty_cache()


ENGINES = {SheetSage2Engine.name: SheetSage2Engine}


def read_waveform(job: Job):
    import numpy

    samples = numpy.fromfile(job.audio_path, dtype="<f4")
    if samples.size != job.samples:
        raise RuntimeError(
            f"{job.audio_path} holds {samples.size} samples and the job decoded "
            f"{job.samples}; the file changed between the decode and the worker"
        )
    return samples


def windows_of(result: dict, cap: int) -> list[dict]:
    """SheetSage2's window records, with whether each ran into its token limit by the
    test SheetSage2's own warning uses (more tokens than `max_output_seq_len`)."""
    rows = []
    for index, window in enumerate(result.get("windows") or []):
        tokens = int(window.get("tokens") or 0)
        rows.append({
            "window": index + 1,
            "start_s": window.get("start"),
            "end_s": window.get("end"),
            "accept_start_s": window.get("accept_start"),
            "accept_end_s": window.get("accept_end"),
            "tokens": tokens,
            "prefix_tokens": window.get("prefix_tokens"),
            "token_cap": cap,
            "reached_token_limit": tokens > cap,
            "events": window.get("events"),
            "accepted_events": window.get("accepted_events"),
            "seconds": None if window.get("elapsed_seconds") is None
            else round(float(window["elapsed_seconds"]), 2),
        })
    return rows


def measure(text):
    """A score's nominal length (scorelength, the reader the audio job checks a song's
    length with) and its size."""
    if text is None:
        return None
    reading: dict = {"chars": len(text), "yue2_tokens": None, "yue2_tokens_note": YUE2_TOKENS_NOTE}
    try:
        reading.update(scorelength.read(text).to_dict())
    except scorelength.ScoreLengthError as exc:
        reading.update({"seconds": None, "seconds_unread": str(exc)})
    return reading


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _sheetsage2_record(result: dict) -> dict:
    return {
        "duration_seconds": result.get("duration_seconds"),
        "prompts": result.get("prompts"),
        "melody_only": result.get("melody_only"),
        "preset": result.get("preset"),
        "dtype": result.get("dtype"),
        "window_seconds": result.get("window_seconds"),
        "overlap_seconds": result.get("overlap_seconds"),
        "lookahead_seconds": result.get("lookahead_seconds"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "peak_gpu_mib": result.get("peak_gpu_mib"),
        "warnings": list(result.get("warnings") or []),
        "diagnostics": list(result.get("diagnostics") or []),
        "abc_error": result.get("abc_error"),
        "melody_notes": result.get("melody_notes"),
        "vocal_notes": result.get("vocal_notes"),
        "instrumental_notes": result.get("instrumental_notes"),
        "abc_measures": result.get("abc_measures"),
        "events": result.get("num_events"),
    }


def _run(engine, job: Job) -> dict:
    progress = Progress(job.request_id)
    started = time.time()
    engine.release()
    progress.enter("reading", 0.0)
    waveform = read_waveform(job)
    os.makedirs(job.output_dir, exist_ok=True)
    result = engine.transcribe(waveform, job, progress)
    peaks = {"transcribing": engine.peak_bytes()}
    progress.enter("normalising", READING + WINDOWS + NOTATION / 2)
    raw = result.get("abc") or None
    wrote = {"raw": False, "midi": False, "score": False, "instrumental": False}
    if raw is not None:
        _write(job.raw_path, raw)
        wrote["raw"] = True
    midi = os.path.join(job.output_dir, SHEETSAGE2_MIDI)
    if os.path.isfile(midi):
        shutil.copyfile(midi, job.midi_path)
        wrote["midi"] = True
    native = None
    if raw is not None:
        native = yue2native.to_yue2(raw, keep_harmony=job.keep_harmony)
        if native.error is None:
            _write(job.score_path, native.score)
            _write(job.instrumental_path, native.instrumental)
            wrote["score"] = wrote["instrumental"] = True
    engine.release()
    record = _sheetsage2_record(result)
    # Only what is published is measured: a score that failed a check is not one.
    made = native if native is not None and native.error is None else None
    return {
        "wrote": wrote,
        "sheetsage2": record,
        "windows": windows_of(result, engine.cap),
        "token_cap": engine.cap,
        "yue2_native": None if native is None else native.to_dict(),
        "lengths": {
            "sheetsage2": measure(raw),
            "score": measure(None if made is None else made.score),
            "instrumental": measure(None if made is None else made.instrumental),
        },
        "dtype": engine.dtype,
        "seconds": round(time.time() - started, 2),
        "stage_seconds": progress.finish(),
        "stage_peak_bytes": peaks,
        "peak_bytes": max(peaks.values()) if peaks else None,
        "versions": engine.versions,
    }


def load(request: dict) -> None:
    name = require(request, "engine", str)
    engine_class = ENGINES.get(name)
    if engine_class is None:
        raise RuntimeError(f"no score engine {name!r}; this worker runs {sorted(ENGINES)}")
    started = time.time()
    try:
        engine = engine_class(request)
    except ImportError as exc:
        raise RuntimeError(
            f"the score env in {sys.executable} cannot import what it needs ({exc}). "
            "Build it with `crucible install score`."
        ) from None
    _STATE["engine"] = engine
    send(
        "ready",
        seconds=round(time.time() - started, 2),
        engine=name,
        device=engine.device,
        versions=engine.versions,
    )
    send("done")


def score(request: dict) -> None:
    engine = _STATE["engine"]
    if engine is None:
        raise RuntimeError(
            "a score request arrived before a load request; the session's first "
            "exchange loads the engine"
        )
    job = Job(request)
    send("ready", request_id=job.request_id)
    try:
        result = _run(engine, job)
    except Cancelled as stopped:
        send("progress", stage="cancelled", during=stopped.stage)
        send("done")
        return
    finally:
        CANCEL.clear(job.request_id)
    send("result", **json.loads(json.dumps(result, default=str)))
    send("done")


OPS = {"load": load, "score": score}

INTERRUPTS = {"cancel": lambda request: CANCEL.ask(request.get("request_id"))}


def main() -> int:
    return workerio.serve(LABEL, OPS, INTERRUPTS)


if __name__ == "__main__":
    sys.exit(main())
