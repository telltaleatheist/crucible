"""Driving the two existing workers, and slicing the audio between them.

`align-longform` is an ORCHESTRATION, not a new engine. Its four stages are

    transcribe   faster-whisper, the `asr` env's worker      (CPU-bound, most of the clock)
    coarse-align the DTW in `coarse.py`                      (pure, no card)
    align        Qwen3-ForcedAligner, the `align` env's worker (the card)
    write        snap + VTT in `cues.py`                     (pure, no card)

and two of those already exist as job types with their own envs. This module is
what lets one job reach both.

WHY THE WORKERS AND NOT NESTED JOBS
-----------------------------------
The obvious shape — submit an `asr` job, then an `align` job — DEADLOCKS. A
Crucible takes one job at a time (`CrucibleBusy`, 409 `server_busy`), so a job
that waits on another job waits forever on a lane it is itself holding. The
worker scripts are the reusable unit, not the job types, and `workers.run_worker`
is the door both already go through.

ONE CARD, ONE THING
-------------------
faster-whisper and the aligner are not on the card together. The transcribe stage
uses `run_worker`, which is start/send/stop in one call, so that worker has
exited before the align stage spawns — true by construction rather than by a
release somebody has to remember. The align stage needs a SESSION instead
(`WorkerSession`: load once, then the whole book down one stdin, because the
aligner is 1.7 GB and a book is hundreds of chunks), and its `stop` is in a
`finally` so the card comes back whether the book finished or not. The aligner is the only one of the two that Crucible
otherwise holds RESIDENT, and this job deliberately does not take that path: a
long-form align is one pass over one book, so the weights are read once either
way, and borrowing the resident holder would evict whatever a client had loaded.

THE ENVIRONMENT IS NOT OPTIONAL
-------------------------------
Both workers get `workerenv.worker_environment(...)`. The `asr` env needs it to
run at all — pip's CUDA libraries are in per-package directories the loader does
not search, and ctranslate2 resolves cuBLAS at the first matrix multiply, so
without it the model loads and the first window dies (measured on owens-pc,
2026-09-15; `crucible doctor` called the job type ready throughout).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable

from ... import workerenv, workers

#: The `asr` worker's window, IMPORTED from that job type rather than copied.
#:
#: This file first carried 600.0/5.0, invented here. The real values are 900/15
#: and they are INTS — the worker refuses a float by name ("the asr request's
#: 'window_s' must be int, got float"), which is how the wrong numbers were
#: caught on the first real run. Importing is what stops the pair drifting: a
#: copy agrees on the day it is written and silently stops agreeing later, and
#: two different ideas of how long a window is would put every word in the
#: second window at the wrong second.
from ..asr import OVERLAP_SECONDS, WINDOW_SECONDS  # noqa: E402

#: What the rough pass runs at. `float16` on the card, and the device is the
#: card: this stage is the long one and running it on CPU would dominate the job.
ROUGH_DEVICE = "cuda"
ROUGH_COMPUTE_TYPE = "float16"


class StageFailed(Exception):
    """A stage could not finish. Carries a code the job turns into its own."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def probe_duration(ffmpeg: str, audio: Path) -> float:
    """Seconds of audio, from ffprobe beside the ffmpeg we were given.

    Asked ONCE and threaded through, because three later decisions depend on it
    — the last chunk's end, the cap check, and the progress denominator — and a
    number re-derived three times is three chances to disagree.
    """
    probe = str(Path(ffmpeg).with_name(Path(ffmpeg).name.replace("ffmpeg", "ffprobe")))
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
            capture_output=True, text=True, timeout=120, check=True,
        ).stdout.strip()
        return float(out)
    except Exception as exc:  # noqa: BLE001 - reported by name below
        raise StageFailed(
            "audio_unreadable",
            f"could not read the duration of {audio.name} with {probe}: {exc}. The job cannot "
            "place cues in audio it cannot measure.",
        ) from None


def transcribe(
    *,
    home: Path,
    python: Path,
    weights_dir: Path,
    ffmpeg: str,
    audio: Path,
    language: str,
    log_path: Path,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[tuple[str, float]]:
    """Stage 1 — the rough transcript, as `(word, start)` in book order.

    `word_timestamps` is TRUE and not a parameter: the coarse stage aligns
    against word positions, so a transcript without them cannot be walked.
    `vad_filter` is FALSE for the same kind of reason — VAD drops audio it
    judges non-speech, and a dropped stretch is a hole the DTW would read as
    the narrator skipping text.
    """
    request = {
        "model_dir": str(weights_dir),
        "ffmpeg": ffmpeg,
        "audio": str(audio),
        "language": language,
        "vad_filter": False,
        "word_timestamps": True,
        "device": ROUGH_DEVICE,
        "compute_type": ROUGH_COMPUTE_TYPE,
        "window_s": WINDOW_SECONDS,
        "overlap_s": OVERLAP_SECONDS,
    }
    try:
        outcome = workers.run_worker(
            python=python,
            script=Path(__file__).resolve().parents[1] / "asr" / "worker.py",
            request=request,
            log_path=log_path,
            ready_silence_timeout=900.0,
            on_progress=on_progress,
            cancelled=cancelled,
            environment=workerenv.worker_environment(
                workerenv.worker_env_dir(home, "asr")
            ),
        )
    except workers.WorkerError as exc:
        raise StageFailed("transcribe_failed", str(exc)) from None

    words: list[tuple[str, float]] = []
    for index, result in enumerate(outcome.results):
        if result.get("error"):
            # A HOLE IS NOT A TRANSCRIPT. The coarse stage would read a wordless
            # stretch of real narration as text the narrator skipped, and drop
            # every sentence in it from the VTT.
            raise StageFailed(
                "transcribe_window_failed",
                f"window {index} of the rough transcript failed ({result['error']}). The "
                "alignment is not attempted on a transcript with holes: a wordless stretch "
                "reads as text the narrator never spoke, and those sentences would be "
                "dropped from the VTT rather than placed.",
            )
        # WINDOW-RELATIVE IN, BOOK-ABSOLUTE OUT, and the expression is the
        # `asr` job type's own (`jobs/asr/__init__.py`: `offset = index *
        # float(WINDOW_SECONDS)`) rather than a second derivation of it. Results
        # carry no index — position in the stream IS the window, which is that
        # worker's rule — so `enumerate` is the whole of the bookkeeping.
        #
        # Written as a carried variable at first, which computed the same numbers
        # and hid that it did. Only ever exercised on a ONE-WINDOW clip until
        # `test_align_longform_stages.py`, because a 15-second probe has no
        # second window to get wrong.
        offset = index * float(WINDOW_SECONDS)
        for segment in result.get("segments") or []:
            for word in segment.get("words") or []:
                words.append((str(word["word"]), float(word["start"]) + offset))
    return words


def slice_chunk(ffmpeg: str, audio: Path, start: float, end: float, out: Path) -> None:
    """Cut `[start, end)` out of the audio for one aligner window.

    16 kHz mono, which is what the aligner's feature extractor was trained at —
    the same decode `align`'s own worker does, done here because this job knows
    the spans and that worker takes files.
    """
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}",
         "-to", f"{end:.3f}", "-i", str(audio), "-ac", "1", "-ar", "16000", str(out)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0 or not out.is_file():
        raise StageFailed(
            "chunk_decode_failed",
            f"ffmpeg could not cut {start:.1f}s-{end:.1f}s out of {audio.name}: "
            f"{(result.stderr or '').strip()[:300]}",
        )


def align_chunks(
    *,
    home: Path,
    python: Path,
    weights_dir: Path,
    ffmpeg: str,
    language_name: str,
    chunk_files: list[Path],
    chunk_texts: list[str],
    max_audio_s: float,
    log_path: Path,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """Stage 3 — the aligner, one result per chunk, BY POSITION.

    No index travels in either direction, which is the `align` worker's own rule
    and not a simplification: an index a worker reports is an index a worker can
    get wrong, and narrator proved that on a 401-chunk book.
    """
    session = workers.WorkerSession(
        python=python,
        script=Path(__file__).resolve().parents[1] / "align" / "worker.py",
        log_path=log_path,
        environment=workerenv.worker_environment(
            workerenv.worker_env_dir(home, "align")
        ),
    )
    try:
        # LOAD AND ALIGN ON ONE PROCESS. The aligner is 1.7 GB of weights and a
        # book is hundreds of chunks, so the worker is started once and fed the
        # whole book on the same stdin — `align`'s own reason for having a
        # session rather than a one-shot worker.
        #
        # `start` takes no `cancelled` hook, deliberately and not by oversight:
        # a load is what the lane is waiting on, and a half-loaded model that was
        # interrupted is a process holding VRAM nothing is tracking.
        session.start(
            {"op": "load", "model_dir": str(weights_dir),
             "device": "cuda", "dtype": "bfloat16"},
            ready_silence_timeout=900.0,
        )
        outcome = session.send(
            {
                "op": "align",
                "language": language_name,
                "max_audio_s": max_audio_s,
                "ffmpeg": ffmpeg,
                # No index in a chunk and none in a result: position is the whole
                # identity, which is the align worker's own rule. An index a
                # worker reports is an index a worker can get wrong, and
                # narrator proved that on a 401-chunk book.
                "chunks": [
                    {"audio": str(path), "text": text}
                    for path, text in zip(chunk_files, chunk_texts)
                ],
            },
            ready_silence_timeout=900.0,
            on_progress=on_progress,
            cancelled=cancelled,
        )
    except workers.WorkerError as exc:
        raise StageFailed("align_failed", str(exc)) from None
    finally:
        # The card is handed back whether this finished or not. This job does not
        # use the resident holder (see the module header), so nothing else will
        # stop this process.
        session.stop()
    return list(outcome.results)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
