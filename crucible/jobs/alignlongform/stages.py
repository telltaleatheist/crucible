from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable

from ... import workers
from ...alignmodels import AlignBackendSpec
from ..align import start_aligner_session
from ..asr import OVERLAP_SECONDS, WINDOW_SECONDS

ROUGH_DEVICE = "cuda"
ROUGH_COMPUTE_TYPE = "float16"


class StageFailed(Exception):

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def probe_duration(ffmpeg: str, audio: Path) -> float:
    probe = str(Path(ffmpeg).with_name(Path(ffmpeg).name.replace("ffmpeg", "ffprobe")))
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
            capture_output=True, text=True, timeout=120, check=True,
        ).stdout.strip()
        return float(out)
    except Exception as exc:
        raise StageFailed(
            "audio_unreadable",
            f"could not read the duration of {audio.name} with {probe}: {exc}. The job cannot "
            "place cues in audio it cannot measure.",
        ) from None


def transcribe(
    *,
    python: Path,
    weights_dir: Path,
    ffmpeg: str,
    audio: Path,
    language: str,
    log_path: Path,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[tuple[str, float]]:
    request = {
        "model_dir": str(weights_dir),
        "ffmpeg": ffmpeg,
        "audio": str(audio),
        "language": language,
        "vad_filter": False,
        "word_timestamps": True,
        "initial_prompt": None,
        "device": ROUGH_DEVICE,
        "compute_type": ROUGH_COMPUTE_TYPE,
        "window_s": WINDOW_SECONDS,
        "overlap_s": OVERLAP_SECONDS,
        "speech": None,
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
            environment=workers.worker_environment(python.parent.parent),
        )
    except workers.WorkerError as exc:
        raise StageFailed("transcribe_failed", str(exc)) from None
    if outcome.ready.get("kept") is not None:
        raise StageFailed(
            "transcribe_failed",
            "the rough transcript came back cut to speech only, and this stage "
            "reads times on the book's own timeline",
        )

    words: list[tuple[str, float]] = []
    for index, result in enumerate(outcome.results):
        if result.get("error"):
            raise StageFailed(
                "transcribe_window_failed",
                f"window {index} of the rough transcript failed ({result['error']}). The "
                "alignment is not attempted on a transcript with holes: a wordless stretch "
                "reads as text the narrator never spoke, and those sentences would be "
                "dropped from the VTT rather than placed.",
            )
        offset = index * float(WINDOW_SECONDS)
        for segment in result.get("segments") or []:
            for word in segment.get("words") or []:
                words.append((str(word["word"]), float(word["start"]) + offset))
    return words


def slice_chunk(ffmpeg: str, audio: Path, start: float, end: float, out: Path) -> None:
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
    python: Path,
    weights_dir: Path,
    ffmpeg: str,
    language_name: str,
    chunk_files: list[Path],
    chunk_texts: list[str],
    max_audio_s: float,
    log_path: Path,
    spec: AlignBackendSpec,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    try:
        session = start_aligner_session(
            python, weights_dir, spec, log_path, ready_silence_timeout=900.0
        )
    except workers.WorkerError as exc:
        raise StageFailed("align_failed", str(exc)) from None
    try:
        outcome = session.send(
            {
                "op": "align",
                "language": language_name,
                "max_audio_s": max_audio_s,
                "ffmpeg": ffmpeg,
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
        session.stop()
    return list(outcome.results)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
