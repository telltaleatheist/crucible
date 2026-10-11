"""The recording a score job reads: decoded here, on the server, before any model loads.

SheetSage2 decodes a file itself (an ffmpeg subprocess to mono 24 kHz float32), and that
would put the first sight of a bad input after the card has been cleared for it: a resident
YuE2 evicted for a file that turns out to be a PDF. So the job decodes the one input with
Crucible's own ffmpeg (hosttools) into exactly what SheetSage2's decoder makes - mono,
24 kHz, little-endian float32 - and hands the worker the samples, which SheetSage2 takes as
a waveform without resampling. Every refusal about the recording is said here, by name.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from ...errors import JobError
from .params import ScoreParams

# SheetSage2's own rate (audio_sheetsage2.SAMPLE_RATE) and its decoder's format.
SAMPLE_RATE = 24_000

SAMPLE_BYTES = 4

# The fewest samples SheetSage2 transcribes (audio_sheetsage2.load_audio refuses fewer).
MIN_SAMPLES = 1025

DECODED_NAME = "recording.f32"

DECODE_TIMEOUT_SECONDS = 600

# How far past the ceiling a whole recording is read: far enough to know it is longer
# without decoding all of a three-hour file to say so.
OVERRUN_SECONDS = 1.0

STDERR_TAIL = 800


@dataclass(frozen=True)
class Recording:
    source: Path
    path: Path
    samples: int
    start_s: float
    max_seconds: float | None

    @property
    def seconds(self) -> float:
        return self.samples / SAMPLE_RATE

    def to_dict(self) -> dict[str, object]:
        return {
            "input": self.source.name,
            "start_s": self.start_s,
            "max_seconds": self.max_seconds,
            "seconds": round(self.seconds, 3),
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
        }


def the_one_input(inputs: dict[str, Path]) -> Path:
    if len(inputs) != 1:
        raise JobError(
            "invalid_inputs",
            f"a score job reads exactly one recording and this one carries "
            f"{len(inputs)} input(s) ({sorted(inputs)}); send the song, in any format "
            "ffmpeg decodes, as the job's one input",
        )
    return next(iter(inputs.values()))


def decode_command(
    ffmpeg: str, source: Path, target: Path, *, start_s: float, read_s: float
) -> list[str]:
    command = [ffmpeg, "-nostdin", "-v", "error", "-y"]
    if start_s > 0:
        command += ["-ss", f"{start_s:.3f}"]
    return command + [
        "-i", str(source),
        "-map", "0:a:0",
        "-vn",
        "-t", f"{read_s:.3f}",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "f32le",
        str(target),
    ]


def decode(
    ffmpeg: str,
    source: Path,
    scratch: Path,
    params: ScoreParams,
    *,
    ceiling_s: int,
    model_id: str,
) -> Recording:
    """Decode `source` from `start_s` for `max_seconds` (or to its end), and refuse a
    recording ffmpeg cannot read, one with too little audio, or one past the ceiling."""
    target = scratch / DECODED_NAME
    read_s = params.max_seconds if params.max_seconds is not None else ceiling_s + OVERRUN_SECONDS
    command = decode_command(ffmpeg, source, target, start_s=params.start_s, read_s=read_s)
    try:
        completed = subprocess.run(
            command, capture_output=True, timeout=DECODE_TIMEOUT_SECONDS, check=False
        )
    except subprocess.TimeoutExpired:
        raise JobError(
            "audio_undecodable",
            f"ffmpeg did not finish decoding {source.name!r} in "
            f"{DECODE_TIMEOUT_SECONDS} s; a recording of at most {ceiling_s} s decodes "
            "in seconds, so the file is damaged or is not audio",
        ) from None
    except OSError as exc:
        raise JobError(
            "audio_undecodable",
            f"could not run ffmpeg ({ffmpeg}) to decode {source.name!r}: {exc}",
        ) from None
    if completed.returncode != 0:
        said = completed.stderr.decode("utf-8", "replace").strip()[-STDERR_TAIL:]
        raise JobError(
            "audio_undecodable",
            f"ffmpeg could not decode {source.name!r} (exit {completed.returncode}): "
            f"{said or 'it said nothing'}. Send a song in a format ffmpeg reads (WAV, "
            "FLAC, MP3, M4A, OGG, a video's audio track, ...)",
        )
    size = target.stat().st_size if target.is_file() else 0
    samples = size // SAMPLE_BYTES
    where = f" from {params.start_s:g} s" if params.start_s > 0 else ""
    if samples < MIN_SAMPLES:
        raise JobError(
            "audio_too_short",
            f"{source.name!r} decodes to {samples / SAMPLE_RATE:.3f} s of audio{where}; "
            f"SheetSage2 needs at least {MIN_SAMPLES} samples at {SAMPLE_RATE} Hz, and "
            "a song to transcribe needs far more"
            + (". start_s may be past the recording's end" if params.start_s > 0 else ""),
        )
    if params.max_seconds is None and samples > ceiling_s * SAMPLE_RATE:
        raise JobError(
            "audio_too_long",
            f"{source.name!r} runs past {ceiling_s} s{where}, {model_id}'s ceiling: a "
            f"score is for YuE2, which composes at most {ceiling_s} s. Send `start_s` "
            "and `max_seconds` to transcribe the part you want",
        )
    return Recording(
        source=source,
        path=target,
        samples=samples,
        start_s=params.start_s,
        max_seconds=params.max_seconds,
    )


__all__ = [
    "DECODED_NAME",
    "MIN_SAMPLES",
    "Recording",
    "SAMPLE_RATE",
    "decode",
    "decode_command",
    "the_one_input",
]
