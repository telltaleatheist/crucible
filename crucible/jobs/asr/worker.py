"""The `asr` worker: faster-whisper, run in its own interpreter.

PHASE4-AUDIO.md section 0. This module is **standalone**. It imports the standard
library, `numpy` and `faster_whisper`, and nothing from `crucible` — the env it
runs in has no `crucible` installed and never will, so an import from the server
package here would be an ImportError at the first real job and a green test suite
right up until then. Crucible runs it as `<env python> <this file>`, hands it one
JSON object on stdin, and reads newline-delimited JSON back.

The wire, in full
-----------------
    stdin   one object: {model_dir, ffmpeg, audio, language, vad_filter,
                         word_timestamps, device, compute_type, window_s,
                         overlap_s}
            `language` is null for auto-detect. Every other key is required and
            an absent one is refused by name — there is no default for anything
            here, because every one of these values changes the transcript.

    fd 1    {"type": "progress", "stage": "decoding"|"transcribing",
             "processed_s", "total_s", "cues"}
            {"type": "ready", "duration_s", "windows", "device", "compute_type"}
            {"type": "result", "segments": [...], "language", ...}   one per window
            {"type": "result", "error": "..."}                       a failed window
            {"type": "failed", "message"}                            the whole run
            {"type": "done"}

Results carry **window-relative** timestamps and no index. The server knows a
result's window from its position in the stream and shifts the timings itself
(`crucible/jobs/asr/__init__.py`), because an index a worker reports is an index
a worker can get wrong — narrator's aligner proved that on a 401-chunk book.

fd 1 is results and nothing else
--------------------------------
The first thing this file does, before importing anything that could print, is
dup fd 1 somewhere safe and point the original at stderr. faster-whisper and
CTranslate2 both log, and a library's logger writing to stdout is exactly what
corrupted narrator's aligner stream. After the dup, a stray `print` lands in the
job's log where it belongs.

Why ffmpeg and not faster-whisper's own decoder
-----------------------------------------------
`faster_whisper.decode_audio` is PyAV, and PyAV's demuxer silently TRUNCATES some
assembled m4b files: a discontinuity a few hours in ends the decode early with no
error at all. One real 18-hour book decoded to six hours, and the transcript
stopped dead mid-book while the windowing loop believed it was finished
(BookForge, `electron/scripts/transcribe_audiobook.py`). ffmpeg reads those same
files in full. `-f f32le` is already normalised to [-1, 1], so there is no
rescale and no second copy.

Why windows at all
------------------
Handing `model.transcribe()` an 18-hour file makes the feature extractor frame
the whole signal into one array — about (1, 6.5M, 400) float64, roughly 19 GiB —
which OOMs. Decoding once and transcribing in 900-second windows keeps peak
memory independent of book length. The numbers are the server's, not the
client's, and they arrive in the request.
"""

from __future__ import annotations

import os
import sys

# ---- fd 1 is results, stderr is everything else. Before any other import. ----
_RESULTS_FD = os.dup(1)
os.dup2(2, 1)
_RESULTS = os.fdopen(_RESULTS_FD, "w", encoding="utf-8", buffering=1)

import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

#: faster-whisper works at 16 kHz mono, always. Not a parameter.
SAMPLE_RATE = 16_000

#: How often the decode phase reports, in seconds of wall clock. An 18-hour book
#: is minutes of CPU-only decode before the first transcription line, and a bar
#: that does not move during it looks like a hang.
DECODE_REPORT_SECONDS = 1.0

#: Report transcription progress when the fraction has moved this far, or when
#: this many seconds of wall clock have passed, whichever comes first. Both
#: numbers are BookForge's, measured against real books.
PROGRESS_FRACTION_STEP = 0.002
PROGRESS_WALL_SECONDS = 1.5


def send(message_type: str, **fields: object) -> None:
    """One JSON object, one line, flushed, on the real fd 1."""
    _RESULTS.write(json.dumps({"type": message_type, **fields}) + "\n")
    _RESULTS.flush()


def fail(message: str) -> int:
    send("failed", message=message)
    return 1


def require(request: dict, key: str, kind: type) -> object:
    """One required key, or a refusal naming it and the file it came from.

    Nothing in this request has a default. `language` may be null, and that null
    means auto-detect — a value, not an absence.
    """
    if key not in request:
        raise KeyError(
            f"the asr request has no {key!r}; every parameter is required because "
            "every one of them changes the transcript"
        )
    value = request[key]
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise KeyError(
            f"the asr request's {key!r} must be {kind.__name__}, got "
            f"{type(value).__name__}"
        )
    return value


# ------------------------------------------------------------------- decoding


def probe_duration(ffmpeg: str, audio_path: str) -> float:
    """Container duration in seconds, via the ffprobe beside ffmpeg.

    This is the denominator for decode progress and nothing else — the decode
    itself never depends on it, and the transcription phase uses the decoded
    sample count, which is exact. A container that carries no duration gets 0.0
    and a progress line with no percentage, which is still honest.
    """
    directory = os.path.dirname(ffmpeg)
    base = "ffprobe" + (".exe" if ffmpeg.lower().endswith(".exe") else "")
    ffprobe = os.path.join(directory, base) if directory else base
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                audio_path,
            ],
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[asr] ffprobe unavailable ({exc}); decode progress has no total")
        return 0.0
    if completed.returncode != 0:
        print(f"[asr] ffprobe exited {completed.returncode}; decode progress has no total")
        return 0.0
    text = completed.stdout.decode("utf-8", "replace").strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        print(f"[asr] ffprobe said {text!r}, which is not a duration")
        return 0.0


def decode(ffmpeg: str, audio_path: str, on_progress) -> "object":
    """Decode to a mono float32 waveform at 16 kHz. Raises on any ffmpeg failure.

    Streamed rather than `subprocess.run` so progress can fire as bytes arrive;
    one audio second is `SAMPLE_RATE * 4` bytes of f32le, so the position is
    exact by construction and no ffmpeg stats are parsed. stderr is drained on a
    thread so an error-spewing decode cannot fill the pipe and deadlock.
    """
    import numpy

    process = subprocess.Popen(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            audio_path,
            "-map",
            "0:a:0",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-f",
            "f32le",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    errors: list[bytes] = []

    def drain() -> None:
        for chunk in iter(lambda: process.stderr.read(65536), b""):
            errors.append(chunk)

    pump = threading.Thread(target=drain, daemon=True)
    pump.start()

    buffer = bytearray()
    per_second = SAMPLE_RATE * 4
    while True:
        chunk = process.stdout.read(1 << 20)
        if not chunk:
            break
        buffer += chunk
        on_progress(len(buffer) / per_second)
    process.stdout.close()
    code = process.wait()
    pump.join(timeout=5)
    if code != 0:
        tail = b"".join(errors).decode("utf-8", "replace").strip()[-500:]
        raise RuntimeError(f"ffmpeg exited {code}: {tail}")
    # A view of the buffer, not a copy: the array is gigabytes on a long book.
    return numpy.frombuffer(buffer, dtype=numpy.float32)


# -------------------------------------------------------------- transcription


def serialise(segment, want_words: bool) -> dict:
    """One faster-whisper segment, window-relative, as plain JSON.

    Only the fields a client can use: the text, its span, and the per-word
    timings when they were asked for. Whisper's own diagnostics
    (`avg_logprob`, `no_speech_prob`, the token ids) are deliberately not
    forwarded — they are the engine's, and a field Crucible publishes is a field
    Crucible has to keep publishing.
    """
    row: dict = {
        "start": float(segment.start),
        "end": float(segment.end),
        "text": str(segment.text),
    }
    if not want_words:
        return row
    words = getattr(segment, "words", None)
    # None is what faster-whisper gives for a segment it found no words in; an
    # empty list says the same thing without making the client check two shapes.
    row["words"] = [
        {
            "start": float(word.start),
            "end": float(word.end),
            "word": str(word.word),
            "probability": float(word.probability),
        }
        for word in (words or [])
    ]
    return row


def main() -> int:
    line = sys.stdin.readline()
    if not line.strip():
        return fail("the asr worker was given no request on stdin")
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        return fail(f"the asr request is not JSON: {exc}")
    if not isinstance(request, dict):
        return fail(f"the asr request must be a JSON object, got {type(request).__name__}")

    try:
        model_dir = require(request, "model_dir", str)
        ffmpeg = require(request, "ffmpeg", str)
        audio = require(request, "audio", str)
        device = require(request, "device", str)
        compute_type = require(request, "compute_type", str)
        vad_filter = require(request, "vad_filter", bool)
        word_timestamps = require(request, "word_timestamps", bool)
        window_s = require(request, "window_s", int)
        overlap_s = require(request, "overlap_s", int)
        if "language" not in request:
            raise KeyError(
                "the asr request has no 'language'; null means auto-detect, which "
                "is a choice and has to be made explicitly"
            )
        language = request["language"]
        if language is not None and not isinstance(language, str):
            raise KeyError(
                f"the asr request's 'language' must be a string or null, got "
                f"{type(language).__name__}"
            )
    except KeyError as exc:
        return fail(str(exc.args[0]))

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        return fail(
            f"faster-whisper is not importable in {sys.executable}: {exc}. "
            "The asr env is installed with `crucible install asr`."
        )

    # No CPU fallback. BookForge falls back to CPU int8 once when a CUDA load
    # fails (`transcribe-bridge.ts`); Crucible does not and will not. There is no
    # CPU backend here, and a transcript that quietly ran at int8 on a CPU is a
    # different transcript with nothing in the output to say so
    # (PHASE4-AUDIO.md section 3).
    try:
        model = WhisperModel(model_dir, device=device, compute_type=compute_type)
    except Exception as exc:
        return fail(
            f"could not load {model_dir} on {device} at {compute_type}: "
            f"{type(exc).__name__}: {exc}"
        )

    total_container = probe_duration(ffmpeg, audio)
    last_decode = [0.0]

    def decode_progress(decoded_seconds: float) -> None:
        now = time.time()
        if now - last_decode[0] < DECODE_REPORT_SECONDS:
            return
        last_decode[0] = now
        send(
            "progress",
            stage="decoding",
            processed_s=round(decoded_seconds, 1),
            total_s=round(total_container, 1),
            cues=0,
        )

    try:
        waveform = decode(ffmpeg, audio, decode_progress)
    except Exception as exc:
        return fail(f"could not decode {audio}: {type(exc).__name__}: {exc}")

    total = len(waveform) / float(SAMPLE_RATE)
    if total <= 0:
        return fail(f"{audio} decoded to zero length")

    windows = int(math.ceil(total / window_s))
    send(
        "ready",
        duration_s=total,
        windows=windows,
        device=device,
        compute_type=compute_type,
    )

    emitted = 0
    last_fraction = [-1.0]
    last_wall = [time.time()]

    def transcribe_progress(processed: float) -> None:
        fraction = min(1.0, processed / total)
        now = time.time()
        if (
            fraction - last_fraction[0] < PROGRESS_FRACTION_STEP
            and now - last_wall[0] < PROGRESS_WALL_SECONDS
        ):
            return
        last_fraction[0] = fraction
        last_wall[0] = now
        send(
            "progress",
            stage="transcribing",
            processed_s=round(processed, 1),
            total_s=round(total, 1),
            cues=emitted,
        )

    for index in range(windows):
        start = index * float(window_s)
        boundary = min(start + window_s, total)
        # The window is extended `overlap_s` PAST its own boundary so a sentence
        # straddling the cut is still spoken in full inside it. The next window
        # starts at the boundary, so consecutive windows share those seconds and
        # the duplicate cues they produce are dropped by the server.
        first = int(start * SAMPLE_RATE)
        last = int(min(boundary + overlap_s, total) * SAMPLE_RATE)
        try:
            segments, info = model.transcribe(
                waveform[first:last],
                language=language,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
            )
            rows = []
            for segment in segments:
                rows.append(serialise(segment, word_timestamps))
                emitted += 1
                transcribe_progress(min(total, start + float(segment.end)))
        except Exception as exc:
            # One window's failure is reported and the run continues, so a single
            # bad stretch does not cost the other seventeen hours — and so the
            # server learns about every failure in one run rather than one per
            # re-run. What the server does with a hole is the server's ruling.
            send("result", error=f"{type(exc).__name__}: {exc}")
            continue
        send(
            "result",
            segments=rows,
            language=info.language,
            language_probability=float(info.language_probability),
        )

    send(
        "progress",
        stage="transcribing",
        processed_s=round(total, 1),
        total_s=round(total, 1),
        cues=emitted,
    )
    send("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
