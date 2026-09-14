"""The `asr` worker on `mlx-darwin`: mlx-whisper, run in its own interpreter.

The SECOND worker behind one job type, beside `worker.py`'s faster-whisper. It
is a second worker and not a branch inside the first because the two run in
different envs on different machines and share no import: `worker.py` imports
`faster_whisper` (CTranslate2) and this one imports `mlx_whisper` (MLX), and
neither library exists in the other's env.

**The wire is byte-for-byte `worker.py`'s.** Same request object on stdin, same
newline-delimited JSON out, same `ready` / `progress` / `result` / `failed` /
`done` message types, same window-relative timestamps with no index. That is the
point: `crucible/jobs/asr/__init__.py` assembles `transcript.json` from these
messages with no branch in it for the engine, so BookForge's align and transcript
readers see one artifact shape whichever machine ran the job. Anything this
engine cannot do is a REFUSAL, never a quietly different field.

This module is **standalone**, for `worker.py`'s reason: the env it runs in has
no `crucible` installed and never will.

The three places mlx-whisper is not faster-whisper
---------------------------------------------------
1. **There is no VAD.** faster-whisper ships Silero; mlx-whisper has nothing of
   the kind — `no_speech_threshold` skips a 30-second segment the model itself
   thinks is silent, which is a different mechanism on different evidence. So a
   request with `vad_filter: true` is REFUSED here by name. Transcribing without
   the filter the caller asked for would be a different transcript with nothing
   in the output to say so, which is the same argument PHASE4-AUDIO.md section 3
   makes against the CPU fallback.

2. **`transcribe()` does not report a language probability.** faster-whisper's
   `info.language_probability` has no counterpart in the returned dict. It is
   not invented and it is not dropped: this worker runs whisper's OWN language
   detection first — `model.detect_language()` on the window's first 30 seconds,
   exactly what `mlx_whisper.transcribe` does internally when no language is
   given — takes the best code and ITS probability, and then passes that code to
   `transcribe()` so the detection is not run twice. Measured on the M1 Ultra,
   2026-09-14: `{"detected": "en", "probability": 0.9946824908256531}`, which is
   the same shape and the same meaning as faster-whisper's field. When the
   caller NAMED a language there is nothing to detect, and the probability is
   1.0 because the caller asserted it.

3. **The segments carry more than faster-whisper's.** mlx-whisper's rows have
   `seek`, `id`, `tokens`, `temperature`, `avg_logprob`, `compression_ratio` and
   `no_speech_prob` beside the three fields Crucible publishes. `serialise()`
   forwards the same three (plus `words`) and nothing else, for `worker.py`'s
   stated reason: a field Crucible publishes is a field Crucible has to keep
   publishing.

Why the window loop is still here
----------------------------------
mlx-whisper does its own 30-second sliding window internally, so it would not
OOM on a long array the way `faster_whisper.transcribe` does. The 900-second
windows stay anyway, because the ARTIFACT is windowed: `transcript.json` records
`window_s`, `overlap_s` and `windows`, the server shifts each window's
timestamps by its position, and a file produced on the Mac must be the same
document as one produced on the PC. One engine windowing and the other not would
be a difference a reader could see.
"""

from __future__ import annotations

import os
import sys

# ---- fd 1 is results, stderr is everything else. Before any other import. ----
# mlx-whisper prints a tqdm bar and its language-detection notice; both would
# land in the middle of a JSON line otherwise.
_RESULTS_FD = os.dup(1)
os.dup2(2, 1)
_RESULTS = os.fdopen(_RESULTS_FD, "w", encoding="utf-8", buffering=1)

import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

#: whisper works at 16 kHz mono, always. Not a parameter. The same constant
#: `worker.py` declares, because it is the same model family's requirement.
SAMPLE_RATE = 16_000

#: The window mlx-whisper's own decoder slides, in mel frames and in seconds.
#: Used only for the language-detection slice below.
DETECT_SECONDS = 30

DECODE_REPORT_SECONDS = 1.0
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
    """One required key, or a refusal naming it. Nothing here has a default."""
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

    `worker.py`'s function, verbatim in behaviour: the denominator for decode
    progress and nothing else. A container that carries no duration gets 0.0 and
    a progress line with no percentage, which is still honest.
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
        print(
            f"[asr] ffprobe exited {completed.returncode}; decode progress has no total"
        )
        return 0.0
    text = completed.stdout.decode("utf-8", "replace").strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        print(f"[asr] ffprobe said {text!r}, which is not a duration")
        return 0.0


def decode(ffmpeg: str, audio_path: str, on_progress) -> "object":
    """Decode to a mono float32 waveform at 16 kHz. Raises on any ffmpeg failure.

    The same decode as `worker.py`'s and for the same reason — PyAV silently
    truncates some assembled m4b files, which ends a transcript hours early with
    no error — so both engines read the same samples out of the same container.
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


def serialise(segment: dict, want_words: bool) -> dict:
    """One mlx-whisper segment, window-relative, as plain JSON.

    Exactly the three fields `worker.py.serialise` publishes, plus `words` in
    the same four-key shape (`word`, `start`, `end`, `probability`) — which is
    mlx-whisper's own shape, verified on the M1 Ultra on 2026-09-14:
    `{"word": " the", "start": 0.0, "end": 0.26, "probability": 0.0922...}`.
    """
    row: dict = {
        "start": float(segment["start"]),
        "end": float(segment["end"]),
        "text": str(segment["text"]),
    }
    if not want_words:
        return row
    # A segment mlx-whisper found no words in has no `words` key at all; an
    # empty list says the same thing without making the client check two shapes,
    # which is `worker.py`'s rule for the `None` its engine returns.
    row["words"] = [
        {
            "start": float(word["start"]),
            "end": float(word["end"]),
            "word": str(word["word"]),
            "probability": float(word["probability"]),
        }
        for word in segment.get("words") or []
    ]
    return row


def detect_language(model_dir: str, window, dtype) -> tuple[str, float]:
    """whisper's own language detection on the first 30 s, and its probability.

    This is what `mlx_whisper.transcribe` does internally when `language` is
    None, lifted out so the PROBABILITY can be reported — the engine's own
    function keeps the code and throws the number away. Running it here and then
    passing the code into `transcribe()` means the detection happens once, not
    twice.
    """
    import mlx.core as mx
    from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.transcribe import ModelHolder

    model = ModelHolder.get_model(model_dir, dtype)
    mel = pad_or_trim(
        log_mel_spectrogram(window[: DETECT_SECONDS * SAMPLE_RATE],
                            n_mels=model.dims.n_mels),
        N_FRAMES,
        axis=-2,
    ).astype(dtype)
    _, probabilities = model.detect_language(mel)
    # `detect_language` is typed `List[dict]` and returns a bare dict for a
    # single mel (measured 2026-09-14). Both shapes are accepted rather than
    # one of them being assumed, because the assumption is one release from
    # being a KeyError in the middle of a book.
    table = probabilities[0] if isinstance(probabilities, list) else probabilities
    best = max(table, key=table.get)
    return str(best), float(table[best])


def main() -> int:
    line = sys.stdin.readline()
    if not line.strip():
        return fail("the asr worker was given no request on stdin")
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        return fail(f"the asr request is not JSON: {exc}")
    if not isinstance(request, dict):
        return fail(
            f"the asr request must be a JSON object, got {type(request).__name__}"
        )

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

    if vad_filter:
        # The server refuses this before the job is queued
        # (`crucible/jobs/asr/__init__.py`), so reaching it here is a bug in
        # Crucible rather than a caller's mistake — and it is still a refusal,
        # because the one thing this worker must never do is produce a
        # transcript under rules the caller did not ask for.
        return fail(
            "this request asks for vad_filter and mlx-whisper has no VAD at all "
            "(faster-whisper's is Silero; mlx-whisper's no_speech_threshold is a "
            "per-segment judgement by the model itself, which is a different "
            "thing on different evidence). Transcribing without it would be a "
            "different transcript with nothing in the file to say so"
        )

    try:
        import mlx.core as mx
        import mlx_whisper
    except ImportError as exc:
        return fail(
            f"mlx-whisper is not importable in {sys.executable}: {exc}. "
            "The asr env is installed with `crucible install asr`."
        )

    # mlx-whisper's own default, stated rather than inherited: `transcribe`
    # reads `fp16` out of its decode options and picks `mx.float16` unless told
    # otherwise. The server sends `float16` on this backend for that reason, and
    # anything else is a refusal rather than a silent substitution.
    dtypes = {"float16": mx.float16, "float32": mx.float32}
    dtype = dtypes.get(compute_type)
    if dtype is None:
        return fail(
            f"compute_type {compute_type!r} is not one mlx-whisper runs; it takes "
            f"{sorted(dtypes)}"
        )
    if device != "metal":
        # MLX has one device and it is the Apple GPU. A request naming anything
        # else has come from a server that thinks this is a different engine.
        return fail(
            f"device {device!r} is not mlx-whisper's; MLX runs on Metal and "
            "nothing else, and the server sends 'metal' on this backend"
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
        # straddling the cut is still spoken in full inside it — `worker.py`'s
        # rule, and the server drops the duplicate cues either way.
        first = int(start * SAMPLE_RATE)
        last = int(min(boundary + overlap_s, total) * SAMPLE_RATE)
        window = waveform[first:last]
        try:
            if language is None:
                code, probability = detect_language(model_dir, window, dtype)
            else:
                # Nothing was detected, so nothing is reported as detected: the
                # caller asserted the language and 1.0 is that assertion, not a
                # measurement the model made.
                code, probability = language, 1.0
            result = mlx_whisper.transcribe(
                window,
                path_or_hf_repo=model_dir,
                language=code,
                word_timestamps=word_timestamps,
                fp16=dtype is mx.float16,
                verbose=None,
            )
            rows = []
            for segment in result["segments"]:
                rows.append(serialise(segment, word_timestamps))
                emitted += 1
                transcribe_progress(min(total, start + float(segment["end"])))
        except Exception as exc:
            # One window's failure is reported and the run continues, so a
            # single bad stretch does not cost the other seventeen hours.
            # What the server does with a hole is the server's ruling.
            send("result", error=f"{type(exc).__name__}: {exc}")
            continue
        send(
            "result",
            segments=rows,
            language=result["language"],
            language_probability=probability,
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
