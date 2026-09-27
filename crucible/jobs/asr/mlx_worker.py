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
engine cannot do is a REFUSAL, never a quietly different field. That includes
`speech` (2026-09-27): the same `speechonly.py`, numpy on the CPU, the same
`samples` / `speech_s` / `kept` in `ready` (see `worker.py`).

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import workerio  # noqa: E402

sys.path.pop(0)

# mlx-whisper prints a tqdm bar and its language-detection notice; both would
# land in the middle of a JSON line otherwise.
workerio.claim_stdout()

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

from workerio import SAMPLE_RATE, decode, fail, probe_duration, send  # noqa: E402

speechonly = workerio.load_sibling("speechonly", __file__)

#: The window mlx-whisper's own decoder slides, in mel frames and in seconds.
#: Used only for the language-detection slice below.
DETECT_SECONDS = 30

PROGRESS_FRACTION_STEP = 0.002
PROGRESS_WALL_SECONDS = 1.5


def require(request: dict, key: str, kind: type) -> object:
    """One required key, or a refusal naming it. Nothing here has a default."""
    return workerio.require(
        request,
        key,
        kind,
        "asr",
        "every parameter is required because every one of them changes the "
        "transcript",
    )


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


def require_prompt(request: dict) -> "str | None":
    """`initial_prompt`: required as a KEY, null means no prompt.

    `worker.py`'s rule, verbatim in behaviour. mlx-whisper's `transcribe` calls
    `initial_prompt.strip()`, so anything but a string or null would be an
    AttributeError in the first window rather than a refusal before the first.
    """
    if "initial_prompt" not in request:
        raise KeyError(
            "the asr request has no 'initial_prompt'; null means no prompt, and "
            "the server sends the key either way"
        )
    value = request["initial_prompt"]
    if value is None:
        return None
    if not isinstance(value, str):
        raise KeyError(
            f"the asr request's 'initial_prompt' must be a string or null, got "
            f"{type(value).__name__}"
        )
    if value.strip() == "":
        raise KeyError("the asr request's 'initial_prompt' is blank; null means none")
    return value


def check_prompt_length(model_dir: str, dtype, prompt: str) -> "str | None":
    """A refusal if the prompt is longer than whisper keeps, else None.

    mlx-whisper's own arithmetic: `transcribe` builds its tokenizer with
    `get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
    ...)` and encodes `" " + initial_prompt.strip()`; `DecodingTask`'s
    `_get_initial_tokens` then keeps `prompt_tokens[-(n_ctx // 2 - 1):]` with
    `n_ctx = model.dims.n_text_ctx`. A longer prompt loses its BEGINNING with no
    error. The language and task only choose special tokens, which `encode`
    never emits, so they are left at the library's own defaults here.
    """
    from mlx_whisper.tokenizer import get_tokenizer
    from mlx_whisper.transcribe import ModelHolder

    # The same cached holder `transcribe()` loads through, so this is the one
    # load of the run and not a second one.
    model = ModelHolder.get_model(model_dir, dtype)
    tokenizer = get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages
    )
    ceiling = int(model.dims.n_text_ctx) // 2 - 1
    count = len(tokenizer.encode(" " + prompt.strip()))
    if count <= ceiling:
        return None
    return (
        f"initial_prompt is {count} tokens and whisper keeps only the last "
        f"{ceiling} of its prompt history, so the first {count - ceiling} would "
        "be dropped without a word. Send a shorter prompt: the title and the "
        "names in it, not the text"
    )


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
        initial_prompt = require_prompt(request)
        speech = speechonly.from_request(request)
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

    if initial_prompt is not None:
        failure = check_prompt_length(model_dir, dtype, initial_prompt)
        if failure is not None:
            return fail(failure)

    total_container = probe_duration(ffmpeg, audio, "asr")
    try:
        waveform = decode(
            ffmpeg,
            audio,
            workerio.decode_reporter(total_s=round(total_container, 1), cues=0),
        )
    except Exception as exc:
        return fail(f"could not decode {audio}: {type(exc).__name__}: {exc}")

    samples = len(waveform)
    source_total = samples / float(SAMPLE_RATE)
    if source_total <= 0:
        return fail(f"{audio} decoded to zero length")
    kept = None
    if speech is not None:
        # `worker.py`'s speech only, the same file and the same numpy on the
        # CPU: this is not mlx-whisper's VAD (it has none), it is Crucible's,
        # run before whisper hears anything.
        try:
            waveform, kept = speechonly.cut_for_worker(
                waveform,
                speech,
                audio,
                workerio.decode_reporter(total_s=round(source_total, 1), cues=0),
            )
        except Exception as exc:
            return fail(f"speech detection failed: {type(exc).__name__}: {exc}")
    total = len(waveform) / float(SAMPLE_RATE)

    windows = int(math.ceil(total / window_s))
    send(
        "ready",
        duration_s=source_total,
        windows=windows,
        device=device,
        compute_type=compute_type,
        samples=samples,
        speech_s=None if kept is None else total,
        kept=kept,
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
                # Every window, for `worker.py`'s reason: each window is its own
                # `transcribe()` call and each call starts its history empty.
                initial_prompt=initial_prompt,
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
