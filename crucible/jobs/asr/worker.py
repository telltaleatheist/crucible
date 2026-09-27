from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import workerio

sys.path.pop(0)
workerio.claim_stdout()

import math
import time

from workerio import SAMPLE_RATE, decode, fail, probe_duration, send

speechonly = workerio.load_sibling("speechonly", __file__)

PROGRESS_FRACTION_STEP = 0.002
PROGRESS_WALL_SECONDS = 1.5


def require(request: dict, key: str, kind: type) -> object:
    return workerio.require(
        request,
        key,
        kind,
        "asr",
        "every parameter is required because every one of them changes the "
        "transcript",
    )


def require_prompt(request: dict) -> "str | None":
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


def prompt_too_long(count: int, ceiling: int) -> str:
    return (
        f"initial_prompt is {count} tokens and whisper keeps only the last "
        f"{ceiling} of its prompt history, so the first {count - ceiling} would "
        "be dropped without a word. Send a shorter prompt: the title and the "
        "names in it, not the text"
    )


def serialise(segment, want_words: bool) -> dict:
    row: dict = {
        "start": float(segment.start),
        "end": float(segment.end),
        "text": str(segment.text),
    }
    if not want_words:
        return row
    words = getattr(segment, "words", None)
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
    request = workerio.read_request("asr")
    if request is None:
        return 1

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

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        return fail(
            f"faster-whisper is not importable in {sys.executable}: {exc}. "
            "The asr env is installed with `crucible install asr`."
        )

    try:
        model = WhisperModel(model_dir, device=device, compute_type=compute_type)
    except Exception as exc:
        return fail(
            f"could not load {model_dir} on {device} at {compute_type}: "
            f"{type(exc).__name__}: {exc}"
        )

    if initial_prompt is not None:
        ceiling = int(model.max_length) // 2 - 1
        count = len(
            model.hf_tokenizer.encode(
                " " + initial_prompt.strip(), add_special_tokens=False
            ).ids
        )
        if count > ceiling:
            return fail(prompt_too_long(count, ceiling))

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
        first = int(start * SAMPLE_RATE)
        last = int(min(boundary + overlap_s, total) * SAMPLE_RATE)
        try:
            segments, info = model.transcribe(
                waveform[first:last],
                language=language,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                initial_prompt=initial_prompt,
            )
            rows = []
            for segment in segments:
                rows.append(serialise(segment, word_timestamps))
                emitted += 1
                transcribe_progress(min(total, start + float(segment.end)))
        except Exception as exc:
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
