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

DETECT_SECONDS = 30

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


def serialise(segment: dict, want_words: bool) -> dict:
    row: dict = {
        "start": float(segment["start"]),
        "end": float(segment["end"]),
        "text": str(segment["text"]),
    }
    if not want_words:
        return row
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
    from mlx_whisper.tokenizer import get_tokenizer
    from mlx_whisper.transcribe import ModelHolder

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
    table = probabilities[0] if isinstance(probabilities, list) else probabilities
    best = max(table, key=table.get)
    return str(best), float(table[best])

NO_VAD = (
    "this request asks for vad_filter and mlx-whisper has no VAD at all "
    "(faster-whisper's is Silero; mlx-whisper's no_speech_threshold is a "
    "per-segment judgement by the model itself, which is a different "
    "thing on different evidence). Transcribing without it would be a "
    "different transcript with nothing in the file to say so"
)


class WorkerFailed(Exception):
    pass


def require_language(request: dict) -> "str | None":
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
    return language


def parse_request(request: dict) -> dict:
    return {
        "model_dir": require(request, "model_dir", str),
        "ffmpeg": require(request, "ffmpeg", str),
        "audio": require(request, "audio", str),
        "device": require(request, "device", str),
        "compute_type": require(request, "compute_type", str),
        "vad_filter": require(request, "vad_filter", bool),
        "word_timestamps": require(request, "word_timestamps", bool),
        "window_s": require(request, "window_s", int),
        "overlap_s": require(request, "overlap_s", int),
        "language": require_language(request),
        "initial_prompt": require_prompt(request),
        "speech": speechonly.from_request(request),
    }


def import_mlx():
    try:
        import mlx.core as mx
        import mlx_whisper
    except ImportError as exc:
        raise WorkerFailed(
            f"mlx-whisper is not importable in {sys.executable}: {exc}. "
            "The asr env is installed with `crucible install asr`."
        ) from None
    return mx, mlx_whisper


def choose_dtype(mx, compute_type: str, device: str):
    dtypes = {"float16": mx.float16, "float32": mx.float32}
    dtype = dtypes.get(compute_type)
    if dtype is None:
        raise WorkerFailed(
            f"compute_type {compute_type!r} is not one mlx-whisper runs; it takes "
            f"{sorted(dtypes)}"
        )
    if device != "metal":
        raise WorkerFailed(
            f"device {device!r} is not mlx-whisper's; MLX runs on Metal and "
            "nothing else, and the server sends 'metal' on this backend"
        )
    return dtype


def check_prompt_fits(model_dir: str, dtype, initial_prompt: "str | None") -> None:
    if initial_prompt is None:
        return
    failure = check_prompt_length(model_dir, dtype, initial_prompt)
    if failure is not None:
        raise WorkerFailed(failure)


def decode_source(ffmpeg: str, audio: str):
    total_container = probe_duration(ffmpeg, audio, "asr")
    try:
        waveform = decode(
            ffmpeg,
            audio,
            workerio.decode_reporter(total_s=round(total_container, 1), cues=0),
        )
    except Exception as exc:
        raise WorkerFailed(f"could not decode {audio}: {type(exc).__name__}: {exc}") from None
    if len(waveform) / float(SAMPLE_RATE) <= 0:
        raise WorkerFailed(f"{audio} decoded to zero length")
    return waveform


def keep_speech(waveform, speech, audio: str, source_total: float):
    if speech is None:
        return waveform, None
    try:
        return speechonly.cut_for_worker(
            waveform,
            speech,
            audio,
            workerio.decode_reporter(total_s=round(source_total, 1), cues=0),
        )
    except Exception as exc:
        raise WorkerFailed(f"speech detection failed: {type(exc).__name__}: {exc}") from None


class TranscribeProgress:
    def __init__(self, total: float) -> None:
        self.total = total
        self.emitted = 0
        self.last_fraction = -1.0
        self.last_wall = time.time()

    def report(self, processed: float) -> None:
        fraction = min(1.0, processed / self.total)
        now = time.time()
        if (
            fraction - self.last_fraction < PROGRESS_FRACTION_STEP
            and now - self.last_wall < PROGRESS_WALL_SECONDS
        ):
            return
        self.last_fraction = fraction
        self.last_wall = now
        self.send(processed)

    def send(self, processed: float) -> None:
        send(
            "progress",
            stage="transcribing",
            processed_s=round(processed, 1),
            total_s=round(self.total, 1),
            cues=self.emitted,
        )


def transcribe_window(backend, waveform, params: dict, index: int, progress: TranscribeProgress) -> None:
    mx, mlx_whisper, dtype = backend
    window_s, total, model_dir = params["window_s"], progress.total, params["model_dir"]
    start = index * float(window_s)
    boundary = min(start + window_s, total)
    first = int(start * SAMPLE_RATE)
    last = int(min(boundary + params["overlap_s"], total) * SAMPLE_RATE)
    window = waveform[first:last]
    try:
        if params["language"] is None:
            code, probability = detect_language(model_dir, window, dtype)
        else:
            code, probability = params["language"], 1.0
        result = mlx_whisper.transcribe(
            window,
            path_or_hf_repo=model_dir,
            language=code,
            word_timestamps=params["word_timestamps"],
            fp16=dtype is mx.float16,
            initial_prompt=params["initial_prompt"],
            verbose=None,
        )
        rows = []
        for segment in result["segments"]:
            rows.append(serialise(segment, params["word_timestamps"]))
            progress.emitted += 1
            progress.report(min(total, start + float(segment["end"])))
    except Exception as exc:
        send("result", error=f"{type(exc).__name__}: {exc}")
        return
    send(
        "result",
        segments=rows,
        language=result["language"],
        language_probability=probability,
    )


def transcribe(params: dict) -> None:
    if params["vad_filter"]:
        raise WorkerFailed(NO_VAD)
    mx, mlx_whisper = import_mlx()
    dtype = choose_dtype(mx, params["compute_type"], params["device"])
    check_prompt_fits(params["model_dir"], dtype, params["initial_prompt"])
    waveform = decode_source(params["ffmpeg"], params["audio"])
    samples = len(waveform)
    source_total = samples / float(SAMPLE_RATE)
    waveform, kept = keep_speech(waveform, params["speech"], params["audio"], source_total)
    total = len(waveform) / float(SAMPLE_RATE)
    windows = int(math.ceil(total / params["window_s"]))
    send(
        "ready",
        duration_s=source_total,
        windows=windows,
        device=params["device"],
        compute_type=params["compute_type"],
        samples=samples,
        speech_s=None if kept is None else total,
        kept=kept,
    )
    progress = TranscribeProgress(total)
    for index in range(windows):
        transcribe_window((mx, mlx_whisper, dtype), waveform, params, index, progress)
    progress.send(total)
    send("done")


def main() -> int:
    request = workerio.read_request("asr")
    if request is None:
        return 1
    try:
        params = parse_request(request)
    except KeyError as exc:
        return fail(str(exc.args[0]))
    try:
        transcribe(params)
    except WorkerFailed as exc:
        return fail(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
