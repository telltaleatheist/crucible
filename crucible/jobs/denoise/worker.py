"""The `denoise` worker: audio-separator, run in the rvc env's interpreter.

PHASE4-AUDIO.md sections 0 and 4.2. This module is **standalone**. It imports
the standard library, `soundfile` and `audio_separator` — and nothing from
`crucible`, because the env it runs in has no `crucible` in it and never will.

Why it imports the library instead of running the CLI
-----------------------------------------------------
The opposite of `jobs/rvc/worker.py`, and for the opposite reason. urvc is
spawned because the 96-file recycle needs a process to *die*; here there is one
input and one model load, so a subprocess would only add an interpreter start
and a second checkpoint read to a job that already has exactly one of each.
BookForge reached the same place from the other direction: its per-block
`python run_audio_separator.py` spawn became a resident worker importing
`Separator` precisely because the spawn cost 10-25 s of every ~85 s block
(`electron/scripts/separator_worker.py`).

What this does NOT change, from the app's version
-------------------------------------------------
Every separation parameter left alone here is an argparse default byte-identical
to the constructor default it would replace — normalization 0.9, amplification
0.0, sample_rate 44100, and the mdx/vr/demucs/mdxc parameter blocks — verified
by BookForge against audio-separator 0.31.1's `cli.py`. Nothing about the maths
is Crucible's.

`use_autocast` is the one parameter that is passed, on `cuda-linux` only:
measured by BookForge on 2026-08-29 against a 120 s real-speech block at 20.1x
realtime without it and 29.2x with, output delta peak -72.6 dB / RMS -91.5 dB
relative — below the 16-bit noise floor. It is CUDA-only by audio-separator's
own documentation, so the SERVER decides it from the backend and hands it over.

fd 1 is results and nothing else
--------------------------------
The first thing this file does is dup fd 1 somewhere safe and point the original
at stderr. audio-separator logs to stderr and draws tqdm bars there, but a
library that prints once to stdout would otherwise corrupt the result stream —
which is exactly how narrator's aligner lost a 401-chunk book on 2026-09-05.

The wire, in full
-----------------
    stdin   one object: {model_file_dir, model_filename, input, output_dir,
                         output_format, sample_rate, use_autocast}

    fd 1    {"type": "ready", "model", "sample_rate", "frames", "seconds",
             "channels"}                         before the model is loaded
            {"type": "progress", "stage", "processed", "total"}
            {"type": "result", "stems": [...], "load_seconds", "separate_seconds"}
            {"type": "failed", "message"}
            {"type": "done"}

One result, because the unit of work is one input. The stems ride inside it
rather than being one result each: a result is matched to work by position, and
a run that produced three stems has not done three units of work.
"""

from __future__ import annotations

import os
import sys

# ---- fd 1 is results, stderr is everything else. Before any other import. ----
_RESULTS_FD = os.dup(1)
os.dup2(2, 1)
_RESULTS = os.fdopen(_RESULTS_FD, "w", encoding="utf-8", buffering=1)

import json  # noqa: E402
import time  # noqa: E402


def send(message_type: str, **fields: object) -> None:
    """One JSON object, one line, flushed, on the real fd 1."""
    _RESULTS.write(json.dumps({"type": message_type, **fields}) + "\n")
    _RESULTS.flush()


def fail(message: str) -> int:
    send("failed", message=message)
    return 1


def require(request: dict, key: str, kind):
    """One required key, or a refusal naming it. Every key is required."""
    if key not in request:
        raise KeyError(f"the denoise request has no {key!r}; every key is required")
    value = request[key]
    kinds = kind if isinstance(kind, tuple) else (kind,)
    wrong = not isinstance(value, kinds) or (
        isinstance(value, bool) and bool not in kinds
    )
    if wrong:
        raise KeyError(
            f"the denoise request's {key!r} must be "
            f"{'/'.join(k.__name__ for k in kinds)}, got {type(value).__name__}"
        )
    return value


def main() -> int:
    line = sys.stdin.readline()
    if not line.strip():
        return fail("the denoise worker was given no request on stdin")
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        return fail(f"the denoise request is not JSON: {exc}")
    if not isinstance(request, dict):
        return fail(
            f"the denoise request must be a JSON object, got {type(request).__name__}"
        )

    try:
        model_file_dir = require(request, "model_file_dir", str)
        model_filename = require(request, "model_filename", str)
        source = require(request, "input", str)
        output_dir = require(request, "output_dir", str)
        output_format = require(request, "output_format", str)
        expected_rate = require(request, "sample_rate", int)
        use_autocast = require(request, "use_autocast", bool)
    except KeyError as exc:
        return fail(str(exc.args[0]))

    import soundfile

    try:
        info = soundfile.info(source)
    except Exception as exc:  # noqa: BLE001 - any unreadable input is the same news
        return fail(
            f"{os.path.basename(source)} could not be read as audio: "
            f"{type(exc).__name__}: {exc}"
        )

    # Reported BEFORE the model is loaded, so a job's event log says what it was
    # given even when the load is what fails.
    send(
        "ready",
        model=model_filename,
        sample_rate=int(info.samplerate),
        frames=int(info.frames),
        seconds=round(float(info.frames) / float(info.samplerate), 3)
        if info.samplerate
        else 0.0,
        channels=int(info.channels),
    )

    if int(info.samplerate) != expected_rate:
        # NOT RESAMPLED HERE, and that is the whole point. The model is
        # 44.1 kHz native and its librosa front-end crashes on other rates; a
        # server that quietly resampled would return a stem whose offsets no
        # longer match the audio the client sliced by, and nothing in the output
        # would say so. The client resamples, because the client is the one that
        # knows what rate it wants back.
        return fail(
            f"this input is {info.samplerate} Hz and {model_filename} is "
            f"{expected_rate} Hz native. Nothing was resampled: a stem returned "
            "at a rate the caller did not send is a stem whose sample offsets no "
            "longer mean anything. Resample before sending"
        )

    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        return fail(
            f"{output_dir} is not empty; this worker identifies the separator's "
            "outputs by what appears in it, so it must start empty"
        )

    started = time.perf_counter()
    from audio_separator.separator import Separator

    separator = Separator(
        model_file_dir=model_file_dir,
        output_dir=output_dir,
        output_format=output_format,
        # CUDA-only by audio-separator's own docs; the server decides.
        use_autocast=use_autocast,
    )
    try:
        separator.load_model(model_filename=model_filename)
    except Exception as exc:  # noqa: BLE001
        return fail(
            f"audio-separator could not load {model_filename} from "
            f"{model_file_dir}: {type(exc).__name__}: {exc}"
        )
    load_seconds = time.perf_counter() - started

    send("progress", stage="separating", processed=0, total=1)
    began = time.perf_counter()
    try:
        separator.separate(source)
    except Exception as exc:  # noqa: BLE001
        return fail(
            f"audio-separator failed on {os.path.basename(source)}: "
            f"{type(exc).__name__}: {exc}"
        )
    separate_seconds = time.perf_counter() - began

    # THE OUTPUT DIRECTORY IS THE ANSWER, not `separate()`'s return value. Which
    # of the two that call gives back — bare filenames or absolute paths — has
    # changed between audio-separator versions, and the directory was empty a
    # moment ago, so everything in it is this separation's. One fact, read from
    # the filesystem that holds it.
    stems = []
    for name in sorted(os.listdir(output_dir)):
        produced = os.path.join(output_dir, name)
        if not os.path.isfile(produced):
            continue
        try:
            stem_info = soundfile.info(produced)
        except Exception as exc:  # noqa: BLE001
            return fail(
                f"{name} came out of the separator but could not be read back as "
                f"audio: {type(exc).__name__}: {exc}"
            )
        stems.append(
            {
                "name": name,
                "sample_rate": int(stem_info.samplerate),
                "frames": int(stem_info.frames),
                "channels": int(stem_info.channels),
                "bytes": os.path.getsize(produced),
            }
        )
    if not stems:
        return fail(
            f"audio-separator finished and wrote nothing into {output_dir}"
        )

    send(
        "result",
        stems=stems,
        load_seconds=round(load_seconds, 2),
        separate_seconds=round(separate_seconds, 2),
    )
    send("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
