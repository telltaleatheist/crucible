"""The `denoise` worker: audio-separator, run in the rvc env's interpreter and HELD.

PHASE4-AUDIO.md sections 0 and 4.2. This module is **standalone**. It imports
the standard library, `soundfile` and `audio_separator` — and nothing from
`crucible`, because the env it runs in has no `crucible` in it and never will.

It outlives a job, and that is the whole point
----------------------------------------------
This worker used to load the checkpoint, separate one block and exit. That is
one model load per block, and the client sends a book's worth of blocks: the app
concatenates a session's sentences into ~22-minute blocks, and **a 15-hour book
is ~44 of them**. BookForge had already measured what that costs and had already
fixed it — `electron/scripts/separator_worker.py` (bookforge `019afa52`) replaced
a per-block spawn with a resident worker because the fixed cost was "being paid
44 times (10-25 s each) for ~85 s of real work per block", which
`electron/denoise-bridge.ts:27-32` puts at *"roughly a third of the pass. One
load now serves the whole book."* The warm figure from that commit is 7.4 s of
an 11.0 s one-shot.

Crucible reintroduced it, in writing, by reasoning that "a separator loads once
per job either way" — true, and exactly the error: the client sends ~44 jobs.
Every one of them succeeded and every log was clean, which is how it hid.

So this reads request after request from stdin until EOF, and
`crucible.workers.WorkerSession` on the other side is what holds it open —
`jobs/align/worker.py`'s shape, for `jobs/align/worker.py`'s reason. Owen's
ruling, 2026-09-15.

The wire, in full
-----------------
    stdin   one object per line. Two ops, and the op is required:

            {"op": "load", "model_file_dir", "model_filename", "use_autocast"}
                -> ready {seconds}, done

            {"op": "separate", "input", "output_dir", "output_format",
             "sample_rate"}
                -> ready {sample_rate, frames, seconds, channels}
                   progress {stage, processed, total}
                   result {stems, separate_seconds}
                   done

    fd 1    the five message kinds every phase 4 worker speaks, and no other.

**`output_format` and `sample_rate` ride on the SEPARATE request, not the load.**
The format is what the stems are written as and the rate is what the input is
checked against; neither is a property of the checkpoint on the card, and putting
them on the load would freeze a job's answer to whatever the first job of the
session happened to ask for.

Per-request output dir, and the assertion that guards it
---------------------------------------------------------
`Separator.load_model()` bakes `output_dir` into the architecture instance's
common config, and `common_separator.write_audio_pydub` is the ONLY place it is
read (`os.path.join(self.output_dir, stem_path)`), at write time. So a
per-request output directory is re-pointing that one attribute — on BOTH the
separator and the model instance — before each `separate()`. This is not a
guess: it is `separator_worker.py`'s own mechanism, verified against
audio-separator 0.31.1, and like that worker this one ASSERTS the attribute
exists at LOAD time rather than discovering it missing mid-book and writing 44
blocks' stems into one directory.

What this does NOT change, from the app's version
-------------------------------------------------
Every separation parameter left alone here is an argparse default byte-identical
to the constructor default it would replace — normalization 0.9, amplification
0.0, sample_rate 44100, and the mdx/vr/demucs/mdxc parameter blocks — verified by
BookForge against audio-separator 0.31.1's `cli.py`. Reusing one `Separator`
across files is the library's OWN sanctioned multi-file usage:
`audio_separator.utils.cli:main` loops `separate()` over every input with the
same instance, clearing the file-specific paths and the GPU cache between files.
Nothing about the maths is Crucible's, and nothing about it changed here.

`use_autocast` is the one parameter that is passed, on `cuda-linux` only:
measured by BookForge on 2026-08-29 against a 120 s real-speech block at 20.1x
realtime without it and 29.2x with, output delta peak -72.6 dB / RMS -91.5 dB
relative — below the 16-bit noise floor. It is CUDA-only by audio-separator's
own documentation, so the SERVER decides it from the backend and hands it over.
It belongs on the LOAD because it is what the model was loaded with, and the
resident row records it for exactly that reason.

fd 1 is results and nothing else
--------------------------------
The first thing this file does is dup fd 1 somewhere safe and point the original
at stderr. audio-separator logs to stderr and draws tqdm bars there, but a
library that prints once to stdout would otherwise corrupt the result stream —
which is exactly how narrator's aligner lost a 401-chunk book on 2026-09-05.

One result per separate request, because the unit of work is one input. The
stems ride inside it rather than being one result each: a result is matched to
work by position, and a run that produced three stems has not done three units
of work.
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

#: The loaded separator and the model instance whose `output_dir` each request
#: re-points. A dict and not bare names so the load op can be idempotent about
#: saying what is loaded, exactly as the aligner's `_STATE` is.
_STATE: dict = {"separator": None, "model_instance": None, "model_filename": None}


def send(message_type: str, **fields: object) -> None:
    """One JSON object, one line, flushed, on the real fd 1."""
    _RESULTS.write(json.dumps({"type": message_type, **fields}) + "\n")
    _RESULTS.flush()


def fail(message: str) -> None:
    send("failed", message=message)


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


# -------------------------------------------------------------------- loading


def load(request: dict) -> None:
    """The `load` op: put the checkpoint on the card and say how long it took."""
    model_file_dir = require(request, "model_file_dir", str)
    model_filename = require(request, "model_filename", str)
    use_autocast = require(request, "use_autocast", bool)

    started = time.perf_counter()
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise RuntimeError(
            f"the rvc env in {sys.executable} cannot import audio_separator "
            f"({exc}). Build it with `crucible install rvc`."
        ) from None

    separator = Separator(
        model_file_dir=model_file_dir,
        # A REAL directory this process owns, replaced per request before every
        # separation. audio-separator's constructor does `os.makedirs` on it, so
        # it cannot be a placeholder that does not exist — and it must not be a
        # directory any job's stems land in, because a job identifies its own
        # outputs by what appears in a directory that was empty a moment ago.
        output_dir=os.path.join(model_file_dir, ".crucible-separator-unset"),
        # The format is the SEPARATE request's and is re-pointed with the
        # directory; this is the constructor's own default standing in until the
        # first request names one.
        output_format="flac",
        use_autocast=use_autocast,
    )
    separator.load_model(model_filename=model_filename)

    # The per-request output dir rides on this attribute. If a future
    # audio-separator moves it, fail HERE — loudly, at load — rather than
    # silently writing every block's stems into the wrong directory.
    # `separator_worker.py:106-108`'s check, kept word for word in intent.
    model_instance = getattr(separator, "model_instance", None)
    if model_instance is None:
        raise RuntimeError(
            "audio-separator loaded no model_instance — this worker cannot "
            "separate, and cannot re-point a per-request output directory"
        )
    if not hasattr(model_instance, "output_dir"):
        raise RuntimeError(
            "audio-separator's model instance has no output_dir attribute — this "
            "worker's per-request output directory mechanism no longer applies "
            "to this version. Every block's stems would land in one directory "
            "and each job would claim the previous job's outputs"
        )

    seconds = time.perf_counter() - started
    _STATE.update(
        separator=separator,
        model_instance=model_instance,
        model_filename=model_filename,
    )
    send("ready", seconds=seconds)
    send("done")


# ------------------------------------------------------------------ separating


def separate(request: dict) -> None:
    """The `separate` op: one block in, its stems out."""
    separator = _STATE["separator"]
    if separator is None:
        raise RuntimeError(
            "a separate request arrived before a load request; the session's "
            "first exchange loads the model"
        )
    model_instance = _STATE["model_instance"]
    model_filename = _STATE["model_filename"]

    source = require(request, "input", str)
    output_dir = require(request, "output_dir", str)
    output_format = require(request, "output_format", str)
    expected_rate = require(request, "sample_rate", int)

    import soundfile

    try:
        info = soundfile.info(source)
    except Exception as exc:  # noqa: BLE001 - any unreadable input is the same news
        raise RuntimeError(
            f"{os.path.basename(source)} could not be read as audio: "
            f"{type(exc).__name__}: {exc}"
        ) from None

    # Reported BEFORE anything is separated, so a job's event log says what it
    # was given even when the separation is what fails.
    send(
        "ready",
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
        raise RuntimeError(
            f"this input is {info.samplerate} Hz and {model_filename} is "
            f"{expected_rate} Hz native. Nothing was resampled: a stem returned "
            "at a rate the caller did not send is a stem whose sample offsets no "
            "longer mean anything. Resample before sending"
        )

    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        raise RuntimeError(
            f"{output_dir} is not empty; this worker identifies the separator's "
            "outputs by what appears in it, so it must start empty"
        )

    # THE PER-REQUEST OUTPUT DIRECTORY, on both objects. See the module docstring
    # — `write_audio_pydub` reads the model instance's copy, and the separator's
    # own is what its logging and its file-path bookkeeping read.
    separator.output_dir = output_dir
    model_instance.output_dir = output_dir
    # THE FORMAT ON BOTH OBJECTS TOO, for the output directory's reason
    # (2026-09-25). audio-separator's model instance copies `output_format` at
    # load (`CommonSeparator.__init__`, common_separator.py L74) and names and
    # writes each stem with ITS copy (L384-386), so setting only the separator's
    # left every stem in the load's format: FLAC, where the server asks for WAV.
    # ContentStudio found a vocals stem arriving as .flac on 1.0.38.
    separator.output_format = output_format
    model_instance.output_format = output_format

    send("progress", stage="separating", processed=0, total=1)
    began = time.perf_counter()
    try:
        separator.separate(source)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"audio-separator failed on {os.path.basename(source)}: "
            f"{type(exc).__name__}: {exc}"
        ) from None
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
            raise RuntimeError(
                f"{name} came out of the separator but could not be read back as "
                f"audio: {type(exc).__name__}: {exc}"
            ) from None
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
        raise RuntimeError(
            f"audio-separator finished and wrote nothing into {output_dir}"
        )
    # THE CONTAINER ASKED FOR IS THE CONTAINER RETURNED, checked rather than
    # trusted: a future audio-separator that moved the attribute again would
    # otherwise change what every client receives, silently.
    wrong = [s["name"] for s in stems if not s["name"].lower().endswith("." + output_format.lower())]
    if wrong:
        raise RuntimeError(
            f"the separator was asked for {output_format} and wrote {wrong}; the "
            "stem container is part of the contract and is not substituted"
        )

    send("result", stems=stems, separate_seconds=round(separate_seconds, 2))
    send("done")


# ----------------------------------------------------------------------- main


OPS = {"load": load, "separate": separate}


def main() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"the denoise request is not JSON: {exc}")
            return 1
        if not isinstance(request, dict):
            fail(
                f"the denoise request must be a JSON object, got "
                f"{type(request).__name__}"
            )
            return 1
        op = request.get("op")
        handler = OPS.get(op)
        if handler is None:
            fail(f"the denoise request's op is {op!r}; this worker takes {sorted(OPS)}")
            return 1
        try:
            handler(request)
        except KeyError as exc:
            fail(str(exc.args[0]))
            return 1
        except Exception as exc:  # noqa: BLE001
            # A failure of the WHOLE request: the model would not load, the
            # input was unreadable, the separator raised. The session is over
            # either way — unlike `align`, where one bad chunk of many is
            # reported and the run continues, a separate request carries ONE
            # block and there is nothing left of it to continue.
            fail(f"{type(exc).__name__}: {exc}")
            return 1
    # EOF on stdin: the session was stopped politely. Exit 0 so `stop()` sees a
    # worker that went when it was asked.
    return 0


if __name__ == "__main__":
    sys.exit(main())
