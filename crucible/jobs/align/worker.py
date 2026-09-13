"""The `align` worker: Qwen3-ForcedAligner, run in its own interpreter and HELD.

PHASE4-AUDIO.md sections 0 and 2. This module is **standalone**. It imports the
standard library, `numpy`, `soundfile`, `torch` and `qwen_asr`, and nothing from
`crucible` — the env it runs in has no `crucible` installed and never will, so an
import from the server package here would be an ImportError at the first real job
and a green test suite right up until then.

It is the first worker that outlives a job
------------------------------------------
`asr` loads, transcribes one file and exits. An aligner that did that would read
1.7 GB of weights per chunk, and a book is hundreds of chunks — narrator caches
the model per worker process for exactly this reason
(`python/narrator/align/aligner.py:519`, "one load per worker process, reused for
the whole book"). So this reads request after request from stdin until EOF, and
`crucible.workers.WorkerSession` on the other side is what holds it open.

The wire, in full
-----------------
    stdin   one object per line. Two ops, and the op is required:

            {"op": "load", "model_dir", "device", "dtype"}
                -> ready {seconds, device, dtype}, done

            {"op": "align", "language", "max_audio_s", "ffmpeg",
             "chunks": [{"audio": "<path>", "text": "..."}]}
            `language` is the model's own ENGLISH LANGUAGE NAME ("English",
            "Cantonese"), not the ISO code the client sent: `model.align` takes
            the name. The mapping is the server's, checked before the job is
            queued, so an unsupported language never reaches a loaded model.
                -> ready {chunks}
                   result {items: [{text, start, end}]}   one per chunk
                   result {error: "..."}                  a chunk that failed
                   done

    fd 1    the five message kinds every phase 4 worker speaks, and no other.

**A chunk carries no index and a result carries no index.** The server knows a
result's chunk from its POSITION in the stream, because an index a worker reports
is an index a worker can get wrong — which is not a theory: narrator's aligner
answers job k by deal position for this reason (`align/env.py:317`), on a book
where the alternative had already gone wrong.

fd 1 is results and nothing else
--------------------------------
The first thing this file does, before importing anything that could print, is
dup fd 1 somewhere safe and point the original at stderr. THIS IS WHOSE LESSON IT
IS: on Owen's first in-app Higgs book (witches, 401 chunks, 2026-09-05) whisperx's
logger wrote a `Failed to align segment` warning through a
`StreamHandler(sys.stdout)`; it landed between two result lines, the parent's
`json.loads` died with "Extra data", and the whole book failed with no traceback
because the stdout tail won over stderr. Three chunks aligned fine. A protocol
channel any library can write to is not a protocol channel
(`python/narrator/align/worker.py:56`).

What this worker does NOT do
----------------------------
It does not map items onto words, it does not check that the model returned the
text it was given, it derives no scores and it writes no coverage report. All of
that stays in BookForge (PHASE4-AUDIO.md section 2), which is most of the value
of the feature and none of the value of a server. What comes back is one
timestamped item per *the model's own* tokenization — 665 items for a 668-word
English window, measured 2026-09-08 — and Crucible asserts nothing about words.
"""

from __future__ import annotations

import os
import sys

# ---- fd 1 is results, stderr is everything else. Before any other import. ----
_RESULTS_FD = os.dup(1)
os.dup2(2, 1)
_RESULTS = os.fdopen(_RESULTS_FD, "w", encoding="utf-8", buffering=1)

import json  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

#: Qwen3-ForcedAligner works at 16 kHz mono, always. Not a parameter — it is the
#: sample rate the feature extractor was trained at, and the client should not
#: have to know it exists (`python/narrator/align/aligner.py:97`).
SAMPLE_RATE = 16_000

#: Report at most this often while aligning. A chunk is seconds of work, so this
#: is about not filling an SSE stream with a line per chunk on a 1,400-chunk book
#: — the per-chunk news is the `cue` event, which the server emits from the
#: result.
PROGRESS_WALL_SECONDS = 1.0

#: The model is loaded once and lives here for the process's life. A dict and not
#: a bare name so the load op can be idempotent about saying what is loaded.
_STATE: dict = {"model": None, "model_dir": None, "device": None, "dtype": None}


def send(message_type: str, **fields: object) -> None:
    """One JSON object, one line, flushed, on the real fd 1."""
    _RESULTS.write(json.dumps({"type": message_type, **fields}) + "\n")
    _RESULTS.flush()


def fail(message: str) -> None:
    send("failed", message=message)


def require(request: dict, key: str, kind: type):
    """One required key, or a refusal naming it.

    Nothing in either request has a default. A `language` Crucible did not send
    is not "English" — it is a producer bug, and it says so, for the reason
    narrator's own worker gives about `REQUIRED_JOB_FIELDS`: a producer that
    stopped sending `device` would have aligned on CPU while the operator
    believed otherwise.
    """
    if key not in request:
        raise KeyError(
            f"the align request has no {key!r}; every parameter is required "
            "because every one of them changes the alignment"
        )
    value = request[key]
    kinds = kind if isinstance(kind, tuple) else (kind,)
    # bool is a subclass of int, so a bool where a number is wanted passes
    # `isinstance` and is still wrong.
    wrong = not isinstance(value, kinds) or (
        isinstance(value, bool) and bool not in kinds
    )
    if wrong:
        raise KeyError(
            f"the align request's {key!r} must be "
            f"{'/'.join(k.__name__ for k in kinds)}, got {type(value).__name__}"
        )
    return value


# ------------------------------------------------------------------- decoding


def decode(ffmpeg: str, audio_path: str):
    """One audio file -> a mono float32 array at 16 kHz. Raises on any failure.

    ffmpeg rather than a python decoder, for the same reason narrator uses it:
    the chunk files are whatever the renderer wrote (FLAC today, WAV yesterday)
    and ffmpeg reads them all identically. `-f f32le` is already normalised to
    [-1, 1], so there is no rescale and no second copy. stderr is drained on a
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
        for block in iter(lambda: process.stderr.read(65536), b""):
            errors.append(block)

    pump = threading.Thread(target=drain, daemon=True)
    pump.start()

    buffer = bytearray()
    while True:
        block = process.stdout.read(1 << 20)
        if not block:
            break
        buffer += block
    process.stdout.close()
    code = process.wait()
    pump.join(timeout=5)
    if code != 0:
        tail = b"".join(errors).decode("utf-8", "replace").strip()[-500:]
        raise RuntimeError(f"ffmpeg exited {code} on {audio_path}: {tail}")
    return numpy.frombuffer(buffer, dtype=numpy.float32)


# -------------------------------------------------------------------- loading


def load(request: dict) -> None:
    """The `load` op: put the checkpoint on the device and say how long it took."""
    model_dir = require(request, "model_dir", str)
    device = require(request, "device", str)
    dtype_name = require(request, "dtype", str)

    try:
        import torch
        from qwen_asr import Qwen3ForcedAligner
    except ImportError as exc:
        raise RuntimeError(
            f"the align env in {sys.executable} cannot import what it needs "
            f"({exc}). Build it with `crucible install align`."
        ) from None

    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise RuntimeError(
            f"torch has no dtype {dtype_name!r}; the manifest's dtype reaches this "
            "line verbatim"
        )

    started = time.time()
    model = Qwen3ForcedAligner.from_pretrained(
        model_dir, dtype=dtype, device_map=device
    )
    seconds = time.time() - started

    _STATE.update(
        model=model, model_dir=model_dir, device=device, dtype=dtype_name
    )
    send("ready", seconds=seconds, device=device, dtype=dtype_name)
    send("done")


# ------------------------------------------------------------------- aligning


def align_one(model, audio, text: str, language: str, max_audio_s: float) -> list:
    """One chunk -> the model's own items, or a raise naming why not.

    The model takes a PATH (or a URL), not an array, so the decoded audio is
    written to a temporary 16 kHz PCM_16 wav for the call and deleted after. That
    is a documented property of the API as verified in the `qwen-align` env on
    2026-09-08, not an assumption, and narrator does exactly the same thing
    (`aligner.py:750`).
    """
    import soundfile

    duration = audio.size / float(SAMPLE_RATE)
    if duration > max_audio_s:
        # THE REFUSAL, NOT A SPLIT. The model card's own limit is timestamps
        # "within up to 5 minutes"; cutting a longer chunk into pieces here would
        # change the alignment and nothing in the output would say it had
        # happened. Narrator chunks are ~90 s, so anything past this is a
        # caller's bug and is reported as one.
        raise ValueError(
            f"{duration:.1f}s of audio; Qwen3-ForcedAligner places timestamps "
            f"within {max_audio_s:.0f}s and says nothing about longer input. Cut "
            "the chunk before aligning it — Crucible will not, because a split "
            "here is a different alignment with nothing to say so."
        )

    handle, wav_path = tempfile.mkstemp(prefix="crucible-align-", suffix=".wav")
    os.close(handle)
    try:
        soundfile.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
        results = model.align(audio=wav_path, text=text, language=language)
    finally:
        # One temp wav per chunk and a book is hundreds of them; leaving them
        # behind fills the temp directory with a book's worth of audio.
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    # ONE LIST PER AUDIO, and one audio was passed.
    items = results[0]
    if not items:
        raise ValueError(
            "the model returned no items at all for this chunk; it was given "
            "audio it could not place this text in"
        )
    return [
        {
            "text": str(item.text),
            "start": float(item.start_time),
            "end": float(item.end_time),
        }
        for item in items
    ]


def align(request: dict) -> None:
    """The `align` op: one result per chunk, in the order the chunks arrived."""
    model = _STATE["model"]
    if model is None:
        raise RuntimeError(
            "an align request arrived before a load request; the session's first "
            "exchange loads the model"
        )
    language = require(request, "language", str)
    ffmpeg = require(request, "ffmpeg", str)
    max_audio_s = float(require(request, "max_audio_s", (int, float)))
    chunks = require(request, "chunks", list)
    if not chunks:
        raise RuntimeError("an align request with no chunks is not a request")

    send("ready", chunks=len(chunks))

    last_report = [0.0]

    def report(done: int) -> None:
        now = time.time()
        if now - last_report[0] < PROGRESS_WALL_SECONDS and done != len(chunks):
            return
        last_report[0] = now
        send("progress", stage="aligning", processed=done, total=len(chunks))

    for position, chunk in enumerate(chunks):
        try:
            if not isinstance(chunk, dict):
                raise ValueError(f"chunk {position} is not an object")
            audio = decode(ffmpeg, require(chunk, "audio", str))
            items = align_one(
                model, audio, require(chunk, "text", str), language, max_audio_s
            )
        except Exception as exc:
            # One chunk's failure is reported and the run continues, so a single
            # bad chunk does not cost the rest of the book — and so the server
            # learns about every failure in one run rather than one per re-run.
            # NO RETRY AND NO SECOND BACKEND, EVER (Owen's ruling, 2026-09-05):
            # what comes back is what the model said, or why it said nothing.
            send("result", error=f"{type(exc).__name__}: {exc}")
        else:
            send("result", items=items)
        report(position + 1)

    send("done")


# ----------------------------------------------------------------------- main


OPS = {"load": load, "align": align}


def main() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"the align request is not JSON: {exc}")
            return 1
        if not isinstance(request, dict):
            fail(f"the align request must be a JSON object, got {type(request).__name__}")
            return 1
        op = request.get("op")
        handler = OPS.get(op)
        if handler is None:
            fail(f"the align request's op is {op!r}; this worker takes {sorted(OPS)}")
            return 1
        try:
            handler(request)
        except KeyError as exc:
            fail(str(exc.args[0]))
            return 1
        except Exception as exc:
            # A failure of the WHOLE request rather than of one chunk: the model
            # would not load, or the request was malformed. The session is over
            # either way, so this exits rather than waiting for another line it
            # could not honour.
            fail(f"{type(exc).__name__}: {exc}")
            return 1
    # EOF on stdin: the session was stopped politely. Exit 0 so `stop()` sees a
    # worker that went when it was asked.
    return 0


if __name__ == "__main__":
    sys.exit(main())
