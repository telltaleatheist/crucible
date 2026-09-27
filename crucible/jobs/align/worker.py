from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import workerio

sys.path.pop(0)
workerio.claim_stdout()

import json
import tempfile
import time

from workerio import (
    SAMPLE_RATE,
    cap_memory,
    decode,
    fail,
    memory_line,
    send,
)

PROGRESS_WALL_SECONDS = 1.0

_STATE: dict = {"model": None, "model_dir": None, "device": None, "dtype": None}


def require(request: dict, key: str, kind: type):
    return workerio.require(
        request,
        key,
        kind,
        "align",
        "every parameter is required because every one of them changes the "
        "alignment",
    )


def load(request: dict) -> None:
    model_dir = require(request, "model_dir", str)
    device = require(request, "device", str)
    dtype_name = require(request, "dtype", str)
    if "memory_cap_bytes" not in request:
        raise KeyError(
            "the align request has no 'memory_cap_bytes'; it is required, and "
            "null where there is no CUDA cap"
        )
    memory_cap_bytes = request["memory_cap_bytes"]

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

    fraction = cap_memory(torch, memory_cap_bytes)
    started = time.time()
    model = Qwen3ForcedAligner.from_pretrained(
        model_dir, dtype=dtype, device_map=device
    )
    seconds = time.time() - started
    memory_line(torch, "after load")

    _STATE.update(
        model=model, model_dir=model_dir, device=device, dtype=dtype_name
    )
    send(
        "ready",
        seconds=seconds,
        device=device,
        dtype=dtype_name,
        memory_cap_bytes=memory_cap_bytes,
        memory_fraction=fraction,
        alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    )
    send("done")


def align_one(model, audio, text: str, language: str, max_audio_s: float) -> list:
    import soundfile

    duration = audio.size / float(SAMPLE_RATE)
    if duration > max_audio_s:
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
        try:
            os.unlink(wav_path)
        except OSError:
            pass

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


def _torch():
    import torch

    return torch


def align(request: dict) -> None:
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
            send("result", error=f"{type(exc).__name__}: {exc}")
        else:
            send("result", items=items)
        memory_line(_torch(), f"chunk {position}")
        report(position + 1)

    send("done")


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
            fail(f"{type(exc).__name__}: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
