"""The wire and the decode every job worker shares.

A worker runs in its own env (`asr`, `align`, `llm`, `rvc`, `mlx-audio`), and
none of those has `crucible` installed, so this module imports nothing but the
standard library at load; numpy is imported inside `decode`, which only the
workers that decode call. A worker loads it by path from the directory above
its own (`crucible/jobs/`), and never leaves that directory on `sys.path`:
`crucible/jobs/queue.py` would shadow the standard library's `queue`.

fd 1 is results and nothing else
--------------------------------
`claim_stdout` dups fd 1 somewhere safe and points the original at stderr,
before any library is imported. A library that prints once to stdout would
otherwise land between two result lines — which is exactly how narrator's
aligner lost a 401-chunk book on 2026-09-05 (the parent's `json.loads` died
with "Extra data").
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import threading
import time

#: Every decoding worker's models work at 16 kHz mono. Not a parameter: it is
#: the rate their feature extractors were trained at.
SAMPLE_RATE = 16_000

#: How often the decode phase reports, in seconds of wall clock. An 18-hour book
#: is minutes of CPU-only decode before the first transcription line, and a bar
#: that does not move during it looks like a hang.
DECODE_REPORT_SECONDS = 1.0

#: fd 1, once `claim_stdout` has moved it out of every library's reach.
_RESULTS = None

#: `send` may be called from a heartbeat thread as well as the main one
#: (denoise), and two writes interleaved on one line would be a line the server
#: cannot parse.
_SEND_LOCK = threading.Lock()


def claim_stdout() -> None:
    """fd 1 is results, stderr is everything else. Idempotent."""
    global _RESULTS
    if _RESULTS is not None:
        return
    results_fd = os.dup(1)
    os.dup2(2, 1)
    _RESULTS = os.fdopen(results_fd, "w", encoding="utf-8", buffering=1)


def send(message_type: str, **fields: object) -> None:
    """One JSON object, one line, flushed, on the real fd 1."""
    line = json.dumps({"type": message_type, **fields}) + "\n"
    with _SEND_LOCK:
        _RESULTS.write(line)
        _RESULTS.flush()


def fail(message: str) -> int:
    send("failed", message=message)
    return 1


def require(request: dict, key: str, kind, label: str, why: str):
    """One required key of the stated type, or a KeyError naming it.

    `label` names the request (`"the {label} request"`) and `why` is the
    worker's sentence for a missing key. A bool where a number is wanted passes
    `isinstance` (bool is an int) and is still refused.
    """
    if key not in request:
        raise KeyError(f"the {label} request has no {key!r}; {why}")
    value = request[key]
    kinds = kind if isinstance(kind, tuple) else (kind,)
    wrong = not isinstance(value, kinds) or (
        isinstance(value, bool) and bool not in kinds
    )
    if wrong:
        names = "/".join(getattr(k, "__name__", str(k)) for k in kinds)
        raise KeyError(
            f"the {label} request's {key!r} must be {names}, got "
            f"{type(value).__name__}"
        )
    return value


def load_sibling(name: str, beside: str):
    """Module `name` from the directory holding the file `beside`."""
    here = os.path.dirname(os.path.abspath(beside))
    if here not in sys.path:
        sys.path.insert(0, here)
    return importlib.import_module(name)


# ------------------------------------------------------------------- decoding


def probe_duration(ffmpeg: str, audio_path: str, label: str) -> float:
    """Container duration in seconds, via the ffprobe beside ffmpeg.

    The denominator for decode progress and nothing else — the decode itself
    never depends on it. A container that carries no duration gets 0.0 and a
    progress line with no percentage, which is still honest.
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
        print(f"[{label}] ffprobe unavailable ({exc}); decode progress has no total")
        return 0.0
    if completed.returncode != 0:
        print(
            f"[{label}] ffprobe exited {completed.returncode}; decode progress has no total"
        )
        return 0.0
    text = completed.stdout.decode("utf-8", "replace").strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        print(f"[{label}] ffprobe said {text!r}, which is not a duration")
        return 0.0


def decode_reporter(**fields: object):
    """An `on_progress` for `decode` that sends a `decoding` frame at most every
    DECODE_REPORT_SECONDS, carrying `fields` after `processed_s`."""
    last = [0.0]

    def report(seconds: float) -> None:
        now = time.time()
        if now - last[0] < DECODE_REPORT_SECONDS:
            return
        last[0] = now
        send("progress", stage="decoding", processed_s=round(seconds, 1), **fields)

    return report


def decode(ffmpeg: str, audio_path: str, on_progress=None):
    """Any container -> mono float32 at 16 kHz, through ffmpeg. Raises on failure.

    ffmpeg rather than a python decoder: PyAV silently truncates some assembled
    m4b files, which ends a transcript hours early with no error, and ffmpeg
    reads every container the renderers write identically. `-f f32le` is
    already normalised to [-1, 1]. Streamed so `on_progress(seconds)` can fire
    as bytes arrive (one audio second is `SAMPLE_RATE * 4` bytes, so the
    position is exact); stderr is drained on a thread so an error-spewing
    decode cannot fill the pipe and deadlock.
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
    per_second = SAMPLE_RATE * 4
    while True:
        block = process.stdout.read(1 << 20)
        if not block:
            break
        buffer += block
        if on_progress is not None:
            on_progress(len(buffer) / per_second)
    process.stdout.close()
    code = process.wait()
    pump.join(timeout=5)
    if code != 0:
        tail = b"".join(errors).decode("utf-8", "replace").strip()[-500:]
        raise RuntimeError(f"ffmpeg exited {code} on {audio_path}: {tail}")
    # A view of the buffer, not a copy: the array is gigabytes on a long book.
    return numpy.frombuffer(buffer, dtype=numpy.float32)


# --------------------------------------------------------- the torch allocator
#
# Crucible's `workers` note has the measurement. On CUDA the server spawns a
# torch worker with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and sends
# its admitted share as `memory_cap_bytes`. A worker held across inputs of
# different lengths is how a caching allocator strands blocks until the card
# spills into system memory.


def cap_memory(torch, memory_cap_bytes):
    """Cap this process's CUDA reservation at its admitted share, before any weights.

    `None` is not CUDA (the server sends a cap only there), so nothing is set.
    Past the cap the allocator frees its cache and retries; a true overrun is an
    OOM naming the fraction, in this job's report, not a card paging the host.
    """
    if memory_cap_bytes is None:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"a CUDA memory cap of {memory_cap_bytes} bytes was sent and this "
            "process sees no CUDA device"
        )
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, memory_cap_bytes / total)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return fraction


def memory_line(torch, label):
    """allocated / peak / reserved on CUDA to the engine log, then a fresh peak."""
    if not torch.cuda.is_available():
        return
    gib = 1024 ** 3
    print(
        f"crucible memory {label}: allocated "
        f"{torch.cuda.memory_allocated(0) / gib:.2f} GiB, peak "
        f"{torch.cuda.max_memory_allocated(0) / gib:.2f} GiB, reserved "
        f"{torch.cuda.memory_reserved(0) / gib:.2f} GiB",
        file=sys.stderr,
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats(0)
