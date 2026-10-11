from __future__ import annotations

import collections
import importlib
import json
import os
import subprocess
import sys
import threading
import time

SAMPLE_RATE = 16_000

DECODE_REPORT_SECONDS = 1.0

_RESULTS = None

_SEND_LOCK = threading.Lock()


def claim_stdout() -> None:
    global _RESULTS
    if _RESULTS is not None:
        return
    results_fd = os.dup(1)
    os.dup2(2, 1)
    _RESULTS = os.fdopen(results_fd, "w", encoding="utf-8", buffering=1)


def send(message_type: str, **fields: object) -> None:
    line = json.dumps({"type": message_type, **fields}) + "\n"
    with _SEND_LOCK:
        _RESULTS.write(line)
        _RESULTS.flush()


def fail(message: str) -> int:
    send("failed", message=message)
    return 1


def _parsed(line: str, label: str):
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        fail(f"the {label} request is not JSON: {exc}")
        return None
    if not isinstance(request, dict):
        fail(f"the {label} request must be a JSON object, got {type(request).__name__}")
        return None
    return request


def read_request(label: str):
    line = sys.stdin.readline()
    if not line.strip():
        fail(f"the {label} worker was given no request on stdin")
        return None
    return _parsed(line, label)


def _answer(line: str, label: str, ops: dict):
    request = _parsed(line, label)
    if request is None:
        return 1
    op = request.get("op")
    handler = ops.get(op)
    if handler is None:
        return fail(f"the {label} request's op is {op!r}; this worker takes {sorted(ops)}")
    try:
        handler(request)
    except KeyError as exc:
        return fail(str(exc.args[0]))
    except Exception as exc:
        return fail(f"{type(exc).__name__}: {exc}")
    return None


class _Pending:
    def __init__(self) -> None:
        self._lines: collections.deque = collections.deque()
        self._ready = threading.Condition()

    def put(self, line) -> None:
        with self._ready:
            self._lines.append(line)
            self._ready.notify()

    def get(self):
        with self._ready:
            while not self._lines:
                self._ready.wait()
            return self._lines.popleft()


def _interrupt_of(line: str, interrupts: dict):
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(request, dict):
        return None, None
    return interrupts.get(request.get("op")), request


def _feed(pending: _Pending, interrupts: dict) -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        handler, request = _interrupt_of(line, interrupts)
        if handler is None:
            pending.put(line)
        else:
            handler(request)
    pending.put(None)


def _lines(label: str, interrupts: dict | None):
    if not interrupts:
        return (line for line in sys.stdin if line.strip())
    pending = _Pending()
    threading.Thread(
        target=_feed, args=(pending, interrupts), name=f"{label}-stdin", daemon=True
    ).start()
    return iter(pending.get, None)


def serve(label: str, ops: dict, interrupts: dict | None = None) -> int:
    for line in _lines(label, interrupts):
        code = _answer(line, label, ops)
        if code is not None:
            return code
    return 0


def require(request: dict, key: str, kind, label: str, why: str):
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
    here = os.path.dirname(os.path.abspath(beside))
    if here not in sys.path:
        sys.path.insert(0, here)
    return importlib.import_module(name)


def probe_duration(ffmpeg: str, audio_path: str, label: str) -> float:
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
    last = [0.0]

    def report(seconds: float) -> None:
        now = time.time()
        if now - last[0] < DECODE_REPORT_SECONDS:
            return
        last[0] = now
        send("progress", stage="decoding", processed_s=round(seconds, 1), **fields)

    return report


def decode(ffmpeg: str, audio_path: str, on_progress=None):
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
    return numpy.frombuffer(buffer, dtype=numpy.float32)


# How every worker writes the audio it hands back (an audio job's song, a denoise job's
# stems): FLAC and WAV as 24-bit PCM, mp3 as 192 kbps CBR.
FLAC_SUBTYPE = "PCM_24"

WAV_SUBTYPE = "PCM_24"

# libsndfile's MP3 writer (LAME, in libsndfile >= 1.1) takes a compression level, not a
# bitrate. In CONSTANT mode 0.40-0.45 all give 192 kbps CBR - measured with ffprobe on
# libsndfile 1.2.2, the one both audio envs carry (0.3 gives 224 kbps, 0.5 160 kbps).
# 0.42 sits in the middle of that band.
MP3_CBR_192_LEVEL = 0.42


def write_audio(path: str, samples, sample_rate: int, audio_format: str) -> None:
    import soundfile

    if audio_format == "mp3":
        soundfile.write(
            path, samples, sample_rate, format="MP3", subtype="MPEG_LAYER_III",
            bitrate_mode="CONSTANT", compression_level=MP3_CBR_192_LEVEL,
        )
        return
    if audio_format not in ("flac", "wav"):
        raise ValueError(f"no encoding for {audio_format!r}; flac, wav or mp3")
    subtype = FLAC_SUBTYPE if audio_format == "flac" else WAV_SUBTYPE
    soundfile.write(path, samples, sample_rate, format=audio_format.upper(), subtype=subtype)


def cap_memory(torch, memory_cap_bytes):
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
