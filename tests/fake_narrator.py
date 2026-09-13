"""A stand-in for `python -m narrator.serve`, faithful to its JSON-lines protocol.

`tests/fake_engine.py` exists because the only things Crucible needs from vLLM are a
process it can start and SIGTERM and an HTTP surface. This file is the same idea for
`tts`, and it has to work harder, because narrator's wire is not HTTP and not
request/response: it is newline-delimited JSON over stdin and stdout, it answers a
`ready` line on stdout rather than a route, it streams sub-sentence audio while a row
is still generating, and it retires rows out of order within a batch.

Every message shape below is copied from `python/narrator/serve/worker.py`'s module
docstring in the BookForge repo, which that file states is unchanged from the wire
`electron/orpheus-worker-pool.ts` has always parsed. Where this file simplifies, it
simplifies by doing *less* — it never invents a field narrator does not send, because
a fake that is generous is a fake that lets a real bug through.

    stdin   {"action": "load", "voice": ..., "caps": {...}, "warm": true}
            {"action": "generate", "text": ..., "stream": true}
            {"action": "generate_batch", "items": [{"i": 0, "text": ..., "stream": true}]}
            {"action": "cancel" | "stop" | "quit"}

    stdout  {"type": "ready", "device": ..., "backend": ...}
            {"type": "loaded", "voice", "backend", "engine", "sampleRate", "pads", "edgeFadeMs"}
            {"type": "chunk", "seq", "format": "pcm16", "data", "duration", "sampleRate"}
            {"type": "done", "duration", "chunks", "cancelled"}
            {"type": "batch_chunk", "i", "seq", "format": "pcm16", "data", "duration", "sampleRate"}
            {"type": "batch_item", "i", "duration", "chunks"}
            {"type": "batch_done"}
            {"type": "audio", "format": "pcm16", "data", "duration", "sampleRate"}
            {"type": "error", "message"}
            {"type": "stopped"}

TWO FIELDS HERE ARE AHEAD OF THE REAL WIRE, and this is the honest place to say
so. `chars` and `capped` ride the retiring row below; `serve/worker.py` sends
neither, and the frame cap it computed (`HiggsBudget.cap_frames`, clamped by
`sgl_served.frame_cap`) never leaves the engine. They are here because
PHASE3-TTS.md section 6 puts `capped` on the `chunk` event and calls it the
difference between a long sentence and a runaway — so the render door reads them
when they are there and publishes `null` when they are not, which is what it will
do against the real narrator until narrator grows them. A test that asserts
`capped is True` is asserting about THIS file, not about narrator.

Run it exactly as Crucible will run the real thing — as an argv, from a subprocess:

    [sys.executable, str(FAKE_NARRATOR), "--engine", "higgs-v3"]

WHAT IT LETS A TEST DO TO IT. Everything steerable is an environment variable rather
than a flag, because the thing under test is `Engine.start()`, which owns the argv and
must not be bent into passing test fixtures through it:

    CRUCIBLE_FAKE_READY_DELAY_S   seconds before the `ready` line. Default 0.
                                  Non-zero is how a `warming` stream gets something to
                                  stream.
    CRUCIBLE_FAKE_READY_NEVER     print no `ready` at all and sit there, so the
                                  readiness timeout is a real timeout and not a crash.
    CRUCIBLE_FAKE_EXIT_CODE       exit with this code immediately. How a load that
                                  dies before it is ready gets tested — the commonest
                                  real failure on `cuda-linux`, where two of the three
                                  blockers found in phase 2 were environment facts that
                                  killed the engine seconds in.
    CRUCIBLE_FAKE_CHARS_PER_SEC   how much audio a chunk's characters are worth.
                                  Default 15.0, which is Higgs's configured pace. This
                                  is what makes an assertion about a reported
                                  `chars_per_sec` mean something.
    CRUCIBLE_FAKE_CAP_CHARS       a chunk with more characters than this stops at the
                                  cap and reports `capped`. How a runaway is tested
                                  without a model that runs away.
    CRUCIBLE_FAKE_FAIL_ROW        a row index (batch `i`, or the literal `0` for a
                                  single generate) that comes back as a per-item
                                  failure. Its neighbours must still succeed: "a failed
                                  chunk is reported and the run continues" is a rule in
                                  every one of these contracts and it needs a test.
                                  The failure shape is `{i, message}` — narrator's own
                                  (`serve/worker.py`), where a row that failed is told
                                  apart from one that worked by having a `message` and
                                  no `data`.
    CRUCIBLE_FAKE_CHUNK_MS        milliseconds of audio per streamed sub-sentence
                                  chunk. Default 200.
    CRUCIBLE_FAKE_IGNORE_SIGTERM  ignore SIGTERM, so `stop()`'s refusal to escalate to
                                  SIGKILL can be tested. Crucible never SIGKILLs a
                                  process holding CUDA — that wedges WSL until Windows
                                  reboots — so the timeout path is a real path and not
                                  a theoretical one. A test that sets this MUST kill
                                  the process itself afterwards.
    CRUCIBLE_FAKE_TRANSCRIPT      a path. Every line this process reads on stdin is
                                  appended to it verbatim, so a test can assert on what
                                  Crucible actually sent rather than on what it meant
                                  to send.

The audio is a 440 Hz sine at 24 kHz mono PCM16 — the sample rate every real engine on
this path uses, so a test that decodes it is exercising the same arithmetic a FLAC
writer will.
"""

from __future__ import annotations

import base64
import json
import math
import os
import signal
import struct
import sys
import threading
import time

SAMPLE_RATE = 24_000
TONE_HZ = 440.0

#: The `pads` and `edgeFadeMs` a real `loaded` line carries. Higgs v3 bakes no silence
#: into a chunk and Orpheus does; Crucible must not care, and carrying the fields here
#: is how "must not care" gets to be a tested claim rather than an assumption.
LOADED_PADS = {"head": 0.0, "tail": 0.0}
LOADED_EDGE_FADE_MS = {"in": 5, "out": 5}

_stdout_lock = threading.Lock()
_cancelled = threading.Event()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    return None if raw is None or raw == "" else int(raw)


def send(message_type: str, **fields: object) -> None:
    """One JSON object, one line, flushed. The lock is narrator's own discipline.

    Streamed chunks are emitted from the same thread as everything else here, but
    the real worker has a reader thread and holds a stdout lock for exactly this
    reason: two half-written lines interleaved are not two messages, they are a
    protocol error that looks like corrupted audio.
    """
    line = json.dumps({"type": message_type, **fields})
    with _stdout_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def tone(seconds: float, phase: float = 0.0) -> tuple[bytes, float]:
    """`seconds` of 24 kHz mono PCM16, and the phase to continue from.

    The phase is threaded through so consecutive sub-sentence chunks of one row
    concatenate into a continuous tone. A test that joins the chunks and finds a
    discontinuity has found a chunk Crucible dropped or reordered, which is the whole
    point of streaming them separately.
    """
    count = max(1, int(SAMPLE_RATE * seconds))
    step = 2.0 * math.pi * TONE_HZ / SAMPLE_RATE
    samples = bytearray()
    for index in range(count):
        value = int(20_000 * math.sin(phase + index * step))
        samples += struct.pack("<h", value)
    return bytes(samples), phase + count * step


def _duration_for(text: str) -> tuple[float, bool, int]:
    """Seconds of audio this text is worth, whether it hit the cap, and its length.

    A real engine's duration is a property of the tokens it generated. Here it is a
    property of the characters, which is enough for every measurement Crucible
    reports — duration, characters, characters per second, capped — to be a number a
    test can predict exactly.
    """
    chars = len(text)
    cap = _env_int("CRUCIBLE_FAKE_CAP_CHARS")
    capped = cap is not None and chars > cap
    spoken = cap if capped else chars
    return spoken / _env_float("CRUCIBLE_FAKE_CHARS_PER_SEC", 15.0), capped, chars


def _emit_row(text: str, streamed: bool, row: int | None) -> None:
    """One row of work: either streamed sub-sentence chunks, or one whole answer.

    `row` is the batch position, or None for a bare `generate`. It decides the message
    names (`batch_chunk`/`batch_item` against `chunk`/`done`) exactly as the real
    worker does, and nothing else about the work differs between the two shapes.
    """
    fail_row = _env_int("CRUCIBLE_FAKE_FAIL_ROW")
    position = 0 if row is None else row
    if fail_row is not None and fail_row == position:
        if row is None:
            send("error", message=f"fake narrator was told to fail row {position}")
        else:
            # `message`, not `error`. Corrected 2026-09-13 while the render door
            # was being built against this file: `serve/worker.py` reports a
            # per-row failure as `{'i': ..., 'message': ...}` in all five of the
            # places it can happen ('No audio generated', 'cancelled',
            # 'Model not loaded', the row's own exception text, and 'Batch
            # generation failed'), and it is the ABSENCE of `data` plus the
            # PRESENCE of `message` that tells a consumer a row failed. A fake
            # sending a key narrator never sends is a fake that lets a real bug
            # through, which is this file's own rule.
            send("batch_item", i=row,
                 message=f"fake narrator was told to fail row {row}")
        return

    seconds, capped, chars = _duration_for(text)
    chunk_seconds = _env_float("CRUCIBLE_FAKE_CHUNK_MS", 200.0) / 1000.0

    if not streamed:
        payload, _ = tone(seconds)
        fields = {
            "format": "pcm16",
            "data": base64.b64encode(payload).decode("ascii"),
            "duration": seconds,
            "sampleRate": SAMPLE_RATE,
            "chars": chars,
            "capped": capped,
        }
        if row is None:
            send("audio", **fields)
        else:
            send("batch_item", i=row, **fields)
        return

    phase = 0.0
    seq = 0
    remaining = seconds
    while remaining > 1e-9:
        if _cancelled.is_set():
            # A cancel mid-row is the streaming door's whole reason for existing, and
            # the real worker answers it the same way: the row closes where it is and
            # says it was cancelled. It does not pretend to have finished.
            break
        span = min(chunk_seconds, remaining)
        payload, phase = tone(span, phase)
        fields = {
            "seq": seq,
            "format": "pcm16",
            "data": base64.b64encode(payload).decode("ascii"),
            "duration": span,
            "sampleRate": SAMPLE_RATE,
        }
        if row is None:
            send("chunk", **fields)
        else:
            send("batch_chunk", i=row, **fields)
        seq += 1
        remaining -= span

    cancelled = _cancelled.is_set()
    emitted = seconds - max(0.0, remaining)
    if row is None:
        send("done", duration=emitted, chunks=seq, cancelled=cancelled,
             chars=chars, capped=capped and not cancelled)
    else:
        send("batch_item", i=row, streamed=True, duration=emitted, chunks=seq,
             cancelled=cancelled, chars=chars, capped=capped and not cancelled)


def _handle(message: dict) -> bool:
    """Act on one line. Returns False when the process should exit."""
    action = message.get("action")

    if action == "load":
        send(
            "loaded",
            voice=message.get("voice"),
            backend="fake",
            engine="fake",
            sampleRate=SAMPLE_RATE,
            pads=LOADED_PADS,
            edgeFadeMs=LOADED_EDGE_FADE_MS,
        )
        return True

    if action == "generate":
        _cancelled.clear()
        _emit_row(message.get("text", ""), bool(message.get("stream")), None)
        return True

    if action == "generate_batch":
        _cancelled.clear()
        items = message.get("items") or []
        # Retired in reverse, deliberately. Rows come back out of order within a batch
        # on the real engine (a short row finishes while a long one is still going),
        # and a consumer that quietly relies on arrival order is a consumer that will
        # be wrong the first time it meets a real one.
        for item in reversed(items):
            if _cancelled.is_set():
                send("batch_item", i=item.get("i"), message="cancelled")
                continue
            _emit_row(item.get("text", ""), bool(item.get("stream")), item.get("i"))
        send("batch_done")
        return True

    if action in ("cancel", "stop"):
        _cancelled.set()
        send("stopped")
        return True

    if action == "quit":
        return False

    send("error", message=f"fake narrator does not know action {action!r}")
    return True


def main() -> int:
    exit_code = _env_int("CRUCIBLE_FAKE_EXIT_CODE")
    if exit_code is not None:
        # Before the `ready` line and before reading a byte: an engine that dies during
        # its own start-up, which on `cuda-linux` is the commonest failure there is.
        sys.stderr.write("fake narrator: told to exit before becoming ready\n")
        return exit_code

    if os.environ.get("CRUCIBLE_FAKE_IGNORE_SIGTERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    transcript_path = os.environ.get("CRUCIBLE_FAKE_TRANSCRIPT")

    delay = _env_float("CRUCIBLE_FAKE_READY_DELAY_S", 0.0)
    if delay > 0:
        # On stderr, because the real worker writes its progress there and Crucible's
        # `warming` messages are the tail of the engine log. A test that asserts on a
        # warming message needs something to be in that log before `ready`.
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            sys.stderr.write("fake narrator: loading\n")
            sys.stderr.flush()
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    if os.environ.get("CRUCIBLE_FAKE_READY_NEVER") == "1":
        while True:
            time.sleep(0.1)

    send("ready", device="fake", backend="fake")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if transcript_path:
            with open(transcript_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            send("error", message=f"fake narrator could not read a line: {exc}")
            continue
        if not _handle(message):
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
