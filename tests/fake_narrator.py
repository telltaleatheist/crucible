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

STDIN IS READ ON THE MAIN THREAD AND GENERATION RUNS ON ANOTHER, and that is not
an implementation detail — it is the property the streaming door is built on.
`electron/orpheus-worker-pool.ts` cancels mid-generation today, which means the
real worker acts on a `cancel` line **while a `generate_batch` is in flight**, so
its stdin loop cannot be the thing doing the generating. This file read and
generated on one thread until 2026-09-13, which made `{"action": "cancel"}` a
line that sat unread until the batch it was meant to abort had already finished:
a fake that could not be cancelled, testing a door whose whole subject is
cancelling. One generation runs at a time (narrator is one engine); a second
arriving while one is in flight is answered with an `error` rather than queued,
because Crucible converses once at a time and a fake that quietly tolerated two
would hide the day it stopped doing so.

A STREAMED BATCH INTERLEAVES ITS ROWS, one sub-sentence chunk each per turn. The
real engine emits `batch_chunk` lines from several rows at once — PHASE3-TTS.md
section 7 calls out "sub-sentence audio emitted *while a row is still
generating*" and "`done` for one row may arrive while another is still emitting"
as requirements — so a fake that finished each row before starting the next would
let a consumer that keys on arrival order alone pass. Rows are stepped in reverse
list order, which keeps the reverse-retirement property a non-streamed batch has
had since this file was written.

TWO FIELDS HERE ARE AHEAD OF THE REAL WIRE, and this is the honest place to say
so. `chars` and `capped` ride the retiring row below; `serve/worker.py` sends
neither, and the frame cap it computed (`HiggsBudget.cap_frames`, clamped by
`sgl_served.frame_cap`) never leaves the engine. They are here because
PHASE3-TTS.md section 6 puts `capped` on the `chunk` event and calls it the
difference between a long sentence and a runaway — so the render door reads them
when they are there and publishes `null` when they are not, which is what it will
do against the real narrator until narrator grows them. A test that asserts
`capped is True` is asserting about THIS file, not about narrator.

`guard` IS NOT one of them — it is on the real wire. `serve/worker.py`'s
`_emit_batch_item` puts `GuardPlan.verdict()`'s object on a retiring row of a
batch an engine guarded itself, and `_emit_guarded_batch` is the arm that
produces it (Owen's ruling of 2026-09-13, crucible/docs/PHASE6-REMOTE-RENDER.md
section 3). It is OPTIONAL there in exactly the way it is optional here: a row
with no verdict carries no `guard` key at all, which is what `CRUCIBLE_FAKE_GUARD`
reproduces by naming the rows that have one. This file does not model the
verdict's contents and must not start: the tests hand it a literal, because the
property being tested is that Crucible carries an object it does not understand
across unchanged.

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
    CRUCIBLE_FAKE_GUARD           a JSON OBJECT keyed by row index (as a string),
                                  whose value is that row's `guard` verdict,
                                  attached verbatim to its retiring `batch_item`.
                                  A row the object does not name carries no
                                  `guard` key at all — so one batch carries both
                                  cases, which is what "null means narrator did
                                  not say, and an absent key is not a missing
                                  measurement" needs to be tested against. Not
                                  valid JSON, or not an object: this process dies
                                  loudly rather than rendering a batch that
                                  silently ignored what a test asked for.
    CRUCIBLE_FAKE_FAIL_ROW        a row index (batch `i`, or the literal `0` for a
                                  single generate) that comes back as a per-item
                                  failure. Its neighbours must still succeed: "a failed
                                  chunk is reported and the run continues" is a rule in
                                  every one of these contracts and it needs a test.
    CRUCIBLE_FAKE_SAMPLING_LOG    a path this process appends one JSON line to per
                                  item it is asked to render:
                                  `{"i", "sampling", "take"}` — the rung exactly as
                                  it arrived, BOTH halves (null / absent when the
                                  item carried none). The ONLY way a test can see
                                  what Crucible put on the wire, because narrator
                                  does not echo a rung back and a fake that did
                                  would be a fake asserting about itself.
    CRUCIBLE_FAKE_NO_ITEM_TAKE    omit `itemTake` from the `ready` line, which is
                                  what a narrator built before
                                  `engine/item_sampling.py` looks like: it has no
                                  per-item rung channel at all — neither the
                                  numbers nor the seed lane — and DROPS both keys
                                  without a word. That is not hypothetical — the
                                  tts env was pinned to one on 2026-09-15 and two
                                  takes of one sentence came back byte-identical.
                                  Default: the key is sent, because this fake
                                  parses a rung.
    CRUCIBLE_FAKE_SAMPLING_LEVERS a comma-separated subset of the four levers this
                                  fake honours; anything else is
                                  `sampling_not_supported` for that ROW. Default:
                                  all four. It is narrator's own `levers` argument
                                  (`engine/item_sampling.py:parse_item_sampling`),
                                  which is how the MLX arm refuses a repetition
                                  penalty it has no sampler for.
                                  The failure shape is `{i, message}` — narrator's own
                                  (`serve/worker.py`), where a row that failed is told
                                  apart from one that worked by having a `message` and
                                  no `data`.
    CRUCIBLE_FAKE_ROW_DELAY_MS    milliseconds of REAL time one WHOLE (non-streamed)
                                  row costs. Default 0, because a fake that is slow
                                  by default makes every suite slow. It is the
                                  whole-row twin of CRUCIBLE_FAKE_CHUNK_DELAY_MS
                                  and exists for the same reason: the render door
                                  sends no `stream` flag anywhere, so a cancel of a
                                  RENDER has nowhere to land unless a row takes
                                  measurable time. Without it a cancel test on that
                                  door proves only that a batch nobody interrupted
                                  completes.
    CRUCIBLE_FAKE_IGNORE_CANCEL   read the cancel and keep rendering anyway — the
                                  real narrator, exactly, until 2026-09-15. Its
                                  stdin reader set the flag the moment a cancel
                                  landed and the arm Crucible's render door drives
                                  (`serve/worker.py:_emit_guarded_batch`) never read
                                  it, so a cancelled render of `thirdreich` went on
                                  retiring rows for eleven minutes with the door
                                  saying `cancelling` the whole time. A fake that
                                  cannot reproduce that cannot test the bound
                                  Crucible now puts on the cooperation
                                  (`engines/narrator.py:CANCEL_GRACE_SECONDS`).
                                  The `stopped` acknowledgement is still sent: this
                                  is an engine that HEARD and did not act, which is
                                  the failure, rather than one that went deaf.
    CRUCIBLE_FAKE_CHUNK_MS        milliseconds of audio per streamed sub-sentence
                                  chunk. Default 200.
    CRUCIBLE_FAKE_CHUNK_DELAY_MS  milliseconds of REAL time to spend on each of
                                  them. Default 0, because a fake that is slow by
                                  default makes every suite slow. Non-zero is how
                                  a row gets to still be generating while a test
                                  cancels it or drops its connection — the two
                                  things PHASE3-TTS.md section 7 exists for, and
                                  neither is reachable against an engine that
                                  finishes a paragraph in microseconds.
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
from typing import Callable, Iterator

SAMPLE_RATE = 24_000
TONE_HZ = 440.0

#: The `pads` and `edgeFadeMs` a real `loaded` line carries. Higgs v3 bakes no silence
#: into a chunk and another engine may; Crucible must not care, and carrying the fields
#: here is how "must not care" gets to be a tested claim rather than an assumption.
LOADED_PADS = {"head": 0.0, "tail": 0.0}
LOADED_EDGE_FADE_MS = {"in": 5, "out": 5}

_stdout_lock = threading.Lock()
_cancelled = threading.Event()

#: The one generation in flight, if any. narrator is one engine, so there is
#: never a second — see `_start_work`, which refuses rather than queues.
_worker: threading.Thread | None = None


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


def _guard_for(row: int | None) -> dict[str, object]:
    """`{"guard": ...}` for this row, or `{}` — never `{"guard": None}`.

    The empty dict is the point: the real worker omits the key entirely for a row
    whose engine reached no verdict (`_emit_batch_item`'s
    `**({'guard': guard} if guard is not None else {})`), and a fake that sent an
    explicit null instead would make Crucible's "absent means the same as null"
    path untestable by making it unreachable.

    A single `generate` (`row is None`) never carries one: the guarded arm is
    `generate_batch`'s, because a retake doubles the latency a listener is
    already waiting on and the interactive door was deliberately left alone.
    """
    raw = os.environ.get("CRUCIBLE_FAKE_GUARD")
    if raw is None or raw == "" or row is None:
        return {}
    verdicts = json.loads(raw)
    if not isinstance(verdicts, dict):
        raise TypeError(
            f"CRUCIBLE_FAKE_GUARD must be a JSON object keyed by row index, got "
            f"{type(verdicts).__name__}"
        )
    key = str(row)
    return {"guard": verdicts[key]} if key in verdicts else {}


#: narrator's per-item sampling vocabulary, verbatim from
#: `engine/item_sampling.py:WIRE_KEYS`. The document's camelCase, deliberately:
#: the per-load channel and the per-item channel are one set of levers and must
#: not grow two spellings.
_WIRE_KEYS = ("temperature", "topP", "topK", "repetitionPenalty")


def _levers() -> set[str]:
    raw = (os.environ.get("CRUCIBLE_FAKE_SAMPLING_LEVERS") or "").strip()
    if not raw:
        return set(_WIRE_KEYS)
    return {part.strip() for part in raw.split(",") if part.strip()}


#: The highest take narrator accepts (`engine/item_sampling.py:MAX_TAKE`).
_MAX_TAKE = 999


def _record_sampling(row: int | None, sampling: object, take: object = None) -> None:
    """Append what this item was asked to render under, for a test to read.

    BOTH HALVES OF THE RUNG. narrator's wire takes a `take` with no `sampling`
    beside it, and two takes whose numbers match are still two draws, so a log
    that recorded only `sampling` could not tell take 1 from take 0 — which is
    the exact defect the seed lane was added to fix on 2026-09-15.
    """
    path = (os.environ.get("CRUCIBLE_FAKE_SAMPLING_LOG") or "").strip()
    if not path:
        return
    with _stdout_lock:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"i": row, "sampling": sampling, "take": take}) + "\n"
            )


def _refuse_take_as_narrator_would(take: object, where: str) -> str | None:
    """narrator's own `take_malformed`, by name, for one item.

    `engine/item_sampling.py:parse_item_take` is the real one and this is its
    shape: a whole number >= 0 within `MAX_TAKE`. A bool is NOT a take
    (`isinstance(True, int)` is True in Python, and `take: true` from a client
    that meant `stream: true` would otherwise render take 1); a float is not,
    `2.0` included, because a take indexes a rung and rounding one would render
    the rung next door and report the one asked for.

    **None is not a refusal**: an item with no `take` renders at take 0, which
    is the seed rule narrator has always had. This fake has every lane, so
    `take_not_supported` is Orpheus's answer and never reachable here.
    """
    if take is None:
        return None
    if isinstance(take, bool) or not isinstance(take, int):
        return (
            f"take_malformed: {where} carries take {take!r} "
            f"({type(take).__name__}); a take is a whole number >= 0 naming a "
            "rung of the voice's ladder, and 0 (or no key at all) is take 0."
        )
    if take < 0:
        return (
            f"take_malformed: {where} carries take {take!r}. A take names a rung "
            "of the ladder and counts up from 0; there is no rung below take 0."
        )
    if take > _MAX_TAKE:
        return (
            f"take_malformed: {where} carries take {take!r}, above MAX_TAKE "
            f"({_MAX_TAKE})."
        )
    return None


def _refuse_sampling_as_narrator_would(sampling: object, where: str) -> str | None:
    """narrator's own per-item refusals, by name, for one item.

    `engine/item_sampling.py` is the real one and this is its shape: a rung is
    a non-empty object of known levers with positive numeric values (`topK` a
    whole one), a key outside the vocabulary is `sampling_malformed`, and a
    key this ENGINE has no lever for is `sampling_not_supported` — one is a
    typo and the other is the wrong backend, and the difference matters to
    whoever sent it. **None is not a refusal**: an item with no `sampling`
    renders at the voice's loaded default, which is take 0.
    """
    if sampling is None:
        return None
    if not isinstance(sampling, dict) or not sampling:
        return (
            f"sampling_malformed: {where} carries sampling {sampling!r}; a rung "
            f"is a non-empty object with any of {', '.join(sorted(_WIRE_KEYS))}."
        )
    unknown = sorted(set(sampling) - set(_WIRE_KEYS))
    if unknown:
        return (
            f"sampling_malformed: {where} carries sampling key(s) {unknown}; the "
            f"levers are {sorted(_WIRE_KEYS)}."
        )
    unsupported = sorted(set(sampling) - _levers())
    if unsupported:
        return (
            f"sampling_not_supported: {where} asks for sampling {unsupported}, "
            "which this engine has no lever for; it honours "
            f"{sorted(_levers())}."
        )
    for key, value in sampling.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            return (
                f"sampling_malformed: {where} carries sampling {key}={value!r}, "
                "which is not a positive number."
            )
        if key == "topK" and int(value) != value:
            return (
                f"sampling_malformed: {where} carries sampling topK={value!r}; "
                "top_k is a whole number of candidates."
            )
    return None


def _refused_this_row(item: dict) -> bool:
    """Answer one item's rung the way `_resolve_row` does, per ROW.

    A per-item refusal fails THE ONE ITEM and not the batch — narrator's own
    rule, and the reason `_resolve_row` exists at all ("one stray voice used to
    fail all 16 sentences"). It reaches the client as a `batch_item` carrying
    `message` and no `data`, which is how every per-row failure travels.
    """
    row = item.get("i")
    sampling = item.get("sampling")
    take = item.get("take")
    _record_sampling(row, sampling, take)
    where = f"generate_batch row i={row!r}"
    refusal = (_refuse_sampling_as_narrator_would(sampling, where)
               or _refuse_take_as_narrator_would(take, where))
    if refusal is None:
        return False
    send("batch_item", i=row, message=refusal)
    return True


def _told_to_fail(row: int | None) -> bool:
    """Answer `CRUCIBLE_FAKE_FAIL_ROW` for this row, and say whether it did."""
    fail_row = _env_int("CRUCIBLE_FAKE_FAIL_ROW")
    position = 0 if row is None else row
    if fail_row is None or fail_row != position:
        return False
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
        send("batch_item", i=row, message=f"fake narrator was told to fail row {row}")
    return True


def _emit_whole_row(text: str, row: int | None) -> None:
    """One row answered in a single message — the render door's shape."""
    if _told_to_fail(row):
        return
    delay = _env_float("CRUCIBLE_FAKE_ROW_DELAY_MS", 0.0) / 1000.0
    if delay > 0:
        time.sleep(delay)
    seconds, capped, chars = _duration_for(text)
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
        send("batch_item", i=row, **fields, **_guard_for(row))


def _stream_row(text: str, row: int | None) -> Iterator[None]:
    """One streamed row, advanced **one sub-sentence chunk per `next()`**.

    A generator rather than a loop so that a batch can interleave its rows; the
    retirement message is sent as the generator finishes, which is what makes a
    short row retire while a long one is still emitting.
    """
    if _told_to_fail(row):
        return

    seconds, capped, chars = _duration_for(text)
    chunk_seconds = _env_float("CRUCIBLE_FAKE_CHUNK_MS", 200.0) / 1000.0
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
        delay = _env_float("CRUCIBLE_FAKE_CHUNK_DELAY_MS", 0.0) / 1000.0
        if delay > 0:
            time.sleep(delay)
        yield

    cancelled = _cancelled.is_set()
    emitted = seconds - max(0.0, remaining)
    if row is None:
        send("done", duration=emitted, chunks=seq, cancelled=cancelled,
             chars=chars, capped=capped and not cancelled)
    else:
        send("batch_item", i=row, streamed=True, duration=emitted, chunks=seq,
             cancelled=cancelled, chars=chars, capped=capped and not cancelled)


def _interleave(rows: list[Iterator[None]]) -> None:
    """Step every live row once per turn, until none is left."""
    live = list(rows)
    while live:
        still: list[Iterator[None]] = []
        for row in live:
            try:
                next(row)
            except StopIteration:
                continue
            still.append(row)
        live = still


def _ignores_the_cancel() -> bool:
    """Is this process standing in for a narrator whose ladder never read the flag?

    See CRUCIBLE_FAKE_IGNORE_CANCEL in the module docstring. It gates only the
    WHOLE-row arm, which is the render door's, because that is the arm the real
    hole was in — the streamed arm honoured its cancel then and honours it now,
    and a knob that broke both would be modelling a narrator that never existed.
    """
    return os.environ.get("CRUCIBLE_FAKE_IGNORE_CANCEL") == "1"


def _run_generate(message: dict) -> None:
    text = message.get("text", "")
    if message.get("stream"):
        _interleave([_stream_row(text, None)])
    else:
        _emit_whole_row(text, None)


def _run_batch(message: dict) -> None:
    items = message.get("items") or []
    # Reversed, deliberately. Rows come back out of order within a batch on the
    # real engine (a short row finishes while a long one is still going), and a
    # consumer that quietly relies on arrival order is a consumer that will be
    # wrong the first time it meets a real one. For a streamed batch reversal
    # decides only which row wins a tie, because the rows are interleaved.
    ordered = list(reversed(items))
    # THE RUNG IS ANSWERED FIRST, per item, before anything is rendered —
    # `_resolve_row`'s position in the real worker, and for its reason: a rung
    # this engine cannot honour must fail its own row while its neighbours are
    # ordinary sentences. A refused row is already answered, so it never
    # reaches the generators below.
    ordered = [item for item in ordered if not _refused_this_row(item)]
    streamed = [item for item in ordered if item.get("stream")]
    whole = [item for item in ordered if not item.get("stream")]

    for item in whole:
        if _cancelled.is_set() and not _ignores_the_cancel():
            send("batch_item", i=item.get("i"), message="cancelled")
            continue
        _emit_whole_row(item.get("text", ""), item.get("i"))

    if streamed:
        _interleave([_stream_row(i.get("text", ""), i.get("i")) for i in streamed])
    send("batch_done")


#: Which of narrator's engines this process is standing in for — `--engine` on
#: the argv, exactly where `FakeNarratorEngine.command()` puts it.
_engine = ""


def _refuse_load_as_narrator_would(message: dict) -> str | None:
    """narrator's OWN refusals at the `load` message, for the Higgs v3 arms.

    Both are real and both were hit by real renders on 2026-09-14. narrator's
    `HiggsV3Engine.resolve_load_voice` and `HiggsV3MlxEngine.resolve_load_voice`
    refuse a `modelDir` by name — the served model is the launch script's
    argument, the MLX model is the voice document's — and then look the voice
    up in the JSON document `NARRATOR_HIGGS_VOICES` names, refusing an unset
    variable, a missing file and an absent voice each by name
    (`engine/higgs/config.py`: `voices_path`, `load_voice`). The messages are
    narrator's, near enough that a Crucible-side test asserting on one would
    also match the real worker's.

    An engine outside `DOCUMENT_READERS` keeps the message narrator's wire
    always had: `modelDir` on the load, no
    document at all.
    """
    if _engine != "higgs-v3":
        return None
    model_dir = message.get("modelDir")
    if model_dir:
        return (
            f"Higgs v3 load carried modelDir={model_dir!r}. The served model is "
            "the launch script's argument, not a per-load field."
        )
    path = (os.environ.get("NARRATOR_HIGGS_VOICES") or "").strip()
    if not path:
        return (
            "NARRATOR_HIGGS_VOICES is not set. A Higgs voice is reference clips "
            "plus their book-exact transcripts, which cannot be passed as a "
            "voice name; point NARRATOR_HIGGS_VOICES at the JSON document that "
            "defines them."
        )
    if not os.path.isfile(path):
        return f"NARRATOR_HIGGS_VOICES points at {path}, which does not exist."
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    name = (message.get("voice") or "").strip()
    if name not in document:
        return (
            f"Higgs voice '{name}' is not in {path}. It defines: "
            f"{', '.join(sorted(document))}."
        )
    entry = document[name]
    if entry.get("kind") == "checkpoint" and not entry.get("checkpointDir"):
        return (
            f"{path}: voice '{name}' is kind 'checkpoint' with no "
            "'checkpointDir'. The checkpoint IS the voice - there is nothing to "
            "serve without it."
        )
    if entry.get("kind") == "checkpoint" and entry.get("maxChars") is None:
        return (
            f"{path}: voice '{name}' is a fine-tune (kind 'checkpoint') and "
            "carries no 'maxChars'."
        )
    # A CLIPS VOICE'S CLIPS ARE FILES ON THIS HOST, and narrator checks that
    # they exist — `config.load_voices` calls `os.path.isfile` on every
    # `clips[].path` and refuses a clip with no `path` or no `transcript`
    # before any of them. So does this: a Crucible that wrote the document but
    # not the wav, or wrote a clip with no transcript, fails in the suite
    # rather than on somebody's book. (The transcript refusal is narrator's own
    # law — a clone conditioned on an absent transcript is a whole book in a
    # subtly wrong voice, reported as success.)
    if entry.get("kind") == "clips" and not entry.get("clips"):
        return (
            f"{path}: voice '{name}' is a reference clone with no clips. A "
            "zero-shot clone with no reference is the model's own voice, which "
            "is a different thing."
        )
    for clip in entry.get("clips") or []:
        for key in ("path", "transcript"):
            if not (clip.get(key) or "").strip():
                return (
                    f"{path}: voice '{name}' has a clip with no '{key}'. The "
                    "transcript is the book-exact text spoken in the clip - the "
                    "corpus row or the narration copy, never a transcription."
                )
        if not os.path.isfile(clip["path"]):
            return (
                f"{path}: voice '{name}' names a clip that does not exist: "
                f"{clip['path']}"
            )
    return None


def _handle(message: dict) -> bool:
    """Act on one line. Returns False when the process should exit.

    Runs on the **stdin thread**, so everything here must be prompt: the whole
    point of this file's shape is that a `cancel` is acted on while a batch is
    still generating on the worker thread.
    """
    action = message.get("action")

    if action == "load":
        refusal = _refuse_load_as_narrator_would(message)
        if refusal is not None:
            send("error", message=refusal)
            return True
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
        _start_work(_run_generate, message)
        return True

    if action == "generate_batch":
        _start_work(_run_batch, message)
        return True

    if action in ("cancel", "stop"):
        _cancelled.set()
        send("stopped")
        return True

    if action == "quit":
        return False

    send("error", message=f"fake narrator does not know action {action!r}")
    return True


def _start_work(target: Callable[[dict], None], message: dict) -> None:
    """Hand one generation to the worker thread, refusing a second in flight.

    `_cancelled` is cleared **here**, on the stdin thread, before the worker
    starts — so a `cancel` that arrives after this line can never be cleared by
    the generation it was sent to abort.
    """
    global _worker
    if _worker is not None and _worker.is_alive():
        send(
            "error",
            message=(
                "fake narrator was sent a second generation while one was in "
                "flight; narrator is one engine and Crucible converses once at "
                "a time"
            ),
        )
        return
    _cancelled.clear()
    _worker = threading.Thread(
        target=target, args=(message,), name="fake-narrator-work", daemon=True
    )
    _worker.start()


def main() -> int:
    global _engine
    if "--engine" in sys.argv:
        _engine = sys.argv[sys.argv.index("--engine") + 1]

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

    # `itemTake` by default, because this fake DOES parse a rung — both halves,
    # `_refuse_sampling_as_narrator_would` and `_refuse_take_as_narrator_would` —
    # and a handshake that hid that would be a fake lying about itself. ONE key
    # for the two facts, which is narrator's own shape since 2026-09-15: a build
    # has both or neither. Withheld on demand, which is the only way to test the
    # narrator that caused this: one pinned a day before
    # `engine/item_sampling.py`, which dropped every rung in silence.
    if os.environ.get("CRUCIBLE_FAKE_NO_ITEM_TAKE") == "1":
        send("ready", device="fake", backend="fake")
    else:
        send("ready", device="fake", backend="fake", itemTake=True)

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
