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

LOADED_PADS = {"head": 0.0, "tail": 0.0}
LOADED_EDGE_FADE_MS = {"in": 5, "out": 5}

_stdout_lock = threading.Lock()
_cancelled = threading.Event()

_worker: threading.Thread | None = None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    return None if raw is None or raw == "" else int(raw)


def _gap_fields() -> dict[str, object]:
    if (os.environ.get("CRUCIBLE_FAKE_OMIT_GAP") or "").strip():
        return {}
    return {"gapSec": _env_float("CRUCIBLE_FAKE_GAP_SEC", 0.6)}


def send(message_type: str, **fields: object) -> None:
    line = json.dumps({"type": message_type, **fields})
    with _stdout_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def tone(seconds: float, phase: float = 0.0) -> tuple[bytes, float]:
    count = max(1, int(SAMPLE_RATE * seconds))
    step = 2.0 * math.pi * TONE_HZ / SAMPLE_RATE
    samples = bytearray()
    for index in range(count):
        value = int(20_000 * math.sin(phase + index * step))
        samples += struct.pack("<h", value)
    return bytes(samples), phase + count * step


def _duration_for(text: str) -> tuple[float, bool, int]:
    chars = len(text)
    cap = _env_int("CRUCIBLE_FAKE_CAP_CHARS")
    capped = cap is not None and chars > cap
    spoken = cap if capped else chars
    return spoken / _env_float("CRUCIBLE_FAKE_CHARS_PER_SEC", 15.0), capped, chars


def _guard_for(row: int | None) -> dict[str, object]:
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


def _pause_cuts_for(row: int | None) -> dict[str, object]:
    raw = os.environ.get("CRUCIBLE_FAKE_PAUSE_CUTS")
    if raw is None or raw == "" or row is None:
        return {}
    cuts = json.loads(raw)
    key = str(row)
    return {"pauseCuts": cuts[key]} if key in cuts else {}


_WIRE_KEYS = ("temperature", "topP", "topK", "repetitionPenalty")


def _levers() -> set[str]:
    raw = (os.environ.get("CRUCIBLE_FAKE_SAMPLING_LEVERS") or "").strip()
    if not raw:
        return set(_WIRE_KEYS)
    return {part.strip() for part in raw.split(",") if part.strip()}


_MAX_TAKE = 999


def _record_sampling(row: int | None, sampling: object, take: object = None) -> None:
    path = (os.environ.get("CRUCIBLE_FAKE_SAMPLING_LOG") or "").strip()
    if not path:
        return
    with _stdout_lock:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"i": row, "sampling": sampling, "take": take}) + "\n"
            )


def _refuse_take_as_narrator_would(take: object, where: str) -> str | None:
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
    fail_row = _env_int("CRUCIBLE_FAKE_FAIL_ROW")
    position = 0 if row is None else row
    if fail_row is None or fail_row != position:
        return False
    if row is None:
        send("error", message=f"fake narrator was told to fail row {position}")
    else:
        send("batch_item", i=row, message=f"fake narrator was told to fail row {row}")
    return True


def _emit_whole_row(text: str, row: int | None) -> None:
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
        send("audio", **fields, **_gap_fields())
    else:
        send("batch_item", i=row, **fields, **_guard_for(row), **_pause_cuts_for(row))


def _stream_row(text: str, row: int | None) -> Iterator[None]:
    if _told_to_fail(row):
        return

    seconds, capped, chars = _duration_for(text)
    chunk_seconds = _env_float("CRUCIBLE_FAKE_CHUNK_MS", 200.0) / 1000.0
    phase = 0.0
    seq = 0
    remaining = seconds
    while remaining > 1e-9:
        if _cancelled.is_set():
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
             chars=chars, capped=capped and not cancelled, **_gap_fields())
    else:
        send("batch_item", i=row, streamed=True, duration=emitted, chunks=seq,
             cancelled=cancelled, chars=chars, capped=capped and not cancelled,
             **_gap_fields())


def _interleave(rows: list[Iterator[None]]) -> None:
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
    return os.environ.get("CRUCIBLE_FAKE_IGNORE_CANCEL") == "1"


def _run_generate(message: dict) -> None:
    text = message.get("text", "")
    if message.get("stream"):
        _interleave([_stream_row(text, None)])
    else:
        _emit_whole_row(text, None)


def _record_batch(message: dict) -> None:
    path = (os.environ.get("CRUCIBLE_FAKE_BATCH_LOG") or "").strip()
    if not path:
        return
    with _stdout_lock:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "language": message.get("language"),
                        "retake": message.get("retake"),
                        "band": message.get("band"),
                        "width": message.get("width"),
                        "keys": sorted(message),
                        "items": len(message.get("items") or []),
                    }
                )
                + "\n"
            )


def _run_batch(message: dict) -> None:
    _record_batch(message)
    items = message.get("items") or []
    ordered = list(reversed(items))
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


_engine = ""


def _refuse_load_as_narrator_would(message: dict) -> str | None:
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
        sys.stderr.write("fake narrator: told to exit before becoming ready\n")
        return exit_code

    if os.environ.get("CRUCIBLE_FAKE_IGNORE_SIGTERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)

    transcript_path = os.environ.get("CRUCIBLE_FAKE_TRANSCRIPT")

    delay = _env_float("CRUCIBLE_FAKE_READY_DELAY_S", 0.0)
    if delay > 0:
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            sys.stderr.write("fake narrator: loading\n")
            sys.stderr.flush()
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    if os.environ.get("CRUCIBLE_FAKE_READY_NEVER") == "1":
        while True:
            time.sleep(0.1)

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
