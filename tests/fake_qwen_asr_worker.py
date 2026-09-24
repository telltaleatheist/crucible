"""A stand-in for `crucible/jobs/asr/qwen_worker.py`, faithful to its wire.

`fake_align_worker.py`'s idea for the Qwen3-ASR session: a real subprocess, run
as the server runs the real one, looping over requests until stdin closes, and
steered entirely by environment variables. No vLLM, no MLX, no audio.

A "wav" this fake writes is a small JSON file, not audio: `{"start", "duration",
"window"}` in ABSOLUTE seconds of the job's input. That is what lets a re-split
of one piece (the loop guard's next rung) know where in the stream it is, so a
test can say "the loop is at 200 s" and have it follow the audio through every
re-cut, the way a real loop follows the real audio.

    CRUCIBLE_FAKE_QWEN_DURATION_S   the input's duration (default 400).
    CRUCIBLE_FAKE_QWEN_LOOP_AT      an absolute second. The piece covering it
                                    loops when decoded.
    CRUCIBLE_FAKE_QWEN_LOOP_KIND    `token_limit` (the decode runs out its
                                    budget), `repeat` (one line 60 times, inside
                                    the budget), or `collapse` (reads like speech
                                    but carries the aligner fake's collapse
                                    marker, so only the aligner can tell).
    CRUCIBLE_FAKE_QWEN_LOOP_ABOVE_S the loop happens only in pieces decoded at a
                                    window LARGER than this; at this window and
                                    below it the piece transcribes cleanly. 0 (the
                                    default) means it never recovers.
    CRUCIBLE_FAKE_QWEN_SILENT_AT    an absolute second whose piece transcribes
                                    to nothing (a silent stretch).
    CRUCIBLE_FAKE_QWEN_LOAD_FAIL    answer the load with `failed`.
    CRUCIBLE_FAKE_QWEN_TRANSCRIPT   a path; every request line is appended to it.
"""

from __future__ import annotations

import json
import math
import os
import sys

COLLAPSE_MARKER = "zzcollapse"


def send(results, message_type: str, **fields: object) -> None:
    results.write(json.dumps({"type": message_type, **fields}) + "\n")
    results.flush()


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _covers(start: float, duration: float, second: float | None) -> bool:
    return second is not None and start <= second < start + duration


def handle_load(results, request: dict) -> None:
    if os.environ.get("CRUCIBLE_FAKE_QWEN_LOAD_FAIL") == "1":
        send(results, "failed", message="fake qwen worker was told not to load")
        raise SystemExit(1)
    context = request["context"]
    send(
        results,
        "ready",
        seconds=2.0,
        engine=request["engine"],
        device="cuda" if request["engine"] == "vllm" else "Device(gpu, 0)",
        dtype=request["dtype"],
        context_tokens=0 if context is None else len(context.split()),
    )
    send(results, "done")


def handle_split(results, request: dict) -> None:
    source = request["source"]
    window = float(request["max_piece_s"])
    try:
        with open(source, encoding="utf-8") as handle:
            piece = json.load(handle)
        base, total = float(piece["start"]), float(piece["duration"])
    except (OSError, ValueError, KeyError):
        # The job's own input: not one of this fake's pieces.
        base, total = 0.0, _float("CRUCIBLE_FAKE_QWEN_DURATION_S", 400.0)
    os.makedirs(request["out_dir"], exist_ok=True)
    count = max(1, math.ceil(total / window))
    send(results, "progress", stage="decoding", processed_s=total)
    send(results, "ready", duration_s=total, pieces=count)
    stem = os.path.splitext(os.path.basename(source))[0]
    for position in range(count):
        offset = position * window
        duration = min(window, total - offset)
        path = os.path.join(request["out_dir"], f"{stem}.{position:05d}.wav")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"start": base + offset, "duration": duration, "window": window},
                handle,
            )
        send(results, "result", offset_s=offset, duration_s=duration, wav=path)
    send(results, "done")


def transcribe_one(piece: dict, max_tokens: int) -> dict:
    with open(piece["wav"], encoding="utf-8") as handle:
        where = json.load(handle)
    start, duration, window = where["start"], where["duration"], where["window"]
    loop_at = os.environ.get("CRUCIBLE_FAKE_QWEN_LOOP_AT")
    silent_at = os.environ.get("CRUCIBLE_FAKE_QWEN_SILENT_AT")
    above = _float("CRUCIBLE_FAKE_QWEN_LOOP_ABOVE_S", 0.0)
    if _covers(start, duration, None if silent_at is None else float(silent_at)):
        return {"text": "", "tokens": 1, "hit_token_limit": False}
    if _covers(start, duration, None if loop_at is None else float(loop_at)) and (
        window > above
    ):
        kind = os.environ.get("CRUCIBLE_FAKE_QWEN_LOOP_KIND", "token_limit")
        line = "and so we went back to the start "
        if kind == "token_limit":
            return {"text": line * 400, "tokens": max_tokens, "hit_token_limit": True}
        if kind == "repeat":
            return {"text": line * 60, "tokens": 540, "hit_token_limit": False}
        if kind == "collapse":
            return {
                "text": f"um so {COLLAPSE_MARKER} " + "words " * 20,
                "tokens": 40,
                "hit_token_limit": False,
            }
        raise SystemExit(f"unknown loop kind {kind!r}")
    return {
        "text": f"Um, the piece at {start:.0f} seconds, uh, says hello.",
        "tokens": 16,
        "hit_token_limit": False,
    }


def handle_transcribe(results, request: dict) -> None:
    pieces = request["pieces"]
    send(results, "ready", pieces=len(pieces))
    for position, piece in enumerate(pieces):
        send(results, "result", **transcribe_one(piece, int(piece["max_tokens"])))
        send(
            results,
            "progress",
            stage="transcribing",
            processed=position + 1,
            total=len(pieces),
        )
    send(results, "done")


HANDLERS = {"load": handle_load, "split": handle_split, "transcribe": handle_transcribe}


def main() -> int:
    results_fd = os.dup(1)
    os.dup2(2, 1)
    results = os.fdopen(results_fd, "w", encoding="utf-8", buffering=1)
    transcript = os.environ.get("CRUCIBLE_FAKE_QWEN_TRANSCRIPT")
    for line in sys.stdin:
        if not line.strip():
            continue
        if transcript:
            with open(transcript, "a", encoding="utf-8") as handle:
                handle.write(line)
        request = json.loads(line)
        handler = HANDLERS.get(request.get("op"))
        if handler is None:
            send(results, "failed", message=f"unknown op {request.get('op')!r}")
            return 1
        handler(results, request)
    return 0


if __name__ == "__main__":
    sys.exit(main())
