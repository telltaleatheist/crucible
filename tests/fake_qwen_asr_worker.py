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
    window = float(request["max_piece_s"])
    overlap = float(request["overlap_s"])
    total = _float("CRUCIBLE_FAKE_QWEN_DURATION_S", 400.0)
    region = request["region_s"]
    base, end = (0.0, total) if region is None else (float(region[0]), float(region[1]))
    os.makedirs(request["out_dir"], exist_ok=True)
    count = max(1, math.ceil((end - base) / window - 1e-9))
    send(results, "progress", stage="decoding", processed_s=total)
    send(results, "ready", duration_s=total, pieces=count)
    stem = os.path.splitext(os.path.basename(request["source"]))[0]
    tag = "" if region is None else f".r{base:g}"
    for position in range(count):
        start = base + position * window
        duration = min(window, end - start)
        audio_start = max(0.0, start - overlap)
        audio_end = min(total, start + duration + overlap)
        path = os.path.join(request["out_dir"], f"{stem}{tag}.{position:05d}.wav")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"start": start, "duration": duration, "window": window,
                 "lead": start - audio_start},
                handle,
            )
        send(results, "result", offset_s=start, duration_s=duration,
             audio_offset_s=audio_start, audio_duration_s=audio_end - audio_start,
             wav=path)
    send(results, "done")


def transcribe_one(piece: dict, max_tokens: int) -> dict:
    with open(piece["wav"], encoding="utf-8") as handle:
        where = json.load(handle)
    start, duration, window = where["start"], where["duration"], where["window"]
    loop_at = os.environ.get("CRUCIBLE_FAKE_QWEN_LOOP_AT")
    silent_at = os.environ.get("CRUCIBLE_FAKE_QWEN_SILENT_AT")
    above = _float("CRUCIBLE_FAKE_QWEN_LOOP_ABOVE_S", 0.0)
    echo_at = os.environ.get("CRUCIBLE_FAKE_QWEN_ECHO_AT")
    if _covers(start, duration, None if silent_at is None else float(silent_at)):
        return {"text": "", "tokens": 1, "hit_token_limit": False}
    if _covers(start, duration, None if echo_at is None else float(echo_at)):
        text = os.environ["CRUCIBLE_FAKE_QWEN_ECHO_TEXT"]
        return {"text": text, "tokens": len(text.split()), "hit_token_limit": False}
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
