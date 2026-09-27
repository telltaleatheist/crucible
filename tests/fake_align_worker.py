from __future__ import annotations

import json
import os
import signal
import sys
import time


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    return None if raw is None or raw == "" else int(raw)


def send(results, message_type: str, **fields: object) -> None:
    results.write(json.dumps({"type": message_type, **fields}) + "\n")
    results.flush()


def handle_load(results, request: dict) -> None:
    delay = _env_float("CRUCIBLE_FAKE_ALIGN_LOAD_DELAY_S", 0.0)
    if delay > 0:
        time.sleep(delay)
    if os.environ.get("CRUCIBLE_FAKE_ALIGN_LOAD_FAIL") == "1":
        send(results, "failed", message="fake align worker was told not to load")
        raise SystemExit(1)
    send(
        results,
        "ready",
        seconds=1.5,
        device=request["device"],
        dtype=request["dtype"],
    )
    if os.environ.get("CRUCIBLE_FAKE_ALIGN_LOAD_RESULT") == "1":
        send(results, "result", items=[])
    send(results, "done")


def handle_align(results, request: dict) -> None:
    chunks = request["chunks"]
    send(results, "ready", chunks=len(chunks))

    junk = os.environ.get("CRUCIBLE_FAKE_ALIGN_JUNK_LINE")
    if junk:
        results.write(junk + "\n")
        results.flush()

    fail_at = _env_int("CRUCIBLE_FAKE_ALIGN_FAIL_CHUNK")
    long_at = _env_int("CRUCIBLE_FAKE_ALIGN_LONG_CHUNK")
    die_after = _env_int("CRUCIBLE_FAKE_ALIGN_DIE_AFTER")
    slow = _env_float("CRUCIBLE_FAKE_ALIGN_SLOW_S", 0.0)
    emit = (
        len(chunks) - 1
        if os.environ.get("CRUCIBLE_FAKE_ALIGN_SHORT") == "1"
        else len(chunks)
    )

    for position in range(emit):
        if slow > 0:
            time.sleep(slow)
        if die_after is not None and position >= die_after:
            sys.stderr.write("fake align worker: told to die mid-align\n")
            raise SystemExit(9)
        if fail_at is not None and fail_at == position:
            send(
                results,
                "result",
                error=f"RuntimeError: fake align worker failed chunk {position}",
            )
            continue
        if long_at is not None and long_at == position:
            send(
                results,
                "result",
                error=(
                    f"ValueError: {request['max_audio_s'] + 61:.1f}s of audio; "
                    "Qwen3-ForcedAligner places timestamps within "
                    f"{request['max_audio_s']:.0f}s and says nothing about longer "
                    "input"
                ),
            )
            continue
        words = chunks[position]["text"].split()
        marker = os.environ.get("CRUCIBLE_FAKE_ALIGN_COLLAPSE_MARKER")
        if marker and marker in words:
            send(
                results,
                "result",
                items=[{"text": word, "start": 1.2, "end": 1.2} for word in words],
            )
            continue
        lead = 0.0
        try:
            with open(chunks[position]["audio"], encoding="utf-8") as handle:
                lead = float(json.load(handle).get("lead", 0.0))
        except (OSError, ValueError, AttributeError):
            lead = 0.0
        send(
            results,
            "result",
            items=[
                {"text": word, "start": lead + at / 10.0, "end": lead + (at + 1) / 10.0}
                for at, word in enumerate(words)
            ],
        )
        send(
            results,
            "progress",
            stage="aligning",
            processed=position + 1,
            total=len(chunks),
        )

    if os.environ.get("CRUCIBLE_FAKE_ALIGN_NO_DONE") == "1":
        raise SystemExit(0)
    send(results, "done")


HANDLERS = {"load": handle_load, "align": handle_align}


def main() -> int:
    results_fd = os.dup(1)
    os.dup2(2, 1)
    results = os.fdopen(results_fd, "w", encoding="utf-8", buffering=1)

    exit_code = _env_int("CRUCIBLE_FAKE_ALIGN_EXIT_CODE")
    if exit_code is not None:
        sys.stderr.write("fake align worker: told to exit before saying anything\n")
        return exit_code

    if os.environ.get("CRUCIBLE_FAKE_ALIGN_IGNORE_SIGTERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)

    transcript = os.environ.get("CRUCIBLE_FAKE_ALIGN_TRANSCRIPT")

    for line in sys.stdin:
        if not line.strip():
            continue
        if transcript:
            with open(transcript, "a", encoding="utf-8") as handle:
                handle.write(line)
        request = json.loads(line)

        if os.environ.get("CRUCIBLE_FAKE_ALIGN_SILENT") == "1":
            while True:
                time.sleep(0.05)

        handler = HANDLERS.get(request.get("op"))
        if handler is None:
            send(results, "failed", message=f"unknown op {request.get('op')!r}")
            return 1
        handler(results, request)

    if os.environ.get("CRUCIBLE_FAKE_ALIGN_IGNORE_EOF") == "1":
        while True:
            time.sleep(0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
