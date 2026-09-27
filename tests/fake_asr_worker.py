from __future__ import annotations

import json
import math
import os
import signal
import sys
import time

WINDOW_SEGMENTS = ((0.0, 5.0), (895.0, 905.0))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    return None if raw is None or raw == "" else int(raw)


def send(results, message_type: str, **fields: object) -> None:
    results.write(json.dumps({"type": message_type, **fields}) + "\n")
    results.flush()


def main() -> int:
    results_fd = os.dup(1)
    os.dup2(2, 1)
    results = os.fdopen(results_fd, "w", encoding="utf-8", buffering=1)

    exit_code = _env_int("CRUCIBLE_FAKE_ASR_EXIT_CODE")
    if exit_code is not None:
        sys.stderr.write("fake asr worker: told to exit before saying anything\n")
        return exit_code

    if os.environ.get("CRUCIBLE_FAKE_ASR_IGNORE_SIGTERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)

    line = sys.stdin.readline()
    transcript = os.environ.get("CRUCIBLE_FAKE_ASR_TRANSCRIPT")
    if transcript:
        with open(transcript, "a", encoding="utf-8") as handle:
            handle.write(line)
    request = json.loads(line)

    if "initial_prompt" not in request:
        send(results, "failed", message="the asr request has no 'initial_prompt'")
        return 1
    if request["initial_prompt"] is not None and not isinstance(
        request["initial_prompt"], str
    ):
        send(
            results,
            "failed",
            message="the asr request's 'initial_prompt' must be a string or null",
        )
        return 1

    if os.environ.get("CRUCIBLE_FAKE_ASR_SILENT") == "1":
        while True:
            time.sleep(0.05)

    total = _env_float("CRUCIBLE_FAKE_ASR_DURATION_S", 1800.0)
    window_s = request["window_s"]
    windows = int(math.ceil(total / window_s))

    for index in range(int(_env_float("CRUCIBLE_FAKE_ASR_DECODE_LINES", 2))):
        send(
            results,
            "progress",
            stage="decoding",
            processed_s=(index + 1) * 100.0,
            total_s=total,
            cues=0,
        )

    delay = _env_float("CRUCIBLE_FAKE_ASR_READY_DELAY_S", 0.0)
    if delay > 0:
        time.sleep(delay)

    send(
        results,
        "ready",
        duration_s=total,
        windows=windows,
        device=request["device"],
        compute_type=request["compute_type"],
    )

    junk = os.environ.get("CRUCIBLE_FAKE_ASR_JUNK_LINE")
    if junk:
        results.write(junk + "\n")
        results.flush()

    fail_window = _env_int("CRUCIBLE_FAKE_ASR_FAIL_WINDOW")
    emit = windows - 1 if os.environ.get("CRUCIBLE_FAKE_ASR_SHORT") == "1" else windows
    slow = _env_float("CRUCIBLE_FAKE_ASR_SLOW_S", 0.0)

    for index in range(emit):
        if slow > 0:
            time.sleep(slow)
        if fail_window is not None and fail_window == index:
            send(results, "result", error=f"fake asr worker was told to fail window {index}")
            continue
        segments = []
        for start, end in WINDOW_SEGMENTS:
            segment = {
                "start": start,
                "end": end,
                "text": f" window {index} at {start:.0f}.",
            }
            if request["word_timestamps"]:
                segment["words"] = [
                    {
                        "start": start,
                        "end": (start + end) / 2,
                        "word": f" w{index}",
                        "probability": 0.9,
                    },
                    {
                        "start": (start + end) / 2,
                        "end": end,
                        "word": f" s{start:.0f}",
                        "probability": 0.8,
                    },
                ]
            segments.append(segment)
        send(
            results,
            "result",
            segments=segments,
            language=request["language"] or "en",
            language_probability=0.98,
        )
        send(
            results,
            "progress",
            stage="transcribing",
            processed_s=min(total, (index + 1) * float(window_s)),
            total_s=total,
            cues=(index + 1) * len(WINDOW_SEGMENTS),
        )

    if os.environ.get("CRUCIBLE_FAKE_ASR_NO_DONE") == "1":
        return 0
    send(results, "done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
