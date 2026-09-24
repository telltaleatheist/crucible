"""A stand-in for `crucible/jobs/asr/worker.py`, faithful to its JSON-lines wire.

`tests/fake_engine.py` exists because the only things Crucible needs from vLLM are
a process it can start and SIGTERM and an HTTP surface, and `tests/fake_narrator.py`
does the same for narrator's stdin/stdout protocol. This is that idea for the
phase 4 workers: a **real subprocess**, run exactly as the server runs the real
one (`<python> <this file>`, request on stdin, newline-delimited JSON on fd 1),
steered entirely by environment variables.

Environment variables and not flags, for `fake_narrator.py`'s reason: the thing
under test is `crucible.workers.run_worker`, which owns the argv and must not be
bent into passing test fixtures through it.

    CRUCIBLE_FAKE_ASR_EXIT_CODE       exit with this code before saying anything.
                                      How a worker whose env is broken — the
                                      commonest real failure — gets tested.
    CRUCIBLE_FAKE_ASR_SILENT          say nothing and sit there, so the silence
                                      timeout is a real timeout and not a crash.
                                      A test that sets this MUST let run_worker
                                      terminate it.
    CRUCIBLE_FAKE_ASR_DECODE_LINES    how many `progress {stage: decoding}` lines
                                      to send before `ready`. Default 2. These
                                      are what make the ready timeout a SILENCE
                                      timeout worth testing: a worker decoding an
                                      18 h book is quiet about its windows for
                                      minutes while being noisy about its bytes.
    CRUCIBLE_FAKE_ASR_READY_DELAY_S   seconds to wait before `ready`, spent after
                                      the decode lines. With SILENCE this is the
                                      whole run.
    CRUCIBLE_FAKE_ASR_DURATION_S      the decoded duration to claim. Default
                                      1800.0, which is two windows at the
                                      server's 900 s.
    CRUCIBLE_FAKE_ASR_FAIL_WINDOW     a window index that comes back as
                                      `result {error}`. Its neighbours still
                                      succeed: "a failed window is reported and
                                      the run continues" is the worker's rule and
                                      it needs a test on both sides.
    CRUCIBLE_FAKE_ASR_SHORT           emit one fewer result than `ready` promised
                                      windows, so the positional-count refusal
                                      has something to refuse.
    CRUCIBLE_FAKE_ASR_NO_DONE         exit 0 without `done`, which is an answer
                                      that stopped early rather than a short one.
    CRUCIBLE_FAKE_ASR_JUNK_LINE       write this text to fd 1 verbatim before the
                                      first result. This is a library logging to
                                      stdout — the bug that corrupted narrator's
                                      aligner stream on a 401-chunk book — and
                                      the server must refuse rather than skip it.
    CRUCIBLE_FAKE_ASR_SLOW_S          seconds to sleep between windows, so a
                                      cancel has something to interrupt.
    CRUCIBLE_FAKE_ASR_IGNORE_SIGTERM  ignore SIGTERM, so the refusal to escalate
                                      to SIGKILL can be tested. Crucible never
                                      SIGKILLs a process that may hold CUDA. A
                                      test that sets this MUST kill the process
                                      itself afterwards.
    CRUCIBLE_FAKE_ASR_TRANSCRIPT      a path. The request line is appended to it
                                      verbatim, so a test can assert on what
                                      Crucible actually sent — the 900 s window,
                                      the 15 s overlap, `float16`, and the
                                      language it resolved `auto` to.

The segments are arranged so that both of the server's arithmetic jobs are
checkable exactly. Every window emits two, at window-relative 0.0-5.0 and
895.0-905.0, so:

  * shifting by position turns window 1's first segment into 900.0-905.0, which
    proves the server used the position and not something the worker said; and
  * that segment begins inside window 0's straddler (895.0-905.0 absolute), which
    is precisely the overlap duplicate the 15 s back-reach produces, so a correct
    dedup drops exactly one segment out of four.
"""

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
    # The real worker dups fd 1 away and points the original at stderr before it
    # imports anything that logs. This one does the same, so that a `print` in a
    # test double cannot accidentally pass for a result.
    results_fd = os.dup(1)
    os.dup2(2, 1)
    results = os.fdopen(results_fd, "w", encoding="utf-8", buffering=1)

    exit_code = _env_int("CRUCIBLE_FAKE_ASR_EXIT_CODE")
    if exit_code is not None:
        sys.stderr.write("fake asr worker: told to exit before saying anything\n")
        return exit_code

    if os.environ.get("CRUCIBLE_FAKE_ASR_IGNORE_SIGTERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # win32's polite signal is CTRL_BREAK_EVENT, which arrives as SIGBREAK
        # (`crucible/procgroup.py`); ignoring the POLITE signal means that one.
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)

    line = sys.stdin.readline()
    transcript = os.environ.get("CRUCIBLE_FAKE_ASR_TRANSCRIPT")
    if transcript:
        with open(transcript, "a", encoding="utf-8") as handle:
            handle.write(line)
    request = json.loads(line)

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
            # `auto` reaches the worker as null and the model answers with what it
            # detected, so the fake answers the same way rather than echoing null.
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
