"""A stand-in for `crucible/jobs/align/worker.py`, faithful to its JSON-lines wire.

`tests/fake_asr_worker.py`'s idea, for the first worker that OUTLIVES A JOB: a
**real subprocess**, run exactly as the server runs the real one (`<python> <this
file>`, requests on stdin, newline-delimited JSON on fd 1), steered entirely by
environment variables, and looping until stdin closes rather than exiting after
one exchange.

Environment variables and not flags, for `fake_narrator.py`'s reason: the thing
under test is `crucible.workers`, which owns the argv and must not be bent into
passing test fixtures through it.

    CRUCIBLE_FAKE_ALIGN_EXIT_CODE      exit with this code before saying
                                       anything. How a worker whose env is
                                       broken — the commonest real failure —
                                       gets tested.
    CRUCIBLE_FAKE_ALIGN_SILENT         say nothing and sit there, so the silence
                                       timeout is a real timeout and not a crash.
                                       A test that sets this MUST let the server
                                       terminate it.
    CRUCIBLE_FAKE_ALIGN_LOAD_FAIL      answer the LOAD with `failed`, which is a
                                       model that would not load.
    CRUCIBLE_FAKE_ALIGN_LOAD_DELAY_S   seconds to wait before the load's `ready`.
    CRUCIBLE_FAKE_ALIGN_LOAD_RESULT    answer the load with a `result` line too,
                                       which a load must never produce.
    CRUCIBLE_FAKE_ALIGN_FAIL_CHUNK     a chunk POSITION (0-based) that comes back
                                       as `result {error}`. Its neighbours still
                                       succeed: "a failed chunk is reported and
                                       the run continues" is the ruling and it
                                       needs a test on both sides.
    CRUCIBLE_FAKE_ALIGN_LONG_CHUNK     a chunk position that comes back as the
                                       over-300-seconds refusal, in the wording
                                       the real worker uses.
    CRUCIBLE_FAKE_ALIGN_SHORT          emit one fewer result than `ready`
                                       promised chunks, so the positional-count
                                       refusal has something to refuse.
    CRUCIBLE_FAKE_ALIGN_NO_DONE        exit 0 without `done` after the align.
    CRUCIBLE_FAKE_ALIGN_JUNK_LINE      write this text to fd 1 verbatim before
                                       the first result. This is a library
                                       logging to stdout — the bug that corrupted
                                       narrator's aligner stream on a 401-chunk
                                       book — and the server must refuse rather
                                       than skip it.
    CRUCIBLE_FAKE_ALIGN_SLOW_S         seconds to sleep between chunks, so a
                                       cancel has something to interrupt.
    CRUCIBLE_FAKE_ALIGN_DIE_AFTER      exit(9) after this many chunks, mid-align,
                                       which is the resident worker dying while
                                       the residency still advertises it.
    CRUCIBLE_FAKE_ALIGN_IGNORE_EOF     do not exit when stdin closes, so that
                                       `stop()`'s polite door is refused and its
                                       SIGTERM backstop is what has to work.
    CRUCIBLE_FAKE_ALIGN_IGNORE_SIGTERM ignore SIGTERM, so the refusal to escalate
                                       to SIGKILL can be tested. A test that sets
                                       this MUST kill the process itself
                                       afterwards.
    CRUCIBLE_FAKE_ALIGN_TRANSCRIPT     a path. Every request line is appended to
                                       it verbatim, so a test can assert on what
                                       Crucible actually sent — the dtype, the
                                       device, the 300 s ceiling, the mapped
                                       language NAME, and that a chunk carries no
                                       index.

The items a chunk comes back with are derived from its TEXT — one item per
whitespace-separated word, a tenth of a second each — so a test can prove the
server put each result against the right chunk without the worker ever reporting
an index.
"""

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
        # Items derived from the TEXT, so a result can be traced back to the
        # chunk it answers without the worker having reported an index.
        words = chunks[position]["text"].split()
        send(
            results,
            "result",
            items=[
                {"text": word, "start": at / 10.0, "end": (at + 1) / 10.0}
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
    # The real worker dups fd 1 away and points the original at stderr before it
    # imports anything that logs. This one does the same, so that a `print` in a
    # test double cannot accidentally pass for a result.
    results_fd = os.dup(1)
    os.dup2(2, 1)
    results = os.fdopen(results_fd, "w", encoding="utf-8", buffering=1)

    exit_code = _env_int("CRUCIBLE_FAKE_ALIGN_EXIT_CODE")
    if exit_code is not None:
        sys.stderr.write("fake align worker: told to exit before saying anything\n")
        return exit_code

    if os.environ.get("CRUCIBLE_FAKE_ALIGN_IGNORE_SIGTERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # win32's polite signal is CTRL_BREAK_EVENT, which arrives as SIGBREAK
        # (`crucible/procgroup.py`); ignoring the POLITE signal means that one.
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)

    transcript = os.environ.get("CRUCIBLE_FAKE_ALIGN_TRANSCRIPT")

    # The loop is the point: this worker lives across requests, and stdin closing
    # is what ends it.
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

    # EOF on stdin. The real worker returns 0 here, which is `stop()`'s polite
    # door; this one can be told to sit there instead, so that the SIGTERM
    # backstop — and the refusal to escalate past it — has something to act on.
    if os.environ.get("CRUCIBLE_FAKE_ALIGN_IGNORE_EOF") == "1":
        while True:
            time.sleep(0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
