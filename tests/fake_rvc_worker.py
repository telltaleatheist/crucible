"""A stand-in for `crucible/jobs/rvc/worker.py`, faithful to its JSON-lines wire.

`tests/fake_asr_worker.py`'s idea again: a **real subprocess**, run exactly as the
server runs the real one (`<python> <this file>`, one request on stdin,
newline-delimited JSON on fd 1), steered entirely by environment variables.

It does not run ultimate-rvc — obviously — but it does everything around it that
the server depends on: it batches at the size it was given, it reports urvc's
per-file progress with the batch offset applied, it WRITES an output file per
input into the directory it was told to, and it answers one result per input in
the order they arrived.

    CRUCIBLE_FAKE_RVC_EXIT_CODE      exit with this code before saying anything.
    CRUCIBLE_FAKE_RVC_SILENT         say nothing and sit there, for the silence
                                     timeout. A test that sets this MUST let the
                                     server terminate it.
    CRUCIBLE_FAKE_RVC_SKIP           a comma-separated list of input NAMES to
                                     write no output for, which is the failure
                                     `rvc` exists to refuse: every input must
                                     produce an output.
    CRUCIBLE_FAKE_RVC_BATCH_FAIL     a 1-based batch number to fail outright,
                                     with `failed`, as urvc exiting non-zero.
    CRUCIBLE_FAKE_RVC_SHORT          emit one fewer result than there are inputs.
    CRUCIBLE_FAKE_RVC_NO_DONE        exit 0 without `done`.
    CRUCIBLE_FAKE_RVC_JUNK_LINE      write this text to fd 1 verbatim before the
                                     first result — a library logging to stdout.
    CRUCIBLE_FAKE_RVC_SLOW_S         seconds to sleep per batch, so a cancel has
                                     something to interrupt.
    CRUCIBLE_FAKE_RVC_IGNORE_SIGTERM ignore SIGTERM, so the refusal to escalate
                                     to SIGKILL can be tested. A test that sets
                                     this MUST kill the process itself.
    CRUCIBLE_FAKE_RVC_TRANSCRIPT     a path. The request line is appended to it,
                                     followed by a JSON line holding the engine
                                     environment variables this process actually
                                     INHERITED — which is the only way to assert
                                     that `KMP_DUPLICATE_LIB_OK` and the rest of
                                     the hardening reached the engine, since they
                                     travel in the environment and not in the
                                     request.

The output it writes is the input's bytes with a marker appended, so a test can
tell a converted file from a copied one.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time

#: What the server sets for urvc, and what a test asserts arrived. The names are
#: `crucible.jobs.rvc.ENGINE_ENVIRONMENT`'s; they are listed again here because a
#: test double that imported from `crucible` would be testing a relationship that
#: does not exist at runtime — a worker env has no `crucible` in it.
ENGINE_KEYS = (
    "URVC_SKIP_INIT",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "KMP_DUPLICATE_LIB_OK",
    "OMP_NUM_THREADS",
    "PYTHONUNBUFFERED",
)

MARKER = b"\n[converted by the fake rvc worker]\n"


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

    exit_code = _env_int("CRUCIBLE_FAKE_RVC_EXIT_CODE")
    if exit_code is not None:
        sys.stderr.write("fake rvc worker: told to exit before saying anything\n")
        return exit_code

    if os.environ.get("CRUCIBLE_FAKE_RVC_IGNORE_SIGTERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # win32's polite signal is CTRL_BREAK_EVENT, which arrives as SIGBREAK
        # (`crucible/procgroup.py`); ignoring the POLITE signal means that one.
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)

    line = sys.stdin.readline()
    transcript = os.environ.get("CRUCIBLE_FAKE_RVC_TRANSCRIPT")
    if transcript:
        with open(transcript, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write(
                json.dumps({key: os.environ.get(key) for key in ENGINE_KEYS}) + "\n"
            )
    request = json.loads(line)

    if os.environ.get("CRUCIBLE_FAKE_RVC_SILENT") == "1":
        while True:
            time.sleep(0.05)

    inputs = request["inputs"]
    batch_size = request["batch_size"]
    input_dir = request["input_dir"]
    output_dir = request["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    batches = [
        inputs[at : at + batch_size] for at in range(0, len(inputs), batch_size)
    ]
    send(
        results,
        "ready",
        files=len(inputs),
        batches=len(batches),
        batch_size=batch_size,
        model=request["model_name"],
    )

    skip = {
        name
        for name in (os.environ.get("CRUCIBLE_FAKE_RVC_SKIP") or "").split(",")
        if name
    }
    batch_fail = _env_int("CRUCIBLE_FAKE_RVC_BATCH_FAIL")
    slow = _env_float("CRUCIBLE_FAKE_RVC_SLOW_S", 0.0)

    done_before = 0
    for number, batch in enumerate(batches, start=1):
        if slow > 0:
            time.sleep(slow)
        if batch_fail is not None and batch_fail == number:
            send(
                results,
                "failed",
                message=(
                    f"batch {number} of {len(batches)} failed: RuntimeError: urvc "
                    "convert-dir exited 1: fake rvc worker was told to fail it"
                ),
            )
            return 1
        for at, name in enumerate(batch, start=1):
            if name not in skip:
                with open(os.path.join(input_dir, name), "rb") as source:
                    payload = source.read()
                with open(os.path.join(output_dir, name), "wb") as sink:
                    sink.write(payload + MARKER)
            send(
                results,
                "progress",
                stage="converting",
                processed=done_before + at,
                total=len(inputs),
                batch=number,
                batches=len(batches),
            )
        done_before += len(batch)

    junk = os.environ.get("CRUCIBLE_FAKE_RVC_JUNK_LINE")
    if junk:
        results.write(junk + "\n")
        results.flush()

    emit = (
        inputs[:-1] if os.environ.get("CRUCIBLE_FAKE_RVC_SHORT") == "1" else inputs
    )
    for name in emit:
        produced = os.path.join(output_dir, name)
        if not os.path.isfile(produced):
            send(results, "result", error=f"urvc wrote no output for {name}")
            continue
        send(results, "result", bytes=os.path.getsize(produced))

    if os.environ.get("CRUCIBLE_FAKE_RVC_NO_DONE") == "1":
        return 0
    send(results, "done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
