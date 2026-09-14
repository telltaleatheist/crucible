"""A stand-in for `crucible/jobs/denoise/worker.py`, faithful to its wire.

`tests/fake_rvc_worker.py`'s idea again: a **real subprocess**, run exactly as
the server runs the real one (`<python> <this file>`, one request on stdin,
newline-delimited JSON on fd 1), steered entirely by environment variables.

It does not run audio-separator, and it does not import soundfile either —
neither is in the interpreter running this suite. What it does is everything
around them that the server depends on: it reads the input's shape from a JSON
sidecar the test wrote beside it, it WRITES a stem file per output into the
directory it was told to, and it answers one result carrying every stem, in the
shape the real worker answers with.

    CRUCIBLE_FAKE_DENOISE_INPUT      JSON: {"sample_rate", "frames", "channels"}
                                     — what this worker "reads" off the input.
                                     Defaults to 44.1 kHz and 441000 frames.
    CRUCIBLE_FAKE_DENOISE_STEMS      JSON list of stem descriptors, each
                                     {"name", optional "sample_rate", "frames"}.
                                     Defaults to one `(Dry)` stem matching the
                                     input exactly, which is the good case.
    CRUCIBLE_FAKE_DENOISE_EXIT_CODE  exit with this code before saying anything.
    CRUCIBLE_FAKE_DENOISE_LOAD_FAIL  fail as the model load would.
    CRUCIBLE_FAKE_DENOISE_SILENT     say nothing and sit there, for the silence
                                     timeout. A test that sets this MUST let the
                                     server terminate it.
    CRUCIBLE_FAKE_DENOISE_NO_DONE    exit 0 without `done`.
    CRUCIBLE_FAKE_DENOISE_JUNK_LINE  write this text to fd 1 verbatim before the
                                     result — a library logging to stdout.
    CRUCIBLE_FAKE_DENOISE_SLOW_S     seconds to sleep before separating, so a
                                     cancel has something to interrupt.
    CRUCIBLE_FAKE_DENOISE_TRANSCRIPT a path. The request line is appended to it,
                                     followed by a JSON line holding the engine
                                     environment variables this process actually
                                     INHERITED — the only way to assert that the
                                     OpenMP hardening reached the engine, since
                                     it travels in the environment.
"""

from __future__ import annotations

import json
import os
import sys
import time

#: What the server sets for the shared rvc env, and what a test asserts arrived.
#: Listed again here rather than imported: a test double that imported from
#: `crucible` would be testing a relationship that does not exist at runtime.
ENGINE_KEYS = ("KMP_DUPLICATE_LIB_OK", "OMP_NUM_THREADS", "PYTHONUNBUFFERED")

DEFAULT_INPUT = {"sample_rate": 44100, "frames": 441000, "channels": 2}
MARKER = b"[a stem written by the fake denoise worker]\n"


def send(results, message_type: str, **fields: object) -> None:
    results.write(json.dumps({"type": message_type, **fields}) + "\n")
    results.flush()


def _env_json(name: str, default):
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else json.loads(raw)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    return None if raw is None or raw == "" else int(raw)


def main() -> int:
    results_fd = os.dup(1)
    os.dup2(2, 1)
    results = os.fdopen(results_fd, "w", encoding="utf-8", buffering=1)

    exit_code = _env_int("CRUCIBLE_FAKE_DENOISE_EXIT_CODE")
    if exit_code is not None:
        sys.stderr.write("fake denoise worker: told to exit before saying anything\n")
        return exit_code

    line = sys.stdin.readline()
    transcript = os.environ.get("CRUCIBLE_FAKE_DENOISE_TRANSCRIPT")
    if transcript:
        with open(transcript, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write(
                json.dumps({key: os.environ.get(key) for key in ENGINE_KEYS}) + "\n"
            )
    request = json.loads(line)

    if os.environ.get("CRUCIBLE_FAKE_DENOISE_SILENT") == "1":
        while True:
            time.sleep(0.05)

    source = _env_json("CRUCIBLE_FAKE_DENOISE_INPUT", DEFAULT_INPUT)
    send(
        results,
        "ready",
        model=request["model_filename"],
        sample_rate=source["sample_rate"],
        frames=source["frames"],
        seconds=round(source["frames"] / source["sample_rate"], 3),
        channels=source["channels"],
    )

    if source["sample_rate"] != request["sample_rate"]:
        send(
            results,
            "failed",
            message=(
                f"this input is {source['sample_rate']} Hz and "
                f"{request['model_filename']} is {request['sample_rate']} Hz "
                "native. Nothing was resampled"
            ),
        )
        return 1

    if os.environ.get("CRUCIBLE_FAKE_DENOISE_LOAD_FAIL") == "1":
        send(
            results,
            "failed",
            message=(
                f"audio-separator could not load {request['model_filename']} from "
                f"{request['model_file_dir']}: RuntimeError: told to fail"
            ),
        )
        return 1

    send(results, "progress", stage="separating", processed=0, total=1)

    slow = os.environ.get("CRUCIBLE_FAKE_DENOISE_SLOW_S")
    if slow:
        time.sleep(float(slow))

    base = os.path.splitext(os.path.basename(request["input"]))[0]
    default_stems = [
        {"name": f"{base}_(Dry)_denoise_mel_band_roformer.wav"},
        {"name": f"{base}_(Other)_denoise_mel_band_roformer.wav"},
    ]
    wanted = _env_json("CRUCIBLE_FAKE_DENOISE_STEMS", default_stems)

    output_dir = request["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    stems = []
    for stem in wanted:
        produced = os.path.join(output_dir, stem["name"])
        with open(produced, "wb") as handle:
            handle.write(MARKER + stem["name"].encode("utf-8"))
        stems.append(
            {
                "name": stem["name"],
                "sample_rate": stem.get("sample_rate", source["sample_rate"]),
                "frames": stem.get("frames", source["frames"]),
                "channels": stem.get("channels", source["channels"]),
                "bytes": os.path.getsize(produced),
            }
        )

    junk = os.environ.get("CRUCIBLE_FAKE_DENOISE_JUNK_LINE")
    if junk:
        results.write(junk + "\n")
        results.flush()

    send(results, "result", stems=stems, load_seconds=1.5, separate_seconds=8.25)

    if os.environ.get("CRUCIBLE_FAKE_DENOISE_NO_DONE") == "1":
        return 0
    send(results, "done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
