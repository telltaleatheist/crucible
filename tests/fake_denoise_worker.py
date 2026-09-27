from __future__ import annotations

import json
import os
import sys
import time

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


def _record(line: str) -> None:
    transcript = os.environ.get("CRUCIBLE_FAKE_DENOISE_TRANSCRIPT")
    if not transcript:
        return
    with open(transcript, "a", encoding="utf-8") as handle:
        handle.write(line if line.endswith("\n") else line + "\n")
        handle.write(
            json.dumps({key: os.environ.get(key) for key in ENGINE_KEYS}) + "\n"
        )


def do_load(results, request: dict) -> int:
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
    send(results, "ready", seconds=1.5)
    send(results, "done")
    return 0


def do_separate(results, request: dict) -> int:
    source = _env_json("CRUCIBLE_FAKE_DENOISE_INPUT", DEFAULT_INPUT)
    send(
        results,
        "ready",
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
                f"this input is {source['sample_rate']} Hz and the model is "
                f"{request['sample_rate']} Hz native. Nothing was resampled"
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

    send(results, "result", stems=stems, separate_seconds=8.25)

    if os.environ.get("CRUCIBLE_FAKE_DENOISE_NO_DONE") == "1":
        raise SystemExit(0)
    send(results, "done")
    return 0


OPS = {"load": do_load, "separate": do_separate}


def main() -> int:
    results_fd = os.dup(1)
    os.dup2(2, 1)
    results = os.fdopen(results_fd, "w", encoding="utf-8", buffering=1)

    exit_code = _env_int("CRUCIBLE_FAKE_DENOISE_EXIT_CODE")
    if exit_code is not None:
        sys.stderr.write("fake denoise worker: told to exit before saying anything\n")
        return exit_code

    for line in sys.stdin:
        if not line.strip():
            continue
        _record(line)
        if os.environ.get("CRUCIBLE_FAKE_DENOISE_SILENT") == "1":
            while True:
                time.sleep(0.05)
        request = json.loads(line)
        handler = OPS.get(request.get("op"))
        if handler is None:
            send(
                results,
                "failed",
                message=f"the denoise request's op is {request.get('op')!r}",
            )
            return 1
        code = handler(results, request)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
