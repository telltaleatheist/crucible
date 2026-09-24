"""A stand-in for `crucible/jobs/asr/mlx_worker.py`, faithful to its wire.

`tests/fake_asr_worker.py` stands in for the faster-whisper worker and carries
the long note about why these doubles are real subprocesses steered by
environment variables. This is its sibling, and it is a SEPARATE file for the
same reason the real workers are separate modules: the thing worth testing here
is that the server picks the RIGHT ONE and hands it the right envelope, and a
double shared between the two engines could not tell those apart.

What it adds over the faster-whisper double, and it is only what the real mlx
worker adds:

  * it REFUSES `vad_filter: true` — mlx-whisper has no voice-activity detector
    at all, so the real worker fails by name rather than transcribing under
    rules the caller did not ask for. The server refuses this at preflight, so a
    request that reaches here with the flag set is a bug in Crucible, and the
    double proves the worker would still catch it.
  * it REFUSES any `device` but `metal`. MLX has one device; `mps` is torch's
    name for the same silicon and belongs to the `align` worker. A double that
    accepted either spelling would let the server send the wrong one forever.

    CRUCIBLE_FAKE_MLX_ASR_TRANSCRIPT  a path. The request line is appended to it
                                      verbatim, so a test can assert that THIS
                                      worker — not the faster-whisper one — was
                                      the one Crucible ran, and that it was sent
                                      `metal` and `float16`.
    CRUCIBLE_FAKE_MLX_ASR_DURATION_S  the decoded duration to claim. Default
                                      1800.0, two windows at the server's 900 s.

The segments are `fake_asr_worker`'s, at the same window-relative positions and
for the same reason: window 1's first segment lands at 900.0 absolute only if
the server shifted by POSITION, and its straddler is the overlap duplicate a
correct dedup drops.
"""

from __future__ import annotations

import json
import math
import os
import sys

WINDOW_SEGMENTS = ((0.0, 5.0), (895.0, 905.0))


def send(results, message_type: str, **fields: object) -> None:
    results.write(json.dumps({"type": message_type, **fields}) + "\n")
    results.flush()


def main() -> int:
    results_fd = os.dup(1)
    os.dup2(2, 1)
    results = os.fdopen(results_fd, "w", encoding="utf-8", buffering=1)

    line = sys.stdin.readline()
    transcript = os.environ.get("CRUCIBLE_FAKE_MLX_ASR_TRANSCRIPT")
    if transcript:
        with open(transcript, "a", encoding="utf-8") as handle:
            handle.write(line)
    request = json.loads(line)

    # The real worker requires the KEY and takes a string or null (null is no
    # prompt), so the double refuses what it would refuse. A double that
    # accepted a missing key would let the server stop sending it unnoticed.
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

    if request["vad_filter"]:
        send(
            results,
            "failed",
            message=(
                "this request asks for vad_filter and mlx-whisper has no VAD at all"
            ),
        )
        return 1
    if request["device"] != "metal":
        send(
            results,
            "failed",
            message=(
                f"device {request['device']!r} is not mlx-whisper's; MLX runs on "
                "Metal and nothing else"
            ),
        )
        return 1

    raw = os.environ.get("CRUCIBLE_FAKE_MLX_ASR_DURATION_S")
    total = 1800.0 if raw is None or raw == "" else float(raw)
    window_s = request["window_s"]
    windows = int(math.ceil(total / window_s))

    send(results, "progress", stage="decoding", processed_s=100.0,
         total_s=total, cues=0)
    send(
        results,
        "ready",
        duration_s=total,
        windows=windows,
        device=request["device"],
        compute_type=request["compute_type"],
    )

    emitted = 0
    for index in range(windows):
        rows = []
        for start, end in WINDOW_SEGMENTS:
            row = {
                "start": start,
                "end": end,
                "text": f"window {index} at {start:.0f}s",
            }
            if request["word_timestamps"]:
                row["words"] = [
                    {
                        "start": start,
                        "end": end,
                        "word": " window",
                        "probability": 0.9,
                    }
                ]
            rows.append(row)
            emitted += 1
        send(
            results,
            "result",
            segments=rows,
            language=request["language"] or "en",
            # The real worker gets this from whisper's own `detect_language`
            # when the caller sent null, and reports 1.0 when the caller named
            # the language — because then nothing was detected.
            language_probability=1.0 if request["language"] else 0.99,
        )
        send(results, "progress", stage="transcribing",
             processed_s=min(total, (index + 1) * float(window_s)),
             total_s=total, cues=emitted)

    send(results, "done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
