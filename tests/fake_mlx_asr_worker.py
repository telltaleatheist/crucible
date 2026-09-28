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
        samples=int(total * 16000),
        speech_s=total if request.get("speech") is not None else None,
        kept=[[0, int(total * 16000)]] if request.get("speech") is not None else None,
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
            language_probability=1.0 if request["language"] else 0.99,
        )
        send(results, "progress", stage="transcribing",
             processed_s=min(total, (index + 1) * float(window_s)),
             total_s=total, cues=emitted)

    send(results, "done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
