"""The real segment worker with the models replaced by drawn masks.

Everything but the model runs for real: workerio, the request, Pillow opening
the input, the mask and cutout PNGs, coverage, cancel. `cutout` draws a soft
ellipse in the middle of the picture; `select` fills the box, or a disc around
the label-1 points, so a test can tell which prompt reached the worker.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REAL_WORKER = Path(__file__).resolve().parents[1] / "crucible" / "jobs" / "segment" / "worker.py"


def _load_real_worker():
    spec = importlib.util.spec_from_file_location("crucible_segment_worker", REAL_WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = _load_real_worker()

DISC_RADIUS = 4


def _transcribe(request: dict) -> None:
    transcript = os.environ.get("CRUCIBLE_FAKE_SEGMENT_TRANSCRIPT")
    if transcript:
        with open(transcript, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(request) + "\n")


class FakeEngine:
    def __init__(self, request: dict) -> None:
        _transcribe({**request, "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")})
        if os.environ.get("CRUCIBLE_FAKE_SEGMENT_LOAD_FAIL") == "1":
            raise RuntimeError("fake segment worker was told not to load")
        self.name = request["engine"]
        self.device = request["device"]
        self.versions = {"fake": "1.0"}

    def peak_bytes(self) -> int:
        return 7

    def release(self) -> None:
        return None

    def _wait(self, job) -> None:
        pause = float(os.environ.get("CRUCIBLE_FAKE_SEGMENT_PAUSE_S") or 0)
        deadline = time.monotonic() + pause
        while time.monotonic() < deadline:
            if worker.CANCEL.asked(job.request_id):
                raise worker.Cancelled("segmenting")
            time.sleep(0.02)

    def mask(self, picture, job):
        from PIL import Image, ImageDraw

        _transcribe({"op": "segment", "request_id": job.request_id, "points": job.points, "box": job.box})
        self._wait(job)
        width, height = picture.size
        mask = Image.new("L", picture.size, 0)
        draw = ImageDraw.Draw(mask)
        if job.kind == "cutout":
            draw.ellipse((width // 4, height // 4, 3 * width // 4, 3 * height // 4), fill=200)
            return mask, {"score": None, "multimask": None}
        if job.box:
            x0, y0, x1, y1 = job.box
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
        for point in job.points or ():
            fill = 255 if point["label"] == 1 else 0
            x, y = point["x"], point["y"]
            draw.ellipse((x - DISC_RADIUS, y - DISC_RADIUS, x + DISC_RADIUS, y + DISC_RADIUS), fill=fill)
        multimask = bool(job.points) and len(job.points) == 1 and not job.box
        return mask, {"score": 0.93, "multimask": multimask}


worker.ENGINES["birefnet"] = FakeEngine
worker.ENGINES["sam2"] = FakeEngine


if __name__ == "__main__":
    sys.exit(worker.main())
