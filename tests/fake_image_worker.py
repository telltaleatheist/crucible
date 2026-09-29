from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path

REAL_WORKER = Path(__file__).resolve().parents[1] / "crucible" / "jobs" / "image" / "worker.py"


def _load_real_worker():
    spec = importlib.util.spec_from_file_location("crucible_image_worker", REAL_WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = _load_real_worker()


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


class TinyPng:
    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)

    def save(self, path: str, format: str) -> None:
        width, height = self.size
        rows = b"".join(b"\x00" + b"\x80\x40\x20" * width for _ in range(height))
        header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        Path(path).write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress(rows))
            + _chunk(b"IEND", b"")
        )


def _transcribe(request: dict) -> None:
    transcript = os.environ.get("CRUCIBLE_FAKE_IMAGE_TRANSCRIPT")
    if transcript:
        with open(transcript, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(request) + "\n")


class FakeEngine:
    def __init__(self, request: dict) -> None:
        _transcribe(request)
        if os.environ.get("CRUCIBLE_FAKE_IMAGE_LOAD_FAIL") == "1":
            raise RuntimeError("fake image worker was told not to load")
        self.device = request["device"]
        self.versions = {"fake": "1.0"}

    def nbytes(self, encoded):
        return len(encoded["prompt_embeds"])

    def generate(self, job, progress, cached):
        encodes = cached is None
        _transcribe({"op": "generate", "request_id": job.request_id, "seed": job.seed, "encoded": encodes})
        pause = float(os.environ.get("CRUCIBLE_FAKE_IMAGE_STEP_S") or 0)
        encode_pause = float(os.environ.get("CRUCIBLE_FAKE_IMAGE_ENCODE_S") or 0)
        progress.enter("encoding")
        if encodes:
            time.sleep(encode_pause)
            cached = {"prompt_embeds": job.prompt.encode("utf-8")}
        progress.enter("denoising")
        for _ in range(job.steps):
            time.sleep(pause)
            progress.stepped()
        progress.enter("decoding")
        return TinyPng(job.width, job.height), {"encoding": 3, "denoising": 5, "decoding": 2}, cached


worker.ENGINES["mflux"] = FakeEngine
worker.ENGINES["diffusers"] = FakeEngine


if __name__ == "__main__":
    sys.exit(worker.main())
