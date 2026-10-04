from __future__ import annotations

import json
import math
import os
import struct
import sys
import time
import wave
from pathlib import Path

JOBS_DIR = Path(__file__).resolve().parents[1] / "crucible" / "jobs"
sys.path.insert(0, str(JOBS_DIR))
import workerio

sys.path.pop(0)
workerio.claim_stdout()
audiocore = workerio.load_sibling("audiocore", str(JOBS_DIR / "audio" / "audiocore.py"))

FAKE_SCORE = "X:1\nT:fake song\nM:4/4\nK:C\n\"C\"CDEF|\"G\"GABc|\n"

SPANS = (("encoding", 0.1), ("denoising", 0.7), ("decoding", 0.1), ("saving", 0.1))


def _crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def _crc16(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _streaminfo(block: int, rate: int, channels: int, frames: int) -> bytes:
    packed = (rate << 44) | ((channels - 1) << 41) | (15 << 36) | frames
    return struct.pack(">HH", block, block) + b"\x00" * 6 + packed.to_bytes(8, "big") + b"\x00" * 16


def _frame(number: int, channels: list[list[int]]) -> bytes:
    count = len(channels[0])
    header = bytes([0xFF, 0xF8, 0x70, ((len(channels) - 1) << 4) | 0x08, number]) + struct.pack(">H", count - 1)
    header += bytes([_crc8(header)])
    body = b"".join(b"\x02" + struct.pack(f">{count}h", *samples) for samples in channels)
    frame = header + body
    return frame + struct.pack(">H", _crc16(frame))


class TinyAudio:
    def __init__(self, frames: int, sample_rate: int, channels: int) -> None:
        self.frames = frames
        self.sample_rate = sample_rate
        self.channels = channels

    def _samples(self) -> list[int]:
        return [int(8000 * math.sin(2 * math.pi * 440 * i / self.sample_rate)) for i in range(self.frames)]

    def save(self, path: str, audio_format: str) -> None:
        samples = self._samples()
        if audio_format == "wav":
            with wave.open(path, "wb") as handle:
                handle.setnchannels(self.channels)
                handle.setsampwidth(2)
                handle.setframerate(self.sample_rate)
                handle.writeframes(b"".join(struct.pack("<h", s) * self.channels for s in samples))
            return
        block = 4096
        frames = [
            _frame(index, [samples[start : start + block]] * self.channels)
            for index, start in enumerate(range(0, self.frames, block))
        ]
        info = _streaminfo(block, self.sample_rate, self.channels, self.frames)
        Path(path).write_bytes(b"fLaC" + bytes([0x80, 0, 0, 34]) + info + b"".join(frames))


def _transcribe(request: dict) -> None:
    transcript = os.environ.get("CRUCIBLE_FAKE_AUDIO_TRANSCRIPT")
    if transcript:
        with open(transcript, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(request) + "\n")


class FakeEngine:
    spans = SPANS

    def __init__(self, request: dict) -> None:
        _transcribe(request)
        if os.environ.get("CRUCIBLE_FAKE_AUDIO_LOAD_FAIL") == "1":
            raise RuntimeError("fake audio worker was told not to load")
        self.device = request["device"]
        self.versions = {"fake": "1.0", "engine": request["engine"]}
        self.notes = None

    def generate(self, job, progress):
        _transcribe({"op": "generate", "request_id": job.request_id, "seed": job.seed, "kind": job.kind})
        pause = float(os.environ.get("CRUCIBLE_FAKE_AUDIO_STEP_S") or 0)
        steps = job.steps or 4
        progress.enter("encoding")
        progress.enter("denoising", steps)
        for step in range(1, steps + 1):
            time.sleep(pause)
            progress.reached(step)
        progress.enter("decoding")
        seconds = job.duration_s or 1.0
        frames = min(int(seconds * job.sample_rate), job.sample_rate // 10)
        score = FAKE_SCORE if job.kind == "song" else None
        return TinyAudio(frames, job.sample_rate, job.channels), score, {"denoising": 7, "decoding": 9}


if __name__ == "__main__":
    sys.exit(audiocore.Worker("fake audio", {"stable-audio-3": FakeEngine, "yue2": FakeEngine}).serve())
