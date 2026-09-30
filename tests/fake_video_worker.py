"""A stand-in for crucible/jobs/video/ltx_worker.py (engine `ltx`, the PC) and
ltx2mlx_worker.py (engine `ltx-2-mlx`, the Mac, with its refining pass): videocore's own Worker,
Progress, Job and Clip, with a fake engine and a mux that writes a small MP4
box structure instead of calling ffmpeg. Runs no model and needs no GPU.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from pathlib import Path

JOBS_DIR = Path(__file__).resolve().parents[1] / "crucible" / "jobs"
sys.path.insert(0, str(JOBS_DIR))
import workerio

sys.path.pop(0)
workerio.claim_stdout()
videocore = workerio.load_sibling("videocore", str(JOBS_DIR / "video" / "videocore.py"))

SAMPLE_RATE = 48000


def _transcribe(request: dict) -> None:
    transcript = os.environ.get("CRUCIBLE_FAKE_VIDEO_TRANSCRIPT")
    if transcript:
        with open(transcript, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(request) + "\n")


def _box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", 8 + len(body)) + kind + body


def fake_mux(ffmpeg: str, clip, output_path: str) -> str:
    frames = sum(len(frame) for frame in clip.frames)
    body = json.dumps(
        {"ffmpeg": ffmpeg, "frames": clip.count, "frame_bytes": frames, "audio_bytes": len(clip.pcm or b"")}
    ).encode()
    Path(output_path).write_bytes(_box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2mp41") + _box(b"free", body))
    return "fake-h264"


class FakeEngine:
    def __init__(self, request: dict) -> None:
        _transcribe(request)
        if os.environ.get("CRUCIBLE_FAKE_VIDEO_LOAD_FAIL") == "1":
            raise RuntimeError("fake video worker was told not to load")
        self.device = request["device"]
        self.versions = {"fake": "1.0", "engine": request["engine"]}
        self._seen: set = set()
        self._two_pass = request["engine"] == "ltx-2-mlx"
        if self._two_pass:
            self.spans = videocore.TWO_PASS_SPANS

    def _steps(self, stage: str, steps: int, pause: float, progress) -> None:
        progress.enter(stage, steps)
        for step in range(1, steps + 1):
            time.sleep(pause)
            progress.reached(step)

    def generate(self, job, progress):
        _transcribe({
            "op": "generate", "request_id": job.request_id, "seed": job.seed, "mode": job.mode,
            "image_path": job.image_path, "num_frames": job.num_frames, "audio": job.audio,
            "ffmpeg": job.ffmpeg, "steps": job.steps, "refine_steps": job.refine_steps,
            "width": job.width, "height": job.height,
        })
        pause = float(os.environ.get("CRUCIBLE_FAKE_VIDEO_STEP_S") or 0)
        hit = job.prompt_key in self._seen
        if not hit:
            progress.enter("encoding")
            if not self._two_pass:
                progress.enter("connecting")
            self._seen.add(job.prompt_key)
        if job.image_path is not None and not self._two_pass:
            progress.enter("conditioning")
        self._steps("denoising", job.steps, pause, progress)
        if self._two_pass:
            self._steps("refining", job.refine_steps, pause, progress)
        progress.enter("decoding")
        frame = bytes(3 * 4 * 4)
        pcm = None
        if job.audio:
            progress.enter("audio_decoding")
            pcm = b"\x00\x00" * 2 * round(job.num_frames / job.fps * SAMPLE_RATE)
        clip = videocore.Clip(
            job.width, job.height, job.fps, (frame for _ in range(job.num_frames)),
            job.num_frames, pcm=pcm, channels=2 if pcm else 0, sample_rate=SAMPLE_RATE if pcm else 0,
        )
        peaks = {"denoising": 19, "decoding": 8}
        if self._two_pass:
            peaks["refining"] = 23
        return clip, peaks, {
            "prompt_cache": "hit" if hit else "miss",
            "quantization": {"fake": True},
            "sampling": {"passes": ["half", "full"] if self._two_pass else ["full"]},
        }


if __name__ == "__main__":
    engines = {"ltx": FakeEngine, "ltx-2-mlx": FakeEngine}
    sys.exit(videocore.Worker("fake video", engines, mux_with=fake_mux).serve())
