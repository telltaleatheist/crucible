"""The video worker's engine-independent half: the request, progress, cancel, the mux.

Runs inside the video env (and inside the fake worker the tests start). It
needs only the standard library: an engine hands back a `Clip` of raw RGB
frames and 16-bit PCM, and this module writes the WAV and pipes the frames
into the ffmpeg Crucible ships (crucible/hosttools.py), which encodes
H.264 and AAC into video.mp4.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import wave

import workerio
from workerio import send

WHY_REQUIRED = "every parameter is required because each one changes the video"

H264_ENCODERS: tuple[str, ...] = ("libx264", "libopenh264", "h264_nvenc")

OPENH264_BITS_PER_PIXEL = 0.3

MIN_BITRATE = 2_000_000

AAC_BITRATE = "192k"

SPANS: tuple[tuple[str, float], ...] = (
    ("encoding", 0.08),
    ("connecting", 0.04),
    ("conditioning", 0.03),
    ("denoising", 0.58),
    ("decoding", 0.17),
    ("audio_decoding", 0.03),
    ("muxing", 0.07),
)

MUX_TIMEOUT_SECONDS = 1800


class Cancelled(Exception):
    def __init__(self, stage: str, step: int) -> None:
        super().__init__(f"cancelled while {stage}, after step {step}")
        self.stage = stage
        self.step = step


class CancelBox:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_id = None

    def ask(self, request_id) -> None:
        with self._lock:
            self._request_id = request_id

    def asked(self, request_id: str) -> bool:
        with self._lock:
            return self._request_id == request_id

    def clear(self, request_id: str) -> None:
        with self._lock:
            if self._request_id == request_id:
                self._request_id = None


CANCEL = CancelBox()


class Progress:
    def __init__(self, request_id: str, spans: tuple = SPANS) -> None:
        self.request_id = request_id
        self._spans = dict(spans)
        self._order = [name for name, _ in spans]
        self.stage = None
        self.step = 0
        self.steps = None
        self.stage_seconds: dict = {}
        self._started = time.time()

    @property
    def asked_to_stop(self) -> bool:
        return CANCEL.asked(self.request_id)

    def check(self) -> None:
        if self.asked_to_stop:
            raise Cancelled(self.stage, self.step)

    def _fraction(self) -> float:
        done = sum(self._spans[name] for name in self._order[: self._order.index(self.stage)])
        within = self.step / self.steps if self.steps else 0.0
        return round(min(1.0, done + self._spans[self.stage] * within), 4)

    def _say(self) -> None:
        send("progress", stage=self.stage, step=self.step, steps=self.steps, fraction=self._fraction())

    def enter(self, stage: str, steps=None) -> None:
        self.check()
        now = time.time()
        if self.stage is not None:
            self.stage_seconds[self.stage] = round(now - self._started, 2)
        self.stage, self.step, self.steps, self._started = stage, 0, steps, now
        self._say()

    def reached(self, step: int) -> None:
        self.step = step
        self._say()
        self.check()

    def finish(self) -> dict:
        if self.stage is not None:
            self.stage_seconds[self.stage] = round(time.time() - self._started, 2)
        return self.stage_seconds


class Job:
    def __init__(self, request: dict, label: str) -> None:
        def required(key, kind):
            return workerio.require(request, key, kind, label, WHY_REQUIRED)

        def nullable(key, kind):
            if key not in request:
                raise KeyError(f"the {label} request has no {key!r}; it is required and null when unset")
            return None if request[key] is None else required(key, kind)

        self.request_id = required("request_id", str)
        self.mode = required("mode", str)
        self.prompt = required("prompt", str)
        self.width = required("width", int)
        self.height = required("height", int)
        self.num_frames = required("num_frames", int)
        self.fps = required("fps", int)
        self.seed = required("seed", int)
        self.steps = required("steps", int)
        self.audio = required("audio", bool)
        self.image_path = nullable("image_path", str)
        self.output_path = required("output_path", str)
        self.ffmpeg = required("ffmpeg", str)
        self.revision = required("revision", str)
        self.backend = required("backend", str)

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps

    @property
    def prompt_key(self) -> tuple:
        return (self.prompt, self.revision, self.backend)


class Clip:
    """What an engine hands back: raw frames and, when asked for, 16-bit PCM.

    `frames` yields each frame as width*height*3 bytes of RGB; `pcm` is
    interleaved little-endian int16 with `channels` channels at `sample_rate`.
    """

    def __init__(self, width: int, height: int, fps: int, frames, count: int,
                 pcm: bytes | None = None, channels: int = 0, sample_rate: int = 0) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = frames
        self.count = count
        self.pcm = pcm
        self.channels = channels
        self.sample_rate = sample_rate

    @property
    def audio_seconds(self) -> float | None:
        if self.pcm is None or not self.channels or not self.sample_rate:
            return None
        return round(len(self.pcm) / (2 * self.channels * self.sample_rate), 3)


def write_wav(path: str, pcm: bytes, channels: int, sample_rate: int) -> None:
    with wave.open(path, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def available_encoders(ffmpeg: str) -> set:
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=60
    )
    if completed.returncode != 0:
        tail = (completed.stderr.strip() or completed.stdout.strip())[-300:]
        raise RuntimeError(f"`{ffmpeg} -encoders` exited {completed.returncode}: {tail}")
    found = set()
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            found.add(parts[1])
    return found


def pick_encoder(ffmpeg: str) -> str:
    offered = available_encoders(ffmpeg)
    for name in H264_ENCODERS:
        if name in offered:
            return name
    raise RuntimeError(
        f"{ffmpeg} has no H.264 encoder (looked for {', '.join(H264_ENCODERS)}); "
        "Crucible's own ffmpeg carries libopenh264, so this is some other ffmpeg "
        "ahead of it on PATH"
    )


def video_options(encoder: str, width: int, height: int, fps: int) -> list:
    if encoder == "libx264":
        options = ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    elif encoder == "libopenh264":
        bitrate = max(MIN_BITRATE, round(width * height * fps * OPENH264_BITS_PER_PIXEL))
        options = ["-c:v", "libopenh264", "-b:v", str(bitrate)]
    elif encoder == "h264_nvenc":
        options = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"]
    else:
        raise RuntimeError(f"no settings for encoder {encoder!r}; this worker knows {list(H264_ENCODERS)}")
    return options + ["-pix_fmt", "yuv420p"]


def mux_command(ffmpeg: str, encoder: str, clip: Clip, wav_path, output_path: str) -> list:
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{clip.width}x{clip.height}", "-framerate", str(clip.fps),
        "-i", "pipe:0",
    ]
    if wav_path is not None:
        command += ["-i", wav_path]
    command += ["-map", "0:v:0"]
    if wav_path is not None:
        command += ["-map", "1:a:0"]
    command += video_options(encoder, clip.width, clip.height, clip.fps)
    if wav_path is not None:
        command += ["-c:a", "aac", "-b:a", AAC_BITRATE, "-shortest"]
    else:
        command += ["-an"]
    return command + ["-movflags", "+faststart", "-f", "mp4", output_path]


def mux(ffmpeg: str, clip: Clip, output_path: str) -> str:
    """Encode the clip into output_path; returns the H.264 encoder used."""
    encoder = pick_encoder(ffmpeg)
    wav_path = None
    if clip.pcm is not None:
        wav_path = output_path + ".audio.wav"
        write_wav(wav_path, clip.pcm, clip.channels, clip.sample_rate)
    command = mux_command(ffmpeg, encoder, clip, wav_path, output_path)
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame in clip.frames:
                process.stdin.write(frame)
        except BrokenPipeError:
            pass
        # communicate() flushes and closes stdin itself. Closing it here first made
        # CPython's POSIX communicate() raise "flush of closed file" after every stage
        # of the first real video had finished (PC, 2026-09-30).
        _, said = process.communicate(timeout=MUX_TIMEOUT_SECONDS)
        if process.returncode != 0:
            tail = said.decode("utf-8", "replace").strip()[-600:]
            raise RuntimeError(f"ffmpeg ({encoder}) exited {process.returncode}: {tail}")
    finally:
        if wav_path is not None and os.path.exists(wav_path):
            os.remove(wav_path)
    return encoder


class Worker:
    def __init__(self, label: str, engines: dict, mux_with=mux) -> None:
        self.label = label
        self.engines = engines
        self.engine = None
        self._mux = mux_with

    def load(self, request: dict) -> None:
        name = workerio.require(request, "engine", str, self.label, WHY_REQUIRED)
        engine_class = self.engines.get(name)
        if engine_class is None:
            raise RuntimeError(f"no {self.label} engine {name!r}; this worker runs {sorted(self.engines)}")
        started = time.time()
        try:
            self.engine = engine_class(request)
        except ImportError as exc:
            raise RuntimeError(
                f"the video env in {sys.executable} cannot import what it needs ({exc}). "
                "Build it with `crucible install video`."
            ) from None
        send(
            "ready",
            seconds=round(time.time() - started, 2),
            engine=name,
            device=self.engine.device,
            versions=self.engine.versions,
        )
        send("done")

    def _run(self, job: Job) -> dict:
        progress = Progress(job.request_id)
        started = time.time()
        clip, peaks, extra = self.engine.generate(job, progress)
        progress.enter("muxing")
        encoder = self._mux(job.ffmpeg, clip, job.output_path)
        return {
            "path": job.output_path,
            "mode": job.mode,
            "width": clip.width,
            "height": clip.height,
            "num_frames": clip.count,
            "fps": clip.fps,
            "duration_s": round(clip.count / clip.fps, 3),
            "audio": clip.pcm is not None,
            "audio_seconds": clip.audio_seconds,
            "audio_sample_rate": clip.sample_rate if clip.pcm is not None else None,
            "audio_channels": clip.channels if clip.pcm is not None else None,
            "encoder": encoder,
            "bytes": os.path.getsize(job.output_path),
            "seconds": round(time.time() - started, 2),
            "stage_seconds": progress.finish(),
            "stage_peak_bytes": peaks,
            "peak_bytes": max(peaks.values()) if peaks else None,
            "versions": self.engine.versions,
            **extra,
        }

    def generate(self, request: dict) -> None:
        if self.engine is None:
            raise RuntimeError(
                "a generate request arrived before a load request; the session's first "
                "exchange loads the engine"
            )
        job = Job(request, self.label)
        send("ready", request_id=job.request_id, steps=job.steps)
        try:
            result = self._run(job)
        except Cancelled as stopped:
            send("progress", stage="cancelled", step=stopped.step, during=stopped.stage)
            send("done")
            return
        finally:
            CANCEL.clear(job.request_id)
        send("result", **result)
        send("done")

    def serve(self) -> int:
        ops = {"load": self.load, "generate": self.generate}
        interrupts = {"cancel": lambda request: CANCEL.ask(request.get("request_id"))}
        return workerio.serve(self.label, ops, interrupts)
