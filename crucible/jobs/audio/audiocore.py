from __future__ import annotations

import sys
import threading
import time

import workerio
from workerio import send, write_audio

WHY_REQUIRED = "every parameter is required because each one changes the sound"


# /proc/self/status fields a song's host memory is read from (Linux; bytes once read).
HOST_MEMORY_FIELDS = {"VmRSS": "rss_bytes", "RssAnon": "anon_bytes", "RssFile": "file_bytes"}


def host_memory(reset_peak: bool = False):
    """This worker's host memory now, from /proc/self/status: resident, anonymous (what
    the OOM killer counts against the guest) and file-backed (mapped weights the kernel
    can drop and re-read). `reset_peak` first restarts the kernel's peak-resident mark
    (`5` to /proc/self/clear_refs), so the next reading's `peak_rss_bytes` is the peak
    since. None where there is no /proc (macOS): there the card and host memory are one
    pool, which the job's own peak already reports."""
    if not sys.platform.startswith("linux"):
        return None
    if reset_peak:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as handle:
            handle.write("5")
    found = {}
    with open("/proc/self/status", encoding="ascii") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            if key in HOST_MEMORY_FIELDS or key == "VmHWM":
                found[key] = int(value.split()[0]) * 1024
    reading = {name: found[key] for key, name in HOST_MEMORY_FIELDS.items()}
    reading["peak_rss_bytes"] = found["VmHWM"]
    return reading


class Cancelled(Exception):
    def __init__(self, stage: str, step: int) -> None:
        super().__init__(f"cancelled while {stage}, after step {step}")
        self.stage = stage
        self.step = step


class Refused(Exception):
    """A request the engine will not render, said by name (`code`) before it renders
    anything: the worker answers it as a result and stays loaded, where any other error
    ends the worker. Only for what the request carries, never for what went wrong."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


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
    def __init__(self, request_id: str, spans: tuple) -> None:
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

    def note(self, text: str) -> None:
        """A sentence for the job's events beside its progress (the server makes it a
        `note`): what a stage decided, such as a planning set sized toward a length."""
        send("progress", stage=self.stage, step=self.step, steps=self.steps,
             fraction=self._fraction(), note=text)

    def reached(self, step: int, steps=None) -> None:
        self.step = step
        if steps is not None:
            self.steps = steps
        self._say()
        self.check()

    def finish(self) -> dict:
        if self.stage is not None:
            self.stage_seconds[self.stage] = round(time.time() - self._started, 2)
        return self.stage_seconds


class Throttled:
    def __init__(self, progress: Progress, every: int) -> None:
        self._progress = progress
        self._every = every
        self.count = 0

    def tick(self, steps=None) -> None:
        self.count += 1
        if self.count % self._every == 0:
            self._progress.reached(self.count, steps)
        elif self._progress.asked_to_stop:
            self._progress.check()


class Job:
    def __init__(self, request: dict, label: str) -> None:
        def required(key, kind):
            return workerio.require(request, key, kind, label, WHY_REQUIRED)

        def nullable(key, kind):
            if key not in request:
                raise KeyError(f"the {label} request has no {key!r}; it is required and null when unset")
            return None if request[key] is None else required(key, kind)

        self.request_id = required("request_id", str)
        self.kind = required("kind", str)
        self.prompt = nullable("prompt", str)
        self.tags = nullable("tags", str)
        self.lyrics = nullable("lyrics", str)
        self.planning_lyrics = nullable("planning_lyrics", str)
        self.negative_prompt = nullable("negative_prompt", str)
        self.duration_s = nullable("duration_s", (int, float))
        self.seed = required("seed", int)
        self.steps = nullable("steps", int)
        self.cfg = nullable("cfg", (int, float))
        self.instrumental = required("instrumental", bool)
        # The range a song's score is checked against before composing (seconds; either
        # may be null), the model's longest song, and whether the planning lyrics are
        # the server's pool set, which the engine may resize to land in the range.
        self.min_duration_s = nullable("min_duration_s", (int, float))
        self.max_duration_s = nullable("max_duration_s", (int, float))
        self.longest_s = required("longest_s", (int, float))
        self.planning_resizable = required("planning_resizable", bool)
        # A cover's score, ready to render ({kind, abc, cot, lyrics}: the server checked it
        # and made any transfer, crucible/jobs/audio/cover.py); null for a planned song.
        self.cover = nullable("cover", dict)
        # The song's CFG negative branch as the client wrote it ({tags, lyrics}); null for
        # the engine's stock negative.
        self.negative = nullable("negative", dict)
        # The planned score's header fields the client fixed ({key, meter, bpm}, null
        # where not sent); null when none was.
        self.header = nullable("header", dict)
        self.sample_rate = required("sample_rate", int)
        self.channels = required("channels", int)
        self.format = required("format", str)
        self.output_path = required("output_path", str)
        self.score_path = nullable("score_path", str)
        self.revision = required("revision", str)
        self.backend = required("backend", str)


class ArrayAudio:
    def __init__(self, samples, sample_rate: int) -> None:
        self.samples = samples
        self.sample_rate = sample_rate

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else int(self.samples.shape[1])

    def save(self, path: str, audio_format: str) -> None:
        write_audio(path, self.samples, self.sample_rate, audio_format)


class Worker:
    def __init__(self, label: str, engines: dict) -> None:
        self.label = label
        self.engines = engines
        self.engine = None

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
                f"the {name} env in {sys.executable} cannot import what it needs ({exc}). "
                "Build it with `crucible install audio`."
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
        progress = Progress(job.request_id, self.engine.spans)
        started = time.time()
        before = host_memory(reset_peak=True)
        audio, score, peaks = self.engine.generate(job, progress)
        progress.enter("saving")
        audio.save(job.output_path, job.format)
        if score is not None and job.score_path is not None:
            with open(job.score_path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(score)
        frames, sample_rate, channels = audio.frames, audio.sample_rate, audio.channels
        # The song's audio is released before the after-reading, so `after` is what the
        # worker holds between songs: a worker that keeps something of each song shows it
        # as `after` climbing from one song to the next.
        del audio
        after = host_memory()
        return {
            "path": job.output_path,
            "score_path": job.score_path if score is not None else None,
            "audio_seconds": round(frames / sample_rate, 3),
            "sample_rate": sample_rate,
            "channels": channels,
            "seconds": round(time.time() - started, 2),
            "stage_seconds": progress.finish(),
            "stage_peak_bytes": peaks,
            "peak_bytes": max(peaks.values()) if peaks else None,
            "versions": self.engine.versions,
            # What the engine did to this one render beyond its parameters (YuE2's
            # instrumental voice transfer); None when nothing.
            "notes": self.engine.notes,
            # Each token-decoding stage's own account - tokens, cap, how it ended, the
            # path it ran on, its speed (yue2_worker.decode_facts); None for an engine
            # that decodes no tokens.
            "decode_stages": self.engine.decode_stages,
            # The score's nominal length, and the range asked and what it did
            # (yue2_worker.length_record); None for an engine that writes no score.
            "length": self.engine.length,
            # A cover's score in YuE2's own tokens (yue2_worker._given_plan); None for a
            # song the engine planned, or an engine that takes no score.
            "score_tokens": self.engine.score_tokens,
            # The CFG negative branch's prefix in tokens, stock or written by the client
            # (yue2_worker.YuE2Engine._compose); None when no negative branch ran.
            "negative_prefix_tokens": self.engine.negative_prefix_tokens,
            # The key, meter and tempo the rendered score states (scoreheader.read); None
            # for an engine that writes no score.
            "score_header": self.engine.score_header,
            # This worker's host memory as the song began and once it was saved, with the
            # peak between (audiocore.host_memory); `host_homes_bytes` is what the engine
            # keeps in host memory on purpose (yue2_worker.HostHomes), None for an engine
            # that keeps nothing there. None on macOS.
            "host_memory": None if before is None else {
                "before": {k: v for k, v in before.items() if k != "peak_rss_bytes"},
                "after": {k: v for k, v in after.items() if k != "peak_rss_bytes"},
                "peak_rss_bytes": after["peak_rss_bytes"],
                "host_homes_bytes": None if self.engine.host_homes is None else dict(self.engine.host_homes),
            },
        }

    def generate(self, request: dict) -> None:
        if self.engine is None:
            raise RuntimeError(
                "a generate request arrived before a load request; the session's first "
                "exchange loads the engine"
            )
        job = Job(request, self.label)
        send("ready", request_id=job.request_id)
        try:
            result = self._run(job)
        except Cancelled as stopped:
            send("progress", stage="cancelled", step=stopped.step, during=stopped.stage)
            send("done")
            return
        except Refused as refused:
            send("result", refused={"code": refused.code, "message": refused.message})
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
