"""How busy the GPU is while a job runs: one cheap sample a second, attributed to stages.

The server samples, not the worker, so any job type can use it: a job starts a
`GpuBusySampler`, marks its stages as they begin, and stops it when the job ends; `summary()`
is what goes into the done event. It samples the whole GPU, whoever is using it (the desktop
included), since that is what the person at the machine feels.

- mlx-darwin: `ioreg -r -d 1 -c IOAccelerator`, the driver's own "Device Utilization %" in
  each accelerator's PerformanceStatistics. No sudo, about 20 ms a call.
- cuda-linux: `nvidia-smi --query-gpu=utilization.gpu`, the percent of the last sample period
  in which a kernel was running.

It fails soft: a tool that is missing, slow or unparseable gives no sample, three failures in a
row stop the sampler, and the summary's fields are then null. It never fails a job.
docs/internals/video.md, "GPU headroom: measuring how busy a render keeps the GPU", says how the
numbers are used.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from typing import Any, Callable

from .backend import CUDA_LINUX, MLX_DARWIN

INTERVAL_S = 1.0

IOREG_PATH = "/usr/sbin/ioreg"

READ_TIMEOUT_S = 5.0

FAILURES_BEFORE_GIVING_UP = 3

PINNED_PCT = 98.0

IOREG_SOURCE = "ioreg IOAccelerator PerformanceStatistics \"Device Utilization %\""

NVIDIA_SOURCE = "nvidia-smi utilization.gpu"

_DEVICE_UTILIZATION = re.compile(r'"Device Utilization %"\s*=\s*(\d+(?:\.\d+)?)')

Reader = Callable[[], "float | None"]


def parse_ioreg(text: str) -> float | None:
    """The busiest accelerator's "Device Utilization %" in `ioreg -c IOAccelerator` output."""
    found = [float(value) for value in _DEVICE_UTILIZATION.findall(text)]
    return max(found) if found else None


def parse_nvidia_smi(text: str) -> float | None:
    found = []
    for line in text.splitlines():
        try:
            found.append(float(line.strip()))
        except ValueError:
            continue
    return max(found) if found else None


def _run(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=READ_TIMEOUT_S
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout if completed.returncode == 0 else None


def ioreg_reader() -> Reader | None:
    exe = shutil.which("ioreg")
    if exe is None and os.path.exists(IOREG_PATH):
        exe = IOREG_PATH
    if exe is None:
        return None

    def read() -> float | None:
        text = _run([exe, "-r", "-d", "1", "-c", "IOAccelerator"])
        return None if text is None else parse_ioreg(text)

    return read


def nvidia_reader() -> Reader | None:
    from .accelerator import nvidia_smi_path

    exe = nvidia_smi_path()
    if exe is None:
        return None

    def read() -> float | None:
        text = _run([exe, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        return None if text is None else parse_nvidia_smi(text)

    return read


def reader_for(backend_kind: str) -> tuple[Reader | None, str | None]:
    """The sampler for this backend and the name of what it reads, or (None, None)."""
    try:
        if backend_kind == MLX_DARWIN:
            reader = ioreg_reader()
            return reader, IOREG_SOURCE if reader else None
        if backend_kind == CUDA_LINUX:
            reader = nvidia_reader()
            return reader, NVIDIA_SOURCE if reader else None
    except Exception:
        return None, None
    return None, None


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean_pct": None, "max_pct": None, "samples": 0}
    return {
        "mean_pct": round(sum(values) / len(values), 1),
        "max_pct": round(max(values), 1),
        "samples": len(values),
    }


class GpuBusySampler:
    """Samples a reader every `interval` seconds on a daemon thread until stopped."""

    def __init__(self, reader: Reader | None, source: str | None, interval: float | None = None) -> None:
        self._reader = reader
        self._source = source
        self._interval = INTERVAL_S if interval is None else interval
        self._lock = threading.Lock()
        self._stage: str | None = None
        self._samples: list[tuple[str | None, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "GpuBusySampler":
        if self._reader is not None:
            self._thread = threading.Thread(target=self._loop, name="gpu-busy", daemon=True)
            self._thread.start()
        return self

    def mark(self, stage: str | None) -> None:
        with self._lock:
            self._stage = stage

    def _loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                value = self._reader() if self._reader is not None else None
            except Exception:
                value = None
            if value is None:
                failures += 1
                if failures >= FAILURES_BEFORE_GIVING_UP:
                    return
            else:
                failures = 0
                with self._lock:
                    self._samples.append((self._stage, float(value)))
            self._stop.wait(self._interval)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=READ_TIMEOUT_S + self._interval + 1.0)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
        values = [value for _, value in samples]
        overall = _stats(values)
        stages: dict[str, list[float]] = {}
        for stage, value in samples:
            if stage is not None:
                stages.setdefault(stage, []).append(value)
        return {
            "gpu_busy_mean_pct": overall["mean_pct"],
            "gpu_busy_max_pct": overall["max_pct"],
            "gpu_busy_samples": overall["samples"],
            "gpu_busy_pinned_samples": sum(1 for value in values if value >= PINNED_PCT),
            "gpu_busy_stages": {stage: _stats(found) for stage, found in stages.items()},
            "gpu_busy_source": self._source if samples else None,
        }


def start(backend_kind: str) -> GpuBusySampler:
    reader, source = reader_for(backend_kind)
    return GpuBusySampler(reader, source).start()


def verdict(summary: dict[str, Any], target_pct: float | None) -> bool | None:
    """Whether the run's mean GPU busy stayed at or under the target; None when unknowable."""
    mean = summary.get("gpu_busy_mean_pct")
    if target_pct is None or mean is None:
        return None
    return bool(mean <= target_pct)


__all__ = [
    "GpuBusySampler",
    "INTERVAL_S",
    "parse_ioreg",
    "parse_nvidia_smi",
    "reader_for",
    "start",
    "verdict",
]
