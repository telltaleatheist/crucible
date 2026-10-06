from __future__ import annotations

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import workerio

sys.path.pop(0)
workerio.claim_stdout()

import threading
import time

from workerio import cap_memory, memory_line, send

_STATE: dict = {"separator": None, "model_instance": None, "model_filename": None}

HEARTBEAT_SECONDS = 30.0


class _Heartbeat:
    def __init__(self, began: float) -> None:
        self._began = began
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            send(
                "progress",
                stage="separating",
                processed=0,
                total=1,
                elapsed_s=round(time.perf_counter() - self._began, 1),
            )


def require(request: dict, key: str, kind):
    return workerio.require(request, key, kind, "denoise", "every key is required")


def load(request: dict) -> None:
    model_file_dir = require(request, "model_file_dir", str)
    model_filename = require(request, "model_filename", str)
    use_autocast = require(request, "use_autocast", bool)
    overlap = require(request, "overlap", int)
    if "memory_cap_bytes" not in request:
        raise KeyError(
            "the denoise request has no 'memory_cap_bytes'; it is required, and "
            "null where there is no CUDA cap"
        )
    memory_cap_bytes = request["memory_cap_bytes"]
    import torch

    fraction = cap_memory(torch, memory_cap_bytes)

    started = time.perf_counter()
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise RuntimeError(
            f"the rvc env in {sys.executable} cannot import audio_separator "
            f"({exc}). Build it with `crucible install rvc`."
        ) from None

    # The library's own MDXC settings with only the overlap replaced (the manifest's), so
    # its other defaults stay whatever this version of audio-separator says they are.
    mdxc_params = dict(inspect.signature(Separator.__init__).parameters["mdxc_params"].default)
    if "overlap" not in mdxc_params:
        raise RuntimeError(
            f"audio-separator's default mdxc_params {sorted(mdxc_params)} has no "
            "'overlap'; this worker cannot set the manifest's overlap on this version"
        )
    mdxc_params["overlap"] = overlap
    separator = Separator(
        model_file_dir=model_file_dir,
        output_dir=os.path.join(model_file_dir, ".crucible-separator-unset"),
        output_format="flac",
        use_autocast=use_autocast,
        mdxc_params=mdxc_params,
    )
    separator.load_model(model_filename=model_filename)

    model_instance = getattr(separator, "model_instance", None)
    if model_instance is None:
        raise RuntimeError(
            "audio-separator loaded no model_instance — this worker cannot "
            "separate, and cannot re-point a per-request output directory"
        )
    if not hasattr(model_instance, "output_dir"):
        raise RuntimeError(
            "audio-separator's model instance has no output_dir attribute — this "
            "worker's per-request output directory mechanism no longer applies "
            "to this version. Every block's stems would land in one directory "
            "and each job would claim the previous job's outputs"
        )

    seconds = time.perf_counter() - started
    memory_line(torch, "after load")
    _STATE.update(
        separator=separator,
        model_instance=model_instance,
        model_filename=model_filename,
    )
    send(
        "ready",
        seconds=seconds,
        memory_cap_bytes=memory_cap_bytes,
        memory_fraction=fraction,
        alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    )
    send("done")


def separate(request: dict) -> None:
    separator = _STATE["separator"]
    if separator is None:
        raise RuntimeError(
            "a separate request arrived before a load request; the session's "
            "first exchange loads the model"
        )
    model_instance = _STATE["model_instance"]
    model_filename = _STATE["model_filename"]

    source = require(request, "input", str)
    output_dir = require(request, "output_dir", str)
    output_format = require(request, "output_format", str)
    expected_rate = require(request, "sample_rate", int)

    import soundfile

    try:
        info = soundfile.info(source)
    except Exception as exc:
        raise RuntimeError(
            f"{os.path.basename(source)} could not be read as audio: "
            f"{type(exc).__name__}: {exc}"
        ) from None

    send(
        "ready",
        sample_rate=int(info.samplerate),
        frames=int(info.frames),
        seconds=round(float(info.frames) / float(info.samplerate), 3)
        if info.samplerate
        else 0.0,
        channels=int(info.channels),
    )

    if int(info.samplerate) != expected_rate:
        raise RuntimeError(
            f"this input is {info.samplerate} Hz and {model_filename} is "
            f"{expected_rate} Hz native. Nothing was resampled: a stem returned "
            "at a rate the caller did not send is a stem whose sample offsets no "
            "longer mean anything. Resample before sending"
        )

    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        raise RuntimeError(
            f"{output_dir} is not empty; this worker identifies the separator's "
            "outputs by what appears in it, so it must start empty"
        )

    separator.output_dir = output_dir
    model_instance.output_dir = output_dir
    separator.output_format = output_format
    model_instance.output_format = output_format

    send("progress", stage="separating", processed=0, total=1)
    began = time.perf_counter()
    try:
        with _Heartbeat(began):
            separator.separate(source)
    except Exception as exc:
        raise RuntimeError(
            f"audio-separator failed on {os.path.basename(source)}: "
            f"{type(exc).__name__}: {exc}"
        ) from None
    separate_seconds = time.perf_counter() - began

    stems = []
    for name in sorted(os.listdir(output_dir)):
        produced = os.path.join(output_dir, name)
        if not os.path.isfile(produced):
            continue
        try:
            stem_info = soundfile.info(produced)
        except Exception as exc:
            raise RuntimeError(
                f"{name} came out of the separator but could not be read back as "
                f"audio: {type(exc).__name__}: {exc}"
            ) from None
        stems.append(
            {
                "name": name,
                "sample_rate": int(stem_info.samplerate),
                "frames": int(stem_info.frames),
                "channels": int(stem_info.channels),
                "bytes": os.path.getsize(produced),
            }
        )
    if not stems:
        raise RuntimeError(
            f"audio-separator finished and wrote nothing into {output_dir}"
        )
    wrong = [s["name"] for s in stems if not s["name"].lower().endswith("." + output_format.lower())]
    if wrong:
        raise RuntimeError(
            f"the separator was asked for {output_format} and wrote {wrong}; the "
            "stem container is part of the contract and is not substituted"
        )

    import torch

    memory_line(torch, os.path.basename(source))
    send("result", stems=stems, separate_seconds=round(separate_seconds, 2))
    send("done")


OPS = {"load": load, "separate": separate}


def main() -> int:
    return workerio.serve("denoise", OPS)


if __name__ == "__main__":
    sys.exit(main())
