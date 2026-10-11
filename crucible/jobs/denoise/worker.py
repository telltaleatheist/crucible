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

stemaudio = workerio.load_sibling("stemaudio", __file__)

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
    hop_s = require(request, "hop_s", int)
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

    # The library's own MDXC settings with only `overlap` replaced, so its other defaults
    # stay whatever this version of audio-separator says they are. For a Roformer that key
    # is the hop in seconds between windows (step = overlap * sample_rate), not a count.
    mdxc_params = dict(inspect.signature(Separator.__init__).parameters["mdxc_params"].default)
    if "overlap" not in mdxc_params:
        raise RuntimeError(
            f"audio-separator's default mdxc_params {sorted(mdxc_params)} has no "
            "'overlap'; this worker cannot set the manifest's hop_s on this version"
        )
    mdxc_params["overlap"] = hop_s
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

    source = require(request, "input", str)
    output_dir = require(request, "output_dir", str)
    work_dir = require(request, "work_dir", str)
    separator_format = require(request, "separator_format", str)
    output_format = require(request, "output_format", str)
    separator_rate = require(request, "sample_rate", int)
    ffmpeg = require(request, "ffmpeg", str)
    ffprobe = require(request, "ffprobe", str)

    for directory in (work_dir, output_dir):
        os.makedirs(directory, exist_ok=True)
        if os.listdir(directory):
            raise RuntimeError(
                f"{directory} is not empty; this worker identifies the separator's "
                "outputs by what appears in it, so it must start empty"
            )
    separated_dir = os.path.join(work_dir, "separated")
    os.makedirs(separated_dir)

    plan = stemaudio.prepare(ffmpeg, ffprobe, source, work_dir, separator_rate)
    send(
        "ready",
        sample_rate=plan["sample_rate"],
        frames=plan["frames"],
        seconds=round(plan["frames"] / float(plan["sample_rate"]), 3),
        channels=plan["channels"],
        codec=plan["codec"],
        container=plan["container"],
        decoded=plan["decoded"],
        resampled=plan["resampled"],
        separator_rate=plan["separator_rate"],
        separator_frames=plan["separator_frames"],
    )

    # The separator always writes lossless WAV: the one lossy encode a client may ask for
    # (mp3) happens once, after, in stemaudio.finish.
    separator.output_dir = separated_dir
    model_instance.output_dir = separated_dir
    separator.output_format = separator_format
    model_instance.output_format = separator_format

    send("progress", stage="separating", processed=0, total=1)
    # The logger the library's stem writer reports to; a stem it skips as near-silent is
    # filled in as silence below rather than lost (stemaudio.NearSilentWatch).
    watch = stemaudio.NearSilentWatch()
    library_logger = model_instance.logger
    library_logger.addHandler(watch)
    began = time.perf_counter()
    try:
        with _Heartbeat(began):
            separator.separate(plan["separator_input"])
    except Exception as exc:
        raise RuntimeError(
            f"audio-separator failed on {os.path.basename(source)}: "
            f"{type(exc).__name__}: {exc}"
        ) from None
    finally:
        library_logger.removeHandler(watch)
    separate_seconds = time.perf_counter() - began
    silent_stems = stemaudio.fill_silent(separated_dir, watch.silent, plan)

    produced = sorted(
        name for name in os.listdir(separated_dir)
        if os.path.isfile(os.path.join(separated_dir, name))
    )
    if not produced:
        raise RuntimeError(
            f"audio-separator finished and wrote nothing into {separated_dir}"
        )
    wrong = [name for name in produced if not name.lower().endswith("." + separator_format.lower())]
    if wrong:
        raise RuntimeError(
            f"the separator was asked for {separator_format} and wrote {wrong}; the "
            "stem container is part of the contract and is not substituted"
        )

    stems = []
    for name in produced:
        try:
            stems.append(
                stemaudio.finish(
                    os.path.join(separated_dir, name), output_dir, plan, output_format
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"{name} came out of the separator but could not be made into a "
                f"{output_format} stem: {type(exc).__name__}: {exc}"
            ) from None

    import torch

    memory_line(torch, os.path.basename(source))
    send(
        "result",
        stems=stems,
        silent_stems=silent_stems,
        separate_seconds=round(separate_seconds, 2),
    )
    send("done")


OPS = {"load": load, "separate": separate}


def main() -> int:
    return workerio.serve("denoise", OPS)


if __name__ == "__main__":
    sys.exit(main())
