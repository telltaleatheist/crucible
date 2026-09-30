from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import workerio

sys.path.pop(0)
workerio.claim_stdout()

import gc
import threading
import time

from workerio import cap_memory, send

LABEL = "segment"

WHY_REQUIRED = "every parameter is required because each one changes the mask"

SPANS = (("reading", 0.1), ("segmenting", 0.8), ("saving", 0.1))

IMAGENET_MEAN = (0.485, 0.456, 0.406)

IMAGENET_STD = (0.229, 0.224, 0.225)

_STATE: dict = {"engine": None}


class Cancelled(Exception):
    def __init__(self, stage: str) -> None:
        super().__init__(f"cancelled while {stage}")
        self.stage = stage


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


def require(request: dict, key: str, kind):
    return workerio.require(request, key, kind, LABEL, WHY_REQUIRED)


def nullable(request: dict, key: str, kind):
    if key not in request:
        raise KeyError(f"the {LABEL} request has no {key!r}; it is required and null when unset")
    return None if request[key] is None else require(request, key, kind)


class Progress:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._spans = dict(SPANS)
        self._order = [name for name, _ in SPANS]
        self.stage = None
        self.stage_seconds: dict = {}
        self._started = time.time()

    def check(self) -> None:
        if CANCEL.asked(self.request_id):
            raise Cancelled(self.stage)

    def enter(self, stage: str) -> None:
        self.check()
        now = time.time()
        if self.stage is not None:
            self.stage_seconds[self.stage] = round(now - self._started, 2)
        self.stage, self._started = stage, now
        done = sum(self._spans[name] for name in self._order[: self._order.index(stage)])
        send("progress", stage=stage, fraction=round(done, 4))

    def finish(self) -> dict:
        if self.stage is not None:
            self.stage_seconds[self.stage] = round(time.time() - self._started, 2)
        return self.stage_seconds


class Job:
    def __init__(self, request: dict) -> None:
        self.request_id = require(request, "request_id", str)
        self.kind = require(request, "kind", str)
        self.image_path = require(request, "image_path", str)
        self.width = require(request, "width", int)
        self.height = require(request, "height", int)
        self.points = nullable(request, "points", list)
        self.box = nullable(request, "box", list)
        self.mask_path = require(request, "mask_path", str)
        self.cutout_path = require(request, "cutout_path", str)
        self.revision = require(request, "revision", str)
        self.backend = require(request, "backend", str)


def _version_of(distribution: str):
    from importlib import metadata

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _torch_dtype(torch, name: str):
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise RuntimeError(f"torch has no dtype {name!r}; the manifest's dtype reaches this line verbatim")
    return dtype


class TorchEngine:
    """What both engines share: the device, the dtype, the memory cap and the peaks."""

    def __init__(self, request: dict) -> None:
        import torch

        self._torch = torch
        self.model_dir = require(request, "model_dir", str)
        self.device = require(request, "device", str)
        self.dtype = _torch_dtype(torch, require(request, "dtype", str))
        self.working_side = require(request, "working_side", int)
        if self.device == "cuda":
            cap_memory(torch, nullable(request, "memory_cap_bytes", int))
            torch.set_float32_matmul_precision("high")
        self.versions = {
            "torch": torch.__version__,
            "torchvision": _version_of("torchvision"),
            "transformers": _version_of("transformers"),
        }

    def on_device(self, model):
        """Loaded on the CPU by from_pretrained; only the cast copy goes to the device."""
        model.eval()
        return model.to(dtype=self.dtype).to(self.device)

    def peak_bytes(self) -> int:
        torch = self._torch
        if self.device == "cuda":
            return int(torch.cuda.max_memory_reserved(0))
        if self.device == "mps":
            return int(torch.mps.driver_allocated_memory())
        return 0

    def release(self) -> None:
        gc.collect()
        torch = self._torch
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
        elif self.device == "mps":
            torch.mps.empty_cache()


class BiRefNetEngine(TorchEngine):
    """Salient-object matting: one picture in, the main subject's soft mask out."""

    name = "birefnet"

    def __init__(self, request: dict) -> None:
        super().__init__(request)
        import timm
        from transformers import AutoModelForImageSegmentation

        model = AutoModelForImageSegmentation.from_pretrained(self.model_dir, trust_remote_code=True)
        self._model = self.on_device(model)
        self.versions.update({"timm": timm.__version__, "kornia": _version_of("kornia")})

    def mask(self, picture, job: Job):
        import numpy
        from PIL import Image
        from torchvision import transforms

        torch = self._torch
        side = self.working_side
        prepare = transforms.Compose(
            [
                transforms.Resize((side, side)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        batch = prepare(picture.convert("RGB")).unsqueeze(0).to(self.device).to(self.dtype)
        with torch.no_grad():
            predicted = self._model(batch)[-1].sigmoid()
        small = predicted[0].squeeze().float().clamp(0, 1).cpu().numpy()
        del batch, predicted
        levels = numpy.rint(small * 255).astype(numpy.uint8)
        mask = Image.fromarray(levels).resize(picture.size, Image.Resampling.BILINEAR)
        return mask, {"score": None, "multimask": None}


class Sam2Engine(TorchEngine):
    """Promptable selection: the object under the caller's points and/or box."""

    name = "sam2"

    def __init__(self, request: dict) -> None:
        super().__init__(request)
        from transformers import Sam2Model, Sam2Processor

        self._processor = Sam2Processor.from_pretrained(self.model_dir)
        self._model = self.on_device(Sam2Model.from_pretrained(self.model_dir))

    def _prompts(self, job: Job) -> dict:
        prompts: dict = {}
        if job.points:
            prompts["input_points"] = [[[[float(p["x"]), float(p["y"])] for p in job.points]]]
            prompts["input_labels"] = [[[int(p["label"]) for p in job.points]]]
        if job.box:
            prompts["input_boxes"] = [[[float(v) for v in job.box]]]
        return prompts

    def mask(self, picture, job: Job):
        from PIL import Image

        torch = self._torch
        inputs = self._processor(images=picture.convert("RGB"), return_tensors="pt", **self._prompts(job))
        original_sizes = inputs["original_sizes"]
        sent = {
            key: inputs[key].to(self.device)
            for key in ("input_points", "input_labels", "input_boxes")
            if key in inputs
        }
        sent["pixel_values"] = inputs["pixel_values"].to(self.device).to(self.dtype)
        # One click is ambiguous (a shirt, or the person wearing it): SAM's own advice is
        # to ask for its three candidates and keep the best-scored. More points or a box
        # say which, and one mask is the answer.
        multimask = bool(job.points) and len(job.points) == 1 and not job.box
        with torch.no_grad():
            outputs = self._model(**sent, multimask_output=multimask)
        scores = outputs.iou_scores[0, 0].float().cpu()
        best = int(scores.argmax())
        masks = self._processor.post_process_masks(outputs.pred_masks.float().cpu(), original_sizes)[0]
        chosen = masks[0, best].numpy().astype("uint8") * 255
        del sent, outputs
        return Image.fromarray(chosen), {"score": round(float(scores[best]), 4), "multimask": multimask}


ENGINES = {BiRefNetEngine.name: BiRefNetEngine, Sam2Engine.name: Sam2Engine}


def open_picture(path: str):
    from PIL import Image

    with Image.open(path) as opened:
        opened.load()
        return opened.copy()


def cutout_of(picture, mask):
    """The input as RGBA with the mask as its alpha; a pixel the input already made clear stays clear."""
    from PIL import ImageChops

    has_alpha = "A" in picture.getbands() or "transparency" in picture.info
    rgba = picture.convert("RGBA")
    alpha = ImageChops.multiply(rgba.getchannel("A"), mask) if has_alpha else mask
    rgba.putalpha(alpha)
    return rgba


def coverage_of(mask) -> float:
    histogram = mask.histogram()
    total = sum(histogram)
    weighted = sum(level * count for level, count in enumerate(histogram))
    return round(weighted / (255 * total), 4) if total else 0.0


def _run(engine, job: Job) -> dict:
    progress = Progress(job.request_id)
    started = time.time()
    engine.release()
    progress.enter("reading")
    picture = open_picture(job.image_path)
    if picture.size != (job.width, job.height):
        raise RuntimeError(
            f"{job.image_path} opens as {picture.size[0]}x{picture.size[1]} but the job read "
            f"{job.width}x{job.height} from its header; the points were checked against the header"
        )
    progress.enter("segmenting")
    mask, said = engine.mask(picture, job)
    peaks = {"segmenting": engine.peak_bytes()}
    if mask.mode != "L" or mask.size != picture.size:
        raise RuntimeError(
            f"the {engine.name} engine made a {mask.mode} mask of {mask.size}, not an L mask of "
            f"{picture.size}"
        )
    progress.enter("saving")
    mask.save(job.mask_path, format="PNG")
    cutout_of(picture, mask).save(job.cutout_path, format="PNG")
    engine.release()
    width, height = picture.size
    return {
        "mask_path": job.mask_path,
        "cutout_path": job.cutout_path,
        "width": width,
        "height": height,
        "coverage": coverage_of(mask),
        **said,
        "seconds": round(time.time() - started, 2),
        "stage_seconds": progress.finish(),
        "stage_peak_bytes": peaks,
        "peak_bytes": max(peaks.values()) if peaks else None,
        "versions": engine.versions,
    }


def load(request: dict) -> None:
    name = require(request, "engine", str)
    engine_class = ENGINES.get(name)
    if engine_class is None:
        raise RuntimeError(f"no segment engine {name!r}; this worker runs {sorted(ENGINES)}")
    started = time.time()
    try:
        engine = engine_class(request)
    except ImportError as exc:
        raise RuntimeError(
            f"the segment env in {sys.executable} cannot import what it needs ({exc}). "
            "Build it with `crucible install segment`."
        ) from None
    _STATE["engine"] = engine
    send(
        "ready",
        seconds=round(time.time() - started, 2),
        engine=name,
        device=engine.device,
        versions=engine.versions,
    )
    send("done")


def segment(request: dict) -> None:
    engine = _STATE["engine"]
    if engine is None:
        raise RuntimeError(
            "a segment request arrived before a load request; the session's first "
            "exchange loads the engine"
        )
    job = Job(request)
    send("ready", request_id=job.request_id)
    try:
        result = _run(engine, job)
    except Cancelled as stopped:
        send("progress", stage="cancelled", during=stopped.stage)
        send("done")
        return
    finally:
        CANCEL.clear(job.request_id)
    send("result", **result)
    send("done")


OPS = {"load": load, "segment": segment}

INTERRUPTS = {"cancel": lambda request: CANCEL.ask(request.get("request_id"))}


def main() -> int:
    return workerio.serve(LABEL, OPS, INTERRUPTS)


if __name__ == "__main__":
    sys.exit(main())
