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

LABEL = "image"

WHY_REQUIRED = "every parameter is required because each one changes the picture"

_STATE: dict = {"engine": None}

_CANCEL = {"request_id": None}

_CANCEL_LOCK = threading.Lock()


class Cancelled(Exception):
    def __init__(self, stage: str, step: int) -> None:
        super().__init__(f"cancelled while {stage}, after step {step}")
        self.stage = stage
        self.step = step


def require(request: dict, key: str, kind):
    return workerio.require(request, key, kind, LABEL, WHY_REQUIRED)


def optional(request: dict, key: str, kind):
    if request.get(key) is None:
        if key not in request:
            raise KeyError(f"the {LABEL} request has no {key!r}; it is required and null when unset")
        return None
    return require(request, key, kind)


class Progress:
    def __init__(self, request_id: str, steps: int) -> None:
        self.request_id = request_id
        self.steps = steps
        self.stage = None
        self.step = 0
        self.stage_seconds: dict = {}
        self._stage_started = time.time()

    def _check(self) -> None:
        with _CANCEL_LOCK:
            asked = _CANCEL["request_id"] == self.request_id
        if asked:
            raise Cancelled(self.stage, self.step)

    def enter(self, stage: str) -> None:
        self._check()
        now = time.time()
        if self.stage is not None:
            self.stage_seconds[self.stage] = round(now - self._stage_started, 2)
        self.stage, self._stage_started = stage, now
        send("progress", stage=stage, step=self.step, steps=self.steps)

    def stepped(self) -> None:
        self.step += 1
        send("progress", stage=self.stage, step=self.step, steps=self.steps)
        self._check()

    def finish(self) -> dict:
        self.stage_seconds[self.stage] = round(time.time() - self._stage_started, 2)
        return self.stage_seconds


class Job:
    def __init__(self, request: dict) -> None:
        self.request_id = require(request, "request_id", str)
        self.prompt = require(request, "prompt", str)
        self.negative_prompt = optional(request, "negative_prompt", str)
        self.width = require(request, "width", int)
        self.height = require(request, "height", int)
        self.seed = require(request, "seed", int)
        self.steps = require(request, "steps", int)
        self.guidance = float(require(request, "guidance", (int, float)))
        self.image_path = optional(request, "image_path", str)
        self.image_strength = optional(request, "image_strength", (int, float))
        self.output_path = require(request, "output_path", str)


class MfluxEngine:
    name = "mflux"

    def __init__(self, request: dict) -> None:
        import mflux
        import mlx.core as mx
        from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

        self._mx = mx
        self._model_class = QwenImage21
        self._model_dir = require(request, "model_dir", str)
        mx.set_cache_limit(require(request, "mlx_cache_limit_bytes", int))
        self.device = "metal"
        self.versions = {"mflux": _version_of(mflux, "mflux"), "mlx": mx.__version__}

    def peak_bytes(self) -> int:
        return int(self._mx.get_peak_memory())

    def _release(self) -> None:
        gc.collect()
        self._mx.clear_cache()

    def generate(self, job: Job, progress: Progress) -> "tuple[object, dict]":
        mx = self._mx
        peaks: dict = {}
        mx.reset_peak_memory()
        progress.enter("encoding")
        model = self._model_class(model_path=self._model_dir)
        model.callbacks.register(_MfluxStages(self, model, progress, peaks))
        try:
            generated = model.generate_image(
                seed=job.seed,
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                width=job.width,
                height=job.height,
                num_inference_steps=job.steps,
                guidance=job.guidance,
                image_path=job.image_path,
                image_strength=job.image_strength,
            )
            peaks["decoding"] = self.peak_bytes()
            return generated.image, peaks
        finally:
            del model
            self._release()


def _cached_embeddings(model) -> list:
    return [array for pair in model.prompt_cache.values() for array in pair if array is not None]


class _MfluxStages:
    def __init__(self, engine: MfluxEngine, model, progress: Progress, peaks: dict) -> None:
        self._engine = engine
        self._model = model
        self._progress = progress
        self._peaks = peaks

    def _close_stage(self, name: str) -> None:
        self._peaks[name] = self._engine.peak_bytes()
        self._engine._release()
        self._engine._mx.reset_peak_memory()

    def call_before_loop(self, **_: object) -> None:
        self._model.text_encoder = None
        self._engine._mx.eval(*_cached_embeddings(self._model))
        self._close_stage("encoding")
        self._progress.enter("denoising")

    def call_in_loop(self, latents, **_: object) -> None:
        self._engine._mx.eval(latents)
        self._progress.stepped()

    def call_after_loop(self, **_: object) -> None:
        self._model.transformer = None
        self._close_stage("denoising")
        self._progress.enter("decoding")


class DiffusersEngine:
    name = "diffusers"

    def __init__(self, request: dict) -> None:
        import diffusers
        import torch
        from diffusers import QwenImage21Pipeline

        self._torch = torch
        self._diffusers = diffusers
        self._model_dir = require(request, "model_dir", str)
        self._dtype = _torch_dtype(torch, require(request, "dtype", str))
        self.device = require(request, "device", str)
        cap_memory(torch, optional(request, "memory_cap_bytes", int))
        self._pipe = QwenImage21Pipeline.from_pretrained(
            self._model_dir,
            text_encoder=None,
            transformer=None,
            vae=None,
            torch_dtype=self._dtype,
        )
        self.versions = {
            "diffusers": diffusers.__version__,
            "torch": torch.__version__,
            "transformers": _version_of(None, "transformers"),
        }

    def peak_bytes(self) -> int:
        return int(self._torch.cuda.max_memory_reserved(0))

    def _release(self) -> None:
        gc.collect()
        self._torch.cuda.empty_cache()
        self._torch.cuda.reset_peak_memory_stats(0)

    def _component(self, loader, subfolder: str):
        return loader.from_pretrained(
            self._model_dir,
            subfolder=subfolder,
            torch_dtype=self._dtype,
            device_map=self.device,
        )

    def _encode(self, job: Job) -> dict:
        from transformers import Qwen3VLForConditionalGeneration

        pipe = self._pipe
        pipe.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(self._model_dir, "text_encoder"),
            dtype=self._dtype,
            device_map=self.device,
        )
        try:
            with self._torch.no_grad():
                return self._embeddings(job)
        finally:
            pipe.text_encoder = None

    def _embeddings(self, job: Job) -> dict:
        pipe = self._pipe
        embeds, mask, _ = pipe.encode_prompt(prompt=job.prompt, device=self.device)
        encoded = {"prompt_embeds": embeds, "prompt_embeds_mask": mask}
        if job.negative_prompt is not None:
            negative, negative_mask, _ = pipe.encode_prompt(
                prompt=job.negative_prompt, device=self.device
            )
            encoded["negative_prompt_embeds"] = negative
            encoded["negative_prompt_embeds_mask"] = negative_mask
        return encoded

    def _denoise(self, job: Job, encoded: dict, progress: Progress):
        pipe = self._pipe
        pipe.transformer = self._component(
            self._diffusers.QwenImage21Transformer2DModel, "transformer"
        )

        def on_step(pipeline, index, timestep, tensors):
            progress.stepped()
            return tensors

        try:
            return pipe(
                **encoded,
                true_cfg_scale=job.guidance,
                width=job.width,
                height=job.height,
                num_inference_steps=job.steps,
                generator=self._torch.Generator(self.device).manual_seed(job.seed),
                output_type="latent",
                callback_on_step_end=on_step,
            ).images
        finally:
            pipe.transformer = None

    def _decode(self, job: Job, latents):
        pipe = self._pipe
        torch = self._torch
        vae = self._component(self._diffusers.AutoencoderKLQwenImage21, "vae")
        try:
            latents = pipe._unpack_latents(latents, job.height, job.width, pipe.vae_scale_factor)
            latents = latents.to(vae.dtype)
            shape = (1, vae.config.z_dim, 1, 1, 1)
            mean = torch.tensor(vae.config.latents_mean).view(shape).to(latents.device, latents.dtype)
            std = torch.tensor(vae.config.latents_std).view(shape).to(latents.device, latents.dtype)
            with torch.no_grad():
                decoded = vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
            return pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        finally:
            del vae

    def generate(self, job: Job, progress: Progress) -> "tuple[object, dict]":
        peaks: dict = {}
        self._release()
        stages = (
            ("encoding", lambda _: self._encode(job)),
            ("denoising", lambda encoded: self._denoise(job, encoded, progress)),
            ("decoding", lambda latents: self._decode(job, latents)),
        )
        carried = None
        for stage, run in stages:
            progress.enter(stage)
            carried = run(carried)
            peaks[stage] = self.peak_bytes()
            self._release()
        return carried, peaks


ENGINES = {MfluxEngine.name: MfluxEngine, DiffusersEngine.name: DiffusersEngine}


def _version_of(module, distribution: str) -> str | None:
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)
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


def load(request: dict) -> None:
    engine_name = require(request, "engine", str)
    engine_class = ENGINES.get(engine_name)
    if engine_class is None:
        raise RuntimeError(f"no image engine {engine_name!r}; this worker runs {sorted(ENGINES)}")
    started = time.time()
    try:
        engine = engine_class(request)
    except ImportError as exc:
        raise RuntimeError(
            f"the image env in {sys.executable} cannot import what it needs ({exc}). "
            "Build it with `crucible install image`."
        ) from None
    _STATE["engine"] = engine
    send(
        "ready",
        seconds=round(time.time() - started, 2),
        engine=engine_name,
        device=engine.device,
        versions=engine.versions,
    )
    send("done")


def _run(engine, job: Job) -> dict:
    progress = Progress(job.request_id, job.steps)
    started = time.time()
    image, peaks = engine.generate(job, progress)
    progress.enter("saving")
    image.save(job.output_path, format="PNG")
    width, height = image.size
    return {
        "path": job.output_path,
        "width": width,
        "height": height,
        "seconds": round(time.time() - started, 2),
        "stage_seconds": progress.finish(),
        "stage_peak_bytes": peaks,
        "peak_bytes": max(peaks.values()) if peaks else None,
    }


def generate(request: dict) -> None:
    engine = _STATE["engine"]
    if engine is None:
        raise RuntimeError(
            "a generate request arrived before a load request; the session's first "
            "exchange loads the engine"
        )
    job = Job(request)
    send("ready", steps=job.steps)
    try:
        result = _run(engine, job)
    except Cancelled as stopped:
        send("progress", stage="cancelled", step=stopped.step, steps=job.steps, during=stopped.stage)
        send("done")
        return
    finally:
        with _CANCEL_LOCK:
            if _CANCEL["request_id"] == job.request_id:
                _CANCEL["request_id"] = None
    send("result", **result)
    send("done")


def cancel(request: dict) -> None:
    with _CANCEL_LOCK:
        _CANCEL["request_id"] = request.get("request_id")


OPS = {"load": load, "generate": generate}

INTERRUPTS = {"cancel": cancel}


def main() -> int:
    return workerio.serve(LABEL, OPS, INTERRUPTS)


if __name__ == "__main__":
    sys.exit(main())
