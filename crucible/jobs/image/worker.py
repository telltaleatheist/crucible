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
from collections import OrderedDict

from workerio import cap_memory, send

inpaint = workerio.load_sibling("inpaint", __file__)

LABEL = "image"

GENERATED_NAME = "generated.png"

WHY_REQUIRED = "every parameter is required because each one changes the picture"

_STATE: dict = {"engine": None}

_CANCEL = {"request_id": None}

PROMPT_CACHE_ENTRIES = 32

PROMPT_CACHE_BYTES = 256 * 1024 * 1024

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


class PromptCache:
    def __init__(self, entries: int = PROMPT_CACHE_ENTRIES, byte_cap: int = PROMPT_CACHE_BYTES) -> None:
        self.entries = entries
        self.byte_cap = byte_cap
        self._held: OrderedDict = OrderedDict()
        self.bytes = 0

    def __len__(self) -> int:
        return len(self._held)

    def __contains__(self, key) -> bool:
        return key in self._held

    def get(self, key):
        found = self._held.get(key)
        if found is None:
            return None
        self._held.move_to_end(key)
        return found[0]

    def put(self, key, value, size: int) -> None:
        if size > self.byte_cap:
            return
        if key in self._held:
            self.bytes -= self._held.pop(key)[1]
        self._held[key] = (value, size)
        self.bytes += size
        while len(self._held) > self.entries or self.bytes > self.byte_cap:
            self.bytes -= self._held.popitem(last=False)[1][1]

    def clear(self) -> None:
        self._held.clear()
        self.bytes = 0


_PROMPTS = PromptCache()


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
        self.mask_path = optional(request, "mask_path", str)
        self.mask_blur = optional(request, "mask_blur", int)
        self.mask = None
        self.blend_steps = None
        self.output_path = require(request, "output_path", str)
        self.revision = require(request, "revision", str)
        self.backend = require(request, "backend", str)

    @property
    def start_step(self) -> int:
        """mflux's init_time_step: the step image-to-image starts at, 0 for pure noise."""
        if self.image_strength is None:
            return 0
        return max(1, int(self.steps * self.image_strength))

    @property
    def prompt_key(self) -> tuple:
        return (self.prompt, self.negative_prompt, self.guidance > 1.0, self.revision, self.backend)


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

    def nbytes(self, encoded: dict) -> int:
        return sum(int(array.nbytes) for pair in encoded.values() for array in pair if array is not None)

    def generate(self, job: Job, progress: Progress, cached: "dict | None") -> "tuple[object, dict, dict]":
        mx = self._mx
        peaks: dict = {}
        encoded: dict = {}
        mx.reset_peak_memory()
        progress.enter("encoding")
        model = self._model_class(model_path=self._model_dir)
        if cached is not None:
            model.prompt_cache.update(cached)
            model.text_encoder = None
        try:
            model.callbacks.register(_MfluxStages(self, model, progress, peaks, encoded))
            if job.mask is None:
                picture = model.generate_image(
                    seed=job.seed,
                    prompt=job.prompt,
                    negative_prompt=job.negative_prompt,
                    width=job.width,
                    height=job.height,
                    num_inference_steps=job.steps,
                    guidance=job.guidance,
                    image_path=job.image_path,
                    image_strength=job.image_strength,
                ).image
            else:
                repaint = self._repaint(model, job)
                picture = _mflux_generate(model, job, repaint)
                job.blend_steps = repaint.calls
            peaks["decoding"] = self.peak_bytes()
            return picture, peaks, encoded
        finally:
            del model
            self._release()

    def _repaint(self, model, job: Job) -> "_MfluxRepaint":
        """What the per-step blend needs, built the way mflux builds its image-to-image start:
        the input VAE-encoded and packed (LatentCreator.encode_image, Qwen21LatentCreator), and
        the very noise generate_image starts from (create_noise with the job's seed)."""
        from mflux.models.common.latent_creator.latent_creator import LatentCreator
        from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator

        mx = self._mx
        encoded = LatentCreator.encode_image(
            vae=model.vae,
            image_path=job.image_path,
            height=job.height,
            width=job.width,
            tiling_config=model.tiling_config,
        )
        clean = Qwen21LatentCreator.pack_latents(encoded, job.height, job.width)
        noise = Qwen21LatentCreator.create_noise(job.seed, job.height, job.width)
        # the mask packed by mflux's own pack_latents, spread over the channels as the clean
        # latents are, so its token order is theirs whatever that order is
        channels = clean.shape[-1]
        height, width = job.mask.grid.shape
        spread = mx.broadcast_to(mx.array(job.mask.grid)[None, None], (1, channels, height, width))
        mask = Qwen21LatentCreator.pack_latents(spread, job.height, job.width, channels)[:, :, :1]
        mask = mask.astype(clean.dtype)
        mx.eval(clean, noise, mask)
        return _MfluxRepaint(clean, noise, mask)


class _MfluxRepaint:
    """The latent blend on mflux: takes the latents a scheduler step made and returns the
    ones the next step reads. After the step at `t` they are at `sigmas[t + 1]` (the shifted
    schedule with a 0 appended, so the last step puts the input back clean)."""

    def __init__(self, clean, noise, mask) -> None:
        self.clean = clean
        self.noise = noise
        self.mask = mask
        self.calls = 0

    def __call__(self, latents, step: int, config):
        self.calls += 1
        sigma = float(config.scheduler.sigmas[step + 1].item())
        blended = inpaint.blend_step(latents, self.clean, self.noise, self.mask, sigma)
        return blended.astype(latents.dtype)


def _mflux_denoise(latents, config, predict, ctx, evaluate, blend=None):
    """The denoising loop of mflux 0.20.0's `QwenImage21.generate_image`, step for step,
    with the blend applied to the value the loop carries into the next step.

    mflux's own loop hands the in-loop callback its latents and reads nothing back, so a
    blend there cannot reach the next step (an in-place write did not, on the Mac,
    2026-09-29). Here the blend's return value is what the next step reads.
    """
    for t in config.time_steps:
        latents = config.scheduler.scale_model_input(latents, t)
        noise = predict(t, latents)
        latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
        if blend is not None:
            latents = blend(latents, t, config)
        ctx.in_loop(t, latents)
        evaluate(latents)
    return latents


def _mflux_parts():
    from types import SimpleNamespace

    import mlx.core as mx
    from mflux.models.common.config import ModelConfig
    from mflux.models.common.config.config import Config
    from mflux.models.common.latent_creator.latent_creator import LatentCreator
    from mflux.models.common.vae.vae_util import VAEUtil
    from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
    from mflux.utils.image_util import ImageUtil

    return SimpleNamespace(
        mx=mx,
        ModelConfig=ModelConfig,
        Config=Config,
        LatentCreator=LatentCreator,
        VAEUtil=VAEUtil,
        Qwen21LatentCreator=Qwen21LatentCreator,
        Qwen21PromptEncoder=Qwen21PromptEncoder,
        ImageUtil=ImageUtil,
    )


def _mflux_generate(model, job: Job, blend, parts=None):
    """mflux 0.20.0's `QwenImage21.generate_image` for a masked job: the same config, start
    latents, prompt encoding, callbacks, loop and decode, with `blend` inside the loop.
    Returns the PIL picture. Unmasked jobs still call mflux's own `generate_image`."""
    parts = parts or _mflux_parts()
    config = parts.Config(
        width=job.width,
        height=job.height,
        guidance=job.guidance,
        scheduler="linear",
        image_path=job.image_path if job.image_strength is not None else None,
        image_strength=job.image_strength,
        model_config=model.model_config,
        num_inference_steps=job.steps,
    )
    # LatentCreator.create_for_txt2img_or_img2img, from the clean latents and the noise the
    # blend already holds (built the same way), so the input is VAE-encoded once
    if config.init_time_step == 0:
        latents = blend.noise
    else:
        sigma = config.scheduler.sigmas[config.init_time_step]
        latents = parts.LatentCreator.add_noise_by_interpolation(clean=blend.clean, noise=blend.noise, sigma=sigma)
    latents = latents.astype(parts.ModelConfig.precision)
    encode = parts.Qwen21PromptEncoder.encode_prompt
    prompt_embeds, prompt_mask = encode(
        prompt=job.prompt,
        prompt_cache=model.prompt_cache,
        tokenizer=model.tokenizers["qwen21"],
        text_encoder=model.text_encoder,
    )
    do_true_cfg = config.guidance > 1.0 and bool(job.negative_prompt)
    if do_true_cfg:
        negative_embeds, negative_mask = encode(
            prompt=job.negative_prompt,
            prompt_cache=model.prompt_cache,
            tokenizer=model.tokenizers["qwen21"],
            text_encoder=model.text_encoder,
        )

    def predict(t, hidden):
        noise = model.transformer(
            t=t,
            config=config,
            hidden_states=hidden,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
        )
        if do_true_cfg:
            negative = model.transformer(
                t=t,
                config=config,
                hidden_states=hidden,
                encoder_hidden_states=negative_embeds,
                encoder_hidden_states_mask=negative_mask,
            )
            noise = negative + config.guidance * (noise - negative)
        return noise

    ctx = model.callbacks.start(seed=job.seed, prompt=job.prompt, config=config)
    ctx.before_loop(latents)
    latents = _mflux_denoise(latents, config, predict, ctx, parts.mx.eval, blend)
    ctx.after_loop(latents)
    latents = parts.Qwen21LatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
    decoded = parts.VAEUtil.decode(vae=model.vae, latent=latents, tiling_config=model.tiling_config)
    return parts.ImageUtil.to_pil(decoded)


def _cached_embeddings(model) -> list:
    return [array for pair in model.prompt_cache.values() for array in pair if array is not None]


class _MfluxStages:
    def __init__(self, engine: MfluxEngine, model, progress: Progress, peaks: dict, encoded: dict) -> None:
        self._engine = engine
        self._model = model
        self._progress = progress
        self._peaks = peaks
        self._encoded = encoded

    def _close_stage(self, name: str) -> None:
        self._peaks[name] = self._engine.peak_bytes()
        self._engine._release()
        self._engine._mx.reset_peak_memory()

    def call_before_loop(self, **_: object) -> None:
        self._model.text_encoder = None
        self._engine._mx.eval(*_cached_embeddings(self._model))
        self._encoded.update(self._model.prompt_cache)
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

    def nbytes(self, encoded: dict) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in encoded.values() if tensor is not None)

    def _encoded(self, job: Job, cached: "dict | None") -> dict:
        if cached is not None:
            return _moved(cached, self.device)
        return self._encode(job)

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

    def _latent_side(self, pixels: int) -> int:
        return 2 * (int(pixels) // (self._pipe.vae_scale_factor * 2))

    def _encode_start_image(self, job: Job):
        """The input picture as packed, normalised latents at the job's size.

        Stretched to width x height like mflux's scale_to_dimensions, so both arms
        start from the same framing. RGBA, as the pipeline itself converts its
        condition images: the 2.1 VAE's first convolution takes four channels.
        """
        pipe = self._pipe
        vae = self._component(self._diffusers.AutoencoderKLQwenImage21, "vae")
        pipe.vae = vae
        try:
            picture = inpaint.open_picture(job.image_path).convert("RGBA")
            pixels = pipe.image_processor.preprocess(picture, height=job.height, width=job.width)
            pixels = pixels.unsqueeze(2).to(device=self.device, dtype=vae.dtype)
            with self._torch.no_grad():
                latents = pipe._encode_vae_image(pixels, generator=None)
            channels = latents.shape[1]
            side_h, side_w = self._latent_side(job.height), self._latent_side(job.width)
            return pipe._pack_latents(latents, 1, channels, side_h, side_w)
        finally:
            pipe.vae = None
            del vae

    def _start_schedule(self, job: Job, clean):
        """mflux's image-to-image start, on the diffusers scheduler.

        mflux begins at step `max(1, int(steps * strength))` of the full schedule and
        blends the input with noise at that step's sigma, so a higher strength keeps
        more of the input. A masked job without a strength starts at step 0, whose
        sigma is 1: pure noise, the whole region regenerated. The pipeline shifts
        whatever sigmas it is handed by the image's sequence length; the shift and the
        terminal stretch are per-sigma given the same last value, so handing it the tail
        of the unshifted schedule reproduces the tail of the shifted one. The start sigma is read from a copy
        of the scheduler set exactly as the pipeline will set it.
        """
        import copy

        import numpy as np
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
        from diffusers.utils.torch_utils import randn_tensor

        pipe = self._pipe
        torch = self._torch
        tail = np.linspace(1.0, 1 / job.steps, job.steps)[job.start_step :].tolist()
        config = pipe.scheduler.config
        mu = calculate_shift(
            clean.shape[1],
            config.get("base_image_seq_len", 256),
            config.get("max_image_seq_len", 4096),
            config.get("base_shift", 0.5),
            config.get("max_shift", 1.15),
        )
        probe = copy.deepcopy(pipe.scheduler)
        probe.set_timesteps(sigmas=tail, device=self.device, mu=mu)
        sigma = float(probe.sigmas[0])
        channels = clean.shape[2]
        side_h, side_w = self._latent_side(job.height), self._latent_side(job.width)
        generator = torch.Generator(self.device).manual_seed(job.seed)
        noise = randn_tensor(
            (1, 1, channels, side_h, side_w), generator=generator, device=self.device, dtype=clean.dtype
        )
        noise = pipe._pack_latents(noise, 1, channels, side_h, side_w)
        return (1.0 - sigma) * clean + sigma * noise, tail, noise

    def _denoise(self, job: Job, encoded: dict, progress: Progress):
        pipe = self._pipe
        start = {}
        repaint = None
        if job.image_path is not None and (job.image_strength is not None or job.mask is not None):
            clean = self._encode_start_image(job)
            self._release()
            latents, sigmas, noise = self._start_schedule(job, clean)
            start = {"latents": latents, "sigmas": sigmas}
            if job.mask is not None:
                mask = self._torch.from_numpy(job.mask.latent).to(device=self.device, dtype=clean.dtype)
                repaint = _DiffusersRepaint(clean, noise, mask)
        pipe.transformer = self._component(
            self._diffusers.QwenImage21Transformer2DModel, "transformer"
        )

        def on_step(pipeline, index, timestep, tensors):
            if repaint is not None:
                tensors["latents"] = repaint(pipeline, index, tensors["latents"])
            progress.stepped()
            return tensors

        try:
            made = pipe(
                **encoded,
                **start,
                true_cfg_scale=job.guidance,
                width=job.width,
                height=job.height,
                num_inference_steps=job.steps,
                generator=self._torch.Generator(self.device).manual_seed(job.seed),
                output_type="latent",
                callback_on_step_end=on_step,
            ).images
            job.blend_steps = None if repaint is None else repaint.calls
            return made
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

    def generate(self, job: Job, progress: Progress, cached: "dict | None") -> "tuple[object, dict, dict]":
        peaks: dict = {}
        kept: dict = {}
        self._release()

        def encode(_):
            encoded = self._encoded(job, cached)
            kept.update(cached if cached is not None else _moved(encoded, "cpu"))
            return encoded

        stages = (
            ("encoding", encode),
            ("denoising", lambda encoded: self._denoise(job, encoded, progress)),
            ("decoding", lambda latents: self._decode(job, latents)),
        )
        carried = None
        for stage, run in stages:
            progress.enter(stage)
            carried = run(carried)
            peaks[stage] = self.peak_bytes()
            self._release()
        return carried, peaks, kept


class _DiffusersRepaint:
    """The latent blend on diffusers, from `callback_on_step_end`.

    QwenImageInpaintPipeline's loop, without forking QwenImage21Pipeline: after the scheduler
    step at index i the latents are at `scheduler.sigmas[i + 1]` (the step index scale_noise
    reads under set_begin_index(0); the last is the appended 0, so the last step puts the input
    back clean), and outside the mask they are replaced by the input noised to that sigma with
    the noise the start was made from. The pipeline reads `latents` back from the callback.
    """

    def __init__(self, clean, noise, mask) -> None:
        self._clean = clean
        self._noise = noise
        self._mask = mask
        self.calls = 0

    def __call__(self, pipeline, index: int, latents):
        self.calls += 1
        sigma = float(pipeline.scheduler.sigmas[index + 1])
        blended = inpaint.blend_step(latents, self._clean, self._noise, self._mask, sigma)
        return blended.to(latents.dtype)


def _moved(tensors: dict, device: str) -> dict:
    return {name: None if tensor is None else tensor.to(device) for name, tensor in tensors.items()}


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
    _PROMPTS.clear()
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
    if job.mask_path is not None:
        job.mask = inpaint.load_mask(job.mask_path, job.image_path, job.width, job.height, job.mask_blur)
    cached = _PROMPTS.get(job.prompt_key)
    image, peaks, encoded = engine.generate(job, progress, cached)
    if cached is None and encoded:
        _PROMPTS.put(job.prompt_key, encoded, engine.nbytes(encoded))
    progress.enter("saving")
    drift = None
    if job.mask is not None:
        image.save(os.path.join(os.path.dirname(job.output_path), GENERATED_NAME), format="PNG")
        drift = inpaint.outside_drift(image, job.mask.original, job.mask.region)
        image = inpaint.paste_back(image, job.mask.original, job.mask.feather)
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
        "prompt_cache": "miss" if cached is None else "hit",
        "prompt_cache_bytes": _PROMPTS.bytes,
        "mask_coverage": None if job.mask is None else round(job.mask.coverage, 4),
        "mask_outside_drift": None if drift is None else round(drift, 2),
        "mask_blend_steps": job.blend_steps,
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
    except inpaint.MaskRefused as refused:
        send("result", refused={"code": refused.code, "message": str(refused)})
        send("done")
        return
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
