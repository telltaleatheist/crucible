"""LTX-2.5 (distilled) through diffusers, one component on the card at a time.

The image worker's shape (crucible/jobs/image/worker.py, DiffusersEngine):
each stage loads what it needs onto the card, runs, deletes it and empties
the cache, and its peak is recorded before the next begins.

- encoding: the Gemma 4 12B text encoder from the Lightricks shards, quantized
  to int8 weight-only by torchao tensor by tensor as transformers loads it (a
  bf16 copy never exists, on the card or in host memory).
- connecting: the text connectors in bfloat16 (6.3 GB). Their output is what
  the transformer reads, and it is small, so it is what the prompt cache keeps.
- conditioning (image-to-video only): the VAE encodes the start frame.
- denoising: the distilled transformer from the GGUF Q6_K file, streamed from
  a read-only memmap straight onto the card (diffusers' own GGUF reader copies
  the whole file into host memory first, which the PC's 13 GB WSL cap cannot
  hold), dequantized one layer at a time; the stock LTX2 pipeline runs its
  loop with the connectors stage's output handed in by a stand-in.
- decoding: the VAE, spatially and temporally tiled.
- audio_decoding: the audio VAE and the vocoder (48 kHz stereo).

docs/internals/video.md has the memory arithmetic behind each stage.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import workerio

sys.path.pop(0)
workerio.claim_stdout()

import gc
import warnings
from collections import OrderedDict

videocore = workerio.load_sibling("videocore", __file__)
ltxkeys = workerio.load_sibling("ltxkeys", __file__)

LABEL = "ltx"

PROMPT_CACHE_ENTRIES = 16

PROMPT_CACHE_BYTES = 256 * 1024 * 1024

FRAMES_PER_COPY = 8

NON_QUANTIZED = ("F32", "F16", "BF16")


def require(request: dict, key: str, kind):
    return workerio.require(request, key, kind, LABEL, videocore.WHY_REQUIRED)


def optional(request: dict, key: str, kind):
    if request.get(key) is None:
        return None
    return require(request, key, kind)


def _version_of(distribution: str):
    from importlib import metadata

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def gguf_state_dict(path: str) -> dict:
    """The transformer's tensors under diffusers names, backed by the file.

    Each tensor is a view of GGUFReader's read-only memmap: nothing is copied
    here, and diffusers' loader moves them to the card one at a time, so what
    host memory holds is page cache the kernel can drop.
    """
    import gguf
    import torch
    from diffusers.quantizers.gguf.utils import GGUFParameter

    reader = gguf.GGUFReader(path)
    state: dict = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        for tensor in reader.tensors:
            name = ltxkeys.diffusers_name(tensor.name)
            if name is None:
                continue
            data = torch.from_numpy(tensor.data)
            if tensor.tensor_type.name in NON_QUANTIZED:
                state[name] = data
            else:
                state[name] = GGUFParameter(data, quant_type=tensor.tensor_type)
    return state


class _Connected:
    """Stands in for the connectors inside the stock pipeline's loop.

    The connecting stage already ran (and freed) the real connectors; this
    hands the pipeline their output on the card, as the real module would.
    It is built as a torch module at call time so this file imports without
    torch in the tests.
    """

    @staticmethod
    def of(outputs: tuple, device: str):
        import torch

        class Connected(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._outputs = outputs
                self._device = torch.device(device)

            @property
            def device(self):
                return self._device

            @property
            def dtype(self):
                return self._outputs[0].dtype

            def forward(self, *args, **kwargs):
                return self._outputs

        return Connected()


class LtxEngine:
    name = "ltx"

    def __init__(self, request: dict) -> None:
        import diffusers
        import torch
        from diffusers import LTX2ImageToVideoPipeline, LTX2Pipeline
        from diffusers.models.autoencoders import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
        from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

        self._torch = torch
        self._model_dir = require(request, "model_dir", str)
        self._transformer_path = require(request, "transformer_path", str)
        self._dtype = getattr(torch, require(request, "dtype", str))
        self.device = require(request, "device", str)
        workerio.cap_memory(torch, optional(request, "memory_cap_bytes", int))
        self._require_files()
        self._vae = AutoencoderKLLTX2Video.from_pretrained(
            self._model_dir, subfolder="vae", torch_dtype=self._dtype
        )
        self._vae.enable_tiling()
        self._vae.use_framewise_decoding = True
        self._audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
            self._model_dir, subfolder="audio_vae", torch_dtype=self._dtype
        )
        self._vocoder = LTX2VocoderWithBWE.from_pretrained(
            self._model_dir, subfolder="vocoder", torch_dtype=self._dtype
        )
        self._t2v = LTX2Pipeline.from_pretrained(
            self._model_dir,
            vae=self._vae,
            audio_vae=self._audio_vae,
            vocoder=self._vocoder,
            text_encoder=None,
            connectors=None,
            transformer=None,
            processor=None,
            prompt_enhancer=None,
            duration_head=None,
            torch_dtype=self._dtype,
        )
        self._i2v = LTX2ImageToVideoPipeline(**self._t2v.components)
        self._prompts: OrderedDict = OrderedDict()
        self._prompt_bytes = 0
        self.versions = {
            "diffusers": diffusers.__version__,
            "torch": torch.__version__,
            "transformers": _version_of("transformers"),
            "torchao": _version_of("torchao"),
            "gguf": _version_of("gguf"),
        }

    def _require_files(self) -> None:
        wanted = [
            self._transformer_path,
            os.path.join(self._model_dir, "text_encoder", "model.safetensors.index.json"),
            os.path.join(self._model_dir, "connectors", "config.json"),
            os.path.join(self._model_dir, "transformer", "config.json"),
        ]
        absent = [path for path in wanted if not os.path.isfile(path)]
        if absent:
            raise RuntimeError(
                f"the LTX-2.5 weights are incomplete: {absent} are missing. "
                "Run `crucible models pull ltx-2.5-distilled`; it fetches only what is missing"
            )

    def peak_bytes(self) -> int:
        return int(self._torch.cuda.max_memory_reserved(0))

    def _release(self) -> None:
        gc.collect()
        self._torch.cuda.empty_cache()
        self._torch.cuda.reset_peak_memory_stats(0)

    def _close(self, stage: str) -> int:
        workerio.memory_line(self._torch, stage)
        peak = self.peak_bytes()
        self._release()
        return peak

    def _encode(self, job) -> tuple:
        from torchao.quantization import Int8WeightOnlyConfig
        from transformers import AutoModelForImageTextToText, TorchAoConfig

        torch = self._torch
        encoder = AutoModelForImageTextToText.from_pretrained(
            os.path.join(self._model_dir, "text_encoder"),
            dtype=self._dtype,
            device_map=self.device,
            quantization_config=TorchAoConfig(Int8WeightOnlyConfig()),
        )
        pipe = self._t2v
        pipe.text_encoder = encoder
        try:
            with torch.no_grad():
                embeds, mask, _, _ = pipe.encode_prompt(
                    prompt=job.prompt,
                    do_classifier_free_guidance=False,
                    device=self.device,
                    dtype=self._dtype,
                )
            return embeds, mask
        finally:
            pipe.text_encoder = None
            del encoder

    def _connect(self, embeds, mask) -> tuple:
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

        connectors = LTX2TextConnectors.from_pretrained(
            self._model_dir, subfolder="connectors", torch_dtype=self._dtype, device_map=self.device
        )
        try:
            with self._torch.no_grad():
                video, audio, attention = connectors(embeds, mask, padding_side="left")
            return tuple(tensor.to("cpu") for tensor in (video, audio, attention))
        finally:
            del connectors

    def _remember(self, job, connected: tuple) -> None:
        size = sum(tensor.numel() * tensor.element_size() for tensor in connected)
        if size > PROMPT_CACHE_BYTES:
            return
        self._prompts[job.prompt_key] = (connected, size)
        self._prompt_bytes += size
        while len(self._prompts) > PROMPT_CACHE_ENTRIES or self._prompt_bytes > PROMPT_CACHE_BYTES:
            self._prompt_bytes -= self._prompts.popitem(last=False)[1][1]

    def _recall(self, job):
        found = self._prompts.get(job.prompt_key)
        if found is None:
            return None
        self._prompts.move_to_end(job.prompt_key)
        return found[0]

    def _condition(self, job):
        import numpy
        from diffusers.pipelines.ltx2.utils import LTX2_5_IMAGE_CRF, apply_image_conditioning_crf
        from PIL import Image

        torch = self._torch
        vae = self._vae.to(self.device)
        try:
            with Image.open(job.image_path) as opened:
                picture = opened.convert("RGB")
            picture = Image.fromarray(
                apply_image_conditioning_crf(numpy.array(picture), LTX2_5_IMAGE_CRF)
            )
            pixels = self._t2v.video_processor.preprocess(
                picture, height=job.height, width=job.width, resize_mode="crop"
            )
            pixels = pixels.to(device=self.device, dtype=vae.dtype).unsqueeze(2)
            with torch.no_grad():
                first = vae.encode(pixels).latent_dist.mode()
            frames = (job.num_frames - 1) // self._t2v.vae_temporal_compression_ratio + 1
            return first.repeat(1, 1, frames, 1, 1).to(dtype=torch.float32)
        finally:
            self._vae.to("cpu")

    def _load_transformer(self):
        from accelerate import init_empty_weights
        from diffusers import GGUFQuantizationConfig, LTX2VideoTransformer3DModel

        state = gguf_state_dict(self._transformer_path)
        config = LTX2VideoTransformer3DModel.load_config(self._model_dir, subfolder="transformer")
        with init_empty_weights():
            expected = set(LTX2VideoTransformer3DModel.from_config(config).state_dict())
        missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
        if missing or extra:
            raise RuntimeError(
                f"{self._transformer_path} does not match the transformer diffusers builds "
                f"from transformer/config.json: {len(missing)} parameter(s) missing "
                f"({missing[:5]}) and {len(extra)} left over ({extra[:5]}). The GGUF and "
                "the Lightricks snapshot must be the pinned pair"
            )
        return LTX2VideoTransformer3DModel.from_single_file(
            state,
            config=self._model_dir,
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=self._dtype),
            torch_dtype=self._dtype,
            device=self.device,
        )

    def _denoise(self, job, connected: tuple, start, progress) -> tuple:
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

        torch = self._torch
        pipe = self._i2v if start is not None else self._t2v
        transformer = self._load_transformer()
        on_card = tuple(tensor.to(self.device) for tensor in connected)
        pipe.transformer = transformer
        pipe.connectors = _Connected.of(on_card, self.device)
        extra = {}
        if start is not None:
            extra = {"latents": start.to(self.device), "noise_scale": 1.0}

        def on_step(pipeline, index, timestep, tensors):
            progress.reached(index + 1)
            return tensors

        placeholder = torch.zeros((1, 1, 1), dtype=self._dtype, device=self.device)
        try:
            return pipe(
                prompt_embeds=placeholder,
                prompt_attention_mask=torch.ones((1, 1), dtype=torch.long, device=self.device),
                width=job.width,
                height=job.height,
                num_frames=job.num_frames,
                frame_rate=float(job.fps),
                sigmas=list(DISTILLED_SIGMA_VALUES[: job.steps]),
                guidance_scale=1.0,
                audio_guidance_scale=1.0,
                stg_scale=0.0,
                audio_stg_scale=0.0,
                modality_scale=1.0,
                audio_modality_scale=1.0,
                generator=torch.Generator(self.device).manual_seed(job.seed),
                output_type="latent",
                return_dict=False,
                callback_on_step_end=on_step,
                **extra,
            )
        finally:
            pipe.transformer = None
            pipe.connectors = None
            del transformer, on_card

    def _decode(self, job, latents):
        torch = self._torch
        vae = self._vae.to(self.device)
        try:
            latents = latents.to(device=self.device, dtype=vae.dtype)
            timestep = None
            if vae.config.timestep_conditioning:
                timestep = torch.zeros((latents.shape[0],), device=self.device, dtype=latents.dtype)
            with torch.no_grad():
                video = vae.decode(latents, timestep, return_dict=False)[0]
            del latents
            frames = []
            for first in range(0, video.shape[2], FRAMES_PER_COPY):
                chunk = video[0, :, first : first + FRAMES_PER_COPY]
                chunk = ((chunk.float() + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
                frames.append(chunk.permute(1, 2, 3, 0).contiguous().cpu())
            del video
            clip = torch.cat(frames).numpy()
        finally:
            self._vae.to("cpu")
        if clip.shape[:3] != (job.num_frames, job.height, job.width):
            raise RuntimeError(
                f"the VAE decoded {clip.shape[:3]} (frames, height, width) for a "
                f"{job.num_frames}x{job.height}x{job.width} request"
            )
        return clip

    def _decode_audio(self, latents) -> tuple:
        torch = self._torch
        audio_vae = self._audio_vae.to(self.device)
        vocoder = self._vocoder.to(self.device)
        try:
            with torch.no_grad():
                mel = audio_vae.decode(latents.to(self.device, audio_vae.dtype), return_dict=False)[0]
                wave = vocoder(mel)[0].float().clamp(-1.0, 1.0)
            channels = int(wave.shape[0])
            pcm = (wave.t() * 32767.0).round().to(torch.int16).contiguous().cpu().numpy().tobytes()
            return pcm, channels, int(vocoder.config.output_sampling_rate)
        finally:
            self._audio_vae.to("cpu")
            self._vocoder.to("cpu")

    def generate(self, job, progress) -> tuple:
        peaks: dict = {}
        self._release()
        connected = self._recall(job)
        cache = "miss" if connected is None else "hit"
        if connected is None:
            progress.enter("encoding")
            embeds, mask = self._encode(job)
            peaks["encoding"] = self._close("encoding")
            progress.enter("connecting")
            connected = self._connect(embeds, mask)
            del embeds, mask
            peaks["connecting"] = self._close("connecting")
            self._remember(job, connected)
        start = None
        if job.image_path is not None:
            progress.enter("conditioning")
            start = self._condition(job)
            peaks["conditioning"] = self._close("conditioning")
        progress.enter("denoising", job.steps)
        video_latents, audio_latents = self._denoise(job, connected, start, progress)
        del start
        peaks["denoising"] = self._close("denoising")
        progress.enter("decoding")
        frames = self._decode(job, video_latents)
        del video_latents
        peaks["decoding"] = self._close("decoding")
        pcm, channels, rate = None, 0, 0
        if job.audio:
            progress.enter("audio_decoding")
            pcm, channels, rate = self._decode_audio(audio_latents)
            peaks["audio_decoding"] = self._close("audio_decoding")
        del audio_latents
        clip = videocore.Clip(
            job.width,
            job.height,
            job.fps,
            (frames[index].tobytes() for index in range(frames.shape[0])),
            int(frames.shape[0]),
            pcm=pcm,
            channels=channels,
            sample_rate=rate,
        )
        return clip, peaks, {
            "prompt_cache": cache,
            "quantization": {
                "text_encoder": "torchao int8 weight-only (per row), bfloat16 activations",
                "transformer": "GGUF Q6_K, dequantized per layer to bfloat16",
            },
        }


if __name__ == "__main__":
    sys.exit(videocore.Worker("video", {LtxEngine.name: LtxEngine}).serve())
