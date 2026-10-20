"""LTX-2.5 (distilled) on the Mac through ltx-2-mlx, one stage in unified memory at a time.

dgrauet/ltx-2-mlx (MIT; the commit is pinned in crucible/envs/video/ltx-2-mlx-mlx-darwin.txt)
is a pure-MLX port of Lightricks' LTX-2 pipelines. Its `DistilledPipeline` mirrors upstream's:
the distilled checkpoint's 8 steps at half the size (on a 2.5 pack the ancestral, SDE sampler),
the latent upscaled 2x by the pack's spatial upscaler, then 3 deterministic steps at full size.
Its own `generate_and_save` runs all of that and writes an mp4 through the ffmpeg on PATH; this
worker drives the same methods one stage at a time instead, so each stage has its own MLX peak,
a batch that keeps it resident can skip the text stage for a prompt it has read, and the sound
and the frames come back to videocore's mux like the PC's:

- encoding: the pack's Gemma 4 text encoder (int8) and its connector; freed after.
- audio_encoding (a supplied soundtrack only): the audio VAE's encoder turns the soundtrack's
  first clip-length into audio tokens; freed after. Both passes then hold those tokens fixed
  (a frozen audio state, denoise mask 0) and denoise only the picture: the port's
  a2vid_two_stage applied inside the distilled pipeline (see `stepping_pipeline`).
- denoising: the distilled transformer (int8, all of it in memory unless the machine's
  [video_desktop] table asks for block streaming), the VAE encoder and the upscaler load; the
  start picture of an image-to-video job is encoded here; 8 steps at half the size.
- refining: the 2x latent upscale, then 3 steps at full size; the transformer is freed after.
- decoding: the conv VAE decoder, tiled to DECODE_BUDGET_BYTES, into 8-bit RGB frames.
- audio_decoding: the audio VAE and the vocoder with bandwidth extension, 48 kHz stereo. With
  a supplied soundtrack there is none: the clip's sound is the soundtrack itself, decoded
  through PyAV to 48 kHz stereo and cut to the clip.

With a [video_desktop] table the job also sends a `desktop` dict: block streaming, and a
temporal (and optional spatial) tiling of each pass through the port's modality tiling, so
no single kernel spans the whole clip. The MLX and ltx-2-mlx env knobs of the same table
arrive in the environment. docs/internals/video.md, "The Mac arm", has the memory arithmetic
behind each stage, and "Keeping the desktop responsive: [video_desktop]" the knobs.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import workerio

sys.path.pop(0)
workerio.claim_stdout()

import gc
import io
from collections import OrderedDict

videocore = workerio.load_sibling("videocore", __file__)

LABEL = "ltx-2-mlx"

PROMPT_CACHE_ENTRIES = 16

PROMPT_CACHE_BYTES = 256 * 1024 * 1024

DECODE_BUDGET_BYTES = 12 * 1024**3

IMAGE_CRF = 18

AUDIO_SAMPLE_RATE = 48000

# The audio VAE's encoder reads 16 kHz stereo (ltx_core_mlx/model/audio_vae/processor.py) and
# makes 25 latents a second (ltx_core_mlx/utils/positions.py, AUDIO_LATENTS_PER_SECOND).
ENCODER_SAMPLE_RATE = 16000

AUDIO_LATENTS_PER_SECOND = 25

# A clip's audio tokens are round(seconds x 25), which can reach half a latent past its last
# frame, and the encoder's framing (a centred STFT, then 4x downsampling) can make one latent
# fewer than the samples' length suggests. So the encoder is given this many latents' worth
# beyond the tokens it must make: more of the soundtrack where it goes on, silence where it
# ends. No frame of the clip is paired with that tail; the tokens past `count` are dropped.
ENCODER_MARGIN_LATENTS = 2

DEFAULT_REFINE_STEPS = 3

DESKTOP_ENVIRONMENT: tuple = ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER", "LTX2_DIT_EVAL_EVERY")

REQUIRED_FILES: tuple = (
    "embedded_config.json",
    "quantize_config.json",
    "text_encoder_config.json",
    "text_encoder.safetensors",
    "connector.safetensors",
    "tokenizer.json",
    "transformer-distilled.safetensors",
    "spatial_upscaler_x2_v1_0.safetensors",
    "vae_encoder_conv.safetensors",
    "vae_decoder_conv.safetensors",
    "audio_vae.safetensors",
    "vocoder.safetensors",
)


def require(request: dict, key: str, kind):
    return workerio.require(request, key, kind, LABEL, videocore.WHY_REQUIRED)


def _version_of(distribution: str):
    from importlib import metadata

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def missing_files(model_dir: str) -> list:
    return [name for name in REQUIRED_FILES if not os.path.isfile(os.path.join(model_dir, name))]


def even(value: int) -> int:
    return value + (value & 1)


def prepare_start_image(image_path: str, out_path: str, crf: int = IMAGE_CRF) -> None:
    """The start picture recompressed as one H.264 frame at `crf`, written as a PNG.

    LTX-2 was trained on frames that carry H.264 artefacts, and upstream feeds a start picture
    through libx264 before encoding it. ltx-2-mlx does that through the ffmpeg on PATH with
    libx264, which Crucible's LGPL Mac ffmpeg does not have; PyAV's wheel does. CRF 18 is the
    LTX-2.5 value (diffusers' LTX2_5_IMAGE_CRF), which the PC's arm uses too; the library's
    own default, 33, is the LTX-2.3 one. The pipeline then crops and resizes the PNG with the
    recompression turned off (crf 0), so it happens once, on the picture as sent.
    """
    import av
    import numpy
    from PIL import Image

    with Image.open(image_path) as opened:
        rgb = numpy.asarray(opened.convert("RGB"), dtype=numpy.uint8)
    height, width = rgb.shape[:2]
    padded = numpy.zeros((even(height), even(width), 3), dtype=numpy.uint8)
    padded[:height, :width] = rgb
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=1)
        stream.width, stream.height = padded.shape[1], padded.shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "veryfast"}
        frame = av.VideoFrame.from_ndarray(padded, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    buffer.seek(0)
    with av.open(buffer) as container:
        decoded = next(container.decode(video=0)).to_ndarray(format="rgb24")
    Image.fromarray(decoded[:height, :width]).save(out_path, format="PNG")


def _tile_fits(size: int, tiles: int, overlap: int) -> bool:
    """Whether ltx-2-mlx's split_by_count accepts `tiles` tiles over `size` latent cells.

    It raises when there are more tiles than cells or when a tile is no longer than the
    overlap (ltx_core_mlx/model/video_vae/tiling.py, split_by_count), so a count that
    would not fit is lowered instead of failing the job.
    """
    return tiles <= 1 or (tiles <= size and (size + overlap * (tiles - 1)) // tiles > overlap)


def _tile_size(size: int, tiles: int, overlap: int) -> int:
    if tiles <= 1:
        return size
    return -(-(size + overlap * (tiles - 1)) // tiles)


def tile_plan(latent: tuple, desktop) -> dict:
    """How one denoising pass over a (frames, height, width) latent grid is tiled.

    Temporal tiles are chosen so each tile's video tokens stay near max_tile_tokens (the
    size of one attention or feed-forward kernel is what holds the GPU); spatial tiles are
    the table's fixed count per axis. None of it applies without a [video_desktop] table.
    """
    frames, height, width = latent
    if desktop is None:
        return {"tiles": [1, 1, 1], "tile_tokens": frames * height * width}
    overlap = int(desktop["tile_overlap"])
    rows = cols = int(desktop["tile_spatial"])
    while rows > 1 and not _tile_fits(height, rows, overlap):
        rows -= 1
    while cols > 1 and not _tile_fits(width, cols, overlap):
        cols -= 1
    tile_rows, tile_cols = _tile_size(height, rows, overlap), _tile_size(width, cols, overlap)
    spans = 1
    ceiling = int(desktop["max_tile_tokens"])
    if ceiling > 0 and frames * tile_rows * tile_cols > ceiling:
        spans = -(-(frames * tile_rows * tile_cols) // ceiling)
        while spans > 1 and not _tile_fits(frames, spans, overlap):
            spans -= 1
    tokens = _tile_size(frames, spans, overlap) * tile_rows * tile_cols
    return {"tiles": [spans, rows, cols], "tile_tokens": tokens}


def tile_config(plan: dict, overlap: int):
    """The TileCountConfig ltx-2-mlx's pipelines take as `tile_count` (None: untiled)."""
    if plan["tiles"] == [1, 1, 1]:
        return None
    from ltx_core_mlx.model.video_vae.tiling import DimensionTilingConfig, TileCountConfig

    frames, rows, cols = plan["tiles"]
    return TileCountConfig(
        frames=DimensionTilingConfig(num_tiles=frames, overlap=overlap if frames > 1 else 0),
        height=DimensionTilingConfig(num_tiles=rows, overlap=overlap if rows > 1 else 0),
        width=DimensionTilingConfig(num_tiles=cols, overlap=overlap if cols > 1 else 0),
    )


class DutyCycle:
    """An mx.eval that, after the GPU finishes, leaves it idle for a share of that time.

    ltx-2-mlx syncs through the module-level `_mx_eval` of
    ltx_core_mlx/model/transformer/model.py after every Nth transformer block
    (LTX2_DIT_EVAL_EVERY) and after every block when streaming. The worker replaces that name
    with this, so each sync is followed by a pause of busy x (100 - duty) / duty: at 85 the
    GPU is left idle about 15% of the time in short, regular gaps, which is what lets the
    desktop's own GPU work in and keeps the GPU off 100%. `busy` is the time the eval waited,
    which is the GPU's time for those blocks less whatever it had already finished.
    """

    def __init__(self, evaluate, duty_pct: int, clock=None, sleep=None) -> None:
        import time

        self._evaluate = evaluate
        self._duty = duty_pct
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self.busy_s = 0.0
        self.paused_s = 0.0

    def reset(self) -> None:
        self.busy_s = 0.0
        self.paused_s = 0.0

    def __call__(self, *arrays) -> None:
        started = self._clock()
        self._evaluate(*arrays)
        busy = self._clock() - started
        self.busy_s += busy
        if self._duty < 100:
            pause = busy * (100 - self._duty) / self._duty
            self._sleep(pause)
            self.paused_s += pause


def freeze_soundtrack(state, tokens, zeros) -> None:
    """Make `state`, a pass's audio LatentState, the supplied soundtrack held fixed.

    The state a2vid_two_stage builds (ltx_pipelines_mlx/a2vid_two_stage.py, audio_state_1 and
    audio_state_2): the soundtrack's tokens as both the latent and the clean latent, a denoise
    mask of 0 (every token preserved: apply_denoise_mask hands back the clean latent after each
    step, and a step from a latent to itself stays there) and `frozen`, which conditions the
    model on audio sigma 0 (utils/samplers.py, _frozen_sigma_kwargs). The positions the pipeline
    built stay: compute_audio_positions of the same token count. LatentState is a plain mutable
    dataclass and the loop reads these fields after the flush.
    """
    if tuple(state.latent.shape) != tuple(tokens.shape):
        raise RuntimeError(
            f"the pipeline built an audio state of {tuple(state.latent.shape)} for this pass, "
            f"but the soundtrack was encoded to {tuple(tokens.shape)}; both are counted from "
            "the same frames and frame rate and must agree"
        )
    state.latent = tokens
    state.clean_latent = tokens
    state.denoise_mask = zeros((1, tokens.shape[1], 1))
    state.frozen = True


def stepping_pipeline(base):
    """`base` (DistilledPipeline) with the prompt's embeddings, and any soundtrack, handed in.

    Its stage 1 encodes the prompt itself; here the encoding stage has already run (or the
    prompt cache answered), so loading the text encoder is a no-op and encoding returns what
    that stage produced. Built at call time so this file imports without mlx.

    With `soundtrack` set (its encoded tokens) both passes run audio-to-video. _stage1 and
    _stage2 (ltx_pipelines_mlx/distilled.py) each build a noised audio state and hand that same
    object to `self._pre_denoise_flush` and then to `self._run_denoise_loop`; the flush is where
    it becomes the frozen soundtrack, in both passes, so stage 2 holds the original tokens, not
    stage 1's output. And stage 1 runs the deterministic Euler loop instead of the 2.5 pack's
    ancestral one: custom-audio lip sync on LTX-2.5 distilled was found to work with plain
    Euler in both passes, unguided (huggingface.co/Lightricks/LTX-2.5/discussions/44, 2026-08).
    """

    class Stepping(base):
        embeds = None
        soundtrack = None

        def _load_text_encoder(self) -> None:
            return None

        def _encode_text(self, prompt: str):
            if self.embeds is None:
                raise RuntimeError(
                    "the pipeline asked for the prompt's embeddings before the encoding "
                    "stage produced them"
                )
            return self.embeds

        def _pre_denoise_flush(self, video_state, audio_state) -> None:
            if self.soundtrack is not None:
                import mlx.core as mx

                freeze_soundtrack(
                    audio_state, self.soundtrack, lambda shape: mx.zeros(shape, dtype=mx.bfloat16)
                )
            super()._pre_denoise_flush(video_state, audio_state)

        def _run_denoise_loop(self, **arguments):
            if self.soundtrack is not None:
                arguments["ancestral"] = False
            return super()._run_denoise_loop(**arguments)

    return Stepping


def decode_soundtrack(path: str, sample_rate: int, fmt: str, samples: int, level_mono: bool):
    """The soundtrack's first `samples` stereo samples per channel at `sample_rate` in `fmt`
    (a PyAV sample format), as a numpy array of (2, samples), through PyAV (the video env has
    it; the port's own load_audio shells out to an ffmpeg on PATH and answers None for any
    failure). Fewer come back when the soundtrack ends first; the caller says what it needed.

    A mono soundtrack becomes stereo by ffmpeg's pan law (each side 3 dB down), which is what
    the port's `-ac 2` hands the audio encoder; with `level_mono` it is instead copied to both
    sides at its own level, so the clip sounds as loud as the file it was made from.
    """
    import av
    import numpy

    kept: list = []
    have = 0
    with av.open(path) as container:
        if not container.streams.audio:
            raise RuntimeError(f"{os.path.basename(path)} has no audio stream")
        mono = level_mono and container.streams.audio[0].codec_context.channels == 1
        width = 1 if mono else 2
        resampler = av.AudioResampler(
            format=fmt, layout="mono" if mono else "stereo", rate=sample_rate
        )

        def take(frames) -> None:
            nonlocal have
            for frame in frames:
                block = frame.to_ndarray()
                if not frame.format.is_planar:
                    block = block.reshape(-1, width).T
                kept.append(block)
                have += block.shape[1]

        for frame in container.decode(container.streams.audio[0]):
            take(resampler.resample(frame))
            if have >= samples:
                break
        else:
            take(resampler.resample(None))
    if not kept:
        raise RuntimeError(f"{os.path.basename(path)} decoded to no sound at all")
    return numpy.repeat(numpy.concatenate(kept, axis=1)[:, :samples], 3 - width, axis=0)


class StepReport:
    """Stands in for ltx-2-mlx's stepwise previews: the only per-step hook its loops offer.

    `bind` is what the pipeline calls for each denoising loop; the callback it returns runs
    after every step's prediction, which it evaluates (the next step needs it anyway) so the
    step reported is the step done, and a cancel lands between steps.
    """

    def __init__(self, mx, progress, evaluate=None) -> None:
        self._mx = mx
        self._progress = progress
        self._evaluate = evaluate or mx.eval

    def bind(self, **_geometry):
        def on_step(step_index, _steps, video_x0, _sigma) -> None:
            self._evaluate(video_x0)
            self._progress.reached(step_index + 1)

        return on_step


class LtxMlxEngine:
    name = "ltx-2-mlx"
    spans = videocore.TWO_PASS_SPANS

    def __init__(self, request: dict) -> None:
        import mlx.core as mx
        from ltx_pipelines_mlx.distilled import DistilledPipeline

        self._mx = mx
        self._model_dir = require(request, "model_dir", str)
        self.device = require(request, "device", str)
        mx.set_cache_limit(require(request, "mlx_cache_limit_bytes", int))
        absent = missing_files(self._model_dir)
        if absent:
            raise RuntimeError(
                f"the LTX-2.5 MLX pack at {self._model_dir} is incomplete: {absent} are "
                "missing. Run `crucible models pull ltx-2.5-distilled`; it fetches only "
                "what is missing"
            )
        desktop = request.get("desktop")
        self._desktop = dict(desktop) if isinstance(desktop, dict) else None
        low_ram = bool(self._desktop and self._desktop.get("low_ram"))
        # low_ram_streaming: the transformer's 48 blocks are read one at a time from the
        # mmap'd file with an mx.eval after each; the pipeline also sets MLX's cache limit
        # to 0 before anything loads (ltx_pipelines_mlx/_base.py, BasePipeline.__init__).
        self._pipe = stepping_pipeline(DistilledPipeline)(
            self._model_dir, low_memory=True, low_ram_streaming=low_ram
        )
        self._cache_limit = 0 if low_ram else require(request, "mlx_cache_limit_bytes", int)
        self._duty = None
        duty = int(self._desktop.get("gpu_duty_pct", 100)) if self._desktop else 100
        if duty < 100:
            from ltx_core_mlx.model.transformer import model as dit

            self._duty = DutyCycle(dit._mx_eval, duty)
            dit._mx_eval = self._duty
        if not self._pipe._is_25:
            raise RuntimeError(
                f"{self._model_dir} is not an LTX-2.5 pack (its embedded_config.json does "
                "not set ff_bias false); the manifest pins dgrauet/ltx-2.5-mlx-q8"
            )
        self._pipe.verbose = True
        self._prompts: OrderedDict = OrderedDict()
        self._prompt_bytes = 0
        self.versions = {
            "ltx-pipelines-mlx": _version_of("ltx-pipelines-mlx"),
            "ltx-core-mlx": _version_of("ltx-core-mlx"),
            "mlx": mx.__version__,
            "mlx-arsenal": _version_of("mlx-arsenal"),
            "transformers": _version_of("transformers"),
            "av": _version_of("av"),
        }

    def _release(self) -> None:
        gc.collect()
        self._mx.clear_cache()
        self._mx.reset_peak_memory()

    def _close(self, stage: str) -> int:
        mx = self._mx
        peak = int(mx.get_peak_memory())
        gib = 1024**3
        print(
            f"crucible memory {stage}: MLX peak {peak / gib:.2f} GiB, active "
            f"{mx.get_active_memory() / gib:.2f} GiB, cache {mx.get_cache_memory() / gib:.2f} GiB",
            file=sys.stderr,
            flush=True,
        )
        self._release()
        return peak

    def _unload(self) -> None:
        pipe = self._pipe
        pipe.embeds = None
        pipe.soundtrack = None
        pipe.stepwise = None
        pipe.dit = None
        pipe.upsampler = None
        pipe._loaded = False
        pipe._tile_count = None
        pipe.prompt_encoder.free()
        pipe.image_conditioner.free()
        pipe.audio_conditioner.free()
        pipe.video_decoder_block.free()
        pipe.audio_decoder_block.free()
        gc.collect()
        self._mx.clear_cache()

    def _encode(self, job) -> tuple:
        encoder = self._pipe.prompt_encoder
        try:
            video, audio = encoder.encode(job.prompt)
            self._mx.eval(video, audio)
            return video, audio
        finally:
            encoder.free()

    def _remember(self, job, embeds: tuple) -> None:
        size = sum(int(array.nbytes) for array in embeds)
        if size > PROMPT_CACHE_BYTES:
            return
        self._prompts[job.prompt_key] = (embeds, size)
        self._prompt_bytes += size
        while len(self._prompts) > PROMPT_CACHE_ENTRIES or self._prompt_bytes > PROMPT_CACHE_BYTES:
            self._prompt_bytes -= self._prompts.popitem(last=False)[1][1]

    def _recall(self, job):
        found = self._prompts.get(job.prompt_key)
        if found is None:
            return None
        self._prompts.move_to_end(job.prompt_key)
        return found[0]

    def _encode_soundtrack(self, job):
        """The soundtrack's audio tokens for this clip: (1, round(seconds x 25), 128)."""
        import numpy
        from ltx_core_mlx.model.audio_vae import encode_audio
        from ltx_core_mlx.utils.positions import compute_audio_token_count

        mx = self._mx
        pipe = self._pipe
        count = compute_audio_token_count(job.num_frames, frame_rate=float(job.fps))
        wanted = (count + ENCODER_MARGIN_LATENTS) * ENCODER_SAMPLE_RATE // AUDIO_LATENTS_PER_SECOND
        wave = decode_soundtrack(
            job.soundtrack_path, ENCODER_SAMPLE_RATE, "fltp", wanted, level_mono=False
        )
        clip = job.num_frames * ENCODER_SAMPLE_RATE // job.fps
        if wave.shape[1] < clip:
            raise RuntimeError(
                f"the soundtrack decoded to {wave.shape[1] / ENCODER_SAMPLE_RATE:.3f} s, short of "
                f"the {job.duration_s:.3f} s clip the server measured it long enough for: its "
                "container states a longer length than its sound"
            )
        wave = numpy.pad(wave.astype(numpy.float32), ((0, 0), (0, wanted - wave.shape[1])))
        encoder, processor = pipe.audio_conditioner.load()
        try:
            latent = encode_audio(mx.array(wave)[None], ENCODER_SAMPLE_RATE, encoder, processor)
            if latent.shape[2] < count:
                raise RuntimeError(
                    f"the audio VAE made {latent.shape[2]} latents of {wanted} samples, fewer "
                    f"than the {count} audio tokens of a {job.num_frames}-frame clip at "
                    f"{job.fps} fps"
                )
            tokens, _ = pipe.audio_patchifier.patchify(latent[:, :, :count, :])
            mx.eval(tokens)
            return tokens
        finally:
            pipe.audio_conditioner.free()

    def _soundtrack_pcm(self, job) -> bytes:
        """The soundtrack itself for the mux, 48 kHz stereo 16-bit, cut to the clip."""
        samples = job.num_frames * AUDIO_SAMPLE_RATE // job.fps
        pcm = decode_soundtrack(
            job.soundtrack_path, AUDIO_SAMPLE_RATE, "s16", samples, level_mono=True
        )
        if pcm.shape[1] < samples:
            raise RuntimeError(
                f"the soundtrack decoded to {pcm.shape[1] / AUDIO_SAMPLE_RATE:.3f} s at "
                f"{AUDIO_SAMPLE_RATE} Hz, short of the {job.duration_s:.3f} s clip"
            )
        return pcm.T.astype("<i2").tobytes()

    def _start_images(self, job):
        if job.image_path is None:
            return None, None
        from ltx_pipelines_mlx.utils.args import ImageConditioningInput

        prepared = job.output_path + ".start.png"
        prepare_start_image(job.image_path, prepared)
        return [ImageConditioningInput(path=prepared, frame_idx=0, strength=1.0, crf=0)], prepared

    def _half_size(self, job, images):
        stage1, frames, height, width = self._pipe._stage1(
            job.prompt,
            job.height,
            job.width,
            job.num_frames,
            frame_rate=float(job.fps),
            seed=job.seed,
            stage1_steps=job.steps,
            image=None,
            images=images,
            prompt_relay=None,
            generated_keyframes=0,
            enable_teacache=False,
        )
        if (frames, height, width) != (job.num_frames, job.height, job.width):
            raise RuntimeError(
                f"ltx-2-mlx settled on {frames} frames at {width}x{height} for a "
                f"{job.num_frames}-frame {job.width}x{job.height} request; the job's own "
                "checks should have refused a size its grid rounds"
            )
        return stage1

    def _full_size(self, job, stage1, refine_steps: int) -> tuple:
        pipe = self._pipe
        half = pipe.video_patchifier.unpatchify(stage1.video_tokens, stage1.latent_dims)
        upscaled = pipe._upsample_latent(half)
        del half
        video, audio = pipe._stage2(
            stage1,
            upscaled,
            num_frames=job.num_frames,
            frame_rate=float(job.fps),
            seed=job.seed,
            stage2_steps=refine_steps,
            extra_conditionings=[],
        )
        self._mx.eval(video, audio)
        return video, audio

    def _decode(self, job, latent) -> list:
        import numpy
        from ltx_core_mlx.model.video_vae.video_vae import (
            _compute_decode_tiling,
            decode_cache_limit,
        )

        mx = self._mx
        block = self._pipe.video_decoder_block
        decoder = block.load()
        tiling = _compute_decode_tiling(
            tuple(latent.shape), frame_rate=float(job.fps), budget_bytes=DECODE_BUDGET_BYTES
        )
        frames: list = []
        try:
            with decode_cache_limit():
                for chunk in decoder.tiled_decode(latent, tiling):
                    pixels = mx.round((mx.clip(chunk[0], -1.0, 1.0) + 1.0) * 127.5).astype(mx.uint8)
                    pixels = mx.contiguous(pixels.transpose(1, 2, 3, 0))
                    mx.eval(pixels)
                    host = numpy.array(pixels)
                    frames.extend(host[index].tobytes() for index in range(host.shape[0]))
                    del chunk, pixels, host
                    gc.collect()
                    mx.clear_cache()
        finally:
            block.free()
        size = job.width * job.height * 3
        if len(frames) != job.num_frames or any(len(frame) != size for frame in frames):
            raise RuntimeError(
                f"the VAE decoded {len(frames)} frames of "
                f"{len(frames[0]) if frames else 0} bytes for a {job.num_frames}-frame "
                f"{job.width}x{job.height} request"
            )
        return frames

    def _decode_audio(self, latent) -> tuple:
        import numpy

        mx = self._mx
        block = self._pipe.audio_decoder_block
        try:
            wave = block(latent)[0]
            if wave.ndim == 1:
                wave = wave[None]
            channels = int(wave.shape[0])
            pcm = mx.round(mx.clip(wave.astype(mx.float32), -1.0, 1.0) * 32767.0).astype(mx.int16)
            pcm = mx.contiguous(pcm.T)
            mx.eval(pcm)
            return numpy.array(pcm).tobytes(), channels
        finally:
            block.free()

    def generate(self, job, progress) -> tuple:
        pipe = self._pipe
        peaks: dict = {}
        refine_steps = job.refine_steps if job.refine_steps is not None else DEFAULT_REFINE_STEPS
        prepared = None
        tokens = None
        latent_frames = (job.num_frames - 1) // 8 + 1
        plans = [
            tile_plan((latent_frames, job.height // 64, job.width // 64), self._desktop),
            tile_plan((latent_frames, job.height // 32, job.width // 32), self._desktop),
        ]
        if self._duty is not None:
            self._duty.reset()
        self._release()
        embeds = self._recall(job)
        cache = "miss" if embeds is None else "hit"
        try:
            if embeds is None:
                progress.enter("encoding")
                embeds = self._encode(job)
                peaks["encoding"] = self._close("encoding")
                self._remember(job, embeds)
            pipe.embeds = embeds
            if job.soundtrack_path is not None:
                progress.enter("audio_encoding")
                pipe.soundtrack = self._encode_soundtrack(job)
                tokens = int(pipe.soundtrack.shape[1])
                peaks["audio_encoding"] = self._close("audio_encoding")
            pipe.stepwise = StepReport(self._mx, progress, self._duty)
            images, prepared = self._start_images(job)
            progress.enter("denoising", job.steps)
            self._tile(plans[0])
            stage1 = self._half_size(job, images)
            peaks["denoising"] = self._close("denoising")
            progress.enter("refining", refine_steps)
            self._tile(plans[1])
            video, audio = self._full_size(job, stage1, refine_steps)
            del stage1
            pipe.dit = None
            pipe._loaded = False
            peaks["refining"] = self._close("refining")
            progress.enter("decoding")
            frames = self._decode(job, video)
            del video
            peaks["decoding"] = self._close("decoding")
            pcm, channels, rate = None, 0, 0
            if job.soundtrack_path is not None:
                pcm, channels, rate = self._soundtrack_pcm(job), 2, AUDIO_SAMPLE_RATE
            elif job.audio:
                progress.enter("audio_decoding")
                pcm, channels = self._decode_audio(audio)
                rate = AUDIO_SAMPLE_RATE
                peaks["audio_decoding"] = self._close("audio_decoding")
            del audio
        finally:
            self._unload()
            if prepared is not None and os.path.exists(prepared):
                os.remove(prepared)
        clip = videocore.Clip(
            job.width, job.height, job.fps, iter(frames), len(frames),
            pcm=pcm, channels=channels, sample_rate=rate,
        )
        frozen = job.soundtrack_path is not None
        first = "euler" if frozen else "euler ancestral"
        held = ", the soundtrack's audio frozen" if frozen else ""
        return clip, peaks, {
            "prompt_cache": cache,
            "soundtrack": {"audio_tokens": tokens} if frozen else None,
            "sampling": {
                "passes": [
                    {"size": "half", "steps": job.steps,
                     "sampler": f"{first}, the LTX-2.5 distilled sigma schedule, unguided{held}"},
                    {"size": "full", "steps": refine_steps,
                     "sampler": f"euler, the LTX-2.5 stage-2 distilled sigmas, unguided{held}",
                     "upscaler": "spatial_upscaler_x2_v1_0"},
                ],
            },
            "quantization": {
                "text_encoder": "MLX int8 (group size 64), bfloat16 activations",
                "transformer": "MLX int8 (group size 64), bfloat16 activations",
            },
            "desktop": self._desktop_report(plans),
        }

    def _tile(self, plan: dict) -> None:
        # _stage1 and _stage2 each read `_tile_count` when they build their model and wrap
        # the transformer in TiledLTXModel when it is set (ltx_pipelines_mlx/distilled.py).
        overlap = int(self._desktop["tile_overlap"]) if self._desktop else 0
        self._pipe._tile_count = tile_config(plan, overlap)

    def _desktop_report(self, plans: list) -> dict | None:
        if self._desktop is None:
            return None
        return {
            **self._desktop,
            "environment": {name: os.environ.get(name) for name in DESKTOP_ENVIRONMENT},
            "mlx_cache_limit_bytes": self._cache_limit,
            "synced_s": None if self._duty is None else round(self._duty.busy_s, 2),
            "paused_s": None if self._duty is None else round(self._duty.paused_s, 2),
            "passes": [{"size": size, **plan} for size, plan in zip(("half", "full"), plans)],
        }


if __name__ == "__main__":
    sys.exit(videocore.Worker("video", {LtxMlxEngine.name: LtxMlxEngine}).serve())
