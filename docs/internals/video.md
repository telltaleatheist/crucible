# Video generation internals

The `video` job type makes a clip with synchronized sound from a prompt, or from a prompt and a
start picture, with LTX-2.5 (Lightricks, released 2026-08-21: a 22B-parameter audio-video
diffusion transformer read by a Gemma 4 12B text encoder). The caller's side (params, result,
prompting) is [../VIDEO.md](../VIDEO.md). This page is how it runs on owens-pc and on the Mac
Studio, and why each number is what it is. The PC's arm has run (2026-09-30: a 5 s 1280x704
clip with sound in 143 s; stage peaks encoding 15.3 GB, connecting 8.1 GB, denoising 19.1 GB,
decoding 3.7 GB, audio 0.5 GB) but its manifest figures are still the **declared** ones. The
Mac's arm ("The Mac arm", below) has not run at all. The sections "What the first runs on the
PC must measure" and "What the first run on the Mac must measure" list what replaces them.

## Shape

The same shape as `image` and `audio`:

- `crucible/video/<id>.toml` is the manifest, parsed by `crucible/videomodels.py`. One model
  ships: `ltx-2.5-distilled`.
- `crucible/videoweights.py` pulls the gated Lightricks snapshot through `weights.pull` (only
  the files the manifest names) and the transformer GGUF as a hash-verified companion through
  `weights.pull_files`, the way `audioweights.py` pulls YuE2's decoder. The model is installed
  only when both are. The Mac's block has no companion (its transformer is in its one repo),
  so there `weights.pull` is the whole pull.
- `crucible/jobs/video/` is a `ResidentWorker` with lease-on-load: `video`, `load-video` and
  `unload-video`, resident kind `video` (noun `video generator`, so the unload refusal is
  `video_generator_not_resident`), capability class `video`, lease act `video`.
- The workers (`ltx_worker.py` on the PC, `ltx2mlx_worker.py` on the Mac) are workerio
  sessions. Their engine-independent half (`videocore.py`: the request, stage progress, cancel
  between steps, the mux) needs only the standard library, and `tests/fake_video_worker.py`
  runs the same `Worker` with a fake engine for either arm. `tests/test_video_mac_worker.py`
  runs the real Mac worker against stand-in `mlx` and ltx-2-mlx packages.
- `[jobs] enable_video` turns it on (off by default, like `enable_image` and `enable_audio`);
  `crucible install video` builds the env and places Crucible's ffmpeg.

## Backends

**cuda-linux and mlx-darwin**, one engine each (`videomodels.VIDEO_BACKEND_ENGINES`): `ltx`
(diffusers, below) on the PC and `ltx-2-mlx` (dgrauet/ltx-2-mlx, "The Mac arm") on the Mac.
Both are blocks of the one model, `ltx-2.5-distilled`: the same Lightricks checkpoint, the same
params, the same `video.mp4` and the same `done.video` fields, so a client sends the same job
to either server and never branches on the host. What differs between the arms is data in
their blocks: the size grid, the image-to-video ceiling, the refining steps, the memory. The
capability row reads the block's largest stage against the host: "can make video" on a 24 GB
card and on a Mac with 48 GB or more (29 GB after the 16 GB desktop allowance), "cannot make
video" on anything smaller. Windows is never a video backend: the PC runs Crucible in WSL2, and
a `llama-windows` host is refused by name.

owens-pc's facts that decide everything below: RTX 3090 Ti, 24 GB, Ampere (sm_86): int8 and
bf16 tensor cores, **no fp8 and no fp4 hardware**, so NVFP4 checkpoints cannot run and fp8 is
at best a storage format. Crucible's desktop allowance leaves about 21 GB for any one stage.
WSL's memory is capped at 13 GB (32 GB host) with 100 GB of slow swap, so **no stage may
materialize a bf16 component in host memory**: the text encoder is 24 GB in bf16, the
transformer 38 GB.

## Weights

The PC's; the Mac's are under "The Mac arm".

| repo | revision | what is pulled | bytes |
| --- | --- | --- | --- |
| `Lightricks/LTX-2.5-Diffusers` (gated, "Agree and Access") | `426936f8b22dc28e4def61e515478b0b7e4a53cc` | `model_index.json`, `scheduler/`, `tokenizer/`, `text_encoder/` (5 bf16 shards), `connectors/` (the 2-shard set and its index), `transformer/config.json` only, `vae/`, `audio_vae/`, `vocoder/`, the README | 32.1 GB |
| `Abiray/LTX-2.5-Distilled-GGUF` | `7b0c2025441f1bf12c18eac375ad21f5e3d3c9e0` | `LTX-2.5-Distilled-Q6_K.gguf`, sha256 `ee8835ff…6a58` | 18.6 GB |

About **50.7 GB** in all. What is deliberately not pulled: `transformer/`'s bf16 shards (two
38 GB sets; the GGUF replaces them), `transformer_full/` (the undistilled DiT), the 9.7 GB
distilled LoRA, `prompt_enhancer/` (a separate Gemma 4 for prompt rewriting, not used),
`diffusion_decoder/` (a DiT decoder that needs NATTEN kernels from the Hub), the latent
upsamplers (the two-stage recipe), `duration_head/` (a job always states its length), and the
unsharded `connectors/diffusion_pytorch_model.safetensors` (the same weights as the 2-shard
set; diffusers prefers the sharded set whenever its index is present, and 5 GB shards are the
smaller read).

Which transformer folder is the distilled one: the Diffusers README's layout table says
`transformer/` is the distilled DiT (the default in `model_index.json`) and `transformer_full/`
the full SFT one. `transformer/` holds two shard sets (4 x 10 GB and 8 x 5 GB) from the one
release commit; diffusers' `scripts/convert_ltx2_to_diffusers.py` at the pinned commit saves a
component on its own with `save_pretrained`'s 10 GB default and a whole pipeline with
`max_shard_size="5GB"`, which is where the two sets come from. The index that says which one
`from_pretrained` would read is behind the gate. The worker never reads either set, so the
question does not arise; this is also why the transformer comes from a GGUF and not from
Lightricks' own shards (next section).

Only `transformer/config.json` is pulled from that folder: `from_single_file` builds the model
from it.

The GGUF repo is a community quantization of the distilled transformer, ungated, under the same
LTX-2.x licence (its model card and the `license` key in the file's own header). Its header
also carries `model_version = 2.5.0` and the original transformer config. The companion is
pinned by revision and sha256, so a changed upload fails the pull (`weights.pull_files`: "NOTHING
was placed") instead of running.

## Quantization, per stage, and what was rejected

| stage | component | how it is on the card | weights on the card | peak declared |
| --- | --- | --- | --- | --- |
| encoding | Gemma 4 12B text encoder (bf16 shards, 24 GB; loaded with `AutoModelForImageTextToText`, whose config is behind the gate) | torchao `Int8WeightOnlyConfig` (per-row int8, bf16 activations) through transformers' `TorchAoConfig`, quantized tensor by tensor as it loads | ~13 GB (torchao leaves the token embedding in bf16) | 16 GB |
| connecting | text connectors (3.2 B parameters) | bfloat16 | 6.3 GB | 8 GB |
| conditioning (image-to-video only) | video VAE encoder | bfloat16, tiled | 1.45 GB | 4 GB |
| denoising | distilled transformer (18.99 B parameters) | GGUF Q6_K through diffusers' `GGUFQuantizationConfig`, dequantized one layer at a time to bf16 | 16.1 GB | **20.5 GB** |
| decoding | video VAE decoder | bfloat16, spatial tiles (512 px) and temporal tiles (16 frames) | 1.45 GB | 9 GB |
| audio_decoding | audio VAE and vocoder (48 kHz, with bandwidth extension) | bfloat16 | 0.36 GB | 2 GB |

`memory_bytes_estimate` is the largest stage, 20.5 GB, and the parser refuses a manifest whose
estimate is anything else. `stage_memory_bytes` in the manifest carries the table's last column.

**The transformer.** In bf16 it is 38 GB and cannot be on the card at all. Its candidates:

- **int8 weight-only from Lightricks' bf16 shards** (torchao or bitsandbytes quantize-on-load):
  19.0 GB of weights before any activation. At 1280x704 x 6 s the activations are 2 to 3 GB,
  so a stage needs 21 to 22 GB plus the CUDA context: over the budget. bitsandbytes' 8-bit path
  also casts activations to fp16 (`MatMul8bitLt: inputs will be cast ... to float16`), which a
  model trained in bf16 cannot be trusted to survive.
- **NF4 from the bf16 shards** (bitsandbytes 4-bit): about 10.5 GB and it fits easily, but
  4-bit normal-float is well below Q6_K in quality, and the brief is the best quality that fits.
- **GGUF Q8_0**: 23.6 GB, over the budget.
- **GGUF Q6_K (chosen)**: 6.56 bits a weight in k-quant blocks with per-block scales, generally
  close to Q8 for diffusion transformers; 16.1 GB on the card once its 2,603 float
  tensors are cast to bf16, leaving about 4 GB for activations. Q5_K_M (18.1 GB file) saves
  almost nothing over Q6_K in this upload, so it buys no room.
- **Lightricks' ComfyUI int8 "convrot" transformer** (21.5 GB, official): ComfyUI's own format,
  which no pinned library loads outside ComfyUI, and at 21.5 GB it does not leave room anyway.
- **NVFP4** (18.7 GB, official): needs fp4 hardware (Blackwell). The 3090 Ti has none.
- **Lightricks' own `ltx-core` / `ltx-pipelines`** (1.0.0 on PyPI, 2026-03-18): predates
  LTX-2.5 by five months; its fp8 path is storage plus upcast, which on sm_86 is the int8 size
  problem again.

diffusers supports GGUF for this model class (`LTX2VideoTransformer3DModel` is single-file
loadable, and `GGUFQuantizationConfig` supports Q6_K), with two gaps at the pinned commit that
the worker closes, both checked against the source:

1. `convert_ltx2_transformer_to_diffusers` (`loaders/single_file_utils.py`) has only the
   LTX-2.0 rename table. LTX-2.3/2.5 add `prompt_adaln_single` and `audio_prompt_adaln_single`,
   which its own `scripts/convert_ltx2_to_diffusers.py` renames (`LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT`,
   loaded there with `strict=True`). `crucible/jobs/video/ltxkeys.py` restates that script's table;
   the worker renames every tensor itself and drops the connector tensors (the connectors come
   from Lightricks' own `connectors/`). `from_single_file` then sees a state dict whose keys
   already match and skips its converter (`_should_convert_state_dict_to_diffusers`). Before
   loading, the worker builds the model on the meta device and refuses by name if any parameter
   is missing or left over. `tests/data/ltx25_gguf_tensor_names.txt` is the real header's names
   (top level, block 0, connector block 0) and the tests map every one of them.
2. `load_gguf_checkpoint` (`models/model_loading_utils.py`) builds each tensor with
   `torch.from_numpy(tensor.data.copy())`: all 18.6 GB in anonymous host memory, which the 13 GB
   WSL cap would push into swap. The worker instead wraps `GGUFReader`'s read-only memmap
   views (`torch.from_numpy(tensor.data)`, no copy) and hands diffusers that dict;
   `load_model_dict_into_meta` then moves them to the card one tensor at a time
   (`GGUFQuantizer.create_quantized_param` calls `.to(device)`; float tensors are cast to bf16
   one at a time on the CPU). What host memory holds is page cache for the file, which the
   kernel drops under pressure.

**The text encoder.** bf16 is 24 GB. torchao int8 weight-only keeps activations in bf16 (the
path in `torchao/quantization/quantize_/workflows/int8/int8_tensor.py` is a bf16 matmul against
the int8 weight cast per call), so Gemma's large activations never meet fp16. transformers 5.17
applies it through a per-tensor conversion op (`integrations/torchao.py`, `TorchAoQuantize`) as
each tensor is read from its safetensors shard onto the card, so neither the card nor host
memory ever holds the bf16 model. NF4 would fit too and was not needed. The GGUF text encoder
(`elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF`) was rejected: transformers dequantizes a GGUF to
full precision on load, which is the 24 GB problem again, and that repo is a third party's
repackaging with a projection baked in for ComfyUI.

## Loading, and the stand-in connectors

The worker's `load` puts nothing on the card. It reads the VAE, audio VAE and vocoder into host
memory (1.8 GB, kept for the life of the worker: the pipelines read their latent statistics and
configs from them) and builds `LTX2Pipeline` and `LTX2ImageToVideoPipeline` around them with no
text encoder, connectors or transformer. Each job then runs the stages in order; after each one
the worker records `torch.cuda.max_memory_reserved`, deletes the component, runs the garbage
collector, empties the cache and resets the peak, as the image worker does.

The denoising stage uses the stock pipeline `__call__` (so the reference loop, sigma handling,
RoPE coordinates and audio length are diffusers' own), in the distilled recipe the Diffusers
README gives: `sigmas=DISTILLED_SIGMA_VALUES` (8 values), guidance 1.0, STG 0.0, modality 1.0,
video and audio alike. The pipeline calls its own `connectors` on the text encoder's output;
since the connectors ran and were freed in the stage before, the worker sets a small module in
their place that returns that stage's output, and hands the pipeline a one-element placeholder
for `prompt_embeds` (it reads only the batch size and dtype from it). With no guidance the
negative prompt is never encoded, which is why `negative_prompt` is refused by name.

Image-to-video: the conditioning stage recompresses the picture as H.264 at CRF 18
(`LTX2_5_IMAGE_CRF`, what LTX-2.5 was trained against; the pipeline would pick 33, the LTX-2.3
value, because it finds no Gemma 4 text encoder loaded at that point), crops and resizes it to
the job's size (`resize_mode="crop"`: no stretching), encodes it with the VAE (the latent's
mode), and passes the latent to the image-to-video pipeline with `noise_scale=1.0`, which keeps
the first latent frame clean and makes the rest noise, as the pipeline's own image path does.

The prompt cache holds the connectors' output (about 12 MB a prompt, 16 prompts, 256 MB at
most) keyed on the prompt, revision and backend. A worker only lives across jobs under a lease,
so the cache pays off in a leased batch: the second clip of a prompt skips the text encoder and
connectors stages entirely.

## Limits

Refused by name before anything loads (`crucible/jobs/video/params.py`); every figure is
declared, sized to the memory table above:

| limit | value | refusal |
| --- | --- | --- |
| frame size | multiples of 32, sides 256 to 1280, at most 901,120 pixels (1280x704 either way up) | `video_size_not_supported`, `video_too_large` |
| frame count | 8k+1 (the VAE's causal grid), at most 145 | `video_frames_not_supported`, `video_too_long` |
| frame rate | 24 or 25 | `video_param_unsupported` |
| text-to-video | at most 16,720 video tokens: 1280x704 x 145 frames, 6.04 s at 24 fps | `video_too_large` |
| image-to-video | at most 8,800 video tokens: 1280x704 x 73 frames (3.04 s), 960x544 x 129 frames (5.4 s) | `video_too_large`, checked in `run` before the worker starts |
| steps | exactly 8 | `video_param_unsupported` |

A video token is one latent cell: `((frames - 1) / 8 + 1) x (width / 32) x (height / 32)`.
Denoising memory grows with it: about 130 KB a token for text-to-video (the self-attention's
RoPE tables and the widest block's feed-forward intermediates in bf16) on top of 16.1 GB of
weights and 1.2 GB of dequant scratch and allocator slack, 19.5 GB at 16,720 tokens.
Image-to-video conditions each token on its own timestep (`video_timestep` is per token, so
`temb` is `[tokens, 9 x 4096]` and each block builds a same-sized modulation tensor), about
150 KB more a token, which is why its ceiling is lower.

## Env

`crucible/envs/video/ltx-cuda-linux.txt` (env key `video-ltx`, Python 3.11): the image recipe's
resolution, which pins the same `torch==2.14.0`, `transformers==5.17.0` and diffusers at
`5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2` (2026-09-28; no released diffusers has LTX-2.5:
0.40.0 came out the day before the model), plus `gguf==0.19.0` (diffusers' GGUF reader; its
dependencies are already in the resolution), `torchao==0.18.0` (weight-only int8; its compiled
extensions are optional and the weight-only path is plain torch) and `av==18.1.0` (the CRF
recompression of a start picture; the last `av` with cp311 wheels). Like the image recipe it is
a resolution, not yet a `pip freeze` of a working env; replace it with one after the first
install on the PC. Smoke import: `import diffusers, gguf, torchao, av`. `# archive-bytes:` is
the image recipe's 3.07 GB plus those three wheels from PyPI (2026-09-29), 3.11 GB.

The Mac's recipe is `crucible/envs/video/ltx-2-mlx-mlx-darwin.txt` (env key
`video-ltx-2-mlx`, Python 3.11, no torch): a `uv pip compile --python-platform
aarch64-apple-darwin` resolution (macOS 15 wheel tags; the Mac runs 26) of the two ltx-2-mlx
packages at the pinned commit plus the pins below, 39 lines. ltx-2-mlx is a uv workspace
monorepo (`packages/ltx-core-mlx`, `packages/ltx-pipelines-mlx`, `packages/ltx-trainer`, each
a hatchling project), so pip takes each package as
`<name> @ git+https://github.com/dgrauet/ltx-2-mlx@<commit>#subdirectory=packages/<name>`, the
way the tts recipe takes narrator; `ltx-pipelines-mlx` depends on `ltx-core-mlx` by name and
the direct reference satisfies it. The trainer is not installed. `mlx==0.32.2` and
`mlx-metal==0.32.2` are the image env's (mflux's) and also what the port's own `uv.lock`
holds. `mlx-arsenal` is pinned to the lock's 0.2.4: left alone the resolver takes 0.16.0, and
the transformer's timestep embedding and the Euler step come from that package.
`transformers==5.17.0` (the other Mac envs'; the lock has 5.3.0) is there only for Gemma 4's
`AutoTokenizer`. `av==18.1.0` is Crucible's own addition (the start picture, "The Mac arm").
Smoke import: `import ltx_pipelines_mlx, ltx_core_mlx, mlx_arsenal, transformers, av`.
`# archive-bytes:` is the 37 wheels' sizes from PyPI (the macOS arm64 cp311 wheel of each,
99.2 MB) plus the source archive at the commit (5.9 MB), 2026-09-30. Like the PC's recipe it is
a resolution, not yet a `pip freeze` of a working env.

## Output and the mux

The worker turns the decoded frames into 8-bit RGB (the pipeline's own postprocess arithmetic,
eight frames at a time on the card) and the vocoder's waveform into 16-bit PCM, writes the PCM
as a WAV beside the output, and pipes the frames into the ffmpeg Crucible ships
(`crucible/hosttools.py`), whose path the job resolves before the worker starts
(`ffmpeg_missing` at submit otherwise). That ffmpeg is BtbN's LGPL build, which has no x264
(`scripts.d/50-x264.sh` is disabled for `lgpl*` variants) but always builds openh264 and the
NVENC headers, so the encoder is picked from `ffmpeg -encoders` in the order libx264,
libopenh264, h264_nvenc, h264_videotoolbox: libopenh264 on the PC, at 0.3 bits a pixel
(7.6 Mb/s at 1280x704 x 24). The Mac's ffmpeg is Crucible's own static LGPL n8.1.3 build
(`--enable-videotoolbox --enable-audiotoolbox`, no x264, no openh264; its configure line is in
the binary), so there it is h264_videotoolbox at 0.4 bits a pixel (8.7 Mb/s at 1280x704 x 24;
Apple's encoder needs more bits than x264 for the same picture), `-profile:v high`, and
`-allow_sw 1` so a session that cannot open the hardware encoder falls back to Apple's software
one instead of failing. Audio is AAC (ffmpeg's own encoder, in both builds) at 192 kb/s, 48 kHz
stereo, `-shortest`, `+faststart`. `done.video.encoder` says which ran. The 1.0.68 fix stands:
`communicate()` closes ffmpeg's stdin itself, on both arms.

## Licence

The LTX-2.x Community License Agreement, licence date 2026-08-11
(`Lightricks/LTX-2@a95ab856bf29407b6b066ede0abe1846050db56c`, file `LICENSE-2_x`), which the
Diffusers repo and the GGUF repo both name. Free for any purpose, commercial and production
included, for an entity whose annual revenue with its affiliates is under US$10,000,000
(section 2.1). At or above that, a paid Commercial Use Agreement is required
(ltxv-licensing@lightricks.com), except for the narrow Non-Commercial Purposes of section 2.2
(an individual's personal, non-business use; a commercial entity's non-production testing and
evaluation). Lightricks claims no rights in outputs (section 5). Section 6 requires keeping any
safety, disclosure, watermark or provenance features the model or its outputs carry; Crucible
adds and removes nothing. The Hugging Face gate also asks the account to accept Lightricks'
privacy policy and marketing consent. Redistributing the weights or a derivative requires
passing on the licence and its Attachment A use restrictions (section 3).

The Mac's weights are a derivative under the same terms: `dgrauet/ltx-2.5-mlx-q8`'s card sets
`license: other`, `license_name: ltx-2-community-license-agreement`, `base_model:
Lightricks/LTX-2.5`, and says the weights are "distributed under the same terms as the model
they were converted from"; its `LICENSE` (34,441 bytes) is the pinned `LICENSE-2_x` text line
for line, only indented. The repo is gated with automatic approval: one "Agree and Access" on
its page, then the same HF token. The library, ltx-2-mlx, is MIT (Copyright (c) 2025 dgrauet),
and so is mlx-arsenal.

## The Mac arm

owens-mac-studio: M1 Ultra, 64 GB unified memory, macOS 26. Crucible keeps a 16 GB desktop
allowance there, so any one stage may use about 48 GB, and a block needs its largest stage and
no more (29 GB here). M1 has no bf16 arithmetic in hardware (M2 and later do); MLX runs bf16 on
it in software, which costs time, not correctness. Every number the port publishes was measured
on an M2 Pro 32 GB or an M3 Max, so none of its timings transfer.

**The library.** dgrauet/ltx-2-mlx v0.15.12, commit
`1724ca673d59f023a8a95efee06e5d36d61c2765` (2026-09-27, the latest release; the repo's
`pushed_at` of 2026-09-28 is a branch push after it), MIT: a pure-MLX port of Lightricks'
`ltx-core` and `ltx-pipelines` with LTX-2.5 support (Gemma 4 text encoder, audio, the 2.5 sigma
tables and ancestral sampler), every public pipeline mirroring an upstream class. Read at that
commit: `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/distilled.py`, `_base.py`,
`ti2vid_two_stages.py`, `utils/blocks.py`, `utils/samplers.py`, `utils/_orchestration.py`,
`utils/media_io.py`, `scheduler.py`, `cli.py`, and in `ltx-core-mlx`
`text_encoders/gemma/encoders/gemma4_encoder.py`, `encoder_configurator.py`, `gemma4.py`
(`load_from_pack`), `model/video_vae/video_vae.py`, `loader/`, `utils/weights.py`.

**The recipe: the distilled two-pass, not the PC's one pass.** The port has no distilled
one-pass pipeline: its `--one-stage` is the dev transformer with CFG, and its `--distilled` is
`DistilledPipeline`, which mirrors upstream's `DistilledPipeline` exactly: the distilled
transformer for 8 steps at half the size (on a 2.5 pack the Euler ancestral, SDE sampler on
`LTX_2_5_DISTILLED_SIGMAS`, the noise seeded from the job's seed plus 10,000), the half-size
latent upscaled 2x by `spatial_upscaler_x2_v1_0`, re-noised to 0.909375, then 3 deterministic
Euler steps at full size on `LTX_2_5_STAGE_2_DISTILLED_SIGMAS`, no guidance in either pass. Its
docstring gives the reason upstream does it this way: running the distilled model directly at
full resolution "can produce out-of-distribution artefacts". It is the path the port calls
Stable and validated end to end on 2.5 q8 packs. Crucible takes it as it is rather than
assembling a one-pass loop out of the port's internals that nobody has run. What follows:

- the size grid is 64 pixels (the half-size pass must itself sit on the VAE's 32-pixel grid);
  the Mac block says `size_multiple = 64`, so a size off it is refused by name rather than
  floored by the port's `snap_output_dimensions`. 1280x704, the default, is on it;
- `steps` is still exactly 8 (the half-size pass); the block declares `refine_steps = 3`, the
  request carries it, and `done.video` reports it with a `sampling` entry per pass;
- a seed reproduces a clip on its own arm only: the two arms use different samplers and
  different libraries.

**The worker drives it one stage at a time.** `generate_and_save` would run everything and
write an mp4 through the ffmpeg on PATH with libx264. `ltx2mlx_worker.py` builds a
`DistilledPipeline` subclass once, at load (it reads only `embedded_config.json` and the 3.8 MB
duration head, and refuses a pack that is not 2.5), and calls the pinned commit's methods in
order:

| stage | what runs | in memory (MLX) | declared |
| --- | --- | --- | --- |
| encoding | `PromptEncoder.encode`: Gemma 4 12B at int8 (group 64, `quantize_config.json`) from `text_encoder.safetensors` with its tokenizer (transformers' `AutoTokenizer`), and the connector (`connector.safetensors`, plus the aggregate-embed projections that live in the text encoder file), prompt left-padded to 1024 tokens; freed after | 16.0 + 4.0 GB of weights | 23 GB |
| denoising | `_stage1`: the distilled transformer (`transformer-distilled.safetensors`, int8, all 48 blocks in memory: no `--low-ram` block streaming, which the 64 GB Mac does not need and which costs time), the VAE encoder and the upscaler load; a start picture is encoded here; 8 steps at half size | 20.6 + 0.6 + 1.0 GB of weights, activations at a quarter of the tokens | 26 GB |
| refining | `_upsample_latent`, then `_stage2` (it frees the VAE encoder and the upscaler before its 3 steps); then the transformer is dropped | 20.6 GB of weights plus about 150 KB a video token at full size | **29 GB** |
| decoding | the conv VAE decoder's `tiled_decode`, tiled by the port's own `_compute_decode_tiling` to a 12 GiB budget (`DECODE_BUDGET_BYTES`; the port's default is half of unified memory, 32 GB here) inside its `decode_cache_limit()`, each chunk turned into 8-bit RGB and copied out | 0.8 GB of weights, the tile | 16 GB |
| audio_decoding | the `AudioDecoder` block: audio VAE and the vocoder with bandwidth extension (its weights upcast to fp32), 48 kHz stereo, into 16-bit PCM | 0.1 + 0.5 GB | 3 GB |

After each stage the worker reads `mx.get_peak_memory()` (MLX's own high-water mark, not the
process footprint), logs a `crucible memory` line, runs the garbage collector, clears MLX's
cache and resets the peak. The cache limit is the block's `mlx_cache_limit_bytes`, 4 GB, as for
`image` on the Mac. At 1280x704 x 145 frames the full-size pass is 16,720 video tokens (the
half-size pass 4,180); the refining figure is the transformer's 20.6 GB plus the widest block's
feed-forward (16,384 wide), the attention inputs (the port uses `mx.fast.scaled_dot_product_attention`,
which never holds the score matrix) and, for image-to-video, the per-token modulation: about
2.5 GB at the text-to-video ceiling and 2 GB more for image-to-video at its own, leaving about
6 GB of the 29 declared for MLX's lazily built graph and allocator.

Three hooks, all read from the pinned source: stage 1 encodes the prompt through
`_load_text_encoder` and `_encode_text`, which the subclass answers with the encoding stage's
output (so the prompt cache works as on the PC: a leased batch skips encoding for a prompt it
has read); the per-step progress comes from the only per-step hook the port's loops have, its
stepwise-preview `bind`, answered by a small object that evaluates the step's prediction (the
next step needs it anyway) and reports the step, which is also where a cancel lands; and the
decode and the audio go through the port's blocks rather than its muxer. `_stage1`,
`_stage2`, `_upsample_latent` and `_compute_decode_tiling` are private in the port, so **a new
ltx-2-mlx commit is a code review, not a pin bump**.

**The start picture.** Upstream feeds a start picture through libx264 before the VAE sees it,
because LTX-2 was trained on frames carrying H.264 artefacts. The port does that by running
`ffmpeg -c:v libx264` from PATH, and Crucible's Mac ffmpeg has no libx264. The worker does the
round trip itself with PyAV (the macOS arm64 `av` 18.1.0 wheel links `libx264.165.dylib`;
checked in the wheel's own `libavcodec` configure line), at CRF 18 (diffusers'
`LTX2_5_IMAGE_CRF`, which the PC uses; the port's default, 33, is the LTX-2.3 value), writes a
PNG beside the output and hands the pipeline that with `crf=0`, so the picture is recompressed
once, before the pipeline crops it to the clip's shape (never stretched). The picture is
encoded inside the denoising stage, so the Mac has no separate `conditioning` stage.

**The weights.** `dgrauet/ltx-2.5-mlx-q8`, revision
`746ca9aacb697d2c739f544d68b584214dedcc75` (2026-08-28), made by the port's author with
mlx-forge (`convert ltx-2.5 --quantize --bits 8`: int8, group size 64, the transformer blocks'
and the text encoder's Linear weights). Gated (automatic approval). The block pulls 20 of its
26 files, 43.5 GB of 74.7 GB:

| pulled | bytes |
| --- | --- |
| `transformer-distilled.safetensors` | 20,595,222,296 |
| `text_encoder.safetensors` (Gemma 4, int8) | 16,013,053,557 |
| `connector.safetensors` | 4,032,375,032 |
| `spatial_upscaler_x2_v1_0.safetensors` + its config | 995,745,611 |
| `vae_decoder_conv.safetensors` | 814,351,176 |
| `vae_encoder_conv.safetensors` | 637,887,026 |
| `vocoder.safetensors` | 258,315,824 |
| `audio_vae.safetensors` | 106,510,732 |
| `tokenizer.json`, `duration_head.safetensors`, the JSON configs, `chat_template.jinja`, README, LICENSE | 36,043,942 |

Not pulled: `transformer-dev.safetensors` (20.6 GB, the undistilled transformer the CFG
pipelines use), `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (8.9 GB, for the dev
pipelines), `temporal_upscaler_x2_v1_0` (DFR's temporal rounds), and the `vae_*_av` diffusion
decoder and its encoder (the port's experimental `--video-decoder diffusion`). The duration head
is pulled although a job always states its length, because the pipeline loads it at
construction and it is 3.8 MB. The configs behind the gate (`embedded_config.json` and the
rest) could not be read from here; what the code reads from them is named above.

Why int8 (q8): the bf16 pack's distilled transformer is 38.0 GB and its text encoder 26.2 GB,
so the full-size pass would need about 42 GB of the 48 and the text stage 30; q8 is the pack
the port calls "Recommended; validated e2e on all 2.5 pipelines". The q4 pack (11.3 GB
transformer) would fit a 32 GB Mac but the port lists it as "Lower quality", and the Studio has
the room for q8.

**Limits on the Mac**, refused by name before anything loads, declared like the PC's:

| limit | value |
| --- | --- |
| frame size | multiples of **64**, sides 256 to 1280, at most 901,120 pixels (1280x704 either way up) |
| frame count | 8k+1, at most 145; frame rate 24 or 25 |
| text-to-video | at most 16,720 video tokens (1280x704 x 145 frames, 6.04 s), the same as the PC |
| image-to-video | at most **14,080** video tokens (1280x704 x 121 frames, 5.04 s), higher than the PC's 8,800: the Mac has the memory for the per-token modulation |
| steps | exactly 8, then 3 refining steps |

A video token is counted at full size, `((frames - 1) / 8 + 1) x (width / 32) x (height / 32)`,
on both arms (`videomodels.video_tokens`; the 64-pixel grid does not change the VAE's 32-pixel
cell).

## What the first runs on the PC must measure

The first 5 s clip ran on 2026-09-30 (the figures at the top). Still owed on owens-pc, with
nothing else on the card:

1. `crucible install video`, then replace the recipe with a `pip freeze` of the env that
   imports.
2. `crucible models pull ltx-2.5-distilled` with an HF token whose account accepted the
   licence (50.7 GB; the GGUF is hashed after download).
3. One 768x512 x 49-frame text-to-video, then one 1280x704 x 145-frame (the text-to-video
   ceiling), then one image-to-video at 1280x704 x 73 frames (its ceiling), reading
   `done.video.stage_peak_bytes`, the worker log's `crucible memory` lines, nvidia-smi's peak
   and the WSL memory high-water mark (`free -g` during encoding and denoising).
4. Replace `stage_memory_bytes` and `memory_bytes_estimate` with the measured peaks plus the
   CUDA context, and set `memory_basis = "measured"`. If denoising at the ceiling passes the
   20.5 GB cap it fails inside the worker as a CUDA out-of-memory, never on the desktop, and the
   fix is a lower `max_video_tokens`. If there is room, raise the ceilings.
5. Look at the clips: Q6_K against the reference quality, the start frame of image-to-video,
   the sound, and the mux (plays in a browser, audio in sync).
6. Time each stage. Every job reads about 50 GB from the model store (text encoder, connectors,
   GGUF), since no component stays on the card between stages; on slow storage that dominates.

## What the first run on the Mac must measure

Nothing of the Mac arm has run: not the env, not the pull, not a stage. Owed on
owens-mac-studio through the installed Crucible, with nothing else resident:

1. `crucible install video`: that pip builds both ltx-2-mlx packages from the git commit (the
   Mac needs git for the `git+` references, as it already does for the tts recipe) and the
   smoke import passes; then replace the recipe with a `pip freeze` of that env.
2. Accept the licence at https://huggingface.co/dgrauet/ltx-2.5-mlx-q8 with the account the
   token belongs to, then `crucible models pull ltx-2.5-distilled` (43.5 GB). That
   `AutoTokenizer` loads the pack's tokenizer offline (`HF_HUB_OFFLINE=1`) from files alone,
   since the pack has no `config.json` and its `tokenizer_config.json` is behind the gate.
3. One 768x512 x 49-frame text-to-video, one 1280x704 x 121 (the default), one 1280x704 x 145
   (the text-to-video ceiling) and one image-to-video at 1280x704 x 121 (its ceiling), reading
   `done.video.stage_peak_bytes`, the worker log's `crucible memory` lines, and the process
   footprint (Activity Monitor's "Memory", or `footprint <pid>`), which MLX's peak does not
   include. Replace the stage table with measured figures and `memory_basis = "measured"`; if
   refining at the ceiling goes past 29 GB, lower `max_video_tokens` (or the image-to-video
   ceiling) rather than raise the declaration past what a 48 GB Mac leaves.
4. Whether the macOS GPU watchdog (`MTLCommandBufferErrorInternal`, code 14) fires on a
   full-size step at 16,720 tokens. The port evaluates once per step, which the worker's step
   hook keeps; if it fires, the port's modality tiling (`--tile-spatial`) or block streaming
   (`low_ram_streaming=True`, `mx.set_cache_limit(0)`) are the knobs, at a cost in time.
5. Time each stage on the M1 Ultra (bf16 in software): the text stage and the transformer
   load read 36.6 GB from the model store per clip, and the full-size pass is the O(N^2) one.
   The worker's silence timeout is 30 minutes between messages; a step that takes longer than
   that at the ceiling means a lower ceiling.
6. Look at the clips: the two-pass picture against the PC's one-pass one, the start frame of
   image-to-video after the CRF 18 round trip, the sound (48 kHz stereo), and that
   h264_videotoolbox's file plays in a browser with the sound in sync. Check the encoder opens
   from Crucible's launchd session; `-allow_sw 1` falls back to Apple's software encoder if not.
