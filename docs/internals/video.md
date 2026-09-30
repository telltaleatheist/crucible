# Video generation internals

The `video` job type makes a clip with synchronized sound from a prompt, or from a prompt and a
start picture, with LTX-2.5 (Lightricks, released 2026-08-21: a 22B-parameter audio-video
diffusion transformer read by a Gemma 4 12B text encoder). The caller's side (params, result,
prompting) is [../VIDEO.md](../VIDEO.md). This page is how it runs on owens-pc and why each
number is what it is. Every memory figure here is **declared** from the files and the library
source, not measured through Crucible; the section "What the first runs on the PC must measure"
lists what replaces them.

## Shape

The same shape as `image` and `audio`:

- `crucible/video/<id>.toml` is the manifest, parsed by `crucible/videomodels.py`. One model
  ships: `ltx-2.5-distilled`.
- `crucible/videoweights.py` pulls the gated Lightricks snapshot through `weights.pull` (only
  the files the manifest names) and the transformer GGUF as a hash-verified companion through
  `weights.pull_files`, the way `audioweights.py` pulls YuE2's decoder. The model is installed
  only when both are.
- `crucible/jobs/video/` is a `ResidentWorker` with lease-on-load: `video`, `load-video` and
  `unload-video`, resident kind `video` (noun `video generator`, so the unload refusal is
  `video_generator_not_resident`), capability class `video`, lease act `video`.
- The worker (`ltx_worker.py`) is a workerio session. Its engine-independent half
  (`videocore.py`: the request, stage progress, cancel between steps, the mux) needs only the
  standard library, and `tests/fake_video_worker.py` runs the same `Worker` with a fake engine.
- `[jobs] enable_video` turns it on (off by default, like `enable_image` and `enable_audio`);
  `crucible install video` builds the env and places Crucible's ffmpeg.

## Backends

**cuda-linux only.** The manifest has no `mlx-darwin` block, `videomodels.VIDEO_BACKEND_ENGINES`
names only `cuda-linux`, and `jobenv.video_envs("mlx-darwin")` refuses by name. On the Mac the
capability row says "cannot make video" (no candidate for the backend, the way YuE2 songs read
there), `crucible install video` fails naming cuda-linux, and a job is refused
`backend_unsupported` with `declared: ["cuda-linux"]`. Windows is never a video backend: the
PC runs Crucible in WSL2.

owens-pc's facts that decide everything below: RTX 3090 Ti, 24 GB, Ampere (sm_86): int8 and
bf16 tensor cores, **no fp8 and no fp4 hardware**, so NVFP4 checkpoints cannot run and fp8 is
at best a storage format. Crucible's desktop allowance leaves about 21 GB for any one stage.
WSL's memory is capped at 13 GB (32 GB host) with 100 GB of slow swap, so **no stage may
materialize a bf16 component in host memory**: the text encoder is 24 GB in bf16, the
transformer 38 GB.

## Weights

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

## Output and the mux

The worker turns the decoded frames into 8-bit RGB (the pipeline's own postprocess arithmetic,
eight frames at a time on the card) and the vocoder's waveform into 16-bit PCM, writes the PCM
as a WAV beside the output, and pipes the frames into the ffmpeg Crucible ships
(`crucible/hosttools.py`), whose path the job resolves before the worker starts
(`ffmpeg_missing` at submit otherwise). That ffmpeg is BtbN's LGPL build, which has no x264
(`scripts.d/50-x264.sh` is disabled for `lgpl*` variants) but always builds openh264 and the
NVENC headers, so the encoder is picked from `ffmpeg -encoders` in the order libx264,
libopenh264, h264_nvenc: libopenh264 on the PC, at 0.3 bits a pixel (7.6 Mb/s at 1280x704 x
24). Audio is AAC at 192 kb/s, 48 kHz stereo, `-shortest`, `+faststart`. `done.video.encoder`
says which ran.

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

## What the first runs on the PC must measure

Nothing here has run on a GPU. Owed on owens-pc, with nothing else on the card:

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
