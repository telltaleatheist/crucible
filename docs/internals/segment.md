# Segmentation internals

Covers `crucible/jobs/segment/` (the `segment`, `load-segment` and `unload-segment` job types,
`params.py`, `picture.py` and the worker `worker.py`), `crucible/segmentmodels.py`,
`crucible/segment/*.toml`, `crucible/envs/segment/*.txt`, the seventh resident kind
(`KIND_SEGMENT`, noun `segmenter`, unload refusal `segmenter_not_resident`) and the capability
classes `cutout` and `select`. The caller's page is [SEGMENT.md](../SEGMENT.md). The machinery
it plugs into is [jobs-runtime.md](jobs-runtime.md); [audio.md](audio.md) and
[image.md](image.md) are the two job types whose shape this copies (audio's per-model class and
refusals, image's one worker env and input picture); install, recipes and weights are
[config-envs-weights.md](config-envs-weights.md).

Why it exists (2026-09-29): Owen's rule is that Crucible supplies tools and the apps
(ContentStudio, a Photoshop-like editor) build the features around them. Apps need pixel masks,
"cut out the subject / remove the background" and "select the object I clicked", and nothing in
Crucible made one. The mask is a plain PNG (white selected, black not, the input's size) so an
app can pass it straight back as an input, for example to inpainting.

## Shape

- One family, `segment` (`[jobs] enable_segment`, default off; absent reads off, a rewriter
  that does not name it keeps it). Three types: `segment` (makes the segmenter resident and
  reuses it), `load-segment` (loads and leaves it resident; `LEAVES_IT_RESIDENT`) and
  `unload-segment`.
- One manifest per model in `crucible/segment/`, loaded by `segmentmodels.py`. `[model].kind` is
  `cutout` or `select` and decides three things: which capability class lists the model, whether
  the job takes `points` and `box` (`PROMPTED_KINDS`), and the lease act a job must send
  (`require_lease_request(act=kind)`; an act must be a capability class, so it is the class, not
  `segment`). `catalog_is_complete`: an undeclared id is refused.
- Each `[backends.<kind>]` arm declares the engine (`birefnet` or `sam2`, and the engine must
  make the model's kind), the pinned repo and revision, dtype, the memory estimate with its basis
  and note, the files to pull, `working_side` (what the model sees: 1024) and `max_pixels` (the
  largest input it takes, 40 MP).
- The job refuses by name before anything loads: params against the model's class
  (`params.py`, `refuse_what_the_model_cannot_take`), then, in `run`, the input
  (`input_picture`: one input, PNG/JPEG/WebP magic, the size from its header, `max_pixels`), then
  every point and the box against that size (`refuse_outside`). A worker that raised on a bad
  point would exit (workerio ends the session on an error), so nothing the caller can get wrong
  reaches it.
- One `segment` request per job; `done.segment` carries the effective params, revision,
  per-stage seconds and peaks, SAM's `score`, `multimask`, the mask's `coverage`, and the
  estimate with its basis. Two artifacts, `mask.png` and `cutout.png`, each checked for the PNG
  signature before it is published.

## Backends and engines

| model | cuda-linux | mlx-darwin |
| --- | --- | --- |
| `birefnet` | `transformers` remote code (`trust_remote_code`, the repo's own `birefnet.py`), CUDA, float16 | the same on Metal (MPS), float32, `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `sam2.1-hiera-large` | `transformers` `Sam2Model` / `Sam2Processor`, CUDA, float32 | the same on MPS, float32 |

- **BiRefNet** (`ZhengPeng7/BiRefNet` at `e2bf8e44`, 2026-02-04, the commit that made its code
  work with transformers 5's meta-device loading and `all_tied_weights_keys`). The repo is
  code plus weights: `config.json`'s `auto_map` points `AutoModelForImageSegmentation` at
  `birefnet.py`, which imports timm, kornia, einops and torchvision. The worker loads it from
  Crucible's own copy (`HF_HUB_OFFLINE=1`; transformers copies the two `.py` files into its
  modules cache), on the CPU, casts it and only then moves it to the device. Preprocessing is the
  model card's: resize to 1024x1024, ImageNet mean and std; the last output through a sigmoid is
  the mask, scaled back to the input's size with bilinear resampling, soft edges kept. The weights
  are stored in float16 (the author's 2025-03-31 commit); CUDA runs them in float16, which the
  author validated at "~0 decrease of performance" (the BiRefNet README, 2025-01-06).
- **BiRefNet on the Mac.** Its decoder uses `torchvision.ops.deform_conv2d`, which has no MPS
  kernel. The worker env on mlx-darwin sets `PYTORCH_ENABLE_MPS_FALLBACK=1`, so those layers run on
  the CPU (unified memory, so the copy is cheap) and the rest on Metal. float32 there, because the
  CPU fallback of a half-precision deformable convolution is not something to rely on. This is the
  arm most likely to need a change after the first real run (below).
- **SAM 2.1** (`facebook/sam2.1-hiera-large` at `665f8e2a`, 2025-08-15, "Add Transformers
  weights"). The same repo carries Meta's original `.pt` and the transformers-format
  `model.safetensors`; the arm pulls only the latter and the processor configs. transformers has
  had `Sam2Model` since 4.56; the pinned 5.17.0 has it, so the official `sam2` package is not
  needed. The checkpoint's config is the video model's (`model_type: sam2_video`); `Sam2Model`
  loading it is what the model card's own example does, and transformers 5.17.0's own
  mask-generation table maps `sam2_video` to `Sam2Model`. The image model leaves the video
  model's memory-attention weights unused. The prompts go through `Sam2Processor` (points, labels and
  boxes normalised to the 1024 frame), `multimask_output` is true only for a single point with no
  box (SAM's advice for an ambiguous click; the best of three by `iou_scores` is kept), and
  `post_process_masks` scales the low-resolution logits to the input's size and thresholds them
  at 0, so SAM's mask is hard. float32 on both backends: the model is small and the official
  predictor's bfloat16 autocast is a speed choice this job does not need.
- **Why not `rembg` / `transparent-background` / the `sam2` package.** Each is a wrapper with its
  own model download at run time; Crucible pins every byte it runs. transformers covers both
  models from pinned repos.
- Windows (`llama-windows`) is never a segment backend (`WSL_ONLY_JOB_TYPES`).

## Envs

One env, `envs/segment`, for both models: `Env("segment", worker=True)`, so it rides the worker
env loops (install, install-on-submit, `tasks.validate`, doctor's `worker_envs`) with no code of
its own, the way `image` does. Headline package `transformers`; the install's smoke import is
`transformers.models.sam2, timm, kornia, einops, torchvision` (`jobenv.SEGMENT_SMOKE_IMPORT`), so a
build that could not run either model is not called installed.

Why not share an existing env: the image env on the PC already holds torch 2.14 and transformers
5.17, but not timm, kornia or einops, and the image env on the Mac is mflux's; adding BiRefNet's
code deps to the image recipes would rebuild the image env on both machines for a job that does
not use them, and couple two job types' pins. A separate env costs a second copy of torch on disk.

Each recipe is a full lock, resolved on 2026-09-29 with `uv pip compile` (uv 0.12.21) for Python
3.11 from the top-level pins `torch==2.14.0`, `torchvision==0.29.0`, `transformers==5.17.0`,
`timm==1.0.30`, `kornia==0.8.3`, `einops==0.8.2`, `pillow==12.3.0` and unpinned `numpy`,
`safetensors`, `huggingface_hub`: `--python-platform x86_64-manylinux_2_28` for cuda-linux (which
brings torch's CUDA 13 wheels and triton, the same torch the image env installs) and
`aarch64-apple-darwin` with `MACOSX_DEPLOYMENT_TARGET=14.0` for mlx-darwin (torch 2.14 ships
`macosx_14_0_arm64` wheels only; the image, align and asr envs already run it on the Mac). The two
resolutions agree on every package they share. The `# archive-bytes:` figures are the chosen
wheels summed from PyPI: 3.08 GB cuda-linux, 0.19 GB mlx-darwin.

## Weights

Pulled through `weights.pull` into `models/<id>/<backend>/` with the arm's `files` as allow
patterns, adopting a Hugging Face cache snapshot at the pinned revision when one exists. BiRefNet:
`config.json`, `BiRefNet_config.py`, `birefnet.py`, `model.safetensors` (444 MB). SAM 2.1:
`config.json`, `model.safetensors` (898 MB), `preprocessor_config.json`, `processor_config.json`,
`video_preprocessor_config.json`; not `sam2.1_hiera_large.pt` (the same weights for Meta's package)
nor the yaml. Neither repo is gated.

Licences, read at the pinned revisions (2026-09-29): neither repo has a LICENSE file. BiRefNet's
model card declares `license: mit` and links the MIT text in `ZhengPeng7/BiRefNet` on GitHub; SAM
2.1's card declares `license: apache-2.0`, and `facebookresearch/sam2` carries the Apache 2.0 text.
Each manifest's `licence`, `licence_url` and `commercial_use` say so.

## Memory

Every arm is `memory_basis = "declared"`: nothing has run through Crucible yet. On CUDA the
estimate is also the worker's hard cap (`set_per_process_memory_fraction`), so it is set with
headroom; on the Mac it is what the unified-memory guard admits against.

| arm | estimate | basis |
| --- | --- | --- |
| birefnet cuda | 4.5 GB | the BiRefNet README's table: 3.5 GB inference at 1024x1024 in FP16 (4.8 GB in FP32); 444 MB of float16 weights |
| birefnet mps | 7.0 GB | float32 weights 0.89 GB, the README's 4.8 GB FP32 figure, the CPU fallback's copies and the torch process in one pool |
| sam2.1 cuda | 4.0 GB | 898 MB of float32 weights (224 M parameters); the Hiera-L encoder at 1024x1024 with windowed attention; masks upscaled on the CPU |
| sam2.1 mps | 6.0 GB | the CUDA figure plus the Metal driver's slack |

The input's size does not move the model's peak (both see 1024x1024); it moves the CPU side (the
decoded picture, SAM's upscaled float32 logits, the RGBA cutout), which `max_pixels` bounds at
40 MP. The worker reports `peak_bytes` after the model's pass: `max_memory_reserved` on CUDA, the
Metal driver's allocation on MPS.

## What the first runs through Crucible must measure

After deployment, one job per arm through Crucible (never a bare script) settles:

1. That BiRefNet's remote code loads offline from the pulled folder under transformers 5.17.0
   (`trust_remote_code` from a local path, its relative `BiRefNet_config` import), and on the PC
   in float16 under the 4.5 GB cap.
2. BiRefNet on MPS: that `PYTORCH_ENABLE_MPS_FALLBACK=1` carries `deform_conv2d` to the CPU and
   the mask is right, and how long a 1024 pass takes. If the fallback fails, the Mac arm moves to
   the CPU device (a manifest change: the engine already runs anywhere torch does).
3. That `Sam2Model.from_pretrained` accepts the `sam2_video` checkpoint without error, and that a
   point plus a box together give one sensible mask.
4. `peak_bytes` per arm at a large input (a 24 MP photo), replacing each `declared` estimate with
   a measured one and its note.

## The worker

`worker.py` speaks the stdlib protocol (`workerio.serve`): `load` builds the engine (`birefnet` or
`sam2`) and answers `ready` with the versions; `segment` answers `ready`, a progress event per
stage (`reading`, `segmenting`, `saving`, each with its `fraction`), one `result` and `done`;
`cancel` is an interrupt read on the stdin thread and honoured between stages. Pillow opens the
input without applying EXIF orientation, as the job's header reader (`picture.py`) does, and the
worker refuses a picture whose decoded size differs from the size the job checked the points
against. `cutout_of` puts the mask in the alpha channel, multiplied by the input's own alpha when
it has one; `coverage_of` is the mask's mean. `tests/fake_segment_worker.py` runs the real worker
with the two models replaced by drawn masks, so the tests exercise the protocol, Pillow's reading
and writing, both artifacts, leases and cancel on Windows.
