# Image generation internals

Covers `crucible/jobs/image/` (the `image`, `load-image` and `unload-image` job types and `worker.py`),
`crucible/imagemodels.py`, `crucible/image/*.toml`, `crucible/envs/image/*.txt`, the fifth
resident kind (`KIND_IMAGE`, noun `generator`) and the `image` capability class. The caller's
page is [IMAGE.md](../IMAGE.md). The job-type machinery it plugs into is
[jobs-runtime.md](jobs-runtime.md); install, recipes and weights are
[config-envs-weights.md](config-envs-weights.md).

Owen, 2026-09-28: *"i just added a qwen image generation model to my mac. id like to add it to
crucible so we can use it on any system and crucible can protect from OOM problems. that means
we'll have to set it up on the pc as well."* And: *"it only runs one set of weights at a time. on
the mac it doesnt exceed 16 gb at once, even at bf16."*

## Shape

- One family, `image` (`[jobs] enable_image`, default off; an older config without the key
  reads as off and a rewriter that does not name it keeps what the file says). Three types:
  `image` (makes the generator resident and reuses it, like `align`), `load-image` (loads it
  and leaves it resident, like `load-voice`; its card effect reuses what it names, so warming
  the model a lease already holds is a no-op, not `409 leased`) and `unload-image`
  (`UnloadJobType`, `generator_not_resident`). `load-model` stays text-only: its card effect
  makes an LLM resident and its manifests are the LLM catalog.
- One env per backend, `image`, built by `crucible install image`, installed on submit like
  every worker env. One manifest, `qwen-image-2.1`, pulled with `crucible models pull` (the
  `models` weights family, like the ASR and align manifests). `catalog_is_complete`: an id
  the directory does not declare is refused, never pulled.
- The worker is a `WorkerSession` (`workerio.serve`, ops `load` and `generate`, interrupt
  `cancel`) held by `Residency` as the resident. Settlement unloads it when the job ends
  unless a lease holds it (jobs-runtime.md section 6). A batch sends `params.lease` on its
  first picture (below).
- One artifact per job, `image.png` (a masked job adds `generated.png`, the picture before the
  paste-back), and `done.image` holding every effective parameter
  (seed included, chosen by the server when the caller sent none), the revision, the engine,
  the per-stage seconds and peak bytes, and the estimate with its basis.

## Backends

| backend | engine | weights | size multiple | image-to-image | inpaint |
| --- | --- | --- | --- | --- | --- |
| `mlx-darwin` | mflux 0.20.0 (`QwenImage21`, MLX) | `Qwen/Qwen-Image-2.1` bf16 | 16 | yes (`image_strength`) | yes (`mask`) |
| `cuda-linux` | diffusers at `5ff8e59f` (`QwenImage21Pipeline`) | the same repo and revision, bf16 | 32 | yes (`image_strength`) | yes (`mask`) |

Windows (`llama-windows`) is never an image backend: the class answers the one `NEEDS_WSL`
sentence there, and `crucible install image` refuses it `needs_wsl`.

- Both arms load the same checkpoint at the same revision. The Mac runs mflux, the library
  Owen's `mflux-generate-qwen-2.1` is, through its Python API; the PC runs diffusers. The
  samplers differ, so a seed reproduces a picture within an arm, not across arms.
- **Sizes.** mflux packs latents per 16-pixel tile, so 1280x720 works on the Mac. diffusers
  checks `vae_scale_factor * 2 = 32` and would otherwise floor the size silently, so the CUDA
  arm refuses a side that is not a multiple of 32 (`image_size_not_supported`) instead of
  handing back a different size.
- **Image-to-image.** mflux starts from the input image at `init_time_step = steps *
  image_strength`, so higher keeps more of the input (Owen's `--image photo.png`). diffusers'
  Qwen-Image 2.1 pipeline takes images only as editing conditions (a different operation with
  no strength), so the CUDA worker builds mflux's start itself: it VAE-encodes the stretched
  input, hands the pipeline the tail of the unshifted sigma schedule from that step (the
  resolution shift and terminal stretch are per-sigma, so the tail shifts to the tail of the
  full schedule), and blends input and seeded noise at the first shifted sigma, read from a
  copy of the scheduler set the same way. Both blocks say `image_to_image = true`.
- **Inpainting.** Below, in its own section. Both blocks say `inpaint = true`, a required key
  like `image_to_image`; an arm without it refuses `mask` with `inpaint_unsupported`.
- **Guidance.** Both pipelines sample Qwen-Image 2.1 without guidance (1.0). A negative prompt
  is read only when guidance is above 1.0, and guidance above 1.0 does nothing without a
  negative prompt, so each without the other is refused (`invalid_params`) rather than
  silently ignored.
- **No quantize param.** mflux can quantize the transformer (`-q 8`), but no arm needs it to fit
  and none declares it, so the param does not exist and is refused as unknown.

## Staged placement: why the estimate is below the download

The repo is 33.1 GB: the text encoder (Qwen3-VL-8B, 17,534,339,488 bytes of bf16
safetensors), the transformer (7B, 14,230,284,408 bytes) and the VAE (1,350,989,512 bytes).
A picture needs them one after another, never together: the encoder turns the prompt into
embeddings, the transformer denoises for `steps` steps, the VAE decodes. Both workers hold at
most one of them:

- **mflux.** `generate` builds a fresh `QwenImage21` from the weights directory. Its arrays are
  lazy: a weight is read from the file when the graph that uses it is evaluated. The worker's
  `before_loop` callback drops the text encoder and only then evaluates the prompt embeddings,
  so each encoder layer's weights are read, used and freed in turn (the encoding stage peaked
  at 2.4 GB for a 17.5 GB encoder); `after_loop` drops the transformer before the VAE decodes.
  `gc` and `mx.clear_cache()` run between stages, and `mx.set_cache_limit` is the manifest's
  `mlx_cache_limit_bytes` (4 GB, Owen's `--mlx-cache-limit-gb 4`), so freed buffers do not pile
  up across steps. Between jobs the worker holds the process and nothing else (0.2 to 0.8 GB,
  once 2.8 GB right after a 1280x720).
- **diffusers.** The pipeline is built with `text_encoder=None, transformer=None, vae=None`
  (processor and scheduler only). Each stage loads its component straight onto the card with
  `device_map="cuda"`, runs, and is released (`gc`, `torch.cuda.empty_cache()`). The embeddings
  and then the latents are what cross between stages. `model_cpu_offload` was rejected: it
  keeps every component in host RAM, and owens-pc's WSL VM is capped at 13 GB
  (`.wslconfig`), so 33 GB of offloaded weights would page to the swap disk.

So the manifest's `memory_bytes_estimate` is the **peak of the largest stage at the largest
size** the block admits (`max_pixels`), not the sum, and it is what capability fits, what the
accelerator guard admits and, on CUDA, the per-process cap
(`set_per_process_memory_fraction`), so a stage that outgrows its number fails inside the
worker with a CUDA OOM instead of spilling into Windows shared memory. A request above
`max_pixels` or `max_side` is refused before anything loads (`image_too_large`).

The price is a read of each component per picture. What stays "resident" between the
pictures of a leased batch is the worker process (its imports and Metal or CUDA context:
15.4 s cold, 1.8 s warm on the Mac); the weights themselves are read again for each picture,
from the page cache when the machine has room for it. On the Mac that read is inside the
measured times below (the encoder stage took 2.9 to 3.9 s including it).

## Measured

On the Mac Studio (M1 Ultra, 64 GB), 2026-09-28, mflux 0.20.0 / MLX 0.32.2 in Crucible's
3.11 interpreter, run through `worker.py` against the weights already in
`~/.cache/huggingface`. Peak bytes are `mx.get_peak_memory()` per stage (the worker's
`stage_peak_bytes`); footprint is `top`'s MEM for the worker process (whole GiB). Owen's
parsec batch (`mflux-generate-qwen-2.1` at 384x512) was running on the same GPU the whole
time and never paused, so every **time** below is contended; the memory is per process and is
not.

| picture | steps | encoding | transformer stage | VAE decode | footprint | wall (contended) |
| --- | --- | --- | --- | --- | --- | --- |
| 512x512 | 8 | 0.01 GB (lazy) | 14.86 GB | 6.21 GB | 16 GiB | 33.5 to 41.3 s, 2.9 s/step |
| 512x512 from an input image, strength 0.6 | 2 of 8 run | 2.42 GB | 15.05 GB | 6.54 GB | 16 GiB | 11.7 s |
| 1280x720 | 8 | 0.01 GB (lazy) | 15.35 GB | 14.56 GB | 16 GiB | 84.0 s, 9.6 s/step |
| 1024x1024 | 2 | 2.41 GB | 15.50 GB | **16.44 GB** | 15 GiB | 28.8 s |
| 1024x1024 | 4 | 0.01 GB (lazy) | 15.50 GB | 16.44 GB | 17 GiB | 49.1 s, 8.3 s/step |

- **Owen's "never more than 16 GB" holds.** The largest stage at the largest admitted size
  (1,048,576 pixels) is the **VAE decode**, 16,441,695,780 bytes; the transformer stage is
  15.5 GB at any size up to that. The manifest's 17,200,000,000 is that peak plus the worker
  process's own 0.2 to 0.8 GB. Rows marked "lazy" ran before the worker evaluated the
  embeddings in `before_loop`: the encoder then ran inside step 1, which is why those rows'
  first step took 7 to 12 s.
- The decode grows with the picture (6.2 GB at 0.26 MP, 14.6 at 0.92, 16.4 at 1.05), which is
  why `max_pixels` is where the estimate was measured and a larger picture is refused rather
  than tried.
- A 40-step 1280x720 was not run; memory does not depend on the step count, and at the
  contended 9.6 s per step it is about 6.5 minutes (Owen's uncontended figure: about 5).
- Starting the worker (imports, Metal) took 15.4 s cold and 1.8 s warm; building the model
  and running the encoder is inside each picture's time.
- A cancel sent after step 3 of 8 stopped after step 4 (the step in flight finishes), the
  worker answered `cancelled` and `done`, and the next two pictures ran in the same process.

## Why the CUDA arm is bf16 and staged (and what was rejected)

The 3090 Ti has 24,564 MiB, less owens-pc's stated 3 GiB desktop allowance: 22.5 GB for a
model. Candidates, in the order considered:

1. **bf16, staged, straight to the card** (chosen). The largest stage is the text encoder:
   17.5 GB of weights plus activations for a prompt and the CUDA context, declared at
   19,000,000,000 bytes, 3.5 GB under the budget. Same weights and revision as the Mac, no
   quality question.
2. **bf16 with `enable_model_cpu_offload()`.** The same card peak, but every component lives in
   host RAM between uses: 33 GB against a 13 GB WSL VM. Rejected.
3. **Everything resident on the card.** 33 GB. Does not fit.
4. **A quantized transformer (bitsandbytes nf4/int8, torchao fp8, a GGUF through diffusers) so
   transformer and encoder could both stay resident.** nf4 encoder (~5.5 GB) plus bf16
   transformer (14.2 GB) plus activations and VAE is ~22 GB, at the budget's edge; fp8 needs
   sm_89 (the 3090 Ti is sm_86); each is a quality change nobody has looked at. Rejected while
   bf16 fits (Owen, 2026-09-28: drop quantization unless measurement shows bf16 does not fit).

**Owed on the PC** (the card was busy with the training run on 2026-09-28, so nothing ran
there): `crucible install image` on owens-pc (the recipe is a resolver output, below, not yet
a freeze of a working env), `crucible models pull qwen-image-2.1`, then one 512x512 at 8
steps and one 1280x704 at 40 steps with nothing else on the card, reading
`done.image.stage_peak_bytes` and nvidia-smi's peak, then one 1024x1024 at 2 steps for the
largest admitted size. Replace the block's `memory_bytes_estimate` with the largest stage peak
plus the CUDA context and set `memory_basis = "measured"`. Watch the VAE decode: on the Mac it
was the largest stage at 1024x1024 (about 15 GB of activations on a 1.35 GB VAE). If the
PC's decode passes the 19 GB cap it fails inside the worker as a CUDA OOM (never on the
desktop), and the fix is a lower `max_pixels` on the cuda block or VAE tiling, measured.

## Recipes

- **mlx-darwin** is a `pip freeze` of `mflux==0.20.0` installed into Crucible's own 3.11
  interpreter on the Mac Studio (2026-09-28): 55 lines, `mlx==0.32.2`, `torch==2.14.0` (mflux
  loads safetensors through it), `transformers==5.17.0`. mflux 0.20.0 is the release Owen's
  `~/.local/bin/mflux-generate-qwen-2.1` runs and the first with Qwen-Image 2.1.
- **cuda-linux** is a resolution, not yet a freeze: `pip install --dry-run --report` for
  cp311 manylinux x86_64 of `torch==2.14.0`, `transformers==5.17.0`, `accelerate`,
  `safetensors`, `pillow` and diffusers at commit `5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2`
  (no released diffusers has `QwenImage21Pipeline`: 0.40.0 predates it, main gained it in
  #14804 on 2026-09-18). The resolver ran on Windows, whose markers drop torch's Linux-only
  wheels, so the `nvidia-*`, `cuda-*` and `triton` lines are the align recipe's, which pins the
  same `torch==2.14.0` and runs on owens-pc. `torchvision==0.29.0` (the torch 2.14.0 build) is
  there because transformers' Qwen3-VL processor needs it. Replace this file with a `pip
  freeze` after the first install on the PC.
- `# archive-bytes:` on both is the sum of the pinned wheels' sizes on PyPI (2026-09-28):
  0.28 GB and 3.07 GB.

## Weights and the Hugging Face cache

`weights.pull` now looks in the Hugging Face cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`,
else `~/.cache/huggingface/hub`) for a snapshot at exactly the pinned revision, and
hard-links its large files into the Crucible store before the download: only files whose
cache blob is content-addressed (a 64-hex name, the LFS and xet blobs). The hub download then
checks each linked file against the pinned revision (an LFS file with no local metadata is
hashed against its sha256) and fetches only what is missing or different. Small git-stored
files are left to the hub, which copies them out of the cache itself: linking them made that
copy fail with `SameFileError` (found on the Mac, 2026-09-28). A link that cannot be made
(another filesystem, or a cache of plain copies, as on Windows) is simply not made. Removing
the Crucible copy unlinks it and leaves the cache alone, and the reverse. The stamp records
`linked_bytes`.

Measured on the Mac, 2026-09-28, into a scratch home with the branch's `weights.pull`: Owen's
`Qwen/Qwen-Image-2.1` at `790c9263` (mflux's download: every component and the processor)
linked 33,127,036,062 bytes, the pull finished in 6.2 s, and only the ~8 MB mflux had not
fetched (`model_index.json`, `scheduler/`, the README, the licence, a 3.3 MB picture) came
from the network. No second copy on disk. The installed Crucible was not touched.

## The lease on an image job

Owen, 2026-09-28: *"does crucible have a way to allow a user to retain a lease and keep a model
loaded if we're generating multiple images? ... if the lease expires or something closes or
stops, it can release the lease and the model from memory."*

Before, a batch had to lease with `POST /v1/models/{id}/lease` after the first picture made the
model resident and before settlement took it off again: a race the client could lose.
`image` and `load-image` now take `params.lease` (`jobs/leaseonload.py`'s `LeaseOnLoad`, the
same block `load-model` and `load-voice` take):

- **Admission.** `preflight` runs `require_lease_request(lease, act="image")`: the act must be
  a known capability class (`unknown_act`) and exactly `image` (`lease_act_mismatch`), the ttl
  30 to 3600 (`invalid_ttl`). A bad block is refused at submit, before anything loads.
- **The moment.** `run` gets the resident worker (`ResidentWorker._session`: the loaded one,
  or a load), then `hold_for_load` opens the lease, then the picture is made. Settlement runs
  only when the job ends, and by then the lease holds the card, so a long first picture cannot
  race it.
- **No second lease.** `hold_for_load` first looks at the open lease: if it holds the same kind
  and subject for the same client, it is heartbeated and its id returned (`opened=False`)
  instead of `Leases.open` refusing `409 leased`. `open_lease_for_load` (load-model,
  load-voice) goes through the same rule. A lease held by someone else is still `leased`.
- **Failure.** A picture that fails or is cancelled after its job *opened* the lease releases
  it (`let_go_of`): the caller never received the id, so nothing would heartbeat or release
  it, and settlement then clears the card as for any failed job. A renewed lease is left
  alone.
- **The end.** `done` carries `lease_id` (null without `lease`). Heartbeat and release are the
  ordinary lease routes; an unrenewed lease lapses and the lane's idle pass
  (`settle_for_lapsed_lease`) unloads the model, a released one is cleared before
  `DELETE /v1/leases/{id}` answers.

## The prompt-embedding cache

Measured on owens-pc (3090 Ti, bf16, staged): every picture spent 17 to 25 s in `encoding`,
because the diffusers arm loads the 17.5 GB text encoder, encodes, frees it and only then
loads the transformer. The worker process survives between leased pictures; the encoder did
not. The embeddings it produces are small, so `worker.py` keeps them:

- **Key** (`Job.prompt_key`): prompt, negative prompt, whether guidance is above 1.0 (only
  then is the negative encoded), the manifest revision and the backend, the last two sent by
  the controller in every `generate` request. Seed, size, steps and guidance's value are not
  in it: none of them reaches the encoder.
- **Value.** diffusers: the four tensors `encode_prompt` returned (`prompt_embeds`, its mask,
  and the negative pair when used), copied to host RAM at the end of the encoding stage and
  moved back to the card on a hit. mflux: the `(embeds, mask)` pairs mflux itself keeps in
  `model.prompt_cache`, keyed by prompt text; on a hit they are put into the fresh model's
  `prompt_cache` and `text_encoder` is dropped before `generate_image`, so mflux never calls
  the encoder and, its arrays being lazy, never reads the encoder's weights. mflux exposes no
  separate encode call, but its prompt cache is that seam.
- **Size.** One entry is `tokens x 4096 x 2` bytes per prompt (bf16; 4096 is the text
  encoder's `hidden_size`) plus the mask: a 40-token prompt is about 0.33 MB, a 300-token one
  about 2.5 MB, twice that with a negative prompt. The worker reports the total as
  `prompt_cache_bytes`.
- **Bounds.** `PromptCache`: least recently used first out, at most 32 entries
  (`PROMPT_CACHE_ENTRIES`) and 256 MiB (`PROMPT_CACHE_BYTES`); an entry larger than the byte
  cap is not kept. It lives in the worker process, is emptied on `load`, and goes when the
  worker stops (unload, settlement, lease lapse, crash).
- **Report.** The worker's result says `prompt_cache: "hit" | "miss"` and the controller copies
  it into `done.image`. A hit still enters the `encoding` stage (to move the tensors back), so
  `stage_seconds.encoding` is present and near 0.
- **Not measured on hardware.** The saving (the whole encoder stage on a hit) is shown by the
  fake worker's stage report only; a PC run of two same-prompt pictures is owed, and must go
  through the installed Crucible, never a scratch worker beside it.

## Inpainting and outpainting (`mask`)

Owen, 2026-09-29: Crucible supplies the tools and the apps build the features. ContentStudio
lets a user brush or lasso a region, or takes a mask from a `segment` job, and regenerates only
that region from a prompt; outpainting is the app padding the canvas and masking the border.
Cropping, running image-to-image and compositing in the app leaves seams, so the image job
takes the mask itself.

**The request.** `params.mask` names the input that carries the mask. The job then carries
exactly two inputs, the mask and the image (the other one, whatever its name). A one-input
`image_strength` request, which apps already send, is unchanged. `image_strength` is optional
with a mask. Without it the job starts at step 0 (sigma 1, pure noise), which regenerates the
region fully: that is the usual inpainting and the default. With it the job starts at mflux's
`init_time_step` as image-to-image does, so a region can be restyled while it keeps its layout.
`mask_blur` (0 to 256, default 8 with a mask) is refused without `mask`.

**Refusals, before the model loads where possible.** `input_pictures` in `jobs/image/__init__.py`
checks at run time, before `_worker` loads anything: the named input is there
(`invalid_inputs`), there are exactly two inputs, both are PNG, JPEG or WebP, and the two
headers give the same size (`mask_size_mismatch`, from `inpaint.picture_size`, a plain-Python
header reader, since the controller has no Pillow). The worker checks the size again after
applying EXIF orientation, and refuses a mask that selects nothing (`mask_empty`). It cannot
know that without decoding the pixels. A refusal from the worker is a `result` carrying
`refused: {code, message}` followed by `done`, not an exception, so the loaded model survives.
The controller raises it as a `JobError` with that code.

**Shared code, `jobs/image/inpaint.py`** (numpy and Pillow, loaded beside the worker with
`workerio.load_sibling`):

- `region_of`: the mask as grey, stretched to width x height (bilinear), 128 and up is the
  region.
- `feathered`: the paste-back weight, `clip(2 * gaussian(region, blur / 2) - 1, 0, 1) * region`.
  It is 1 deep inside, falls to 0 at the edge over about `mask_blur` pixels, and is exactly 0
  outside, so a kept pixel is never touched (the fade is inward, unlike diffusers'
  `blur_factor`, which is centred on the edge and would change kept pixels).
- `latent_grid`: the region on the latent grid by max over each 16x16 tile (2.1's VAE scale
  factor is 16 and its latents are unpatched). This is max, not diffusers' nearest
  `interpolate`, so a thin stroke still reaches its latent. `packed` flattens it row-major to
  `(1, h*w, 1)`, the layout both engines pack 2.1 latents in (diffusers' `_pack_latents` is
  `view(B, C, H*W).transpose(1, 2)`; mflux's `Qwen21LatentCreator.pack_latents` transposes to
  channels-last and reshapes). A test checks both.
- `blend_step(latents, clean, noise, mask, sigma)`:
  `mask * latents + (1 - mask) * ((1 - sigma) * clean + sigma * noise)`. It is plain arithmetic,
  so it works on torch tensors, MLX arrays and numpy.
- `paste_back`: `feather * generated + (1 - feather) * original`, with the original
  EXIF-oriented, RGB and stretched to the job's size. It runs in the worker's `_run` for both
  engines, inside the `saving` stage.

**CUDA (diffusers).** The loop is `QwenImageInpaintPipeline`'s (`pipelines/qwenimage/
pipeline_qwenimage_inpaint.py` at `5ff8e59f`): after `scheduler.step`, the input latents are
noised with `scheduler.scale_noise` to the next timestep and blended in with
`(1 - init_mask) * init_latents_proper + init_mask * latents`. Under `set_begin_index(0)`,
`scale_noise` reads `sigmas[step_index]`, and after step `i` that is `sigmas[i + 1]`, the
appended 0 after the last step. `QwenImage21Pipeline` is not forked: `callback_on_step_end`
receives `latents` after the scheduler step and the pipeline reads them back
(`callback_outputs.pop("latents", latents)`), so `_DiffusersRepaint` reads
`pipeline.scheduler.sigmas[index + 1]` and returns the blend. The clean latents and the noise
are the ones `_encode_start_image` and `_start_schedule` already made for image-to-image
(`_start_schedule` now also returns the noise and starts at `Job.start_step`, 0 without a
strength). The start image is now EXIF-oriented before encoding, as mflux does.

**Mac (mflux 0.20.0).** mflux has no inpainting for Qwen-Image 2.1. Its fill (`flux/variants/
fill`) is the separate FLUX.1 Fill model, and `Config.masked_image_path` is read only by that
model. Its `QwenImage21.generate_image` loop calls `ctx.in_loop(t, latents)` after each
`scheduler.step` and reads nothing back.

The first version (1.0.64) wrote the blend into the loop's array from that callback
(`latents[...] = blended`), relying on MLX item assignment updating the array in place. On the
Mac Studio (2026-09-29) an inpainted box came back as a separate little scene with a hard seam
on every side, and both outpaint strips were unrelated landscapes. The paste-back still kept
every pixel outside the mask exact. That was taken for the write not reaching the loop.

So a masked job no longer goes through `generate_image`. `_mflux_generate` is that method's
body for 0.20.0, step for step: the same `Config` (linear scheduler), prompt encoding through
`Qwen21PromptEncoder` and the model's prompt cache, true CFG, the `callbacks.start` context
with `before_loop`, `in_loop` and `after_loop` (so the worker's stage, memory and progress
callbacks run as before), the unpack, and `VAEUtil.decode`. The loop, `_mflux_denoise`, is
mflux's own: `scale_model_input`, the transformer, `scheduler.step`. The one difference is that
the blend's return value is what the loop carries into the next step. The start latents are
`LatentCreator.create_for_txt2img_or_img2img`'s, built from the clean latents and the noise
the blend already holds (built the same way): the noise at `init_time_step` 0, else
`add_noise_by_interpolation` at `sigmas[init_time_step]`. So the input is VAE-encoded once
even with a strength. `ImageUtil.to_pil` makes the picture (`to_image`'s metadata is not
used). Unmasked jobs still call mflux's `generate_image`, the path measured above.
`test_the_mac_loop_feeds_each_blend_to_the_next_step` runs both functions against stubs that
record what each step receives.

The sigma is `config.scheduler.sigmas[t + 1]` (`LinearScheduler`'s shifted schedule, with a 0
appended). The clean latents are built as mflux's own image-to-image builds them
(`LatentCreator.encode_image`, then `Qwen21LatentCreator.pack_latents`) during the encoding
stage. The noise is `Qwen21LatentCreator.create_noise(seed, ...)`, the array mflux's txt2img
starts from.

**1.0.66 on the Mac was bit-identical to 1.0.64 (2026-09-30), and the Mac fault is still
open.** The rewritten loop produced the same pixels, so the in-place write had reached the
loop after all, and the fault is somewhere else. The coordinator's candidates were each
checked against mflux 0.20.0's own code, not against a reading of it.
`tests/mflux_pinned_check.py` runs the vendored mflux files (`tests/fixtures/mflux_0_20_0`:
`config.py`, `linear_scheduler.py`, `qwen21_latent_creator.py`, `latent_creator.py`, MIT)
on a numpy stand-in for `mlx.core`, with a stub VAE whose latents encode their own channel,
row and column. The test is a non-square 160x96 picture with an off-centre box.

- **Token order.** `_repaint`'s clean latents put the encode's `(c, y, x)` at token
  `y * w + x`, channel `c`. The mask token at `y * w + x` is the grid's `(y, x)`.
  `Qwen21LatentCreator.unpack_latents` returns the encode exactly. 2.1 latents are unpatched:
  `pack_latents` reshapes to `(1, 64, H/16, W/16)`, transposes to channels-last and flattens,
  and `Qwen21VAE` has `spatial_scale = 16` and `latent_channels = 64`.
- **Sigma.** At every step the kept tokens the transformer receives equal
  `(1 - s) * clean + s * noise`, where `s` is `config.scheduler.sigmas[t]`, the sigma
  `Qwen21Transformer._compute_timestep` gives it for that step. That holds from `t = 0`
  (pure noise) and from `init_time_step = 4` with strength 0.4. The kept tokens decode to the
  encode exactly after the last step.
- **Noise and start.** The blend's noise is `create_noise(seed, ...)`, and the first step's
  input equals mflux's own `create_for_txt2img_or_img2img`, with and without a strength.
- **Scale.** `Qwen21VAE.encode` returns `(x - LATENTS_MEAN) / LATENTS_STD`, and `decode`
  applies `x * LATENTS_STD + LATENTS_MEAN` first. The loop's latents are the normalised ones,
  as mflux's own image-to-image relies on.

The mask is now also packed by mflux's own `pack_latents` (the grid spread over the 64
channels like the clean latents), so its token order is the clean latents' by construction,
whatever mflux's packing is. What the vendored files cannot see is the real encoder and
transformer. So a masked job now reports what the next run needs to place the fault:
`mask_blend_steps` (the blend ran on every step) and `mask_outside_drift` (how far the
decoded picture differs from the input outside the mask before the paste-back). It also
publishes `generated.png`, the picture before the paste-back. The last step puts the clean
latents back outside the mask, so:

- **Drift of a few units, seams still there:** the kept latents decode to the input, and the
  transformer draws the region without regard to them. The fault is in the model port or the
  schedule, not in the mask code.
- **Drift of tens with `mask_blend_steps` equal to the steps:** the blend held, but the kept
  latents are not the input's. The fault is in the encode (`LatentCreator.encode_image` on the
  real VAE) as the worker calls it.
- **`mask_blend_steps` missing or 0:** the blend never ran in that process. Check the worker
  log's first line, which names the `worker.py` the server started.

**A seam on one side of an outpaint (PC, 2026-09-29).** A 768x768 photo centred on 1024x768
came back with the left strip continuing the scene and a hard vertical seam on the right,
where the generated trees did not line up with the photo's. The mask handling is symmetric.
The mask was white over x < 140 and x >= 884. The latent grid regenerates tiles 0 to 8
(x < 144) and 55 to 63 (x >= 880): 4 pixels past the mask on each side, both pasted back from
the original. The inward feather is the same on both edges.
`test_the_mask_is_handled_the_same_on_the_left_and_the_right` checks that region, feather
and grid are mirror images.

Stopping the blend over the last steps would not help. The last step already blends at sigma
0, so it puts back the un-noised input, which is what diffusers' inpaint pipeline ends with.
Where the trees go is decided in the first, high-noise steps, and freeing the kept side at the
end would only let the model change pixels that the paste-back then replaces. What is left is
the model: a model that was not trained for inpainting sometimes continues a busy edge (trees)
badly on one side and well on the other, differently for each seed. The remedies are the
caller's: another seed, a wider `mask_blur` (24 to 32) with the white reaching further into
the photo (24 to 32 pixels), and narrower strips.

**Owed on hardware.** Owed on each machine, through the installed Crucible: the inpaint (a
centre box, no strength) and the outpaint (a 1024x768 canvas from a centred 768x768), reading
`mask_blend_steps`, `mask_outside_drift` and `generated.png` on both. The PC's drift is the
healthy baseline. Then one masked job at 1,048,576 pixels, reading
`stage_peak_bytes`: the extra VAE encode runs in the encoding stage on the Mac (in the
transformer stage's setup on the PC, before the transformer loads), and neither was measured.

## Cancel between steps

A cancel used to mean SIGTERM to the worker, which for a resident generator would throw away
the loaded process. `WorkerSession.send(..., cancel_request=...)` now writes the request's
`cancel` line to the worker's stdin instead; `workerio.serve(label, ops, interrupts)` reads
stdin on a thread and runs an interrupt as soon as its line arrives, while `generate` is still
running. The worker checks the flag between steps and between stages, answers
`progress {stage: "cancelled"}` and `done`, and stays up. A worker that has not stopped
`CANCEL_GRACE_SECONDS` (120 s) after being asked is stopped the old way (SIGTERM, never
SIGKILL). The cancel carries the request's id, so a cancel that arrives after its picture
finished cannot stop the next one.

The same change fixed a latent bug for every session worker: `_Reader.lines` yielded for as
long as lines kept arriving, so a worker reporting progress more often than every 0.5 s was
never checked for a cancel. It now returns to the caller at least every `POLL_SECONDS`.
