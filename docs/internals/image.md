# Image generation internals

Covers `crucible/jobs/image/` (the `image` and `unload-image` job types and `worker.py`),
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
  reads as off and a rewriter that does not name it keeps what the file says). Two types:
  `image` (makes the generator resident and reuses it, like `align`) and `unload-image`
  (`UnloadJobType`, `generator_not_resident`).
- One env per backend, `image`, built by `crucible install image`, installed on submit like
  every worker env. One manifest, `qwen-image-2.1`, pulled with `crucible models pull` (the
  `models` weights family, like the ASR and align manifests). `catalog_is_complete`: an id
  the directory does not declare is refused, never pulled.
- The worker is a `WorkerSession` (`workerio.serve`, ops `load` and `generate`, interrupt
  `cancel`) held by `Residency` as the resident. Settlement unloads it when the job ends
  unless a lease holds it (jobs-runtime.md section 6): a batch of pictures leases the model
  after the first, as a book leases its voice.
- One artifact per job, `image.png`, and `done.image` holding every effective parameter
  (seed included, chosen by the server when the caller sent none), the revision, the engine,
  the per-stage seconds and peak bytes, and the estimate with its basis.

## Backends

| backend | engine | weights | size multiple | image-to-image |
| --- | --- | --- | --- | --- |
| `mlx-darwin` | mflux 0.20.0 (`QwenImage21`, MLX) | `Qwen/Qwen-Image-2.1` bf16 | 16 | yes (`image_strength`) |
| `cuda-linux` | diffusers at `5ff8e59f` (`QwenImage21Pipeline`) | the same repo and revision, bf16 | 32 | not yet |

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
  Qwen-Image 2.1 pipeline takes images only as editing conditions, which is a different
  operation with no strength, so the CUDA block says `image_to_image = false` and the job is
  refused by name rather than run as an edit.
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

- **mflux.** `generate` builds a fresh `QwenImage21` from the weights directory (its arrays
  are lazy; a weight is read when first used), and the worker's own callbacks drop the text
  encoder before the first step and the transformer after the last, with `gc` and
  `mx.clear_cache()` between stages. `mx.set_cache_limit` is the manifest's
  `mlx_cache_limit_bytes` (4 GB, Owen's `--mlx-cache-limit-gb 4`), so freed buffers do not pile
  up across steps. Between jobs the worker holds the process and nothing else.
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

The price is a read of each component per picture. On the Mac the files stay in the page
cache between pictures of a batch; the measured numbers below include that read.

## Measured

MEASUREMENTS_PLACEHOLDER

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
`done.image.stage_peak_bytes` and nvidia-smi's peak. Replace the block's
`memory_bytes_estimate` with the largest stage peak plus the CUDA context and set
`memory_basis = "measured"`.

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
hard-links its files into the Crucible store before the download. The hub download then
hashes each linked file against the hub's sha256 and skips it when it matches, so a model
someone already fetched with another tool costs no second copy and no download, and a file
that does not match is fetched as usual. A link that cannot be made (another filesystem) is
simply not made. Removing the Crucible copy unlinks it and leaves the cache alone, and the
reverse. The stamp records `linked_bytes`.

Owen's Mac had `Qwen/Qwen-Image-2.1` at `790c9263` in `~/.cache/huggingface` (31 GB, what
mflux downloads: every component's safetensors and configs, and the processor), so a pull
there links those and downloads only what mflux skipped (`model_index.json`, `scheduler/`,
the README, the licence and a 3.3 MB picture).

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
