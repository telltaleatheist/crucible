# Audio generation internals

Covers `crucible/jobs/audio/` (the `audio`, `load-audio` and `unload-audio` job types,
`params.py`, and the workers `audiocore.py`, `stable_audio_worker.py`, `yue2_worker.py`),
`crucible/audiomodels.py`, `crucible/audioweights.py`, `crucible/audio/*.toml`,
`crucible/envs/audio/*.txt`, the sixth resident kind (`KIND_AUDIO`, noun `audio generator`,
unload refusal `audio_generator_not_resident`) and the capability classes `sfx`, `music` and
`song`. The caller's page is [AUDIO.md](../AUDIO.md). The machinery it plugs into is
[jobs-runtime.md](jobs-runtime.md) and [image.md](image.md), whose shape this copies; install,
recipes and weights are [config-envs-weights.md](config-envs-weights.md).

Owen, 2026-09-28: *"lets get the sound effects and the music one, and get yue2 as well. hook
them up to crucible. we're going to need them on the mac crucible so get those as well and wire
it up so we can use them."*

## Shape

- One family, `audio` (`[jobs] enable_audio`, default off; absent reads off, a rewriter that does
  not name it keeps it). Three types: `audio` (makes the generator resident and reuses it),
  `load-audio` (loads and leaves it resident; `LEAVES_IT_RESIDENT`) and `unload-audio`.
- One manifest per model in `crucible/audio/`, loaded by `audiomodels.py`. `[model].kind` is
  `sfx`, `music` or `song` and decides two things: which capability class lists the model,
  and which text param it reads (`prompt`, or `tags` plus `lyrics`).
  `catalog_is_complete`: an undeclared id is refused.
- Each `[backends.<kind>]` arm declares the engine, the pinned repo and revision, `gated`, dtype,
  memory estimate with basis and note, the files to pull, sample rate, channels,
  `max_duration_s`, `takes` (the optional params it reads, from `duration_s`, `steps`, `cfg`,
  `negative_prompt`) with their defaults and ceilings, `not_taken` (the reason a param is refused,
  quoted in the refusal), and `companions` (other repos the arm needs, below).
- The job validates against the arm before anything loads (`params.py`,
  `refuse_what_the_model_cannot_take`), settles defaults from the arm (`settle`), and sends one
  `generate` request. `done.audio` carries the effective params, the revision, the per-stage
  seconds and peaks, the versions the worker reported, and the estimate with its basis.

## Backends and engines

| model | cuda-linux | mlx-darwin |
| --- | --- | --- |
| `stable-audio-3-small-sfx` | `stable-audio-3` package, CUDA, float16 | the same package on Metal (MPS), float32 |
| `stable-audio-3-medium` | `stable-audio-3` package + flash-attn 2.8.3, CUDA, float16 | the same package on MPS, float32, SDPA fallback |
| `yue2-3b` | `yue2-infer`, CUDA, bfloat16 | none |

- **Why the `stable-audio-3` package.** It is Stability's official inference library for 3.0
  (the model cards and README use it). diffusers 0.40.0 added `StableAudio3Pipeline`, but the
  official repos carry no diffusers layout (`model_index.json`) at the pinned revisions, and
  stable-audio-tools is the research library. The package is not on PyPI: the recipe pins the
  GitHub commit `779434a9`.
- **Loading from Crucible's own copy.** `StableAudioModel.from_pretrained` only resolves names
  through the Hugging Face cache, so the worker builds the model the way it does internally
  (`load_diffusion_cond` + `StableAudioModel`) from `model_config.json` in the pulled directory,
  after rewriting the T5Gemma conditioner's `repo_id`/`subfolder` to a `model_path` inside it
  (`localise_text_encoder`). The worker runs with `HF_HUB_OFFLINE=1`. The config is behind the
  licence gate, so the exact keys it names were not read before this shipped; the first real
  load through Crucible confirms it (below).
- **Medium on CUDA wants flash-attn.** Stability's README: Medium requires Flash Attention 2
  (its SAME-L autoencoder uses sliding-window attention), and "static glitch" output means a
  broken flash-attn. The cuda recipe installs the official Dao-AILab wheel
  `flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311`, pinned by its sha256.
- **The Mac runs the same package on MPS.** The package picks MPS when there is no CUDA and then
  forces float32. Small SFX is listed by Stability as a CPU-capable model. Medium is listed as a
  CUDA model; without flash-attn the package falls back to its own chunked-halo SDPA for the
  sliding window. That arm is the package's no-CUDA path, not a configuration Stability
  benchmarks. Stability's fast Mac path (`optimized/mlx` in their repo, weights in the
  ungated `stabilityai/stable-audio-3-optimized`) is a folder of scripts with no package
  metadata, so pip cannot install it; adopting it means vendoring code, left for a later pass.
- **YuE2 on mlx-darwin runs the official `yue2-infer` on torch 2.14.0, device `mps`.** Its pinned
  torch 2.10 corrupts bfloat16 causal SDPA prefill on some Apple chips (issue #176, fix PR #181
  unmerged; fixed in torch 2.13). `envs/audio/yue2-mlx-darwin.txt` pins 2.14.0 and marks
  `yue2-infer` `# crucible: no-deps` (jobenv installs it apart, after the rest, so its own
  torch pin cannot pull 2.10 back; every dependency is still pinned and drift-checked).
  `yue2_worker.mps_causal_is_sound` re-proves the kernel on each Mac load (relative error of
  `is_causal` vs an explicit tril at lengths 17/128/705; sound ~0.003, leaking 0.3-0.6; limit
  0.02) and refuses otherwise. Measured 2026-10-03: the issue's repro is clean on the M1 Ultra
  under both torch 2.10 and 2.14. The Mac memory figure (13 GB) is declared from the issue's
  M4 Pro report until measured here. Community MLX ports are not used.
- Windows (`llama-windows`) is never an audio backend (`WSL_ONLY_JOB_TYPES`).

## Envs

Stable Audio pins `torch==2.7.1` and `transformers>=5.8`; `yue2-infer` pins `torch==2.10.0`,
`transformers==4.57.6` and `huggingface-hub==0.36.2`. They cannot share a venv, so audio has one
env per engine, the way tts has one per narrator engine: `envs/audio-stable-audio-3` and (PC
only) `envs/audio-yue2`, recipes `crucible/envs/audio/<engine>-<backend>.txt`
(`jobenv.audio_env`, `jobenv.audio_envs`). `Env("audio", worker=False)` keeps them out of the
one-env-per-type worker loops; `crucible install audio` builds every engine env of the backend
in turn (a failure keeps the ones already built and says a rerun builds only what is missing),
install-on-submit maps `envs/audio-*` back to the `audio` installer and sums the recipes'
archive sizes, `tasks.validate.env_installed("audio")` is true only when every engine env is,
and `crucible doctor` reports `audio_envs` per engine.

Each recipe is a full freeze: pip's own resolution for Python 3.11 on the target platform
(2026-09-29), with the Linux-only CUDA wheels torch pulls in added by evaluating its markers
for that platform. The `# archive-bytes:` figures are those wheels summed from PyPI (plus the
flash-attn wheel from GitHub, 256 MB): stable-audio-3 cuda 3.30 GB, yue2 cuda 4.17 GB,
stable-audio-3 mlx 0.11 GB.

A direct reference is pinned by a commit (`@<40 hex>`) or, for a wheel URL, by a
`#sha256=<64 hex>` fragment; drift is read back from pip's `direct_url.json`
(`vcs_info.commit_id`, or `archive_info.hashes.sha256`).

## Weights

- Pulled through `weights.pull` into `models/<id>/<backend>/` with the manifest's `files` as
  allow patterns, adopting a Hugging Face cache snapshot at the pinned revision (hard links, no
  second copy) when one exists. The Mac has none of these today.
- **Companions.** YuE2 needs a second repo, `m-a-p/YuE2-Vae`, its decoder. An arm's
  `companions` are pulled with `weights.pull_files` (every file verified against its sha256)
  into `models/<id>/<backend>/<name>/`, with their own stamp. `audioweights.installed` is true
  only when the main snapshot and every companion are; `audioweights.pull` fetches only what is
  missing; the load request passes the directories as `parts`.
- **Gated repos.** `stabilityai/stable-audio-3-small-sfx` and `-medium` are gated (`auto`:
  acceptance is instant). `gated = true` in the arm makes a pull without an HF token refuse
  before downloading anything, and a job refuse `409 model_gated` (not `model_not_installed`, so
  install-on-submit does not start a pull that cannot succeed); both say the page to accept, the
  token page, and the command to rerun (`weights.gated_message`, also used when the hub itself
  answers `GatedRepoError`). Crucible never works around a gate.

## Memory

Every arm is `memory_basis = "declared"`: nothing has run through Crucible yet.

| arm | estimate | basis |
| --- | --- | --- |
| small-sfx cuda | 3.5 GB | 3.45 GB float32 files cast to float16 (1.73 GB); Stability's README: 2.40 GB peak at 120 s on an H200 |
| medium cuda | 8.0 GB | 10.4 GB float32 files cast to float16 (5.2 GB); Stability's README: 6.52 GB peak at 380 s with flash-attn, unchunked decode |
| small-sfx mps | 6.0 GB | float32 weights 3.45 GB plus float32 activations, one unified pool |
| medium mps | 16.0 GB | float32 weights 10.4 GB plus float32 activations without flash-attn |
| yue2 cuda | 16.0 GB | YuE2 README: 11.18 GiB on an RTX 4090 with the full score, 14.08 GiB at the full 24,576-token context |

Medium at 8 GB fits the PC's 24 GB card with its 3 GB desktop allowance, so the PC runs Medium,
not Small. YuE2 caps its own process (`memory_budget_gib`, the estimate plus the 2 GiB reserve
it subtracts), and on CUDA every other worker gets `set_per_process_memory_fraction` from the
estimate. The worker reports peaks per stage: `max_memory_reserved` on CUDA, the Metal driver's
allocation on MPS.

## What the first runs through Crucible must measure

After deployment, one job per arm through Crucible (never a bare script) settles:

1. The Stable Audio `model_config.json` keys: that `localise_text_encoder` points T5Gemma at the
   pulled folder and the load succeeds offline.
2. `peak_bytes` per arm at the longest duration (120 s sfx, 380 s music, a long song), replacing
   each `declared` estimate with a measured one and its note.
3. Medium on MPS: that the SDPA fallback sounds right (Stability's "static glitch" symptom) and
   how long 380 s takes; if it does not, the Mac music arm moves to Stability's MLX path.
4. The flash-attn wheel's ABI against torch 2.7.1 on the PC. `crucible install audio` smoke-imports
   `stable_audio_3, flash_attn` on cuda-linux, and the worker refuses to load on CUDA when the
   package's own `flash_attn_func` came out `None` (it swallows the ImportError and would
   otherwise produce static); the first Medium job proves the kernels run.
5. YuE2's time per song on the 3090 Ti, and whether `synthesizing` fits beside the desktop.

## The workers

Both engines speak the stdlib protocol (`workerio.serve`): `load` builds the engine and answers
`ready` with the versions; `generate` answers `ready`, progress events, one `result` and `done`;
`cancel` is an interrupt read on the stdin thread. `audiocore.py` holds what the two share:
`Job` (the request, every key required, null when unset), `Progress` (stage spans to a
`fraction`, cancel checked at every step), `Throttled` (a progress event every 64 tokens),
`ArrayAudio` (soundfile writes FLAC or WAV, 24-bit) and `Worker`. The job checks the file's magic
(`fLaC`, `RIFF....WAVE`) before publishing it and publishes `score.abc` when the worker wrote
one. `tests/fake_audio_worker.py` runs the real `audiocore` with a fake engine that writes a
genuine FLAC (verbatim subframes, CRC-8 and CRC-16), so the tests exercise the protocol, the
artifact and cancel on Windows.
