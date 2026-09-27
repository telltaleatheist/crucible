# Crucible on the Mac — parity audit, 2026-09-14 (read-only)

Read-only Opus audit of the Mac Studio's server and the repo, run the evening of 2026-09-14
for PHASE15-HOST.md section 4.6. Nothing was written on the Mac. Owen: *"we'll have to make
sure crucible works on mac as well … it would just function out of the box with mlx-audio and
everything we have configured for mac bookforge."*

**Backend as measured:** `crucible 0.6.0 (api 1)`, `backend: mlx-darwin on darwin/arm64`,
`apple Apple M1 Ultra (64.0 GiB) — mlx 0.32.2`, launchd label `com.crucible.serve`, served
from a conda env (`/opt/homebrew/Caskroom/miniconda/base/envs/crucible`), home `~/.crucible`.

## The one structural fact behind all three gaps

`crucible/manifests.py:48-51` — **one engine per backend**, and `manifests.py:857-862` refuses
any manifest naming another:

```python
BACKEND_ENGINES: dict[str, str] = {CUDA_LINUX: "vllm", MLX_DARWIN: "mlx-lm"}
```

Same shape in `crucible/asrmodels.py:53-55` (`{cuda-linux: "faster-whisper"}`) and
`crucible/alignmodels.py:61-63` (`{cuda-linux: "qwen3-forced-aligner"}`). `capability.decide()`
turns an empty candidate list into exactly the three doctor lines ("this build ships none with a
mlx-darwin block"). `crucible/workerenv.py:59` — `WORKER_JOB_TYPES = ("align", "asr", "rvc")`:
`asr` and `align` are subprocess WORKERS, not servers, so a library with no HTTP server fits
them directly; `pages` is the hard one because it rides the `llm` SERVER.

## 1. What each `mlx-darwin` block needs

### `align` — nearly done, outside Crucible

Qwen3-ForcedAligner does not run under MLX (`mlx-community/Qwen3-ForcedAligner-0.6B` is HTTP
404) and does not need to: it is plain torch, and `mlx-darwin` already runs torch-on-Metal
(`envs/rvc/mlx-darwin.txt` pins `torch==2.7.0`). narrator's aligner picks the dtype itself
(`python/narrator/align/aligner.py:620`, `mps` listed at `:465-471`).

`envs/align/mlx-darwin.md` is out of date: it says "nobody has measured it", but BookForge's
`electron/components/qwen-align-env.ts:8-9` records *"Measured on this M1 Ultra on 2026-09-08:
97x realtime warm on MPS in bfloat16 (33 s cold, model load included, for 95 s of audio)."*
What is still unmeasured is the timestamp comparison that file asks for.

The env exists on the Mac (`…/envs/qwen-align`, python 3.11.16, 94 packages, zero `nvidia-*`),
with the same three pins as `envs/align/cuda-linux.txt`:

| pin | cuda-linux.txt | Mac `qwen-align` |
|---|---|---|
| `qwen-asr` | 0.0.6 | 0.0.6 |
| `torch` | 2.14.0 (+15 `nvidia-*`) | 2.14.0 (Metal) |
| `transformers` | 4.57.6 | 4.57.6 |

Packed and published already: `qwen-align-env-macos-arm64.tar.gz`, sha256
`69b4bb14c644fa94cf7b243de071758d2197b5b1c248bf887c062db9fda415f2`, 499,191,377 bytes
(`qwen-align-env.ts:68-77`). Weights identical to the existing pin:
`Qwen/Qwen3-ForcedAligner-0.6B` @ `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`, 1,840,072,459 B.

Work: `envs/align/mlx-darwin.txt` = that freeze; `mlx-darwin: "qwen3-forced-aligner"` in
`ALIGN_BACKEND_ENGINES`; a `[backends.mlx-darwin]` block with its OWN measured
`memory_bytes_estimate` (cuda declares 3,446,157,280 — do not copy it); an `align/mlx-darwin`
row in `.github/workflows/envpacks.yml`.

### `asr` — mlx-whisper serves it; nothing on the Mac runs it today

faster-whisper is CTranslate2, which has no Metal backend, so this is a SECOND ENGINE (new
manifests, `mlx-whisper` in `ASR_BACKEND_ENGINES`, a second worker). Package
`mlx-whisper==0.4.3` (pulls torch). New ids, not `mlx-darwin` blocks on the six existing ones.
Measured via the HF API (sum of file sizes, 2026-09-14):

| would-be id | repo | main sha | bytes |
|---|---|---|---|
| `mlx-whisper-tiny` | `mlx-community/whisper-tiny-mlx` | `6caf9c55601caafbe6508a8b0d216bdf4783c4e8` | 74,420,620 |
| `mlx-whisper-base` | `mlx-community/whisper-base-mlx` | `1e3e249fb8d01c655324bd6841b1deadffd6d04c` | 143,726,326 |
| `mlx-whisper-small` | `mlx-community/whisper-small-mlx` | `45f3915923c7a79a5a5b5a7d909d39aeb0e5630e` | 481,309,720 |
| `mlx-whisper-medium` | `mlx-community/whisper-medium-mlx` | `7fc08c4eac4c316526498f147dfdee6f6303f975` | 1,524,927,044 |
| `mlx-whisper-large-v3` | `mlx-community/whisper-large-v3-mlx` | `49e6aa286ad60c14352c404340ded53710378a11` | 3,083,522,487 |
| `mlx-whisper-large-v3-turbo` | `mlx-community/whisper-large-v3-turbo` | `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb` | 1,613,979,758 |
| `mlx-whisper-distil-large-v3` | `mlx-community/distil-whisper-large-v3` | `e1c3c155644be59f8b477c0186719442f7e3fbb0` | 1,509,132,231 |

mlx-whisper was hand-run once on the Mac (memory `mac-gpu-whisper-and-wsl-oom`) in a conda env
`transcription` that no longer exists; mlx_whisper is installed nowhere on the Mac now, and
BookForge's code never referenced it — the app's Mac ASR is CPU faster-whisper. (Homebrew
`whisper-cpp 1.7.5` is present as a binary; not pack-shaped.)

### `pages` — weights, package and dialect proven; the engine slot is the blocker

Foundry's Mac route is real and measured: `mlx-vlm 0.6.10` + `mlx-community/dots.ocr-4bit`,
0.80% character error against the PDF's text layer, ~27 s/page on the M1 Ultra
(`foundry/src/vlm/models.ts:159-161, 191-194`; env `foundry-env-mac-arm64-v1`). It answers in
`parseDotsPage`'s `dots-json` dialect (Foundry's spec README says the MLX and vLLM builds do).
The env is on the Mac as conda `vlmtest` (mlx-vlm 0.6.10, mlx 0.32.0, mlx-lm 0.31.3).

Weights: `mlx-community/dots.ocr-4bit` @ `4ab989e403d4f8cafa5fdeede5b2290a706c2405`,
3,538,472,109 B (18 files, `model.safetensors` 3.524 GB). Foundry pins no revision; Crucible's
manifest loader requires the 40-char sha, so that commit is the pin.

The blocker: `mlx-darwin`'s engine is `mlx-lm` (`crucible/engines/mlx_lm.py` starts
`python -m mlx_lm server`, text only). Serving dots needs a second engine class (mlx-vlm ships
its own server: fastapi/uvicorn/starlette/python-multipart are its dependencies) and
`BACKEND_ENGINES` must stop being one-engine-per-backend. mlx-vlm 0.6.10's pins (`mlx>=0.32.0`,
`mlx-lm>=0.31.3`, `transformers>=5.14.0`) are satisfied by `envs/llm/mlx-darwin.txt` as it
stands — one env can hold both. `memory_bytes_estimate` for all three classes is owed a
measurement; the byte figures above are weight floors.

## 2. The ffmpeg PATH report — not a defect

| fact | evidence |
|---|---|
| ffmpeg exists | `/opt/homebrew/bin/ffmpeg -> ../Cellar/ffmpeg/8.1.2/bin/ffmpeg` |
| the plist carries the good PATH | written Sep 13 23:53; `<key>PATH</key>` includes `/opt/homebrew/bin` |
| the running service has it | `launchctl print gui/501/com.crucible.serve` → `environment = { PATH => …:/opt/homebrew/bin:… }`; `ps -Eww -p <pid>` agrees |
| `ssh mac '<cmd>'` gets the bare PATH | `/usr/bin:/bin:/usr/sbin:/sbin` |

`crucible/hosttools.py:56-58` is `shutil.which` over the CALLING process's PATH, and `doctor`
is a CLI. With `PATH=/opt/homebrew/bin:$PATH` the same doctor reports `job tts: ready` and ends
`healthy`. The Mac does not lack ffmpeg, the plist was not installed before the Phase 14 PATH
fix, and `service install` does not need re-running. Improvement owed: `doctor` should name the
service's PATH beside the shell's, since Crucible wrote the plist and can read it back.

## 3. The upgrade off conda, once v0.6.x publishes

All three job envs are venvs parented on the conda env (`~/.crucible/envs/{llm,rvc,tts}/
pyvenv.cfg`: `home = …/envs/crucible/bin`). Removing conda first kills 3.0 GB of working envs.
`~/.crucible/server` and `~/.crucible/downloads` do not exist yet; stamps are build stamps with
no `pack_sha256`. CI publishes four `mlx-darwin` packs (`server`, `rvc`, `llm`, `tts`).

Checklist (order is the risk):

1. The release exists: `envpacks.json`, the four mlx-darwin packs, `install.sh` on the tag.
2. `crucible service stop`; leave the plist.
3. `install.sh` → `mlx-darwin`, server pack into `~/.crucible/server`; `init` skipped because a
   config exists; the token is kept.
4. `crucible install llm`, `tts`, `rvc` from packs (denoise rides the rvc env).
5. `crucible service install` FROM A LOGIN SHELL (the recorded PATH is the installing shell's).
6. `crucible capability --write`; `doctor` from a login shell. `pages`/`asr`/`align` still NO
   until section 1 lands.
7. Only then remove the conda env `crucible`. The Mac's other conda envs (`narrator-mlx`,
   `vlmtest`, `qwen-align`, `bookforge-urvc`, `finetune`, …) are BookForge's and Foundry's.

Survives untouched: `config.toml` incl. the token, `models/` 33 G, `voices/` 57 G (7 voices),
`rvc/`, `rvc-base/`, `denoise-models/`, `logs/`. Lost or stale: `jobs/`, `uploads/` (fine);
`~/.crucible/hf-token.txt` (38 B) — Crucible never reads it (`weights.hf_token()` is `$HF_TOKEN`
else `[hf] token` in config.toml; the Mac's config has no `[hf]` section) — move or delete;
`narrator-higgs-voices.json` is regenerated (today names only `deathstalker` of seven).

Collectable now: `envs/rvc/mlx-darwin.txt`'s freeze — the real install happened and its
substitutions verified (`torch==2.7.0`, `torchvision==0.22.0`, `torchaudio==2.7.0`,
`onnxruntime==1.27.0`, `audio-separator==0.31.1`, `static_ffmpeg==3.0`, `static-sox==1.0.2`;
104 packages).

## 4. Mac capabilities `mlx-darwin` does not serve, beyond the three

| capability | evidence | door |
|---|---|---|
| Resemble-Enhance on MPS (the Enhance tab) | `electron/enhance-bridge.ts:148-164, 819-860`; `components/resemble-env.ts` | no job type (`denoise` is separation, not enhancement) |
| Foundry NLI zero-shot entailment on MPS | `foundry/src/analyze/nli_worker.py:129-143`; `env-catalog.ts:119-130` (null artifact) | no job type — PHASE15 §6 |
| MLX LoRA training | memory `mlx-lora-training-facts` | outside Crucible by design |
| Qwen3-14B GGUF via llama.cpp on :8090 (titles) | memory `triton-mac-14b-serving` | `llm` in kind; the llama engine is `llama-windows`, refused off win32 |
| headline32b MLX shim | memory `mlx-random-state-is-per-thread` | not in the catalog |
| Higgs zero-shot clip on the MLX arm | memory `higgs-zero-shot-path` | door exists, never exercised |
| urvc MPS memory patch | `envs/rvc/mlx-darwin.txt` names its absence | served, known softness |

Noted and NOT a defect: `modules/bookforge.toml` omits `denoise` from `job_types` on purpose —
`crucible install rvc` writes both flags (`workerenv.JOB_TYPES_SERVED_BY_ENV`) and naming it
would be refused; the file's comment says so.

## Bottom line

`tts`, `llm`, `rvc`, `denoise`: already out of the box, healthy, seven voices renderable.
`align`: nearly free. `asr`: a clean second engine with small measured weights. `pages`: proven
model and package; needs one engine per (backend, class-family) — decided in PHASE15 §4.6.
