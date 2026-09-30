# Config, environments, weights and the model catalog

Internals for `crucible/config.py`, `settings.py`, `upstreams.py`, `jobenv.py`,
`envpatches.py`, `envs/**`, `weights.py`, `manifests.py`, `models/*.toml`,
`catalog.py`, `tasks.py`, `modules.py`, `apiclient.py` and the package constants in
`crucible/__init__.py`. This covers what the code alone does not say: the
constraints, the measured numbers and the owner's rulings behind them.

## Package constants (`crucible/__init__.py`)

- `VERSION` is the build. `API_VERSION` is the HTTP contract, and it changes only on a
  breaking change. `API_HEADER`, `CLIENT_HEADER` and `USER_AGENT_HEADER` live in
  `crucible/protocol.py` rather than in the api package because the controller half
  (`peer.py`, running inside the Windows tray) has to send them and must not import
  FastAPI or uvicorn.
- `CLIENT_HEADER` exists because browsers cannot set `User-Agent`. It is a
  forbidden header, so a browser client would otherwise be recorded as a
  `Mozilla/5.0 ...` string.
- `KEEP_ALIVE_SECONDS = 75` is shared by both uvicorn entry points
  (`cli.cmd_serve`, `host/child_lifecycle.run_owned_server`). The server must hold an
  idle connection longer than the client pool does. uvicorn's default is 5 s and
  undici keeps idle sockets about 4 s. With those two defaults, align's first request
  after a render sometimes got `ECONNRESET` (measured 2026-09-18 to 09-20). 75 s is
  well above both and above the usual 60 s proxy idle timeout. It was chosen, not
  measured. The SDK also retries an idempotent request once on a reset that arrives
  before any response byte.

## Leaf modules

`config.py` sits under nearly everything, so it imports only modules that import
nothing heavier than `backend` and `errors`. Each fact below has one owner, and every
caller imports it from there (the older locations no longer answer; `tests/test_leaf_modules.py` holds them to it):

| leaf | owns | used to be imported from |
|---|---|---|
| `capabilityrecord` | `CapabilityRow`, `CapabilityRecord`, `desktop_reserve_words`, the `DESKTOP_BASIS_*` names | `config` |
| `classnames` | `CLASS_NAMES`, `ROUTABLE_CLASSES`, `SELECTABLE_CLASSES` | `capability` (gone; `capabilityclasses` refuses to import if its `CLASSES` disagree) |
| `narratorengines` | `HIGGS_V3`, `NARRATOR_ENGINE_SAMPLING`, `NARRATOR_ENGINES`, `DOCUMENT_READERS`, `ESTIMATE_BASES`, `EngineFootprint`, `declared_tts_footprints`, `VoicesDocumentView` | `config`, `voices`, `engines`, `narratorvoices`, `ttsplan` |
| `enginespec` | `dtype_of`, `declared_dtype`, `stated_dtype`, `run_dtype`, `dtype_on`, `bf16_fallback`, `card_needs`, `card_args`, `UNSTATED_ENGINE_CONCURRENCY` | `engines.vllm`, `decide` |
| `tomltable` | `check_table` and the `REVISION_PATTERN`, `MODEL_ID_PATTERN`, `HF_REPO_PATTERN`, `VOICE_ID_PATTERN`, `SHA256_PATTERN` rules | `manifests` |
| `upstreamrecord` | `UPSTREAM_NAMES`, `UPSTREAM_FIELD`, `UPSTREAM_DISPLAY`, `UpstreamRecord` and the `require_*` / `record_from_patch` validators | `upstreams` |

- A stated dtype has one rule (`enginespec.declared_dtype`): the block's `dtype`,
  else `--dtype` in its `engine_args`; `auto` or neither means none stated.
  `capability`, `precision` and the vLLM adapter all read it.
- Every manifest-table checker (`manifests`, `asrmodels`, `alignmodels`,
  `rvcmodels`, `denoisemodels`, `rvcbase`, `voices`, `voicerepo`, `config`) calls
  `tomltable.check_table` with its own error class.

## config.toml

### Location and writing

- The home is `$CRUCIBLE_HOME` when set, otherwise `~/.crucible`, and on Windows
  `%LOCALAPPDATA%\Crucible`. A dot-directory in the Windows profile may be roamed or
  backed up, and this directory holds tens of GB of weights. A Windows session
  without `LOCALAPPDATA` is refused rather than falling back to another location.
- `write_config` creates the home at 0700 and the file at 0600. It serialises the
  whole document with `tomli_w` into a temp file, fsyncs it and replaces the
  original, so a failed write cannot truncate the token or provider keys. It writes
  no comment lines. The token line is `token = "<...>"`, and the ladder reads it
  with `^\s*token\s*=\s*"([^"]+)"`.
- **Every rewriter must pass the loaded values.** `write_config` rebuilds the whole
  document from its arguments. A caller that leaves out `routes`, `upstreams`,
  `local_models`, `tts_engines` or `retention_days` silently erases them. If
  `[tts.*]` is lost, every repo-manifest voice fails with `engine_footprint_unset`.
  `install_on_submit=None` and `enable_image=None` keep whatever the file on disk says
  (`_kept_jobs_flag`), so an older rewriter cannot turn an operator's `false` back to
  true, nor a rewriter written before `image` existed turn the image type off. A config
  with no `enable_image` key (every config before 2026-09-28) reads it as false.
  `enable_audio` (2026-09-29) and `enable_segment` (2026-09-29) follow the same rule.
- `carried_tables` (used only by `crucible init --config-from`) copies whole tables
  unchanged, so keys this build does not know are preserved. A table that is both
  typed and carried is refused.
- The pairing file is written only by `crucible/pairing.py`.

### Defaults and rulings

Each default is set once as a module constant. An absent key means the ruled
default. Values of the wrong type are refused, not coerced; for example,
`open_pairing = "false"` is a truthy string.

| key | default | ruling |
|---|---|---|
| `[auth] open_pairing` | true | Owen 2026-09-17: *"ollama allows anybody to connect if they can reach it. make that the case with crucible servers as well"* |
| `[jobs] retention_days` | 7 | Owen 2026-09-18. This is a backstop: a job is reaped as soon as its artifacts are fetched. 0 and negative values are refused, because retention cannot be turned off. |
| `[jobs] install_on_submit` | true | Owen 2026-09-26: *"yes, we need to install a missing environment when a job is submitted"*. `POST /v1/jobs` reads it on every request. |

Always-written keys (`install_on_submit`, `retention_days`,
`desktop_allowance_basis`) are written so an operator can find them in the file.
Tables that are optional (`routes`, `local_models`, `tts`, `upstreams`) are written
only when they are non-empty.

### Desktop reserve (`[accelerator]`)

- On `cuda-linux` the reserve is VRAM the desktop holds that belongs to no job. The
  WSL2 driver shim does not list desktop apps as compute apps. `crucible init`
  measures it on NVIDIA (`ladder.measure_desktop_reserve`). The measurement is capped
  at `DEFAULT_DESKTOP_ALLOWANCE_BYTES` (3 GiB), which is also the fallback when
  nothing can be measured. 3 GiB was right for owens-pc, which streams, and held
  back half of kylies-pc's 6 GB card.
- On `mlx-darwin` the reserve is `MLX_DESKTOP_ALLOWANCE_FRACTION` (25%) of unified
  memory. That is the complement of Metal's `recommendedMaxWorkingSetSize`, which is
  about 75%. A flat 3 GiB would let the 64 GB Studio select the bf16 27B and leave
  macOS 8.5 GB. At 25% it selects the 4-bit, which is what Owen runs.
- `desktop_allowance_basis` is one of `measured`, `declared` or `stated`. A
  **stated** reserve is never changed automatically (Owen 2026-09-26: owens-pc keeps
  3 GiB on purpose because it streams). Only `crucible capability --measure-desktop`
  replaces it, and it prints the old and new values. `desktop_reserve_words` is the
  one sentence every surface uses to report the reserve.

### Tables and records

- `[capability]` is a record of the last decision, not the authority; `[jobs]
  enable_*` is the authority. An absent table means nothing has decided yet. It must
  never become an empty record, because that would read as "probed and nothing
  fit". `backend_kind` and `total_bytes` are the inputs the decision was made on,
  which is how `crucible doctor` notices a swapped GPU. TOML has no null, so
  `selected = ""` and `shortfall_bytes = 0` are used instead. `desktop_allowance_bytes`
  is stored twice, in `[accelerator]` (the authority) and in the record (the input it
  was decided on); `load_config` refuses a record whose `backend_kind` or
  `desktop_allowance_bytes` disagrees with the authority (`_record_agrees`), naming
  both values and `crucible capability --write`, which is the command that rewrites
  the record. `load_config(..., tolerate_stale_record=True)` is for that command and
  for `doctor`, which report the disagreement instead.
- `tts_engine_footprints` raises the `ConfigError` for a missing or unreadable file
  rather than answering `{}`; `voicerepo.voice_for_pin` turns it into one
  `config_unreadable` refusal for that pin, and `footprint_unset` names the command
  that writes a `[tts.<engine>]` table (`crucible init --force` for the engines
  `declared_tts_footprints` knows).
- `[routes]`: no entry means the class runs locally. `"local"` is never written.
  Upstreams are validated first, then routes, in the same order a `PUT
  /v1/settings` applies them.
- `[upstreams.*]`: the table's presence is what "configured" means. An entry
  without its one field (`key` or `url`) fails to load.
- `[local_models]`: no entry means "automatic". Whether the chosen model fits is
  decided by `capability`, not stored here.
- `[tts.<engine>]` holds the serving footprint for one narrator engine on this
  machine. It used to be repeated in all seven voice manifests with the same values,
  which showed it describes the machine and engine rather than a voice.
  `declared_tts_footprints` holds the values `crucible init` writes (the owner may
  instead leave the table unset until measured). Sources:
  - cuda-linux `19_000_000_000`: the SGLang `--mem-fraction-static 0.60`
    reservation on the 3090 Ti (BookForge, 2026-09-05). Basis `declared`.
  - mlx-darwin `12_133_000_000`: an 11.3 GiB peak at a 900-character chunk
    (mlx-audio 0.4.8, 2026-09-05). Basis `declared`.
  - `max_num_seqs = 16`: vllm-omni's stage-0 default. On owens-pc, 16 in flight
    measured 11,387-11,584 chars/min, and 32 stalled.
  - `llama-windows` serves no narrator engine and writes no `[tts]` table.

  A declared number must carry a note and a measured number must not.
  `mem_fraction` and `context_length` are machine facts here because
  `[voice.serving]` is refused in a repo manifest.
- `[server] advertise`, `tailscale_advertise` and `lan_advertise` are authorities
  (`host[:port]`), not URLs, and a scheme is refused. These are addresses that
  forward to this server from outside it; for example, `tailscale serve` on Windows
  forwards into WSL. They are added to the derived addresses, never used instead of
  them. Each has its own owner, so disabling the LAN door does not drop a tailnet
  address.
- `own_engine_backend`: a config with no `[server]` section runs no engine. On
  Owen's PC the Windows half is an orchestrator-only config.

### One Config per process

`Config.adopt` replaces the whole document in place, because every route,
`Residency`, `JobStore` and plugin closes over the same object. Rebinding them
would leave two sources of truth. `adopt` is the only place that writes to the
frozen dataclass. It loops over `__dataclass_fields__`, so a new field is adopted
automatically. A config from a different path or home is refused, because that is a
different server. `follow_file()` re-reads when the file's `(mtime_ns, size)` stamp
changes, so a write from another process (`crucible install`, `crucible capability
--write`) reaches a running server. The stamp is taken before the read. A file that
does not parse leaves the last good document in place.

## The settings door (`settings.py`)

Owen 2026-09-14: apps configure Crucible through the engine, and an app's settings
page is a window onto the server's settings, not a copy of them.

- **A refusal applies nothing.** The whole patch is resolved in memory, validated,
  and only then written. Resolution order: upstreams, then routes, then the
  allowance, then local-model choices. Removing an upstream is checked against the
  final routes, so one PUT can re-route and remove together.
- `resolve` is a loop over `SECTION_RESOLVERS`, (patch key, resolver) pairs in that
  order, then `tailscale_advertise` and `lan_advertise`. Each `_resolve_<section>`
  refuses and records only its own section; `_validate` checks the whole result. A new
  settings section is one resolver and one row in that tuple.
- **Keys are write-only.** A key is returned only as `key_hint`: `…` (U+2026) plus
  the last four characters. Foundry renders the hint verbatim. Keys never appear in
  a response, log line or activity row.
- **Routes, never fallbacks** (Owen, 2026-09-14). Work goes to an upstream only because
  a route the operator (or an app on the operator's behalf) set says so, visible on the
  page and in `/v1/capability` before any request. Nothing is sent to an upstream because
  something local failed.
- A route to an unconfigured upstream is never stored, and a hand-edited `[routes]` table
  that names one is refused when the config loads (`route_bad_model`,
  `route_upstream_unconfigured`), so a server never starts with a route it cannot serve.
- The document carries `upstream_labels` (`upstreamrecord.UPSTREAM_DISPLAY`), the display
  name of each upstream, so the operator page has no copy of that table.
- Setting a local model to null restores the automatic choice. A model that is not
  installed is **not** refused (Owen 2026-09-16). A model that does not fit is
  refused, with the arithmetic in the message.
- A local-model choice is refused while there is no `[capability]` record, because
  nothing has probed the card.
- Capability is recomputed on every write, using the card numbers from the record
  (not a new probe) and the live vendor and card facts.
- `apply` writes the file, then adopts it into the running Config. History
  (`HISTORY_LIMIT`) is in memory only and records field paths, never values of keys.

## Upstreams (`upstreams.py`)

- There are exactly three: `anthropic`, `openai` and `ollama`. Hosted base URLs are
  fixed, with no per-upstream `base_url`, so a typo cannot send a key to another
  host.
- The chat request is **never retried**, because a request that reached the
  upstream may already have been billed. A 429 is passed back with `Retry-After`.
  The model list is never invented or cached. An upstream rejection is `502`,
  never `401`, because `401` means the Crucible token was wrong.
- Anthropic requires `max_tokens`. The default is 4096, reported as `upstream
  default 4096`. `response_format` with a JSON schema becomes a forced tool
  (`structured_answer`), and `tool_use` maps back to `stop`. Leading system messages
  become the top-level `system`. `thinking` is dropped for Anthropic and OpenAI and
  reported as `dropped`.
- **Ollama uses its native `/api/chat`.** The OpenAI-compatible endpoint has no
  `num_ctx`, so long prompts were silently cut from the front. Every Ollama chat
  sends `options.num_ctx`. The value comes from the request's `context_tokens`,
  else the tag's own `PARAMETER num_ctx` (it wins over the trained maximum: Owen's
  `qwen3.8:27b-24g` is 98304 because 262144 does not fit 24 GB), else
  `model_info.<arch>.context_length`. It is never a made-up default.
  `X-Crucible-Context` reports the value and its source.
- `OllamaContexts` caches by (address, tag) and is valid only while `/api/tags`
  reports the same digest. Rebuilding a tag with `ollama create` changes the digest.
- The context lookup (`/api/show`, `/api/tags`) retries: 3 attempts, 10 s timeout,
  0.5 s and 2 s waits. A 4xx is not retried.
- Only the fields in `_OLLAMA_READS` are accepted; any other field is refused
  (`upstream_field_unsupported`).
- An Anthropic stream that stops mid-message gets a closing `[DONE]`. An Ollama
  stream without `done: true` gets **no** `[DONE]`, because that last line is the
  only evidence the answer is complete.

## Code, not environments

Owen, 2026-09-18: *"the unit of deployment is the code."* A deploy ships Crucible's code and
nothing that is published elsewhere; an environment is downloaded once, at install, from its
publisher, and is never rebuilt, re-hosted or re-downloaded because code changed.

- **A release carries only our code**: the wheel and sdist, the two SDK tarballs and the two
  generated installers (`scripts.md`, "What a release carries"). No environment pack, no
  rootfs, no re-hosted wheel.
- **Everything else comes from its publisher, pinned by version and digest**: CPython from
  python-build-standalone (`interpreter.py`), Crucible's own dependencies from PyPI
  (`pyproject.toml`), a job env from PyPI and the indexes its recipe names
  (`envs/<type>/<recipe>.txt`, exact pins), narrator from bookforge git at a commit, the WSL
  image from Canonical with Canonical's `SHA256SUMS`, `llama-server` from ggml-org's release,
  weights from Hugging Face.
- **An upgrade** is `pip install` of the new wheel into the interpreter already there, then a
  restart. A job env is touched only when its recipe's hash changed, and then by
  `pip install -r` into the existing venv, never delete-and-rebuild (the plan below). A moved
  narrator commit alone reinstalls that one line.
- **Repair is install.** `crucible doctor` names a missing or drifted env and `crucible
  install <job>` repairs it from its recipe: one path for a first install and a repair.
- **A normal deploy runs no tests** (`scripts.md`, "Tests"). The tests that matter ran on the
  branch before the merge; nothing that touches a GPU is on a release or deploy path.

### Archive sizes (`# archive-bytes:`)

Every recipe's first line is `# archive-bytes: <int> <citation>`, which the disk guard reads
before a fresh build (`jobenv.refuse_without_room`). The figures are the sizes of the
third-party wheels one install downloads, measured on 2026-09-18 when those wheels were last
re-hosted as packs. They are archive sizes, so they are floors for the unpacked env.

| env | archive bytes |
|---|---|
| `tts` (cuda-linux) | 5.3 GB |
| `llm` (cuda-linux) | 3.3 GB |
| `rvc` (cuda-linux) | 3.3 GB |
| `align` (cuda-linux) | 2.9 GB |
| `asr` (cuda-linux) | 1.3 GB |
| every mlx-darwin env | 0.9 GB, the combined mlx total, which can only overstate |

A recipe measured on its own replaces its row here and its header line together.

## Job environments (`jobenv.py`)

- There is one venv per job type (`~/.crucible/envs/<key>/`), built from a recipe
  under `envs/<job type>/`. The server never imports torch, vLLM, mlx or narrator.
  Worker types (`align`, `asr`, `rvc`) run their library as `<env python>
  worker.py`. `denoise` has no env of its own and runs in `rvc`'s, because
  audio-separator wants the same torch and a second env would be another 3 GB.
- On cuda-linux, `tts` has one env **per narrator engine** (`tts-higgs-v3`),
  because engines pin conflicting serving stacks and torches. On mlx-darwin every
  engine shares `tts`.
- **Serving stack: SGLang-Omni.** Owen 2026-09-15: *"we dont use vllm-omni. we use
  sglang. vllm-omni doesnt work for higgs."* vllm-omni's batched talker corrupts the
  newest batch row. Measured on the same 50 chunks: vllm-omni had 13/50 damaged at
  10,752 chars/min; SGLang had 5/50 at 26,666. `CUDA_LINUX_SERVING_STACK` names the
  stack the recipe installs. On mlx-darwin narrator renders in-process and reads no
  `HIGGS_STACK`. The `HIGGS_SGL_*` knobs are left unset because
  `serve_higgs_sgl.sh` already defaults them to the catalog's values: mem 0.60,
  7500 new tokens, CUDA-graph max batch = `HIGGS_MAX_NUM_SEQS`.
- **Interpreter.** `RECIPE_PYTHON` pins `higgs-v3-cuda-linux` to 3.12, because
  sglang-omni 0.1.4 pulls torch 2.13.0+cu130 and flashinfer for 3.12. Any other
  version is either the server's own interpreter or a pinned
  python-build-standalone, downloaded and digest-checked into
  `<home>/interpreters/`. Crucible never searches `PATH` for a Python.
- **Stamp.** `crucible-env.json` is written only by `_write_stamp`, only after pip
  returned 0 and the patches were applied. It records two halves: the environment
  hash (all lines except direct references) and the commit of each `name @ url`
  line, plus the recipe text.
- **Status.** `env_status` builds `EnvStatus` in one place. `_env_findings` walks venv,
  stamp, stamp readable, backend, then `_contents_findings` compares pins
  (`_pin_drift`) and direct references (`_reference_drift`); the first finding that is
  not installed is the reason. `install_env` is `_build_venv` (fresh build),
  `_install_recipe` or `_reinstall_references` (the plan), the patches, then the stamp.
- **Hashing.** The only normalisation is CRLF to LF. It is not the git blob hash,
  because recipes ship inside the wheel, where there is no git. A worktree with
  `core.autocrlf=true` once hashed differently from main. The environment hash
  ignores comment lines.
- **Plan** (`plan_install`, shared with `crucible doctor`):
  - A moved direct reference only: reinstall those lines with `--no-deps
    --force-reinstall`.
  - A moved pin: `pip install -r` into the existing venv.
  - A changed option line (`--index-url` and similar) or a removed requirement:
    rebuild. pip does not re-fetch a pin that is already satisfied, and it never
    removes packages. See `unverifiable_recipe_changes`.
  - A comment-only edit: nothing.
- **Disk guard.** `refuse_without_room` runs only for a fresh build, before pip
  touches the network. It reads the recipe's `# archive-bytes: <int> <citation>`
  line. That comment line is data and must stay. The value is an archive size
  ("Archive sizes" above, measured 2026-09-18), so it is a floor. Every mlx recipe
  carries the combined mlx total (0.9 GB), which can only overstate. Replace it
  when one mlx env is measured on its own.
- `recipe_index_urls` reads every index from the recipes, plus pip's default and
  `HF_ENDPOINT`, for the network probe (`docs/history/PHASE19-AUTOMATIC-WSL.md` 2.12).
- **Compiler.** The standalone CPython records `CC = clang`, and WSL Ubuntu has no
  clang. When the recorded compiler is missing, `build_environment` sets gcc, else
  cc, else clang, with the matching C++ compiler. This was measured on `diffq`
  (2026-09-26).
- `PipFailure` extracts one line naming the failing package and reason. The cause
  used to scroll out of the tail, 40 lines up.
- A direct reference must pin a 40-character commit. `pip list` reports declared
  versions only (for example, `ultimate-rvc` is 0.5.11 in both the fork and PyPI),
  so commits are checked against PEP 610 `direct_url.json`.

## Recipes (`envs/*/*.txt`)

Every recipe is a `pip freeze` of a real, working env, recorded exactly so an
install does not drift to a different torch. `crucible doctor` reports NOT READY if
any line is missing or at a different version. Regenerate after a deliberate bump
with `<env>/bin/python -m pip freeze`. Nothing is pruned: packages such as gradio,
flask and twine are declared dependencies, and trimming them needs its own measured
install.

- **llm cuda-linux**: `vllm==0.29.0` chosen. Resolved on owens-pc 2026-09-12:
  196 s, 8.0 GB env.
- **llm mlx-darwin**: `mlx-lm==0.31.3` chosen. Resolved on the Studio 2026-09-12,
  14 s. It includes `mlx-vlm==0.7.1` (the pages engine, run in-process by
  `engines/mlx_vlm_serve.py`), resolved with every older pin held as a constraint.
  0.6.10 fails on `mlx==0.32.2` (`mx.repeat` with an array count), and 0.7.1 fixes
  that. It also pulls `mlx-audio==0.5.5`, which is unused here; the tts env pins
  its own.
- **asr cuda-linux** (faster-whisper): numpy is 2.4.6, because 2.5.x requires
  Python 3.12 or later. The cuBLAS 12 and cuDNN 9 wheels are pinned so the env is
  self-contained; a system cuDNN of the wrong major version aborts inside
  CTranslate2. **Never add torch**: torch and CTranslate2 in one process corrupt
  CUDA state.
- **asr mlx-darwin**: `mlx-whisper==0.4.3`. `torch` is present only because
  mlx-whisper declares it. `mlx-metal` is the GPU half; without it the env silently
  runs on the CPU.
- **align** (Qwen3-ForcedAligner, `qwen-asr==0.0.6`, `torch==2.14.0`): never add
  faster-whisper or ctranslate2. The Mac recipe is the cuda set without the 19
  `nvidia-*`/`cuda-*` wheels and `triton`. The torch wheel for macOS arm64 already
  includes Metal support.
- **rvc**: `ultimate-rvc` is pinned to a commit of Owen's fork, because only the fork
  has `generate convert-dir`, which loads the model once per batch. torchaudio must
  be a CUDA build, because a CPU build silently moves resampling off the card.
  `diffq==0.2.4` comes from Crucible's `wheels` GitHub release (`--find-links`),
  because a stock WSL guest has no compiler. `onnxruntime-gpu==1.26.0`, not 1.27.0:
  1.27.0 links CUDA 13 (`libcudart.so.13`), while this env is CUDA 12.8.
  static_ffmpeg is gone on cuda-linux, where Crucible's own ffmpeg comes first on
  `PATH`, and kept on the Mac until a darwin build is pinned. The base assets
  (contentvec, rmvpe) are not in the recipe; they are the `rvc-base` subject. On the
  Mac, BookForge's MPS memory patch for urvc is not applied. The 96-file process
  recycle bounds memory. `use_autocast` is CUDA-only.
- **image** (mflux 0.20.0 on the Mac, diffusers at a pinned commit on CUDA): see
  [image.md](image.md), "Recipes". The cuda-linux recipe is a resolution, not yet a freeze.
- **audio** (one env per engine: `stable-audio-3-<backend>.txt`, `yue2-cuda-linux.txt`): see
  [audio.md](audio.md), "Envs". Resolutions, not yet freezes of a working env. The cuda
  Stable Audio recipe installs the flash-attn wheel from its GitHub release by URL, pinned by a
  `#sha256=` fragment: a direct reference is pinned by `@<commit>` or, for a wheel URL, by its
  digest, and `installed_direct_references` reads either back from pip's `direct_url.json`.
- **segment** (`envs/segment`, one worker env for BiRefNet and SAM 2.1: torch 2.14,
  transformers 5.17.0, timm, kornia, einops): see [segment.md](segment.md), "Envs". `uv pip
  compile` locks for Python 3.11 on each platform, not yet freezes of a working env; the
  headers are the wheels summed from PyPI (3.08 GB cuda-linux, 0.19 GB mlx-darwin).
- **tts cuda-linux** (`higgs-v3-sgl` extra, Python 3.12): this is the working
  BookForge `sglomni` env (Owen: *"mirror it. it should be exact"*). `uv` is kept
  because it is part of that env. `flashinfer-jit-cache` comes from flashinfer's own
  per-CUDA index. Without it the server compiles kernels on every cold start. The
  narrator pin must be at or after bookforge `edca6c22`, which ships
  `serve_higgs_sgl.sh`.
- **tts mlx-darwin**: `mlx-lm==0.31.3`, the version the batched fast path is pinned
  to. `mlx-audio==0.4.8`, because 0.5.1 raised TypeError on every generate under
  transformers 5.x. That was measured on a since-removed engine, so a re-measurement
  is owed. Narrator must be at or after bookforge `fbc235bf`
  (`item_sampling.py`), otherwise take rungs are silently ignored.
- **CUDA toolkit links** (`envpatches.ensure_cuda_toolkit_links`): flashinfer
  compiles with the nvcc inside `nvidia/cu13` only when that directory looks like a
  toolkit (`lib64 -> lib`, `libcudart.so -> libcudart.so.13`). The links are
  relative and created after pip and before the stamp. An existing link that points
  elsewhere is refused, not replaced. Without the links, installs succeed and the
  first render fails.

### ASR on the Mac

CTranslate2 has no Metal backend (`device="mps"` is a ValueError). Running it on the
CPU would silently change `compute_type` to int8 and produce a different
transcript, so the Mac runs mlx-whisper behind the same job type and wire. Each
whisper id has two conversions, and the transcript sidecar records which one ran.
`vad_filter: true` is refused on mlx (`vad_unsupported_by_engine`), because
mlx-whisper has no VAD. `language_probability` comes from `detect_language()` on
the first 30 s of the window. Measured on the M1 Ultra, 2026-09-14, over one 900 s
window with word timestamps: turbo peaked at 2,654,916,970 B (25.4x realtime) and
tiny at 549,418,642 B (75.3x). Accuracy between the two engines has not been
compared.

## Site-packages patches (`envpatches.py`, `envs/llm/patches/`)

A patch is defined by its distribution, target file, marker, an optional string
that must be absent, and a current-version marker; a patch with only an older
marker is reported `stale`. It is selected by the recipe's pins: a distribution
that is not pinned gives `not_applicable`, never a silent skip. `apply_patches`
runs after pip and before the stamp, and success is proven by `check_patches`, not
by the script's exit code. The markers in each script and in the envpatches table
must match exactly.

Script contract: `<env python> <script> <env prefix>` (or `CRUCIBLE_LLM_ENV`).
Scripts are idempotent by marker and patch the live file, never `.orig`. They are
version-pinned to mlx-lm 0.31.3 (`VERSION_MISMATCH`, exit 2) and all-or-nothing
(`ANCHOR_NOT_FOUND`, exit 2). `NOT_FOUND` and `AMBIGUOUS` are reported for a
missing file or two site-packages trees. `site-packages` is found by glob and
deduplicated by real path. Line endings are preserved.

The four mlx-lm 0.31.3 patches (`llm` env, mlx-darwin only).
`MlxLmEngine.start` refuses `llm_env_unpatched` unless all four report `applied`:

1. **top_logprobs 11 → 40.** The server validator is the only limit
   (`_format_top_logprobs` accepts any `top_n`). The decide door asks for
   labels + 4, and Briefcase needs 11 and 26 labels.
2. **float32 logprobs.** Stock mlx-lm normalises in bf16, which introduces an error
   of up to 0.0625, so `label_mass` came back 0.94-1.06 (qwen3.5-2b, 95th
   percentile 1.055). The patch returns float32 logprobs from all three sites; the
   sampler still reads the stock values, so generation is unchanged.
3. **Fatal generation thread.** An exception in `ResponseGenerator._generate`
   ended only that thread, and requests waited forever. ContentStudio saw 17-20
   minute hangs on 2026-09-25 (upstream ml-explore/mlx-lm#1672). The patch wraps
   the thread target so an exception prints the traceback and calls
   `os._exit(70)`. `sys.exit` would end only the thread.
4. **Cache counters.** `left_padding`, `lengths`, `offset` and `_idx` updated lazily
   grow an unevaluated graph that the prompt cache stores. On Qwen3.5/3.8
   (ArraysCache) that exceeds Metal's buffer limit (499000). This was reproduced on
   the 27B with ContentStudio's real requests. The patch adds the counters to the
   decode step's existing `mx.async_eval`.

## Weights store (`weights.py`)

- The layout is `~/.crucible/<family>/<id>/<backend>/`. The families (`models`,
  `voices`, `rvc`) are separate namespaces, so a voice pull cannot overwrite a model
  with the same id. A directory counts as installed only when
  `crucible-pull.json` exists, the revision matches the manifest, and every file the
  spec names is present. For example, a `dots-ocr` GGUF without its `mmproj` is not
  installed.
- There are three pull shapes, and all write the same stamp:
  - `pull`: a whole-repo snapshot, or only the named files on `llama-windows`. A
    GGUF repo holds every quantisation.
  - `pull_archive`: one `.tar.gz`, digest-checked before unpacking, used for RVC
    models. The tar member check is written out because `filter="data"` requires
    Python 3.12 or later.
  - `pull_files`: named files placed exactly where an engine looks, used for
    rvc-base and the separators. All digests are verified before any file is
    placed. `stamp_name` gives one stamp per set, for flat directories such as
    `denoise-models`. `force` replaces only this set's files.
  - `pull_archive` and `pull_files` fetch through one `_hub_download` of a `HubFile`
    and map hub exceptions through one `download_error` table: gated first, then a
    missing repo, revision or entry, then anything else.
- **Pinned vs local.** A local voice path is never fetched, stamped or deleted. It
  is reported with `pulled = None` and may disappear between jobs.
- **Aliases** (`[model] weights_of`, Owen 2026-09-23: *"One
  copy on disk, two fit rows in the catalog."*). An alias's weights are its base's
  folder and stamp. The alias owns only its `extra_files` (the llama-windows
  projector) and a record beside the base's stamp. Removing a base is refused
  (`weights_shared`) while a pulled and installed alias holds it. Removing an alias
  takes only its own files. Pulling an alias pulls the base first.
- **Cancel.** The progress hook is the only point where a `snapshot_download` can
  be interrupted, and it runs for every chunk. `PullCancelled` is never wrapped in
  `WeightsError`, and the partial files are removed. `_Reporting` counts bytes
  itself, because a disabled tqdm bar (`HF_HUB_DISABLE_PROGRESS_BARS`) does not
  call `update`, which would remove the cancel point. Only byte bars are reported.
- `_QuietUnauthenticated` filters only the Hub's "unauthenticated requests"
  warning. All pulled repos are public unless gated.
- **Gated repos.** A `GatedRepoError`, from a snapshot or a single-file download, becomes
  `gated_message`: the repo's page to accept the licence on, the token page, where the token goes
  (`$HF_TOKEN` or `[hf] token`), and the pull command to run again. Nothing is downloaded around
  a gate. An audio arm that declares `gated = true` refuses before the first request when there
  is no token at all (audio.md, "Weights").
- **Companions** (audio only, `audioweights.py`): a second repo an arm needs (YuE2's decoder)
  is a `pull_files` set in a subdirectory of the model's folder, with its own stamp; the model
  counts as installed only with every companion.
- `resolve_revision` resolves a repo to its current head in the engine, which
  already holds the HF token, so apps do not need a copy.
- `stranded` reports directories that no manifest declares. It never deletes them.
- **The Hugging Face cache.** A snapshot pull first hard-links whatever the Hugging Face cache
  holds at exactly the pinned revision (`adopt_hub_cache`, `$HF_HUB_CACHE`, else
  `$HF_HOME/hub`, else `~/.cache/huggingface/hub`); the hub download then hashes each linked
  file against the hub's sha256 and fetches only what is missing or different. The stamp
  records `linked_bytes`. Measured on the Mac's Qwen-Image 2.1 (image.md, "Weights and the
  Hugging Face cache").
- Every weights subject (model, ASR, align, voice, RVC and denoise manifests)
  answers `pull_command` (the exact command that fetches it) and `aliases()` (the
  manifests that share its weights; empty for kinds that cannot be aliased).
  `require_installed` names `pull_command` and `aliases_holding` asks `aliases()`,
  so `weights.py` imports no manifest module.

## Model manifests (`manifests.py`)

- **One validator per table.** `_parse` runs `_check_document` (top-level tables),
  `_parse_model`, `_parse_backends` (each `[backends.<kind>]` by `_parse_backend`, its
  `memory` by `_parse_memory`), `_parse_defaults` and `_parse_local` (the builder per
  kind is `_LOCAL_BUILDERS`). Each starts with `tomltable.check_table`, so a new key
  goes in that table's validator and its allowed set together. `resolve_weights_of` is
  `_weights_base`, then `_check_shared_facts` and `_check_shared_pins`.
- Validation is strict: an unknown key is refused, because a typo such as
  `memory_bytes_estimat` must not load with no estimate. `bool` is not accepted
  where `int` is expected.
- **Engine per (backend, class family).** The family (`text` or `pages`) is derived
  from what the block **serves** (`serves`, defaulting to `[model] modalities`,
  and allowed only to narrow them), never declared. mlx-darwin needs two engines:
  mlx-lm cannot take images, and `mlx-vlm` is Crucible's own dots-specific page
  server. `llama-windows` uses `llama-server` for both families, with `--mmproj` as
  the difference.
- **Image rules.** `--skip-mm-profiling` is refused when `image` is served, because
  pages would then meet an engine with no memory reserved for them.
  `--language-model-only` is refused when `image` is served, because the engine
  would answer pages it cannot see. On llama-windows, `file` is required, `mmproj`
  is required when images are served, and `mmproj` is refused otherwise.
- **GGUF floor.** Owen 2026-09-26: *"we can quantize if we need to. no less than 4."*
  Q3, Q2 and IQ2 files are refused.
- `[defaults]` may contain only keys the engines honour. A field stated in the
  request wins over the manifest, and a field neither states is left to the engine.
  `crucible/sampling.py` applies this and reports the source of each value. An
  integer `temperature = 0` is accepted.
- `params_b` is load-bearing: the 9B floor for clean, translate, simplify and
  analysis compares against it, so 0.8 is not rounded. `trained_context`
  (`max_position_embeddings` at the pin) is required. Without it the card-derived
  ceiling was unbounded: the 9B on the Mac was offered 1,389,135 tokens.
- `display` and `description` are required when `[local]` exists. `[local]` is
  base-only on aliases and feeds only `lineup.py`.
- `max_context` is the largest context a load may request. When it is absent, the
  ceiling is the served context. It must be ≤ `trained_context` and ≥ the served
  context. Owen 2026-09-23: requesting more than this *"throws an error back to the
  app"*.
- **Memory terms.** `engine_total = weights + overhead + kv_bytes_per_token × context
  × concurrency`. The terms must add up to `memory_bytes_estimate` within
  `MEMORY_TERMS_TOLERANCE` (5%). That tolerance catches GB/GiB mix-ups (7.4%) and
  computed-vs-measured KV errors (24%). `overhead_bytes = 0` is allowed.
  `basis` is `measured`, `computed` (a floor) or `declared`.
- Local ids may not contain `/` (`manifest_model_id_slash`), because the slash marks
  an upstream model id.
- `load_all_manifests` sorts by id, not path. `-` sorts before `.`, so path order
  would put `qwen3.8-27b-4bit` before `qwen3.8-27b`. `/v1/models` lists in this
  order.
- `resolve_weights_of` codes: `weights_of_unknown`, `weights_of_chain`,
  `weights_of_backend_missing`, `weights_of_pin_mismatch` (hf_repo, revision or
  file) and `weights_of_fact_mismatch` (family, params_b, trained_context,
  defaults).

## Model catalog numbers (`models/*.toml`)

### Rules that hold across the catalog

- **vLLM `--gpu-memory-utilization` is a budget, not a demand.** vLLM fills whatever
  the budget leaves with KV: `KV = util × total − non-KV`. It does not subtract the
  Windows desktop. On blocks with `[memory]` terms the value is ignored at runtime,
  because `vram.py` appends `--kv-cache-memory-bytes` and its own utilisation after
  the manifest args, and argparse keeps the last value.
- **Computed KV slopes are too small.** vLLM pads the attention page up to the
  gated-delta recurrent state. Measured against computed: 9B 40,337 vs 32,768
  (23% under), 27B-4bit 86,251 vs 65,536 (24%), 0.8B 18,023 vs 12,288 (32%). Do
  not compute a vLLM slope. llama.cpp KV is exact f16 and unpadded.
- **Only full-attention layers grow with context.** Qwen3.5/3.8 use
  `full_attention_interval` 4. The other layers are gated-delta with state per
  sequence, not per token.
- **mlx-lm prefill residual.** Prefill runs in 2048-token steps with logits over
  the 248,320-token vocabulary, so one step's logits are 2.03 GB. The 9B's residual
  is 2,069,045,094 B, and smaller mlx blocks carry it because the vocabulary is the
  same size. Computed weights + KV came out 34% under the measured value on the 27B
  at 98k context, so no long-context mlx estimate is computed.
- **mlx-lm batch sizing** (2026-09-24, measurement owed): the limit is Metal's
  `max_recommended_working_set_size`, 55,662,788,608 B (51.84 GiB) on the M1 Ultra,
  above which the Mac pages. The per-sequence cost is KV + float32 gated-delta
  state + conv state, held by the decode batch and by each of
  `--prompt-cache-size` cached sequences, plus `--prompt-concurrency − 1` extra
  prefill residuals. All three flags are always stated (`MlxLmEngine.start` refuses
  an argv without them). The chat door admits `--decode-concurrency + 1`.
- **Budgets used for `max_context`** (computed 2026-09-23, load tests owed): the
  3090 Ti has 24,564 MiB less 3 GiB = 22,535,995,392 B, and the Studio has 64 GiB
  less 25% = 51,539,607,552 B. 131072 is a cap chosen because longer prefills are
  not worth the time, not a memory limit. On mlx-darwin it is an admission ceiling
  only, because mlx-lm takes no context flag.
- **llama-windows blocks are declared**: file sizes plus Foundry's 1.5 GB
  `OVERHEAD_GB`. No Windows card has served them yet.
- **Local form.** The Ollama tag uses the same weights as the served form (for
  example, bf16 rather than the default q8_0). `needs_bytes` is the download plus
  1.5 GB, declared.
- Qwen3.5 has `thinking = false` by default. With a bounded budget the model spends
  it all on reasoning and returns no content. Both apps used to send this
  themselves.

### Per model

- **qwen3.5-9b** (the cleanup model): `context_default = 16384`. Four client records
  support that value; the earlier 12288 was the wrong tier of BookForge's
  `numCtxMaxForModel`.
  - cuda-linux, measured 2026-09-12 on the 3090 Ti: engine peak 19.52 GiB. The
    usable utilisation window is about 0.80-0.88: at 0.79 there is no KV pool, and
    0.85 with the default `--max-num-seqs` thrashed while capturing 51 CUDA graphs.
    `--max-num-seqs 16` gives 9 graphs in 5 s.
  - `--skip-mm-profiling` is worth 1.90 GiB (measured A/B).
    `--language-model-only` skips loading the 912,020,960 B vision tower (vLLM
    `no_init_weights`), which roughly doubles the KV pool at the same utilisation.
  - Calibrated 2026-09-18: weights 18,038,862,643, overhead 1,476,395,008,
    slope 40,337. The estimate was deliberately not lowered.
  - mlx: peak 20.38 GB at 12,198 tokens. RSS reads lower because weights are
    memory-mapped. At batch 16 the total is 32.14 GiB of 51.84.
  - llama-windows: Q8_0, because cleanup is where a wrong word is silently wrong.
  - The 9B is the local floor for translate and simplify (Owen 2026-09-16: *"they
    cant pick smaller than 9b"*).
- **qwen3.8-27b-4bit** (the PC's 27B):
  - cuda-linux uses `avyukth/...-AWQ-INT4`, one 18.57 GB file. cyankiwi's build
    (21.02 GB) does not fit with KV.
  - At 98304 context the guard refused it (23.3 GiB needed), so cuda-linux serves
    16384 and has `max_context` 32768. The measured peak is 20.15 GiB, the only
    fully measured row.
  - Terms: weights 18,061,981,175 (the card figure less the 921,460,192 B tower),
    overhead 1.44 GiB, slope 86,251.
  - `--gpu-memory-utilization 0.86` is the smallest that starts at 16384 (0.85
    fails).
  - fp8 KV might double the context but is unverified on Ampere.
  - mlx peak is 31.55 GiB at 98,220 tokens (two methods, 0.6% apart). The residual
    is 11,381,997,601 B. This block has no `[memory]` table: one point cannot
    separate intercept from slope.
  - llama-windows uses `UD-Q4_K_M` (there is no plain Q4_K_M) at 16384 context,
    because 98304 would exceed the declared allowance.
  - Its Ollama local form is the published parent `qwen3.8:27b`, since
    `qwen3.8:27b-24g` is a private Modelfile.
- **qwen3.8-27b-8bit** (Mac only; Owen 2026-09-17: *"put 8 bit on the Mac. 4 bit
  for pc"*; 2026-09-23: *"we shouldnt have an 8 bit 27b on here ... wont fit in the
  gpu"*): the weights (29,501,218,479 B) are the hub's blob sum. The residual is
  the 4-bit's (a double-counted KV term was removed 2026-09-23). At decode 8 and
  prompt 1 the total is 49.66 GiB of 51.84. A second concurrent prefill would page.
- **qwen3.5-4b / 2b / 0.8b** (decide only, below the 9B floor): official Qwen repos.
  The 2B is full precision everywhere (Owen 2026-09-24: *"full quant when
  possible"*). Images are served on cuda-linux and llama-windows, text only on
  mlx-darwin.
  - The 0.8B's text half was measured (2026-09-23): 1.53 GiB weights,
    0.75 GiB overhead, slope 18,023. The 2B's slope equals the 0.8B's exactly, and
    the 4B's equals the 9B's, because those pairs have identical attention
    configurations.
  - The 1.90 GiB image reserve (the 9B's A/B) is carried as an upper bound.
  - vLLM holds a constant 138,033,759 B beyond the language model and does not
    load `mtp.*`.
  - `--limit-mm-per-prompt {"image": 8, "video": 0}` matches `decide.MAX_IMAGES`.
- **qwen3.5-9b-vl / qwen3.8-27b-4bit-vl**: aliases that serve the vision tower.
  Switching between an alias and its base is a full engine reload (about 20 s on
  the 9B). Neither fits the 3090 Ti: the 9B-vl leaves 1,700 tokens of KV, and the
  27B-vl has no KV room at all. No mlx block exists. On llama-windows they apply
  only on machines without WSL.
- **dots-ocr** (the page reader):
  - The repo was renamed to `dots-studio/dots.ocr`; the old name redirects. A page
    at 200 dpi is about 3,450 image tokens, hence the 32768 context.
  - cuda-linux is declared at a 0.5 utilisation budget (12,878,610,432 B). Both
    apps reserve this; a larger reservation once OOM-killed the host under WSL,
    because reserved VRAM is also committed host RAM. A measurement is owed.
  - The GGUF pair is `ggml-org/dots.ocr-GGUF` Q8_0 plus its Q8_0 mmproj. The
    anthonym21 projector lacks `clip.vision.projector.scale_factor`, which upstream
    llama.cpp requires. The F16 projector is the next pin if Q8 reads worse.
  - mlx (measured 2026-09-21): `--width 12` equals `PAGE_CONCURRENCY`. Width 2 is
    reproducibly slower than serial. A peak of 13,963,861,086 B was measured on
    twelve 1300×2232 pages. The engine embeds one image at a time (the Metal
    watchdog kills twelve large pages in one call) and keeps prefill equal to the
    batch size.

## Catalog (`catalog.py`)

- `subjects()` is the single list behind both `GET /v1/catalog` and pull tasks.
  Kinds are listed in `KINDS` order. `engine` exists only on `llama-windows`, and
  `rvc-base` has the single id `base`.
- `license` is always null, because no manifest schema carries a license.
  `expected_bytes` is null for models and voices (a whole-repo snapshot has no
  declared size), and 0 or null for aliases, whose download is counted on the base.
- Local voices are not catalog subjects, because pull and remove would both refuse
  them.
- `backends_declaring` is on every row of a module file, because some subjects
  exist on one machine only (the Mac's 8-bit 27B).
- `stranded_weights` reports folders under an alias id as stranded, because an
  alias owns no folder.
- `Removals` is kept in memory only and records the last 20 deletions with who
  asked.
- `locate_installed` and `remove_subject` are the door's questions in the door's
  order (unknown kind, unknown id, not installed, held), each a `RemoveRefused` with
  the route's status and code. `remove_subject` takes a `holder` callback so the
  server can answer "resident, leased or named by a task" and the CLI, with no
  server, can answer "nobody". The `subject_unknown` text points at
  `crucible api catalog` (there is no top-level `crucible catalog`).

## Tasks (`tasks.py`)

- A task is work done to the server (`pull`, `install`, `module`, `engine`,
  `engine-restart`), not with its card. Tasks run one at a time, are not persisted,
  have no queued state, and are kept in memory (`HISTORY` = 50). A pull may run
  alongside jobs. An install may not start while any of the four facts
  (`settle.py`) hold the card, because it ends by swapping the registry. The
  exception is an install started by `POST /v1/jobs`, which adds plugins without
  replacing any.
- **Pull cancel** is cooperative: `DELETE` sets a flag, and the progress hook raises
  `PullCancelled` on its next call. The cancel check is not throttled; progress
  events are (0.5 s). **Install cancel** sends SIGTERM to the subprocess, with a
  10 s grace period.
- Installs run the `crucible` console script found next to the interpreter, then on
  `PATH`, never `python -m crucible`, which would import a `crucible/` folder in the
  current directory. `CRUCIBLE_HOME` is set explicitly. The last `crucible: ` line
  of output is the failure reason.
- A single pull that is already installed is refused. A module entry that is
  already installed is skipped.
- A module is validated as a whole before anything runs. Class needs are resolved
  per server from its capability record. A class this server cannot serve becomes
  an `unmet` row with the capability row's own reason; it does not fail the module.
- Engine move and restart are relayed from the host or orchestrator door (NDJSON)
  line by line, with no read deadline. Unreadable lines are forwarded as events.
  Endings:

  | condition | code |
  |---|---|
  | no `$CRUCIBLE_HOST_DOOR` | `engine_move_needs_host` or `engine_restart_needs_orchestrator` |
  | connection refused or timed out | `host_unreachable` |
  | stream ends with no terminal event | `host_install_failed` |
  | a `failed` event | that event's own code, verbatim |

  A restart's stream dies with the process, so clients should believe `/v1/info`
  rather than the stream.

## Module files (`modules.py`)

Module files are generated by `scripts/gen-modules.py` and never written by hand;
`--check` guards them. A declaration lists job types, capability classes and named
subjects. Classes are sent to the server unresolved, and each server resolves them
from its own record. At generation time the generator only checks that a class
exists and selects a model. A class with several candidates and no floor must be
named explicitly. Only model classes resolve (`capability.models_by_class`, the
same table `classes_for_model` reads); voices and other kinds must be named.
Every subject carries `backends`, which the app strips before posting. The version
is `<crucible version>+<first 12 hex of the sha256 of the content>`, excluding the
version field itself.

## `crucible api` (`apiclient.py`)

- The client verbs live under `api` because `crucible models` and `crucible voices`
  already describe this build, not a server. One binary provides both, as Ollama
  does.
- **Output is always JSON on stdout.** A single request prints one indented
  document. A followed stream prints one compact JSON line per event, then the final
  state. The only exception is `job artifact`, which writes raw bytes. There is no
  `--json` flag.
- **Refusals are printed verbatim** to stderr with exit code 1. For example,
  `server_busy` is kept as-is so it can be searched for. Transport failures and
  client refusals have their own codes. Two refusals are mapped to a sentence
  instead of the JSON (`_next_step`): `unauthorized` names the host and
  `crucible pair <address>`, and the `api_version_*` codes say which side is older
  from `details.server_api_version` / `client_api_version`. `error_in` is the one
  parser of the `{"error": ...}` body, shared with `_pair_call`.
- `server_unreachable` names the resolved connection (`unreachable`): the server's
  name, its URL and which flag or variable named it, then the next step on that host
  (`crucible doctor`, and `crucible lan enable` there for a remote refusal). It never
  says "the local engine" for a `--server`, `--pairing` or `$CRUCIBLE_PAIRING` target.
- **Connection.** Exactly one source: `--url` with `--token`; `--pairing`;
  `--pairing-file`, `$CRUCIBLE_PAIRING` or `--server NAME`; or nothing, meaning the
  local engine. `--url` without `--token` is refused and never falls back to the
  local token. The environment variable exists because `argv` is visible to `ps`.
- `follow` has no read timeout (the server sends a keepalive every 15 s) and no
  automatic reconnect. Resume by passing `--since <last id>`.
- `BrokenPipeError` is caught before `OSError`, so `| head` is not reported as the
  server being unreachable.
- `--params` is JSON validated by the server; job params are not duplicated as
  flags. `--artifacts-dir` does not imply `--follow`.
- `crucible pair <address>` fetches the pairing line through the connect door,
  verifies it on an authenticated route, and saves it to
  `<home>/servers/<slug>.pairing` without printing the token.
