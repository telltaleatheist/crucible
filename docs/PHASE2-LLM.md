# Phase 2: the `llm` job type

Contract for the first real capability. Extends DESIGN.md; where the two disagree this
file wins for `llm`. Decided 2026-09-12.

## 1. Models are manifests

`models/<id>.toml` in this repo, one per model id. The id is Crucible's, stable across
backends; the weights differ per backend.

```toml
[model]
id = "qwen3.5-9b"
family = "qwen3.5"
params_b = 9
context_default = 12288        # Owen's pinned cleanup ctx

[backends.cuda-linux]
engine = "vllm"
hf_repo = "Qwen/Qwen3.5-9B"
revision = "<commit sha, pinned by the builder after inspecting the repo>"
memory_bytes_estimate = 21000000000   # weights + KV at context_default; measured, not guessed
engine_args = ["--dtype", "bfloat16", "--gpu-memory-utilization", "0.85"]

[backends.mlx-darwin]
engine = "mlx-lm"
hf_repo = "mlx-community/Qwen3.5-9B-bf16"
revision = "<pinned>"
memory_bytes_estimate = 21000000000
```

Phase 2 ships two manifests: `qwen3.5-9b` and `qwen3.5-27b`. The 27B has a cuda-linux
block too; the server will refuse to load it on a 24 GB card by name (see 4), which is
the point: the manifest says what the model needs, the host says what it has.

Weights live under `~/.crucible/models/<id>/<backend>/`, pulled by `crucible models pull
<id>` with `huggingface_hub` at the pinned revision. Never from GitHub Releases. The HF
token comes from `HF_TOKEN` in the environment or `[hf] token` in config; a private repo
without one is refused by name.

## 2. Envs are recipes

`envs/llm/cuda-linux.txt` and `envs/llm/mlx-darwin.txt`: pinned pip requirements
(`vllm==0.29.0` / `mlx-lm==0.31.3` plus whatever each needs). `crucible install llm`
creates `~/.crucible/envs/llm/` as a venv from the server's own interpreter and installs
the recipe for the host's backend from PyPI and the PyTorch index. Heavy wheels never
touch GitHub Releases. `crucible doctor` reports the env's presence and the pinned
versions actually installed.

## 3. Engines are managed subprocesses

`crucible/engines/vllm.py` and `crucible/engines/mlx_lm.py`, both behind one small
`Engine` interface: `start(model_dir, served_name, port, args)`, `ready()` (polls the
engine's `/v1/models`), `stop()` (SIGTERM, wait for exit, **never SIGKILL**: a killed
CUDA process wedges WSL), `base_url`. The engine binds `127.0.0.1` on a free port; only
Crucible talks to it. Engine stdout/stderr goes to `~/.crucible/logs/engine-<id>.log`.

Phase 2 residency rule: **one resident model at a time**. Loading a second unloads the
first. LRU across several comes later.

## 4. The accelerator guard

Before an engine starts on `cuda-linux` the server runs `nvidia-smi
--query-compute-apps` and refuses with `accelerator_busy` if any process not owned by
Crucible holds more than 1 GiB, naming the process. It also refuses with
`insufficient_memory` if free VRAM is below the manifest's estimate, naming both numbers.
On `mlx-darwin` the check is free unified memory against the estimate. There is no
eviction of other people's processes, ever.

## 5. API additions

| Route | Auth | Returns |
|---|---|---|
| `GET /v1/models` | yes | `[{id, family, params_b, backend_supported, installed, resident, loadable, reason (when not loadable), memory_bytes_estimate, context_default}]` |
| `POST /v1/jobs {type: "load-model", model}` | yes | a normal job. Events: `queued`, `warming {message}` streamed from the engine's readiness (several), `done {resident: id}`. Refusals by name before queuing: `unknown_model`, `model_not_installed`, `backend_unsupported`, `accelerator_busy`, `insufficient_memory`, `env_missing`. |
| `POST /v1/jobs {type: "unload-model", model}` | yes | a normal job; `done`. `model_not_resident` if it isn't. |
| `POST /v1/openai/chat/completions` | yes | proxied verbatim to the resident engine, streaming or not. `model` in the body must equal the resident id, else **409 `model_not_resident`** naming the resident model (or none). Never loads implicitly. |
| `GET /v1/openai/models` | yes | the resident model in OpenAI's list shape, or an empty list. |

`GET /v1/info` gains `capabilities: [{job_type: "llm", models: [...same as /models]}]`.
`GET /v1/health` reports `warming` while a load job runs and `resident_models: [id]`.

A load job runs on the exclusive lane like everything else, so it waits behind a running
job and a chat request never races a load.

## 6. SDK additions (`@crucible/client`)

`models()`, `loadModel(id)` → job id (use `events()` as usual), `unloadModel(id)` → job
id, `chat({model, messages, temperature?, topP?, maxTokens?, stop?, signal?})` → the
OpenAI response typed minimally (`id, model, choices[0].message.content, usage`), and
`chatStream(...)` → `AsyncIterable<string>` of content deltas. 409 `model_not_resident`
surfaces as `CrucibleRefused` with that code. `signal` aborts the fetch.

## 7. BookForge consumer

A new provider `crucible` in `AIProviderConfig`: `crucible: {server, model}` where
`server` names an entry in the registry. The cleanup path resolves the server, checks the
model is resident (refuses by name if not; never auto-loads inside a cleanup run) and
calls `chat()` with the existing system prompt, temperature and abort signal. CLI:
`--ai-cleanup --provider crucible --server <name> --model qwen3.5-9b`, plus
`--crucible-models`, `--crucible-load`, `--crucible-unload`, `--crucible-chat` for the
operator. Nothing in the app UI changes.

## 8. Verification (the GPU is touched for the first time)

Card check before any load: `nvidia-smi` must show only the desktop (about 2.3 GB) or the
run is refused; Owen's other work on the card is never evicted. On the PC: install the
env in WSL, pull `qwen3.5-9b`, load it (measure the real VRAM and record it in the
manifest), run a chat completion through the proxy and a streamed one, unload, and
confirm the card returns to the desktop-only figure. On the Mac: the same with mlx-lm
and the bf16 conversion; then load `qwen3.5-27b` there to prove the 27B routing story.
On the PC, `load-model qwen3.5-27b` must be refused with `insufficient_memory` naming
54 GB against 24 GB. Everything through the CLI and the SDK, nothing hand-rolled.
