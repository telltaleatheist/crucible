# Crucible

An inference server for Owen's apps. One server binary per backend, one API, any client.

BookForge, Foundry, Content Studio and Briefcase all need the same class of hardware: a
GPU with a language model, a vision-language model, a TTS model, an aligner or a voice
converter resident on it. Crucible is the thing that gets hot. It runs the models. It
never knows what an audiobook or a cleanup pass is.

See `docs/DESIGN.md` for the architecture and `docs/PLAN.md` for the build order.

**Status: phase 2 (`llm`).** The handshake is real — config, token, backend detection,
API v1, the job queue, provenance sidecars, the `echo` test job type, and the TypeScript
client that speaks to all of it. Phase 2 adds the first real capability: model manifests,
a per-job-type env, managed vLLM / mlx-lm engines, the accelerator guard, and an
OpenAI-compatible proxy to one resident model. `tts`, `vlm-pages`, `align` and `rvc`
arrive in phases 3-4.

## Hosts

| Backend | Host | Detected by |
|---|---|---|
| `cuda-linux` | Linux with an NVIDIA card | `nvidia-smi` (PATH, then `/usr/lib/wsl/lib/nvidia-smi`) |
| `mlx-darwin` | Apple Silicon macOS | `arm64` + `import mlx.core` |

**Windows is not a backend.** On Windows, Crucible runs inside WSL2; the `crucible`
command refuses to do anything on win32 and says so. Anything else — Linux without a
working `nvidia-smi`, an Intel Mac, any other platform — is refused by name, with no
CPU fallback.

## Install

One conda env per host, not a shared one:

```bash
conda create -n crucible python=3.11 -y
conda activate crucible
pip install -e .            # add [test] for the test suite: pip install -e '.[test]'
```

## Use

```bash
crucible init --enable-echo     # detect the backend, mint a token, write the config
crucible doctor                 # probe the host; exit 0 only when healthy
crucible doctor --json          # the same report, machine-readable
crucible token --show           # print the bearer token
crucible serve                  # foreground; 127.0.0.1:7100 by default
crucible install llm            # build the llm env for this host's backend
crucible models list            # model manifests and their standing here
crucible models pull <id>       # fetch a model's weights at its pinned revision
```

`crucible init` refuses if a config already exists (`--force` replaces it and mints a
**new** token, which every client then needs). It refuses outright if no backend is
viable, naming the reason.

### Binding and the tailnet

`crucible serve` binds `127.0.0.1:7100` by default, which only this host can reach. To
serve other machines — the expectation is Owen's headscale tailnet, where no port is
public — bind wider:

```bash
crucible serve --host 0.0.0.0          # every interface
crucible serve --host 100.x.y.z        # just the tailnet address
```

The bearer token is the lock either way. `serve` prints which of the two it did.

### State on disk

Everything lives under `$CRUCIBLE_HOME`, default `~/.crucible`:

```
config.toml        mode 0600 — server name, bind defaults, backend, and the token
jobs/<id>/inputs/  the job's inputs, materialised before it is queued
jobs/<id>/artifacts/   its outputs and their .provenance.json sidecars
uploads/<blob_id>  blobs from POST /v1/uploads
envs/llm/          the llm job type's venv, built by `crucible install llm`
models/<id>/<backend>/  weights, stamped with the revision they were pulled at
logs/engine-<id>.log    one engine's stdout and stderr, command line first
```

`CRUCIBLE_HOME` is read on every call, so a second server (or a test, or the live
keeper) can be pointed somewhere else:

```bash
CRUCIBLE_HOME=/tmp/crucible-a1 crucible init --enable-echo
CRUCIBLE_HOME=/tmp/crucible-a1 crucible doctor --json
```

### The token

`crucible init` mints 32 random bytes (`secrets.token_urlsafe(32)`) and writes them to
`[auth] token` in `config.toml`. The file is created at mode 0600 under a 0700 home, so
it is never briefly world-readable. Nothing else stores it: no keyring, no env var, no
copy. `crucible token --show` prints it (the `--show` flag is required so it cannot be
printed by accident); `crucible doctor` reports the file's mode and calls the server
unhealthy if it is not 0600.

## API v1

Base `http://<host>:<port>/v1`. Every route except `ping` needs both
`Authorization: Bearer <token>` and `X-Crucible-Api: 1`. Auth is checked first, so a
request with neither is answered 401.

| Route | Auth | Notes |
|---|---|---|
| `GET /ping` | no | `{crucible, name, api_version}` — tells "wrong token" from "not a Crucible" |
| `GET /info` | yes | server, host (platform/arch/backend/gpu), capabilities |
| `GET /health` | yes | `{status, queue_depth, resident_models}` |
| `POST /uploads` | yes | multipart `file=@...` → `{blob_id, bytes, sha256}` |
| `POST /jobs` | yes | `{type, model?, params, inputs}` → 202 `{job_id}` |
| `GET /jobs/{id}` | yes | status, progress, position, error, artifacts |
| `GET /jobs/{id}/events` | yes | SSE, resumable with `Last-Event-ID` |
| `GET /jobs/{id}/artifacts/{name}` | yes | bytes; `{name}.provenance.json` always exists |
| `DELETE /jobs/{id}` | yes | cancel |

Errors are always `{"error": {"code", "message", "details"?}}`. Refusals name the thing
refused: `unauthorized`, `api_version_mismatch` (426, naming both versions),
`unknown_job_type`, `job_type_disabled`, `unknown_model`, `unknown_blob`, `unknown_job`.

Inputs come either inline or by blob:

```json
{"type": "echo",
 "params": {"delay_ms": 25},
 "inputs": {"page.png": {"blob_id": "…"},
            "note.txt": {"inline_base64": "aGVsbG8="}}}
```

SSE events are `queued`, `progress {fraction, message}`, `artifact {name}`, `done`,
`failed {error}`, `cancelled {status}`, each with an integer `id`. Reconnect with
`Last-Event-ID: <n>` to get everything after `n`, including the replay of a job that has
already finished.

### `echo`

The phase-1 test job type: it copies each input to an artifact of the same name,
emitting progress between the copies. It is registered only when
`[jobs] enable_echo = true` (`crucible init --enable-echo`); otherwise
`POST /jobs {"type": "echo"}` is refused with `job_type_disabled`. Its one parameter is
`delay_ms` (0-60000, default 25), which exists so SSE ordering and cancellation are
testable. Unknown parameters are refused, not ignored.

Cancellation is cooperative: a queued job is cancelled immediately, a running one is
told to stop and ends `cancelled` at its next checkpoint (`DELETE` answers
`{"status": "cancelling"}` in that case).

### `llm`

The first capability that touches the accelerator. Crucible runs one language model at a
time and puts an OpenAI-compatible surface in front of it; the prompts, the chunking and
the rubrics stay in the app.

```bash
crucible init --enable-llm        # or add [jobs] enable_llm = true to an existing config
crucible install llm              # build ~/.crucible/envs/llm and install the host recipe
crucible models list              # what this build ships and where each one stands here
crucible models pull qwen3.5-9b   # ~19 GB from HuggingFace at the manifest's pinned sha
crucible doctor                   # reports the env's presence and the versions installed
```

Then, over the API (`AUTH` is the two headers every route needs):

```bash
# load it — a normal job, on the same exclusive lane as everything else
curl "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d '{"type": "load-model", "model": "qwen3.5-9b"}' "$BASE/jobs"
curl -N "${AUTH[@]}" "$BASE/jobs/$JOB_ID/events"    # queued, warming..., done {resident}

# chat — proxied verbatim to the engine, streaming or not
curl "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.5-9b", "messages": [{"role": "user", "content": "hello"}]}' \
  "$BASE/openai/chat/completions"

# unload — the card comes back
curl "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d '{"type": "unload-model", "model": "qwen3.5-9b"}' "$BASE/jobs"
```

| Route | Notes |
|---|---|
| `GET /v1/models` | every manifest, with `backend_supported`, `installed`, `resident`, `loadable`, `reason`, `memory_bytes_estimate`, `context_default` |
| `POST /v1/jobs {type: "load-model", model}` | events `queued`, `warming {message}` (several, from the engine's readiness), `done {resident}` |
| `POST /v1/jobs {type: "unload-model", model}` | a normal job; `done {resident: null}` |
| `POST /v1/openai/chat/completions` | proxied to the resident engine, `stream` honoured |
| `GET /v1/openai/models` | the resident model in OpenAI's list shape, or an empty list |

`GET /v1/info` gains an `llm` capability carrying the `/v1/models` rows, and
`GET /v1/health` reports `warming` while a load runs plus `resident_models`.

#### One resident model at a time

Phase 2 holds exactly one model on the accelerator. Loading a second **unloads the
first** — you see that in the load job's `warming` stream. LRU across several comes
later.

The chat proxy never loads a model. If the body's `model` is not the resident one the
answer is **409 `model_not_resident`**, naming what is resident (or that nothing is):

```json
{"error": {"code": "model_not_resident",
           "message": "'gpt-4' is not resident on this server; 'qwen3.5-9b' is. Crucible never loads a model to answer a chat request — submit a {\"type\": \"load-model\"} job first.",
           "details": {"requested": "gpt-4", "resident": "qwen3.5-9b"}}}
```

#### Models are manifests

`models/<id>.toml` in this repo is the whole definition of a model: its id, family,
default context, and one block per backend naming the engine, the HuggingFace repo, the
**pinned commit sha**, a memory estimate and the engine's args. Validation is strict —
an unknown key is refused, every listed key is required, and a branch name is not a pin.
Weights land in `~/.crucible/models/<id>/<backend>/` and are only considered installed
once the pull has stamped `crucible-pull.json` there at the revision the manifest names.

The HF token for a private repo comes from `$HF_TOKEN` or `[hf] token` in `config.toml`;
without one, a private repo is refused by name.

#### Envs are recipes

`envs/llm/cuda-linux.txt` (vLLM) and `envs/llm/mlx-darwin.txt` (mlx-lm) are pinned pip
requirements. `crucible install llm` builds `~/.crucible/envs/llm/` as a venv from the
server's own interpreter and installs the recipe for this host's backend from PyPI —
heavy wheels never come from GitHub Releases. Engines are spawned from that venv, so the
API server process never imports torch or mlx.

#### The accelerator guard

Before any engine starts, Crucible looks at the card and **refuses rather than
competing**. It never evicts anything.

They are checked in this order — what can never be fixed, then what an install or a pull
would fix, then what the accelerator says right now — so a 27B on a 24 GB card is refused
for being a 27B on a 24 GB card rather than for needing a 55 GB download first:

| Refusal | When |
|---|---|
| `unknown_model` (400) | no manifest with that id |
| `backend_unsupported` (400) | the manifest has no block for this host |
| `insufficient_memory` (409) | the estimate exceeds the accelerator's **total** — never loadable here |
| `env_missing` (409) | `~/.crucible/envs/llm` is not installed |
| `model_not_installed` (409) | no weights at the manifest's pinned revision |
| `accelerator_busy` (409) | a process that is not Crucible's holds more than 1 GiB — named, with its pid |
| `insufficient_memory` (409) | not enough **free** memory right now — both numbers named |

All of these happen **before the job is queued**, so a client is told by name instead of
watching a job fail a minute later.

> **Measured limitation, WSL2.** The driver shim inside WSL2 answers
> `nvidia-smi --query-compute-apps` with an **empty list** even while a process in that
> same VM holds 17 GB of the card. `memory.free` under WSL2 *is* accurate for the whole
> card, so Crucible also refuses when VRAM is in use that no listed compute app accounts
> for, past `[accelerator] desktop_allowance_bytes` — the host desktop's own graphics
> memory, a declared fact in `config.toml` (3 GiB by default; use `0` on a headless box).
> On Apple Silicon that rule does not apply: "used" unified memory is the OS and the
> user's apps, so the free figure is the whole check.

#### Logs

Each engine's stdout and stderr go to `~/.crucible/logs/engine-<model id>.log`, starting
with the exact command line that was run. That is where a failed load's reason is, and
the `engine_failed` error quotes its last 40 lines.

#### Engines stop with SIGTERM

`stop()` signals the engine's process group and waits. Crucible **never** SIGKILLs a
process holding CUDA — that wedges WSL2 until Windows reboots. If an engine will not go
within 180 s, the refusal says so and names the log rather than escalating.

## The client

`sdk/ts/` is `@crucible/client`, the TypeScript client for this API: ESM and CommonJS
builds, `.d.ts` for both, and **no runtime dependencies** — Node 20+, bun and the
Electron main process all have `fetch`, `ReadableStream`, `FormData` and `Blob`.

```bash
npm install https://github.com/telltaleatheist/crucible/releases/download/v0.1.0/crucible-client-0.1.0.tgz
```

```ts
const crucible = new CrucibleClient({ url: 'http://127.0.0.1:7100', token, clientName: 'bookforge' });
const jobId = await crucible.submit({ type: 'echo', params: {}, inputs: { 'note.txt': { inline: bytes } } });
for await (const event of crucible.events(jobId)) console.log(event.event, event.data);
```

See `sdk/ts/README.md` for the whole surface and the error types. There is no npm
registry publish: the tarball on the release is the distribution.

## Releases

One version, one tag, one release. `v<ver>` carries all three artefacts —
`crucible-<ver>.tar.gz`, `crucible-<ver>-py3-none-any.whl` and
`crucible-client-<ver>.tgz` — because the server and the client that speaks to it share
a version, so a client can never be paired with a server nobody tested it against.

```bash
./scripts/release.sh --dry-run   # build and check, create nothing
./scripts/release.sh             # cut v<ver> from main
```

The version is read from `crucible/__init__.py`, `sdk/ts/package.json` and
`sdk/ts/src/version.ts`; a disagreement between any two of them is a refusal. So is a
dirty tree, an unpushed HEAD, and a tag that already exists.

## Tests

```bash
pip install -e '.[test]'
pytest                       # in-process, FastAPI TestClient, temp CRUCIBLE_HOME
./scripts/keeper-live.sh     # a real server on a free port, driven with curl
./scripts/keeper-llm-live.sh # a real server, a real engine, a real model
```

Both exit non-zero on any failure — trust the exit code, not the log. The pytest suite
never touches a real `~/.crucible`: every test gets a `CRUCIBLE_HOME` under pytest's
`tmp_path`, and backend detection is monkeypatched, so the suite is host-independent.
The live keeper is not: it runs `crucible init`/`serve`/`doctor` for real and asserts
that `/info` reports one of the two real backends.

`keeper-llm-live.sh` goes one further and loads a model. It needs the host ready — the
llm env installed and the model pulled — and **refuses by name** if either is missing
rather than skipping. It checks the card is idle first, loads `qwen3.5-9b` (override with
`CRUCIBLE_LLM_MODEL`), reads the `warming` stream, runs a non-streamed and a streamed
chat through the proxy, asserts the 409 for a wrong model name, unloads, and confirms the
accelerator returned to the figure it started at. The unit suite proves the same refusals
with monkeypatched probes and an in-process `Engine`, so a machine with no GPU still runs
every rule.

The SDK has its own two:

```bash
cd sdk/ts && npm ci && npm run test:unit   # the error map, in-process http fixture
./scripts/e2e.sh                           # Linux/macOS: real server, real client
./scripts/e2e-from-windows.sh              # Windows (Git Bash): server in WSL2, client native
```

The e2e needs `CRUCIBLE_URL` and `CRUCIBLE_TOKEN` and **fails by name** if either is
missing — it never skips. Both scripts set them up around a throwaway server on a free
port and stop it with SIGTERM.

Run the Python suites on Linux or macOS. On Windows the CLI refuses by design, so they
must run inside WSL2 — which is exactly what `scripts/e2e-from-windows.sh` arranges,
with the client staying native so the Windows -> WSL2 seam is what gets tested.
