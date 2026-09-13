# Crucible — design

Status: v1 API, phase 1 (handshake). Decided with Owen 2026-09-12.

## 1. What it is

One inference server, many client apps. Crucible runs models; it never knows what an
audiobook, a cleanup pass, or a PDF conversion is. Clients (BookForge, Foundry, Content
Studio, Briefcase) send bytes, Crucible returns bytes.

Three roles, kept distinct even when they ship together:

| Role | Lives in | Knows about |
|---|---|---|
| **Client SDK** | each app (`@crucible/client`, TypeScript) | servers, jobs, artifacts. Zero GPU or platform code. |
| **Bootstrapper** | each app's settings UI (later) | how to detect the host, install a local server, start/stop it, report health. |
| **Server** | this repo (`crucible`, Python) | backends, envs, models, VRAM, the queue. |

The client speaks HTTP to the server **even when it just spawned that server on
localhost**. There is no in-process shortcut. One code path for Owen's PC, a friend's
box, the Mac across the room, and a rented droplet.

## 2. Backends

A backend is a (platform, accelerator) pair the server can run on. The list is explicit
and short:

| Backend | Host | Engines |
|---|---|---|
| `cuda-linux` | Linux with an NVIDIA card. On Windows this is the Linux server inside WSL2 (later: Docker Desktop). | vLLM, SGLang, torch/CUDA |
| `mlx-darwin` | Apple Silicon Mac | MLX, mlx-lm, mlx-audio |

**Windows native is never a backend.** SGLang and vLLM do not run there. Crucible has no
Windows code path at all; the bootstrapper on a Windows client drives `wsl.exe -d <distro>
--exec bash -c ...` (never the implicit shell, which pre-expands `$var`).

The server detects its backend at startup. If nothing is viable it refuses to serve and
says why. There is no CPU fallback.

## 3. Job types

The vocabulary the server offers. Each is a plugin module under `crucible/jobs/<type>/`
declaring: the env it needs, the models it can serve, a VRAM estimate per model, and a
`run(job, ctx)`.

| Type | In | Out | Notes |
|---|---|---|---|
| `llm` | chat messages, model id, sampling | text | OpenAI-compatible endpoint (`/v1/openai/...`), so vLLM / SGLang / mlx-lm batching comes for free. Phase 2. |
| `vlm-pages` | PDF or page images | per-page structured markup | dots 3B. Send the PDF; the server rasterises. Phase 3. |
| `tts` | text chunks, voice id, sampling, cap | audio (FLAC per chunk) + progress | Higgs via narrator. Cap certificate is per (model, backend). Phase 3. |
| `align` | audio + text | VTT/JSON cues | Qwen3-ForcedAligner. Phase 4. |
| `rvc` | audio, model id, params | audio | Phase 4. |
| `echo` | any blob | the same blob, with progress events | Test-only, enabled by config flag. Proves the stream and artifact path. Phase 1. |

Prompts, chunking, rubrics, edit lists, retake ladders: **app logic, stays in the app.**

### 3.1 The division of knowledge (ruled by Owen, 2026-09-12)

BookForge is being split into two pieces, and the line is this: **the client knows what
the operator ordered and which server to send it to; the server knows how to run it.**
BookForge never learns whether a server is running SGLang, vLLM, vllm-omni or MLX. It
knows it has an order to narrate this book, that it has three servers registered, and
that the operator (or the queue) said "server 2".

Everything that tunes an engine to a model lives in **Crucible's own configuration**, not
on the wire: sampling defaults per backend, Orpheus's EOS controls, cap certificates,
token-budget formulas, dtype, engine flags. A `tts` request carries the text chunks, a
voice id and the take number; the server's voice config decides the rest. Where the app
genuinely needs to steer a knob (a temperature the operator set, a cap override), the
job contract names that knob explicitly, one at a time, with a reason. The default is
that it does not cross the seam.

This is the reading of `docs/CLIENT-SURFACES.md` section 10, tier 3: those rows describe
what the *server* must implement, not what the *client* must send.

## 4. API v1

Base: `http://<host>:<port>/v1`. Auth: `Authorization: Bearer <token>` on every route
except `ping`. Clients send `X-Crucible-Api: 1`; a major mismatch is refused with **426**
and a body naming both versions. The contract is the **API version**, separate from the
build version.

| Route | Auth | Returns |
|---|---|---|
| `GET /ping` | no | `{crucible: true, name, api_version}`. Lets a client tell "wrong token" from "not a Crucible". |
| `GET /info` | yes | server `{name, version, api_version}`, host `{platform, arch, backend, gpu: {vendor, name, vram_bytes}}`, `capabilities: [{job_type, models}]`. A capability's model rows are `{id, revision, source, resident, vram_bytes}` — **except `llm`**, whose rows are `GET /v1/models`' rows verbatim (PHASE2-LLM.md section 5). One model, one description: a client reads a model's standing in one shape wherever it finds it, and never reconciles two. |
| `GET /health` | yes | `{status: ok / warming / busy, queue_depth, resident_models}` |
| `POST /uploads` | yes | multipart → `{blob_id, bytes, sha256}`. For inputs too big to inline. |
| `POST /jobs` | yes | `{type, model?, params, inputs: {name: {blob_id} or {inline_base64}}}` (exactly one per input, unknown keys refused) → `{job_id}` (202). Refuses unknown type / model by name (400). |
| `GET /jobs/{id}` | yes | `{job_id, type, model, status: queued / running / done / failed / cancelled, progress, position, error (null when none), artifacts: [name], created, started, finished}` |
| `GET /jobs/{id}/events` | yes | SSE, ids strictly increasing from 1: `queued {position}`, `warming` (phase 2, payload TBD), `progress {fraction, message}` (the first, `fraction 0`, marks the job running), `artifact {name}`, `done {artifacts}`, `failed {error}`, `cancelled {status}`. Resumable with `Last-Event-ID`. |
| `GET /jobs/{id}/artifacts/{name}` | yes | bytes. `.../{name}.provenance.json` always exists (see section 7). |
| `DELETE /jobs/{id}` | yes | cancel: 200 `{status: "cancelling"}` on a running job (cooperative), 200 `{status: "cancelled"}` on a queued one, 409 `job_not_cancellable` on a terminal one. |
| `/openai/*` | yes | OpenAI-compatible passthrough for `llm` (phase 2). |

Errors are JSON `{error: {code, message}}`. No route ever degrades to a different
model or backend than asked for.

## 5. Server runtime

Python 3.11+, FastAPI + uvicorn, package `crucible`, CLI `crucible`:

```
crucible init             # mint token, write config, detect backend. Refuses if none.
crucible serve            # foreground; systemd/launchd units later
crucible doctor           # host probe + per-job-type env/model status, exit code = health
crucible install <type>   # create that job type's env and pull its models
crucible update           # self-update from GitHub Releases; drains the queue first
crucible token --show     # print the bearer token (owner only)
```

Config: `~/.crucible/config.toml` (mode 0600, holds the token), models under
`~/.crucible/models/`, envs under `~/.crucible/envs/`, job scratch under
`~/.crucible/jobs/<id>/` (pruned by age). On Windows all of this is inside WSL.

Env per job type, never one giant env. Envs are created by the installer from **PyPI,
the PyTorch index, and the SGLang/vLLM wheel indexes** using the recipe in
`envs/<type>/`. **Heavy wheels are never hosted on GitHub Releases.** Releases host: the
`crucible` sdist/wheel, the SDK tarball, and versioned env recipes.

Models: stock weights from their home (HF hub, e.g. Qwen 27B), Owen's fine-tunes from
HuggingFace (`owenmorgan/...`, private ones need the HF token on the server). Never on
GitHub Releases. A model manifest (`models/<id>.toml`) names source, revision, files,
per-backend caps.

## 6. Residency and the queue

The server owns the accelerator. One exclusive lease per GPU: `tts`, `rvc`, `align`
jobs run one at a time; `llm`/`vlm-pages` run through the engine's own continuous
batching. Models load on demand and are evicted least-recently-used when a job needs
VRAM the resident set can't give. Clients never see a lock file; they see `position`
and `warming`.

Two clients, one server: the queue is the arbiter. Two servers, one client: **one job,
one server**. A book is never split across backends (an MLX render and an SGLang render
are two different voices).

## 7. Provenance

Every artifact has a sibling `<name>.provenance.json`:
`{server: {name, version}, backend, job_type, model: {id, revision, fingerprint} or null for a model-less type, params, started, finished}`. Keys stay snake_case in every client; the sidecar is persisted verbatim.
Clients must persist it with the output. A finished audiobook says which server rendered it —
and, since `fingerprint` (`<id>@<revision>`, PHASE2-LLM.md section 5), which weights.

## 8. Versioning and updates

- `api_version` (integer, bumped on breaking change) is what clients check.
- `version` (semver) is the build.
- The server **updates itself** (`crucible update`) from GitHub Releases when its owner
  asks. The client never pushes an update to a server. Never auto-update: a release that
  changes a cap certificate changes the audio.

## 9. Auth and transport

Bearer token, always. `crucible init` mints it; the client stores it per server entry.
The transport is expected to be Owen's headscale tailnet (droplets join it too), so no
port is public; the token is the lock even if one is.

## 10. Non-goals

- No web UI on the server. Health is `GET /health` and `crucible doctor`.
- No library, no project files, no per-user state. Stateless per job.
- No automatic load-balancing across servers (later, and only across the same backend).
- No fallbacks anywhere: unknown model, missing env, no GPU, wrong API version: refuse by name.
