# Crucible — design

Status: v1 API. **Every job type in section 3 is built, tested and merged** (2026-09-13).
`llm` and page reading are verified live on real cards with measured numbers in their
manifests; `tts`, `asr`, `align` and `rvc` were built against fake engines and their
manifests say so (`estimate_basis = "declared"`). `docs/PLAN.md`'s top block is the list
of what a free card and Owen's rulings still have to settle.

This file is the shape of the whole thing. Each job type's exact wire contract lives in its
own file and **wins over this one where they disagree**, because this one was written before
any of them had met an accelerator: `PHASE2-LLM.md`, `PHASE3-TTS.md`, `PHASE3-VLM.md`,
`PHASE4-AUDIO.md`. `CLIENT-SURFACES.md` is the audit of every model call the client apps
actually make, and is what all four were written from. Decided with Owen 2026-09-12.

## 1. What it is

One inference server, many client apps. Crucible runs models; it never knows what an
audiobook, a cleanup pass, or a PDF conversion is. Clients (BookForge, Foundry, Content
Studio, Briefcase) send bytes, Crucible returns bytes.

Three roles, kept distinct even when they ship together:

| Role | Lives in | Knows about |
|---|---|---|
| **Client SDK** | each app (`@crucible/client`, TypeScript) | servers, jobs, artifacts. Zero GPU or platform code. |
| **Bootstrapper** | THIS REPO (`@crucible/bootstrap`, TypeScript), consumed by each app's settings UI | detect the host, install a local server, ensure the SERVICE is running, report health. Owen ruled 2026-09-13 that a local Crucible is a service and not an app's child process, and that one bootstrapper ships with the server rather than one per app — PHASE5-APPS.md section 6.0. |
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
| `llm` | chat messages, model id, sampling | text | OpenAI-compatible endpoint (`/v1/openai/...`), so vLLM / SGLang / mlx-lm batching comes for free. Phase 2, `PHASE2-LLM.md`. |
| `tts` | text chunks, voice id, take | audio (FLAC per chunk) + measurements | Higgs and Orpheus through narrator. Two doors: a render job and a streaming connection. Voices are the server's, and so is every knob that tunes an engine to one. Phase 3b, `PHASE3-TTS.md`. |
| `align` | audio + text | timestamped items | Qwen3-ForcedAligner-0.6B, resident across a whole book. Phase 4, `PHASE4-AUDIO.md`. |
| `asr` | one audio file | transcript with word timestamps | faster-whisper, six sizes, no default. Phase 4, `PHASE4-AUDIO.md`. |
| `rvc` | audio + model id + params | audio | ultimate-rvc. Phase 4, `PHASE4-AUDIO.md`. |
| `echo` | any blob | the same blob, with progress events | Test-only, enabled by config flag. Proves the stream and artifact path. Phase 1. |

There is **no `vlm-pages` type**, and the reason is the one piece of this table that research
overturned. Both apps rasterise locally at a pinned 200 dpi and send an ordinary chat
completion whose first content part is a data-URI PNG, so a server receives pictures and
never PDFs, and page reading is the `llm` proxy plus a model whose manifest says it takes
images. `PHASE3-VLM.md` has the whole argument. Sending the PDF and rasterising server-side
is a real and larger job type; it is simply not the one either app needs, and building it
would have been a second residency and a second proxy for a chat completion.

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
| `GET /info` | yes | server `{name, version, api_version}`, host `{platform, arch, backend, gpu: {vendor, name, vram_bytes}}`, `job_types: [...]` (what this server accepts in `POST /jobs`), `capabilities: [{job_type, models}]` — **one row per capability, not one per postable type**: `llm` is operated through `load-model` and `unload-model`, neither of which is a capability, and listing them as such made one model appear three times in two shapes. A capability's model rows are `{id, revision, source, installed, resident, vram_bytes}` — `installed` is the puller's stamp at the pinned revision (the weights are on disk), `resident` is an engine serving them now, and neither implies the other — **except `llm`**, whose rows are `GET /v1/models`' rows verbatim (PHASE2-LLM.md section 5), and `tts`, whose rows are `GET /v1/voices`' (PHASE3-TTS.md section 8). One model, one description: a client reads a model's standing in one shape wherever it finds it, and never reconciles two. |
| `GET /health` | yes | `{status: ok / warming / busy, queue_depth, resident_models}` |
| `POST /uploads` | yes | multipart → `{blob_id, bytes, sha256}`. For inputs too big to inline. |
| `POST /jobs` | yes | `{type, model?, params, inputs: {name: {blob_id} or {inline_base64}}}` (exactly one per input, unknown keys refused) → `{job_id}` (202). Refuses unknown type / model by name (400). **Refuses `409 server_busy` when the lane is occupied** — the server admits one job at a time and does not queue (section 6). The refusal carries `details: {holder, job_id, type, model, status, since, progress, message}`, where `holder` is the recorded User-Agent (null when the client sent none), so a client can say *"GPU busy: foundry, 42% through a sigma render"* instead of polling blind. |
| `GET /jobs/{id}` | yes | `{job_id, type, model, status: queued / running / done / failed / cancelled, progress, position, error (null when none), artifacts: [name], created, started, finished}` |
| `GET /jobs/{id}/events` | yes | SSE, ids strictly increasing from 1: `queued {position}`, `warming` (phase 2, payload TBD), `progress {fraction, message}` (the first, `fraction 0`, marks the job running), `artifact {name}`, `done {artifacts}`, `failed {error}`, `cancelled {status}`. Resumable with `Last-Event-ID`. |
| `GET /jobs/{id}/artifacts/{name}` | yes | bytes. `.../{name}.provenance.json` always exists (see section 7). |
| `DELETE /jobs/{id}` | yes | cancel: 200 `{status: "cancelling"}` on a running job (cooperative), 200 `{status: "cancelled"}` on a queued one, 409 `job_not_cancellable` on a terminal one. |
| `/openai/*` | yes | OpenAI-compatible passthrough for `llm` (phase 2). |
| `GET /setup` | yes | **The operator door** (PHASE13-OPERATOR.md section 3.1). `{name, version, backend, bind, urls, token, pairing, job_types, config_path}` — everything an app needs to be pointed here, in one read. `urls` is the bind address made reachable (a wildcard bind becomes one entry per non-loopback IPv4 interface, read from `getifaddrs(3)`; never a hostname lookup), and `pairing` is one `crucible://<name>@<host>:<port>/#<token>` line per url. It returns the token and reveals nothing: only a caller who already has it can reach the route. |
| `GET /catalog` | yes | `{rows: [{kind, id, name, job_type, installed, installed_bytes, expected_bytes, floors, license, source, resident}]}` — every pullable subject this BACKEND can hold, installed or not. `kind ∈ model, voice, rvc, rvc-base, denoise`. Every field is derived from something the server already owns; a subject with no block for this backend is absent rather than listed as unsupported. |
| `POST /tasks` | yes | One operator operation on the server itself: `{type: "pull", kind, id}`, `{type: "install", job_type, narrator_engine?}` or `{type: "module", module}` → `{task_id}` (202). **One task at a time** (`409 task_busy`); an `install` additionally waits for the card (`409 server_busy`, whose `details.fact` names which of the four holders it is). Refusals by name at POST: `unknown_subject`, `already_installed`, `unknown_job_type`, `job_type_installed`, `narrator_engine_required`, `narrator_engine_refused`, `invalid_module`. |
| `GET /tasks` | yes | `{tasks: [...]}` — the last 50, newest first, in memory. A restart forgets them. |
| `GET /tasks/{id}` | yes | `{task_id, type, request, state: running / done / failed / cancelled, error, created, started, finished}`. There is no `queued`: a task is admitted and running in the same act. |
| `GET /tasks/{id}/events` | yes | SSE, the job stream's envelope: `started {type}`, `step {name, index, total, job_types?}`, `progress` (`{bytes_done, bytes_total, file}` for a pull, `{line}` for an install), `skipped {reason}`, `done`, `failed {code, message}`, `cancelled`. |
| `DELETE /tasks/{id}` | yes | cancel: 200 `{status: "cancelling"}`, 409 `not_running` once terminal. A pull stops at its next chunk and its partial directory is removed; an install is SIGTERMed. |
| `GET /` and `GET /ui/*` | **no** | The operator page, served as package data from `crucible/ui/`. Public because there is no secret in any of it: the page asks for the token, or reads it out of the URL fragment a pairing line put there, and a fragment never reaches the server. |

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
crucible token --url      # print the pairing line an app's connect door takes
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

## 6. Residency, admission, and the queue

The server owns the accelerator. One exclusive lease per GPU: `tts`, `rvc`, `align`
jobs run one at a time; `llm`/`vlm-pages` run through the engine's own continuous
batching. Models load on demand and are evicted least-recently-used when a job needs
VRAM the resident set can't give. Clients never see a lock file; they see `position`
and `warming`.

**Queues belong to clients; admission belongs to the server** (Owen, 2026-09-13;
ARCHITECTURE.md section 3). *"if the server is busy, it cant receive a new job. if its
not busy, it receives the next job requested."* So `POST /jobs` **refuses `409
server_busy`** while a job is on the lane, rather than appending behind it — the same
policy the streaming door has always had (`409 stream_session_open`), applied to the
door that disagreed with it. The reason is not simplicity: the client is the only thing
that knows the chain, the pin, the priority and which book is being watched, so a
server-side FIFO can only be a dumb queue that the smart client queue then has to model.
Two arbitrators, one strictly less informed.

The refusal is informative or it is useless — a bare "busy" makes clients poll, and
polling is a *worse* queue than FIFO, since the winner becomes whoever polls at the
luckiest moment rather than whoever asked first. So it names the holder and what it is
doing (section 4).

**The lane is not the only way to be busy.** A streaming session holds the resident
engine without occupying the lane, so the job types that would talk to it or move it —
`load-model`, `unload-model`, `load-voice`, `unload-voice`, `tts`, `align`,
`unload-aligner` — additionally refuse `409 engine_in_use`, naming the holder. `asr` and
`rvc` do not: they never touch the resident engine, and what they contend for is memory,
which `accelerator.guard` already refuses by name.

`crucible/jobs/queue.py` keeps the lane, the deque, `position`, `queue_depth`, cancel,
events and provenance. One policy decision changed, at admission; `queue_depth` is now
honestly 0 or 1, and restoring queueing is the same one line.

Two clients, one server: the first to ask gets it and the second is told who has it.
Two servers, one client: **one job, one server**. A book is never split across backends
(an MLX render and an SGLang render are two different voices).

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
