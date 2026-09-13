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

Phase 2 ships three manifests: `qwen3.5-9b`, `qwen3.8-27b` and `qwen3.8-27b-4bit`. The
bf16 27B has a cuda-linux block too; the server will refuse to load it on a 24 GB card by
name (see 4), which is the point: the manifest says what the model needs, the host says
what it has. `qwen3.8-27b-4bit` is the same 27B at int4 — the one that does fit 24 GB —
and carries the 98304-token context of Owen's Ollama tag `qwen3.8:27b-24g`, so
`context_default` is a per-model number and not a constant. Nothing else about a model's
*use* belongs here: sampling is the client's, sent per request.

A backend block may also carry a `context_default` of its own, and then that is the
context that backend serves — `--max-model-len`, the `/v1/models` row, the resident
model. `[model] context_default` is what the model is FOR; the backend's is what that
accelerator has room for. `qwen3.8-27b-4bit` is the case that made this necessary:
98304 on 64 GB of unified memory, 16384 on a 24 GB card, both measured. It is optional and
never a silent default — absent means the model's number, and everything that needs a
context asks `ModelManifest.context_for(backend)` rather than reading either field.

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
| `GET /v1/models` | yes | `[{id, family, params_b, revision, fingerprint, backend_supported, installed, resident, loadable, reason (when not loadable), memory_bytes_estimate, context_default, max_model_len}]` |
| `POST /v1/jobs {type: "load-model", model}` | yes | a normal job. Events: `queued`, `warming {message}` streamed from the engine's readiness (several), `done {resident: id}`. Refusals by name before queuing: `unknown_model`, `model_not_installed`, `backend_unsupported`, `accelerator_busy`, `insufficient_memory`, `env_missing`. |
| `POST /v1/jobs {type: "unload-model", model}` | yes | a normal job; `done {resident: null}` — the same field the load reports, saying what is resident *now*, which after an unload is nothing. `model_not_resident` if it isn't. |
| `POST /v1/openai/chat/completions` | yes | proxied to the resident engine, streaming or not, verbatim but for `model` (see below). `model` in the body must equal the resident id, else **409 `model_not_resident`** naming the resident model (or none). Never loads implicitly. |
| `GET /v1/openai/models` | yes | the resident model in OpenAI's list shape (`{id, object, created, owned_by, engine_model_name, revision, fingerprint, max_model_len}`), or an empty list. |

`revision` is the pin in **this host's** backend block, so a client records the same sha
the puller used; it is `null` — not `""` — when `backend_supported` is false, because a
model this host cannot serve has no revision here to name. `memory_bytes_estimate` is
`null` in that same case and for the same reason: both figures live in the backend block
this manifest does not have, and `0` would read as "needs nothing".

`context_default` and `max_model_len` are two fields because they answer two questions.
`context_default` is the **manifest's intent**: what this host would serve this model at,
its backend block's own number where it has one (section 1). `max_model_len` is **what is
being served right now**: for the resident model it is the number the engine was actually
started with, read off the engine's record and not re-derived from the manifest, so a
manifest edited under a running engine cannot make this row promise a context nothing is
serving; for every other model it is what this host would start it with. They agree on a
server nobody has edited underneath, and the one moment they disagree is the moment a
client needs to be able to tell them apart. `max_model_len` is `null` when
`backend_supported` is false, alongside `revision` and `memory_bytes_estimate`.

`GET /v1/openai/models` carries `max_model_len` too, under OpenAI's own field name, and
that is the door that matters: Foundry reads the OpenAI-shaped listing rather than
`/v1/models`, and its `capFor` (`vllm.ts:208`) sizes `max_tokens` as
`max_model_len − (⌈chars/2.5⌉ + 256)` — **with no clamp at all when the server does not
report the field** (CLIENT-SURFACES.md section 6.1). An unclamped request is a 400 from
the engine, so the field being absent costs a whole call.

### The served name, and what a client writes down

**`fingerprint` is `<id>@<revision>`, and it is what belongs in a record — never the bare
id.** A client does not merely display the model it talked to; Foundry hashes the served
model id into its cleanup cache key and its translate bank, and BookForge stamps it into a
book's OPF (CLIENT-SURFACES.md section 6.5). Two consequences follow, and they pull in
opposite directions, which is why the rule has two halves:

- **The id is stable, so a cache stays warm.** Changing the name Crucible reports for the
  same weights re-asks every block of every book. `qwen3.5-9b` is that name on both
  backends, and the proxy puts it back on the way out precisely so that a book cleaned on
  the Mac and a book cleaned on the PC are filed under one name.
- **The revision travels with it, so a record is not a lie.** The same id serves different
  bytes on different hosts — `qwen3.5-9b` is Qwen's own repo on `cuda-linux` and the bf16
  conversion on `mlx-darwin`, at two different shas — and a manifest can be re-pinned. A
  record that says only `qwen3.5-9b` cannot tell those apart afterwards.

On `/v1/models`, `fingerprint` is `id` and `revision` from that same row joined, so it can
never disagree with them, and it is `null` wherever `revision` is — an unpinned fingerprint
would be worse than none, because it would look like a pin. On `/v1/openai/models` both
come off the resident engine, because that entry describes what is **running**. The
provenance sidecar (DESIGN.md section 7) carries all three — `{id, revision, fingerprint}`
— and its `revision` is this host's backend pin, which is a statement about bytes and not
about a file: a load refuses weights pulled at any other revision, so the pin the manifest
names is the pin the engine read.

**A Crucible id carries its dtype when the dtype is not bf16.** That is why
`qwen3.8-27b-4bit` is a separate id from `qwen3.8-27b` rather than a flag on it: a server
reports one served name, so two books cleaned at two precisions would otherwise be
indistinguishable in their records, and the int4 and bf16 answers to the same prompt are
not the same answer. bf16 is the unmarked case and takes no suffix. The precision is part
of the *id* and not of the revision because it is a choice about which weights to serve,
which the client may legitimately care about; the revision is which commit of those
weights, which it only records.

`GET /v1/info` gains `capabilities: [{job_type: "llm", models: [...]}]` whose rows are the
`/v1/models` rows **verbatim**, produced by the same function. That is the one exception to
DESIGN.md section 4's capability row shape, and it is deliberate: two descriptions of one
model is how a client ends up reconciling them.
`GET /v1/health` reports `warming` while a load job runs and `resident_models: [id]`.

The proxy is verbatim in both directions **except for the one field it owns**, `model`. A
reasoning model's `reasoning` comes back untouched, and per-request template controls the
client sends — `chat_template_kwargs`, which mlx-lm reads per request and vLLM honours by
the same name — are forwarded as they arrive; the server neither adds them nor strips
them. `model` is different because the proxy already substitutes it on the way in: mlx-lm
has no `--served-model-name` and answers to the resolved weights directory, so a request
for `qwen3.5-9b` reaches the engine naming a path. OpenAI engines echo the name they were
asked for, so Crucible puts its own id back on the way out — in the completion and in
every streamed chunk. Without it the answer to "what did I just talk to" would be a path
on the server's disk on `mlx-darwin` and the Crucible id on `cuda-linux`, where vLLM does
take a served name. One id, both directions, both backends. (On a backend where the two
names already agree there is nothing to undo and the stream is relayed byte for byte.)

That holds on the way **in** as well, and literally: where the engine answers to the
Crucible id there is nothing to substitute, so the bytes the client sent are the bytes the
engine reads — the proxy does not parse-and-re-serialise a body it has no field to change
in. Two things in particular ride on that and are tested (`tests/test_llm_api.py`):

- **`response_format: {type: "json_schema", json_schema: {...}, strict: true}`** is the only
  structured-output mechanism either app uses — Foundry's analyze verdicts and both tag
  calls (CLIENT-SURFACES.md section 6.2) — and its `schema` is a grammar the engine's
  guided-decoding backend compiles. Re-encoding somebody else's grammar in transit is not
  the proxy's job. The same applies to every other field the OpenAI dialect defines and
  Crucible has never been taught about: `logprobs`, `top_logprobs`, `seed`, `stop`,
  `logit_bias`.
- **`finish_reason` is never touched**, streamed or not, including `tool_calls`, whose
  `content` is `null`. Foundry turns `length` into a degradation rather than a wrong answer
  and BookForge's audiobook analysis throws by name on it; a proxy that normalised the
  field would turn a caught truncation into silent corruption.

An engine's own refusal is relayed with the engine's status code and body, not rewrapped in
Crucible's `{"error": {code, message}}` envelope. A schema vLLM will not compile is a 400
the *engine* made, and the message naming the part of the grammar to fix is the useful half
of it. A streamed request is no different: the upstream response is opened before anything
is returned, so a refusal arrives as a refusal and never as a 200 whose stream turns out to
be an error.

**A caller who goes away takes the engine's request with them.** Neither app can cancel a
chat any other way — Foundry's `Transport` has no abort member at all and BookForge chains
an `AbortSignal` to the fetch, so for both of them the cancel *is* dropping the connection
(CLIENT-SURFACES.md, closing section). Crucible runs one job at a time, so tokens generated
for somebody who has hung up are not wasted in the abstract; they are the next job's time.
A streamed completion's upstream is closed by the response that owns it, on every path out
of it. A non-streamed one races the upstream POST against the caller's own socket and
cancels the POST when that socket closes, which is what closes the engine's end: a bare
`await client.post(...)` watches nothing and would sit there to the end of the token
budget. The handler then answers **499 `client_disconnected`** — a status nobody will read,
because the connection it would travel down is gone, written down here so that nothing in
the code has to pretend a completion happened.

A load job runs on the exclusive lane like everything else, so it waits behind a running
job and a chat request never races a load.

## 6. SDK additions (`@crucible/client`)

`models()`, `loadModel(id)` → job id (use `events()` as usual), `unloadModel(id)` → job
id, `chat({model, messages, temperature?, topP?, maxTokens?, stop?, thinking?, signal?})`
→ the OpenAI response typed minimally (`id, model, choices[0].message.content, usage`),
and `chatStream(...)` → `AsyncIterable<string>` of content deltas. 409
`model_not_resident` surfaces as `CrucibleRefused` with that code. `signal` aborts the
fetch.

`thinking` is the one sampling knob that is not OpenAI's: Qwen3.5 and its kind think
before they answer, and a short ceiling spends the whole budget on `reasoning` and returns
a message with **no `content` at all**. `thinking: false` sends
`chat_template_kwargs: {"enable_thinking": false}`, `thinking: true` sends the same field
set to `true`, and omitting it sends nothing and leaves the model's own default alone. The
SDK still requires `content` on a completion — an answer that is not there is not an empty
answer — but names the cause when the message carried `reasoning` and stopped for
`length`, so the operator knows to raise `maxTokens` or turn thinking off.

`info()` reads the `llm` capability's rows with the `/models` reader, so
`ModelInfo.revision` is `string | null` and a capability is a union the client narrows on
`jobType` (`isLlmCapability`). `DoneData.resident` is `string | null`: `null` is what an
unload reports.

## 7. BookForge consumer

A new provider `crucible` in `AIProviderConfig`: `crucible: {server, model}` where
`server` names an entry in the registry. The cleanup path resolves the server, checks the
model is resident (refuses by name if not; never auto-loads inside a cleanup run) and
calls `chat()` with the existing system prompt, temperature and abort signal. CLI:
`--ai-cleanup --provider crucible --server <name> --model qwen3.5-9b`, plus
`--crucible-models`, `--crucible-load`, `--crucible-unload`, `--crucible-chat` for the
operator. Nothing in the app UI changes.

## 8. Verification (the GPU is touched for the first time)

Card check before any load: `nvidia-smi` must show only the desktop (1-3 GB) or the run is
refused; Owen's other work on the card is never evicted. On the PC: install the env in
WSL, pull `qwen3.5-9b`, load it (measure the real VRAM and record it in the manifest), run
a chat completion through the proxy and a streamed one, unload, and confirm the card
returns to the desktop-only figure. On the Mac: the same with mlx-lm and the bf16
conversion; then load `qwen3.8-27b` there to prove the 27B routing story. On the PC,
`load-model qwen3.8-27b` must be refused with `insufficient_memory` naming 54 GB against
24 GB. Everything through the CLI and the SDK, nothing hand-rolled.

**Done, both backends.** `mlx-darwin` 2026-09-12 on the Mac Studio; `cuda-linux` the same
day on the 3090 Ti, from Windows through the BookForge CLI to a server in WSL2 —
`keeper-llm-live.sh` in remote mode, 12 passed, 0 failed. See README's *cuda-linux,
verified* for the figures. Three amendments this section did not anticipate, all measured
and all now in the code or the manifests:

- **A load can fail for reasons that have nothing to do with the card.** Two of the three
  blockers on the PC were environment facts: WSL2's pinned memory disabled by default in
  vLLM, and no CUDA compiler in the llm env for FlashInfer's sampler to JIT against. The
  card check passes and the engine still dies. `crucible/engines/vllm.py` carries both
  reasons and the measurements.
- **`--gpu-memory-utilization` is not "how much of the card may we use".** It is a budget
  vLLM fills — spending the remainder on KV — and it does not subtract the host desktop.
  Left at the contract's example 0.85 it took the card to 139 MiB free. The number to pick
  is the one whose KV pool is the size you want, and a manifest should say which it is.
- **A measurement is worth a load, and the log is worth keeping.** Both computed estimates
  were light (the 9B's total by 6.3%, the 27B-4bit's KV-per-token by 24%), and the first
  failed run destroyed the engine log that explained it, because
  `scripts/measure-llm-memory.sh` deleted its throwaway home on the way out. It keeps the
  log on a non-zero exit now.
