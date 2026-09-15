# Phase 15 — Crucible is the one door: host mode, routes, and settings the apps write through

**Owen, 2026-09-14 (evening):** *"the user will be installing bookforge/foundry (and by
extension, crucible) on windows. they wont be installing it from wsl, theres no front door in
wsl … one centralized location that controls the GPU power … bookforge/foundry gain a simple
contract: send commands to the crucible server. period. they dont have ollama fallbacks or
cloud anything at all … one contract, one SDK, one API, one communication method. and theres a
guarantee that all testing to determine if the user's system is powerful enough is done once."*

And: *"Bookforge and foundry setup/settings should be able to configure crucible settings. If
the user enters an anthropic api key, it should pass through to crucible … the user shouldn't
have to interact with crucible almost at all but should have access to it if they want to."*

This document is the CONTRACT (ARCHITECTURE.md R1: every name on the wire has one owner, and
that owner is this file). Four builds read it: the server (section 3), the Windows host
(section 4), BookForge (section 5) and Foundry (section 5). A build that needs a name this file
does not have adds it HERE first, in its own commit, and says so.

## 0. What is decided, and what it corrects

> **SUPERSEDED IN PART, 2026-09-15 — `docs/PHASE17-ORCHESTRATOR.md`.** Section 4's host
> IS an orchestrator, and Phase 17 gives that relation a name, a `role` on `/v1/info`,
> a claim (`POST /v1/peer/claim`) and a second task (`engine-restart`). Nothing below is
> withdrawn — one server per machine, control is Windows's and data is the card's, the
> apps keep one address — and `crucible host` keeps working as an alias for
> `crucible orchestrator`. Read PHASE17 before touching section 4.

> **AMENDED 2026-09-14, later that evening — Owen:** *"the windows side should still host GPU
> jobs even if WSL isnt present/workable. if the user cant or wont install WSL, we can still
> run dots, qwen 9b, or whatever else from the windows side. just like it runs from the mac
> side. it just wont have the benefits of sglang or vllm or whatever if they dont install
> wsl."* And: *"in an ideal world, though, dots would run from the WSL side of crucible,
> because we can parallelize it with VLLM. it gets significantly faster. but if they dont have
> wsl, they can use windows crucible."*
>
> So **Windows IS a backend: `llama-windows`**, structurally what `mlx-darwin` is — a
> per-model engine child the server spawns, leases, settles and kills — with `llama-server`
> (llama.cpp) as the engine and GGUF as the weights. There is no `backend_kind = "none"`.
> "Host mode" below means *a Crucible server running natively on Windows with the
> `llama-windows` backend*; every sentence that said "no accelerator" is struck by this block
> and rewritten in 3.5 and 3.10. What WSL adds on top is vLLM/SGLang (parallel page reading,
> the faster text path) and the Python job types (`tts`, `asr`, `align`, `rvc`, `denoise`).
> **The WSL engine is the preferred one on every Windows machine that can run it**: the host
> (section 4) offers the WSL install as the upgrade from the first day, and migrates the
> config into the guest when it arrives, exactly as before.

**One server per machine, still.** The Windows half of this phase is NOT a second Crucible and
NOT a relay. Two processes forwarding to each other would be a fact with two owners (two
versions, two health states, a hop on the TTS stream). The server that answers `:7100` on a
Windows machine is the WSL one when WSL is there, and a **host-mode** server (`llama-windows`,
section 3.5) when it is not. Never both. Apps connect to the same address either way.

**Control is Windows's; data is the card's (Owen, 2026-09-14: "the windows side will always
configure the WSL side. the WSL side just orchestrates commands on behalf of windows").** The
host on Windows is the ONLY thing that installs, starts, stops, updates and reconfigures the
WSL engine — one install, one page, one config, one version, and the guest has no installer
and no configuration page of its own. What Windows does NOT do is sit in the request path:
the server that runs the card answers the apps directly on `:7100`, because a relay would be
two server processes with two versions and two health states and a byte-copying hop on every
stream, for nothing an app can see. The apps' contract is one address and one token either
way.

**Windows gets a presence.** WSL has no boot: nothing starts a distro at login, so today the
engine is down after every reboot until an app happens to poke it, and when a clean stop left
it down at 16:10 nothing noticed. The only process that can own "the engine is running" is one
that is itself running on Windows. That is `crucible host` (section 4): a tray icon, a login
task, a watcher, and the installer/guide the wsl-states table (PHASE14 4c) was written for.

**The apps have no provider code.** Ollama, Claude, OpenAI, "does the 27B fit", cloud key rows,
cloud model lists, cloud slots: all of it leaves BookForge and Foundry. The app sends a
capability's work to the engine and the engine's config says where it runs. This
**overrules the 2026-09-14 morning ruling** that cloud keys live in Foundry's cloud card
(BookForge `docs/CRUCIBLE_ROLLOUT_PLAN.md` §3): the keys move INTO the engine, and Foundry's
card becomes a window onto the engine's settings, like BookForge's.

**Routes, never fallbacks.** "Not powerful enough sends it to Anthropic" is a **route the
operator set** (or the app set on the operator's behalf through section 3.2), visible on the
page and in `/v1/capability`, chosen before any request. A request that names an upstream
whose key is not configured is refused by name. Nothing decides to go to the cloud because
something local failed.

**Settings live in the engine and nowhere else.** An app is a window onto them (section 5),
never a copy. BookForge's setup page writes the Anthropic key straight to the engine and reads
capability back; the app's own settings file never holds a key. Two apps and the page can all
edit the same engine because there is one store.

**What stays exactly as it is.** The lease, the four facts, the unload ruling. Job types and
packs (PHASE14). The operator page's existing panels. Each app's server REGISTRY (a list of
servers is per app). The pairing line. Coordinate-on-connect (PHASE14 4a).

## 1. Vocabulary

| word | meaning |
|---|---|
| **class** | a capability class from `crucible/capability.py`: `echo clean translate simplify analysis pages tts asr align rvc denoise`. Unchanged. |
| **route** | where a class's work runs on THIS server: `local` (the selected local model, as today) or an **upstream**. Only the four `llm` classes (`clean translate simplify analysis`) can route upstream in this phase; every other class is `local` and refuses anything else (`route_not_routable`). |
| **upstream** | an HTTP chat-completions service the server forwards to on the operator's account: `anthropic`, `openai`, `ollama`. Exactly these three names. An upstream is configured (has what it needs to be called) or not. |
| **upstream model** | a model id of the form `<upstream>/<model>`, e.g. `anthropic/claude-sonnet-5`, `openai/gpt-5`, `ollama/qwen3.5:9b`. The slash is what tells a chat request apart from a local model id; a local model id never contains `/` (checked at manifest load — `manifest_model_id_slash`). |
| **host mode** | a Crucible server running natively on Windows, `backend_kind = "llama-windows"`: the llm classes and `pages` served by `llama-server` children from GGUF weights (3.10), plus `echo`, the settings door and the upstream routes. The Python job types (`tts asr align rvc denoise`) need WSL2 and say so. |
| **the host** | `crucible host`, the Windows tray process (section 4). Not a server. |
| **pairing file** | the pairing line (PHASE13 2.1) written to a user-only file on the machine the server runs on, so an app on the same machine connects without anyone typing (section 3.6). |

## 2. Config: what `config.toml` gains

```toml
[routes]                 # absent key = "local"
translate = "anthropic/claude-sonnet-5"
simplify  = "anthropic/claude-sonnet-5"
# clean and analysis absent -> local

[upstreams.anthropic]
key = "sk-ant-…"          # file is mode 0600 already (PHASE13 1); a key here is no worse than the token
[upstreams.openai]
key = "sk-…"
[upstreams.ollama]
url = "http://192.168.68.20:11434"   # no key; ollama is reached by address
```

- `backend_kind` gains the value `"llama-windows"` (host mode). `crucible init --backend
  llama-windows` is legal only on win32 and is what `crucible host` runs; on linux/darwin it is
  refused (`backend_not_here`), and `cuda-linux`/`mlx-darwin` on win32 are refused the same way.
- A route's value is an upstream model id. `route = "local"` and an absent key mean the same
  thing; `"local"` is never written.
- The upstream `model` in a route is the operator's to choose; the server does not ship a
  cloud model list. `POST /v1/settings/upstreams/{name}/test` (3.2) returns what the upstream
  itself lists, and the page and the apps show THAT.
- `Config.adopt` (PHASE13 3.4) carries `routes` and `upstreams` so a `PUT /v1/settings` is
  live without a restart. Capability (`config.capability`) is RECOMPUTED in-process on every
  settings write that touches a route, and re-written to the config, because the route is part
  of the capability answer (3.3).

## 3. The server

### 3.1 `GET /v1/settings`

```json
{
  "routes": {
    "clean":     {"route": "local",    "model": "qwen3.5-9b"},
    "translate": {"route": "upstream", "model": "anthropic/claude-sonnet-5"},
    "simplify":  {"route": "upstream", "model": "anthropic/claude-sonnet-5"},
    "analysis":  {"route": "local",    "model": null}
  },
  "upstreams": {
    "anthropic": {"configured": true,  "key_hint": "…k3A9"},
    "openai":    {"configured": false, "key_hint": null},
    "ollama":    {"configured": true,  "url": "http://192.168.68.20:11434"}
  },
  "desktop_allowance_bytes": 3221225472,
  "backend_kind": "cuda-linux"
}
```

- `routes.<class>.model` for `local` is the class's SELECTED local model (the capability row's
  `selected`), or `null` when nothing fits. It is here so a window can show "translate: local,
  qwen3.8-27b-4bit" without a second call. It is **also `null` when this server has no
  capability record at all** (`GET /v1/capability` answers `503 capability_undecided`), and
  that is not the same statement wearing one spelling: this document must be readable before
  anything has probed the card, because writing the key is what an app does FIRST. A window
  that needs the two apart reads capability, which says which it is by name.
- A key is **write-only**. `key_hint` is the last four characters, enough to recognise which
  key is there and nothing else. There is no route that returns a key.

### 3.2 `PUT /v1/settings` — partial, validated, live

Body is any subset of:

```json
{
  "routes":   {"translate": "anthropic/claude-sonnet-5", "simplify": "local"},
  "upstreams": {"anthropic": {"key": "sk-ant-…"}, "ollama": {"url": "http://…"}, "openai": null},
  "desktop_allowance_bytes": 3221225472
}
```

- A route may be `"local"` or an upstream model id. Refusals, by name, each with `details`
  saying which field: `route_not_routable` (a non-llm class), `route_bad_model` (no slash, or
  an upstream name that is not one of the three), `route_upstream_unconfigured` (the named
  upstream has no key/url — configure it in the SAME request or before; the server never
  stores a route it cannot serve).
- `upstreams.<name>: null` removes that upstream. Removing one that a route names is refused
  `upstream_in_use` with the classes that name it — the caller re-routes first, in the same
  request if it likes. Order inside one request: upstreams are applied, then routes, then the
  whole is validated; a refusal applies nothing.
- **Two more refusals the patch door needs, added while building (2026-09-14).**
  `unknown_upstream` (400) — `upstreams` names something other than the three, and a typo
  must not be stored as a fourth upstream nothing can call. `upstream_bad_field` (400) — an
  upstream given the field it does not take (a `url` for `anthropic`, a `key` for `ollama`);
  each name takes exactly one, so a request carrying the other one is a request about a
  different upstream than the one it named. A body that is not an object, or a value of the
  wrong type, is the API's existing `invalid_request` (400) and is not given a name of its
  own.
- **A config edited by hand is refused at LOAD**, with the same three route names in the
  sentence (`route_not_routable`, `route_bad_model`, `route_upstream_unconfigured`), because
  a server that started with a route it cannot serve would refuse every request for that
  class with a sentence about the wrong thing.
- `POST /v1/settings/upstreams/{name}/test` with an optional body `{"key": "…"}` or
  `{"url": "…"}` (to test BEFORE saving) → `200 {"models": [...ids...]}` from the upstream's
  own model listing, unbilled; `502 upstream_unreachable` / `401 upstream_rejected` /
  `400 upstream_unconfigured` otherwise. Anthropic: `GET /v1/models` with `x-api-key`;
  OpenAI: `GET /v1/models` with bearer; Ollama: `GET /api/tags`.
- The response of `PUT` is the full `GET /v1/settings` document after the write, so a window
  never has to guess what took.
- **The `details` keys, pinned (Foundry package I reads exactly these):** every `PUT` refusal
  carries `details.field` (string — the dotted path that was refused, e.g. `"routes.translate"`
  or `"upstreams.anthropic.key"`); `upstream_in_use` additionally carries `details.classes`
  (string[] — the classes that name the upstream). `key_hint` is rendered VERBATIM by a client:
  the value INCLUDES the leading ellipsis (`"…k3A9"`, U+2026 then the last four characters), and
  a client prepends nothing.
- `X-Crucible-Act` (PHASE13) is honoured: a settings write is recorded in `/v1/activity`'s
  history with the act and the client agent, minus any key.

### 3.3 `GET /v1/capability` says the route

Every row gains `route`:

```json
{"capability": "translate", "enabled": true, "selected": "anthropic/claude-sonnet-5",
 "route": "upstream", "reason": "routed to anthropic; the local answer would be: qwen3.8-27b-4bit fits …"}
```

- For `route: "upstream"`, `enabled` is `true` iff the upstream is configured (which 3.2
  guarantees at write time, so it is always true — it is stated anyway so a reader of a
  capability document alone can trust it), and `selected` is the upstream model id, which is
  exactly the `model` an app sends to `/v1/openai/chat/completions`.
- `reason` keeps the LOCAL sentence after "the local answer would be:", so nothing is lost
  when the operator routes back.
- **How a client reads `route`, pinned for both apps (settled with Foundry 2026-09-14):** a
  capability document in which NO row carries `route` comes from a server that predates this
  phase, and every class on such a server IS local — that is a fact the document states, not
  a default the client fills, and both apps read it as `local` (BookForge's helper and
  Foundry's package K alike). A document in which SOME rows carry `route` and one does not is
  a defect and is refused by name (`capability_route_missing`, naming the row); a row whose
  `route` is present and not `local` | `upstream` is refused `capability_route_unknown`. API
  version stays 1: the field is additive.
- `job_types` (PHASE13 3.2a) is unchanged: it lists the installed job types, and an upstream
  route installs nothing.
- **In host mode** the llm classes and `pages` answer from the `llama-windows` fit table
  (3.10: GGUF sizes against free VRAM, or the CPU sentence). The Python-job classes (`tts asr
  align rvc denoise`) answer `enabled: false, route: "local"` with the one reason `this job
  type needs the WSL2 engine (vLLM/SGLang); install it from the console`, the same sentence
  for all five so an app shows it once. The four llm classes may still route upstream.

### 3.4 Chat completions forward to an upstream

`POST /v1/openai/chat/completions` (and `/openai/v1/…`) today: the model must be resident.
Now:

- `model` without `/` → the local path, unchanged in every respect.
- `model` = `<upstream>/<id>` → forwarded to that upstream with the operator's key, streaming
  and non-streaming both, the `inflight` record opened with `act` and `model` as today (so
  `/v1/activity` says "translating on anthropic"), no lease, no lane, the settlement untouched
  (nothing was on the card). Refusals: `upstream_unconfigured` (409), `upstream_rejected`
  (401 from the upstream → 502 here with the upstream's message), `upstream_unreachable` (502),
  `upstream_rate_limited` (429 passed through with `Retry-After` — the CALLER waits; the
  server never retries a billed request).
- Anthropic is not OpenAI-shaped. The server translates: `messages` with a leading `system`
  becomes `system`; `max_tokens` is required by Anthropic (use the request's, else the
  manifest-less default `4096` — stated in the response's `X-Crucible-Sampling` header as
  `max_tokens=4096 (upstream default)` so the audit line PHASE2 section 9 requires still
  says what filled the gap); `response_format` JSON schema → Anthropic tool-use with one forced
  tool (this is how `analysis` gets guided decoding upstream); streaming SSE is re-emitted in
  OpenAI chunk shape. OpenAI and Ollama are already OpenAI-shaped (`/v1/chat/completions`).
- `thinking: false` (BookForge sends it) is dropped for upstreams that do not know it, never
  forwarded blind. It travels in `chat_template_kwargs` (PHASE2-LLM.md section 9) and **none
  of the three upstreams reads that table**, so the whole table is what is dropped, for all
  three. The audit header says so: a fourth source value, **`dropped`**, meaning *the request
  stated it and this server did not forward it, because the upstream does not take it*.
  Saying `request` would claim a value reached the model and saying `engine` would hide that
  the caller asked.
- **The `max_tokens` audit value, spelled exactly.** `X-Crucible-Sampling` names a SOURCE per
  key, so Anthropic's filled-in default is the source string
  **`upstream default 4096`** — the sentence `max_tokens=4096 (upstream default)` written in
  the one place the header has room for it. The number is
  `crucible/upstreams.py`'s `ANTHROPIC_MAX_TOKENS_DEFAULT`, and it is in the string rather
  than only in the constant because a reader holding one response must be able to see what
  was sent without reading the server's source.
- **A `model` with a `/` whose prefix is not one of the three upstreams is refused
  `route_bad_model` (400)** — the same name section 3.2 gives the same malformation, because
  it is the same mistake arriving at a different door, and two names for it would be two
  vocabularies for one fact.
- `GET /v1/openai/models` lists local models as today PLUS, for each configured upstream, the
  routed upstream models (only the ones a route names — not the upstream's whole catalog,
  which is `test`'s job). An upstream row is
  `{"id": "<upstream>/<model>", "object": "model", "owned_by": "<upstream>",
  "upstream": "<upstream>", "routed_for": [classes]}` and carries **no** `created`,
  `max_model_len` or `defaults`: this server did not load it, does not know its context and
  has no manifest for it. A client that sizes `max_tokens` against `max_model_len` already
  skips the clamp when the field is absent (CLIENT-SURFACES.md section 6.1), which is the
  correct behaviour here and not a gap.
- A lease on an upstream model (`POST /v1/models/{id}/lease`) is refused `lease_not_needed`
  with the sentence "an upstream model is never resident; send the chat". Same for
  `{"type": "load-model"}` naming one.

### 3.5 Host mode — the `llama-windows` backend

- `crucible serve` no longer refuses win32. `main()`'s refusal narrows to: on win32 every verb
  runs and `backend_kind` must be `"llama-windows"`; any other kind on win32, or
  `llama-windows` off win32, is `backend_not_here`, because a backend runs where its engine
  runs and nowhere else.
- `llama-windows` is a backend in the full sense `mlx-darwin` is: the residency, the lease,
  the four facts and `settle.py`'s unload ruling all apply; a resident model IS a running
  `llama-server` child; `load-model` spawns it, unload kills it; one child at a time. The
  catalog for this backend lists GGUF variants (3.10 says which); `install` installs the
  engine (the `engine` subject, 3.10) — there is no Python env pack for it, so `crucible
  install llm` / `install pages` on this backend fetch the engine and nothing else, and
  `install tts` (etc.) is refused `needs_wsl` with the sentence from 3.3.
- Host mode has everything else this phase and the previous ones give a server: the
  settings door, the page, `/v1/setup`, `/v1/catalog`, tasks, `/v1/accelerator` (nvidia-smi,
  or `cpu` with the machine's RAM as the figure), `/v1/activity`, chat completions to a
  resident child or to an upstream.
  - **AND THE LOAD GUARD, which is the same probe (T7, first Windows run, 2026-09-14).**
    `crucible/accelerator.py`'s `read_state` was the last hand-written list of backends on
    the load path, so `load-model dots-ocr` on a staged `llama-windows` server came back
    `409 accelerator_unreadable: 'llama-windows' is not a Crucible backend`. It reads the
    card the Windows way now: `nvidia-smi` for the compute apps and the free/total figures
    when this host has an NVIDIA driver, `GlobalMemoryStatusEx`'s available/total RAM when
    it has none. Two deliberate differences from the other two backends, each for a reason
    stated in that file: **detection tolerates a broken driver and the guard does not** (a
    card Crucible cannot read is `accelerator_unreadable`, never a fall back to "here is
    all of your RAM"), and the **`unattributed_bytes` check does not run here** — it exists
    for WSL2's blind compute-app list, while a Windows desktop always holds VRAM that
    belongs to no compute app, so running it would refuse every load on a machine that is
    merely displaying a desktop.
  - **AND THE GUARD ASKS FOR ROOM, NOT FOR SOLITUDE (T7, the SECOND Windows run,
    2026-09-14, on the live card).** Reading the card the Windows way was half the job;
    the other half is that the card's *meaning* is different here, and the first fix
    kept the cuda-linux one. The staged server refused `load-model dots-ocr` with
    `409 accelerator_busy: the accelerator is held by pid 1460 ([Insufficient
    Permissions], memory not reported); pid 6028 (…CrossDeviceResume.exe); pid 11208
    (C:\WINDOWS\explorer.exe); pid 12852 (…SearchHost.exe); pid 12880
    (…StartMenuExperienceHost.exe)` — the compositor and the shell, i.e. Windows
    drawing a desktop. **On `llama-windows` the card is SHARED BY DESIGN**: nvidia-smi
    on Windows names every windowed process on the GPU, several with `[Insufficient
    Permissions]` and no memory figure at all, so "a foreign compute app is on the
    card" is the normal state of the machine and, taken as a refusal, makes every load
    on this backend impossible. The cuda-linux meaning (a foreign GPU process inside
    WSL2 is a trainer or another engine — refuse) does not transfer.
    - **The rule.** The guard asks whether there is **ROOM**: `free VRAM >=
      memory_bytes_estimate`, from nvidia-smi `memory.free`. A shortfall is refused by
      the name that shortfall has always had, **`insufficient_memory`** (the name
      `capability.py` already points at this module for; a second spelling of one
      refusal would be a fact with two owners), naming both figures. The foreign
      processes are **REPORTED** — `details.processes` on that refusal, and
      `/v1/accelerator` lists them regardless — and are **never the reason**.
    - **The one holder is an engine of ours that outlived its run.** A `llama-server`
      Crucible did not start is a previous run's child still on the card and is
      `accelerator_busy`, named. It is found **by image name** (`llama-server`, with
      any directory and any `.exe` stripped, case-folded), never by "any pid that is
      not ours": the pid of a crashed run is not knowable and the image is.
    - **`cuda-linux` and `mlx-darwin` are untouched.** This is a rule about a Windows
      desktop, not a softening of section 4; `tests/test_accelerator.py` pins the same
      process list still refusing `accelerator_busy` on `cuda-linux`.
- `crucible doctor` in host mode prints `backend: llama-windows on windows/x86_64 — llama.cpp
  <tag> (cuda-12.4 | cpu)`, then the engine line, then the upstream lines.
- Nothing in host mode is a stopgap for WSL, and WSL is the better engine (section 0's block:
  parallel page reading under vLLM, the faster text path, the five Python job types). When WSL
  arrives, the host (section 4) moves the config — token, routes, upstreams — into the guest
  and STOPS the Windows server. The token survives the move, so every app that paired stays
  paired.
- **Crucible owns WHERE the weights are, and a model is never stored twice on one machine
  (Owen, 2026-09-14: "crucible will manage the location of the models, too … crucible can
  move the models to WSL instead of storing it in both windows and in wsl").** On Windows the
  server's home is `%LOCALAPPDATA%\Crucible\` and every subject lives under it; in the guest it
  is `~/.crucible` as today. The two engines read DIFFERENT FILES for the same subject —
  `llama-windows` runs GGUF, `cuda-linux` runs the safetensors under vLLM/SGLang — so a file
  cannot be carried across; "move" is a migration STEP the host runs (4.3), by name, per
  installed subject: the guest pulls its own form of each subject that was installed on
  Windows (the catalog's `installed` list is the input, so nothing the operator had is
  forgotten), and only when the guest reports that subject `installed: true` is the Windows
  copy deleted. Throughout, one catalog, one `installed` answer per subject; an app never
  sees a path. Weights Windows never had (voices, whisper, aligners, RVC, denoise) are
  pulled by the module's coordinate step as before. A migration interrupted mid-way leaves
  BOTH copies of the unfinished subject and resumes on the next host start; it never deletes
  first.

### 3.5a A subject can be REMOVED — `DELETE /v1/catalog/{kind}/{id}`

The weights rule (3.5, last bullet) needs a door the host can call to delete a Windows copy
once the guest has its own, and the host must never reach into `weights.py`'s layout from
outside. So the catalog gains one verb: `DELETE /v1/catalog/{kind}/{id}` removes an installed
subject's files (every file the manifest names for THIS backend, and the subject's directory
if it is then empty) and answers `204`. Refusals by name: `subject_unknown` (404),
`subject_not_installed` (409), `subject_in_use` (409, with `details.who` — resident, leased,
or named by a running task), `subject_remove_failed` (500, with the path). Same auth as every
private route; recorded in `/v1/activity` with the act. `crucible remove <kind> <id>` is the
CLI spelling, refusing identically. The page's Catalog panel gains a Remove on installed rows.
An app never calls it on a user's behalf without saying so on screen (BookForge and Foundry:
not in this phase — the host is the only caller for now).

### 3.6 The pairing file

`crucible init` and `crucible service install` write the pairing line to a user-only file
beside the config: `<CRUCIBLE_HOME>/pairing` (mode 0600 on linux/darwin; on Windows the file
is written by the host, section 4.3, with an ACL of the current user only). One line, the
`127.0.0.1` pairing line, trailing newline. `crucible token --url` prints the same. An app on
the same machine reads it (5.1) and never asks a person to type a token. Rotating the token
rewrites it.

**The file's line is the LOOPBACK one, always, whatever the server is bound to**, and
`crucible token --url` therefore prints it first and then the reachable lines
(`reachable_urls`, 3.1) — which on a `127.0.0.1` bind are the same one line, printed once.
The file answers one question, *"an app on THIS machine wants in"*, and the answer to that
is never a LAN address: a wildcard-bound server has no loopback entry in `reachable_urls`
at all, so a file built from that list would hand a local app an address that depends on
which interface the OS listed first.

**Where it is, per platform (pinned 2026-09-14 for Foundry's package J and the host agent):**

| platform | path | written by |
|---|---|---|
| linux (incl. inside the WSL guest), darwin | `<CRUCIBLE_HOME>/pairing` = `~/.crucible/pairing` | `crucible init` / `service install` / token rotation |
| Windows | `%LOCALAPPDATA%\Crucible\pairing` (beside `host\`, 4.4) | the host (4.1/4.3): the host-mode server's line while that runs, then the GUEST's line after the migrate step — same token, host, port |

`CRUCIBLE_HOME` set in the environment overrides the directory on every platform (on Windows
the file is then `%CRUCIBLE_HOME%\pairing`). The Windows file is the host's COPY of the
guest's line, because the guest's own home is inside the distro where no Windows app looks.
An absent file means "no local server" — a fact the app shows, not a fallback it fills. Until
a host exists on a machine (Owen's PC today), an app's "read `config.toml` through `wsl.exe`"
door is how the WSL server gets registered, and that door is deleted when the host lands.

**AMENDED 2026-09-14: `crucible serve` writes it too**, when it is absent or when it does
not match the config (name, host, port, token). `init` and `service install` were the only
writers, so a server that EXISTED before this phase had no pairing file and an app on its
own machine was told there was no engine there — measured on the Mac Studio right after its
upgrade. A file that exists and disagrees is worse than none, because it points an app at a
door with the wrong key, so the comparison is on the LINE itself, which is exactly the four
facts an app needs. The line written is the CONFIG's, never this run's `--host`/`--port`
overrides: 3.6's file answers *"an app on THIS machine wants in"*, and a developer running
`crucible serve --port 7999` for an afternoon must not repoint every app on the box at a
server that is about to stop. A write that fails is PRINTED and does not stop the server —
the file changes whether a person has to type a token, not whether the engine runs.

**Where a READER looks, pinned** (added 2026-09-14 by the BookForge build of 5.1, which had to
open the file before the host existed to write it — the preamble's rule: a name this file did
not have is added here first). Two locations, in this order, and neither is a fallback for the
other — the first is an override the operator set and the second is the only default there is:

1. `$CRUCIBLE_HOME/pairing`, when `CRUCIBLE_HOME` is set and non-empty. Same env var
   `config.py crucible_home()` already honours, so a second server on a second home is found
   by the app the same way the CLI finds it.
2. Otherwise, per platform:
   - **win32:** `%LOCALAPPDATA%\Crucible\pairing`. NOT `~/.crucible`: on Windows the server is
     the WSL guest's or the host-mode child's, and in both cases the thing that writes a
     WINDOWS-side pairing file is `crucible host` (4.3), whose own per-machine root is already
     `%LOCALAPPDATA%\Crucible\` — `wsl\`, `downloads\` (`sdk/bootstrap/src/distro.ts`) and
     `host\` (4.4) are all under it, so the pairing file is its fourth member and the host runs
     with `CRUCIBLE_HOME` set to that directory. `LOCALAPPDATA` is read from the environment and
     never assembled from a username, exactly as `distro.ts` does it; unset is refused by name,
     not guessed.
   - **linux/darwin:** `~/.crucible/pairing`, which is `crucible_home()`'s default.

A reader that finds no file answers `null` — "no engine on this machine" is a FACT, and the
caller's next line is "install one" or "paste a connect code" (5.1). It is never an error and
never a retry.

### 3.7 The page gains a Settings panel

`crucible/ui/` gains **Settings**, between Job types and Connect an app: one row per llm
class with a select (`local — <selected local model or "nothing fits">` / each configured
upstream's routed model / "an upstream model…" free text), three upstream cards (Anthropic key
+ Test + Save, OpenAI key + Test + Save, Ollama url + Test + Save; a configured card shows the
hint and a Remove), the desktop allowance. Every control is a `PUT /v1/settings`; the panel
re-reads the document the PUT returns. No local state. In host mode the page's Status panel
says "llama-windows (llama.cpp <tag>, cuda | cpu) — install the WSL2 engine for faster page
reading and text, and for TTS, ASR, alignment, RVC and denoise", and Job types / Catalog say
what 3.5 says.

### 3.8 The SDK (`@crucible/client`)

`settings()`, `putSettings(patch)`, `testUpstream(name, probe?)` — which does NOT throw on the
three test refusals (a settings window wants one shape to draw): it returns
`{ok: true, models: string[]} | {ok: false, code: 'upstream_unreachable' | 'upstream_rejected' |
'upstream_unconfigured', message: string}`, and throws only for what every call throws
(auth, version, transport) — `readPairingFile(home?)`
(node only; returns the parsed pairing or `null` when absent — `null` is a fact here, not a
fallback: the caller's next line is "install one" or "paste one"). `CapabilityRow.route`,
`SettingsDocument`, `SettingsPatch`, `UpstreamName`, the refusal names above as constants.
Tests for each.

### 3.9 Tests

pytest: settings GET/PUT with every refusal; key never in any response, log line or activity
record; recompute-on-write; chat forwarding for each of the three upstreams against a fake
upstream (streaming and not; Anthropic shape translation incl. system, max_tokens audit and
tool-forced JSON); host mode's answers for every class above; `backend_not_here`; `needs_wsl`; the pairing
file. sdk/ts: the four methods and the types. The count goes UP from 1234 and is reported.

### 3.10 `llama-windows`: page reading on a Windows box without WSL (RULED 2026-09-14, evening)

**Owen (relayed by Foundry pc, verbatim):** *"i think that should go through crucible as well.
at a bare minimum, a crucible server will run on absolutely anything. it's an extension of the
foundry app … if it uses the GPU (as dots does), it should probably be crucible-side … crucible
can decide if the user's system is even capable of running it … it should be a pass-through
thin client UI for the crucible engine."*

**The line is MODEL INFERENCE vs DETERMINISTIC WORK, not GPU vs CPU.** Anything that runs a
model is the engine's, even on a CPU-only laptop (allowed, slow, and the capability row says
so). Rasterising, parsing, EPUB assembly stay in the app. (Foundry's NLI analysis worker falls
on the engine's side of that line too — its own class, a later phase, noted in 6.)

**What it is (as amended by section 0's block).** `llama-windows` is the Windows BACKEND:
`backend_kind = "llama-windows"`, a `llama-server` child per resident model, spawned by
`load-model`, leased and settled like any resident, killed on unload. It serves the four llm
classes from GGUF text models and `pages` from the dots.ocr GGUF pair — dots first because
Foundry's launcher is the spec, the text models by the same mechanism in the same build. It
is NOT a Python env and NOT a pack (the engine is a zip from llama.cpp's release, the `engine`
subject below), and it never serves `tts/asr/align/rvc/denoise` (those are Python job types;
WSL). The spec is Foundry's working launcher, handed over at
`C:\tmp\foundry-page-reader-spec\` (README first — eight load-bearing facts;
`page-reader.ts`'s header argues every constant).

**The catalog for `llama-windows`** (its own block per manifest, like `mlx-darwin`'s): `dots-ocr`
→ the GGUF pair in fact 2; `qwen3.5-9b` → a Q8_0 GGUF of the same weights the `cuda-linux`
row names; `qwen3.8-27b` → a Q4_K_M GGUF (the 27B on a 24 GB card only fits at 4-bit, which is
the same binary-per-server sentence `translate` already carries). Each row records the HF repo,
revision, file name(s), bytes and the floor, read from the repo once and recorded — the agent
that lands this picks the repos, states them in section 7, and the manifests are the one
owner. Fit is the row's floor against free VRAM; with no NVIDIA the row is `enabled: true` with
the reason "cpu build — slow; runs on this machine's CPU" (Owen: a Crucible server runs on
anything). A model whose GGUF is not published is simply absent from this backend's block —
never a guessed row.

**What Crucible decides that the app used to** (README's last section): whether this machine
can bear it (the capability row: NVIDIA + free VRAM ≥ the row's floor → `enabled: true,
selected: dots-ocr, reason: "cuda build, <n> GiB free"`; no NVIDIA → `enabled: true` with the
reason "cpu build — slow; the model runs on this machine's CPU" — Owen: a Crucible server runs
on anything; nothing else refuses it), which build to fetch, where the files live, when they
go.

**Facts the port MUST keep, and the two it changes:**

1. **The build is PINNED, never listed at runtime.** Foundry read the llama.cpp release
   listing and fell back to a pinned tag; here the tag is ONE constant (`LLAMA_CPP_RELEASE`,
   beside `STANDALONE_PYTHON` in `envpack.py`, with the sha256 of each asset read from that
   release's checksums or measured once and recorded — the doc says which). Windows + NVIDIA:
   `llama-<tag>-bin-win-cuda-12.4-x64.zip` PLUS `cudart-llama-bin-win-cuda-12.4-x64.zip`
   into one directory (the server does not start without the cudart DLLs); Windows without
   NVIDIA: the CPU build. Fetched by a `pull` task of a new subject kind `engine` (`{kind:
   "engine", id: "llama-cpp"}`) so the page's Tasks panel shows it like weights; refusals
   `engine_download_failed`, `engine_sha_mismatch`, by name.
2. **The weights** are the existing `pages` catalog subject (`dots-ocr` → `ggml-org/
   dots.ocr-GGUF` @ `2c093a32ca360a396bc6d87d60408636130b9d9b`, `dots.ocr-Q8_0.gguf` +
   `mmproj-dots.ocr-Q8_0.gguf`, 3.24 GB, ~4.74 GB needed) pulled through the catalog like any
   subject. A pull that has the text tower and not the mmproj is INCOMPLETE and the subject
   says `installed: false` — the mmproj is not optional.

   **THE PROJECTOR MUST BE A CONVERSION THIS BUILD CAN READ, and that is a fact about the
   pin, not about llama.cpp.** For `PROJECTOR_TYPE_DOTS_OCR`, `clip.cpp` reads
   `clip.vision.projector.scale_factor` as MANDATORY (`get_u32(KEY_PROJ_SCALE_FACTOR,
   hparams.n_merge)` — no `required = false`; `tools/mtmd/clip.cpp:1542` at `b10970`, the
   same at `b10950`), so a projector converted for a fork that used the older
   `clip.vision.spatial_merge_size` spelling is refused at `clip_init` and the server exits
   before it serves. That is what 7.6's third defect was, and it is why this pin is
   `ggml-org`'s own conversion rather than a third party's. **The llama.cpp pin did NOT move
   and must not be moved for this**: no upstream tag loads the other file.
3. **The spawn**, verbatim from `ensurePageReader()`: `-m <gguf> --mmproj <mmproj> -c 16384
   --parallel 1` on a port Crucible chooses (an ephemeral loopback port, not 8000). `-c 16384`
   because a page at the app's dpi is up to ~8k image tokens plus the answer.
4. **Readiness:** `GET /v1/models` on the child until it lists a model whose name ends in
   `dots.ocr`; 5-minute timeout; stderr filtered; the fatal lines (OOM, missing DLL, bad
   GGUF) end the wait early with `pages_engine_failed` and the line.
5. **CHANGED — no adopting.** Foundry adopted a server already on port 8000 and never stopped
   it. Crucible never adopts a process it did not start (that is a fact with two owners); the
   child is on a port Crucible chose, so there is nothing to adopt. If the chosen port is
   somehow taken the spawn is refused `port_in_use` by name.
6. **Unload = kill the child**, 30 s graceful then kill. Keep-warm is the engine's existing
   unload ruling (`settle.py`'s unload-every-time applies: the child stops when the last
   `pages` job of a run settles), not an app timer.
7. **The request** is the existing `pages` job wire (PHASE4's `vlm-pages`): the server
   forwards each page as OpenAI chat completions with one `image_url` data-URI PNG and dots's
   prompt, `max_tokens` from the model row, temperature 0; the answer (JSON, sometimes fenced)
   is returned as the job's artifact exactly as the cuda-linux path returns it — the app's
   parser (`parseDotsPage`) does not change. `confirmServedModel` is kept: a child whose
   `/v1/models` name does not match is `pages_engine_wrong_model`.
8. **UNMEASURED, and the first thing measured on a card:** whether the Q8 GGUF answers in the
   parser's dialect exactly (the MLX and vLLM builds do); seconds per page on CPU and on a
   small card. Both recorded in section 7 by whoever runs it first; the doc says
   "unmeasured" until then, never a guessed number.
9. **CHANGED — the load guard asks for ROOM, not for solitude** (found by T7 on the live
   card, 2026-09-14; the full record and the refusal names are in 3.5). Foundry never
   asked the question at all — it started a server on port 8000 beside whatever else was
   on the card, because a Windows desktop always shares its GPU. Crucible does ask, and
   on this backend the question is `free VRAM >= memory_bytes_estimate`, refused
   `insufficient_memory` with both figures; the compositor, the shell, the browser and
   the `[Insufficient Permissions]` rows are reported in `details.processes` and are
   never the reason. The single exception is a `llama-server` this Crucible did not
   start — an engine child left behind by a crashed run — which is `accelerator_busy`,
   matched **by image name** because the pid of that run is not knowable.

**Exit for Foundry's package L:** `pages` answers `enabled: true` from a `llama-windows` server
on a clean no-WSL Windows box and a real page comes back parsed; for text, `clean` answers the
same way from the 9B GGUF or an upstream. Then `page-reader.ts` and every
"can this machine do it" line in Foundry go.

## 4. The host is the ORCHESTRATOR (PHASE17) — `crucible orchestrator` on Windows

> **Renamed 2026-09-15 by `docs/PHASE17-ORCHESTRATOR.md`.** Everything this section
> describes is the orchestrator half of the relation Phase 17 names: it manages exactly
> one engine, serves ZERO job types, and never carries a byte of anybody's data. The
> verb is `crucible orchestrator`, with **`crucible host` kept as an alias** (PHASE17
> section 7 says why, and why the Python package stays `crucible/host/`). The word
> "host" below should be read as "orchestrator" throughout; PHASE17 adds the claim
> (2.1), the `role`/`engine` fields on its door's `/v1/info` (3.2) and the
> `engine-restart` task (4.2) to what is written here.

A Windows-only verb in the SAME package (nothing else to version), started at login, shown in
the notification area. It is the front door Owen asked for. It owns exactly four things:

### 4.1 Presence

- **Which server this machine runs**, decided from the wsl-states table (PHASE14 4c): the
  Crucible distro exists → the WSL server; otherwise → the host-mode server, run as a child of
  the host process. Never both, and the host says which on its menu.
- **Start at login.** A shortcut in the user's Startup folder
  (`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Crucible.lnk` → `crucible host`).
  No admin, no Task Scheduler, no service: a per-user login item is what a tray program is.
- **Boot the guest.** With the distro present, the host runs `wsl -d crucible --exec true` at
  start (that boots the distro; systemd + linger then start the enabled unit), waits for
  `GET /v1/ping` on `127.0.0.1:7100`, and if the unit is not up after 30 s runs the two
  recipes found on 2026-09-14 in order and by name: `systemctl --user start crucible`; if the
  user bus is absent, `systemctl restart user@1000` as root (`wsl -d crucible -u root`). If
  neither brings it up the tray shows "engine did not start — open the log", never a spinner.
- **The LAN door.** The WSL server binds loopback and reaches the LAN only through a
  Windows-side forward, so the host owns it: when the WSL engine is active the host keeps a
  `netsh interface portproxy` (or WSL mirrored networking where present — the host detects
  which, by name) from the machine's LAN addresses on `7100` to the guest, and removes it when
  the engine stops. This is what makes `/v1/setup`'s LAN pairing lines true on Windows without
  the user touching netsh; it needs admin once, prompted by name with the sentence that says
  why.
- **Watch.** `GET /v1/ping` every 15 s. Down → the same recipe once, then the tray state
  "engine stopped" with a Start item. It never loops on restart; the systemd unit's own
  `Restart=` handles crashes. (RULING RECORDED HERE: `service.py` moves to `Restart=always`
  with `RestartSec=2`; a clean SIGTERM leaving the engine down was today's defect, and "stop"
  from the host is `systemctl --user stop`, which `Restart=always` respects — the unit is
  stopped, not exited.)

**The names this build gives 4.1.** The host's own root is `%LOCALAPPDATA%\Crucible\` and the
host runs with `CRUCIBLE_HOME` set to it (3.6), so the host-mode config is
`%LOCALAPPDATA%\Crucible\config.toml` and the log is `%LOCALAPPDATA%\Crucible\host.log` — one
file, appended, rolled at 2 MiB to `host.log.1`. A tray that starts at every login for a year
must not grow without bound, and one previous file is enough to read a failure that happened
before the last restart.

Presence is a PAIR, `(distro, engine)`, and every value has a name:

| `distro` | means |
|---|---|
| `present` | `wsl -l -v` lists `crucible`: this machine runs the WSL server |
| `absent` | it does not: this machine runs the host-mode child |
| `unknown` | `wsl.exe` could not be asked — missing, or it errored. The menu says so and still offers the install; it never reads this as `absent`, because "install the engine" on a machine whose WSL merely failed to answer would import a second distro. |

| `engine` | means | title line |
|---|---|---|
| `starting` | the boot, or a recovery recipe, is in flight | `Crucible — starting…` |
| `running` | `GET /v1/ping` answered | `Crucible — running (WSL)` / `Crucible — running (llama-windows)` (section 0's amendment: host mode is a BACKEND, and the title names it) |
| `stopped` | the ping failed and this down-edge's recovery is spent | `Crucible — stopped` |
| `failed` | both recipes ran and neither brought it up | `Crucible — engine did not start — open the log` |
| `installing` | the 4.3 sequence is running | `Crucible — installing…` |

The recovery recipes are named. With the distro: `user-unit-start`
(`systemctl --user start crucible` inside it), then `user-bus-restart`
(`systemctl restart user@1000` as root, through `wsl -d crucible -u root`), in that order, at
most once per down-edge. In host mode there is one: `host-mode-respawn`, which starts
`crucible serve` as a child again (the `llama-windows` server, 3.5 — the host starts and
stops that process and knows nothing about what it runs). `BOOT_WAIT_SECONDS = 30` and `WATCH_SECONDS = 15` are the
two numbers 4.1 states, and they are constants with those names.

**The Startup verbs.** `crucible host --install-startup` writes the shortcut and prints its
path; `crucible host --remove-startup` deletes it and says whether there was one. Both exit
without starting a tray, and both are the ONE owner of that file — `install.ps1` calls the
first rather than writing a `.lnk` of its own. The shortcut is written by
`crucible/host/startup.py` through a PowerShell `WScript.Shell` one-liner (no pywin32, no new
dependency) and its target is `pythonw.exe -m crucible.cli host`, not the `.cmd` (4.4): a
`.cmd` opens a console window, and a tray program has none.

#### 4.1a Presence is a TRIPLE — added 2026-09-15, by the first real run

`crucible host` ran on Owen's PC for the first time on 2026-09-15 and the pair was not
enough to describe that machine. It has a Crucible engine inside **`Ubuntu`**, installed by
hand long before any of this existed, so both halves of the pair are true at once and they
contradict each other: `wsl -l -v` lists no distro NAMED `crucible`, so `distro` is honestly
`absent`, and `GET /v1/ping` on 7100 honestly answers `200`. With only `(distro, engine)`
the host reads the first fact as *"this machine runs the host-mode child"*, spawns the
`llama-windows` server onto a port another Crucible already holds, pings, gets the OTHER
server's answer and reports `running` about a child that lost the bind. **Section 0 is one
server per machine and the host was the thing making it false.**

So presence is `(distro, engine, owner)`, and `owner` has three values and an absence:

| `owner` | means | what the host may do to it |
|---|---|---|
| `wsl-unit` | the guest's unit, in the distro Crucible owns | boot, restart, stop — all of 4.1 |
| `host-child` | the `llama-windows` server this process spawned | start, respawn, stop; Quit takes it down |
| `found` | an engine that was already answering when the host started | **watch it, and nothing else** |
| `none` | there is no engine | start one, if this machine's answer says to |

**A `found` engine is watched and never acted on.** `restart-engine` and `stop-engine` are
disabled, `install-engine` is ABSENT even though the distro is (`install-engine` is offered
when `distro` is absent or unknown — but importing a distro onto a machine that already has
an engine is the one mistake here that pressing the button again cannot undo), a down-edge
runs NO recovery recipe at all, and `quit`'s label is `Quit (the engine keeps running)`
whatever the distro probe said. The refusal lives in the menu model AND at the click: a
disabled item is a drawing, and the thing that must not happen is the act.

**The ping comes FIRST on a machine with no Crucible distro.** The guard is a LOOK, not a
refusal — a machine where nothing answers still gets the host-mode child. With the distro
PRESENT the order is unchanged, because `boot()`'s `wsl --exec true` is a no-op on a distro
that is already up.

**A `found` engine can be RAISED to `wsl-unit` by CONSENT, and only by that —
added 2026-09-15, PHASE17 2.5.** The rule above is right about a distro nobody
has spoken about and useless about the one this machine's engine actually lives
in: `Ubuntu` is `found` for the single reason that Crucible did not import it,
which is all the orchestrator knows about a stranger's distro too. So a person
may name one, once, in the orchestrator's own config:

```toml
# %LOCALAPPDATA%\Crucible\config.toml
[orchestrator]
distro = "Ubuntu"
```

With it, `probe_distro()` looks for THAT name (so it reads `present`), and a
running engine there becomes `wsl-unit` **if
`env XDG_RUNTIME_DIR=/run/user/<uid> systemctl --user is-enabled crucible.service`
answers inside it** (the uid READ with `id -u`, never assumed — and see the
correction to the measurement below) — the claim then lands and `engine-restart`
goes through the unit. If the unit cannot be read the owner stays `found` and the
log says what systemctl said, because consent is permission and not evidence.
Without the setting nothing changes: `found`, watched, never claimed.

**What consent does NOT widen is `user-bus-restart`, and that is not a matter of
permission.** `systemctl restart user@1000` kills every process uid 1000 owns in
that distro — the paragraph below this one is the reason, and it is about what is
RUNNING in somebody's Linux, not about who may manage it. It stays refused by
name (`orchestrator_recipe_not_ours`) in any distro Crucible did not import,
consented or not, at the point the recipe would have run. The predicate is the
rootfs, never the flag.

**The Windows pairing file is the guest's line COPIED, and 3.6 always said so** — *"the
Windows file is the host's COPY of the guest's line"*. The code did not do it: it composed
a line from the host's own `config.toml`, which on a machine whose engine is a guest's is a
DIFFERENT token at the guest's address. 3.6 calls a file that exists and disagrees worse
than none, and it is exactly that: every app that reads it gets 401. Where the line comes
from is decided by the owner and there is no fallback between the two sources —
`wsl-unit`/`found` read it out of the distro (`cat "${CRUCIBLE_HOME:-$HOME/.crucible}/pairing"`
through `--exec`), `host-child` composes it from the host's own config, and when neither can
answer **no file is written**. The distro a `found` engine is in is discovered by asking each
**already-running** distro for its pairing line and keeping the first whose authority is
`127.0.0.1:7100`; only running ones, because `wsl -d <name> --exec` BOOTS a stopped distro and
the host boots no VM it does not own.

**The host HOLDS the distro open** — 7b.4c's owed sentence, now written. A WSL distro
terminates seconds after the last `wsl.exe` session ends, whatever its units say and whatever
`loginctl enable-linger` says; `Restart=always` cannot keep a VM alive because the VM is not
something the guest can hold. 4.1's boot (`wsl -d crucible --exec true`) therefore started the
thing the host is watching and then let it go. The host now keeps one
`wsl.exe -d <distro> --exec sleep infinity` child for as long as a WSL engine is the engine,
takes it again on the watch tick after it dies, and releases it on Quit.

**`user-bus-restart` is safe only in the distro Crucible OWNS, and that is now a rule.**
`systemctl restart user@1000` kills every process uid 1000 owns in that distro. In the
`crucible` rootfs that is Crucible's own processes and the cost is the restart. In a distro
a person also uses it is everything they are running — on the night this was written, a
five-thousand-step LoRA trainer. A `found` engine gets no recipes at all, and this is the
second reason why.

**What the user bus actually did, measured 2026-09-15 (read-only), correcting 7b.4c.**
Linger is now ON (`Linger=yes`) and `/run/user/1000` EXISTS and holds `dbus-1/` and `pulse/`
— and there is still **no `bus` socket in it**, so every door onto the user manager is shut:

```
wsl -d Ubuntu --exec systemctl --user is-active crucible.service
  -> Failed to connect to bus: No such file or directory
wsl -d Ubuntu --exec bash -c 'XDG_RUNTIME_DIR=/run/user/1000 systemctl --user is-active …'
  -> Failed to connect to bus: No such file or directory      (XDG_RUNTIME_DIR was ALREADY set)
wsl -d Ubuntu -u root --exec systemctl --user -M telltale@ is-active crucible.service
  -> Failed to retrieve unit state: Transport endpoint is not connected
```

> **CORRECTED 2026-09-15, 07:34-07:35, and the correction is the fix.** The
> three failures below were read as *the socket is missing*. They are not that,
> or not only that. `systemctl restart user@1000` as root created
> `/run/user/1000/bus` — and a `wsl.exe --exec` session STILL could not reach
> it, because such a session gets no logind seat and therefore no
> `XDG_RUNTIME_DIR`, and systemctl looks for the bus at `$XDG_RUNTIME_DIR/bus`
> and nowhere else. **A missing socket and a missing variable print the
> identical sentence**, which is why re-reading the message could never have
> separated them. Measured on the same distro in the same minute:
> `XDG_RUNTIME_DIR=/run/user/1000 systemctl --user is-active crucible.service`
> answered `active`. Every user-manager call the orchestrator makes now carries
> that prefix, with the uid read by `id -u` (PHASE17 2.5). The second line
> below says *"XDG_RUNTIME_DIR was ALREADY set"* — it was set in the shell the
> measurement was typed in, and not in the `--exec` process that ran systemctl.

So the root-side `-M <user>@` door — the one 4.1 was owed as a written recipe — **does not
work here either**, and the reason is not the missing `XDG_RUNTIME_DIR` 7b.4c blamed: a
`systemd --user` manager for uid 1000 IS running (pid 220) and simply never got a bus,
because it was started before there was a logind session to give it one. The only thing that
would repair it is `systemctl restart user@1000`, which is `user-bus-restart` — and on THIS
distro that is the command the rule above forbids. `RECIPES` stands as 4.1 wrote it; what is
added is that neither recipe can be reached from Windows on a distro in this state, and that
the host must not try on one it does not own.

### 4.2 The menu

`Crucible — running (WSL)` / `running (llama-windows)` / `stopped` / `installing…` as the title
line, then: **Open console** (the pairing line's URL with `#token=`, PHASE13 5.3 — the same
hardened window rule applies to a browser: it is the default browser), **Install the
WSL2 engine (faster pages and text; TTS, ASR…)…** (only when the distro is absent — runs section 4.3), **Restart
engine**, **Stop engine**, **Open log**, **Quit** (stops the host; the WSL server keeps running
because it is systemd's; in host mode the child server stops with it, and the menu says so).

**The menu is a pure function** — `menu_model(distro, engine)` in `crucible/host/menu.py` —
and every item carries an id that the click handler and the tests name it by: `open-console`,
`install-engine`, `restart-engine`, `stop-engine`, `open-log`, `quit`. `install-engine` is
ABSENT rather than disabled unless `distro` is `absent` or `unknown` (4.2 says "only when the
distro is absent"); every other item is always present and carries `enabled`. `open-console`
is enabled only while `engine` is `running`, because the URL it opens comes from the pairing
file and there is nothing to open when nothing answers. `quit`'s LABEL is what says which
Crucible this is: `Quit (the engine keeps running)` with the WSL server, `Quit (stops the
engine)` in host mode.

### 4.3 Install and migrate — the wsl-states table, driven

The existing bootstrap step list (PHASE14 7b.1) and state table (4c) move from "an app's
package" to "the host's job". `@crucible/bootstrap` keeps its `install()` for the apps that
still call it, and its implementation becomes: is the host installed? no → download and run
`install.ps1`; yes → tell the host to install (a named-pipe/loopback request to the host,
`http://127.0.0.1:7101/install`, host-only bearer = the same token) and stream its events. ONE
implementation of the sequence, the host's; the bootstrap is its client.

The host's install: the state table's answer for the machine, each state's sentence shown in a
window (not a console), the ones that need admin (`wsl --install`, the feature enable) run
through a UAC prompt by name with the sentence that explains why, the reboot states say
"reboot, then Crucible continues" and the Startup item makes that true; import the distro;
fetch the server pack in the guest; **move the host-mode config into the guest** (token,
routes, upstreams — `crucible init` in the guest with `--config-from` a file the host wrote,
0600) so the token survives; install the module's job types; service install; linger;
capability write; MIGRATE THE WEIGHTS per subject (3.5: pull in the guest, then delete on Windows, never the reverse); stop the host-mode child; switch the pairing file to the guest's line (the
same line — same token, same host, same port). Every step is one of the existing named steps
or one of the state table's named states; the host adds no new sentence of its own.

**The door, named.** `POST /install` on `127.0.0.1:7101`, bearer = the engine token (the
host-mode config's, and the guest's after the migrate — the same token either way, 3.5). The
response is newline-delimited JSON, one event per line:
`{"event": "state"|"step"|"line"|"done"|"error", …}`. A connection that closes before `done`
is a failure the client names rather than a success it assumes. Refusals, by name:
`host_install_running` (409 — a second POST while one is in flight; there is one install on a
machine and the second caller waits), `host_no_token` (503 — the host has no config yet, which
is only true before its first host-mode `init`), `host_unauthorized` (401), and every state
code from the 4c table verbatim when the machine cannot be carried further. Bootstrap's side
adds two: `host_not_installed` (there is no `%LOCALAPPDATA%\Crucible\host\` on this machine —
the answer is `install.ps1`) and `host_unreachable` (the host pack is installed and the door
did not answer).

**Who calls the door.** Two callers and no third. (1) The Windows SERVER, when the page posts
`POST /v1/tasks {"type": "engine", "target": "wsl"}` (4.7): the server relays this door's
events under its own task id, so the events are shaped like `crucible/tasks.py`'s — a step
name, a state, `bytes_done`/`bytes_total` while something downloads, and the 4c sentence on a
failure. (2) `@crucible/bootstrap`'s `install()`, directly, on a machine that has no server at
all yet — the very first install, before there is a page to open. The shape is the same for
both, because there is one sequence.

**The 4c table crosses into Python by GENERATION, not by a second copy.** `crucible host` is
Python and the table is `sdk/bootstrap/src/wsl-states.ts`, so its DATA — the code, the probe
argv, the sentence, the action — is EMITTED into `crucible/host/wsl_states.py` by the same
`scripts/gen-install-scripts.ts` that writes `install.ps1`, and `npm run gen:install --
--check` refuses a drift exactly as it does for the two scripts. What is spelled twice is only
what `steps.ts` already spells twice for the same reason: the PREDICATES (`means`), which are
code rather than data. `crucible/host/wslstate.py` holds one per code and a test asserts the
two sets are equal — the "tied by a check instead of by an import" seam `envpack.SMOKE_IMPORT`
and `cli.INSTALLABLE_JOB_TYPES` already use.

**The host has NO window of its own (amended by 4.7).** Its UI is the tray menu and the
operator PAGE. Because `llama-windows` runs on any Windows machine, the host starts the
Windows server within seconds of install and opens the page; the WSL install is then the
page's engine switch (4.7), shown as a task in the page's Tasks panel like a pull. The
states that need a reboot are answered by the task saying "reboot, then Crucible continues"
and the Startup item resuming the task and reopening the page. There is no tkinter, no
second progress UI, no second owner of the sequence.

**The migrate step, named.** `crucible init --config-from <file>` is the flag 4.3 asks for:
the file is a TOML document the host wrote at 0600 and deletes afterwards, and `init` takes
exactly three things out of it — `auth.token`, `[routes]` and `[upstreams]` — and nothing
else. The host, the port, the name, the backend and the job flags belong to the machine being
initialised, not to the one being left. A file without `auth.token` is refused
`config_from_no_token`; a file that is not TOML is `config_from_unreadable`.

### 4.4 Packaging

- A Windows host pack: `crucible-env-host-llama-windows-<version>.tar.zst`, built by the same
  `crucible envpack build host` on a `windows-latest` runner: python-build-standalone
  `x86_64-pc-windows-msvc-install_only` (pinned in `STANDALONE_PYTHON` with its
  SHA256SUMS digest like the two others) + the server package + `pystray` + `pillow` (the
  tray). Unpacked to `%LOCALAPPDATA%\Crucible\host\`. It is the ONLY Windows-native Python
  Crucible ever ships, and since section 0's amendment it carries BOTH halves of what runs on
  Windows: the tray, and the `llama-windows` server the tray starts.
- `install.ps1` (generated, PHASE14 7b.2) becomes: download the host pack for this version,
  verify, unpack, write the Startup shortcut, start `crucible host`, and STOP — the host takes
  it from there (4.3) and shows the WSL steps in its own window. `install.sh` is unchanged
  (linux/darwin have no host).
- The release gains the host pack in `envpacks.json` (backend `llama-windows`) and CI gains the
  `windows-latest` job. release.sh's asset list is updated (seven assets now: the installers
  are unchanged in count).
- **Mac:** no host. launchd already supervises; the pairing file (3.6) is what an app reads.
  A menu-bar item is out of this phase.

**CORRECTION TO THE INTERPRETER PIN, read from the release rather than from memory.** There is
no `x86_64-pc-windows-msvc-shared-install_only` asset on python-build-standalone **20260901**.
The `-shared` infix is retired: the word "shared" appears ZERO times in that release's
`SHA256SUMS` (against 90 occurrences of "static"), and the Windows `install_only` build IS the
shared one. The pin is therefore
`cpython-3.11.16+20260901-x86_64-pc-windows-msvc-install_only.tar.gz`, sha256
`6be524fa6752af802146a4adc7d098565425b0b1c166e19a5a7a4c8cccb86bf6`, read on 2026-09-14 from
`https://github.com/astral-sh/python-build-standalone/releases/download/20260901/SHA256SUMS`.
Same release and same CPython (3.11.16) as the two backends, which is the property that
mattered.

**The pack's layout is Windows's, and its entry point is a `.cmd`.** python-build-standalone's
Windows tree is `python.exe` / `pythonw.exe` / `Scripts\` / `Lib\`, not `bin/`, so `envpack`
asks `pack_python(root, backend)` for the interpreter instead of assuming `bin/python`. And
the relocation defect 7.2a solved for POSIX has no POSIX answer here: pip writes
`Scripts\<name>.exe` launchers with the building interpreter's absolute path embedded in the
binary, which a move breaks and which no shebang rewrite can reach. So the Windows build
writes `crucible.cmd` beside `python.exe`:

```bat
@echo off
"%~dp0python.exe" -m crucible.cli %*
```

`%~dp0` is the Windows spelling of the same idea as `$(dirname -- "$0")` — the interpreter is
found from the script's own location — so `relocate_console_scripts()` on this backend writes
one `.cmd` per console script the wheel declares, and the pack keeps the `.exe` launchers only
as the dead weight pip left (nothing in Crucible calls them, and `pack_smoke` proves the
`.cmd`, moved).

- **The pack backend is `llama-windows`, the same word a config on this machine uses**
  (3.5) — CORRECTED 2026-09-14, after section 0's amendment. The pack's name is `host` and its
  archive is what `pack_filename`'s rule yields with no special case:
  `crucible-env-host-llama-windows-<version>.tar.zst`. An earlier draft of this section wrote
  the backend `host-windows` and the asset `crucible-env-host-windows-…`, which is the rule
  with the repeated word quietly elided — two spellings of one name, and a 404 the first time
  `install.ps1` composed the URL from the rule instead of from the prose. `host-windows` also
  gave one machine two names, which is the R1 shape: a Windows Crucible IS `llama-windows`,
  here and in `backend_kind` and in `crucible doctor`, and the pack table's key is the same
  word because pip resolved those wheels for that machine.
- `crucible envpack build host` is refused `pack_not_buildable_here` off win32, and on win32
  every OTHER pack name is refused by the same function and the same rule.
- **The pairing file's ACL** (3.6, "an ACL of the current user only") is set with
  `icacls <file> /inheritance:r /grant:r <user>:(R,W)`, `<user>` read from `%USERNAME%` in the
  environment and never assembled. `icacls` ships with Windows, so this adds no dependency. A
  failure is `pairing_acl_failed` and the file is DELETED rather than left readable by
  everybody with a bearer token in it.

### 4.5 Tests

sdk/bootstrap: the shrunk `install()` (host present / absent) with a fake host. pytest: `crucible
host` is refused off win32 (`host_windows_only`); the menu model as a pure function of
(distro state, ping state); the Startup shortcut path; the migrate step's `--config-from`; the
pairing-file switch. The tray itself is exercised by hand on Owen's PC and the doc records
what was seen.

Added by the build, each because it pins something that would otherwise drift silently: the
`(distro, engine)` table above is walked exhaustively and every cell's title and item set is
asserted; the generated `crucible/host/wsl_states.py` is checked against `wsl-states.ts` by
`gen:install --check` and its codes against `wslstate.py`'s predicates by pytest; the
`icacls` argv; the `.cmd` shim's text and the fact that a MOVED pack still runs it; the
`llama-windows` row in the pack table and its refusal off win32; and the loopback door's four
refusals. Windows-only code paths take the platform as an argument so they run on both — the
suite runs in WSL (`pytest`) and must not be a suite that skips its subject.

### 4.6 The Mac — no host, and what "works out of the box" is measured against

> **BUILT 2026-09-14, and section 7c is what came of it.** `align` and `asr`
> are served, with every estimate MEASURED on that machine rather than
> declared; `pages` got the structural half — `BACKEND_ENGINES` is one engine
> per (backend, class family) — and NOT the manifest block, because mlx-vlm's
> own HTTP server was measured and does not put the image into the prompt.
> `envs/rvc/mlx-darwin.txt`'s owed freeze and `doctor`'s owed PATH line landed
> with them. Read 7c for the numbers; what follows is the decision they came
> from.

**Owen, 2026-09-14:** *"we'll have to make sure crucible works on mac as well. it wouldnt need a
wsl sidecar for mac obviously. it would just function out of the box with mlx-audio and
everything we have configured for mac bookforge."*

The Mac needs no host: `mlx-darwin` is a native backend, launchd supervises it, `install.sh`
installs it, the pairing file (3.6) connects a local app. What "out of the box" is measured
against is the Mac Studio as read on 2026-09-14 (crucible 0.6.0, M1 Ultra 64 GB, macOS
26.3.1) and the read-only audit that followed (`docs/MAC-PARITY-AUDIT-2026-09-14.md`):

- **Served today, healthy:** `tts` (Higgs via mlx-audio 0.4.8, all seven voices renderable),
  `llm` (9B and 27B fit), `rvc`, `denoise`. The `job_type_not_ready: tts: no ffmpeg` line in
  the first read was a NON-LOGIN SSH SHELL's PATH, not the service's — the plist carries
  `/opt/homebrew/bin` and the running process has it; `doctor` reports the shell it runs in.
  Not a defect. (One improvement it exposes, owed: `doctor` should name the SERVICE's PATH
  beside the shell's, since it wrote the plist and can read it back.) The "stale orpheus env"
  was two doctor rows for ONE directory (`jobenv.tts_env()` resolves every engine name to the
  one mlx-darwin env); nothing to delete, the row is already gone at HEAD.
- **Not served, in ascending cost:**
  - `align` — NEARLY FREE. Qwen3-ForcedAligner is plain torch and torch-on-Metal is already
    how `rvc` runs on this backend; BookForge's Mac env for it is built, packed, published
    (`qwen-align-env-macos-arm64.tar.gz`) and MEASURED (97x realtime warm on MPS, bf16,
    2026-09-08); the weights are the identical repo+revision the cuda row pins. Owed: a
    `mlx-darwin` block with its OWN measured `memory_bytes_estimate`, the recipe as that
    env's freeze, the CI row, and the timestamp comparison `envs/align/mlx-darwin.md` asks for.
  - `asr` — a SECOND ENGINE (`mlx-whisper`, CTranslate2 has no Metal), a second worker, NEW
    model ids (different weights at one id would be a lie): `mlx-community/whisper-*-mlx`,
    seven repos measured in the audit (large-v3 2.87 GiB, turbo 1.50 GiB). Nothing on the Mac
    runs mlx-whisper today and BookForge's Mac ASR was CPU faster-whisper, so this is the gap
    with the least standing work behind it.
  - `pages` — **BUILT AND THEN NOT SHIPPED (7c, measured 2026-09-14):** the engine class and the
    per-family table landed, but `models/dots-ocr.toml` has NO mlx-darwin block on purpose:
    mlx-vlm's own HTTP server (0.6.10 and 0.7.1 alike) does not put the image into dots.ocr's
    prompt — in-process `generate` returns 5 correct blocks in the dialect at 16.3 s/page,
    peak 4.93 GB; the same weights over its server answer one `Picture` box in 0.72 s
    (`prompt_tokens=216` vs 3,464 ids in-process). Shipping the block would light `pages: yes`
    on every Mac and answer every page with one box. **RULING OWED (recommended: yes):**
    Crucible serves dots on the Mac through its OWN small server over `mlx_vlm.generate` —
    the same lifecycle as every engine child, the request builder shared with the two other
    backends — rather than waiting on an upstream fix. The original finding follows.
  - `pages` (original) — the model, package and dialect are PROVEN on that machine (Foundry's Mac route:
    `mlx-vlm 0.6.10` + `mlx-community/dots.ocr-4bit` @ `4ab989e4…`, 3.30 GiB, 0.80% CER,
    ~27 s/page), but `mlx-darwin`'s one engine is `mlx-lm`, a text server that cannot read an
    image. **DECIDED here: `BACKEND_ENGINES` stops being one engine per backend and becomes
    one engine per (backend, class-family)** — `cuda-linux` already uses vLLM for both text and
    pages; `mlx-darwin` gets `mlx-vlm`'s own server as the `pages` engine beside `mlx-lm` for
    text, one env holding both (mlx-vlm's pins are satisfied by `envs/llm/mlx-darwin.txt`).
    The lease and the four facts are per resident, not per engine, so nothing about
    arbitration changes; what changes is the residency knowing which server class to start.
- **Upgrade off conda** (Phase 14 deletes the conda requirement): the Mac's three job envs are
  venvs PARENTED on the conda env's python, so removing conda first would kill them. Order is
  the whole risk and the audit's §3 is the checklist: release exists → service stop →
  `install.sh` (server pack; init skipped, token kept) → `install llm|tts|rvc` from packs →
  `service install` FROM A LOGIN SHELL → capability write → doctor → only then delete the
  conda env. Weights, voices, config and token survive untouched. `~/.crucible/hf-token.txt`
  is a stray Crucible never read (`[hf] token` in config.toml is the door) — move or delete.
- Also owed from the same read: `envs/rvc/mlx-darwin.txt`'s freeze (the real install
  happened; its substitutions verified), and the `tts` zero-shot clip on the MLX arm, still
  never exercised.

### 4.7 The engine switch — Windows ⇄ WSL2 is a control on the page, and a task

**Owen, 2026-09-14:** *"crucible will need a wsl install configuration page in its installer, so
the user can configure the wsl side, and the user can flip from windows to wsl. maybe there's
a switch or something on the UI that determines if the crucible server is operating from WSL
or from windows. if the user flips it from windows to WSL then the wsl configuration
activates and it installs all the models and dependencies necessary in wsl, and deletes them
in windows. itll be more than a switch i suppose, since itll be a compute heavy configuration
update to download the necessary models and wheels."*

- The page's Status panel on a Windows machine shows **Engine: Windows (llama.cpp)** or
  **Engine: WSL2 (vLLM/SGLang)** and a control **Move to WSL2…** (only while Windows). The
  page never shows the Python job types as installable on the Windows engine; it shows them
  under the control, as what the move brings.
- Pressing it is `POST /v1/tasks {"type": "engine", "target": "wsl"}` — a task like a pull or an
  install: one at a time (`409 task_busy`), events on `/v1/tasks/{id}/events`, cancellable
  between steps, visible in the Tasks panel with bytes where a step downloads. The Windows
  server does not run it: it hands it to the HOST's loopback door (4.3) and relays the host's
  events under the task id, because only the host can run `wsl.exe`, prompt UAC and survive
  the reboot. A Windows server that was not started by a host (a developer running `crucible
  serve` by hand) refuses `engine_move_needs_host`.

  **`CRUCIBLE_HOST_DOOR`, named here because `crucible/host/` did not have a name for
  it (added 2026-09-14 by the build of this section).** The host spawns `crucible serve`
  as a child (4.1) and the child inherited nothing that said a host was there. So the
  host sets ONE variable on the child's environment — `CRUCIBLE_HOST_DOOR=http://
  127.0.0.1:7101`, its own door's base URL — and **its presence IS the fact**: a server
  with it set was started by a host, and a server without it refuses
  `engine_move_needs_host`. A probe of 7101 is the wrong question twice over, because
  something that is not a host can answer it and a host restarting its own door is still
  the host. **The TOKEN is not carried.** The door's bearer is the ENGINE's token, which
  the server already holds in its own config; a second copy in an environment variable
  would be a secret with two owners and one more place for it to be stale.
  **A DOOR THAT IS SET IS A TASK THAT IS ACCEPTED (T10, first Windows run, 2026-09-14).**
  With `CRUCIBLE_HOST_DOOR` set there is nothing to refuse at submit time — the server
  hands the move to the host and relays — so `POST /v1/tasks` answers `202 {"task_id":
  …}` and **every failure of the door is named in the task's own events**, never in the
  POST body. A client that reads the POST body for the refusal will read a success; it
  reads `/v1/tasks/{id}` or `/v1/tasks/{id}/events` instead. `engine_move_needs_host` is
  the POST-time refusal and it means exactly one thing: the variable is not set.

  **One door, one set of names.** The three other endings are named by
  `@crucible/bootstrap`'s `requestHostInstall` (4.3), which calls the SAME door, and the
  server uses THOSE names rather than a second set of its own:

  | ending | code | where |
  | --- | --- | --- |
  | `$CRUCIBLE_HOST_DOOR` is not set | `engine_move_needs_host` | 409 at the POST |
  | connection refused / timeout | `host_unreachable` | the task's `failed` event |
  | the stream ends with no terminal event | `host_install_failed` | the task's `failed` event |
  | the host sent a `failed` event | the code IT carries, verbatim | the task's `failed` event |

  An HTTP refusal whose body this server cannot read stays `engine_move_needs_host`, for
  its own reason: something answered 7101 and it is not behaving like a host, which is
  the same fact as there being no host there.

- The task's steps are the state table + the step list + the migrate step of 4.3, in order:
  detect the WSL state → the named answer for it (feature enable / `wsl --install` / reboot /
  kernel update / import the Crucible distro) → server pack in the guest → move the config →
  install the job types the apps' modules asked for (the coordinate records the server keeps
  from every connected app say which; nothing is guessed) → pull each installed subject's
  guest form → delete the Windows copies (3.5) → service install, linger, capability → stop
  the Windows server → the guest answers `:7100` with the same token → `done`. The page,
  which lost its server for a few seconds at the switch-over, re-reads `/v1/info` and shows
  **Engine: WSL2**.
- A failed step fails the task by name with the state table's sentence and leaves the Windows
  engine running and untouched — the move is not partial from the app's point of view until
  the final switch-over, and the weights rule (3.5) already says nothing is deleted before
  the guest has it.
- **The reverse (WSL2 → Windows) is not in this phase.** The control shows the one forward
  move; removing the WSL engine is an explicit operator act, written in section 6.
- **"There will never, ever be a local gpu configured"** — the same ruling from BookForge's side
  (5.3): every GPU slot the queue shows is a Crucible endpoint, and the engine question is
  asked and answered HERE, once, not in an app.

## 5. The apps

### 5.1 Connect: three ways, in this order, all automatic

1. **The pairing file** on this machine (`readPairingFile()`), when the app has no `local`
   entry yet → the registry gains `local` from it, coordinate runs (PHASE14 4a). No typing.
2. **A pasted pairing line** (PHASE13 5.1) for a server elsewhere.
3. **Get one on this machine** → `@crucible/bootstrap install()` (4.3's client). On Windows
   that is the host; on Mac it is `install.sh` as today.

The app's setup step (PHASE13 5.5) probes in that order and shows ONE face.

### 5.2 Settings are the engine's; the app draws a window

Each app's AI/engine settings section and its wizard's AI step draw **the engine's settings
document** (3.1) for the selected server and write through with `PUT /v1/settings` (3.2). The
app holds nothing: no key, no route, no model list. Concretely, the wizard's AI step reads
capability; for each llm class that is `enabled: false` locally it says the class's reason and
offers "run it through Anthropic / OpenAI / an Ollama server instead" — entering a key calls
`test`, then one `PUT` that configures the upstream AND sets the route, then capability is
re-read and the step shows the new answer. The settings section shows the same rows plus the
three upstream cards, all write-through. A key field is empty on every draw (write-only) with
the hint beside it.

**What "the things BookForge does affect crucible directly" means on the wire:** every control
in these sections is a request to the engine, and its result is the engine's answer re-read.
There is no Save button that writes an app file and syncs later.

### 5.3 What each app DELETES

**BookForge:** `AIProvider` shrinks to `'crucible'` (and `'local'` until the legacy layer goes
with the in-app pass — that deletion is already scheduled and this phase does not move it).
`claude` and `openai` branches in `ai-bridge.ts`, `text-ai.ts`, `translation-bridge.ts`,
`book-analysis.ts`, `queue-steps/ai-provider.ts`, `web-fetch-bridge.ts`; `ollama-capabilities.ts`
and every Ollama door that the legacy layer does not own; the "cloud keys → Foundry's card"
read added this morning; any cloud model list. The cleanup/OCR/translation/simplify/analysis
doors send `capability.selected` as the model to the registry's server and nothing else.
`shared/queue/slot-sets.ts`: a row whose class routes `upstream` on its server takes that
server's **`[cloud]` lane** (one per server, width 2), no GPU slot, no lease; `runVenueOfRow`
names it. **Every GPU slot the queue shows is a Crucible endpoint (Owen, 2026-09-14: "there
will never, ever be a local gpu configured. there simply wont be an outlet for it").** The
slot sets are: one `[gpu]` set per registered Crucible server, its `[cloud]` lane, and
`local-work [cpu][cpu]` (CPU slots stay local, ruling dd50e8c3). The legacy set with its own
GPU slot is deleted with the legacy spawn layer after Owen's in-app pass — that deletion is
already scheduled and this phase names the end state, not a new date. The keeper suite pins every deleted door by name (as `test-no-e2a-doors.js` does).

**Foundry:** `cloud-providers.ts`, the cloud card, `ComputeSlotKind = 'cloud'`, the cloud
placement in `crucible-dispatch.ts`, `FOUNDRY_ENDPOINT_HEADERS` composed from an app-held key.
The engine (the CLI's `--endpoint`) is pointed at the Crucible server's `/openai/v1` for every
text act, with `capability.selected` as the model — the same as a local act, which is the
point. Hosted, the settings window in 5.2 is Foundry's card drawn from BookForge's registry
selection (`FoundryHost.servers()`), so the two apps show one engine's settings.

### 5.3a A module names CLASSES, and the SERVER resolves them for its backend (found 2026-09-14)

Measured by Foundry against the Mac: foundry.module.json carries `qwen3.8-27b-4bit` and
`dots-ocr` as resolved ids because gen-modules.py resolves a `[[needs]] class` to ONE id at
generation time — the cuda-linux answer. Posted to the Mac, validate_module refuses the whole
module `unknown_subject` (dots-ocr has no mlx-darwin block) and `qwen3.8-27b-4bit` is not what
the Mac's capability selected (`qwen3.8-27b`). The generator was a second owner of a decision
that is the server's (PHASE9: the capability record is the one place a class is resolved).

Ruling: the module carries `needs` as CLASSES on the wire, unresolved:
`{"name":"foundry","version":"…","job_types":[{"type":"llm"}],"needs":[{"class":"clean"},{"class":"translate"},{"class":"pages"}],"subjects":[{"kind":"voice","id":"higgs-default"}]}`.
`subjects` keep explicit ids ONLY for genuine app choices (a voice, a whisper size, the rvc
base). gen-modules.py checks every explicit id exists in SOME backend's block and every class
exists, writes the JSON, resolves nothing. The server's `module` task resolves each class
through ITS capability record (`selected`) and pulls that subject; a class this backend has
DISABLED is not a refusal — the task result carries
`unmet: [{"class":"pages","reason":"<the capability row's reason>"}]` and the app shows "not
on this engine". An explicit subject the backend cannot hold is still `unknown_subject`. SDK:
`ModuleResult.unmet`. Both apps re-vendor their module after the gen changes; coordinate shows
`unmet` beside the pulls it made.

### 5.4 The single re-vendor (BookForge ← Foundry)

Moves again: it now targets the Foundry sha AFTER their 5.3 lands, and carries the items the
rollout plan already lists (`RunOptions.waitFor`, `hosted_placement_not_vendored` deletion,
`slots?()` removal) plus the cloud card's replacement.

## 8. THE BUTTON — the test run, staged so the card is held only for the measurements

**Owen, 2026-09-14:** *"get everything ready so we can just hit a button and have the tests run, and
when it's fully ready to test, ill release the card and let you know when its ready."*

One script, `scripts/testrun-phase15.sh`, run from Git Bash on the PC with no arguments. It
runs the stages below IN ORDER, stops at the first failure with the stage's name and the
failing command's output, and writes `C:	mp\phase15-testrun\<timestamp>
eport.md` as it goes
so a stopped run still says what passed. Every stage prints its wall-clock seconds. Nothing
in it asks a question; anything it needs (a key, a PDF page) is a file it looks for by name
and reports SKIPPED by name when absent — never silently.

**Staged BEFORE the button (no card, no VM memory beyond a single suite):**

- S1 the live WSL clone `/home/telltale/crucible` fast-forwarded to the branch HEAD,
  `pip install -e .` in its env, the service restarted, `/v1/info` answering the new build
  (`pages_engine`, routes present in `/v1/capability`).
- S2 the Windows side staged WITHOUT installing anything system-wide: the host pack built
  (`crucible envpack build host`), unpacked under `C:	mp\phase15-testrun\host\`, a temp
  `CRUCIBLE_HOME=C:	mp\phase15-testrun\home` initialised with `--backend llama-windows` on
  port **7101** (7100 is the WSL server's); the `engine` subject and the `dots-ocr` GGUF pair
  and the `qwen3.5-9b` GGUF PULLED into it (network and disk only — a pull never touches the
  card); the Startup shortcut NOT written; no portproxy.
- S3 BookForge's `dist/` current (`tsc` for electron, `ng build` for the renderer) and the
  keepers green; Foundry's main at the sha they name, its own build green (theirs).
- S4 the two pytest suites' inputs verified: the lock free, no trainer (Owen has released the
  card by then — the script still checks and refuses by name).

**The button, in order:**

| # | stage | needs the card | pass = |
|---|---|---|---|
| T1 | `sdk/bootstrap npm test`, `sdk/ts npm test` | no | 0 failing |
| T2 | pytest, main branch, under the lock | no (VM memory) | 0 failed, count reported |
| T3 | pytest, mac branch worktree (until merged), under the lock | no | 0 failed; `test_lineup.py`'s 4 worktree-environmental failures listed by name and nothing else |
| T4 | BookForge keepers (`node tools/run-keepers.js`) | no | all suites green |
| T5 | upstream route end to end: if `C:	mp\phase15-testrunnthropic-key.txt` exists, `PUT /v1/settings` on the WSL server configures anthropic + routes `translate`, one chat completion through it, then the route is put back to local and the key REMOVED (`upstreams.anthropic: null`) — the key never stays on a test run | no | a completion came back; settings restored; SKIPPED by name if no key file |
| T6 | dots under vLLM in WSL: submit a `vlm-pages` job for `C:	mp\phase15-testrun\page.pdf` (SKIPPED by name if absent) to the WSL server; record seconds/page and that the artifact parses in the {bbox,category,text} dialect | YES | one page parsed, figure recorded |
| T7 | llama-windows: start the staged Windows server on 7101, `load-model dots-ocr`, the same page through the same job wire; the artifact byte-shape identical to T6's; then `load-model qwen3.5-9b` and one cleanup chunk; unload; stop the server | YES | both answered; seconds/page and seconds/chunk recorded; `/v1/activity` shows the acts |
| T8 | the remove door: `DELETE /v1/catalog/model/qwen3.5-9b` on the staged Windows server, then `installed: false` in its catalog | no | 204 then false |
| T9 | Mac: `ssh mac` — the Mac server's `/v1/capability` after ITS upgrade (7c's M-steps; the deploy is a `git pull` of `/Volumes/Callisto/Projects/crucible`, the editable checkout — never a wheel) shows `align` and `asr` enabled; one align job, one asr job (the Mac's card, allowed). `pages` is NOT expected: see 4.6's mlx-vlm finding | Mac's card | two artifacts, figures recorded |
| T10 | the engine task on this PC: `POST /v1/tasks {"type":"engine","target":"wsl"}` on the staged Windows server with the host's door running — on a machine that already has the distro this exercises detection + migrate-config + migrate-weights (the subject from T7 pulled in the guest, then deleted on Windows) and refuses the steps that do not apply, by name | no | task `done`; the Windows copy gone, the guest's present |

Then Owen's in-app pass (BookForge, Foundry) — his, not the script's. The report's last
section is the list of every figure that was "unmeasured" in sections 7/7b/7c and is now a
number, ready to be pasted back into those sections.

## 6. Not in this phase, written so it is not forgotten

- Removing the WSL engine / moving back to Windows (`engine` task with `target: "windows"`): an explicit operator act, not the switch.
- A Mac menu-bar host. launchd covers supervision; the pairing file covers connect.
- Per-request cost or token accounting for upstreams. `/v1/activity` records the act and the
  model; a usage figure is the upstream's dashboard's until somebody asks for it here.
- Foundry's NLI analysis worker as its own engine class (3.10's line puts it on the engine's side).
- Routing any non-llm class upstream (a cloud TTS, a cloud ASR). `route_not_routable` is the
  door, and it opens when there is a reason.
- Deleting BookForge's legacy local spawn layer (`'local'`, the WSL bridges, ollama's remaining
  doors). That is the in-app-pass deletion already on the rollout plan.

## 7. What was built (server half, 2026-09-14)

Written by the agent that landed sections 2, 3.1–3.4, 3.6–3.8 and the foundation of
3.10. It says what was measured, what was decided, what deviates from the sections above
and why, and — for the half that is not built — every fact the next build would otherwise
derive a second time.

### 7.1 Built and tested

| section | what landed | where |
|---|---|---|
| 2 | `[routes]`, `[upstreams.*]`, `Config.adopt` carrying both, `manifest_model_id_slash` | `crucible/config.py`, `crucible/upstreams.py`, `crucible/manifests.py` |
| 3.1–3.2 | `GET`/`PUT /v1/settings`, `POST /v1/settings/upstreams/{name}/test` | `crucible/settings.py`, `crucible/api.py` |
| 3.3 | `route` on every capability row; the local sentence kept after routing | `crucible/capability.py`, `crucible/api.py` |
| 3.4 | chat forwarding to all three upstreams, streaming and not | `crucible/upstreams.py`, `crucible/api.py` |
| 3.6 | `<CRUCIBLE_HOME>/pairing`, written by `init` and `service install` | `crucible/config.py`, `crucible/cli.py` |
| 3.7 | the page's Settings panel | `crucible/ui/` |
| 3.8 | `settings()`, `putSettings()`, `testUpstream()`, `readPairingFile()` | `sdk/ts/src/` |
| 0, 3.3, 3.5, 3.10 | the `llama-windows` BACKEND, its catalog rows and its capability answer | `crucible/backend.py`, `models/`, `crucible/capability.py` |

**Refusals added, by name.** `route_not_routable`, `route_bad_model`,
`route_upstream_unconfigured`, `upstream_in_use`, `unknown_upstream`,
`upstream_bad_field`, `upstream_unconfigured`, `upstream_rejected`,
`upstream_unreachable`, `upstream_rate_limited`, `lease_not_needed`,
`manifest_model_id_slash`. Every `PUT /v1/settings` refusal carries `details.field`, the
dotted path; `upstream_in_use` also carries `details.classes`.

**The exact settings document the fake server produces** (`tests/test_settings_api.py`,
a 24 GB card with a 3 GiB allowance, nothing configured):

```json
{
  "routes": {
    "clean":     {"route": "local", "model": "qwen3.5-9b"},
    "translate": {"route": "local", "model": "qwen3.8-27b-4bit"},
    "simplify":  {"route": "local", "model": "qwen3.8-27b-4bit"},
    "analysis":  {"route": "local", "model": "qwen3.8-27b-4bit"}
  },
  "upstreams": {
    "anthropic": {"configured": false, "key_hint": null},
    "openai":    {"configured": false, "key_hint": null},
    "ollama":    {"configured": false, "url": null}
  },
  "desktop_allowance_bytes": 3221225472,
  "backend_kind": "cuda-linux"
}
```

### 7.2 Decisions this build made, and the deviations

- **`X-Crucible-Sampling` gains two source values**, because the header has to stay honest
  across a hop with no manifest: `dropped` (the request stated `thinking` and this server
  did not forward it, since none of the three upstreams reads `chat_template_kwargs`) and
  `upstream default 4096` (Anthropic requires `max_tokens` and the request stated none).
  The number is in the string so a reader holding one response can see what was sent.
- **Every non-2xx from an upstream except 429 is `upstream_rejected` (502)**, with the
  provider's own message and `details.upstream_status`. Section 3.4 spells out the 401;
  the rest — a model the account cannot reach, an overloaded region — are the same event
  from this server's side, and multiplying the names would lose nothing and cost a client
  a table. 429 is passed through with the upstream's own `Retry-After`, never retried.
- **Capability is recomputed on an ALLOWANCE change too**, not only on a route change
  (section 2 names the route). The allowance is the other input `decide()` reads, and a
  write that changed it and left the record alone would leave the rows describing the old
  reserve.
- **`settings.recomputed_capability` reads the card's numbers from the RECORD** and the
  GPU vendor from the live backend. A settings write is not the door that re-measures a
  card (`crucible capability --write` is, and it needs the host); the vendor is a live
  host fact, and `cmd_serve` already refuses to start when the detected backend and the
  recorded one disagree, so the two cannot drift under a running server.
- **The fit rule on `llama-windows` is AVAILABLE memory, not free VRAM.** Section 3.10
  says free; `crucible/capability.py`'s rule 2 says total-less-the-allowance, with the
  reason that a capability decided on a transient is switched off by an open browser.
  One rule on three backends; the runtime guard still owns "is there room right now" and
  refuses `insufficient_memory` with the measured figure at load.
- **A cardless Windows box still lights its rows**, with `cpu build — slow; the model
  runs on this machine's CPU` appended and the pool called `system memory` rather than
  `card`. The arithmetic is unchanged: RAM is a real limit, and a 27B in 8 GB does not
  run slowly, it thrashes.
- **`qwen3.8-27b-4bit`'s llama-windows file is `UD-Q4_K_M`, not `Q4_K_M`**, because
  `unsloth/Qwen3.8-27B-GGUF` publishes no plain one — its plain Q4s are `Q4_0` and
  `Q4_1`, both worse. A row is never guessed; this is the file that exists.
- **`readPairingFile()` is async**, although it reads one short line. A static
  `import … from 'node:fs'` in a module `index.ts` re-exports would put fs into the graph
  of `import {CrucibleClient}`, which the SDK README already forbids for
  `writeArtifactsTo`; the specifier is assembled at run time exactly as that one is, and
  a dynamic import is a promise.
- **`tests/test_ui_mount.py`'s "reaches for another host" guard was narrowed for
  `app.js`**, not dropped: since 3.7 the page draws a field for an Ollama address and
  `http://host:11434` in its placeholder is an example shown to a person. Every absolute
  URL in the script must now sit on a `placeholder:` line; the HTML and the CSS keep the
  absolute rule, and what the page actually fetches is pinned twice over by the two
  neighbouring tests.

### 7.3 Measured, and unmeasured

- **MEASURED on the card, 2026-09-15.** The button ran. Everything this section listed
  as needing a card has a number, and **3.10's fact 8 is answered** — that section still
  carries the old sentence and is owed the correction.

  | figure | `cuda-linux` / vLLM in WSL (T6) | `llama-windows` / llama.cpp (T7) |
  |---|---|---|
  | `load-model dots-ocr` | **73.1 s** (a first run measured 99.3 s) | **6.2 s** |
  | seconds/page | **4.5** | **3.7** |
  | blocks parsed | **11** | **11** |
  | categories | `List-item`, `Section-header`, `Text` | the same three |
  | dialect | `dots-json`, parsed | `dots-json`, parsed |
  | unload | clean | clean |

  The card is an **RTX 3090 Ti** — in WSL2 for T6, the same card from Windows for T7.
  Earlier drafts of this section said "the 4090"; that was never this machine. Runs
  `20260915-003456` (T6) and `20260915-004350` (T7).

  **The two artifacts are byte-identical.** `t6/shape.json` and `t7/shape.json` — eleven
  objects, each `{bbox, category, text}` with a four-number bbox — `diff` clean. That is
  T7's stated pass criterion met, and it is the answer to *whether the Q8 GGUF answers in
  `parseDotsPage`'s dialect exactly*: it does, on the same page, to the byte.

  `llama-server`'s own timing for that page: prompt **2 595 tokens in 1.26 s**
  (2 058 tok/s), **657 output tokens in 2.44 s** (269 tok/s).

- **`qwen3.5-9b` Q8 under llama.cpp (T7)**: load **10.2 s**, **1.3 s** per cleanup
  chunk, and the chunk came back with the broken word joined. The completion names
  `model: qwen3.5-9b` and `system_fingerprint: b10970-bfdc32183`, which is **`--alias`
  proven against a real `llama-server`** rather than against a string this build wrote.

- **T6's first run, 2026-09-15 00:01, run `20260914-235923`** — kept because the defect
  it found is why the figures above exist. The page WAS read (the journal has the
  `200 OK`) but `scripts/read_one_page.py` unloaded in a `finally`, the server refused
  that unload `409 engine_in_use` because its own settlement had already begun clearing
  the card, and the refusal replaced the result before `answer.json` had been written.
  Both halves fixed in **e34271d**: `crucible/settle.py` (a clearance of the same model
  is the same intent, answered, not a conflict — `tests/test_llm_api.py`,
  `tests/test_tts_api.py`) and the stage, which now records and prints what it measured
  before it tidies up. The re-run is the T6 column above.

- **The three `llama-windows` `memory_bytes_estimate` figures are still DECLARED** (the
  GGUF's own size plus 1.5 GB, Foundry's `OVERHEAD_GB` and the same number every
  `[local] needs_bytes` in `models/` declares), and **one of the three now has a
  measurement beside it**. With `dots-ocr` resident the card read **~6.7 GB total**
  against a **2.5 GB** desktop baseline, so **≈4.2 GB** for the Q8 text tower plus the
  Q8 projector — under the declared 4 738 598 784 (4.74 GB), which is the direction an
  estimate should err. `qwen3.5-9b` (declared 11 027 502 048) is **measured only as a
  load that succeeded; bytes owed**. `qwen3.8-27b-4bit` (declared 17 964 440 224) was
  never loaded.
- **Read from the HuggingFace API on 2026-09-14** (tree API, LFS size):
  `unsloth/Qwen3.5-9B-GGUF` @ `3885219b6810b007914f3a7950a8d1b469d598a5`,
  `Qwen3.5-9B-Q8_0.gguf` 9 527 502 048 B;
  `unsloth/Qwen3.8-27B-GGUF` @ `4ca720788d1e01f1bff70c033e0d0028fd02e502`,
  `Qwen3.8-27B-UD-Q4_K_M.gguf` 16 464 440 224 B. `dots-ocr` reuses the pin its `[local]`
  table already carries.

### 7.4 NOT built, and every fact the next build would otherwise derive twice

The `llama-windows` backend EXISTS — kind, detection, catalog rows, capability. What it
cannot yet do is start a model. Owed, in this order:

1. **The `engine` subject** (3.10, fact 1). `LLAMA_CPP_RELEASE` is to be pinned at
   **`b10970`** (ggml-org/llama.cpp, published 2026-09-14T20:53:17Z). Its three assets and
   their sha256, read from the GitHub releases API on 2026-09-14 — the API publishes a
   `digest` per asset, so these are the release's own checksums and not a local
   measurement:

   | asset | bytes | sha256 |
   |---|---|---|
   | `llama-b10970-bin-win-cuda-12.4-x64.zip` | 254 074 942 | `78c878ae30622a9e4be09e3831066454668ca70398114f23bf74ac814e52dad8` |
   | `cudart-llama-bin-win-cuda-12.4-x64.zip` | 391 443 627 | `8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6` |
   | `llama-b10970-bin-win-cpu-x64.zip` | 18 428 751 | `2c6d6516c04e95caa080d8eb917743e71858c73985acbb6739ad61b14e68b298` |

   NVIDIA takes the first two into ONE directory (the server does not start without the
   cudart DLLs); a cardless machine takes the third. Refusals `engine_download_failed`,
   `engine_sha_mismatch`.
2. **`weights.pull` and `weights.installed` must become FILE-AWARE.** This is the one
   piece of existing machinery that blocks everything else and it is not optional:
   `pull` calls `snapshot_download` on the whole repo, and `unsloth/Qwen3.8-27B-GGUF`
   holds every quantization — hundreds of gigabytes. `BackendSpec.files` already exists
   and is the one owner of "which files does this backend fetch"; `pull` needs it as
   `allow_patterns` and `installed` needs to require every one of them present, which is
   also what makes `dots-ocr` report `installed: false` with the text tower and no mmproj
   (fact 2) and what makes section 3.5's *"the catalog's `installed` list must be exact
   per subject"* true for the host's weights migration.
3. **`LlamaServerEngine`**, a `SubprocessEngine` whose `command()` is `llama-server.exe`
   rather than a Python module. `Residency._engine_args` composes `-m <dir>/<file>`,
   `--mmproj <dir>/<mmproj>`, `-c <context>` and `--alias <crucible id>` from the spec —
   only the server knows where it put the weights, which is why the manifest's
   `engine_args` carries `--parallel 1` and nothing else.
   **One decision to record here: `--alias <crucible id>`, so `engine_model_name()`
   returns the Crucible id and the proxy is verbatim.** Fact 4 asks readiness to wait for
   a `/v1/models` name *ending in* `dots.ocr`; with an alias the check becomes "the name
   equals the id this server started", which is strictly stricter and generalises to the
   two text models the same mechanism serves. `pages_engine_wrong_model` keeps its name.
   `port_in_use` and the fatal-line early exit (`pages_engine_failed`) are unchanged, and
   **nothing is ever adopted** (fact 5).
4. **`install llm|pages` on this backend fetches the engine and nothing else**, and
   `install tts|asr|align|rvc|denoise` is refused `needs_wsl` with
   `capability.NEEDS_WSL_REASON`, which is already the sentence the capability rows carry.
5. **`crucible doctor`** prints the `backend: llama-windows on windows/x86_64 — llama.cpp
   <tag> (cuda-12.4 | cpu)` line; **`/v1/accelerator`** answers from nvidia-smi, or `cpu`
   with `physical_memory_bytes()` as the figure (already written, in
   `crucible/backend.py`).
6. **`main()`'s win32 gate.** Left untouched on purpose: the host agent's branch
   (`feat/phase15-host`) replaces it with an opt-in `win32_ok` flag, and two edits to one
   line is a merge conflict for nothing. Widening it is one flag.
7. **Section 3.5a** (`DELETE /v1/catalog/{kind}/{id}`, `crucible remove`,
   `removeSubject()`) and **section 4.7** (the `engine` task forwarding to the host's
   loopback door) — neither started.

### 7.5 Tests

Run under the WSL lock, single files, because a `train_lora.py` run holds the VM:
`test_settings_api.py` (25), `test_upstream_chat.py` (26), `test_pairing_file.py` (7),
`test_llama_windows.py` (19), plus `test_ui_mount.py`, `test_manifests.py` and
`test_capability.py` re-run green after their fixtures learned `route` and `gpu_vendor`.
**The full suite is owed and is scheduled after the training run** — it was 1234 before
this work. `sdk/ts`: 257, up from 235. (Both were settled on 2026-09-15: the suite is
green and `sdk/ts` is 299 — 7.6's Tests block.)

---

### 7.6 What 7.4's list became, 2026-09-14 (late)

Written by the agent that merged the host and Mac branches into this one and
built the seven things 7.4 handed over. **Numbered 7.6 and not 7.5 because 7.5
is already Tests** — 7.4's list said "§7 gains 7.5" and 7.5 was taken; the
number moved rather than the section it would have replaced.

#### Built

| 7.4 item | what landed | where |
|---|---|---|
| 1 | the `engine` SUBJECT, pinned at `b10970`, both CUDA zips into one directory | `crucible/llamacpp.py` |
| 2 | `weights.pull`/`installed` are FILE-AWARE through `BackendSpec.files` | `crucible/weights.py` |
| 3 | `LlamaServerEngine`, `--alias`, the fatal-line early exit, 30 s then kill | `crucible/engines/llama_server.py` |
| 4 | `install llm` fetches the engine; `install tts\|asr\|align\|rvc\|denoise` is `needs_wsl` | `crucible/cli.py` |
| 5 | `doctor`'s engine row; `/v1/accelerator` was already there | `crucible/cli.py` |
| 6 | `main()`'s win32 gate is GONE, not widened | `crucible/cli.py` |
| 7 | 3.5a `DELETE /v1/catalog/{kind}/{id}`; 4.7 the `engine` task | `crucible/api.py`, `crucible/tasks.py` |
| 3.10 f.7 | ONE page-request definition, published on `/v1/info` | `crucible/pages.py` |
| 5.3a | a module names CLASSES; the server resolves them and reports `unmet` | `crucible/modules.py`, `crucible/tasks.py` |
| 3.6 | `crucible serve` writes the pairing file too | `crucible/cli.py` |
| — | the manifests move INSIDE the package (a wheel never carried them) | `pyproject.toml`, `crucible/` |

**Refusals added, by name.** `engine_download_failed`, `engine_sha_mismatch`,
`engine_not_installed`, `pages_engine_failed`, `port_in_use`,
`subject_unknown`, `subject_not_installed`, `subject_in_use`,
`subject_remove_failed`, `engine_target_unknown`, `engine_move_not_here`,
`engine_move_needs_host`, `backend_not_here` (wired, having only existed as a
sentence), `needs_wsl`, `pairing_file_malformed` (SDK),
`capability_route_missing` and `capability_route_unknown` (SDK).

**The env vars the host door needs.** `crucible/host/` had NONE, so this build
names one: **`CRUCIBLE_HOST_DOOR`**, set by the host on the server it spawns,
holding the door's base URL (`http://127.0.0.1:7101`). **Its presence IS the
fact** that a host started this server, which is what 4.7's
`engine_move_needs_host` turns on; a probe of 7101 would be the wrong
question twice over, because something that is not a host can answer it and a
host restarting its own door is still the host. **The token is NOT carried**:
the door's bearer is the engine's own, which the server already holds, and a
second copy in an environment variable would be a secret with two owners.
Section 4.7 now says all of this.

#### Decisions this build made, and the deviations

- **`LLAMA_CPP_RELEASE` lives in `crucible/llamacpp.py`, not `envpack.py`.**
  7.4 said "beside `STANDALONE_PYTHON`", and the reason given was the
  pinned-download idiom. `envpack.py` is about Python packs; the engine
  subject is not one, and putting a 400-line module's constant in another
  module to be near a similar-looking constant is proximity standing in for
  ownership. The NAME is 7.4's.
- **The test door answers 502 `upstream_rejected`, not 401.** 3.2 says 401. A
  401 from a Crucible route means THIS server refused THIS client's bearer,
  and BookForge measured exactly the consequence: a bad Anthropic key drew
  "crucible refused the request" beside the key field. 7.2 already decided
  502 for every non-2xx but 429 at the chat door; this is the same decision
  at the other one, so one code has one status.
- **`crucible install pages` is refused, with the sentence that names `llm`.**
  3.5 writes it as a command. `pages` is a capability CLASS and not a job
  type (PHASE3-VLM.md section 1: there is no `vlm-pages` job type), so it has
  no installer and never will; it joins `INSTALLER_FOR` the way `denoise`
  did, and the refusal says which command works. The CLI's `choices=` still
  lists only real job types, so `crucible install pages` gets argparse's
  message rather than that one — the HTTP install task gets the good one.
- **The stop deviates from `base.py`'s never-SIGKILL rule, on purpose.** That
  rule is about CUDA inside WSL2 wedging the distro until Windows reboots.
  This engine is native; 30 s graceful (`CTRL_BREAK_EVENT` to the child's own
  group) then `terminate()`, and the reason is written where the deviation is
  made.
- **`config_permissions` is not checked on win32.** A Windows file has no
  POSIX mode and `stat` reports 0o666 whatever the ACL says, so the check
  reported a problem on every healthy Windows server the moment `doctor`
  started running there. Owed: a `doctor` line that reads the ACL back.
- **T9 is not in `scripts/testrun-phase15.sh`.** It drives the Mac's card over
  ssh; a button on the PC whose blast radius reaches another machine's GPU is
  a button that does not say what it does. Run it from the Mac.

#### Measured, at last

**S2 has run end to end on Owen's PC**, 2026-09-14, and the whole
`llama-windows` stack answers short of starting the engine:

| step | seconds | size |
|---|---|---|
| host pack built (`crucible envpack build host`) | 49 | 46 MB archived, 186 MB unpacked |
| unpacked under `C:\tmp\phase15-testrun\host\` | 5 | — |
| `crucible init --backend llama-windows --port 7101` | 3 | — |
| `engine llama-cpp` (both CUDA zips, verified, unpacked) | 19 | 645 MB fetched, 1.17 GB on disk |
| `models pull dots-ocr` (the GGUF pair) | 51 | 4.42 GB |
| `models pull qwen3.5-9b` (ONE Q8 of a repo of quantizations) | 97 | 9.53 GB |
| **S2 total** | **227** | **15 GB** |

`crucible doctor` on that home reports **healthy**, with
`engine: ready — llama.cpp b10970 (cuda-12.4)` and
`job load-model: ready — loadable: ['dots-ocr', 'qwen3.5-9b']`. **S1 has run
too**: `/home/telltale/crucible` is at this branch's HEAD, the unit restarted
(through the root `user@1000` recipe — the user bus was absent, exactly the
case 4.1 names), and `/v1/info` answers `pages_engine: {engine: "vllm",
installed: true}` with the eight-key `request` block, while `/v1/capability`
returns eleven rows all carrying `route`.

Two things the first real Windows run found, which is what running it is for:
`crucible doctor` asked `jobenv` for an `llm` env on a backend that has none
and crashed, and the `config_permissions` check called every healthy Windows
server unhealthy. Both fixed above.

#### What T7 found on the live card, 2026-09-14 (second defect)

The first press of the button with the card released got two stages into
`llama-windows` and stopped at the same place twice, for two different
reasons. The first was `read_state`'s hand-written backend list (3.5, first
sub-bullet). The second is this one, and it is the more interesting of the
two because the code was *correct on the backend it was written for*:

> `409 accelerator_busy: cannot load 'dots-ocr': the accelerator is held by
> pid 1460 ([Insufficient Permissions], memory not reported); pid 6028
> (…CrossDeviceResume.exe); pid 11208 (C:\WINDOWS\explorer.exe); pid 12852
> (…SearchHost.exe); pid 12880 (…StartMenuExperienceHost.exe)`

Every named process is Windows drawing a desktop. The guard's rule — a
foreign compute app on the card means somebody's job is on it, refuse — is
the right rule inside WSL2 and the wrong one on a desktop, where the card is
shared by design and nvidia-smi names every window on it. Carried across
unexamined, it made every load on this backend impossible.

**Fixed in `crucible/accelerator.py`**: on `llama-windows` the guard asks for
ROOM (`free >= memory_bytes_estimate`, refused `insufficient_memory` with both
figures, the neighbours reported in `details.processes` and never the reason);
the one holder is a `llama-server` this Crucible did not start, matched by
IMAGE NAME and refused `accelerator_busy`. `cuda-linux` and `mlx-darwin` are
untouched, and a test pins that same process list still refusing
`accelerator_busy` on `cuda-linux`. 3.5 and 3.10 fact 9 carry the rule; no new
refusal name was minted.

`read_windows_state()`'s handling of `[Insufficient Permissions]` rows was
re-checked against the exact CSV the driver printed here and kept — those rows
parse to `used_bytes: None`, which is not zero and is not treated as zero.

New tests: `test_accelerator.py` +8 (24 in the file), `test_llama_engine.py`
+2 with one CORRECTED — `test_a_busy_windows_card_is_still_accelerator_busy`
asserted the defect and is now
`test_a_full_windows_card_is_refused_for_the_room_and_not_for_the_company`.

#### What T7 found on the live card, 2026-09-15 (third defect)

With the guard fixed, the third press got all the way to the spawn. The
child started and then exited, and for once nothing in this repo was wrong:

> ```
> llama-server.exe -m ...\models\dots-ocr\llama-windows\Dots.Ocr-1.8B-Q8_0.gguf
>   --parallel 1 --mmproj ...\models\dots-ocr\llama-windows\mmproj-Dots.Ocr-F16.gguf ...
> E clip_init: failed to load model '...mmproj-Dots.Ocr-F16.gguf':
>   Key not found: clip.vision.projector.scale_factor
> E srv load_model: failed to load multimodal model
> E srv llama_server: exiting due to model loading error
> ```
> (`C:\tmp\phase15-testrun\home\logs\engine-dots-ocr.log`, 00:27)

**The catalog pinned two halves of two different projects.** Read off the
bytes, both sides:

* The pulled `mmproj-Dots.Ocr-F16.gguf` (GGUF v3, 388 tensors, 25 kv) has NO
  `clip.vision.projector.scale_factor`. It carries
  `clip.vision.spatial_merge_size = 2` — the older spelling, which upstream
  reads for qwen2vl-family projectors and never for this one.
* `tools/mtmd/clip.cpp` at the pinned `b10970`, case
  `PROJECTOR_TYPE_DOTS_OCR`, line 1542: `get_u32(KEY_PROJ_SCALE_FACTOR,
  hparams.n_merge);` — no `required = false`, where every optional read in
  that function passes one. `KEY_PROJ_SCALE_FACTOR` is
  `"clip.vision.projector.scale_factor"` (`clip-impl.h:61`).

**The llama.cpp pin does not move.** That line is byte-identical at `b10950`
— Foundry's own `PINNED_RELEASE` — so this is not a regression between two
builds and no choice of upstream tag makes that file load. The reason is in
the anthonym21 README, which the manifest had quoted only half of: under the
2026-03-23 regeneration note it says *"Requires a llama.cpp build with
DotsOCR support. At the moment, use: anthony-maio/llama.cpp"* — **a fork**.
Those GGUFs were converted by a fork's converter for that fork's loader.

**Fixed in `crucible/models/dots-ocr.toml`** (`[local]` and
`[backends.llama-windows]`, which name one pair by design):

| | was | now |
|---|---|---|
| repo @ rev | `anthonym21/dots.ocr-GGUF` @ `42ab3102…` | `ggml-org/dots.ocr-GGUF` @ `2c093a32ca360a396bc6d87d60408636130b9d9b` |
| text tower | `Dots.Ocr-1.8B-Q8_0.gguf` 1,894,530,336 B | `dots.ocr-Q8_0.gguf` 1,894,530,272 B (sha256 `6f2db1a7…fa1de7e9`) |
| projector | `mmproj-Dots.Ocr-F16.gguf` 2,524,495,808 B (sha256 `b65a1db5…f6970de8`) | `mmproj-dots.ocr-Q8_0.gguf` 1,344,068,512 B (sha256 `20d3d35b…8e822423`) |
| download | 4,419,026,144 (4.42 GB) | 3,238,598,784 (3.24 GB) |
| needs / estimate | 5,919,026,144 (5.92 GB) | 4,738,598,784 (4.74 GB) |

The new projector's header, read over a range request on 2026-09-15 (GGUF v3,
304 tensors, 24 kv): `clip.projector_type = dots_ocr`,
**`clip.vision.projector.scale_factor = 2`**, `clip.vision.image_min_pixels =
3136`, `clip.vision.image_max_pixels = 11289600`, and no
`clip.vision.spatial_merge_size` — the same fact from the other side.

**It is also Foundry's pair, which is the point.** PHASE9-CAPABILITY.md 7.4's
"Found 2" was this disagreement written down before it cost anything: the
constants Wave 61 package E deleted from `app/electron/page-reader.ts` (foundry
@ `06efb83`) were `ggml-org/dots.ocr-GGUF` / `dots.ocr-Q8_0.gguf` /
`mmproj-dots.ocr-Q8_0.gguf`. The catalog is the owner and Foundry reads it, so
this closes the fact with two owners by moving the owner to the half that
works, not by inventing a third pin. The same repo's `mmproj-dots.ocr-f16.gguf`
(2,526,296,992 B) also carries the key and would load; the Q8_0 one is pinned
because it is what both apps once shipped, and if a page proves the quantised
projector reads worse, THAT measurement moves the line to the f16 file at the
same revision. Nothing selects between them at runtime.

`foundry-lineup.json` regenerated (`scripts/gen-foundry-lineup.py`), and
`docs/PHASE9-CAPABILITY.md` 7.4's table and Found 2 say the pair is settled.

Tests updated for the pin: `test_manifests.py`, `test_lineup.py`,
`test_llama_windows.py`, `test_llama_engine.py` — the filenames and sizes were
asserted in four files, which is what kept the change honest.

**Owen pulls; this build did not.** The home is an environment variable, not a
flag (`CRUCIBLE_HOME`, `crucible/config.py`), and `remove` comes FIRST: the two
superseded files sit in
`C:\tmp\phase15-testrun\home\models\dots-ocr\llama-windows\` and the new spec
does not name them, so a bare pull leaves 4.42 GB of a fork's GGUFs beside the
3.24 GB that works and `directory_bytes` reports the sum.

```powershell
$env:CRUCIBLE_HOME = 'C:\tmp\phase15-testrun\home'
crucible remove model dots-ocr     # the subject's whole directory
crucible models pull dots-ocr      # the pair above, at the pinned revision
```

That sequence was run on 2026-09-15 and **the first line did not do what this
paragraph says it does** — see T8's second finding below. The old pair is
still there.

#### The button ran — 2026-09-15, the card released

Owen released the card at ~23:00 on the 14th and the whole list below stopped
being a list of intentions. The figures are in **7.3** (T6/T7), **7c** (T9)
and this section's Tests block (T1/T2/T4); what follows is what each
"unmeasured" line became.

| was unmeasured | now |
|---|---|
| seconds/page, dots under llama.cpp on Windows | **3.7** (T7) |
| whether the Q8 GGUF answers in `parseDotsPage`'s dialect exactly | **it does** — `t7/shape.json` and `t6/shape.json` `diff` clean, eleven `{bbox, category, text}` objects |
| seconds/page, dots under vLLM (on an RTX 3090 Ti, not a 4090) | **4.5** (T6) |
| seconds per cleanup chunk, `qwen3.5-9b` Q8 under llama.cpp | **1.3** (T7) |
| `--alias` against a real `llama-server` | **proven** — readiness passed on both loads and the completion names `model: qwen3.5-9b` |

**Three of the old lines are only PARTLY answered, and say so:**

- **the three `memory_bytes_estimate` figures** — `dots-ocr` measured at
  ≈4.2 GB against a declared 4.74 (7.3); `qwen3.5-9b` **measured only as a
  load that succeeded, bytes owed**; `qwen3.8-27b-4bit` never loaded.
- **the fatal-line table against real llama.cpp output** — the early exit
  DID fire on real output, at 00:27 on the projector defect above
  (`srv llama_server: exiting due to model loading error`, `engine_failed`
  with the last 40 lines). That is one path of eight. The other seven
  substrings are still Foundry's, matched only against strings this build
  wrote.
- **the engine task end to end** — T10 **PASS, 4 s**, after the naming fix
  (**10908f5**: `host_unreachable` / `host_install_failed`, and
  `engine_move_needs_host` reserved for the case where there is no door at
  all). What it proves is that the task is accepted and handed to the door.
  **The move itself is still unmeasured**: no distro was imported and no host
  tray was started, so migrate-config and migrate-weights have run against
  fake servers only (7b.4b) and nothing on this machine has witnessed the
  order between two live catalogs.

#### What T8 found — two stage-order facts, both owed

T8 (the remove door) is **PASS, 1 s**, and on the way it turned up two things
that are about the ORDER of the run rather than about the door:

1. **T8 before a T7 re-run removes the model T7 needs.** `DELETE
   /v1/catalog/model/qwen3.5-9b` does exactly what it says, and the next T7
   then refuses `model_not_installed` by name (run `20260915-004132`). Fixing
   it was a re-pull: **9.53 GB in 88 s, 109 MB/s**. The stage list in §8 is
   written T7-then-T8 and must stay that way, or T8 needs to put back what it
   takes.
2. **`crucible remove model dots-ocr` left the superseded files behind.**
   Remove follows the CURRENT spec's file names, so on the staged home it
   deleted the pair the new `ggml-org` pin names and left the old
   `anthonym21` `Dots.Ocr-1.8B-Q8_0.gguf` / `mmproj-Dots.Ocr-F16.gguf` sitting
   in the same directory — 4.42 GB that `directory_bytes` still counts and
   nothing will ever load. 3.5a says a subject's whole directory goes; the
   implementation reads the spec instead. **Owed.**

#### The Higgs door, on the card

Not a button stage — the TTS half of `cuda-linux`, checked the same night
with the `owen` voice, because a card that reads pages should also be shown to
render.

| | |
|---|---|
| take-0 render job | **166 s** including the engine load; **11.3 s** of audio for 150 characters |
| SGLang on the card | **20.0 GB** at `--mem-fraction-static 0.6` — the 19 GB estimate stands, now measured |
| take-1 render job | **140 s**, and **byte-identical to take 0** |
| `take: 2` | refused `unknown_take` |
| zero-shot load | refused `engine_failed: … does not carry generation_config.json` (a base checkpoint) — **open** |

**The byte-identical take is two facts, not one.** The tts env's pinned
narrator (`0eeb0267`) predates the per-item sampling channel, so the take
never reached the engine; the handshake and a `sampling_not_wired` guard are
restored (**a6c34f2**, **fe709ff**) and the env re-pin is blocked on Owen
pushing BookForge's branch. Underneath that, narrator seeds `1234 + index`,
so **same-take re-rolls are byte-identical by design** — a per-take seed is
being built. Neither of those is the guard failing; both are it working and
saying so.

#### The narrator `HIGGS_ENV` prefix regression

`a7ab9af` put a prefix in front of `HIGGS_ENV` that the serve script could not
resolve; fixed in **96980ce**, which is the HEAD the `20260915-001145` stage
ran at. Recorded here rather than in a BookForge doc because this night's run
is what found it.

#### Tests

Under the WSL lock, single files. New: `test_llama_engine.py` 29,
`test_catalog_remove.py` 14, `test_engine_task.py` 15,
`test_pages_request.py` 11, `test_module_needs.py` 11, `test_wheel.py` 5.
Changed and re-run green: `test_cli.py` 30, `test_envpack.py` 81,
`test_settings_api.py` 25, `test_upstream_chat.py` 26,
`test_llama_windows.py` 19, `test_service.py` 54, `test_pairing_file.py` 11,
`test_manifests.py` 93, `test_engine_readiness.py` 10, `test_capability.py`
36, `test_catalog_route.py` 17, `test_modules.py`, `test_tasks_api.py`,
`test_api.py`, `test_llm_api.py`, `test_ui_mount.py`, `test_activity.py`,
`test_voices.py`, `test_tts_api.py`, `test_tts_render.py`, `test_jobenv.py`,
`test_workerenv.py`, `test_rvcmodels.py`, `test_rvcbase.py`,
`test_rvc_api.py`, `test_denoisemodels.py`, `test_denoise_api.py`,
`test_asrmodels.py`, `test_alignmodels.py`, `test_align_api.py`,
`test_asr_api.py`, `test_lineup.py`, `test_setup_route.py`. `test_host.py`
87 on the **Windows** interpreter (86 passed, 1 skipped) — its subject is
Windows and its platform is injected, so it runs on both.

`sdk/bootstrap` 245. `sdk/ts` **290**, up from 257.

**The full suite is no longer owed — T2 ran, 2026-09-15.** `sdk/bootstrap`
**245 pass**, `sdk/ts` **299 pass**, 33 s (T1, run `20260914-231608`). The
full pytest in WSL under the lock found **three stale tests and nothing else**,
all three stale for a reason this phase created:

| test | why it was wrong |
|---|---|
| `test_backend.py::test_windows_is_never_a_backend` | superseded by 3.5 — Windows IS the `llama-windows` backend now |
| `test_host.py::test_there_is_no_platform_gate_left_in_main` | needed `CRUCIBLE_HOME` on Linux: without `%LOCALAPPDATA%` `crucible_home()` refuses, by design |
| `test_vlm_pages.py::test_the_page_manifest_has_no_mac_block` | asserted one backend block; `dots-ocr` now has two |

Green after **1a0d387**. T4 — BookForge's keepers — is green too, once the
prompt-vendor tier was written (BookForge `392f20c3`).

## 7b. What was built, 2026-09-14 — the host side

Section 4 entirely, plus the pieces of 3.5/3.6 the host needs to exist. Sections 1-3 and 5 are
the server's and the apps'. `crucible/host/` is new; `npm test` in `sdk/bootstrap` is **245
passing**, from 191 at the start of this phase.

### 7b.1 The package, and why it is eleven files

Everything that can be a pure function is one, in its own module, because the tray cannot be
tested and every decision it draws must be. `tray.py` is the only file with no test, and it
contains no `if`.

| file | what it owns |
|---|---|
| `paths.py` | every path, from the environment only. `%LOCALAPPDATA%\Crucible\` and its four members. |
| `log.py` | `host.log`, appended, rolled once at 2 MiB. |
| `menu.py` | `(distro, engine) → MenuModel`. 4.2. |
| `runner.py` | the one door to a subprocess and to `/v1/ping`. Injectable. |
| `presence.py` | the boot, the two recovery recipes, the 15 s watch. 4.1. |
| `wsl_states.py` | **GENERATED** from `sdk/bootstrap/src/wsl-states.ts`. |
| `wslstate.py` | the predicates for that table, and the walk. |
| `landoor.py` | 4.1's LAN forward: which mechanism, and whether it is already open. |
| `catalog.py` | the two catalog ports the weights migration reads, and 3.5a's delete. |
| `installer.py` | 4.7's sequence, including the weights migration. |
| `door.py` | `POST /install` on 127.0.0.1:7101. |
| `startup.py` | the Startup shortcut, and the two verbs that own it. |
| `tray.py` | pystray. Nothing else. |
| `app.py` | the loop that holds them. |

**Importing the package needs neither pystray nor tkinter.** pytest runs in WSL, in an env
that has neither, and a suite that cannot import its subject pins nothing. `tray` is imported
inside the functions that use it and nothing else imports it at all.

### 7b.2 The 4c table crosses into Python by GENERATION

`gen-install-scripts.ts` gained a third output. The table's DATA — code, probe argv, sentence,
action, order — is emitted into `crucible/host/wsl_states.py`, and `npm run gen:install --
--check` refuses a drift in it exactly as it does for `install.sh` and `install.ps1`. Only the
`means` PREDICATES are spelled twice, because they are code; `wslstate.MEANS` holds one per
code and a pytest asserts the two sets are equal. It is the seam `envpack.SMOKE_IMPORT` has
with `cli.INSTALLABLE_JOB_TYPES`, and `steps.ts` has with its three programs.

Getting a template out of a sentence that is a FUNCTION of its evidence took sentinels: the
generator calls each row with values that cannot occur, swaps them for `{said}`,
`{app_distro}`, `{guest_user}`, `{release}`, `{required}`, `{free}`, and **refuses** both when
a sentinel survives and when a placeholder the table used to produce stops appearing. A
sentence that reaches a person with a sentinel in it is worse than a generator that stops.
One row needed thought: `pack_disk`'s two figures are rendered by `gib()`, so the sentinels
there are NUMBERS (`424242 GiB` and a `df` reply) rather than strings.

### 7b.3 `install.ps1` no longer walks the table

It downloads the host pack for this release, verifies it, unpacks it to
`%LOCALAPPDATA%\Crucible\host\`, asks `crucible host --install-startup` for the login item,
starts the tray with `pythonw`, and stops. It needs no admin. Three things it does that the
old one did not:

- **It asks `tar --version` whether this machine's tar carries zstd.** Measured below.
- **It unpacks BESIDE, runs the moved `crucible.cmd --version`, and only then renames.**
- **It never elevates.** The host raises UAC later, by name, when a 4c row needs it.

The old walk is not lost — it moved into `installer.py`, where it can carry a reboot across
(the Startup item) and where the page's engine switch (4.7) drives the same code.

### 7b.4 Measured on Owen's PC, 2026-09-14, with nothing changed

**bsdtar and zstd, which decided two designs.**

```
C:\Windows\system32\tar.exe --version
  bsdtar 3.8.1 - libarchive 3.8.1 zlib/1.2.13.1-motley liblzma/5.4.3 bz2lib/1.0.8
  libzstd/1.5.5 cng/2.0 libb2/bundled
zstd --version                                  -> not found (no zstd.exe on Windows at all)
tar.exe --zstd --options zstd:compression-level=10 -C src -cf out.tar.zst .   -> 0
tar.exe -xf out.tar.zst -C back                                              -> 0, bytes intact
```

So `require_zstd_tar()` ASKS the tar it found rather than demanding a binary the OS does not
ship — and the refusal is not theoretical: **`crucible envpack build host` run from Git Bash
refused by name**, because Git Bash puts its own GNU tar 1.32 ahead of System32 on PATH:

```
crucible: pack_no_zstd: `C:\Program Files\Git\usr\bin\tar.EXE` reports 'tar (GNU tar) 1.32',
which does not name libzstd. … Put that one ahead of this one on PATH. A GNU tar here would
shell out to a `zstd` that is not installed and fail at the compression step, which is after
the interpreter download and the pip run — the expensive end of the build to find out at.
```

That is the check earning its place on the first real run.

**The interpreter pin, read from the release.** `curl` of
`https://github.com/astral-sh/python-build-standalone/releases/download/20260901/SHA256SUMS`
(870 lines): the word "shared" appears **0** times, "static" 90. So 4.4's
`x86_64-pc-windows-msvc-shared-install_only` does not exist and the pin is
`cpython-3.11.16+20260901-x86_64-pc-windows-msvc-install_only.tar.gz`, sha256
`6be524fa6752af802146a4adc7d098565425b0b1c166e19a5a7a4c8cccb86bf6`. 4.4 is corrected.

**WSL and the LAN door, read and deliberately not changed.** `wsl --version` 2.5.7.0,
Windows 10.0.26100 — mirrored networking is supported. `%USERPROFILE%\.wslconfig` has no
`networkingMode`, so this machine is on NAT. `netsh interface portproxy show v4tov4` listed
**nothing**. `landoor.detect()` therefore answers `portproxy / not open`, which is the state
that would prompt for administrator on a real install; **no `netsh` was run.**

**The probes, against the real machine.**

```
paths.crucible_root      C:\Users\tellt\AppData\Local\Crucible
startup.shortcut_path    C:\Users\tellt\AppData\Roaming\Microsoft\Windows\Start Menu\
                         Programs\Startup\Crucible.lnk
probe_distro             absent — wsl -l -v lists Ubuntu and no "crucible"
GET 127.0.0.1:7100/v1/ping   answered
```

`absent` is the honest answer on this machine and it is what makes the menu offer the WSL
install; Owen's own Ubuntu was read and never touched.

**The tray, by hand.** Run from a throwaway venv under `C:\tmp` (pystray + pillow, deleted
afterwards — nothing installed system-wide), against a TEMPORARY `LOCALAPPDATA` and `APPDATA`,
with the install door on a port the OS picked. **No server was started, no distro imported, no
Startup shortcut written, port 7100 untouched, and nothing went near the card.**

The icon appeared in the notification area. Its menu read:

```
Crucible — stopped
  Open console                                            (disabled)
  Install the WSL2 engine (faster pages and text; TTS, ASR…)…
  Restart engine
  Stop engine                                             (disabled)
  Open log
  Quit (stops the engine)
```

`Open log` opened the log in Notepad and wrote `menu: open-log` to it. Driving the state to
`installing` and then to `running` redrew the title to `Crucible — installing…` and then
`Crucible — running (llama-windows)` without the icon flickering. `Quit` returned from
`icon.run()` cleanly. Afterwards `%LOCALAPPDATA%\Crucible` did not exist and the real Startup
folder was unchanged — verified, not assumed.

### 7b.4a The Windows host pack, BUILT — `crucible envpack build host`

On Owen's PC, with `C:\Windows\System32` ahead of Git Bash on PATH, into `C:\tmp\hostpack`.
CPU only: a python-build-standalone download, a `pip install` of the wheel plus `pystray` and
`pillow`, a prune and a tar. **Nothing went near the card.**

| | `host` / `llama-windows` |
|---|---|
| build time | **64 s** (interpreter cached after the first run) |
| unpacked | **185,628,698 B** (186 MB) |
| archive (bsdtar `--zstd`, level 10) | **46,151,197 B** (46 MB) — 25% of the tree |
| parts | **1** |
| sha256 | `1a7ffac5c379c1063dfbf2344662b5761bac9f0d3efc090177464fadd790d8d0` |
| recipe sha256 | `ebef09466a9c…` (`pyproject.toml` — the SAME digest PHASE14 7.2 recorded for the `server` pack, which is the point of 7.3's ruling) |
| python | **3.11.16**, from the pin read out of SHA256SUMS |
| shims written | **13** — `crucible`, `dotenv`, `fastapi`, `hf`, `httpx`, `huggingface-cli`, `idna`, `tiny-agents`, `tqdm`, `uvicorn`, `watchfiles`, `websockets`, `wheel` |
| smoke test | `crucible.cmd --version` in a temp unpack → `crucible 0.6.0` — **passed** |

**And then the pack was moved again, by hand, and the relocation question settled.** Copying
the single part to `C:\tmp\hostcheck\pack.tar.zst` and unpacking it into a directory the build
never saw:

```
tar -tf …            ./  ./crucible.cmd  ./DLLs/  ./dotenv.cmd  ./fastapi.cmd  …
                     — TOP LEVEL, no wrapper directory (PHASE14 7.3a's invariant)
cat crucible.cmd     @echo off
                     "%~dp0python.exe" -m crucible.cli %*
crucible.cmd --version          -> crucible 0.6.0            exit 0
crucible.cmd host --install-startup -> wrote the shortcut     exit 0
Scripts\crucible.exe --version  -> (nothing)                 exit 1
```

**That last line is the whole reason `write_cmd_shims` exists.** pip's launcher binary carries
the build tree's interpreter path inside it, and it is dead the moment the tree moves — the
Windows form of the defect PHASE14 7.2a found on POSIX, where a shebang rewrite fixed it and
here where no rewrite can reach. The `%~dp0` shim beside it runs.

**Two things this by-hand run found that reading the code had not:**

1. **`crucible host --remove-startup` did not parse.** The PowerShell was two adjacent strings
   with only the first an f-string, so the second's escaped braces stayed DOUBLED and
   PowerShell got `} } else {` — "Unexpected token '}'". It was found by using it to undo the
   shortcut the line above had just written, which is exactly the sequence a person performs.
   Fixed, and pinned by two tests that assert the braces are balanced and single.
2. **The shortcut was written to the REAL Startup folder** by that `--install-startup`, which
   this session was told not to do. It was removed with the fixed verb and the folder verified
   back to its two original entries (`LG Monitor App Installer.lnk`, `Ollama.lnk`).

### 7b.4b The weights migration, made real against 3.5a

`DELETE /v1/catalog/{kind}/{id}` (3.5a) is the door 7b.6 asked for, and `migrate-weights` is
now written against it rather than describing what it would do.

**The shape.** `crucible/host/catalog.py` has two ports and three verbs —
`installed_subjects()`, `pull()`, `remove()`. They are two ports and not one because during
the move both servers want `127.0.0.1:7100` and on Windows the Windows one has it: the
Windows server is dialled over loopback from this process, the guest is reached by running
`curl` INSIDE the distro, which is the only address unambiguously its own. They speak the
same verbs and answer the same refusals, so the sequence never branches on which side it is
talking to — **it is the ORDER that carries 3.5's rule, not the transport.** One bearer opens
both, because `migrate-config`, three steps earlier, made the guest's token the Windows one.

**The loop, per round:** read BOTH catalogs; for each subject the Windows engine reports
installed, submit the guest's pull if the guest does not have it, WAIT until the guest's own
catalog says `installed`, then `DELETE` it on Windows. Then re-read and go again. A machine
unplugged at any instant has that subject on one side or on both, never on neither.

**Idempotent by RE-DIFFING, not by a journal.** Every round re-reads the two catalogs and acts
on the difference, so a resume after a crash, a reboot or a Ctrl-C needs no state that
survived the crash — which is the only kind of resume that is true after a power cut. Running
the whole step again on a finished machine reads two catalogs and does nothing, which is a
test.

**`subject_in_use` is waited out, never skipped.** 3.5 says nothing is skipped, so a held
subject comes back on the next round with `details.who` named in the meantime —
`MIGRATE_IN_USE_ROUNDS` (60) x `MIGRATE_POLL_SECONDS` (5 s), five minutes of somebody closing
an app — and then the step fails BY THAT NAME, naming every holder. The bound is deliberate:
the two honest ends are "removed" and "still held, and here is who", and an unbounded wait
would be a third, which is a move that never finishes and never says why. Nothing is lost when
it fails that way — the guest has its copy and the Windows one is still there.

Three more refusals, each of which keeps a Windows copy alive rather than deleting it:
`subject_pull_timeout` (the guest never reported it installed), any other `DELETE` refusal
verbatim (`subject_remove_failed`, …), and `catalog_unreachable` — because **"nothing
installed" and "could not ask" must never be the same answer**, the first being an answer that
would delete every Windows copy on the machine.

**Tested with two fake servers**, which is the only way to witness an order between two
machines: the guest has a subject before the Windows copy goes; a subject the guest already
has is not pulled again; a half-done move resumes from the diff and is a no-op the second
time; `subject_in_use` is retried three times and then succeeds; held forever fails by name
with the holder; a pull that never lands leaves the Windows copy alone; and a non-`in_use`
refusal keeps both copies.

### 7b.4c Two facts about 4.1's presence, measured on the button's night (2026-09-15)

Neither was written down anywhere, both change what the tray has to do, and
both were found by needing the WSL server up for T6 rather than by reading
`presence.py`.

- **The distro terminates seconds after the last `wsl.exe` session ends, even
  with systemd units running.** A `Restart=always` unit and an enabled-linger
  user do not keep the VM alive; only a HELD PROCESS does. 4.1 says the host
  runs `wsl -d crucible --exec true` at start and then watches `/v1/ping` —
  that boots the distro and then lets go of it, so the thing it is watching
  goes away on its own. **The tray owns this**: something of the host's has to
  stay attached for as long as the WSL engine is meant to be the engine. 4.1 is
  owed the sentence.
- **Linger was OFF on this machine.** `loginctl enable-linger telltale`, run as
  root, is now done. That is a precondition 4.3 should set and check, not
  something a person discovers when the unit is not there.
- **The user bus is unreachable from a `wsl.exe --exec` session.**
  `systemctl --user …` answers "Failed to connect to bus", which is 4.1's
  `user-unit-start` recipe failing in exactly the case it exists for — so the
  unit could not be restarted from Windows at all tonight and the server was
  run as a held process instead. The `user-bus-restart` recipe
  (`systemctl restart user@1000` as root) is the one that works from there;
  what is missing is the recipe written out as commands. **§4 is owed it.**

### 7b.5 Decisions, where the doc left a choice

- **`distro = unknown` is a state and not a synonym for `absent`.** `wsl.exe` failing to
  answer offers the install and never claims a server. Reading it as `absent` would import a
  SECOND distro, which is the one mistake here that pressing the button again cannot undo;
  reading it as `present` would hide the only item that can fix the machine.
- **A ping that answers ANY status is a server that is up**, 401 included. Treating a refused
  token as "down" would have the host boot a distro because somebody's bearer was wrong.
- **A failed pairing-file ACL DELETES the file.** A bearer token on disk that everybody on the
  machine can read is worse than no pairing file: absent is a fact an app knows how to handle
  (3.6), readable is a silent leak.
- **`main()`'s win32 gate NARROWED rather than opened.** A subparser opts in with
  `win32_ok=True`; `host` and `envpack` do (4.4 builds the Windows pack on Windows) and
  everything else keeps the old refusal until 3.5's `llama-windows` server lands. Letting
  every verb through now would replace one honest refusal with a `NoViableBackend` from
  somewhere deeper — the same "no" with a worse sentence and a stack trace.
- **The guest half of the install is `install.sh`, run by the host.** Restating its six steps
  in Python would be the third spelling of a list that already has two. The cost is one extra
  `crucible init --force --config-from` afterwards, because `install.sh` mints its own token
  and 3.5 says the Windows one has to survive; two inits and one token beats one init and an
  app that silently stops being paired.
- **`--config-from` is extracted TEXTUALLY, not parsed and re-emitted.** A key is a secret and
  a round trip through a writer is a chance to mangle one. `write_config` copies the carried
  tables verbatim and refuses to shadow a table it owns.
- **The door's `done` carries a result where `crucible/tasks.py`'s carries `{}`.** It has a
  caller tasks.py does not: `install()` is a library function that must RETURN an
  `InstallResult`, and on Windows it cannot go and read the guest's config instead — that
  `wsl.exe` door is one of the things this phase deletes. The relaying server may drop it.
- **The door accepts fields it does not read** (`release`, `job_types`, `home`, `bind`).
  `engine_target_unknown` is reserved for a `target` that is not `wsl`, which is the one field
  whose wrong value would DO the wrong thing. An older door refusing a field a newer client
  was told to send is how two halves of one release stop talking.
- **The migration polls the CATALOG, not the pull task's events.** A task is a report about
  a fact and `installed` is the fact, and the thing that gates a deletion has to be the fact.
  It also makes the step survive losing the event stream, which a six-hour download will.
- **The catalog ports are built per RUN, not once at tray start.** The token they use is the
  one the config holds when the move BEGINS, and `migrate-config` is what makes the guest's
  token the Windows one. A port that captured a token at startup would be holding a token the
  guest never had.
- **The CI job for the Windows pack is a job and not a matrix row.** Every step in that matrix
  is written in `sh` and reclaims disk with `sudo rm -rf`. One job with four Windows-shaped
  steps is honest; a matrix with `if: runner.os` on half its steps is a matrix pretending to
  be one job.

### 7b.6 What the host side could NOT do, and why

- The pack build below DID happen and is in 7b.4a; what is still owed is a `windows-latest`
  run and a published asset.
- **No distro was imported and no install was run end to end.** `crucible-rootfs-<version>.tar.zst`
  is not on any release yet (PHASE14 7b.5 says the same), and importing anything on this
  machine was out of scope by instruction. `installer._import_distro` therefore refuses
  `no_crucible_distro` naming the missing asset rather than importing somebody else's image,
  and the steps after it are exercised only against the scripted runner.
- **`install-job-types` installs NOTHING yet**, and says so on the event stream. Its input is
  the coordinate records the Windows server keeps for each connected app (4.7), and there is
  no door onto them yet. The apps' own coordinate step (PHASE14 4a) installs what they need
  on first connect to the guest, so nothing is lost — it is just not done here.
- **`migrate-weights` is now REAL** (7b.4b) but has never run against two live servers,
  because neither the `llama-windows` catalog nor a guest on this machine exists yet. It is
  exercised against two fake ones, which is the only way to witness an ORDER between two
  machines at all.
- **The LAN forward was not added.** `netsh` needs administrator and Owen's machine was to be
  read, not changed. Detection is measured (7b.4); `add_argv()` / `remove_argv()` are data and
  tested as data.
- **`crucible host` was not run through the CLI verb itself** — only its objects, from a
  harness, so that nothing wrote to the real `%LOCALAPPDATA%` or Startup folder. The verb's
  refusal off win32 IS tested.
- ~~**The full pytest suite was not re-measured at the end.**~~ **Done, 2026-09-15** — T2
  ran under the lock once the trainer let go, found three stale tests and nothing else, and
  is green after `1a0d387`. See 7.6's Tests block.

### 7b.7 Tests

| suite | before | after |
|---|---|---|
| `sdk/bootstrap` `npm test` | 191 | **245** |
| `tests/test_host.py` | — | **85 passed, 1 skipped** |
| `tests/test_envpack.py` | 45 passed / 14 skipped | **67 passed / 14 skipped** |
| pytest, whole tree minus `tests/test_lineup.py` | 1234 (reported) / 1215 measured here | **GREEN, 2026-09-15** (T2, under the lock, after `1a0d387`) — the LoRA trainer that held the VM through every earlier attempt was gone by then |
| `sdk/ts` `npm test` | 290 | **299** (T1) |

`tests/test_lineup.py` fails on this checkout for a reason that is not this phase's: it is a
git WORKTREE, its `.git` is a file pointing at a Windows path, and `git rev-parse` inside WSL
cannot follow it. Four failures, all of them that.

The one skip is `test_on_posix_the_pairing_file_is_0600_from_the_outset`, and it is skipped
on NTFS rather than deleted or faked: a mode is a property of the FILESYSTEM and NTFS has none
to report (it answers `0o666` for every file). It is the only assertion in the file that
cannot be platform-injected, because the thing under test is the filesystem's own answer. The
Windows half of that same rule — the `icacls` ACL and the file being DELETED when it cannot be
set — runs everywhere.

**Every host test runs off Windows.** The platform, the environment and every subprocess are
injected, because a suite that skipped its subject on the machine it runs on would pin
nothing. All fifteen cells of 4.1's `(distro, engine)` table are walked, not sampled.

---

### 7b.8 THE HOST IS INSTALLED AND RUNNING — the first real run, 2026-09-15

Not a harness this time: the pack on disk under `%LOCALAPPDATA%`, the Startup item in the
real Startup folder, the tray started the way the shortcut starts it, against the machine's
own engine. Nothing went near the card; a mistborn LoRA trainer held the 3090 Ti throughout
and was not disturbed.

**What is installed, and where.**

| | |
|---|---|
| host pack | `C:\Users\tellt\AppData\Local\Crucible\host\` — 186,333,005 B unpacked, `crucible.cmd --version` → `crucible 0.6.0` |
| stamp | `…\host\.pack` — `sha256=dbad3ef8ca42a2e10eda557b2bdce5921f1858af8863f49236c58efa8876b296`, `release=0.6.0` |
| Startup item | `…\Roaming\…\Startup\Crucible.lnk` — exactly one, target `…\Crucible\host\pythonw.exe`, args `-m crucible.cli host`, workdir `…\Crucible\host` |
| tray | `pythonw.exe` pid 45504; `…\Crucible\host.pid` holds the same number |
| log | `…\Crucible\host.log` |
| pairing file | `…\Crucible\pairing` — 95 B, one line, ACL `OWENS-PC\telltale:(R,W)` and nothing else |
| door | `127.0.0.1:7101`, listening |

**`install.ps1` could NOT be used, and that is a release fact and not a defect of the
script.** Its step 1 fetches `envpacks.json` from the release and dies `pack_manifest_unreadable`
if it is not there. The latest release is **v0.5.0** and it publishes three assets — the
wheel, the sdist and `crucible-client-0.5.0.tgz`. There is no `envpacks.json`, no rootfs and
no host pack on ANY release, and `install.ps1`'s own default is `-Release 0.6.0`, a tag that
does not exist yet. So the pack was BUILT here (`crucible envpack build host`, 54 s, CPU
only — and **`C:\Windows\System32` must come before Git's `usr\bin` on PATH**,
or the build reaches `pack_no_zstd` after the interpreter download and the pip
run: Git for Windows ships GNU tar 1.32, which names no libzstd and would shell
out to a `zstd.exe` Windows does not have, while System32's bsdtar 3.8.1 carries
libzstd 1.5.5. Hit for real on 2026-09-15 from a Git Bash shell) and `install.ps1`'s steps 5–7 were then performed verbatim against it: join the parts,
verify the sha against the manifest the build wrote, unpack beside, prove `crucible.cmd`
runs, move into place, stamp, `crucible host --install-startup`, `Start-Process -WindowStyle
Hidden pythonw.exe -m crucible.cli host`. **7b.6's owed item is unchanged and is now the
blocking one: a `windows-latest` run and a published asset.**

**The branch it took, and the log it wrote.**

```
crucible host 0.6.0 starting; CRUCIBLE_HOME=C:\Users\tellt\AppData\Local\Crucible
startup: …\Startup\Crucible.lnk now starts `crucible host` at login, with no console window
presence: wsl -l -v lists Ubuntu and no "crucible"
find-engine: the engine on http://127.0.0.1:7100 is the "Ubuntu" distro's, and this host did not start it
presence: absent/running/found — the engine on http://127.0.0.1:7100 is the "Ubuntu" distro's and this host did not start it
hold: "Ubuntu" is held open by pid 41212 — a distro terminates seconds after the last wsl.exe session ends, whatever its units say (7b.4c)
pairing: C:\Users\tellt\AppData\Local\Crucible\pairing
door: listening on 127.0.0.1:7101
```

`absent/running/found` is 4.1a's whole point in one line. **Before this run that same machine
would have read line 3 and started a second server.**

**Nothing was disturbed, verified rather than assumed.** `crucible serve` pid **17517**
before and after; `train_lora.py` pid **18112** before and after; `wsl -l -v` lists `Ubuntu
Running 2` throughout; `GET /v1/ping` 200 throughout; six minutes of the 15 s watch produced
no state-change line, which is the watch saying nothing happened. The only `wsl.exe` calls
the host made are `-l -v`, `-l -v --running`, one `--exec cat` of the guest's pairing file,
and the held `--exec sleep infinity` — all read-only or additive, and `--terminate` /
`--shutdown` appear nowhere on any branch this machine can reach.

**Both app-side readers were exercised against the live file, and neither was told a token.**

```
@crucible/client 0.6.0 (vendored in BookForge)
  cruciblePairingPath()      -> C:\Users\tellt\AppData\Local\Crucible\pairing
  readPairingFile()          -> name crucible@owens-pc-wsl, url http://127.0.0.1:7100, token len 43
BookForge dist/electron/crucible/pairing-file.js
  cruciblePairingFilePath(h) -> the same path
  readCruciblePairingFile(h) -> the same three facts
GET /v1/ping with that token -> 200 {"crucible":true,"name":"crucible@owens-pc-wsl","api_version":1}
```

The 95 bytes are byte-for-byte the guest's own line, which is the copy rule working: the
host's `config.toml` does not exist on this machine at all, and under the old composer there
would have been no file.

**Three defects, all fixed with tests.**

1. `c59dc9d` — 4.1a's whole subject: a second server onto an occupied port, a `running`
   reported about a child that lost the bind, a pairing file composed with the wrong token,
   and a distro nobody held open. Thirteen tests, written with the shape of THIS machine.
2. `ddbac6f` — `crucible envpack build host` shipped a 46 MB pack **with no Crucible in it**
   and called it a success. With `PYTHONPATH` at the source checkout the pack's pip found the
   repo's `crucible.egg-info`, said "Requirement already satisfied", installed every
   dependency and not the wheel; `write_cmd_shims` wrote twelve correct shims for other
   people's commands and none for `crucible`; the smoke test then failed as
   `FileNotFoundError: [WinError 2]`. Now refused at build time by name, with the cause
   named, and the smoke test says "there is no crucible.cmd in it" instead of raising a
   traceback about a path.
3. A test-fixture fact worth keeping: `WINDOWS_ENV`'s fabricated `USERNAME` made the real
   `icacls` answer `No mapping between account names and security IDs was done`. On this PC
   the profile directory is `tellt` and the ACCOUNT is `telltale` — which is precisely why
   `paths.py` and `pairing.py` READ `%USERNAME%` instead of assembling a name from a path.
   The rule earning its keep on the first machine it met.

**Tests.** `tests/test_host.py` **98 passed, 1 skipped** (was 85, 1) and `tests/test_envpack.py`
green but for one pre-existing environmental failure, both on the WINDOWS interpreter
(`scripts/testrun-phase15.sh`'s T2 could not run: its lock refuses while a trainer holds the
VM, and that rule was respected). The one failure is
`test_a_baked_in_shebang_is_rewritten_to_one_that_survives_a_move`, which asserts an
executable bit on the POSIX relocation path — NTFS has no mode to report, the same family as
the already-documented 0600 skip.

**What is left for the morning.** A `windows-latest` CI run and a published host pack, so
`install.ps1` works as written (7b.6). The tray ICON itself was not looked at — the process
is up and its menu model is pinned by tests, but nobody has seen the notification area.
Foundry's reader was not exercised (BookForge's and the SDK's were). And the engine this
host found is still the hand-held `crucible serve` in `Ubuntu`, not a systemd unit the host
can restart: the unit exists and is enabled, linger is on, and its user bus is unreachable
until `user@1000` is restarted — which must not happen while the trainer is in that distro.

## 7c. What was built — the Mac

> **Numbered 7c and not 7**, because sections 7a and 7b belong to the two
> builds 4.6 hands work to and neither is written yet. This is the Mac's, and
> it is the record 4.6 asked for.

Section 4.6 listed three unserved classes in ascending cost and one owed
improvement. Two of the three now run, the third is measured and deliberately
not shipped, and the improvement landed. Everything below was done against the
Mac Studio (M1 Ultra, 64 GiB unified, macOS 26.3.1) on 2026-09-14, except the
T9 block at the end, which is 2026-09-15; no number here is a guess, and where
a figure is absent this section says "unmeasured" rather than inventing one.

### `align` — served

| what | where |
|---|---|
| recipe | `envs/align/mlx-darwin.txt` — the freeze of the Mac's `qwen-align` env, read 2026-09-14. Cuda's set minus the 19 `nvidia-*`/`cuda-*` wheels and `triton`, `platformdirs` one patch newer, everything else identical. Two lines named and left out: `conda-pack` (BookForge's packing step) and `torchaudio` (absent from the working cuda env, which is the proof it is not needed). |
| engine | `mlx-darwin: "qwen3-forced-aligner"` — the SAME engine on both backends. Not a second engine the way `asr` needs one: the aligner is plain torch and torch runs on Metal. |
| weights | `Qwen/Qwen3-ForcedAligner-0.6B` @ `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`, 1,840,072,459 B — the identical repo and revision the cuda block pins, re-read from the hub API rather than copied. |
| device | `crucible/jobs/align/__init__.py` owns a per-backend table with **no default**: `cuda-linux` → `cuda`, `mlx-darwin` → `mps`. A backend nobody decided about is a refusal, never a `cuda` handed to a Mac. The dtype still comes off the manifest. |
| **estimate_basis** | **MEASURED.** 5,885,296,640 B — the high-water of `torch.mps.driver_allocated_memory()`, sampled at 20 Hz, over three back-to-back 300-second chunks (QWEN3_MAX_AUDIO_S, the longest input this job type accepts). It settles at the second chunk and does not move on the third, which is what says ceiling rather than leak. DRIVER-allocated and not current-allocated, because torch's MPS caching allocator keeps what it takes and an align session is resident for a whole book; live tensors peaked at 3,032,348,672. It is LARGER than the cuda block's declared arithmetic, which is the argument for measuring. |
| also measured | ~77x realtime warm (900 s of audio in 11.72 s), beside BookForge's 97x on a 95 s clip. |
| **unmeasured** | **The timestamp comparison.** Nobody has aligned one chapter on both machines and compared the cues. `bfloat16` on MPS is a different numerical path from `bfloat16` on CUDA. `envs/align/mlx-darwin.md` says what the comparison is and what the bar is. |

### `asr` — served, by a second engine

CTranslate2 has no Metal backend, so this is `mlx-whisper` with its own recipe
(`envs/asr/mlx-darwin.txt`, the freeze of a scratch env built from the pin and
deleted after), its own worker (`crucible/jobs/asr/mlx_worker.py`) and **seven
new ids**. The wire is byte-for-byte the faster-whisper worker's, so
`transcript.json` is one document whichever machine made it.

**The ids never cross, and the loader enforces it.** A transcript records the
model id and nothing else about the bytes; the two libraries' "large-v3" are
different conversions at a different quantisation. `ASR_ENGINE_ID_PREFIX`
refuses a manifest whose id does not name its engine.

Every revision below was verified twice against the hub API on 2026-09-14 (main
through `/refs`, then the byte total through `/tree/<sha>?recursive=1`). Every
estimate is **MEASURED**: `mx.get_peak_memory()` over ONE 900-second window —
`WINDOW_SECONDS`, the unit a book is actually cut into — with
`word_timestamps=True`, after `mx.reset_peak_memory()` before the load.

| id | repo @ sha | repo bytes | estimate (MEASURED) | 900 s took |
|---|---|---|---|---|
| `mlx-whisper-tiny` | `mlx-community/whisper-tiny-mlx` @ `6caf9c55601caafbe6508a8b0d216bdf4783c4e8` | 74,420,620 | 549,418,642 | 11.95 s (75.3x) |
| `mlx-whisper-base` | `mlx-community/whisper-base-mlx` @ `1e3e249fb8d01c655324bd6841b1deadffd6d04c` | 143,726,326 | 877,017,662 | 13.11 s (68.6x) |
| `mlx-whisper-small` | `mlx-community/whisper-small-mlx` @ `45f3915923c7a79a5a5b5a7d909d39aeb0e5630e` | 481,309,720 | 1,540,273,318 | 31.11 s (28.9x) |
| `mlx-whisper-medium` | `mlx-community/whisper-medium-mlx` @ `7fc08c4eac4c316526498f147dfdee6f6303f975` | 1,524,927,044 | 2,607,243,002 | 36.39 s (24.7x) |
| `mlx-whisper-large-v3` | `mlx-community/whisper-large-v3-mlx` @ `49e6aa286ad60c14352c404340ded53710378a11` | 3,083,522,487 | 4,153,379,610 | 142.01 s (6.3x) |
| `mlx-whisper-large-v3-turbo` | `mlx-community/whisper-large-v3-turbo` @ `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb` | 1,613,979,758 | 2,654,916,970 | 35.43 s (25.4x) |
| `mlx-whisper-distil-large-v3` | `mlx-community/distil-whisper-large-v3` @ `e1c3c155644be59f8b477c0186719442f7e3fbb0` | 1,509,132,231 | 2,549,972,298 | 25.84 s (34.8x) |

**A finding worth the table: `large-v3` is four times slower than `turbo`** for
the same 128-mel encoder, because it decodes with thirty-two layers against
turbo's four. An 18-hour book is about three hours of Mac on `large-v3` and
about forty minutes on `turbo`.

**One value is refused rather than differing.** mlx-whisper has no VAD at all —
faster-whisper's is Silero; `no_speech_threshold` is the model's own
per-segment judgement, a different mechanism on different evidence. So
`vad_filter: true` is `400 vad_unsupported_by_engine`, asked off the BACKEND so
it lands before the env and weights checks. And one field is reconstructed
rather than dropped: `language_probability`, from whisper's own
`detect_language` on the window's first 30 s (measured: `en` at
0.9946824908256531), with the detected code then passed into `transcribe` so
the detection runs once.

**Unmeasured: ACCURACY.** Nobody has put one book through both engines. The
memory and the speed are watched; which transcript is better on Owen's material
is an open question and `envs/asr/mlx-darwin.md` says so.

### `pages` — the structure landed, the manifest block did NOT, and that is a measurement

4.6's decision is built: **`BACKEND_ENGINES` is now one engine per (backend,
class family)** — `cuda-linux` maps both text and pages to vLLM as it always
did, `mlx-darwin` maps text to `mlx-lm` and pages to `mlx-vlm`. The family is
DERIVED from `modalities` rather than declared, so `qwen3.5-9b` (a vision model
served text-only) states that fact once. `crucible/engines/mlx_vlm.py` is the
new class; `residency.load()` needed no change, because it already picked the
class off `spec.engine`.

**`models/dots-ocr.toml` still has no `[backends.mlx-darwin]` block, and the
reason changed from "nobody measured it" to "it was measured and it is wrong".**

| | |
|---|---|
| in process, `mlx_vlm.generate()` | five blocks in the `dots-json` dialect, the three body paragraphs transcribed VERBATIM, **16.32 s/page**, peak **4,927,004,359 B** |
| over its own server, same weights, same image, same prompt | `[{"bbox": [1, 0, 1008, 1008], "category": "Picture"}]`, 0.72 s |

The discriminator is the token count. The server logs `images=1` and then
`prompt_tokens=216` — the text alone; asked a 21-token question it logs
`prompt_tokens=21`. On the same machine, in the same env,
`mlx_vlm.utils.prepare_inputs(processor, images=[the page], prompts=<the same
formatted prompt>)` returns **3,464** input_ids with pixel_values of
(13800, 588) — the 13,800 patches a 1300x2112 page makes at patch 14, merged
2x2 into 3,450 image tokens. The image placeholder is never expanded on the
server's path.

Not the request shape and not the version: four shapes (image part first, text
part first, a local file path, an explicit `resize_shape`) produced the
byte-identical wrong answer, on mlx-vlm **0.6.10 and 0.7.1** alike, and
`apply_chat_template` was ruled out directly — it produces the same
`<|img|><|imgpad|><|endofimg|>` prompt from a string and from the message list
the server passes. The page was a synthetic 1300x2112 PNG (the size Foundry's
pinned `VLM_DPI = 200` makes of a 468x760 pt page) with known text; no personal
document was read.

Writing the block would light `pages: yes` on every Mac and answer every page
with one `Picture` covering the sheet — a well-formed answer in the right
dialect that is simply not the page. **What is needed is one thing:** an
mlx-vlm whose `/v1/chat/completions` reports `prompt_tokens` in the thousands
for that page. Then the block is `engine = "mlx-vlm"`,
`hf_repo = "mlx-community/dots.ocr-4bit"`,
`revision = "4ab989e403d4f8cafa5fdeede5b2290a706c2405"` (3,538,472,109 B, hub
API verified), `memory_bytes_estimate = 4927004359` re-taken through the
server, and no `--trust-remote-code` (mlx-vlm ships its own `dots_ocr` class,
unlike vLLM). `envs/llm/mlx-darwin.txt` carries the same decision about its
pin: adding `mlx-vlm==0.6.10` takes that env from 34 packages to 54, and
installing it for an engine nothing can reach is weight for a capability that
is off.

Foundry's Mac page reading is unaffected — its `mlx-local` route calls
`generate()` in process, which is the half that works, and that is also why
nobody had found this.

### `rvc` — the recipe stopped being a substitution

`envs/rvc/mlx-darwin.txt` is now `pip freeze` from `~/.crucible/envs/rvc` on
that machine, 104 packages, and the env's own stamp says this file built it
(`recipe: mlx-darwin.txt`, python 3.11.16, 43.4 s). All three CUDA→neutral
substitutions the old notice asked to have checked are confirmed.

### The owed improvement — `doctor` names both PATHs

`crucible doctor` prints `PATH (this shell)` and `PATH (the service)`, the
second read back out of the unit or plist Crucible wrote
(`service.read_recorded_path`, through `plistlib` and systemd's `%%`
unescaping), with a note when they differ saying that is normal. "No service
installed" prints as `none recorded` beside where the definition would be,
rather than as an absent line.

### Mac staging — what T9 needs before the button

Section 8's T9 asks the Mac's own server for `align`, `asr` and `pages` and
then runs one job of each. None of that is true of the Mac today: its server
runs the code from before this work, and `align` and `asr` have no env and no
weights there. This is the exact sequence that gets it there. **Run it after
the branch is merged, on Owen's word, and not before** — every step writes
something on that machine. **(It was run on 2026-09-15 and T9 followed it; the
record is at the end of this section.)**

**One correction to carry into it, and it changes the method.** `docs/PLAN.md`
records "no pack has been built on the Mac — there is no Crucible checkout on
it". There is one: `/Volumes/Callisto/Projects/crucible`, on branch
`feat/phase6-remote-render` at `22eccf0`, clean, with `origin` =
`git@github-crucible:telltaleatheist/crucible.git`. The conda env `crucible`
holds it as an **editable** install (`direct_url.json`:
`{"editable": true, "url": "file:///Volumes/Callisto/Projects/crucible"}`), and
`manifests_dir()` on that machine resolves to
`/Volumes/Callisto/Projects/crucible/models`. So the Mac does not need a wheel
built, scp'd and installed: **a `git pull` IS the deploy**, and it carries the
manifests, the recipes and the new worker with it, because all three live
beside the package rather than inside it. A wheel would in fact be worse than
useless here — `[tool.setuptools.packages.find] include = ["crucible*"]` ships
the package and `crucible/ui/` and nothing else, so a wheel install has no
`models/`, no `asr/`, no `align/` and no `envs/`, and `manifests_dir()` refuses
by name.

Every command below is one `ssh mac '<cmd>'` unless it says otherwise. The env
prefix `$C` is `/opt/homebrew/Caskroom/miniconda/base/envs/crucible` and `$R`
is `/Volumes/Callisto/Projects/crucible`.

| # | command | what it changes on the Mac |
|---|---|---|
| **M0** | `mount \| grep Callisto; /usr/bin/git -C $R status --short; launchctl list \| grep com.crucible` | **Nothing.** The three preconditions, checked before anything is written: the volume the editable install points at is mounted, the checkout is clean (a dirty checkout means somebody is working there and `--ff-only` will refuse anyway), and the service is loaded. A missing Callisto is the one failure that looks like a broken Crucible and is not. |
| **M1** | `/usr/bin/git -C $R fetch origin && /usr/bin/git -C $R checkout <merged-branch> && /usr/bin/git -C $R pull --ff-only origin <merged-branch>` | **The code.** New: `envs/align/mlx-darwin.txt`, `envs/asr/mlx-darwin.txt`, seven `asr/mlx-whisper-*.toml`, `crucible/jobs/asr/mlx_worker.py`, `crucible/engines/mlx_vlm.py`. Changed: `align/qwen3-aligner.toml`, `envs/rvc/mlx-darwin.txt`, the loaders. No `pip install` — the install is editable, `pyproject.toml`'s dependencies are untouched by this work, and a `pip install -e .` would only rewrite a `.pth` that is already right. (If a later branch DOES change `[project] dependencies`, that is the one case: `$C/bin/python -m pip install -e $R`.) |
| **M2** | `launchctl kickstart -k gui/501/com.crucible.serve` | **Restarts the server** on the new code. `-k` kills the running one first; `RunAtLoad` and the plist are untouched, so this is a restart and not a reinstall — in particular the **recorded PATH and the token are not rewritten**, which is exactly what must not happen here (`crucible service install` would rewrite the PATH with whatever shell ran it, and `ssh mac '<cmd>'` is the bare one). |
| **M3** | `sleep 5; curl -s -H "Authorization: Bearer $(grep token $HOME/.crucible/config.toml \| cut -d\" -f2)" http://127.0.0.1:7100/v1/info \| head -c 400` | **Nothing.** Proves M1+M2 took: the server answers, and `capabilities` now lists `asr` with thirteen model ids where it listed six. If it does not answer, `tail ~/.crucible/logs/serve.log` says why and nothing below should be run. |
| **M4** | `$C/bin/crucible install align --build --verbose` | **Builds `~/.crucible/envs/align/`** (a venv on the conda env's python) from `envs/align/mlx-darwin.txt` — 92 pins, ~2 GB with torch 2.14.0. `--build` is REQUIRED and not a preference: no release publishes an `align/mlx-darwin` pack yet (this branch adds the CI row; the pack appears on the next tag), so a plain `crucible install align` refuses `pack_not_published`. On success the command also **writes `[jobs] enable_align = true` into `config.toml`**, which is how the flag gets turned on — do not hand-edit it. |
| **M5** | `$C/bin/crucible install asr --build --verbose` | **Builds `~/.crucible/envs/asr/`** from `envs/asr/mlx-darwin.txt` — 34 pins, ~2 GB (mlx-whisper declares torch). Same `--build` reason, same flag write (`enable_asr = true`). Its headline package is `mlx-whisper`, not `faster-whisper`; a doctor that says otherwise means M1 did not take. |
| **M6** | `$C/bin/crucible models pull qwen3-aligner` | **~1.84 GB into `~/.crucible/models/qwen3-aligner/mlx-darwin/`** plus a `crucible-pull.json` stamp at revision `c7cbfc20…`. **It will re-download even though the snapshot is already in `~/.cache/huggingface/hub`**: `weights.pull` passes `local_dir=`, which writes the tree directly and does not read the shared cache. That is a network cost, not a defect, and it is the reason this step is minutes rather than seconds. |
| **M7** | `$C/bin/crucible models pull mlx-whisper-large-v3-turbo` | **~1.61 GB into `~/.crucible/models/mlx-whisper-large-v3-turbo/mlx-darwin/`** at `a4aaeec0…`. **Turbo and not large-v3**, deliberately: measured at 25.4x realtime against large-v3's 6.3x for the same encoder, so T9's asr job finishes in a quarter of the time and proves exactly as much. Pull `mlx-whisper-large-v3` too only if T9 is meant to measure accuracy, which it is not. |
| **M8** | *(skip — see below)* `$C/bin/crucible models pull dots-ocr` | **Would fetch ~3.5 GB and light nothing.** `models/dots-ocr.toml` has no `mlx-darwin` block, so the pull refuses `backend_unsupported` by name. Leave it out of the staging run; the page half of T9 cannot pass on this machine and the next row says what to do instead. |
| **M9** | `$C/bin/crucible capability --write` | **Rewrites the `[capability]` record in `config.toml`** — the classes, the selected id per class and the reason. After M4–M7 this is what turns `align` and `asr` from "this build ships none with a mlx-darwin block" into `yes`, with `qwen3-aligner` and `mlx-whisper-large-v3-turbo` selected (best-first by declared size; turbo wins only if large-v3 was not pulled — if both are installed the record will name `mlx-whisper-large-v3`, which is correct and slower). |
| **M10** | from a **login** shell: `ssh mac -t 'bash -lc "$C/bin/crucible doctor"'` | **Nothing.** The verification, and `-t 'bash -lc'` is not decoration: a plain `ssh mac '<cmd>'` gets `/usr/bin:/bin:/usr/sbin:/sbin` and would report `job tts: NOT READY — there is no ffmpeg on PATH` on a perfectly healthy host. Since this branch, the doctor prints **both** PATHs, so a run from either shell now shows the discrepancy rather than being misled by it. Expect `align env: ready`, `asr env: ready — mlx-whisper 0.4.3, …`, `capability align: yes`, `capability asr: yes`, and `healthy`. |

**Then T9, with one stage of it struck.** `align` and `asr` will answer: submit
one align job and one asr job against the Mac's server and record the figures
(this repo's own measurements, for comparison: the aligner runs at about 77x
realtime warm on 300-second chunks, `mlx-whisper-large-v3-turbo` at 25.4x on a
900-second window).

**`pages` will report `no` on the Mac and T9's third artifact cannot be
produced.** That is not a staging failure and no amount of installing fixes it:
`models/dots-ocr.toml` has no `mlx-darwin` block because mlx-vlm's own HTTP
server does not put the image into the prompt (the run is above, in this
section). T9 should be amended to two artifacts on the Mac, with the page
reading proved on `cuda-linux` by T6 and on `llama-windows` by T7 as it already
is. If the button is meant to *demonstrate* the Mac reading a page anyway, the
only honest way today is Foundry's in-process `mlx-local` route, which is a
different program and not Crucible answering.

**What the staging does NOT do, and each is deliberate.** It does not run
`crucible service install` (that would rewrite the recorded PATH from a
non-login shell — the exact bug the audit found). It does not touch
`~/.crucible/config.toml`'s token, `voices/`, `rvc/` or the three existing
envs. It does not start the conda removal from the audit's §3 — that is a
separate, ordered operation and doing it in the same session would put a 3 GB
env rebuild in the middle of a test run. And it installs nothing on the Mac
outside `~/.crucible/envs/{align,asr}` and `~/.crucible/models/`, both of which
`crucible` owns.

### T9 — the M-steps ran, and so did T9, 2026-09-15

Run from the Mac, as 7.6 decided it had to be. The M-steps above took the
checkout, the two envs and the two weight sets; **T9 itself ran in 94 s**:

| job | engine | measured |
|---|---|---|
| `asr` | `mlx-whisper-large-v3-turbo` | **27 s** on a 15-second clip, **including the engine start**; the transcript is correct verbatim |
| `align` | `qwen3-aligner` | **49 s**; `alignment.json` 2 509 B |

**And the refusal held where it matters**: `vad_filter: true` came back
`vad_unsupported_by_engine` on mlx-whisper, off the BACKEND and before the env
and weights checks, exactly as this section says it should. That is the one
value in the wire that is refused rather than quietly differing, and it is now
refused on a live server rather than in a test.

**`pages` was struck from T9 before the run, not failed during it** — the
mlx-vlm finding above is why, and nothing about the run changes it. Two
artifacts came back, which is what an amended T9 asks for.

**Both of this section's "unmeasured" lines still stand.** T9 is one clip and
one chunk on one machine; it is not the ALIGN TIMESTAMP COMPARISON (nobody has
aligned a chapter on both machines and diffed the cues) and it is not ASR
ACCURACY (nobody has put a book through both engines). `envs/align/mlx-darwin.md`
and `envs/asr/mlx-darwin.md` still say what those comparisons are.

### What a Mac still cannot do, and it is not on this list by accident

`pages`, until the mlx-vlm defect above is fixed. Everything else in 4.6's
"served today" row is unchanged: `tts`, `llm`, `rvc`, `denoise`. The upgrade
off conda (4.6's checklist, the audit's §3) is untouched by all of this and
still has to be done in that order.
