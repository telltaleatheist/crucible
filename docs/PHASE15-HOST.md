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

`settings()`, `putSettings(patch)`, `testUpstream(name, probe?)`, `readPairingFile(home?)`
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
2. **The weights** are the existing `pages` catalog subject (`dots-ocr` → `anthonym21/
   dots.ocr-GGUF` @ `42ab310215a26d05ebe21ccc55f64db6c2bfc6ce`, `Dots.Ocr-1.8B-Q8_0.gguf` +
   `mmproj-Dots.Ocr-F16.gguf`, 4.42 GB, ~5.9 GB needed) pulled through the catalog like any
   subject. A pull that has the text tower and not the mmproj is INCOMPLETE and the subject
   says `installed: false` — the mmproj is not optional.
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

**Exit for Foundry's package L:** `pages` answers `enabled: true` from a `llama-windows` server
on a clean no-WSL Windows box and a real page comes back parsed; for text, `clean` answers the
same way from the 9B GGUF or an upstream. Then `page-reader.ts` and every
"can this machine do it" line in Foundry go.

## 4. The host — `crucible host` on Windows

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

### 4.2 The menu

`Crucible — running (WSL)` / `running (llama-windows)` / `stopped` / `installing…` as the title
line, then: **Open console** (the pairing line's URL with `#token=`, PHASE13 5.3 — the same
hardened window rule applies to a browser: it is the default browser), **Install the
WSL2 engine (faster pages and text; TTS, ASR…)…** (only when the distro is absent — runs section 4.3), **Restart
engine**, **Stop engine**, **Open log**, **Quit** (stops the host; the WSL server keeps running
because it is systemd's; in host mode the child server stops with it, and the menu says so).

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

**The host has NO window of its own (amended by 4.7).** Its UI is the tray menu and the
operator PAGE. Because `llama-windows` runs on any Windows machine, the host starts the
Windows server within seconds of install and opens the page; the WSL install is then the
page's engine switch (4.7), shown as a task in the page's Tasks panel like a pull. The
states that need a reboot are answered by the task saying "reboot, then Crucible continues"
and the Startup item resuming the task and reopening the page. There is no tkinter, no
second progress UI, no second owner of the sequence.

### 4.4 Packaging

- A Windows host pack: `crucible-env-host-windows-<version>.tar.zst`, built by the same
  `crucible envpack build host` on a `windows-latest` runner: python-build-standalone
  `x86_64-pc-windows-msvc-shared-install_only` (pinned in `STANDALONE_PYTHON` with its
  SHA256SUMS digest like the two others) + the server package + `pystray` + `pillow` (the
  tray). Unpacked to `%LOCALAPPDATA%\Crucible\host\`. It is the ONLY Windows-native Python
  Crucible ever ships, and it never gets a backend.
- `install.ps1` (generated, PHASE14 7b.2) becomes: download the host pack for this version,
  verify, unpack, write the Startup shortcut, start `crucible host`, and STOP — the host takes
  it from there (4.3) and shows the WSL steps in its own window. `install.sh` is unchanged
  (linux/darwin have no host).
- The release gains the host pack in `envpacks.json` (backend `host-windows`) and CI gains the
  `windows-latest` job. release.sh's asset list is updated (seven assets now: the installers
  are unchanged in count).
- **Mac:** no host. launchd already supervises; the pairing file (3.6) is what an app reads.
  A menu-bar item is out of this phase.

### 4.5 Tests

sdk/bootstrap: the shrunk `install()` (host present / absent) with a fake host. pytest: `crucible
host` is refused off win32 (`host_windows_only`); the menu model as a pure function of
(distro state, ping state); the Startup shortcut path; the migrate step's `--config-from`; the
pairing-file switch. The tray itself is exercised by hand on Owen's PC and the doc records
what was seen.

### 4.6 The Mac — no host, and what "works out of the box" is measured against

**Owen, 2026-09-14:** *"we'll have to make sure crucible works on mac as well. it wouldnt need a
wsl sidecar for mac obviously. it would just function out of the box with mlx-audio and
everything we have configured for mac bookforge."*

The Mac needs no host: `mlx-darwin` is a native backend, launchd supervises it, `install.sh`
installs it, the pairing file (3.6) connects a local app. What "out of the box" is measured
against is `crucible doctor` on the Mac Studio, read 2026-09-14 (crucible 0.6.0, M1 Ultra
64 GB, macOS 26.3.1): **tts (Higgs via mlx-audio 0.4.8), llm, rvc, denoise READY, all
seven voices fit**; and three classes that are NOT served on this backend today —
`pages`, `asr`, `align` answer "this build ships none with a mlx-darwin block". Two defects
seen in the same read: launchd's bare PATH has no ffmpeg, so `tts` is `job_type_not_ready`
under the service even though the env is ready (the Phase 14 PATH-in-plist fix post-dates
the Mac's plist); and a stale `tts env (orpheus)` that the Orpheus removal must delete on
upgrade.

Owed, in order, and NOT built in this phase until the read-only audit reports: an
`mlx-darwin` block for `asr` (mlx-whisper), `pages` (dots.ocr under mlx-vlm — Foundry's
Mac path answered the parser's dialect), and `align` (only if an MLX aligner genuinely exists;
"align has no Mac engine" is an honest capability answer and a guessed block is not); the
Mac's upgrade off conda onto the release's server pack; `service install` re-run for the PATH.

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

### 5.4 The single re-vendor (BookForge ← Foundry)

Moves again: it now targets the Foundry sha AFTER their 5.3 lands, and carries the items the
rollout plan already lists (`RunOptions.waitFor`, `hosted_placement_not_vendored` deletion,
`slots?()` removal) plus the cloud card's replacement.

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
