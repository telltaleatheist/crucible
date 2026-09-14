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

**One server per machine, still.** The Windows half of this phase is NOT a second Crucible and
NOT a relay. Two processes forwarding to each other would be a fact with two owners (two
versions, two health states, a hop on the TTS stream). The server that answers `:7100` on a
Windows machine is the WSL one when WSL is there, and a **host-mode** server (section 3.4)
when it is not. Never both. Apps connect to the same address either way.

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
| **host mode** | a server with `backend_kind = "none"`: no accelerator, no card job types, only `echo` and the upstream routes. What runs natively on a Windows machine without WSL. |
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

- `backend_kind` gains the value `"none"` (host mode). `crucible init --backend none` is legal
  only on win32 and is what `crucible host` runs; on linux/darwin it is refused
  (`backend_none_not_here`) because those machines run the real server.
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
  qwen3.8-27b-4bit" without a second call.
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
- **In host mode** every card class answers `enabled: false, route: "local"` with the reason
  `this machine has no accelerator backend; on Windows the engine runs inside WSL2 — install it
  from the console` (one sentence, the same for every class, so an app shows it once). The
  four llm classes may still route upstream and then answer `enabled: true`.

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
  forwarded blind.
- `GET /v1/openai/models` lists local models as today PLUS, for each configured upstream, the
  routed upstream models (only the ones a route names — not the upstream's whole catalog,
  which is `test`'s job).
- A lease on an upstream model (`POST /v1/models/{id}/lease`) is refused `lease_not_needed`
  with the sentence "an upstream model is never resident; send the chat". Same for
  `{"type": "load-model"}` naming one.

### 3.5 Host mode

- `crucible serve` no longer refuses win32 outright. `main()`'s refusal narrows to: on win32,
  every verb runs, and `backend_kind` must be `"none"` — a config naming `cuda-linux` on win32
  is refused `backend_not_here` (the CLI and the doctor both), because **Windows is never a
  backend** and this keeps the sentence true.
- Host mode has: `echo`, the settings door, the page, `/v1/setup`, `/v1/capability` (3.3's
  host answer), `/v1/info` with `backend_kind: "none"`, chat completions to upstreams. It has no
  `install` task (refused `no_backend`), no catalog subjects (`/v1/catalog` returns empty lists
  and `backend_kind: "none"`), no accelerator (`/v1/accelerator` → `no_backend`), no lease.
- `crucible doctor` in host mode prints the one line "backend: none — host mode on
  windows/x86_64; the accelerator engine runs inside WSL2 (see `crucible host`)", then the
  upstream lines.
- Nothing in host mode is a stopgap for WSL. When WSL arrives, the host (section 4) moves the
  config — token, routes, upstreams — into the guest and STOPS the host-mode server. The token
  survives the move, so every app that paired stays paired.

### 3.6 The pairing file

`crucible init` and `crucible service install` write the pairing line to a user-only file
beside the config: `<CRUCIBLE_HOME>/pairing` (mode 0600 on linux/darwin; on Windows the file
is written by the host, section 4.3, with an ACL of the current user only). One line, the
`127.0.0.1` pairing line, trailing newline. `crucible token --url` prints the same. An app on
the same machine reads it (5.1) and never asks a person to type a token. Rotating the token
rewrites it.

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
says "host mode — no accelerator on this machine" and Job types / Catalog say what 3.5 says.

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
tool-forced JSON); host mode's answers for every route above; `backend_not_here`; the pairing
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

**What it is.** Host mode gains one small backend, `llama-windows`: a `llama-server` child the
host-mode server spawns and kills, serving the `pages` class from the dots.ocr GGUF pair. It is
NOT a Python env, NOT a pack, has no lease table of its own (one child at a time IS the
arbitration; the host-mode server has no other card work), and it never serves `tts/asr/
align/rvc/denoise`. `backend_kind` for such a server is still `"none"`; the child is a property
of host mode, reported in `/v1/info` as `pages_engine: "llama-windows"`. The spec is Foundry's
working launcher, handed over at `C:\tmp\foundry-page-reader-spec\` (README first — eight
load-bearing facts; `page-reader.ts`'s header argues every constant).

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

**Exit for Foundry's package L:** `pages` answers `enabled: true` from a host-mode server on a
clean no-WSL Windows box and a real page comes back parsed. Then `page-reader.ts` and every
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
| `running` | `GET /v1/ping` answered | `Crucible — running (WSL)` / `Crucible — running (host mode)` |
| `stopped` | the ping failed and this down-edge's recovery is spent | `Crucible — stopped` |
| `failed` | both recipes ran and neither brought it up | `Crucible — engine did not start — open the log` |
| `installing` | the 4.3 sequence is running | `Crucible — installing…` |

The recovery recipes are named. With the distro: `user-unit-start`
(`systemctl --user start crucible` inside it), then `user-bus-restart`
(`systemctl restart user@1000` as root, through `wsl -d crucible -u root`), in that order, at
most once per down-edge. In host mode there is one: `host-mode-respawn`, which starts
`crucible serve` as a child again. `BOOT_WAIT_SECONDS = 30` and `WATCH_SECONDS = 15` are the
two numbers 4.1 states, and they are constants with those names.

**The Startup verbs.** `crucible host --install-startup` writes the shortcut and prints its
path; `crucible host --remove-startup` deletes it and says whether there was one. Both exit
without starting a tray, and both are the ONE owner of that file — `install.ps1` calls the
first rather than writing a `.lnk` of its own. The shortcut is written by
`crucible/host/startup.py` through a PowerShell `WScript.Shell` one-liner (no pywin32, no new
dependency) and its target is `pythonw.exe -m crucible.cli host`, not the `.cmd` (4.4): a
`.cmd` opens a console window, and a tray program has none.

### 4.2 The menu

`Crucible — running (WSL)` / `running (host mode)` / `stopped` / `installing…` as the title
line, then: **Open console** (the pairing line's URL with `#token=`, PHASE13 5.3 — the same
hardened window rule applies to a browser: it is the default browser), **Install the
accelerator engine (WSL2)…** (only when the distro is absent — runs section 4.3), **Restart
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
capability write; stop the host-mode child; switch the pairing file to the guest's line (the
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

**The 4c table crosses into Python by GENERATION, not by a second copy.** `crucible host` is
Python and the table is `sdk/bootstrap/src/wsl-states.ts`, so its DATA — the code, the probe
argv, the sentence, the action — is EMITTED into `crucible/host/wsl_states.py` by the same
`scripts/gen-install-scripts.ts` that writes `install.ps1`, and `npm run gen:install --
--check` refuses a drift exactly as it does for the two scripts. What is spelled twice is only
what `steps.ts` already spells twice for the same reason: the PREDICATES (`means`), which are
code rather than data. `crucible/host/wslstate.py` holds one per code and a test asserts the
two sets are equal — the "tied by a check instead of by an import" seam `envpack.SMOKE_IMPORT`
and `cli.INSTALLABLE_JOB_TYPES` already use.

**The window** is tkinter (which is in the pinned interpreter, 4.4): one row per step with a
state beside it, no console. `crucible host --install` runs the same sequence in the
FOREGROUND and prints the events, which is how it is tested and how a support session reads
it.

**The migrate step, named.** `crucible init --config-from <file>` is the flag 4.3 asks for:
the file is a TOML document the host wrote at 0600 and deletes afterwards, and `init` takes
exactly three things out of it — `auth.token`, `[routes]` and `[upstreams]` — and nothing
else. The host, the port, the name, the backend and the job flags belong to the machine being
initialised, not to the one being left. A file without `auth.token` is refused
`config_from_no_token`; a file that is not TOML is `config_from_unreadable`.

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

- The pack's name is `host`, its backend kind is `host-windows`, and its archive is
  `crucible-env-host-windows-<version>.tar.zst`. `host-windows` is a PACK BACKEND and never a
  `backend_kind` in a config: a config on this machine says `none` (3.5). The pack table's key
  is about which wheels pip resolved; **Windows is never a backend** and nothing in this pack
  runs a model.
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
`host-windows` row in the pack table and its refusal off win32; and the loopback door's four
refusals. Windows-only code paths take the platform as an argument so they run on both — the
suite runs in WSL (`pytest`) and must not be a suite that skips its subject.

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
names it. The keeper suite pins every deleted door by name (as `test-no-e2a-doors.js` does).

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

- A Mac menu-bar host. launchd covers supervision; the pairing file covers connect.
- Per-request cost or token accounting for upstreams. `/v1/activity` records the act and the
  model; a usage figure is the upstream's dashboard's until somebody asks for it here.
- Foundry's NLI analysis worker as its own engine class (3.10's line puts it on the engine's side).
- Routing any non-llm class upstream (a cloud TTS, a cloud ASR). `route_not_routable` is the
  door, and it opens when there is a reason.
- Deleting BookForge's legacy local spawn layer (`'local'`, the WSL bridges, ollama's remaining
  doors). That is the in-app-pass deletion already on the rollout plan.
