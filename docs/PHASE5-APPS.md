# Phase 5: the apps, the bootstrapper, and friends

What changes in BookForge and Foundry once the server can do the work, in what order, and
what has to be decided before the first file is deleted. Written 2026-09-13 from
`docs/CLIENT-SURFACES.md` sections 2, 5 and 8.

Crucible's own rule up to here has been **the app changes little or nothing**: phases 1 and
2 shipped with the CLI as the only consumer, no UI, no IPC, no settings row. That was right
while the server was unproven. This phase is where it ends, and the order below exists so
that it ends one deletion at a time with a way back from each.

## 1. The finding this phase is built on

`electron/text-server.ts` is 1,364 lines and it is **a line-for-line rehearsal of
Crucible's phase 2**: manifests with pinned revisions, an env, a managed subprocess, a
readiness probe on `/v1/models`, one resident model at a time, refusal by name, an
accelerator guard, a `/v1/openai` passthrough. It already serves `foundry clean-text`,
`translate` and `simplify` on Owen's PC.

That is the good news and it is also the warning. Two of its behaviours are **not** in
Crucible, both deliberate, and both have to be answered here rather than discovered when
the file is deleted:

**Yielding.** The text server registers itself as the *low-priority* holder of the card
(`GPU_OWNER_TEXT = 'vllm:text'`) with an `onYield` that steps off when a render asks for the
GPU. Crucible has no yield: it has a queue, and inside one Crucible the queue is the whole
answer. Between a Crucible and anything else — Owen streaming, a fine-tune, a second
server — there is only `GET /v1/accelerator`, which reports and never evicts. **So the
migration trades a cooperative yield for a serialising queue.** That is a better contract
for jobs that go through the queue and a worse one for the case the yield was built for: a
long render arriving while a short cleanup holds the card. The honest resolution is that
both go through the same lane, and a render no longer has to wait for a *process* to notice
and step aside; it waits for a job to finish, which is bounded and visible. Write that down
in the app when the holder is removed, so nobody re-adds a yield to a server that has a
queue.

**Staging and adoption.** The text server downloads absent weights rather than refusing
(resumable, pinned, progress every two seconds), and it *adopts* a server already answering
on 8300 with the right served name instead of starting a second one. Crucible refuses a
model that is not installed (`model_not_installed`) and owns its own engines. Adoption has
no analogue and needs none — Crucible is the thing that would be adopted. Staging does:
`crucible models pull` exists, but nothing in the app calls it, so the first thing a
BookForge pointed at a fresh Crucible will do is refuse by name. The settings row has to be
able to trigger a pull and show its progress, or every new server is a manual step in a
terminal.

## 2. The Servers row

`electron/crucible/servers.ts` already holds the registry
(`<userData>/crucible-servers.json`, `{name, url, token}`) and has since phase 1 with no UI
over it. The row gives it one:

- add / edit / remove an entry, with a **Test** button that calls `ping` then `info` — the
  two-step matters, because `ping` is unauthenticated and `info` is not, so the pair tells
  "wrong token" from "not a Crucible" from "nothing there", which is exactly why the SDK
  has three different error types for it.
- per server, what it advertises: backend, GPU, models, voices, what is resident.
- **Pull** a model or voice this server does not have, with progress (section 1).
- Load / unload, because an operator deciding what is resident is the model this whole
  design rests on and there is currently nowhere to do it but a CLI flag.

`checkProviderConnection` already takes a `crucibleServer` parameter that the app's IPC
handler deliberately never passes (`ai-bridge.ts`). Passing it is the smallest possible
first step and should be the first commit of this phase.

## 3. A server is a column on a queue row

Owen's ruling, verbatim: *"it needs to know that it got an order from the operator to
narrate this book and it has access to three servers. the operator says to run it on server
2. or the queue says to run it on server 2."*

So a queue row carries a server, chosen by the operator or by the queue, and every GPU step
resolves it through the registry. **One job, one server** — DESIGN.md section 6 — because
a book split across backends is two different voices.

**The hazard, and it is already real:** a stored queue row in a project's `project.json`
**freezes its `model` / `server` / `ollama` / `concurrency` at enqueue time**. A resumed
job asks for what the row recorded, not what settings say now. Pointing a machine at
Crucible does not repoint rows already queued. That is correct behaviour and it will look
like a bug on the first migration, so the migration says it out loud and the row shows
which server it was enqueued against.

## 3.1 Two consequences of a server being somewhere else

Both of these are small, both are already decided by code that exists, and both will look
like bugs on the first migration if they are not written down first.

**A remote Crucible is a `cpu` step.** `resourceForProvider` (`queue-steps/runtime.ts:37`)
routes an AI pass to `cpu` when the provider is `claude` or `openai` and to `gpu` when it is
`ollama` or `local`, on the stated ground that "a hosted API is network latency and nothing
else". A Crucible is whichever of those it happens to be: a server in this machine's WSL is
using **this** card and must hold the single `gpu` slot; a server on the Mac or a droplet is
network latency and must not. So the resource is a property of the **registry entry**, not
of the provider name — and a registry entry needs to say which it is. Resolving it from the
URL (loopback means local) is a guess that is wrong for a tailnet address pointing at this
same box. Ask the server: `/v1/info` already reports the host, so an entry can record
whether that host is this one, at the moment it is added.

**Admission becomes a network call, so it needs a deadline and a memory.** Today a GPU step
holds while `external-gpu-job.lock` exists and while the in-process arbiter reports a
foreign holder — both local reads, both instant. `GET /v1/accelerator` is neither. The
admission check has to cache its answer for the length of its recheck interval (15 s today)
and, crucially, **decide what an unreachable server means**. It means the step does not
start and the row says why: a queue that proceeds when it cannot see the card has
reinvented `gpu-arbiter.ts:88`'s `timeoutMs` path, which proceeds **without** the lock and
is the single worst line in the app's arbitration.

## 4. The 8766 relay, which is how Sunday keeps working

BookForge's TTS WebSocket on 8766 is what the browser extension talks to, and Owen uses it
every Sunday. It **stays exactly where it is.** What changes is what is behind it: instead
of `orpheus-worker-pool.ts` spawning `python -m narrator.serve` and speaking JSON-lines to
it over a pipe, the pool becomes a relay to the streaming session on the chosen server —
`POST /v1/tts/stream`, its SSE event stream, and posted ops.

The extension's protocol does not change, the port does not change, and the failure mode
if the relay is wrong is immediate and obvious rather than subtle. That is the whole reason
to keep the port rather than point the extension at Crucible: the extension is the one
client Owen uses without looking at it.

`PHASE3-TTS.md` section 7's frames were shaped to make this relay thin — client-assigned
row ids, base64 PCM per row, out-of-order retirement — because a relay that has to re-window
or re-order audio is a relay that will drift. The base64 is not a cost here at all: the pool
already hands the extension base64 PCM16, so the relay forwards the encoding it is given
rather than decoding and re-encoding it.

## 5. What retires, in the order it is safe to retire it

Each line only after the thing replacing it has rendered or read something real.

| Goes | Because | After |
|---|---|---|
| `electron/text-server.ts` (1,364 lines) and `docs/TEXT-SERVER.md` | `llm` + phase 3a | Foundry's clean / translate / simplify run through a Crucible URL |
| `electron/vlm-page-server.ts` (479), `wslVlmRefusal`, `useWsl2ForVlm`, `wslVlmCondaEnv`, `wslVlmModel`, the `wsl-server` arm of `resolveVlmRoute` | phase 3c | one real page read matches what the `dots` env produces |
| `foundry-app/electron/vllm-server.ts` | phase 3c | the same — **this is the path that reserves half the card with no arbitration at all** |
| the WSL spawn, path rewriting and VRAM arithmetic in `parallel-tts-bridge.ts`, including the lease bug at `:7229` that returns before acquiring the mutex | `tts` render | a chapter renders through Crucible and assembles unchanged |
| `orpheus-worker-pool.ts`'s spawn half | `tts` stream | the extension plays a chapter on a Sunday |
| the `viaWsl` refusal in `whisperx-align-bridge.ts:642` | `align` + `asr` | whole-m4b alignment runs on the PC **for the first time** |
| the 96-file process-recycling wrapper in `rvc-bridge.ts:356` | `rvc` | a sentence set converts identically |
| `external-gpu-job.lock`, the in-process arbiter's `timeoutMs` path that **proceeds without the lock**, and most of `wsl-lifecycle.ts`'s GPU-teardown ladder | the accelerator probe | `setGpuHolderProbe` points at `/v1/accelerator` |

`mlx-local` stays: it is the Mac's only page-reading route until `mlx-darwin` has a measured
dots.ocr block. `llama-bridge.ts` stays: the bundled llama.cpp is the offline story and
Crucible is not one.

## 6. The bootstrapper

### 6.0 THE RULING: a local Crucible is a SERVICE, and the bootstrapper ships with Crucible

**Owen, 2026-09-13, asked whether a local Crucible is a machine service or an app's child
process: "service".** Two consequences he left to me, both recorded here.

**A local Crucible is a service on the machine. No app owns it.** `crucible serve` becomes
a systemd unit on `cuda-linux` (inside WSL2 on a Windows host) and a launchd agent on
`mlx-darwin`, and an app's job shrinks to *make sure this machine has one, and make sure it
is running*. Everything below follows from that one word, and most of it is a deletion:

- **Nobody owns it, so nobody has to be chosen as the owner.** The question that produced
  this ruling — when Foundry is hosted inside BookForge, which of them installs and starts
  the server — simply stops existing. Both call an idempotent `ensureRunning()` and connect.
  Two apps racing is a no-op rather than a conflict.
- **It survives the app.** A Crucible holding a 19 GB model must not die because somebody
  closed a window, and a render must not die with it. As a child process it would.
- **The orphaned-CUDA hazard goes away by construction.** Foundry's `mount.ts:988` names it:
  a crash leaves a guest process holding the card with nothing to SIGTERM it. Nothing can be
  orphaned that was never anyone's child.
- **Section 7's second question is answered, and it was already leaning this way.**
  "Does BookForge ever start a server it did not install?" — the ordinary case is that it
  *never* starts one: it asks the service. A registry entry whose URL answers `ping` is a
  server, whoever started it.

**The bootstrapper ships WITH Crucible, as `@crucible/bootstrap`** — a sibling of
`@crucible/client`, zero runtime dependencies, released from this repo at the same version
by the same `release.sh`. It does **not** live in each app, which is what DESIGN.md section
1 said until today and what this supersedes.

The reason is the lesson of 2026-09-13, stated in ARCHITECTURE.md as R1. Two apps each
writing "detect the host, install a server, start it, stop it, report health, make a model
resident" is one fact with two owners, and it would be the largest instance yet — because
it is the piece that has to work on a machine nobody has ever logged into. Shipping it with
the server also inherits the property `release.sh` already enforces for the client: **a
bootstrapper can never be paired with a server nobody tested it against.**

Its surface is small, and every verb is idempotent:

```
install()            # this host has a Crucible, at a version this build knows
ensureRunning()      # the service is up; a no-op when it already is
ensureResident(id)   # that model is on the card (PHASE7-LANES.md section 9.2)
health()             # what /v1/activity says, or why it cannot be reached
```

**What consumes it, and what must not.** The app half of an Electron product consumes it.
An *engine* — Foundry's single compiled binary, which runs on machines with no app, no
Electron and no `node_modules` — must never touch it, and that is already Foundry's own
written rule: *"foundry uses a vLLM server and never starts one — launch it with `vllm
serve <model>`, or name the machine that has it."* The engine consumes an endpoint. It does
not stand anything up, and it does not learn what a Crucible is.

**The vendored Foundry app needs none of it.** Hosted inside BookForge it consumes an
endpoint like any other client, so it declares no dependency on `@crucible/bootstrap` and
the drift surface of two declarations over one runtime — which the Foundry session raised —
never opens. Only a standalone Foundry would depend on it, and only to reach the same
service.

### 6.1 What it still has to do

It lives in the app layer, not the server, and it knows how to: detect the host, install a
local server, ensure the service is running, and report health.

On Windows that means **inside WSL2**, driven by `wsl.exe -d <distro> --exec bash -c ...`
and never the implicit shell, which pre-expands `$var`. The app already has every piece of
that machinery — it is what `text-server.ts` and `vlm-page-server.ts` do — so the
bootstrapper is largely those two files' *good* half, kept while their model-serving half
is deleted.

**Starting is `systemctl --user start` / `launchctl kickstart`, not a spawn.** That is the
whole difference in code, and it is why the app half of this is small: the machinery
`text-server.ts` and `vlm-page-server.ts` carry for spawning, supervising, ring-buffering
and tearing down a model server is exactly the half that is being deleted, not kept.

A Docker image for `cuda-linux` is the alternative worth having for a friend, because
"install WSL2, then conda, then an env" is not an onboarding path. The image is the server
plus one job type's env; models and voices are pulled at first use.

Then a friend gets the client tarball, a tailnet invite, and a token — and the reason this
works at all is DESIGN.md section 1's rule that the client speaks HTTP even to a server it
just spawned on localhost. There is no second code path to write for them.

## 7. Two things to decide before building, not during

**Does a resident model ever unload itself?** The text server has `noteTextQueueIdle`, a
keep-warm window with a ceiling of 240 minutes, defaulting to "stop when the queue drains".
Crucible has nothing: what is resident stays resident until an operator or a job says
otherwise. For one machine with one operator that is right — an idle unload is a 110-second
reload the next time anyone types — but it means a forgotten model holds a card overnight.
The probe makes that visible rather than fixing it. ~~**Proposed default: no idle unload,
and the Servers row shows what is resident and for how long.** It is a config key, not a
behaviour change, if Owen disagrees.~~

> **OVERRULED, 2026-09-14. Owen:** *"Models should always be unloaded when we're done with
> them. Every time."*
>
> He disagreed, and the proposal above had already been answered by what happened the night
> before: a 9B sat resident after the Foundry proof until a person unloaded it, and Owen saw
> *"something loaded and nothing happening"*. The card is not storage, and the "forgotten
> model holds a card overnight" this section wrote down as a cost is the whole of the
> defect.
>
> **And it is not a config key**, which is the part of the proposal that was most wrong. A
> keep-warm window is a fact standing in for a guess — ninety idle seconds is evidence of
> nothing — so there is no window, no timer and no key that turns the rule off. "Done" is
> read from **four facts**: no job on the lane, no lease open, no streaming session holding
> the claim, no chat in flight. The moment the last of them goes false, the resident thing
> is unloaded and says why. **PHASE7-LANES.md section 5.3** is the ruling;
> `crucible/settle.py` is the code.
>
> What makes it safe rather than a 44-second reload between every book is section 5.2's
> lease, built the same night. The consequence for the app is therefore this section's to
> carry: **a BookForge chat run with no lease open reloads its model per request**, because
> `electron/ai-bridge.ts`'s `crucible` provider neither loads nor leases. The fix is a lease
> at BookForge's door, and it belongs on this phase's list rather than in the server.

**Does BookForge ever start a server it did not install?** Adoption (section 1) is the text
server's answer to "something is already on 8300". For Crucible the equivalent question is
whether the bootstrapper may attach to a Crucible it finds running rather than starting its
own. **Proposed: yes, and it is not adoption but the ordinary case** — a registry entry
whose URL answers `ping` is a server, whoever started it. The bootstrapper starts one only
when no entry answers.
