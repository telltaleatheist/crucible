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

## 4. The 8766 relay, which is how Sunday keeps working

BookForge's TTS WebSocket on 8766 is what the browser extension talks to, and Owen uses it
every Sunday. It **stays exactly where it is.** What changes is what is behind it: instead
of `orpheus-worker-pool.ts` spawning `python -m narrator.serve` and speaking JSON-lines to
it over a pipe, the pool becomes a relay to `GET /v1/tts/stream` on the chosen server.

The extension's protocol does not change, the port does not change, and the failure mode
if the relay is wrong is immediate and obvious rather than subtle. That is the whole reason
to keep the port rather than point the extension at Crucible: the extension is the one
client Owen uses without looking at it.

`PHASE3-TTS.md` section 7's frames were shaped to make this relay thin — client-assigned
row ids, binary audio frames, out-of-order retirement — because a relay that has to
re-window or re-order audio is a relay that will drift.

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

The third role in DESIGN.md section 1, and the only one still unbuilt. It lives in the app,
not in the server, and it knows how to: detect the host, install a local server, start and
stop it, and report health.

On Windows that means **inside WSL2**, driven by `wsl.exe -d <distro> --exec bash -c ...`
and never the implicit shell, which pre-expands `$var`. The app already has every piece of
that machinery — it is what `text-server.ts` and `vlm-page-server.ts` do — so the
bootstrapper is largely those two files' *good* half, kept while their model-serving half
is deleted.

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
The probe makes that visible rather than fixing it. **Proposed default: no idle unload, and
the Servers row shows what is resident and for how long.** It is a config key, not a
behaviour change, if Owen disagrees.

**Does BookForge ever start a server it did not install?** Adoption (section 1) is the text
server's answer to "something is already on 8300". For Crucible the equivalent question is
whether the bootstrapper may attach to a Crucible it finds running rather than starting its
own. **Proposed: yes, and it is not adoption but the ordinary case** — a registry entry
whose URL answers `ping` is a server, whoever started it. The bootstrapper starts one only
when no entry answers.
