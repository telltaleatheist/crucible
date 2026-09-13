# Phase 7: a server is a machine, and a machine is a set of slots

**Owen, 2026-09-13.** *"Right now there's a queue with a gpu slot and two cpu slots. I
figured we'd have logic that lets a user connect a server to Bookforge by ip address or
something. They make the handshake, and it's added as a new gpu slot. Crucible will have
to have an api endpoint that will report what's on it and its progress so Bookforge can
hit that endpoint and fill that gpu slot with that data."*

and, refining it minutes later:

*"But this would mean every server added will have its own gpu slot. And if we can run
something on the gpu, we should be able to run it on the cpu as well. Which means we'll
have two cpu slots and a gpu slot per server."*

The refinement is the better model and this document is built on it. This is the first
phase in which Crucible changes what BookForge's **queue** is, rather than what one step
inside it does. PHASE5-APPS.md section 3.1 anticipated the consequence in one line —
"admission becomes a network call" — and this is that line worked out.

---

## 1. What the queue is today

`shared/queue/engine-types.ts`:

```ts
export type StepResource = 'gpu' | 'cpu' | 'wait';
export const RESOURCE_SLOTS: Readonly<Record<StepResource, number>> = {
  gpu: 1, cpu: 2, wait: 32,
};
```

One number per resource, fixed at compile time. A step declares `resource: 'gpu'` and the
scheduler runs it when `slotsInUse('gpu') < 1`. "The GPU" is a singular, unnamed,
implicitly-local thing — it is the 3090 Ti in this machine and nothing else could be
meant. Admission is local for the same reason: `queue-engine.ts` checks
`%APPDATA%\BookForge\external-gpu-job.lock` and the gpu-arbiter, both of which describe
*this* machine's card.

---

## 2. What it becomes: capacity is per machine

The unit is not a slot. It is a **machine**, and a machine brings its own slot set:

```ts
type Machine = {
  id: 'local' | string;            // stable, the registry's name for it
  kind: 'local' | 'crucible';
  url?: string;                    // crucible only
  slots: { gpu: 1, cpu: 2 };       // what it contributes
};
```

`RESOURCE_SLOTS` stops being a constant and becomes the sum over connected machines, and
`slotsInUse` becomes per machine. Everything else about the scheduler — parent-done,
queued, one step per slot — is unchanged.

**Why per-machine is right, and not merely Owen's preference.** A single global count of
`gpu: 3` would be a lie the moment two of those cards are on one box, and it makes
admission unanswerable: "is the GPU free" has no meaning across three machines, while "is
*this* machine's card free" is one fact about one machine that that machine can answer
about itself. It also makes the bench drawable — a row per machine showing its own
occupancy — and makes a disconnection comprehensible: a machine leaves and takes its
three slots with it, rather than a global number mysteriously dropping.

**`local` is always present and is never removed.** A disconnected server reduces
capacity; a missing local card is a different and louder problem. With zero servers
connected the queue behaves exactly as it does today — one GPU, two CPU, same scheduler —
which is the migration story.

---

## 3. The CPU half is a real change to Crucible, not just to BookForge

Owen's *"if we can run something on the gpu, we should be able to run it on the cpu as
well"* is correct as an intuition about hardware and needs work before it is true of the
software. Three facts, stated plainly:

**Crucible has exactly one queue lane, and it is exclusive.** Every job type — `llm`,
`tts`, `asr`, `align`, `rvc`, `echo` — is serialised through it. That design is right for
the reason it was chosen: **the card is exclusive**, so two jobs that both want it must
not run at once.

**That reason does not apply to work that does not touch the card.** So a second,
non-exclusive lane in Crucible for CPU work is architecturally *consistent* with the
existing design rather than a hole in it. The single lane was never a statement about
concurrency in general; it was a statement about the GPU.

**Crucible already does CPU work — inside the exclusive lane, where it does not belong.**
`crucible/jobs/tts/render.py` shells out to ffmpeg once per chunk to encode PCM16 into
FLAC, and `asr` refuses `ffmpeg_missing` before it will queue. So a book's worth of FLAC
encoding is currently occupying the card's lane while the card sits idle between chunks.
The split this phase proposes is one the server already wanted.

### "There is no CPU fallback" is not contradicted

DESIGN.md section 2 says: *"The server detects its backend at startup. If nothing is
viable it refuses to serve and says why. There is no CPU fallback."* That sentence is
about **backend detection** — a machine with no accelerator must refuse rather than run a
model slowly on its CPU. It says nothing about ancillary work performed alongside a real
accelerator.

To keep the rule from eroding, this document names the distinction and asks that it be
used:

- **accelerated work** — a model runs. Goes in the exclusive lane. Never on a CPU. The
  rule above governs it absolutely.
- **ancillary work** — encoding, muxing, concatenating, hashing, checksums. No model. Goes
  in the CPU lane. Two at a time, matching BookForge.

A job type declares which it is. `llm`, `tts`, `asr`, `align` and `rvc` are accelerated;
the FLAC encode inside `tts` is ancillary and is the first thing that should move.

### The part that is genuinely hard, and is not solved here

BookForge's CPU steps today are **assembly** — ffmpeg concat, m4b muxing, the coverage
gate, the chapter markers, the `.sentences.vtt` sidecar. Those do not operate on a chunk;
they operate on **the whole session and the library**. Sending assembly to the Mac means
sending it every FLAC in the book and receiving an m4b back: gigabytes in one movement,
against a chunk stream that is incremental and overlaps with generation.

So a remote machine's two CPU slots are **real capacity with, today, almost nothing that
can travel to them**. That is an honest statement of where this lands, not a reason to
drop the CPU half: the slots should exist and be modelled, because the model is right and
because ancillary work will grow into them (the per-chunk encode first, then per-chunk
hashing for phase 6's sha'd resume, then per-chapter alignment). What must not happen is
assembly being made to travel just because a slot is free.

---

## 4. Not every step can travel

Today's `gpu` steps are not one kind of thing:

| step | what it actually runs | can it use a remote machine? |
|---|---|---|
| narration (TTS) | narrator, spawned locally or in WSL | **only once phase 6 lands** |
| AI cleanup | Ollama on localhost, or a Crucible `llm` | yes, via the `crucible` provider |
| Foundry VLM page read | a WSL python env | not yet — section 8 |
| enhance / RVC | a local python env | via Crucible `rvc`, unwired on the client |
| align | local, or Crucible `align` | yes, once wired |
| assembly | local ffmpeg over the whole session | **no** — section 3 |

So an assignment needs an **eligibility** answer, and the honest place for it is the step
module, the only thing that knows what it is about to spawn:

```ts
/** Which machines this step can run on. Default: 'local' — a step that has not
 *  been taught to travel does not travel. */
machines?(config: Record<string, unknown>): 'local' | 'any';
```

Defaulting to `'local'` is the whole safety property. A step that has not been converted
to speak to Crucible keeps behaving exactly as it does now, and adding a server to the
registry can never silently break a render. The alternative default would hand a
locally-spawning narration step a remote machine, and it would either fail on a path that
does not exist or — far worse — run locally while occupying a remote slot, letting two
renders fight over one card.

**The scheduler's preference rule:** a step that can travel prefers an idle **remote**
machine over the local one, because the local card is the scarce thing that a `local`-only
step queued behind it will need. A `local`-only step takes the local slot or waits, even
when three remote machines are idle.

### 4.1 A slot is a STEP, not a request

Stated here because it is a correctness constraint and not a throughput knob, and because
the Foundry session supplied the numbers that prove it.

A VLM page-read step occupies **one** slot and runs **twelve** page requests in flight
inside it (`DEFAULT_VLM_CONCURRENCY`). That twelve is not tuning. `foundry/src/vlm/read.ts`
narrows each page's token cap from the longest page accepted *so far*, so a page is sent
under a band up to N answers out of date — and the 4x margin in `band.ts` and the 2x retry
factor in `models.ts` were both chosen against a lag of exactly twelve. Measured over
**18,202 pages**: at twelve the lag costs zero accepted pages; at twenty-four it costs two.

So a slot model that decided how many requests were in flight would be silently changing
what two other files are allowed to assume. Lowering is always safe; raising past twelve
requires `band.ts` and `models.ts` revisited in the same commit. **The scheduler allocates
steps. What a step does inside its slot is the step's own business.**

### 4.2 Picking the machine

Owen, 2026-09-13: *"The easiest way to do this is probably to disable or enable servers at
will from the queue page. To make sure it goes to the server we want. Do you have a
different idea of how we pick the server to use for a job or a set of jobs"*

The toggle is right and should be built — but it answers a different question from the one
it is being asked to answer, and used for routing it has a failure mode worth avoiding.

**Keep the toggle, as a capacity switch.** "This machine is available to the queue at all"
is a real, standing piece of state with an obvious use: the Mac is fine-tuning tonight,
keep the queue off it. It belongs on the queue page exactly as Owen describes.

**But routing by toggling is modal, and the queue is asynchronous.** Disabling two machines
to force one job onto the third also routes every job that admits afterwards, and Owen is
usually not watching — the pattern this whole system is built around is queueing a night's
work and going to bed. The assignment happens at **admission**, not at enqueue, so with a
global toggle the machine a job lands on is decided by whatever the toggle happened to be
hours later, in an order he did not choose. It also cannot express "these three chapters
on the Mac, those three on the PC", which is the case that motivated the question.

**So: a per-row pin, defaulting to Any.** `Run on: [ Any | This PC | Mac Studio ]`, set when
the row is created and editable while it is still queued. The decision is recorded at the
moment the intent exists — when Owen makes the row — instead of being inferred from global
state at an unpredictable later time. It costs almost nothing, because section 4 already
requires an eligibility function; a pin is one more input to it:

```
eligible(step) = step.machines() ∩ enabled(machine) ∩ (step.pin ?? any)
```

A pin naming a machine that is disabled, unreachable or stale **holds the step and says so
by name**. It never silently falls back to another machine: a pin is an instruction, and
quietly doing something else with Owen's book is the failure this rule exists to prevent.

### 4.3 A JOB IS ATOMIC (Owen's ruling, 2026-09-13)

*"I'd be fine with the premise that a single job must be atomic in that if it starts on
one server, it finishes on one server. We could split later but it would simplify the
pipeline right now."*

**Ruled, and it removes a whole class of design from this phase.** A job that started on a
machine finishes on that machine or fails. There is no migration, no partial hand-off, no
"the Mac died at chunk 300 so the PC picks up at 301" inside one job. Everything section 6
says about a machine going away stays true and gets *simpler*: the step fails by name and
the ordinary resume re-submits it, rather than the scheduler attempting a live transfer
whose half-states nobody could enumerate.

Three things this buys immediately:

- **The pace state has one owner for a job's lifetime.** Phase 6 section 4 carries the
  guard's running median chapter to chapter; atomicity means it never has to be carried
  mid-chapter, which is the case that would have needed the tracker to be serialisable at
  an arbitrary point rather than at a boundary.
- **Artifacts have one home.** Phase 6 section 6.1's fetcher reads one server per job. A
  split job would mean chunk 1-299 on one server's artifact store and 300+ on another's,
  and a resume that had to know which.
- **The failure is one word.** A job either ran on a machine or it did not.

"We could split later" is the right framing and is why this is recorded as a ruling rather
than a constraint: nothing here forecloses it. A future job that wants to split can do it
by being **two jobs**, which the chain already expresses.

#### Two things "atomic" does NOT mean, and both need saying

The word is load-bearing and it has two readings that would each break something that
works today. The Foundry session raised both; they are wording, not design, and this
contract will outlive everyone's memory of what was meant.

**It is about where a job RUNS, not who it may talk to.** A job is *claimed* by one
machine, runs there, and finishes or fails there. It is emphatically **not** "a job does
not involve another machine" — that phrasing would retroactively outlaw Foundry's
`--vlm-endpoint`, which has worked for months and is the only reason this PC's 3090 is
reachable from the Mac at all. A page read runs start to finish on one machine and calls a
GPU over HTTP the way anything calls a database. **The remote server is a service the job
consumes, not a second machine the job runs on.** Ownership is the test; conversation is
not.

**It means atomic in CLAIMING, never in EFFECT. Partial work surviving a failure is a
feature, and deleting it would be the most expensive mistake available in this codebase.**
A killed VLM read is deliberately not rolled back: `foundry/src/vlm/readings.ts` appends
and fsyncs every answer the moment it lands, so a halt at page 400 of 900 banks 399 pages
of GPU time and the re-run pays only for what is missing. That module's header records a
previous version which "safely" rotated banks aside on failure and thereby lost a finished
book. Phase 6's chunk fetcher has exactly the same property for exactly the same reason —
a render that dies at chunk 900 of 1,400 keeps its 900 chunks, which is what BookForge's
resume has always relied on.

So: **no failure cleanup, no rollback-to-clean, no tidying a partial output on the strength
of the word "atomic".** The hazard here is not a bug; it is a well-intentioned future edit
that looks principled while destroying hours of GPU per book. It is the same shape as
section 4.1's twelve: a reasonable-sounding improvement to the slot model silently breaking
a correctness or cost property that lives in a different repo.

### 4.4 A chain STICKS to the machine its first step ran on

This is the part that is not a preference, and it is the reason a per-row pin alone is not
enough.

The two backends are different implementations of the same model. `cuda-linux` renders
Higgs through SGLang (`engine/higgs/v3_served.py`); `mlx-darwin` renders it through
mlx-audio (`engine/higgs/mlx_backend.py`). Different samplers, different RNG, different
batching. **They do not produce the same audio**, and they need not, because until now a
book was rendered on one machine by construction.

Spreading one book across machines breaks that in two ways at once:

1. **The book acquires a seam.** Chapters 1-4 in one engine's voice and 5-8 in another's is
   audible in a way no test asserts.
2. **The guard's pace state is carried across a rate change.** Phase 6 section 4 has the
   running median travelling chapter to chapter; if the two engines speak at even slightly
   different rates, chapter 5 starts centred on chapter 4's machine and fires on healthy
   chunks — which is exactly the failure raising the short factor to 1.3 was meant to stop.

So **machine affinity is inherited down a chain**: the first step to be assigned fixes the
machine, and every step chained behind it inherits the pin unless Owen overrides it
explicitly. A book is a chain. That gets the parallelism where it is safe — two *different*
books on two machines at once — and refuses it where it is not.

---

## 5. `GET /v1/activity` — what is on this server and how far along

The endpoint Owen asked for. One cheap read, the whole server, no job id needed.

```json
{
  "server":   {"name": "owens-mac-studio", "version": "0.5.0", "backend": "mlx-darwin",
               "api_version": 1, "uptime_s": 48213},
  "resident": {"kind": "tts", "id": "sigma", "since": "2026-09-13T18:02:11Z",
               "vram_bytes": 9126805504, "estimate_basis": "measured"},
  "slots":    {"accelerated": {"busy": 1, "of": 1, "queue_depth": 2},
               "ancillary":   {"busy": 0, "of": 2}},
  "running":  [{"job_id": "j_01H…", "type": "tts", "model": "sigma", "lane": "accelerated",
                "progress": 0.421, "message": "rendering 118 of 280 chunk(s)",
                "created": "2026-09-13T18:04:02Z", "started": "2026-09-13T18:04:03Z",
                "client": "bookforge/owens-pc"}],
  "queued":   [{"job_id": "j_01H…", "type": "tts", "model": "sigma", "position": 1,
                "created": "…", "client": "bookforge/owens-mac-studio"}],
  "accelerator": {"total_bytes": …, "used_bytes": …, "free_bytes": …,
                  "unattributed_bytes": …}
}
```

Nearly all of it is assembly of state `JobStore` already holds — `running_id`,
`queue_depth`, `position`, `job.progress`, `Residency`, and the `/v1/accelerator` probe.
`slots.ancillary` is the only field that does not exist yet, and it appears with the
second lane.

### Why a poll and not the SSE stream it already has

Crucible already streams per-job events, and a step that owns a job reads that stream —
fine-grained, push, exactly right for a progress bar on *that* row. `/v1/activity` is a
different question asked by a different part of the UI: *what is this machine doing?*,
asked by a bench widget that has no job and may never have one. Making the widget open an
SSE stream per job per server, to render one line of text, is the wrong shape. The two do
not compete: the step uses the stream, the bench uses the poll.

### `client` is load-bearing, and is new state

Two BookForge instances — the PC and the Mac — can point at the same server, and they
will. A bench that shows a foreign render as its own is actively misleading: Owen would
see progress on the PC for a chapter the Mac is rendering, and cancelling it would be a
surprise.

So a job records **who created it**. The SDK already sends
`<clientName> crucible-client/<version>` as the User-Agent on every call, so the value is
already on the wire and is simply not kept. `JobStore.create` records it, `/v1/activity`
reports it, and BookForge draws a foreign job as **busy — <client>**, with no progress bar
it does not own and no cancel button.

This is not a security boundary. Everything holding one token is one trust domain
(DESIGN.md section 8), and a client that lies about its name is lying to a bench widget.
It is an identification, and it is described as one.

### What it must not become

`/v1/activity` reports. It does not admit, reserve, claim or lock. A client reading "slot
free" and submitting is racing every other client, and **that race is already correctly
handled** by the server's own lane: the second job queues. The whole point of Crucible
owning a queue is that admission is not the client's problem, and an endpoint that let a
client reserve a slot would put it back.

---

## 6. Admission across the seam

Local admission is unchanged: the lock file and the gpu-arbiter, because they describe
this machine.

**Remote admission is the absence of a reason not to, and it is cheap:** the machine's
last `/v1/activity` poll succeeded, its `api_version` matches, and the job type this step
needs is enabled there (`/v1/info`'s `job_types`). It is explicitly **not** "the slot is
idle" — the server queues, so submitting into a busy server is legal and often right.

The failure that needs naming is **the machine that goes away mid-step**. A remote render
is minutes to hours, and a closed laptop lid or a tailnet hiccup must not fail a chapter
that is still rendering on the far side. So:

- a poll failure marks the machine **stale**, not gone; the bench greys its row
- the step reading the job's SSE stream reconnects, and Crucible's event log is replayable
  from an offset, which is what makes that safe
- a machine stale longer than a stated grace window loses its slots, and the step on it
  fails by name — `crucible_server_unreachable`, naming the machine — rather than hanging

The grace window is the number to argue about, and it is a config value rather than a
constant in the scheduler.

---

## 7. Connecting one: the handshake, and the gap in it

`electron/crucible/servers.ts` and `<userData>/crucible-servers.json` already exist, and
`ping` is already the unauthenticated probe that distinguishes "not a Crucible" from
"wrong token" (`CrucibleNotACrucible` vs `CrucibleAuthError`). The mechanism is built;
what is missing is the door.

The flow: Owen types an address → `ping` (no token) confirms it is a Crucible and reports
its api version → Owen supplies the token → `info` confirms it and names the backend, the
job types and what is resident → the row is written to the registry → a machine appears
with its three slots.

**The gap, stated plainly: there is no token exchange.** Every call but `ping` needs a
bearer token, and Crucible has no pairing flow — the token is minted by `crucible init` on
the server and read with `crucible token --show`. So today "connecting a server" means
Owen SSHes to the machine and pastes a token. That is acceptable for three machines he
owns, and it is the reason this is a **settings flow and not a discovery flow**;
pretending otherwise would produce a UI that cannot work. A pairing handshake (a
short-lived code printed by `crucible pair`) is a reasonable phase 8 and is not proposed
here.

**This is also the first BookForge UI Crucible has ever justified.** Every phase so far
held the line at "no UI, no IPC, no settings row", and that line was right — the CLI
proved each seam without moving the app. But a server registry Owen edits by hand in a
JSON file under `<userData>` is not a feature, so this is where the line is deliberately
crossed, for a reason, and the reason is written down here.

---

## 8. Foundry

Foundry enqueues into this queue (`electron/foundry-host-queue.ts`) and its VLM page
reading is a `gpu` step that spawns a WSL python env. Under section 4's default it is
`local` and keeps working untouched, which is the correct first state.

Making it travel is a Foundry-side change — page reading is a Crucible `llm` job with
image content parts (PHASE3-VLM.md), which the server has built and verified on a real
card. It is not on this document's critical path and should not be started until
`/v1/activity` and the machine model exist, or it will be built against a contract that is
still moving. The Foundry session has been told exactly that.

---

## 9. Order, and what depends on what

1. **`/v1/activity` + the recorded client** — server-side, small, independently useful,
   and nothing else can be built without it. **Start here.** *(Built 2026-09-13, branch
   `feat/phase6-remote-render`: the route, the opt-in probe, `Job.client` from the
   User-Agent, `Job.message` from the last progress event.)*
2. **The ancillary lane in Crucible** — a second, non-exclusive lane; the `tts` FLAC
   encode moves into it. Independently valuable (it stops a book's encoding from occupying
   the card's lane) and it is what makes `slots.ancillary` honest.
3. **The machine model in the BookForge scheduler** — per-machine capacity, `machines()`
   defaulting to `local`, a bench row per machine. Testable with the keeper suite's fake
   step modules and a fake server; no card needed.
4. **The settings flow** — address, token, `ping`/`info`, the registry row.
5. **Teaching the first step to travel** — narration, and only after phase 6, because
   until the chunks can come back over the wire there is nothing for a remote machine to
   do.

Phase 6 and steps 1-3 here are independent and can proceed in either order. Step 5 is the
join, and it is the one that needs a real card and Owen's go.
