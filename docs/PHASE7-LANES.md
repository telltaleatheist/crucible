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

## 2. SUPERSEDED — capacity is not per machine; a remote step waits

> **This section is superseded, 2026-09-13, later the same day.** It is kept because the
> refuted idea is the one somebody will think of next, and because the reason it is wrong
> is the most useful thing in this document. The correction is section 2.5. Read that
> first; what follows is the draft it replaces.

## 2.5. The correction: BookForge does not model the server's capacity

Owen, 2026-09-13: *"maybe we should have a system that queues jobs from foundry,
bookforge, or wherever else… that way we dont have to build a queue for crucible as well
and we can rely on the client to not be an idiot and do two jobs at once on the same
server."*

**The premise is already satisfied: Crucible has had a queue since v0.3.0.**
`crucible/jobs/queue.py` is one exclusive lane draining in submission order, reporting
`position` (0 running, 1..n in line) and `queue_depth`. So there is no second queue to
build, and there is no onus on anyone not to submit twice — two clients submit, the second
waits. Since Foundry is hosted inside BookForge, two clients against one server is the
NORMAL case, and a design that asked the user to avoid it would have been a design against
Owen's own setup.

But his instinct — **there must not be two arbitrators** — was right, and it lands on this
document rather than on the server.

> **SUPERSEDED IN ONE DETAIL, 2026-09-13 (ARCHITECTURE.md section 3).** Owen went
> further: *"i think all queuing logic should exist in the clients, not the server. if the
> server is busy, it cant receive a new job."* So the second client **no longer waits —
> it is refused `409 server_busy`, told who has the card and how far along they are.**
>
> The paragraph above is otherwise unchanged and its conclusion is strengthened, not
> weakened: there is still no second queue to build, because the queue that exists is the
> client's, which is the only one that knows the chain, the pin and the priority. What
> changes for this document is one sentence in section 6 and one in section 5 — both
> flagged there — and nothing at all in 2.5's rule below, which is about *routing*.
>
> `crucible/jobs/queue.py` keeps the lane, the deque, `position`, `queue_depth`, cancel,
> events and provenance. `queue_depth` is honestly 0 or 1.

### What section 2 got wrong

It had each connected machine contributing `1 GPU + 2 CPU` to **BookForge's** queue, with
`/v1/activity` polled to decide whether a remote slot was free. That is BookForge keeping a
**stale replica of Crucible's authority and making admission decisions from it** — two
schedulers, one deciding from a cache of the other, with a poll interval's worth of wrongness
built in. Every hard question it created (how stale is too stale, who wins a race, what
happens when the poll and the submit disagree) was self-inflicted.

It also contradicts a ruling Owen had already made. `shared/queue/engine-types.ts`:

> `gpu` is the exclusive **local** resource […] `wait` is not a worker and does not belong
> on the bench. A step declares it when its whole job is to sit until something outside
> this queue happens.

Owen created `wait` on 2026-09-08, watching an export step hold a CPU slot: *"its sitting
in the cpu slot doing nothing for two minutes now."* Holding a worker while waiting starves
a real render behind it.

### The rule

**A step running on a remote machine holds `wait`, not `gpu`.** While the Mac renders, this
PC's card is free. That is not a modelling preference; it is a fact, and the existing
resource vocabulary already expresses it.

Consequences, all of them simplifications:

- **No per-machine slot arithmetic and no remote admission.** `RESOURCE_SLOTS` stays the
  compile-time constant it is. The local queue keeps describing the local machine, which is
  the only machine it can speak for.
- **The protocol is submit-and-be-answered.** Crucible answers the only question that
  matters — *is there room now* — authoritatively, because it is the one holding the card.
  Since the admission ruling that answer is a 202 or a `409 server_busy` naming the
  holder, rather than a 202 and a `position`. The client keeps its own queue either way;
  the difference is that it no longer has a copy of its row inside the server too.
- **Distribution stays emergent and gets better.** Four books pinned to four servers all run
  at once because none of them occupies a local worker — and the local card goes on
  rendering a fifth.
- **Contention surfaces honestly instead of being prevented.** A step that cannot get in
  is refused `server_busy`, and the `client` recorded on each job (section 5) is on the
  refusal, so the message names who is ahead: *"waiting behind owens-mac-studio's job"*.
  (Before the admission ruling this read `position: 3`; the sentence it supports is the
  same one.)
- **`/v1/activity` changes role, not shape.** It is a **bench display** and a **preflight**
  (reachable, API version matches, job type enabled). It is never admission. Display and
  admission are different questions, and only one of them may be answered from a poll.

### What survives from the superseded draft

The pin (4.2), chain stickiness (4.3), atomicity (4.3) and the guarantees (9) all stand
unchanged. Every one of them is about **routing** — which machine a job is for — and
routing is the single scheduling decision anybody makes here. It is the user's, it is made
once, and it is made at the row. Nothing above turns it into a scheduler.

Crucible's ancillary lane (section 3) also stands, but is now plainly independent of this
phase: it is worth doing because a book's FLAC encoding should not occupy the card's lane,
not because BookForge needs to count it.

---

## 2.9. The superseded draft: capacity is per machine

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

> Since the admission ruling (ARCHITECTURE.md section 3) the lane is also what *admission*
> is about: a submission arriving while the lane is occupied is refused, not queued. That
> is the reason `echo` is refused too even though it needs no card — exempting the types
> that want no accelerator would put them straight back on a deque, because the lane is
> exclusive whatever a job wants from it, and the server would be queueing again for
> exactly the jobs it claimed not to queue for.
>
> **When the ancillary lane below is built it gets its own admission answer**, because it
> has its own occupancy — two at a time, not one. `server_busy` will then have to name
> *which* lane is full, and a job type's declaration of which lane it belongs to becomes
> load-bearing at the door rather than only at the scheduler. Nothing about that is
> decided here; it is recorded so the second lane is not built as if admission were still
> a single global fact.

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

**So: a per-row pin — and `Any` is NOT the default, because the machines are not equal.**

Owen, 2026-09-13: *"the mac is significantly slower than the pc. 90% of the time im going
to want to use the pc, hands down. but if im doing lots of jobs at the same time, im going
to want the ability to overflow the queued items to the mac. so its not a situation where
all gpus are equal and will be picked equally."*

An earlier draft of this section had the pin default to `Any`, meaning "the first eligible
machine". **That was wrong, and the name was the tell.** "Any" says the machines are
interchangeable. They are not: `cuda-linux` on the 3090 Ti and `mlx-darwin` on the M1 Ultra
differ by enough that sending a job to the slower one when the faster one is merely BUSY
can finish later than waiting would have. A scheduler that treats them as equal is not
being neutral — it is being wrong about the hardware, silently, in the direction that costs
wall-clock.

**The correction is two independent facts, because Owen's sentence contains two.**

**1. Machines are RANKED, once, in the registry** — not per row. `rank` is a property of a
server entry (this PC 1, the Mac 2), set in settings where a person thinks about hardware
rather than while queueing a book. The default pin for any new row is **the highest-ranked
eligible machine**, which is the 90% case with nothing to click.

**2. Overflow is OPT-IN, per row, and defaults to OFF.** `May overflow if busy`. With it
off — the default — a row waits for its machine, which is what "90% of the time I want the
PC, hands down" means: the PC being busy is a reason to wait, not a reason to go somewhere
slower. With it on, a refusal sends the row to the next machine down the rank.

That is the whole of it, and it maps to his sentence exactly: *the ability* to overflow is
a thing you turn on when you are queueing a night's work, and off the rest of the time.

**What a 409 means therefore depends on one flag, and both behaviours fall out of it:**

| the row says | a `server_busy` means |
|---|---|
| overflow **off** (default) | **wait.** This machine or none. |
| overflow **on** | try the next machine down the rank; wait when the list runs out |
| pinned to a NAMED machine | **wait**, always — a pin is an instruction, and overflow does not override it |

**Distribution is still emergent and still not a scheduler.** Nothing measures throughput,
predicts a finish time, or balances anything. It is a ranked list, a flag, and a refusal
that arrives fast enough to act on.

**What is NOT proposed, and why.** The genuinely optimal answer to "should this job wait for
the fast machine or run now on the slow one" needs how long the current job has left and how
much slower the other machine is for THIS job type. Crucible reports progress but nothing
estimates a finish, and the MLX/CUDA ratio is per workload rather than a constant. A
scheduler that guessed would be wrong in a way nobody could see. A flag the operator sets is
right in a way they can. The decision is recorded at the
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
               "memory_bytes_estimate": 9126805504},
  "slots":    {"accelerated": {"busy": 1, "of": 1, "queue_depth": 1},
               "ancillary":   {"busy": 0, "of": 2}},
  "running":  [{"job_id": "j_01H…", "type": "tts", "model": "sigma",
                "status": "running", "position": 0,
                "progress": 0.421, "message": "rendering 118 of 280 chunk(s)",
                "created": "2026-09-13T18:04:02Z", "started": "2026-09-13T18:04:03Z",
                "client": "bookforge/owens-pc"}],
  "queued":   [],
  "accelerator": {"total_bytes": …, "used_bytes": …, "free_bytes": …,
                  "unattributed_bytes": …}
}
```

`queued` is **all but always empty** since the admission ruling, and `queue_depth` is 0 or
1: a second submission is refused rather than appended, so the only thing that can be in
that array is a job admitted microseconds ago that the lane has not picked up yet. The key
and the `position` field are kept — they are still correct, every phase-2 client reads
them, and the window they describe is real. This example used to show a second client's job
at `position: 1` with `queue_depth: 2`, which is now a shape no client will see.

**THE BLOCK ABOVE IS THE ROUTE'S, CHECKED AGAINST IT.** Two fields in an earlier draft
were invented and are corrected here: `resident` carries `memory_bytes_estimate` (not
`vram_bytes` plus `estimate_basis` — `estimate_basis` is a MANIFEST field describing where
a number came from, and a resident thing has no such provenance), and a `running` row
carries `status` and `position` rather than a `lane` key the route has never emitted. Both
were spotted by the agent that built section 3.1, which declined to guess which side was
meant to be right. **The code is: it is what ships, and a document that disagrees with it
is a trap for whoever writes the next consumer** — the same mistake PHASE6 section 3 made
with the guard verdict, found the same day.

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
free" and submitting is racing every other client, and **that race is settled at the
door**: `POST /v1/jobs` admits one and refuses the other `409 server_busy`, naming the
winner. An endpoint that let a client reserve a slot would be a second place to arbitrate,
and a stale one.

So this route is a **bench display** and a **preflight**, never admission. It is the
honest answer to *"how long until that finishes"*; it is not permission to submit, and a
client must be built to be refused after reading it. Only `POST /v1/jobs` can say yes.

*(Before the admission ruling of 2026-09-13 this paragraph said the race was handled
because "the second job queues". It no longer queues — it is refused with the facts. The
rule the paragraph exists for, that display and admission are different questions and only
one may be answered from a poll, is unchanged and is now literally enforced.)*

---

## 6. Admission across the seam

Local admission is unchanged: the lock file and the gpu-arbiter, because they describe
this machine.

**Remote admission is the absence of a reason not to, and it is cheap:** the machine's
last `/v1/activity` poll succeeded, its `api_version` matches, and the job type this step
needs is enabled there (`/v1/info`'s `job_types`). It is explicitly **not** "the slot is
idle", because a poll cannot know that — the authoritative answer comes back from the
submission itself.

**Since the admission ruling (ARCHITECTURE.md section 3) a busy server refuses**, so
submitting into one is legal but no longer *lands*. The client's own queue holds the row
and retries; it has lost a round trip and nothing else, because it never handed ownership
of that row to the server. What it must not do is treat `409 server_busy` as a failure —
it is the one answer that is never the step's fault, it names the holder and it carries
`progress`, which is enough to back off for about as long as the holder has left.

**The retry belongs in the SDK**, written once, so BookForge and Foundry inherit it and
cannot drift into two different back-off policies against one server.

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

### 7.1 The token, ruled 2026-09-13

Owen ruled on the whole shape at once rather than piecemeal. Three questions had been
getting mixed; they separate cleanly.

**(A) WHERE IT LIVES.** Already settled by DESIGN.md section 9 — *"`crucible init` mints
it; the client stores it per server entry."* So: in the registry that owns the server
list, which by section 4.2's rule is BookForge's `crucible-servers.json` when Foundry is
hosted and Foundry's own settings when it is not.

**(B) HOW IT REACHES THE PROCESS THAT CALLS.** Not as a token — **as an opaque header
map**, in one environment variable set by whoever spawns the engine:

```
{"Authorization": "Bearer <token>", "X-Crucible-Api": "1"}
```

Three reasons this beats passing a token:

- **It keeps the client ignorant of what a Crucible is.** The Foundry session asked not to
  teach its engine the product's name, and it was right: an engine that knows only "this
  endpoint wants these headers" needs no change the day it points at something else.
- **It carries both headers through one mechanism.** The version header is not a secret and
  the token is, but they travel together and neither needs its own path.
- **It is future-shaped.** Any server wanting a different header set is already supported.

The rest of the discipline, unchanged: **never a flag** (a command line gets pasted into a
bug report), **never through a settings echo** that prints values by design, **never
logged**, and **stripped from the environment of any child that does not need it** — a
rasteriser turning pages into PNGs has no use for a credential, and children inherit
everything unless an explicit `env` says otherwise.

**Why environment rather than a settings field alone.** Storage is on disk either way, so
the disk exposure is common to both and cannot break the tie. What differs is the residual
risk: the environment's is inheritance, fixable once with an explicit `env` at each spawn;
the settings field's is that the secret sits in a code path that prints values by design,
where safety depends on every future author remembering to exclude it. A mechanical fix
beats a remembered one.

**(C) HOW THE USER GETS ONE — the question nobody had answered.** Today it is: open a
terminal, `wsl`, `crucible token --show`, select text in a terminal, paste. For the case
Owen actually names — a friend installing Crucible in WSL on their own machine for speed —
that is the product's first impression, and no field design fixes it.

**So invert who mints it. When the CLIENT installs the server, the client mints the
token** and passes it to `crucible init --token <generated>`. The user never sees a token,
never copies one, never learns the word. This is cheap precisely because the client is
already the thing running the install; it belongs in the **bootstrapper**, which is still
unbuilt, so it is a requirement on that work rather than work outstanding here.

Split by provenance:

| How the server got there | Token UX |
|---|---|
| Installed by BookForge (local WSL — the common case) | **none.** The client mints it. |
| Installed by hand, or remote (the Mac, a droplet) | paste into a field beside the URL |
| A pairing code (`crucible pair`) | **deferred** |

Pairing is deferred deliberately: it needs a new **unauthenticated** endpoint, which is a
security surface added to solve a problem that exists only for machines nobody installed
from here — and Owen administers all of his.

### 7.2 The registry file is plaintext, deliberately

Written down so that nobody earnestly encrypts it later without knowing what they are
trading away.

The token sits in plaintext in the registry under `<userData>`. That matches the actual
threat model — a home tailnet, machines Owen owns, and everything holding one token being
one trust domain by DESIGN.md section 8 — and it is consistent with how every other
credential here is already kept, including the HuggingFace token, which is a text file in
`Downloads`.

An OS keychain would be inconsistent with everything around it and would buy
cross-platform work against no threat that exists here. If the threat model changes — a
shared machine, a token that reaches something Owen does not own — this is the paragraph
to come back and argue with.

---

## 8. Foundry

### 8.0 The contract, BUILT 2026-09-13 (Foundry `2d5d411`)

**What BookForge must set on the Foundry engine process it spawns:**

```
FOUNDRY_ENDPOINT_HEADERS={"Authorization":"Bearer <token>","X-Crucible-Api":"1"}
```

A JSON **object** of header name to string value. Every request on both of Foundry's
doors — the VLM page read and the four text acts — carries every pair. **Absent or empty
means send none**, which is today's behaviour exactly, so a server wanting no headers is
unaffected and nothing changes until BookForge starts setting it.

Note what is NOT in that name: Crucible. The header map was chosen so Foundry's engine
learns only *"this endpoint wants these headers"* — the word for this product appears
nowhere in its `src/`, and the day it points at something else nothing there changes.

Verified by them through the compiled `dist/foundry-windows-x64.exe`, not at source: both
headers observed on the wire from the binary BookForge actually spawns, and a malformed
map refused before the network with a sentence naming the variable and never quoting the
value.

What they built alongside it, all of it agreed here first: malformed refuses the run rather
than dropping headers; `content-type`, `content-length` and `host` refused by name; a
settings fallback that the environment overrides and that is never echoed into a run log;
an explicit `env` at all three Python spawns with the map **stripped, not allowlisted**;
and a distinct sentence for 426, 401, 403, 404 and `model_not_resident` — the last one
saying that the server will not load a model to answer a request and that nothing on their
side can, which is section 9.2's cliff stated where a person meets it.

The port-8000 collision is fixed by **refusal rather than by moving a default**, so the
clamp set mirrored on this side needs no change.

### 8.1 What is still unbuilt



Foundry enqueues into this queue (`electron/foundry-host-queue.ts`) and its VLM page
reading is a `gpu` step that spawns a WSL python env. Under section 4's default it is
`local` and keeps working untouched, which is the correct first state.

Making it travel is a Foundry-side change — page reading is a Crucible `llm` job with
image content parts (PHASE3-VLM.md), which the server has built and verified on a real
card. It is not on this document's critical path and should not be started until
`/v1/activity` and the machine model exist, or it will be built against a contract that is
still moving. The Foundry session has been told exactly that.

---

## 9. Two guarantees a multi-server client can rely on

Both promised to the Foundry session on 2026-09-13 and written here because a promise in
a message is not a contract.

### 9.1 The Crucible id is stable across machines. `engine_model_name` is not.

**THE GUARANTEE, stated as the field a client actually reads:**

> The `id` in `GET /v1/openai/models` is the **Crucible id** — a constant in this repo,
> identical on every machine serving that manifest. `engine_model_name` is reported
> **beside** it as a diagnostic and **never appears in `id`**.
>
> So: **persist the listing's `id`. Never persist `engine_model_name`.**

Verified rather than asserted: `crucible/api.py`'s `openai_models` sets `"id":
resident.model_id` and `"engine_model_name": resident.engine_model_name` as separate keys,
and nothing merges them.

**This deliberately permits discovery.** An earlier draft of this section said "key on the
id you asked for, never the name discovery hands back", and the Foundry session was right
to reject it: read literally it outlaws asking a server what it serves, which is a
deliberate feature — `requireServedModel` with no `--model` returns the single served row
and persists its `id`, which is how a friend avoids typing a model name at all and how
Owen's measured 9B run proved the discovery path end to end. The rule above forbids the
thing that actually breaks and permits the thing that works, and it holds whether or not
the client knows what it asked for.

The reason the second half is needed is that the two backends cannot be made to agree:

- `crucible/engines/vllm.py` always launches with `--served-model-name <crucible id>`, so
  on `cuda-linux` the engine answers to the Crucible id.
- `crucible/engines/mlx_lm.py` **has no `--served-model-name`**. mlx-lm lists every
  mlx-looking repo it can see, and `engines.engine_model_name()` therefore returns the
  resolved **weights directory path** — machine-specific, and unfixable by any launcher
  rule.

So two Crucibles serving byte-identical weights at identical precision report different
`engine_model_name`s forever. What is stable is the id, because it comes from a manifest
in this repo. `GET /v1/openai/models` reports `engine_model_name` *beside* the id
precisely so the difference is visible rather than surprising.

**Why this matters more with four servers than with two.** Foundry hashes the served model
name into `cleanKey`, the translate bank, the analyze verdict key and the EPUB narration
stamp, and ruled in September that those keys stay verbatim and are never canonicalised —
correctly, but ruled when the cost of a mismatch was one re-clean. Across four registered
servers a name that varies per machine is a standing tax: a book cleaned on the Mac and
finished on a droplet re-asks every block and stamps different provenance, for no reason
but two engines spelling the same weights differently. Keying on the id removes the tax
without touching that ruling.

### 9.2 A long read loads its own model; chat still does not

`model_not_resident` on the chat door is deliberate (PHASE2-LLM.md section 5): chat is
fine-grained and unattended, so two clients alternating would thrash the card. That rule
stands and is not softening.

But it creates a cliff for anyone whose first act is a conversion: install Crucible for
speed, point a reader at it, and get refused on page one because nobody made the vision
model resident. **A page read is a render, not a chat** — long, attended, one operator's
explicit order — and `tts` render already has exactly this asymmetry for exactly this
reason: it may load its own model and emits `warming` while it does.

So the answer is the scheduler's, not the reader's: **whoever owns the queue submits
`load-model` before it queues the reads.**

**RESIDENCY IS IMPLIED BY NOTHING.** Not by the endpoint being local, not by the app
having started the server, not by the port answering. It is asked for explicitly, on every
path, before the first page.

That is stronger than the rule this section carried an hour ago, and the Foundry session
widened it twice — the second time against their own earlier correction, which is why it is
worth recording how it moved:

1. First draft: *the scheduler submits `load-model` before it queues the reads.* True, and
   silent about where.
2. Their first correction: *the gate inverts.* Their call sits in the right place
   (`app/electron/job-queue.ts:3638`) but is gated on `isLocalVllmEndpoint` — loopback AND
   port 8000, a question about the ADDRESS — while residency is a question about the
   server's KIND. So the branch that must call `ensureResident` is the one that gate
   EXCLUDES. True, and incomplete.
3. **The rule.** It must run on the branch that gate INCLUDES as well, because of an
   assumption nobody had stated: when the app starts a server itself it passes the model on
   the command line (`deriveLaunch` emits `vllm serve <model>`), so *"I started it"* has
   always implied *"the model is on the card"* and nothing ever had to ask.

**An adopted server voids that implication**, and adoption is about to become the ordinary
case rather than the edge. `ensureServer` already probes before it spawns and, finding a
port that answers, says *"Using it as it is; this app will not stop it"* — which is what
makes the transition to a service safe in whatever order it happens, and it is also what
means **no path has asked whether the model this run needs is resident.**

So `ensureResident` is not a renamed `ensureServer`. **`ensureServer` was a question about
a PROCESS; `ensureResident` is a question about a CARD**, and the only thing that ever let
one stand in for the other was a command-line argument.

### 9.3 What an idle service holds: nothing on the card

Owen ruled on 2026-09-13 that a local Crucible is a service (PHASE5-APPS.md section 6.0),
which raises a question the Foundry session asked before it could bite: if `crucible serve`
is always up, has the card been spoken for?

**No. An idle Crucible holds zero VRAM, and this is verified rather than asserted:**

- `Residency.__init__` sets `self._resident = None`. Nothing is resident at boot.
- The app's `lifespan` starts the job lane and nothing else — no engine, no model, no probe.
- An engine subprocess exists **only** while something is resident. It is spawned by
  `load` / `load_voice` / `load_aligner` and torn down by `unload` / `_evict` /
  `shutdown`, each of which calls `engine.stop()`.

So a running-but-idle service is a Python process holding tens of megabytes of RAM and
**nothing at all on the accelerator**. "The service is running" must never come to mean
"the card is spoken for", and it does not.

This matters beyond tidiness: the reason Foundry's own manager reserved a flat half the
card (`GPU_UTIL = 0.5`) was that it could not ask anything how much was free. A service that
idles at zero and answers `GET /v1/accelerator` is what makes that reservation unnecessary
rather than merely inherited. One load, the whole book, one unload. The
machinery exists (`crucible/jobs/llm/` provides `load-model` and `unload-model`); nothing
new is needed but the ordering. A client must still render the refusal well, because it is
what somebody gets when they point at a server nobody loaded — but a good sentence is not
an onboarding, and the cliff belongs to the install story rather than to the transport.

---

## 10. Order, and what depends on what

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
