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

## 2.4. RESTORED, with one correction: a slot set PER SERVER

**Owen, 2026-09-13:** *"if it's set to any, bookforge will send the queue items to any open
gpu. which means bookforge will need one set of slots for each server. one gpu and two cpu
slots for mac, one gpu and two cpu for pc, etc."*

**He is right, and section 2.5 below — which superseded exactly this shape — broke something
I did not notice at the time.** With a single global `gpu` slot, BookForge can only ever
have one GPU step running, so **two books can never render on two machines at once**, which
is the entire purpose of registering a second server. A `wait`-holding remote step avoids
starving local work, but it does not give the queue a way to *decide how many books to have
in flight*, and without that the answer is either one (useless) or all fifteen (fifteen rows
retrying against a server that wants one).

So the slot sets come back:

```
this-pc      [ gpu ] [ cpu ] [ cpu ]
mac-studio   [ gpu ] [ cpu ] [ cpu ]
droplet      [ gpu ] [ cpu ] [ cpu ]
local        [ cpu ] [ cpu ]            ← work BookForge does ITSELF (assembly, muxing)
```

### The correction, and it is the whole of what 2.5 was right about

> **A server's slots count what BOOKFORGE has in flight there. They are not a model of the
> server's capacity, and they are never read to decide whether the server is free.**

That distinction is what makes this safe, and it is worth being exact about because the two
readings look identical in a UI:

- **"The PC's GPU slot is occupied"** means *I have a GPU step running on the PC.* That is
  BookForge's own bookkeeping about its own work. It is local knowledge, it is never stale,
  and no poll produces it.
- It does **not** mean the PC's card is free when the slot is empty. Foundry may have it.
  Another BookForge on the Mac may have it. **The 409 remains the only authority**, exactly
  as section 2.5 insists.

So a free slot licenses an *attempt*, never an assumption. BookForge submits; the server
either takes the job or refuses by name; and a refusal leaves the slot free because nothing
of BookForge's is running there. **Nothing is decided from a cache, because nothing is
cached** — the count is of BookForge's own outstanding work, which it cannot be wrong
about.

### What each part is for

| slot set | bounds | why that number |
|---|---|---|
| a server's `gpu` | 1 | Crucible admits one job at a time (section 3.1). More would be rows guaranteed a 409. |
| a server's `cpu` | 2 | anticipates Crucible's ancillary lane (section 3). **Zero today** — no CPU work is sent to a server yet, and the slots exist so the bench and the model do not change when it is. |
| `local` `cpu` | 2 | assembly, muxing, the coverage gate. This is work BookForge does ITSELF and never sends anywhere — `RESOURCE_SLOTS.cpu` as it is today, unchanged. |

**There is no `local gpu` row**, and its absence is the point of phase 7: a GPU step goes to
a *server*, and "this PC" is a server like any other. The local Crucible is a service
(PHASE5-APPS.md section 6.0), so work for the 3090 Ti occupies `this-pc`'s slot rather than
a special local one.

### How `any` uses them

A book set to `any` takes the first server whose GPU slot is free **in rank order**, and
submits. If that server refuses, the slot it never occupied stays free and the book tries
the next. With two servers and fifteen books, two are in flight and thirteen are queued
locally — which is the behaviour Owen described and the reason the slots have to be
per-server rather than global.

---

## 2.5. What section 2.5 got right, and still does

*(This section superseded the per-machine slot model above. Section 2.4 restores the shape
and keeps this section's principle, which was never about the shape.)*



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

### What section 2 got wrong — and it was NOT the slot sets

**The shape was fine; section 2.4 restores it.** What was wrong was one word in the middle
of it: the slots were described as **BookForge modelling the SERVER's capacity**, with
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

**A step running on a remote machine does not hold the LOCAL card's slot.** While the Mac
renders, this PC's card is free. That is not a modelling preference; it is a fact.

Section 2.4 expresses it by giving each server its own slot set, so a step on the Mac
occupies `mac-studio`'s GPU slot and leaves `this-pc`'s alone. An earlier version of this
section expressed it by having a remote step hold `wait` under one global `RESOURCE_SLOTS`
— which kept the local card free but left the queue no way to decide how many books to have
in flight, so two books could never render on two machines at once. **2.4's shape is the
right one; this section's PRINCIPLE is what survives, and it is the bullet below.**

Consequences:

- **Nothing is read to decide whether a server is free.** A slot set counts BookForge's own
  outstanding work there (2.4), never the server's state. The local queue speaks only for
  what it has itself submitted, which is the only thing it can speak for without a poll.
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

`waitFor` (4.2), one-book-one-GPU (4.4), atomicity (4.3) and the guarantees (9) all stand
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

**So: two levels, and each is ONE field.**

Owen, 2026-09-13, after three drafts of this section: *"i think the solution is giving the
user the ability to choose which GPUs the queue uses, or which GPU the queue ITEMS use.
which registered crucible server should this job wait for? any can be an option."*

**The middle sentence is the design.** *Which registered server should this job wait for?*
is a single question, and a single question takes a single answer. Every earlier draft of
this section answered it with a machine **plus a modifier** — first `Any` with a hidden
preference, then `machine: string | null` with an `overflow: boolean` — and both were the
same answer said in two fields.

### 4.2.1 Per item: which server should this job wait for?

```ts
waitFor: string | 'any'      // a registered server's name, or 'any'
```

| the row says | behaviour |
|---|---|
| **`"this-pc"`** — the default, and Owen's 90% | run on it; if it is busy, **wait**; if it is unreachable, **hold and say so** |
| **`"mac-studio"`**, or any named server | the same, for that one |
| **`"any"`** | the first server that will take it, **preferring rank order**; unreachable servers are simply not candidates |

That is the whole per-item model. Note what it does NOT contain, because each was a real
thing in a previous draft and each is now impossible rather than merely discouraged:

- **No `overflow` boolean.** "May this travel when busy" is not a modifier on a machine; it
  is the difference between naming a machine and naming `any`. Folding it in removes the
  state `{machine: "mac", overflow: true}`, which was storable and meant nothing — the
  defect the Foundry session raised, gone by construction rather than by a UI rule and a
  normalise-on-write.
- **No `null`, and no chosen-versus-inherited distinction.** The default is a real value the
  row displays and the operator can see and change. There is no silent inheritance to
  distinguish from a choice, so the third piece of state that distinction needed — and the
  two rounds spent avoiding it — is not required.
- **No tie-break rule.** Rank is still the registry array's order (section 4.2.2), and
  `any` is the only value that consults it.

**`any` is not "the machines are equal".** It is *"I do not mind which, and I would rather
start than wait."* It still prefers rank, so a free 3090 Ti beats a free M1 every time; it
differs from naming the PC only when the PC is busy. The machines being unequal is why the
DEFAULT is the top-ranked server by name rather than `any` — which was the correction that
started this — and `any` is the opt-in for a night's work.

### 4.2.1a The default is a SETTING, because a default must not manufacture instructions

The Foundry session found the one case where the shape above and Owen's stated workflow
disagree, and it is worth the extra setting.

**The case.** Twenty books are queued; each row defaults to `"this-pc"`, visibly. Owen then
disables this-pc — to do exactly what he described, *"if im doing lots of jobs at the same
time, im going to want the ability to overflow the queued items to the mac."* By the
composition rule below, **all twenty hold**: each names a disabled server, and a named
server is an instruction. But he never touched the picker. **The default wrote an
instruction into every row on his behalf**, and the feature's own motivating case is the one
it breaks.

This is the chosen-versus-inherited problem returning — not as a missing field this time,
but as a default that fabricates choices. It is also *less* visible than the old `null`
was, in one specific way: the row honestly says `this-pc`, so it is truthful about what it
will do and silent about why, and nobody chose it.

**The fix is one setting, beside the drag-order and the enable switches. No type change, no
new per-row state:**

> **New jobs wait for:  ( • ) the top-ranked server   ( ) Any**

The 90% keeps today's behaviour with nothing to click. "Lots of jobs tonight" becomes one
switch flipped once, and every row queued afterwards can travel. The row still stores
exactly one string; `any` is simply what the default writes when Owen has said so. It also
fixes a smaller thing the per-row-only version could not: somebody whose normal mode *is*
overflow would otherwise be editing every row forever.

**And the rows already queued when the switch flips are told, not moved.** Disabling a
server that queued rows name **surfaces them** — *"12 rows are waiting for this PC, which
is now disabled"* — with a one-click bulk change to `any`. They are never silently
re-routed, because a named server is an instruction and re-routing twenty books onto slower
hardware without being asked is the failure this whole section exists to prevent. But
leaving an operator to discover it one row at a time is the other failure, and a surfaced
count with a bulk action is the answer to both.

### 4.2.2 Per queue: which servers may the queue use at all?

The second half of Owen's sentence, and a different question from the first. A server is
**enabled or disabled for the whole queue**, in settings, alongside the drag-order that
defines rank.

This is the capacity switch he described much earlier — *"the Mac is fine-tuning tonight,
keep the queue off it"* — and it is standing state about hardware, not a routing decision
about one book. Keeping it out of the per-item field is what stops it being modal: turning
the Mac off does not silently re-route work that was already queued for it.

****The Servers settings row, as Owen described it 2026-09-13:** an *Add Crucible server*
button, and each connected server a line item that **drag-and-drops to set the priority
order** — *"if 1 is free, use it."* That is the whole UI for both this section and rank:
one list, dragged to order, with an enable switch per row. No rank numbers, because the
list's order IS the rank (section 4.2's "rank needs no field").

**A newly added server lands at the BOTTOM of the order.** Adding a machine must never
silently demote the one every existing row defaults to — a person adding a droplet at
midnight is not thereby saying it should outrank their 3090 Ti. Promotion is a drag, which
is a deliberate act; demotion by side effect is not available.

The two compose, and the composition has one rule worth stating.** A row that names a
disabled server **holds and says which**, exactly as it would for one that is unreachable.
It is not re-routed, because a named server is an instruction and the queue-level switch is
about availability rather than about overriding what a person asked for. A row set to `any`
simply never considers a disabled server.

### 4.2.3 Why this is the third rewrite of one section

Recorded because the sequence is more useful than the answer.

1. **`Any` as the default**, meaning "first eligible". Wrong: it asserts the machines are
   interchangeable, and sending a job to the M1 because the 3090 Ti is merely BUSY can
   finish later than waiting would have.
2. **`machine: string | null` + `overflow: boolean`.** Fixed the default, but a named
   machine made `overflow` dead — a representable state that meant nothing — and the
   chosen-versus-inherited distinction it needed wanted a third field.
3. **One field per level**, above. Both residues vanish, because they were artifacts of
   splitting one answer across two fields rather than defects in the rules.

The lesson is the Foundry session's, generalised: when the options all feel slightly wrong,
the shared assumption is usually the thing to question. Here every draft assumed the row
stores *a machine*, and modelled "may it travel" separately. It stores **an answer to a
question**, and `any` is one of the answers.

**And the technique has a ceiling worth knowing.** It finds assumptions shared by the
options — but both of us were generating options inside one frame, so nothing either
produced could expose the frame itself. What broke it was Owen restating the requirement in
his own words. So the companion habit is: **when two people have converged and it still
feels awkward, do not ask for another option — ask whoever wanted the thing to say what
they want again.**

### Two consequences confirmed as deliberate

**A queued row does NOT move when the drag-order changes.** Re-ranking affects new rows
only. This reverses what an earlier draft concluded — resolve at start, so a re-rank moves
queued work — and the reversal is correct *because the value is now visible*: a row that
says `this-pc` and means it is honest, where a row that silently re-pointed itself
overnight would not be. Recorded here because a frozen row reads as a bug six months on
unless somebody wrote down that it is not.

**`any` with no reachable enabled server holds and NAMES that**, rather than holding
silently. "Waiting for any server; none of the 2 enabled are reachable" is actionable;
a row sitting at `queued` with no explanation is the thing `/v1/activity` and every named
refusal in this document exist to avoid.

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

### 4.4 ONE BOOK = ONE GPU

**Owen, 2026-09-13, after reading the per-step design:** *"i guess we could say if one job
depends on the previous one, we keep the task list on the same GPU until the job is done.
maybe serialized jobs get a single GPU choice. one book = one gpu. i dont think theres a
speed benefit to splitting jobs like that. and it might overcomplicate the system."*

**Ruled, and he is right on both counts — but the speed argument is stronger than "no
benefit". Step-type routing, which this section recommended an hour earlier, would have
been actively SLOWER for his workload. That recommendation was wrong and the arithmetic
below is why.**

> **A dependency chain gets ONE machine choice. Every step of one book runs on the machine
> the book was assigned.**

`waitFor` therefore exists at two levels, not three: the **queue default** (a setting) and
**the book** (its row). There is no per-step override, because there is nothing for it to
buy.

### 4.4.1 Why per-step routing was wrong — the argument, then the numbers

An earlier draft argued that since the machines' speed ratio varies by step type, the best
plan is to put the highest-ratio step — `tts` — on the fast machine and let the rest
overflow.

**The decisive objection needs no arithmetic at all.** Routing all TTS to one card
**serialises every book's dominant step through it**, and leaves the other card doing
minutes of cleanup between hours of idleness. A plan that leaves half the hardware unused
for most of a night is wrong on its face, whatever the alternative saves. Hoarding the
dominant step on the fast machine does not exploit that machine — it makes it the only
machine.

**The numbers, stated second and deliberately so.** Take 15 books; on the PC each is about
`cleanup 10 min → tts 3 h → align 10 min`, so 3.33 h, and call the Mac 2x slower at 6.67 h.

| plan | wall clock | the other card |
|---|---|---|
| all `tts` on the PC | **45.0 h** | Mac busy 10 h, **idle 35** |
| one book = one GPU | **33.3 h** | both busy throughout |

33.3 h is combined throughput: `1/3.33 + 1/6.67 = 0.45 books/h`, and `15 / 0.45 = 33.3`.
So the margin is **1.35x — a third faster, not a multiple.**

**THIS NUMBER HAS BEEN WRONG TWICE AND THAT IS WHY IT IS NOT THE HEADLINE.** The first
draft of this section recommended step-type routing, reasoning from a speed *ratio* without
weighting it by *duration*. The correction then claimed one-book-one-GPU finishes in
"~15-20 h", which is **below the floor for two machines even if the Mac were exactly as
fast as the PC** (`15 x 3.33 / 2 = 25 h`) — arrived at by imagining both cards busy and
halving, without applying the 2x penalty to the Mac's share. Same species of error as the
one it was correcting. Both were caught by the Foundry session; the second was caught
because a contract carrying "3x" is a number somebody quotes later to justify a change that
then underdelivers by a factor of two.

The conclusion survived both errors intact, which is the point: **it rests on the idle
card, not on the margin.**

### 4.4.2 The other two reasons, both of which also hold

**Within one book there is nothing to overlap.** `cleanup → tts → align` is a dependency
chain, so the steps cannot run at the same time whatever machines they are on. Sending
`cleanup` to the slower machine makes that step slower, delays everything behind it, and
adds a transfer between steps. For a single book, splitting is strictly worse.

**The audio seam is satisfied for free.** `cuda-linux` renders Higgs through SGLang and
`mlx-darwin` through mlx-audio — different samplers, different RNG, different batching, and
**they do not produce the same audio**. Chapters 1-4 in one engine's voice and 5-8 in
another's is audible in a way no test asserts, and phase 6's guard would carry its running
median across a rate change on top of it. Under one-book-one-GPU that cannot happen, so it
needs no rule of its own. It was the whole justification in an earlier draft; it is now a
consequence.

### 4.4.3 And the simplicity argument is not a tiebreaker, it is a reason

*"it might overcomplicate the system."* Three levels of override is three places to look
when a book renders somewhere surprising, three states to show in a UI, and a per-step
field on every row that would be `null` for the entire life of almost every queue. Paying
that for a plan the arithmetic says is slower would have been the worst kind of
flexibility: expensive, visible, and wrong.

### 4.4.3a THE PRIOR: a draft that adds a level of override is probably wrong

Five rewrites of this section in one day, and **every wrong version was the more flexible
one**:

| draft | what it added | why it was wrong |
|---|---|---|
| `Any` as the default | a hidden preference | asserted the machines are interchangeable |
| `machine` + `overflow` | a modifier on a machine | a representable state that meant nothing |
| the default writing a name | an instruction nobody gave | broke the overflow case it existed for |
| three routing levels | per-step override | slower, per 4.4.1 |
| per-step-type defaults | a settings block | same |

Owen's corrections have all gone the same direction: **removing something a draft added.**

That is five for five, which is enough to stop treating it as an observation and start
treating it as a prior. Proposed as a rule for this queue, suggested by the Foundry session:

> **When a draft adds a level of override, the null hypothesis is that it is wrong.** The
> burden is on the addition to show a case that exists, is not harmful, and is not already
> served by the level above it.

It generalises a little beyond this section, but not infinitely: it is a claim about a queue
with one operator, two machines and a workload dominated by one long step. It is not a
claim that configurability is bad in general.

### 4.4.4 What this keeps, and what his two modes become

**Kept:** the queue default (section 4.2.1a) and per-book `waitFor` (section 4.2.1). Book 12
of 15 goes to the Mac by setting it on book 12 — every step of book 12 follows.

**Dropped:** per-step `waitFor`, and the per-step-type queue default.

His two modes still fall out of the levels rather than needing a toggle:

- **Whole book on one GPU** — now the only behaviour, so this is simply what a book does.
- **Fill empty slots until the queue is complete** — the queue default at `Any`. Each book
  takes the first server that will have it, so both machines stay busy and no book is split.

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

### 5.1 The work that has no percentage, and the one that looks like it but does

Owen, 2026-09-13: *"a few other places where bookforge touches rendering — there's a
streaming page on bookforge that streams audio. there's also the correct sentences/re-roll
page. and of course the browser extension that streams audio. those places are independent
of a queue but claim a server while they run and need to be accounted for logically in
crucible. that means crucible wont always have a percent complete to hand back."*

He is right about the accounting, and the reading of BookForge that followed found the
three are **not three of a kind**. They fall into two families, and the families want two
different Crucible doors.

**Family A — the streaming scheduler. No denominator, ever.**
`electron/stream-scheduler.ts` is the single owner of every sentence queue for listening,
and three surfaces drive it:

| surface | front door |
|---|---|
| the in-app Play tab | direct, in process |
| **the browser extension** and LAN clients | `electron/tts-api-server.ts`, ws `:8766` |
| the Bookshelf Reader on a phone | `electron/reader-stream-bridge.ts`, riding `:8765` |

All three speak a deliberate subset of one protocol to one scheduler. **So the Crucible
integration is ONE seam, not three** — a scheduler that can dispatch a sentence to a
Crucible streaming session instead of a local worker makes all three work at once, which is
what *"everything above it should be identical"* asked for.

These have no total. Rows arrive one `say` at a time for as long as somebody keeps reading.
`GET /v1/activity`'s `streaming.progress` is therefore **null**, permanently and by
construction, and what it reports instead is counts — `said`, `finished`, `in_flight`,
`seconds`, `chars`. BUILT 2026-09-13, along with the `claim` field, because until then a
bench polling this route read `busy: 0` and drew an idle machine **while the extension was
streaming from it**.

**Family B — correct-sentences / re-roll. It DOES have a percentage.**
`electron/correct-sentences-bridge.ts` does not touch the scheduler at all. It goes through
`parallel-tts-bridge.regenerateSentenceIndices` — the render path — over a **known,
finite set of sentence indices** chosen by the user before the work starts. That is a
denominator. It is a Crucible **`tts` render job**, it gets a real `progress`, and it
belongs in the lane like any other render.

So the rule is not "streaming surfaces have no percentage". It is: **a percentage exists
exactly when the client handed over the whole of the work up front.** A render job did; a
reader has not and never will. Sorting the three surfaces by *which door they already use
in BookForge* gets this right for free, and sorting them by "does it feel like streaming"
gets correct-sentences wrong.

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
