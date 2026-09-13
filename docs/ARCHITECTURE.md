# The system: BookForge, Foundry, Crucible

**Owen, 2026-09-13:** *"i feel like the whole system we have, with bookforge, foundry, and
crucible all connected, is chaotic."*

He is right, and this document is the diagnosis, the rules that follow from it, and the
order to apply them. It is written from a four-part audit of all three repos, with
file:line evidence behind every claim; where a claim could not be verified it says so.

---

## 1. The diagnosis

The chaos is not complexity. Each piece is individually well-built and heavily commented.
It is this:

> **Almost every defect found is one fact with two owners and nothing comparing them.**

Seven instances, found in one day:

| The fact | Copy A | Copy B | Consequence |
|---|---|---|---|
| `thirdreich`'s safe band | overlay: 500-700 | Crucible: 600-1000 | chunks packed into a **measured 12.5% and 18.8% failure zone**. *Fixed.* |
| the acronym list | `caps_acronyms.json`: 15 | Foundry: 14, no COVID | Foundry's validator accepts what BookForge's refuses; **both stamp `n6`**. *Owned by Foundry.* |
| the engine's `log()` count | declared 28 / prose 25 | actual 32 | a documentation guard that went red and was stepped over. *Fixed.* |
| the VLM completion marker | Foundry moved it to `<bank>.completed.json` | BookForge reads `completed.json` | a finished conversion reads as interrupted |
| dots.ocr's vLLM arguments | 3 places | 3 different answers | same model, same card, different KV pool per launcher |
| desktop VRAM allowance | Crucible: 3072 MiB | BookForge: 1500 MiB | two declared facts about one host, 2x apart |
| which Foundry commit is vendored | `VENDORED.md`: `3b13392` | keeper: `9f4ee4e` | the keeper proves a five-week-old engine |

None of these is hard. Every one is invisible until it reaches output. **That is what
"chaotic" feels like from the inside**: not that the system is complicated, but that no
part of it tells you when it has quietly stopped agreeing with another part.

### The second-order finding

Three of the seven were caught by a guard that **already existed and was ignored**: the log
count had been red long enough that nobody remembered, the vendor keeper is anchored five
weeks back, and the marker-strip check was silently satisfied by a stale assumption. A
guard that is red and tolerated is worse than no guard, because it launders the absence of
checking into the appearance of it.

---

## 2. The rules

### R1. One fact, one owner, named in writing.

Every duplicated fact above needs an owner declared where both copies can see the
declaration. Where a copy must exist (Crucible has to run on a machine that never heard of
BookForge), the copy is **derived and checked**, never authored twice.

The band fix is the template: the value corrected, the provenance and sample size written
into the manifest, and `scripts/check-voice-bands.py` added and mutation-tested so the next
drift is caught rather than shipped.

### R2. A guard that is red is a broken guard.

If a check cannot be made green it must be deleted or fixed, never left failing. A tolerated
red is how three of the seven survived.

### R3. Nothing is ever told "maybe."

The sharpest single defect in the audit, `gpu-arbiter.ts:88-99`:

```
timed out after 600s waiting for GPU (held by tts:A); proceeding WITHOUT the lock
```

The promise **resolves normally**, so callers cannot distinguish holding the lock from
having given up. Three call sites then set `holdsGpu = true` on a failed acquire, and their
later release is a silent no-op. Worse: a waiter that times out never becomes the holder, so
its `onYield` is dropped — after a ten-minute wait the text server is up holding ~20 GB and
**nothing can ever ask it to step off again.**

> **You either hold the card or you do not. A caller is never handed an ambiguous answer.**

This is the same rule as section 3's queue policy, one layer down — which is why they are
one piece of work and not two.

### R4. A log line is never load-bearing.

The audit found **39 distinct text contracts** parsed to drive behaviour across the three
repos. Five of them *kill a process* (three watchdogs, plus `GENERATION_ACTIVITY_RE`), and
one greps a **log file to decide whether a 19 GB model may render a book**
(`v3_served.py:1080+`).

None survives a network hop, because a remote job has no stderr to scrape. Every one is
therefore a **defect with a deadline** rather than a style preference. The fix shape is
always the same: **promote the fact to an event, and leave the log line alone** — the two
audiences are different, and "stop logging" is not the instruction.

### R5. Queues belong to clients. Admission belongs to the server.

Section 3.

### R6. Partial work survives failure, always.

Owen's atomicity ruling, correctly scoped (PHASE7-LANES.md section 4.3): a job is atomic in
**claiming**, never in **effect**. A killed VLM read banks the 399 pages it finished; a dead
render keeps its 900 chunks. No failure cleanup, no rollback-to-clean, no tidying a partial
output.

---

## 3. The queue ruling

**Owen, 2026-09-13:** *"i think all queuing logic should exist in the clients, not the
server. if the server is busy, it cant receive a new job. if its not busy, it receives the
next job requested."*

**Ruled, and it is right for a better reason than simplicity.** The client is the only thing
that knows what the user wants — the chain, the pin, the priority, which book is being
watched. A server-side FIFO can only ever be a dumb queue, and having one forces the smart
client queue to *model* it. Two arbitrators, one strictly less informed.

**It is also already half-built, and that inconsistency is itself part of the chaos.**
Crucible's two doors disagree today: the streaming door refuses (`409
stream_session_open`, naming the holder, `ttsstream.py:980-988`) while the job door queues.
One server, two policies, no reason.

### What it means concretely

- **The job door refuses when the lane is busy**, exactly as the stream door already does.
- **The refusal is informative or it is useless.** A bare "busy" forces clients to poll, and
  polling is a *worse* queue than FIFO: the winner is whoever polls at the luckiest moment
  rather than whoever asked first. So the 409 carries `{holder, job_type, model, since,
  progress}` — which is also exactly the *"GPU busy: Foundry"* message Owen wants, for free.
  The `client` field recorded on every job (section 5 of PHASE7-LANES) is what makes it
  nameable.
- **A "card is free" edge signal** so clients need not poll at all. Two clients waking
  together is a millisecond race, which is fine: this is one person with three machines, not
  a multi-tenant system. *(Deferred, with reasons — see 3.2. A lane-free signal would lie
  while a streaming session holds the card.)*
- **The retry lives in the SDK**, written once, so BookForge and Foundry inherit it and
  cannot drift. *(Still owed; the 409 now carries everything it needs.)*

### What it costs: a policy change, not a removal

`crucible/jobs/queue.py` stays. The lane still runs jobs; events, cancel, artifacts and
provenance all keep working. The change is at **admission**: refuse when the lane is busy
instead of appending to the deque. `queue_depth` becomes honestly 0 or 1.

Ripping out 340 tested lines to get behaviour a policy flag gives you is the expensive
version of the right idea — and if queueing is ever wanted back, it is the same one line.

---

### 3.1 BUILT, 2026-09-13 — what shipped, and the three questions it had to answer

`JobStore.refuse_if_busy()` is the whole of it. `POST /v1/jobs` calls it; `enqueue` calls
it again and is the authority. The lane, the deque, `position`, `queue_depth`, cancel,
events and provenance are untouched.

```json
409 {"error": {"code": "server_busy", "message": "…", "details": {
  "holder":  "bookforge/owens-pc crucible-client/0.4.0",
  "job_id":  "51e14d2c…", "type": "tts", "model": "sigma",
  "status":  "running", "since": "2026-09-13T18:02:11Z",
  "progress": 0.42, "message": "rendering 118 of 280"}}}
```

Two fields beyond the sketch above, both earning their place. **`status`** makes `since`
unambiguous — "running" dates from `started`, "queued" from `created` — and without it a
client cannot tell a render that has been going an hour from one admitted two milliseconds
ago. **`message`** is the holder's last progress line, which is what turns *"GPU busy"*
into *"GPU busy: rendering 118 of 280"*. `holder` is null when the client sent no
User-Agent: **null means it did not say**, and a name invented here would make a bench
confidently wrong about whose render is on the card.

**What counts as busy — the lane, and separately the card.** Admission is about **the
lane**, uniformly, `echo` included. Exempting the types that need no accelerator would put
them straight back on a deque, because the lane is exclusive whatever a job wants from it,
and the server would be queueing again for exactly the jobs it claimed not to queue for.

`Residency.warming` needs no check of its own: it is only ever set from inside a load job,
which is on the lane, so it is strictly a sub-state of "a job is running". Adding a second
test for it would be a guard against a bug rather than against a state.

**A streaming session is the other way to be busy**, and it does not occupy the lane — so
`refuse_if_busy` can truthfully say the server is free while the card is not. That half
stays per-job-type in `preflight`, through `Residency.refuse_if_claimed`, because the claim
is about *narrator's one stdin and one stdout* rather than VRAM in general, and the honest
answer differs by type. `align` and `unload-aligner` were **added** to the askers: they are
the third mutator of residency and were being accepted and then failed a minute later at
`_refuse_mutation_if_claimed`. `asr` and `rvc` were deliberately **not** added — they never
touch the resident engine, and what they contend for is memory, which `accelerator.guard`
already refuses by name in their preflights. Making them ask would quietly redefine the
claim from "the wire" to "the card", which is a different rule and would need ruling as one.

**The race.** `enqueue` appends and sets `_wake`; the lane is a task on the same event loop
and does not resume until the current one yields, so a job can be admitted with nothing yet
running. A check that read only `_running_id` would let a second job in and `_pending` would
reach 2 — the queue, back, reachable by two clients a millisecond apart. So admission reads
**the deque**, in the store, where the lane's state is authoritative; the check and the
append then happen in one synchronous stretch on the event loop, with no lock and no HTTP-
layer coordination. A job admitted but not yet started is reported as the holder, with
`status: "queued"` and `since` its `created`.

### 3.2 DEFERRED: the "card is free" edge signal

**Described, not built.** Crucible's SSE machinery is per-resource — a job's event log, a
session's frame log — and a server-wide lane-edge stream is a new shape with three
decisions in it, not a wiring job:

1. **A lane-free signal would lie.** "The lane is free" is not "your job will be admitted":
   with a streaming session open, `tts`, `align` and every load/unload are still refused
   `engine_in_use`. A signal that fires and is then followed by a refusal is worse than
   polling, because a client will build a retry on it. An honest signal must carry the
   claim as well as the lane — or be per-job-type, which is a different endpoint again.
2. **The claim is released off the event loop.** `Residency.release` runs on a worker
   thread, and the stream watchdog closes sessions from another; neither has a loop
   reference or any notification path. Making the claim observable as an edge means giving
   `Residency` a thread-safe publisher, which is a real addition to the one file in this
   server that must not be made subtle.
3. **Connect-time level, then edges.** A client that subscribes while the server is already
   free must be told so immediately or it waits for an edge that has passed. That is a
   cursor-and-replay question, and both existing SSE doors answer it differently.

And the consumer is the SDK, which is a separate package: a server-side edge stream with no
client half is a feature nobody can use, with a wire to keep compatible forever.

**Until it exists, clients poll `GET /v1/activity`** — which is cheap by design (the
`nvidia-smi` probe is opt-in), already reports `running`, `progress` and `client`, and is
now explicitly documented as a display and a preflight, never admission. The refusal itself
carries `progress`, which is enough to back off for roughly as long as the holder has left,
so the polling this avoids is the tight kind rather than the periodic kind.

---

## 4. Where each thing lives

The settled ownership, for reference:

| Concern | Owner | Notes |
|---|---|---|
| the queue, ordering, priority, pins | **the client** | section 3 |
| admission ("is there room now") | **the server** | only it can answer |
| the guard and the retake decision | **the model + its inference** | Owen, 2026-09-13; PHASE6 |
| chunking and the order of work | **the client** | PHASE3-TTS section 1 |
| engine tuning (sampling, caps, bands) | **Crucible config** | never a wire field |
| all text processing | **Foundry** | except `tts-punctuation.ts` |
| scheduling, when hosted | **BookForge** | `foundry-host-queue.ts`: "scheduling crosses, execution does not" |
| execution | **whoever owns the runner** | Foundry's runner writes its own ledger |
| safe bands | **BookForge's overlay** | Crucible derives + checks (R1) |
| `NORMALIZER_VERSION` / `PUNCTUATION_SPEC_VERSION` | **Foundry** | Owen, 2026-09-05 |

---

## 5. The order

Sequenced by what unblocks what, not by severity.

**Now — silent output defects.**
1. ~~`thirdreich`'s band~~ **done**, with a checker.
2. COVID in `SPOKEN_AS_WORD` — Foundry's, raised with them; needs a stamp decision.
3. The VLM completion marker path — BookForge reads where Foundry no longer writes.

**Next — R3, the ambiguous answer.**
4. `acquireGpu` must resolve with a verdict, not a bare resolve; callers must stop setting
   `holdsGpu = true` on a timeout. This is a small change and it is the highest-value one
   in the audit.

**Then — R5, the queue policy.**
5. ~~Crucible's job door refuses when busy, with an informative 409.~~ **done** (3.1),
   with the card half extended to `align` and the free-signal deferred with reasons (3.2).
6. The SDK carries the retry. The free-signal subscription waits on 3.2's three decisions.
7. BookForge shows "GPU busy: <client>" — every field it needs is on the 409 already.

**Then — R4, the log contracts.** Phase 6 removes the guard-event scrape. The remaining 38
are each "promote the fact to an event", sized by the audit's table and done as their
owning feature travels. **PHASE8-LOGS.md** covers the transport.

**Continuously — R1 and R2.** Every fix ships with the check that would have caught it, and
no red guard is left standing.

---

## 6. What was NOT found

Recorded because a clean result is evidence too, and because these were the things most
likely to be quietly rotten:

- **The vendored `foundry-app/` is byte-identical** to Foundry's `app/` on every shared
  source file. The seam is clean; only the keeper's anchor is stale.
- **Foundry's rotation invariant holds**: a run that produces nothing leaves the catalogue
  exactly as it was, on both cancel and failure.
- **The drain/idle ordering is consistent on both sides** of the hosted seam.
- **Crucible never evicts anyone else's process**, and says so in three places that agree.
- **RVC assets agree** across both catalogs, 7/7 on sha and byte count.
