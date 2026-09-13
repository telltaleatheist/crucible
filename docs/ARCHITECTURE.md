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
  a multi-tenant system.
- **The retry lives in the SDK**, written once, so BookForge and Foundry inherit it and
  cannot drift.

### What it costs: a policy change, not a removal

`crucible/jobs/queue.py` stays. The lane still runs jobs; events, cancel, artifacts and
provenance all keep working. The change is at **admission**: refuse when the lane is busy
instead of appending to the deque. `queue_depth` becomes honestly 0 or 1.

Ripping out 340 tested lines to get behaviour a policy flag gives you is the expensive
version of the right idea — and if queueing is ever wanted back, it is the same one line.

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
5. Crucible's job door refuses when busy, with an informative 409.
6. The SDK carries the retry and the free-signal subscription.
7. BookForge shows "GPU busy: <client>".

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
