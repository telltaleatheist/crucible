# Phase 8: getting the evidence to the person

**Owen, 2026-09-13:** *"we're also going to have to find a way to get logs from the server
to the bookforge client that spawned the commands."*

He is right that this is missing, and it stops being a convenience the moment a render
happens on another machine: today the evidence lands on the server and the person is at the
client, and closing that gap means SSH.

This document is the design. It is written against one measurement and one rule.

---

## 1. The measurement that decides the shape

`crucible/jobs/queue.py`:

```python
def append_event(self, job: Job, kind: str, data: dict[str, Any]) -> None:
    event = {"id": len(job.events) + 1, "event": kind, "data": data}
    job.events.append(event)
```

**`job.events` is unbounded, and nothing evicts it.** That is deliberate and correct: the
SSE stream replays from `Last-Event-ID` by walking the list (`api.py`, the `while index <
len(job.events)` loop), so a client that drops its connection and reconnects is owed every
event it missed. Completeness is the feature. `JobStore._jobs` likewise keeps every job the
process has seen.

That is affordable for events, which are small and few — a render emits a handful per chunk.
**It is not affordable for logs.** A single vLLM start-up emits thousands of lines before it
serves anything, and narrator logs per chunk across a 1,400-chunk book. Making log lines
into events would multiply the retained set by an order of magnitude and hold it for the
life of the server, to carry text nobody reads unless something breaks.

So: **logs are not events, and must not share their transport.**

---

## 2. The rule

> **A log line must never be load-bearing. If the client needs a fact, it arrives as an
> event.**

This is the phase 6 lesson generalised. narrator today prints
`[HIGGS3][HIGGS_GUARD_EVENT] {json}` and `parallel-tts-bridge.ts:116` parses it back, which
works only because the renderer and its caller share a stdout. Over a socket it does not
work at all, and phase 6 is already replacing it with a structured `guard` field for exactly
that reason.

Every remaining place where behaviour is driven by parsing a log line is therefore a **known
defect with a deadline**, not a style preference — it is a thing that will break the first
time that work runs on another machine. The audit of those sites is a prerequisite to this
phase, not a follow-up (section 6).

---

## 3. Three tiers, by lifetime and purpose

Conflating these is why "get the logs to the client" sounds like one job and is three.

### Tier 1 — EVENTS. The contract. Already built.

Structured, ordered, replayable, complete, small. `progress`, `chunk`, `artifact`,
`warming`, `done`, `failed`, and phase 6's `guard`. These drive behaviour and are never
lossy. Unbounded retention is the right trade here and does not change.

### Tier 2 — THE LIVE TAIL. New, and deliberately lossy.

For watching a job that is running. A **bounded ring buffer per job** — the last N lines,
N in the low thousands — streamed on request.

Three properties that are not negotiable:

- **Opt-in.** A client asks for it; the default carries nothing. A bench drawing four
  servers must not be streaming four engines' debug output to render one line of text.
- **Filtered at the source.** A level, and a source filter (`narrator`, `sglang`,
  `crucible`), applied on the server. Shipping everything so the client can drop it wastes
  the network on the one path where the network is the scarce thing.
- **It says when it dropped.** A ring buffer that silently discards is worse than no buffer,
  because the reader believes they are seeing everything. A gap is reported as a gap:
  `{"dropped": 412}`. This is the same discipline as `capped: null` meaning "narrator did
  not say" — the absence of information is itself information, and must be transmitted.

### Tier 3 — THE FULL LOG, AS AN ARTIFACT. New, and nearly free.

For post-mortem, which is the case Owen actually described. Every subprocess's stdout and
stderr is written to a file in the job's own scratch directory and registered with
`ctx.artifact()` when the job ends.

It is nearly free because **it reuses machinery that already exists and is already tested**:
artifacts have a registry, an event, a fetch route (`GET /v1/jobs/{id}/artifacts/{name}`),
and an SDK method. A log file is an artifact like a FLAC is an artifact. Nothing new crosses
the wire.

It is also the tier that survives the thing Owen is actually trying to fix: the job is over,
the person is on another machine, and they want to know what happened.

**It is written for a FAILED job too** — in fact especially. A job that fails registers its
log before it finishes, so the artifact is there for the one case it exists to serve. (Note
for the implementer: this interacts with the failure path in `_fail_out_of_band`, which
deliberately bypasses `_finish`; the log must still land.)

---

## 4. What this does NOT do

- **It does not relay the server's own lifecycle logs.** Start-up, backend detection,
  refusals that happen before a job exists — those belong to whoever administers the server,
  not to a client. `GET /v1/activity` already answers "is it alive and what is it doing".
- **It does not make Crucible a log aggregator.** No search, no retention policy beyond the
  job's own life, no cross-job query. If that is ever wanted it is a different product.
- **It does not stream by default.** See tier 2.

---

## 5. The client side

BookForge's job row gains a way to see the tail while running and to fetch the full log
after. The natural home is the queue page's existing per-step detail — a step already has a
message and a progress bar, and this is the same row.

What must NOT happen is a second log-viewing mechanism: BookForge already surfaces spawned
process output for locally spawned work, and a remote job's log should arrive in the same
place. One viewer, two sources.

---

## 6. Prerequisite: the load-bearing log audit

Before tier 2 is built, every place that parses a log line to drive behaviour must be found
and listed. Known already:

- `parallel-tts-bridge.ts:116` — `GUARD_EVENT_PREFIXES`, both engines' guard events.
- the progress regex reading `Converting sentence N/M (P%)` from the render worker.

Each one is a fact that must become an event before the work it watches can travel. Phase 6
does the first. The audit says what else there is, and each finding is a small piece of work
of the same shape: **promote the fact to an event, leave the log line alone.**

Leaving the log line in place matters — it is what a person reads, and the two audiences are
different. The rule is not "stop logging", it is "stop *parsing*".

---

## 7. Order

1. **The load-bearing audit** (section 6). Cheap, and it sizes everything else.
2. **Tier 3**, the artifact. Highest value per unit of work by a wide margin, reuses tested
   machinery, and serves the case Owen described.
3. **Tier 2**, the live tail, once the ring buffer's size and the filter's shape are settled
   by watching a real render.
4. **The client viewer**, last, because it is the only piece that needs a UI decision.
