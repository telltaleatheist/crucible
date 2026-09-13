# Phase 6: the guard belongs to the model, and the chunks travel

**Owen's ruling, 2026-09-13.** *"I think the model and its inference should be
responsible for guarding. The system will still be available to the user. It's just
saved somewhere else. Getting the chunks back to Bookforge, though - that'll be a bit
harder. Because this should be designed to work on another computer. But it'll have to
transfer back to the main system. Which means we'll be changing up how narrator works."*

This document is that ruling worked out, and it **amends PHASE3-TTS.md** in three places
(section 1, section 3, section 6). Where the two disagree, this one is current.

---

## 0. The finding that makes this urgent, not just tidy

Before designing anything I went to read where the guard runs today, and it is not where
either document says it is.

narrator has **two rendering worlds**, and only one of them is guarded.

| | the audiobook world | the serve world |
|---|---|---|
| entry | `convert()` / `convert_batch()` / `convert_many()` | `narrator.serve` over stdin/stdout |
| driver | `truncation.GuardPlan` | a list comprehension |
| guard | PaceTracker, re-roll, split ladder, reject dir | **none** |
| batched | yes, `BATCH_SIZE` takes in flight | **no** for a served Higgs |
| output | `<sentences_dir>/<i>.flac` | base64 PCM16 on a pipe |

**Crucible drives the serve world.** Both doors do — `crucible/jobs/tts/render.py` sends
`generate_batch`, and the streaming door sends the same worker `stream: true` rows. So
for Higgs:

- `serve/worker.py:1057` `_generate_audio` takes the non-Orpheus branch and calls
  `render_audio(clean, index=index)`. One call, one waveform. No ladder.
- `serve/worker.py:1245` — the batch path for a served engine is
  `[self._generate_audio(t, ...) for i, t in enumerate(texts)]`. Sequential **and**
  unguarded.
- `v3_engine.py:1069` `generate_batch_stream` is batched (a `BATCH_SIZE`-wide pool) but
  its `render` closure is also a bare `self.render_audio(text, index=i)`.
- `mlx_backend.py:1760` is the same: `render_audio` on both its slab rung and its serial
  rung.

And `serve/worker.py:1478` says, in a docstring: *"The retake ladder and the truncation
guard for NON-streamed rows live inside generate_batch_stream."* That is **true for
Orpheus and false for both Higgs implementations.** It is the sentence that let the gap
sit unnoticed.

So the honest statement of where we are is not "`capped` is missing from the wire". It
is: **the render door has no guard at all.** PHASE3-TTS.md section 6's promise that
"Crucible never retakes — BookForge's PaceTracker reads those numbers and decides" was
only ever half-true, because the numbers it forwards describe an unguarded single take,
and BookForge's PaceTracker is not in that path either — the bridge reads guard events by
scraping `[HIGGS3][HIGGS_GUARD_EVENT]` off stderr, and a render driven through Crucible
produces none.

Owen's ruling fixes this by construction. A guard that lives with the model cannot be
left out of a path that uses the model.

---

## 1. What the ruling changes in PHASE3-TTS.md

**Section 1 said: "the server measures, the client judges."** It now reads: **the model
judges, the server forwards, the client orders.** The client still decides *what to
render* — the book, the chunking, the order, the voice. It stops deciding *whether a take
was good*, because that judgment needs the engine's own numbers (the frame cap, the seed,
the pace of the takes already shipped) and those never leave the engine.

**Section 3 said: "the ladder's steps are server config; the client asks for take N."**
The ladder's steps are now **engine config**, and the client does not ask for a take at
all. `take` leaves the wire as a request parameter. One request = one **accepted** chunk,
with its take history attached. This closes PLAN.md's owed ruling 5, which is the one
place the division of knowledge had a genuinely arguable alternative — Owen has now
ruled, and this is the answer.

**Section 6's `chunk` event** keeps every field it has and gains `guard`. `capped`
survives as a field but stops being the thing the guard depends on: the cap-hit is now
one input to a verdict the engine has already reached, not a number the client has to
reason from. This dissolves PLAN.md's owed item 2 rather than discharging it — nobody has
to get `capped` onto narrator's wire, because the wire carries the conclusion.

---

## 2. The shape: one renderer, two sinks

The temptation is to read "cut narrator loose from `sentences_dir`" as a deletion. It is
not. `sentences_dir` is a perfectly good thing for a render that is local, and BookForge's
assembly, RVC pass, aligner and resume scan all read a directory of `<i>.flac` and should
keep doing so.

What is wrong is that the *guarded batched driver* lives **below** the file-writing layer
instead of above it. `HiggsV3Engine.convert_many` is the driver, and it writes the files
itself, "here, for the same reason" the plan lives on that thread. So the serve world
could not reach the guard without also acquiring a directory.

The fix is to split `convert_many` at its retirement point:

```
                     ┌─ accepted chunk: (index, audio, verdict) ─┐
  rows ──▶ GuardPlan ┤                                           ├─▶ sink
           (unchanged)└─ retakes re-enter the same pool ─────────┘

  sink = write <sentences_dir>/<i>.flac        ← the audiobook world, unchanged
  sink = emit batch_item {..., guard: {...}}   ← the serve world, newly guarded
```

`GuardPlan`, `PaceTracker`, `render_guarded`, the split ladder, the reject dir, the
`MIN_GUARD_CHARS` short-side rule, the depth-first ordering: **none of it changes.** This
is a refactor of who calls the driver, not of what the driver decides. That matters
because this is the code that renders Owen's books, and a change to the policy would have
to be re-measured against the Mistborn pause map before anyone could trust it. A change to
the plumbing does not.

### The verdict is already being computed

`GuardPlan` takes an `on_event` callback and calls it for every guard decision;
`truncation.emit_event` is the default and writes the `[HIGGS3][HIGGS_GUARD_EVENT] {json}`
log line. So the verdict exists as a structured record today and is being thrown at
stderr. The change is to **collect** those records per chunk and attach them, keeping the
log line as well so the bridge's existing analytics keep counting while both paths are
alive.

---

## 3. narrator's wire gains one field

`batch_item` today:

```json
{"type": "batch_item", "i": 12, "format": "pcm16", "data": "…", "duration": 4.12, "sampleRate": 24000}
```

and gains exactly one key:

```json
"guard": {
  "verdict": "clean" | "short" | "long" | "hole" | "split" | "rejected",
  "takes": [ {"take": 0, "seed": 4711, "chars": 412, "seconds": 3.1,
              "chars_per_sec": 132.9, "capped": true, "outcome": "long"} ],
  "band":  {"reference": 14.8, "short": 11.4, "long": 19.2, "observed": 37, "warm": true},
  "parts": 2
}
```

Three properties this shape is chosen for:

- **Additive.** A client that does not read `guard` is unaffected, so the browser
  extension's Listen path and `orpheus-worker-pool.ts` need no change on the day this
  lands.
- **It is the conclusion, not the evidence.** `verdict` is what the engine decided.
  `takes` is why, for analytics and for Owen's eye, and a client that acts on `takes`
  instead of `verdict` is re-litigating a decision that has already been made.
- **`parts` is how many chunks came back for one requested index**, because the ladder is
  allowed to split. See section 5 — this is the one place the ruling makes the wire harder
  rather than simpler, and it has to be faced rather than papered over.

The Orpheus arm is untouched. Its guard is a different and older mechanism
(`_guard_truncation`, `engine/orpheus/guards.py`) and it already runs in the serve world.
It gains `guard` on its `batch_item` when someone has a reason to want it, not as part of
this.

---

## 4. Crucible forwards, and stays stateless

Two changes, both small, because the server was built for this.

**`chunk` gains `guard`**, forwarded verbatim from narrator, `null` when narrator does not
say. Crucible does not read it, does not validate its contents beyond it being an object,
and does not act on it. That is the same discipline `model_provenance` already has, and it
is what keeps `api_version` at 1.

**The pace state round-trips through the client.** This is the one tension the ruling
creates: the guard re-centres on the running median of the book's own shipped takes, which
is per-**book** state, and DESIGN.md section 10 says Crucible keeps no session, no project
and no per-user state between jobs.

The resolution is that **the client carries it, and the server never remembers it**:

- `tts` params gain `pace` — an opaque object, absent on the first chapter.
- `done_extra` gains `pace` — the tracker's state as the job leaves it.
- BookForge hands chapter two the object chapter one returned.

Crucible stores nothing, and is therefore still stateless in exactly the sense DESIGN.md
means: two jobs from two clients cannot see each other, and restarting the server loses
nothing. The state lives with the thing that owns the book, which is the client, and it is
carried rather than remembered. A job is a **chapter**, which is also the natural unit for
a progress bar and for a resume.

This matters more than it looks. A guard centred on too few chunks fires on healthy ones —
which is precisely the failure raising the short factor to 1.3 was meant to stop — so a
book rendered as 40 cold-started chapters would be measurably worse than the same book
rendered as one run. Carrying the state is not an optimisation.

---

## 5. The hard part: a split chunk breaks the one-index-one-file rule

The ladder is allowed to decide that a chunk is unsalvageable whole and render it as two
halves. In the audiobook world that is invisible: both halves are concatenated into one
`<i>.flac` before it is written, and the caller never learns. `truncation.join_parts`
does it.

That stays true, and it is the right answer: **Crucible returns one artifact per requested
index, always.** `parts` on the verdict says the ladder split, for analytics, and the
audio is already joined. A client that asked for chunk 12 gets `12.flac`.

The alternative — returning two artifacts and letting the client re-index — would push the
ladder's private business into every consumer, break the resume scan's `<i>.flac` test,
and desynchronise the `.sentences.vtt` sidecar from the audio. It is rejected.

---

## 6. BookForge: the chunks arrive over the wire

This is the largest piece of work, and it is BookForge's, not Crucible's.

`electron/parallel-tts-bridge.ts` is 10,586 lines built on one assumption: **the renderer
and the assembler share a filesystem.** Every WSL path rewrite, the
`normalizeWslSessionToWindows` copy, the `\\wsl$` staging, `findE2aProcessDir` — all of it
exists to keep that assumption true across a VM boundary. Phase 3 was always meant to
delete that layer rather than patch it. This is where it gets deleted.

The replacement is deliberately small, because the local directory stays:

**6.1 A fetcher.** As each `artifact` event lands on the job's SSE stream, GET it and
write it to the local `sentences_dir`. Nothing downstream changes — assembly, the RVC
pass, the aligner and the resume scan all still read a local directory of `<i>.flac`. It
is filled by a download instead of by a subprocess. Per-chunk fetch, issued as each
artifact is announced, overlaps with the next chunk still generating, so the transfer is
free in wall-clock terms for any book longer than a few chunks.

**6.2 A listing endpoint and a batched fetch**, for the reconnecting client. A client that
was offline for 200 chunks must not make 200 requests. `GET /v1/jobs/<id>/artifacts`
already exists; the gap-filling fetch is the new part.

**6.3 Resume gets a sha.** The current test is "the file exists and is strictly larger
than 1024 bytes" (`render/worker.py` `RESUME_MIN_BYTES`). That was sound when narrator
wrote into the session directory, because a half-written file was the only corruption
mode a crash could produce. Over a network, **truncation is new**: a socket that closes
mid-body leaves a plausible FLAC of plausible size. Put the chunk's sha256 in the
provenance sidecar, check it on resume, and re-fetch on a mismatch.

**6.4 The guard analytics move off stderr.** `GUARD_EVENT_PREFIXES` at
`parallel-tts-bridge.ts:116` scrapes both engines' log lines. It gains a second source:
the `guard` field on the chunk event. Both live side by side — a locally spawned narrator
still prints the lines, and a remote render has no stderr to scrape.

**6.5 What does NOT change.** The chunk packer, `prepareNarrationInput`, the narration
text pass, the `.sentences.vtt` sidecar, the assembly, the coverage gate, the chapter
markers, the library. Those are the client's business and the ruling does not touch them.

---

## 7. Order of work

1. **narrator N1** — split `convert_many` into driver + sink; route the serve world's
   Higgs batch through the driver; `guard` on `batch_item`. Tested against
   `serve/fake_engine.py`, which already speaks the protocol.
2. **Crucible N2** — forward `guard`; `pace` in and out; drop `take` from the wire; amend
   PHASE3-TTS.md sections 1, 3 and 6 to point here.
3. **BookForge B** — the fetcher, the sha'd resume, the second analytics source. Behind
   the existing `crucible` provider, so nothing moves for a local render until Owen
   switches it.
4. **A keeper on a real card**, which needs Owen's go and a free 3090 Ti: render a chapter
   through the remote path, and compare its guard-fire count and pause map against the
   Mistborn baseline. **Nothing here is trustworthy until that runs** — a guard refactor
   that changes behaviour is exactly the thing a fake engine cannot detect.

Steps 1-3 are CPU-side and testable without a card. Step 4 is the gate.

---

## 8. What this deliberately does not do

- **It does not touch the Orpheus guard.** Different mechanism, different engine, already
  in the serve world.
- **It does not change a single guard decision.** Same `GuardPlan`, same `PaceTracker`,
  same bands, same `MIN_GUARD_CHARS`, same depth-first order. If a measured render
  disagrees with the Mistborn baseline, that is a bug in this work, not a new policy.
- **It does not extract narrator into its own repo.** Still PLAN.md's owed ruling 1, still
  Owen's call, still the thing blocking `crucible install tts`. This work makes the case
  stronger — narrator now has a real wire protocol and not just a private pipe — but it
  does not depend on the answer.
- **It does not make Crucible remember anything.** Section 4 is load-bearing: the moment
  the server holds pace state between jobs it has acquired a session, and DESIGN.md
  section 10 stops being true.
