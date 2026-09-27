# Phase 10: narrator comes apart along the lines that already exist

**Owen, 2026-09-13:** *"i think narrator should probably be split up. isnt it responsible for
handling assembly as well, which is a local cpu task, not a gpu task? we should probably hand
the gpu portions to crucible and the cpu/management parts to bookforge."* — and, on how:
*"let's do this the RIGHT way, not the cheap/fast way."*

He is right, and the code agrees with him more than it looks. This document is the four moves,
in order, and the check each one owes.

---

## 0a. The test that decides which side anything goes

**Owen, 2026-09-13:** *"system-intensive work belongs with crucible. deterministic or cpu work
can remain with the client, since it's simple. crucible is designed to distribute the load of
compute intensive actions."*

This supersedes the GPU/CPU framing used elsewhere in these documents, which is a proxy and
occasionally the wrong one. The operative word is **distribute**: Crucible exists to move load
onto another machine, so the question is not "is this heavy" but **"is this worth moving."**
That is two conditions, and a thing has to pass both:

1. **Heavy enough** that the compute dominates the round trip.
2. **Self-contained enough** that its inputs and outputs fit on a wire without the transfer
   swamping the saving.

Every existing job type passes both — `llm`, `tts`, `asr`, `align`, `rvc`, `vlm-pages` all take
a small input and return a small output after real compute. And the test explains the two
boundaries that would otherwise look arbitrary:

| Work | Heavy? | Self-contained? | Side |
|---|---|---|---|
| a render, a transcription, a page read | yes | yes | **Crucible** |
| **assembly** (ffmpeg, chapters, m4b) | yes | **no** — needs the whole book's audio in one place | client |
| **video-assembly** (subtitle video) | yes | **no** — GBs in, GBs out | client |
| chunking, the gap classifier | **no** — milliseconds | yes | client |

So assembly stays client-side **not because it is CPU** — it is genuinely heavy — but because
nothing is saved by moving a 20-hour book across a network to run ffmpeg on it. That is a better
reason than the one section 3 originally gave, and it survives someone pointing out that
encoding an m4b is not cheap.

### The gap this test finds

**`final-denoise` is a Crucible job type that does not exist.** BookForge runs it as a GPU queue
step (`electron/queue-steps/final-denoise.ts`) — a roformer pass over a session's sentences,
reading a directory and producing a directory. That is **the same shape as `rvc`**, which *is* a
job type. It passes both conditions and belongs on the server side.

Its own docstring is this principle already being applied by hand, one layer down: it was split
out of the assembly because the assembly *"declared itself a GPU step whenever the flag was set —
holding the one GPU slot through the gap pass, the denoise, AND the chapter combine and AAC
encode that follow, which are pure CPU and are the long tail of the job."* Separating the
compute-intensive part from the deterministic tail, so the card frees up, is exactly what this
section says at the scale of a whole machine.

`video-assembly` is the counter-example and stays where it is, despite being marked a GPU step.

**Owen ruled on both, 2026-09-13:** *"denoise and rvc should be crucible work. video assembly
should be client."* `rvc` already is a job type, so `denoise` is the one to build.

### What a `denoise` job type actually costs — and it shares an env with `rvc`

Measured rather than estimated:

- BookForge runs it through `electron/denoise-bridge.ts` — *"block-based mel-band roformer…
  the audio-separator package inside the RVC engine env (rvc-env)"*, model
  `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt`. It slices the audio, processes, and
  splices back at recorded offsets; the roformer **preserves timing exactly**, which is what
  lets it run over a sentence set without moving any cue.
- The same directory holds the **dereverb** and **dereverb-echo** roformers, so this is a
  family, not one checkpoint.
- Crucible's `envs/rvc/cuda-linux.txt` installs `ultimate-rvc`, and `ultimate-rvc`'s own
  `pyproject.toml:86` declares `audio-separator` as a git dependency. **So Crucible's RVC env
  already carries the runtime this needs, transitively.**

What is left is therefore small and conventional: a `denoise/*.toml` manifest family for the
roformer checkpoints, a worker beside `crucible/jobs/rvc/worker.py`, the job type, and an
`enable_denoise` capability flag. No new env.

**One defect to fix on the way, in both repos.** `audio-separator` reaches this code only as
*somebody else's transitive dependency*. `envs/rvc/cuda-linux.txt` does not name it, and neither
does anything on BookForge's side — so the day `ultimate-rvc` drops or moves it, denoise breaks
with no warning and nothing in either repo has claimed to need it. That is R1's shape with the
owner offstage entirely. A job type that depends on a package directly must **pin it by name in
its own recipe**, exactly as the tts recipes pin narrator rather than assuming an env has it.

---

## 0. The rule this whole phase is written under

The 2026-09-13 audit (`ARCHITECTURE.md`) found seven duplicated facts in one day. **Every one
of them sat on a boundary**, and three survived behind guards that were red and tolerated. The
chaos was traced to three extractions — text→Foundry, inference→Crucible, narrator→its own
package — *"each copying facts across a new boundary, each guarded by a check written in the
past tense."*

> **So every move below creates exactly one boundary, ships the check that compares across it,
> and does not begin until the previous move's check is green.**

That is what "the right way" means here, concretely. Not more design — fewer simultaneous
seams, each with a live comparison.

---

## 1. Move one: narrator leaves BookForge, whole

**Not decomposed. Extracted.** These are different decisions with different urgency, and
conflating them would create four boundaries at once.

**The problem it solves.** `crucible install tts` today runs:

```
narrator[higgs-v3-server] @ git+https://github.com/telltaleatheist/bookforge@4ebc529f…#subdirectory=python
```

`telltaleatheist/bookforge` is **private**. So installing an inference dependency requires
credentials to an entire private application repo. On Owen's own machines that is an
inconvenience. Under `PHASE9-CAPABILITY.md` it is a blocker: install is supposed to probe a card
and decide what that server can do, on a 6 GB machine, on a droplet, on any box that is not his.
None of them can install narrator.

**Why it is nearly free.** The boundary already exists — narrator is already consumed as a
pip-installable package from a git URL, pinned to a sha, in three recipe files. Only the URL
changes.

**Live defect this move had to fix first, and it is FIXED (2026-09-13).** The pin was
`4ebc529f`, **five commits behind** on `python/`, and those commits are the whole of phase 6:

```
fe7f35cb  feat(narrator): the guard runs where the model does
6efa16e5  test(narrator): the phase-6 keeper — a guarded render on a real card
c3bedef3  fix(higgs): a base-weights server can be attached to
04784608  fix(text): one acronym list, read by every reader
0630dd1c  fix(narrator): MLX reaches parity, the empty-sentence door is declared
```

`crucible install tts` was building an env whose narrator had **no `render_many`** — the
server-side forwarding was written and the SDK reads `guard`, and the installed narrator could
not emit it, so the field would have been `null` forever with nothing saying why. All three
recipes now pin `770415db591e0af2cc650cf8c889224972e8906d`
(BookForge `feat/narrator-guarded-serve`, pushed so the sha is fetchable), narrator's
`pyproject.toml` is byte-identical across the two shas so every restated pin still holds, and
`test_every_tts_recipe_pins_the_same_narrator_commit` now refuses a bump that lands on two of
the three. **The pin bump is no longer a prerequisite of this move.**

### The check this move owes

`tests/test_cli.py` asserts each of the three recipes pins *a* 40-character sha. It does **not**
assert they pin the *same* one. `tests/test_workerenv.py::test_both_rvc_recipes_pin_the_same_commit`
is exactly that check for RVC and is four lines. Copy it.

---

## 2. Move two: the model's facts move into narrator — and this REVERSES a ruling

### The finding that decides it

**narrator carries no facts about the voices it renders. Not one.** A search of
`python/narrator` for voice data returns only test goldens. Every cap, safe band, sampling
number and context length is authored elsewhere and handed in at call time.

> **The thing that runs the model knows nothing about the model.** That is the defect, and it is
> why this fact keeps having to be copied.

Today it is authored in BookForge's `electron/data/higgs-models.json` and
`electron/data/higgs-safe-bands.json`, mirrored into Crucible's seven `voices/*.toml`, and
compared by `scripts/check-voice-bands.py` — a script that exists precisely because there are
two copies. (It compares the band and nothing else; `max_chars`, the pace rates, the sampling
block, `hf_repo` and `revision` are authored twice and uncompared. That is how `thirdreich`'s
band stayed wrong in one copy while right in the other.)

### The ruling, stated as a reversal

**Owen, 2026-09-13:** *"if model-specific configuration data is contained inside crucible, and
foundry needs that info to run a process, it should be able to retrieve that info from crucible.
a single source of truth."*

The instinct is right. The destination is one step further than Crucible, for two reasons:

1. **Crucible cannot be the source when there is no Crucible.** BookForge must work with no
   server registered. If caps come from a server, a machine without one has no caps.
2. **A safe band is a measured property of the model.** It belongs with the model, not with
   whoever happens to serve it.

So: **the model's facts ship inside narrator.** Crucible serves them on `/v1/voices`; BookForge
reads them from the installed package. One author, two transports, and the version pin makes the
two readers agree *by construction* rather than by a comparison script.

> **THIS REVERSES `ARCHITECTURE.md` §4**, which currently reads *"safe bands: **BookForge's
> overlay** | Crucible derives + checks (R1)."* That line is superseded by this section. The
> reversal is deliberate and is recorded here rather than left to drift, because two things each
> believing they are authoritative is worse than either arrangement on its own.

### The check this move owes

`scripts/check-voice-bands.py` survives, **pointed the other way**: Crucible's manifests and
BookForge's catalog both become derived views, and the checker compares each against narrator's
declaration. It must also widen beyond the band to every field authored on both sides —
`max_chars`, the pace rates, the sampling block, `hf_repo`, `revision` — because those are the
ones that drifted while the checker watched the one field it knew about.

---

## 3. Move three: assembly goes to BookForge

**Measured, 2026-09-13: `assemble/` and `engine/` are already fully decoupled in both
directions.** Neither imports the other. The single fact they share — `edge_fade` / `pads` — is
*deliberately* duplicated in `assemble/engine_profiles.py`, and
`tests/test_engine_protocol.py::test_the_engines_agree_with_the_assemblers_own_table` loads the
assembler's table by path and compares it.

Someone already decided assembly must not import the engine, and defended it with a check. The
seam is cut. This move is recognising it.

Assembly is ffmpeg, chapter markers, the m4b and the `.sentences.vtt` sidecar — no model, no
card, and the most audiobook-specific code in the system. It is BookForge's.

**One dependency to carry with it:** `assemble/sentence_vtt.py` imports `spoken` and
`split_sentences` from `text/paragraph_packer.py`. That resolves for free, because section 4
sends the chunker to BookForge as well — but only if the two moves land together or in that
order.

### The check this move owes

`test_the_engines_agree_with_the_assemblers_own_table` crosses the new repo boundary with the
code. It is already the right shape; it just has to keep running once the two halves are in
different repos.

---

## 4. Move four: `text/` — the hang-up dissolves on inspection

This was held back as "the contested one, needing its own ruling." **Measuring it removes the
contest.** `text/` is 5,242 lines wearing one name and covering four owners, which is why *"where
does text go"* had no answer:

| What it really is | Modules | ~lines | Owner |
|---|---|---|---|
| **Chunking** — flat text into generation chunks | `packer.py`, `paragraph_packer.py`, `chapters.py` | 3,368 | **BookForge** — *"chunking and the order of work"* is the client's (PHASE3-TTS §1) |
| **Book I/O + session staging** | `epub.py`, `prep.py`, `sentences.py` | 935 | **BookForge** |
| **Prosody** — how much silence sits either side of a chunk | `gaps.py` | 129 | **BookForge** — an audio decision that is not text at all, but deterministic arithmetic over chunk metadata, so section 0a keeps it client-side |
| **Text transformation** | `normalize.py`, `lang.py`, `sml.py` | 764 | **Foundry**, by the 2026-09-05 ruling |

So *"all text processing is Foundry's"* applies to **764 of 5,242 lines** — and most of that is
not live. `normalize.py`'s own docstring records that **six whole transforms are gated off** for
Orpheus (abbreviation expansion, the quote rewrite, the punctuation-run collapse, the
letter/digit spacing rule, and two more) because fine-tunes are trained on book-exact text, and
that **number normalization was moved to BookForge's model pass and is permanently skipped
here** (2026-09-02). What remains live is `foreign2latin` and a thin `normalize_text`.

**The hang-up I had asserted — "packing is engine-aware, so it conflicts with Foundry's
ownership of text" — was wrong. Packing is not text processing.** Chunking is the client's by a
ruling already made, so there is no conflict to resolve; there is a misnamed directory to split.

### The one genuinely contested thing, and it is a function

**The caps fold lives inside the chunker and is text transformation.**
`paragraph_packer.fold_caps_run` and `caps_acronyms.json` decide whether `THE NASA FILES` reads
as printed or is title-cased. It is read by three code paths by design, and on 2026-09-13 a
**fourth** was found: `foundry/src/clean/tts-spoken-forms.ts:284` hard-codes a 15-entry copy,
four lines below a comment describing the COVID incident that the same divergence caused eight
weeks earlier.

That is the only part of `text/` that needs a ruling rather than a rename.

### The check this move owes

`tools/test-one-fact-one-owner.js` (added 2026-09-13) already computes both folds' keep-sets
from the JSON and asserts equality, and refuses a second hardcoded set in `listen-text.ts`. It
has to grow the Foundry side — which means Foundry reading the JSON rather than restating it.
Raised on the agent channel; the stamp decision is Foundry's.

---

## 5. The direction this implies, which is not a move

`render/` is the local rendering path and Crucible's `tts` job is the remote one. Those are the
two rendering worlds PHASE6 §0 found, and phase 6 unified only their *guard*.

`DESIGN.md` already says clients send bytes over HTTP *"even to a server on localhost."* Follow
that through and there is **one** render path: BookForge orchestrates, Crucible renders,
localhost is a short network hop. `render/worker.py` does not move — it **goes away**, and the
question of which repo owns it stops mattering.

Stated as a direction, deliberately not scheduled. It is a larger call than any of the four
moves and should be ruled on its own.

---

## 6. Order, and why this order

1. **Extract narrator whole** — unblocks `crucible install` everywhere. One URL, one pin bump,
   one four-line test.
2. **Model facts move into it** — dissolves the R1 duplication instead of relocating it, and
   closes the reversal explicitly.
3. **Assembly to BookForge** — the seam is already cut and already tested.
4. **`text/` splits four ways** — mechanical once 3 has landed, because chunking travels with
   assembly. Only the caps fold needs a ruling.

Moves 1 and 2 are what make Crucible installable on a machine that is not Owen's; that is why
they are first. Moves 3 and 4 are cleanup of a package that will by then be somebody's clearly.
