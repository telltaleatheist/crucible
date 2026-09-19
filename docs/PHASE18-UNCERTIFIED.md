# Phase 18 — uncertified voices: local weights, stated certificates, screening

**Status: a ruling, and one step of it is built.** Sections 0–9 record Owen's rulings of
2026-09-18 and the facts measured for them, so the build can be argued with before it is
written. Section 10 is the build order and says which of its three steps has landed:
**step (1), the source axis, is built** — on `feat/phase18-source-axis`, stacked on the
pace branch of §4.1.1. Steps (2) and (3) are still a ruling only, and every "still to
build" below means them. The header is kept current rather than left saying "nothing is
implemented", because a document read after the work is done must not describe a server
that no longer exists.

---

## 0. The ruling in one paragraph

A Crucible voice may name its weights by **path** as well as by HuggingFace pin, and the
certificates a voice carries — its cap, its pace band, its serving width — become facts a
manifest may decline to state and a render may decline to be bound by. What replaces them
is not a default: every certificate is a decision the caller states in one of three
explicit forms, `"voice"`, a value, or `null`. A voice that has not earned its numbers can
then exist, be served, and be measured, which is the thing that produces the numbers.

---

## 1. The defect this fixes

**A manifest demands the measurements that screening exists to produce.**

`crucible/voices.py` requires of every voice: the pace triple
(`pace_chars_per_sec`, `max_chars_per_sec`, `min_chars_per_sec` — narrator's `_length_band`
"takes them as a triple or not at all", per `crucible/narratorvoices.py`), a per-backend
`max_chars` ("the one field narrator refuses a checkpoint voice without", same file), a
`max_num_seqs` with a written note, and a `revision` that is a full 40-character commit sha
"so a pull is reproducible; branch names are not pins" (`voices.py:893`).

Every one of those is a *result*. A fine-tune that is being screened has none of them, and
not having them is the reason it is on the card. The schema was written for voices that had
already finished the process it now makes impossible to run.

That is a circularity rather than a preference, which is why this is a defect and not a
convenience request. It is also why the fix is not "relax the schema": it is to say which
facts are the server's, which are the engine's, and which are the caller's, and to let the
caller state the ones that are its own.

## 2. Two customers, one missing capability

**The deploy problem (blocked since 2026-09-15).** HuggingFace private storage is full —
74.6 GiB of current trees over 17 private repos, ~89 GiB billed with history
(`orpheus-finetune/docs/HIGGS_FIELD_NOTES.md` §4n.74, open item (a)). Nothing can be pushed,
so no new revision can be pinned, so a newly trained voice cannot enter Crucible at all.
`deathstalker` and `mistborn` were deployed on 2026-09-15 through the **legacy BookForge
path** for exactly this reason, and §4n.74 records it as Owen's to rule.

**The ladder (works today, on its own path).** The fine-tuning ladder screens 5–31 merged
checkpoints per run. Reported by the training session, measured on the `mb_ha_rvcbed1`
ladder of 2026-09-15 overnight:

| | |
|---|---|
| merge (`merge_for_serving.py`, CPU) | 29–31 s |
| merged checkpoint on disk | 8.0 GiB, at `/home/telltale/higgs_v3_merged/<run>_<step>` |
| serve start → first render | 67 s, both models |
| bank | 128 prompts × 4 seeds = **512 renders per model**, 88–1529 chars |
| render 512 at concurrency 4 | 2530 s and 3232 s |
| lifetime of a merge | deleted the moment its renders land |

The merges are **already HF-repo-shaped directories** — `model.safetensors`, index, config,
tokenizer, the same layout `snapshot_download` would have written — and they live inside
WSL, on the same filesystem as the cuda-linux card. So naming one needs no new format.

They are also scratch, emphatically. Owen, 2026-09-18: *"you should be able to fuse the
model, use it, and then delete the fuse. keeping the adapter. fused models are like 9 gb
each, and we might have 30 checkpoints. so we'll have to plug and play the fused models."*
The 30 figure is an incident, not a hypothetical: on 2026-09-14 the screen loop cleaned up
only *partial* merges, 31 completed ones survived at 8.49 GB each, C: fell to 4.7 GB free
and the screen died mid-merge.

**These are one missing capability with two lifetimes**, and that is the cut this document
makes. Not "a durable door and a scratch door" — the same door, registered and left in one
case, registered and deleted four minutes later in the other.

## 3. Ruling — weights have a source, and management is its consequence

A `[voice.backends.<kind>]` block declares **exactly one** source:

    hf_repo + revision      a pin. Crucible fetches it.
    path                    a directory. Somebody else put it there.

Neither is optional in the sense of "omit and the server decides"; a block with both, or
with neither, is refused by name.

**Whether Crucible manages the bytes is not a second axis.** It follows from the source:

| | pinned | path |
|---|---|---|
| fetched by Crucible | yes | no |
| stamped (`weights.py`) | yes | **never** |
| a `/v1/catalog` subject | yes | **never** |
| deleted by `DELETE /v1/catalog/voice/{id}` | yes | **never** |
| may vanish between jobs | no | **yes, and that is not an error** |

The last row is the contract the ladder needs and the one a durable local voice pays
nothing for: a path that is gone at load time is `weights_path_absent` at the moment of use,
not a broken install. Crucible does not own the directory's lifetime and must not act as if
it does.

Downstream this is smaller than it sounds. `residency` already hands narrator a
`weights_dir` (`residency.py:1179`), so a path source is "skip the pull and the stamp, this
*is* the weights dir".

### 3.1 Provenance for a path

A pin's `revision` is **verified** — the sha is what was fetched. A path has no such
guarantee, and Crucible must not pretend otherwise or invent a substitute.

The answer follows the precedent already in `voices.py`, where `estimate_basis` is
`"measured"` or `"declared"` and the basis rides on the `/v1/voices` row "so nothing
downstream can mistake one for the other". A path block carries an identity **asserted by
whoever registered it** — for the ladder, the source adapter and step — and the row says it
is asserted. `fingerprint()` keeps its shape (`<id>@<identity>`) and gains a basis beside it.

**Crucible does not read `merge_manifest.json`.** The fused directory carries one, and it is
the right source for the assertion, but reading it would make the server know what a LoRA
merge is. It never knows what an audiobook is either. The registering client writes the
identity into the manifest; Crucible records and reports it.

## 4. Ruling — a certificate is stated, never assumed

Owen, 2026-09-18: *"all the guards and protections and expectations should be optional
configurations that are sent in."*

The form that makes this safe is three explicit values per certificate, none omittable:

    max_chars:  "voice" | <int> | null
    sampling:   "voice" | {temperature, top_p, top_k}
    guard:      true | false
    take:       <int>                 (see section 5)

`"voice"` means *use the certificate this voice carries* — a stated choice, and the one
BookForge writes. `null` means *enforce nothing* — also a stated choice. **Omitting the key
is refused.**

This is the distinction that keeps the ruling inside the no-fallbacks rule. A **default** is
the server inventing a value for a fact that has a right answer. An **absent constraint** is
nobody having asked for one. "No cap" is a real state; a missing `max_chars` key is not.
A book that renders uncapped must have done so because somebody wrote `null`, not because a
field went missing in a refactor and nothing noticed.

And the manifest side follows: the pace triple, `max_chars`, `target_chars`/`safe_*_chars`
and the serving note become **statable as unknown**. (`target_chars` and the safe band are
already optional — `_PACE_OPTIONAL`. The triple and the cap are not, and both are required
by *narrator*, not by Crucible's own preference, which is why section 6 matters.)

### 4.1 The third relaxation — `[voice.pace]` is omissible, and must be absent rather than zeroed

Raised by the training session on 2026-09-18, against §6.1 as first written, and correct:
**the refusal in §6.1 could not fire under today's schema.** Verified in `crucible/voices.py`:

* `_PACE_REQUIRED` (:150) demands all three rates;
* `_parse` (:855) refuses a voice with no `[voice.pace]` table at all — *"missing the
  [voice.pace] table"*;
* `_check_pace` (:531) enforces `min < pace < max` and refuses any rate `<= 0`.

So every valid voice carries exactly the triple §6.1 treats as the certificate, "a voice that
declines to state them" is not expressible, and `guard: true` would always find a certificate
to point at. There is also no null-ish escape: zeros and placeholders are refused, and a
correctly-ordered triple is mandatory.

Which leaves exactly one source for the numbers — **the predecessor's** — and that is a trap
with a case history. deathstalker's promotion of 2026-09-15 inherited `pace` 16.64 from
`ds_v5_prod` while the weights actually shipping measured 15.91: 4.4% fast, enough to mis-size
narrator's duration guard from the first chunk, and gap (3) of the four things `promote_voice`
cannot finish (`HIGGS_FIELD_NOTES.md` §4n.74). Requiring a triple of an uncertified voice is a
standing invitation to repeat that and then to guard against it.

**The ruling: `[voice.pace]` is omissible in whole.** Absent, never zeroed, never partial — a
table that is present is checked exactly as it is today, ordering included. The reason it is
right rather than convenient: *a screening checkpoint's pace is unknown by definition, and
measuring it is one of the run's outputs.* Pace is the median chars/s over a run's clean
renders — mistborn ckpt-5368 measured 13.90 overall / 13.33 in-band over 239 clean renders,
deathstalker ckpt-3658 15.91 over 222 — and that measured number is what the certified voice
later declares. A screening voice that declared a pace would be asserting the answer to the
question its own render exists to ask.

**Absence propagates as absence.** A `/v1/voices` row for an uncertified voice reports `pace`
as `null` meaning **not measured** — never a default, never an inherited value. This is
`capped`'s rule again, and it is load-bearing for the same reason: an inherited pace is
indistinguishable from a measured one at the point of use, which is exactly how 16.64 survived
a promotion.

#### 4.1.1 This is already built, on an unmerged branch, found from the other end

`fix/manifest-pace-not-invented` (da68edc, 2026-09-18, worktree
`crucible-worktrees/fix-unmeasured-pace`) — *"A voice manifest states the pace it measured, or
none at all"*. It makes the three rates **optional as a group**, refuses a partial triple by
name, strips the invented numbers out of `higgs-default` and `zeroshot`, and carries tests in
`test_voices.py` and `test_narrator_voices.py`. It is not merged.

It was written for a different reason than this document's and arrived at the same rule, which
is the strongest evidence either has. Its reason: `higgs-default` and `zeroshot` satisfied the
mandatory triple by copying narrator's own Higgs v3 defaults back to it (pace 15.0, max 20.0,
min 14.5), and 15.0 is not a narration rate at all — it is `cap_frames()`'s divisor. Since
narrator keeps a band's RATIOS and re-centres them on the book's running median, a band centred
there re-rolled healthy chunks to `MAX_DEPTH`. **The same weights were guarded differently
depending on whether BookForge or Crucible described the voice** — this repo's own recurring
shape, a fact with two owners.

**One correction it forces on this document.** Absence does not mean "no guard": `voice_entry`
omits the three keys, and narrator's `truncation.tracker_for` then uses its engine's default
band. So an uncertified voice is still guardable in the mechanical sense, which is why §6.1's
refusal had to be re-argued from what that fallback costs rather than from the certificate
being missing.

So step (2) of §10 is partly done and needs a review and a decision, not a build.

## 5. Ruling — seeds. The ladder needs no new wire field

This reverses what the design believed on the morning of 2026-09-18, and the reversal is
measured rather than argued.

narrator's seed for any row is (`narrator/engine/higgs/truncation.py`, lines 179–208):

    seed = base + index + REROLL_SEED_STRIDE * (TAKE_REROLL_LANES * take + attempt)

with `REROLL_SEED_STRIDE = 100_003`, `TAKE_REROLL_LANES = 16`, and the mapping
`(take, attempt, index) -> seed` **injective** by construction. `base` is `config.seed`,
default **1234** (`narrator/engine/higgs/config.py:146`), and **Crucible sets no seed
anywhere** — verified by grep across the package.

Therefore the draw is a pure function of `(index, take)` and **does not depend on the
weights**. Prompt *p* at take *k* is the same draw on checkpoint A as on checkpoint B, which
is exactly the matched-cell property the ladder's Wilson-bound comparison rests on. Four
takes are four matched, reproducible seed lanes across every model in a run.

So `seed` does **not** become a request field. What has to change is one Crucible refusal:
`voices.py:_check_takes` rejects a rung above 0 that declares no sampling override —
*"a rung that is the same sampling as the one below it is a different DRAW, which is what a
re-roll is for"*. That rule was written when a take could not move the seed. It can since
2026-09-15, and narrator's own module says so and hands the decision over verbatim
(`narrator/engine/item_sampling.py`):

> *"That rule was written when narrator had no take seed and a different draw was the one
> thing a rung could NOT ask for. This channel makes such a rung expressible; whether to
> allow it is Crucible's ruling, not narrator's."*

**The ruling: allow it.** A rung that declares no numbers is a different draw at the same
sampling, which is a legitimate thing to ask an engine for and is the ladder's entire unit
of work. The refusal's reasoning was true when written and is now false.

Two consequences worth stating. A screening voice declares no `[[voice.takes]]` at all and
the client names takes 0..N — so a take must be nameable **without a declared rung**, which
is the second half of the same relaxation. And the ladder's record, keyed `(id, seed)`
today, becomes keyed `(index, take)` — the same cell, named by what was actually requested
rather than by a number it had to compute.

### 5.1 Two contracts the substitution creates

Both raised by the training session on 2026-09-18, both load-bearing, neither optional.

**`index` is the client's, and stable across partial resubmits.** Because the draw is
`base + index + stride * take`, `index` stops being a position and becomes *identity*. The
ladder resumes by re-submitting only the missing cells — a run that died at 510 of 512
re-renders two, not 512. If `index` were assigned by position within the submitted batch,
those two would come back as 0 and 1 and be **different draws from the ones that failed**,
which is reproducibility failing at exactly the moment it is being relied on.

The mechanism is already right: `TtsParams` refuses duplicate indices because "an index is an
artifact name", `render.py:848` keys by `chunk.index` rather than by position, and
`render.py:882` puts `{"i": chunk.index}` on narrator's wire directly. What is new is that
this is now a **contract** rather than a convenience, and positional assignment anywhere in
the path — client, server or engine — would silently destroy it. The ladder sets `index` to
the prompt's ordinal in the bank file (0..127, fixed for the life of that bank), and a partial
resubmit reuses the original indices.

**A screening sweep's takes must be pure seed lanes — byte-identical sampling.** The
relaxation in section 5 permits a numberless rung; this says the screening client must *use*
only numberless rungs. If take 1 carried a declared rung-1 deviation (the five shipped
fine-tunes declare `temperature = 0.7`), a four-take sweep would measure four **sampling
points** rather than four draws, and would be comparable to nothing in the record: every band
ever measured — back to thirdreich's 500-800 — was measured at exactly 0.8 / 0.95 / 50, and
the entire prior ladder record had to be marked "at the wrong temperature" once already when
the default moved. A screening voice therefore declares no `[[voice.takes]]` at all, and takes
0..N differ in nothing but the seed lane.

## 6. Ruling — the guard is the caller's to decline, and the arm already exists

A `tts` render goes through narrator's `render_many`: the guarded driver, with the
PaceTracker, the re-roll and the split ladder inside it. `truncation.join_parts` rejoins a
split before the chunk retires, so a split shows up only as `parts: 2` inside the verdict.

For screening that is not a feature, it is **falsification**. The ladder is measuring the
failure curve the guard exists to hide: a cell that was silently retaken is a corrupted data
point, and a cell that was split is a cell that did not happen.

The unguarded arm is already built and is not a new code path.
`narrator/serve/worker.py:2268`:

    if rows and _guards_its_own_batch(self.orph):
        self._emit_guarded_batch(rows, emitted)
    elif rows:
        audios = self._generate_audio_batch(...)

`_guards_its_own_batch` is a **capability probe** — `callable(getattr(engine, 'render_many',
None))` — so today the arm is chosen by what the engine offers and the caller has no say.
The change is to let the request choose: `guard: false` takes the `elif` branch that ran for
everything before the 2026-09-13 ruling.

With the guard off, `attempt` is always 0, so a screening draw is
`base + index + TAKE_SEED_STRIDE * take` exactly — deterministic, reproducible, and asserted
by the server rather than echoed back by the client, which is strictly better than what the
ladder gets from sglang today.

### 6.1 On an uncertified voice the guard is not an option — it is a refusal

Pressed by the training session on 2026-09-18, and they are right. `guard: false` for
screening is a **correctness condition**, not a convenience, for two reasons that point the
same way.

**A guarded screen under-counts the thing it measures, invisibly.** The ladder's entire
output is a failure count. If narrator re-rolls a take that failed its own truncation or pace
check, the ladder scores the survivor and never sees the failure — and a successful re-roll
looks exactly like a good first draw. The numbers would be systematically clean in precisely
the way the instrument exists to detect, and would not be comparable to any band in the
record, all of which were measured on a path with no such guard.

**And the guard invalidates section 5's arithmetic.** With the guard live, `attempt` can
exceed 0, so the draw stops being a function of `(index, take)` and the reproducibility
argument lapses. A server cannot assert a draw identity it did not compute.

**The rule that follows is not an extra flag.** An **uncertified voice cannot be guarded**,
so `guard: true` on one is `guard_without_certificate` — refused by name, not silently
downgraded.

**But not for the reason this section first gave, and the correction matters.** It said a
voice with no pace "has nothing to guard against". That is false: narrator's
`truncation.tracker_for` falls back to *the engine's default band, centred on the geometric
mean of its edges*, so an unmeasured voice is guarded — against a number nobody measured for
those weights. The refusal therefore rests on what that fallback DOES, which is documented in
this repo by the branch that discovered it independently (§4.1): centred at 15.0 against a
real book pace nearer 17.2, the band's ratios give 1.333 tolerance short and 1.034 long,
healthy chunks fall under `median x 0.967`, are judged run-ons, and go
re-roll → split → re-roll to `MAX_DEPTH`.

Under a screen those re-rolls are **recorded as the checkpoint's failures**. A screening
render guarded against a generic band measures the band, not the model, and it does so in the
direction that makes a good checkpoint look bad. That is the inversion §6.1 exists to prevent,
and it is worse than the under-count in the first argument above rather than milder than it.

**And the certificate is not merely absent — it is UNKNOWABLE until this render has
happened.** The pace a guard enforces is the median chars/s
of the voice's own clean renders (§4.1), so on a checkpoint that has never been rendered there
is no number to enforce, and the render being refused is how the number comes to exist. That
gives the class a **lifecycle rather than a flag**: a voice is uncertified exactly while its
numbers do not exist, it leaves that class the moment they are measured and written into its
manifest, and the guard becomes available at that point because there is finally something to
guard against. Nothing has to be toggled, and no voice can sit in a state where a guard is
enforcing a figure nobody measured.

(This depends on §4.1's relaxation and does not work without it. As first written, §6.1's
refusal had no reachable case — the schema made a pace triple mandatory, so every voice
carried the certificate the refusal was checking for the absence of. Its second draft had a
reachable case and the wrong reason; this is the third.)

The converse also holds and is stated so nobody builds it by accident: when the guard IS live,
Crucible must report the draw as **unreported** rather than computing one. That is `capped`'s
discipline — `null` means "narrator did not say", never "false" — applied one level up.

## 7. Why this is one door and not two

The two-door alternative — a screening job that speaks sglang's `/v1/audio/speech` directly,
bypassing narrator — was rejected. Its cost is not the door. sglang under Crucible is started
*by narrator*, so a door that bypasses narrator needs Crucible to start sglang bare as well:
a second residency shape, a second env recipe path, a second provenance story. That is not a
second door, it is a second half of the server, and the moment screening wants what it says
it wants — the server asserting which weights were served and what sampling was applied —
that provenance gets built twice.

The reframing behind the ruling: Crucible's charter is *"it runs models and returns bytes,
and it never knows what an audiobook, a cleanup pass or a PDF conversion is"* (DESIGN.md).
The `tts` job today is not that — cap certificate, pace band, take ladder, retake vocabulary
are narration domain living in the server. The ladder is not an awkward second customer. It
is the first customer that asks Crucible to do what Crucible says it does, and the right
response is to **recover the primitive**, not to route around the door that grew a specialty.

## 8. What does not change

**The chunker.** There isn't one, and there must not be. Chunking is the client's
(PHASE3-TTS.md §1), which is why `chunk_too_long` is *"a refusal and not a re-split"* — a
server that quietly cut a chunk in half would return two files where one was asked for. With
`max_chars: null` there is simply nothing to refuse; the text goes to the engine as sent.

**Resume.** Stays the client's. The ladder already keys its record by cell with an `ok` flag,
so it re-submits the missing cells. A server-side resume would be a second owner of a fact
the client holds, and `index` is documented as the client's and never renumbered — which is
what makes client-side resume correct.

**Concurrency.** Already `[voice.serving].max_num_seqs`, which is stage 0's admission width,
`--tts_engine.factory.max_running_requests` on SGLang, and the width of narrator's own batch.
A screening voice states 4.

**The serving stack.** Crucible already runs the stack the ladder drives —
`serve_higgs_sgl.sh`, width from `HIGGS_MAX_NUM_SEQS`, `cuda_graph_max_bs` following it.

**One job per lane.** 512 chunks in one job is already legal (`chunks` has `min_length=1` and
no ceiling), and the ladder's CPU scoring of model N runs in its own process while the lane
renders N+1.

## 9. Refusal names

**Corrected once built.** This section first invented four API error codes. Three of them
were schema problems, and a schema problem does not get a code of its own in this server —
`crucible/voices.py` raises `VoiceError` and `PUT /v1/voices/{id}` surfaces every one of them
as **`voice_invalid`** with the message naming the problem. Adding `weights_source_ambiguous`
as a top-level code would have made this one schema the only one whose refusals are sorted
into codes, and a client would then have had two ways to learn the same thing.

So the new refusals are named **in the message**, under the codes that already exist:

| code | message names | when |
|---|---|---|
| `voice_invalid` | *"declares both hf_repo … and path … names ONE source"* | a block declares a pin and a path |
| `voice_invalid` | *"names no weights"* | it declares neither |
| `voice_invalid` | *"is a pinned block and also carries identity"* | a pin with an asserted identity beside its verified one |
| `voice_invalid` | *"declares path … and no identity"* | a directory that will not say what it holds |
| `voice_invalid` | *"path … is not absolute"* | a path the server would resolve against its own cwd |
| `voice_not_installed` | *"names … and there is no such directory on this server"* | a local voice's bytes are gone — and the message does **not** say to pull |
| — | *"cannot be pulled"* / *"cannot be removed"* | `weights.pull` / `weights.remove` on a local spec |

Still to build, and these two ARE new codes because they are refusals of a REQUEST rather
than of a manifest:

| name | when |
|---|---|
| `certificate_unstated` | a render omits a certificate key instead of stating one of its three values |
| `guard_without_certificate` | `guard: true` on a voice that states no pace triple and no cap (§6.1) |

Retired: `_check_takes`'s refusal of a numberless rung above 0 (section 5); `_parse`'s
*"missing the [voice.pace] table"* for an uncertified voice (section 4.1 — a **present** table
is still checked exactly as today, ordering and positivity included).

Unchanged and still load-bearing: `chunk_too_long` (when a cap was asked for), `unknown_take`,
`sampling_malformed`, `take_malformed`, `voice_in_use`, `server_busy`.

## 10. Build order

1. **The source axis** (section 3) plus the basis on the `/v1/voices` row. Crucible only.
   Unblocks the deploy problem, which is blocked *now* and has been since 2026-09-15.
   **WRITTEN** on `feat/phase18-source-axis`, stacked on the pace branch of §4.1.1 because
   both rewrite `voices.py`. What landed: the two source shapes and the refusals in §9;
   `source` and `identity_basis` on the `/v1/voices` row; `weights.installed` answering a
   local spec off the directory with no stamp; `pull` and `remove` refusing one by name; and
   a local voice kept OUT of `/v1/catalog`, because every row there offers a pull and a
   remove button and both would refuse. Tests in `test_voices.py` (the schema) and
   `test_voices_local_weights.py` (everything downstream), each seen to fail once with the
   check removed.
2. **Measurements statable as unknown** (section 4, manifest half). Crucible only, lands with
   (1). Precondition for a screening voice existing at all. **The pace half of this is already
   written** — `fix/manifest-pace-not-invented`, unmerged, §4.1.1 — so what it needs is Owen's
   test pass and a merge, not a build. `max_chars` and the serving note are still to do.
3. **Stated certificates on the render door** (section 4, request half), the take relaxation
   (section 5) and `guard: false` (section 6). Crucible plus one narrator change, which is a
   flag selecting an arm that already exists.

(1) and (2) are the urgent half and they are small. (3) is an improvement to something that
already works — the ladder renders fine on its own path today — so it earns its way in on
card-sharing and provenance, not on being blocked.

### 10.1 One thing step (3) breaks outside this repo, and it fails silently

Recorded here because this document is what will be read when the work is done, and the defect
is invisible from inside Crucible.

Re-keying a render from `{id}_s{seed}` to `(index, take)` breaks
`orpheus-finetune/pipeline/higgs_pause_screen.py:514,516`, which does not merely use the key
as a dict key — it **parses** it, with a hardcoded literal:

    by_prompt.setdefault(r["key"].rsplit("_s", 1)[0], []).append(r["dur"])

With no `_s` in the key, `rsplit` returns the whole string, every render becomes its own group
of one, `len(sib) > 1` is False, and **`overrun` scores 0 on every render, forever** — no
exception, no warning. `overrun` is one of the four classes the screen ranks checkpoints on,
and it is the class that caught the late-epoch failures on `mb_full_rvc1` (5.4 per 100 renders
against 0.0 for the bedded run, part of why the bed was adopted). A screen scoring 0 overruns
would have ranked those checkpoints clean.

The fix is small and belongs to the training repo: `renders.json` already carries `id` as its
own field, so the grouping should read `r["id"]` instead of parsing `r["key"]`, which means
threading `id` through the `files` tuples at `higgs_pause_screen.py:465`. The same lesson was
already learned once in `band.py`, which used to parse the rung out of the id with a hardcoded
`L(\d+)_` regex and returned empty for banks built with a different prefix.

**Not to be done now** — nothing here is built, the key format has not changed, and the screen
is running on a train that is mid-prep. It lands with step (3) or the substitution is not
finished.

## 11. Open, and not decided here

**Memory fraction.** Crucible leaves `HIGGS_SGL_MEM_FRACTION` unset, so `serve_higgs_sgl.sh`
defaults it to **0.60**; the ladder serves at **0.48** because it scores on the same card. The
cheapest resolution is no new field at all — the ladder scores on CPU (reported 1.46 s/render,
so ~12.5 min for 512, which hides completely inside the next model's 42–54 min render). If a
per-voice fraction is wanted instead it belongs in `[voice.serving]` beside `max_num_seqs`,
with the note that field already requires.

**Engine reuse across merges.** Not designed for. At the ladder, serve start is 67 s against a
42–54 min render — ~2%. At the *screen* it dominates: 5 checkpoints × (30 s merge + 67 s serve
+ 16 renders) is ~23 min, mostly not rendering. Worth revisiting only there, and noting that
merging is inherent — the v3 talker declares no `SupportsLoRA` and neither vLLM nor SGLang
exposes `--enable-lora`, so an adapter must be folded into full weights first.

**Whether a screening render should be guarded-but-reporting.** Not considered here. The
ladder wants no guard; a future caller might want the verdict without the intervention. That
is a third state and nobody has asked for it.
