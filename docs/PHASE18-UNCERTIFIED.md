# Phase 18 — uncertified voices: local weights, stated certificates, screening

**Status: a ruling, and ALL THREE STEPS ARE BUILT.** Sections 0-9 record Owen's rulings of
2026-09-18 and 2026-09-19 and the facts measured for them, so the build can be argued with.
Section 10 is the build order and says where each step landed: **(1) the source axis** and
**(2) measurements statable as unknown** are in **1.0.6**; **(3) the render door** is on
`feat/phase18-retake-flag`, unmerged, and is the only part of this document that describes a
server you cannot yet `git checkout main` and run.

**Sections 4, 5, 6 and 9 were rewritten on 2026-09-19 and the design they describe is not
the one they described before.** The superseded design was a *three-value certificate* per
key — `max_chars: "voice" | <int> | null`, `sampling: "voice" | {...}`, `guard: true|false`,
with omission refused — and it is gone. What replaced it is smaller and says the same thing
with fewer moving parts: **one flag and one band.** A reader who finds the certificate
vocabulary quoted anywhere else in this repo has found something stale; the names that exist
are `retake`, `band` and `width`.

The header is kept current rather than left saying "nothing is implemented", because a
document read after the work is done must not describe a server that no longer exists.

---

## 0. The ruling in one paragraph

A Crucible voice may name its weights by **path** as well as by HuggingFace pin, and the
certificates a voice carries — its cap, its pace band, its serving width — become facts a
manifest may decline to state and a render is no longer bound by. What replaces them is not
a default: a render STATES the two decisions it is making (`retake`, and the `band` a guard
would measure against) and the server acts on nothing it was not told. A voice that has not
earned its numbers can then exist, be served, and be measured, which is the thing that
produces the numbers.

*(Rewritten 2026-09-19. The first form of that middle sentence was a three-value certificate
per key — `"voice"`, a value, or `null`, with omission refused — and it is superseded. See
the header.)*

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

## 4. Ruling — the request states the decision; the manifest may decline to state a fact

Owen, 2026-09-18: *"all the guards and protections and expectations should be optional
configurations that are sent in."* And 2026-09-19, on the other half: *"I don't think it's
crucible's place to refuse chunks outside the band... especially if we add a different tts
engine."*

**REWRITTEN 2026-09-19, and the first draft is worth one paragraph because the replacement
is best read against it.** The first form was three explicit values per certificate, none
omittable — `max_chars: "voice" | <int> | null`, `sampling: "voice" | {...}`, `guard:
true|false` — on the argument that a default is the server inventing a value while an absent
constraint is nobody having asked for one, so the caller must SAY which. That argument is
still right, and it is still the argument below. What was wrong was the size of the
machinery it bought: three keys, each with a three-way vocabulary, each refused when
omitted, to express two decisions and one number.

### 4.0 What the request actually carries

    retake: bool          optional; absent is false
    band:   {pace_chars_per_sec, max_chars_per_sec, min_chars_per_sec}   optional
    width:  int >= 1      optional; absent is the voice's own max_num_seqs

**`retake` chooses the ARM.** `true` is narrator's guarded driver — the PaceTracker, the
re-roll on truncation/runaway/loop, the split ladder — measured against THAT band. `false`,
which is what absence means, is the bare arm: every chunk rendered once as sent, at the
requested take's sampling, nothing judged and nothing retaken. Measurement (`seconds`,
`chars`, `chars_per_sec`, `tokens`) comes back on every chunk on both arms, because those
are the server's own count of bytes that arrived. On the bare arm `guard` on the row is
`null` — nobody judged it, because nobody was asked to, which is the most exact that null
has ever been.

**Absence is a statement here, and that is the same rule the certificate design was built
on.** A DEFAULT is the server inventing a value for a fact that has a right answer. An
ABSENT CONSTRAINT is nobody having asked for one — and "nobody asked me to judge this
render" is a real, sayable state. A book that renders unguarded must have done so because
nobody asked for a guard, not because a field went missing in a refactor.

**`band` is the CALLER's and is never looked up.** All three positive, `min < pace < max`,
or the whole request is `band_malformed`. `retake: true` with no band is
`retake_without_band` — not filled in from the voice, and not silently downgraded to the
bare arm, because a client that asked to be guarded and was not would read every clean row
as a verdict. BookForge echoes back the row it read from `/v1/voices`; the screening ladder
sends nothing and therefore cannot be guarded, which is section 6.1's rule falling out of
the shape rather than needing its own refusal.

A band sent with `retake` false or absent is accepted, CHECKED, and not acted on. Owen:
*"it won't do anything with the number because it wasn't asked to."* Checked anyway, because
a malformed band is a client mistake whether or not this run would have used it.

**`width` is how many of this job's chunks are in flight**, and absent is the resident
voice's own `[voice.serving].max_num_seqs` — the width the ENGINE was started at, which is a
stated number with an owner rather than a default. Above it is `width_over_serving` and
never a clamp: a job that thought it was running 16 wide and was not would report a
throughput nobody can reproduce. Narrowing restarts nothing. Measured 2026-09-19: 0.60 mem
fraction at 16 wide summed to 24.2 GB on a 24 GB card and WDDM then pages to host RAM 4-10x
slower with no error; the ladder's baseline is 4, on voices whose manifests say 16.

### 4.0.1 What the RESULT carries back, and why

Two fields on the job's terminal event, added the same day (this is section 7's promise —
"the server asserting which weights were served and what sampling was applied" — discharged):

    sampling: {temperature, top_p, top_k}     THE FULL TRIPLE AS APPLIED
    voice:    {id, identity, identity_basis}

`sampling` is the voice's take-0 numbers with this take's rung laid over them, **never the
override alone** — a record saying only `temperature: 0.7` says nothing about the top-p and
top-k it ran at, and those are what the ladder's comparison rests on. It exists because
sampling lives on the MANIFEST and not on the request, which is the right shape (PHASE3-TTS
section 3) and leaves exactly one hole: a manifest edited between two runs makes two
incomparable records that both claim "take 0". Not hypothetical — every Higgs measurement
before 2026-09-06 was rendered at temperature 1.0 and the entire prior ladder record had to
be marked "at the wrong temperature" once already when the default moved.

`voice` is the `/v1/voices` row's own three words, so a ladder's record is self-describing
and nobody has to parse a fingerprint to learn whether an identity was a fetched sha
(`verified`) or a directory somebody pointed at (`asserted`).

**No per-request sampling override, explicitly.** The client asks for a take; the server
says what the take means. What the wire gained is the server SAYING what it did, not the
client telling it.

### 4.0.2 And the server stopped refusing a chunk by length

`chunk_too_long` is **retired**, on the render door and the streaming door alike. Not
relaxed — removed. Chunking and packing are the client's (PHASE3-TTS section 1), which never
changed, and the cap is still advertised on `/v1/voices` so a client can pack to it. What
changed is who ACTS on it. Two facts arrived together and made the refusal untenable: a
screening checkpoint has no measured cap to be refused against (section 4.2 below), and a
second TTS engine would have its own frame arithmetic that this number describes nothing
about. An oversize chunk now surfaces as whatever the engine does with it, reported honestly
on the `chunk` row — which is how a sweep finds out what the cap actually is. Crucible still
never re-splits: a server that quietly cut a chunk in half would return two files where one
was asked for.

### 4.1 The third relaxation — `[voice.pace]` is omissible, and must be absent rather than zeroed

Raised by the training session on 2026-09-18, against §6.1 as first written, and correct:
**the refusal in §6.1 could not fire under the schema of that day.** (§6.1 no longer HAS a
refusal of its own — `retake_without_band` enforces the same rule from the request side
since 2026-09-19 — but the argument below is what retired the mandatory triple, so it is
kept as it was made.) Verified in `crucible/voices.py`:

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

**One correction it forced on this document.** Absence does not mean "no guard" mechanically:
`voice_entry` omits the three keys, and narrator's `truncation.tracker_for` then uses its
engine's default band. So an uncertified voice was still guardable, which is why §6.1's rule
had to be re-argued from what that fallback COSTS rather than from a certificate being
missing — and, on 2026-09-19, moved off the voice entirely: with the band on the REQUEST
there is no path to the engine's default band unless somebody typed one.

**Both halves landed.** `fix/manifest-pace-not-invented` is merged and the three rates are
optional as a group in **1.0.6**. The `[voice.pace]` TABLE became omissible on
`feat/phase18-retake-flag`, which is what makes the rule above true of a manifest rather
than only of its contents — until then the way past the refusal was an empty table written
to satisfy a parser, which is a manifest saying something to a loader rather than about a
voice.

### 4.2 The manifest side — every certificate becomes statable as unknown

The pace triple, `max_chars`, `target_chars`/`safe_*_chars`, the `[voice.pace]` table itself
and the serving note are all optional now. (`target_chars` and the safe band always were —
`_PACE_OPTIONAL`.) Each is a RESULT, and requiring a result of a voice that exists to
produce it is the circularity section 1 is about.

**Absence propagates as absence, everywhere.** A `/v1/voices` row reports `max_chars: null`
and `pace: null` meaning NOT MEASURED — never the other arm's number, never a sibling's,
never the engine's — `narratorvoices.voice_entry` omits `maxChars` rather than inventing
one, and `narrator` no longer refuses a checkpoint voice that carries none. `max_chars_basis`
still governs a cap that IS stated, and a basis with no cap is refused as the leftover it is.

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

**AND THE SECOND HALF, ruled 2026-09-19: a take PAST the declared ladder is legal too.**
`unknown_take` is retired with it. A screening voice declares no `[[voice.takes]]` at all
and the client names takes 0..N, so a take must be nameable without a declared rung — and
`VoiceManifest.take` now answers one.

*What a take past the end resolves to was the one question here with two plausible answers,
and the answer is the voice's OWN sampling — take 0's, i.e. no override — not the last
rung's.* Two reasons, pointing the same way. The screening sweep needs takes 0..N to be
PURE SEED LANES at identical sampling (section 5.1), which is the matched-cell property its
Wilson-bound comparison rests on; and BookForge's retake ladder climbs DECLARED rungs k+1
and stops where they stop, so past that it is asking for another draw, which is what it
gets. Resolving to the last rung's numbers would be a clamp under another name — take 4
rendered at take 2's temperature and reported as take 4 — which is exactly what the old
refusal existed to prevent, and preventing it is why the refusal could be retired at all.

A NEGATIVE take is still refused, by the manifest. It is not a lane; it is a bug in a
caller. No door can reach it: both carry `Field(ge=0)`.

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

## 6. Ruling — the guard is the caller's to ask for, and the arm already exists

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
None))` — so until 2026-09-19 the arm was chosen by what the engine offered and the caller
had no say. The change is that the REQUEST chooses: `retake: false` takes the `elif` branch
that ran for everything before the 2026-09-13 ruling, and `retake: true` takes the first.

**The polarity is the correction this section needed.** It was written as `guard: false` —
an opt-OUT of a guarded default — and that is backwards for a server whose charter is to run
models and return bytes. The bare arm IS the primitive; the guard is a narration behaviour a
client asks for. A server that guarded by default would be deciding, on behalf of a run that
is measuring a failure curve, to hide the failures.

With the guard off, `attempt` is always 0, so a screening draw is
`base + index + TAKE_SEED_STRIDE * take` exactly — deterministic, reproducible, and asserted
by the server rather than echoed back by the client, which is strictly better than what the
ladder gets from sglang today.

### 6.1 An uncertified voice cannot be guarded — and it is the SHAPE that enforces it

Pressed by the training session on 2026-09-18, and they were right that `guard: false` for
screening is a **correctness condition** rather than a convenience. Both of their arguments
stand, unchanged:

**A guarded screen under-counts the thing it measures, invisibly.** The ladder's entire
output is a failure count. If narrator re-rolls a take that failed its own truncation or pace
check, the ladder scores the survivor and never sees the failure — and a successful re-roll
looks exactly like a good first draw. The numbers would be systematically clean in precisely
the way the instrument exists to detect, and would not be comparable to any band in the
record, all of which were measured on a path with no such guard.

**And the guard invalidates section 5's arithmetic.** With the guard live, `attempt` can
exceed 0, so the draw stops being a function of `(index, take)` and the reproducibility
argument lapses. A server cannot assert a draw identity it did not compute.

**What CHANGED on 2026-09-19 is how the rule is enforced, and the new way is better.** This
section used to call for a refusal of its own — `guard_without_certificate`, fired when
`guard: true` met a voice with no pace triple. That refusal is gone and nothing is lost,
because the band moved onto the REQUEST: `retake: true` with no band is refused by name
(`retake_without_band`) whatever the voice says, and a screening client has no band to state
precisely because nothing has been measured on those weights yet. The condition enforces
itself, the server never looks a certificate up, and there is no second owner of "what band
is this voice guarded against".

That also disposes of the trap the old rule was fighting. narrator's `truncation.tracker_for`
falls back to *the engine's default band, centred on the geometric mean of its edges*, so an
unmeasured voice guarded through the old path was guarded against a number nobody measured
for those weights: centred at 15.0 against a real book pace nearer 17.2, the band's ratios
give 1.333 tolerance short and 1.034 long, healthy chunks fall under `median x 0.967`, are
judged run-ons, and go re-roll -> split -> re-roll to `MAX_DEPTH`. Under a screen those
re-rolls are recorded as the CHECKPOINT's failures, which inverts the measurement in the
direction that makes a good checkpoint look bad. With the band on the request there is no
path to that band at all unless somebody typed it.

**The lifecycle still holds and is still the honest description.** A voice is uncertified
exactly while its numbers do not exist; it leaves that class the moment they are measured and
written into its manifest; the guard becomes askable-for meaningfully at that point because
there is finally a band to state. Nothing has to be toggled.

The converse also holds and is stated so nobody builds it by accident: when the guard IS
live, Crucible must report the draw as **unreported** rather than computing one. That is
`capped`'s discipline — `null` means "narrator did not say", never "false" — applied one
level up.

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
(PHASE3-TTS.md §1) and stays the client's. What DID change (section 4.0.2) is that the
server stopped refusing an oversize chunk at all: `chunk_too_long` is retired, and the text
goes to the engine as sent whether or not a cap was ever measured. "Never re-split" is the
half that survives and is the half that mattered — a server that quietly cut a chunk in half
would return two files where one was asked for.

**Resume.** Stays the client's. The ladder already keys its record by cell with an `ok` flag,
so it re-submits the missing cells. A server-side resume would be a second owner of a fact
the client holds, and `index` is documented as the client's and never renumbered — which is
what makes client-side resume correct.

**Concurrency — AMENDED 2026-09-19; the manifest's number is now a CEILING, not the whole
answer.** `[voice.serving].max_num_seqs` is still stage 0's admission width,
`--tts_engine.factory.max_running_requests` on SGLang, and the width of narrator's own batch,
and it is still what the engine is STARTED with. What it stopped being is the only say in how
wide a JOB runs: `params.width` narrows a job under it, `width_over_serving` refuses one
above it, and nothing is restarted either way.

The reason is a measurement rather than a preference. 0.60 mem fraction at 16 in flight
summed to 24.2 GB on a 24 GB card and WDDM paged to host RAM 4-10x slower with no error at
all. A screening job needs 4 on voices whose manifests say 16, and a manifest that said 4
would size the SERVER at 4 for every other client of that voice.

**And two more serving fields, same day, same shape** (`<field>` plus a required
`<field>_note`, both optional, both on `/v1/voices`): `mem_fraction` and `context_length`.
They are section 11's first open question, decided — see it.

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

**BUILT 2026-09-19, and these ARE new codes because they are refusals of a REQUEST rather
than of a manifest:**

| name | when |
|---|---|
| `retake_without_band` | `retake: true` and the request states no `band`. Never filled in from the voice and never downgraded to the bare arm. |
| `band_malformed` | a `band` that is not one — a missing rate, a value that is not a number, a rate at or below zero, or an order other than `min < pace < max`. ONE code for all of them, because a band is one statement, and it refuses the WHOLE request because there is no row a band belongs to. |
| `width_over_serving` | `params.width` above the voice's `[voice.serving].max_num_seqs`. Both numbers in the detail. Never clamped. |

**The two names the first draft of this section invented are GONE**, and neither was built:
`certificate_unstated` (a render omitting a certificate key) went with the certificate
design itself, and `guard_without_certificate` is unnecessary because `retake_without_band`
already enforces it from the request side — see §6.1.

**Retired**, each with what happens instead:

| retired | what happens now |
|---|---|
| `chunk_too_long` (both doors) | the chunk is rendered as sent and reported honestly. §4.0.2. |
| `unknown_take` | a take at or past the end of the ladder is a seed lane at the voice's own sampling. Never a clamp. §5. |
| `_check_takes`' refusal of a numberless rung above 0 | allowed: a rung that changes nothing is a different DRAW since narrator grew the take seed on 2026-09-15. §5. |
| `_parse`'s *"missing the [voice.pace] table"* | an absent table means exactly what an empty one means — nothing was measured. A PRESENT table is still checked exactly as before, ordering, positivity and symmetry included. §4.1. |
| `max_chars` as `_BACKEND_REQUIRED` | a backend block may state no cap; the row says `null`, meaning not measured, and `voice_entry` omits `maxChars` rather than inventing one. §4.2. |

Unchanged and still load-bearing: `sampling_malformed`, `take_malformed`,
`sampling_not_wired`, `voice_in_use`, `server_busy`, `voice_kind_unsupported`.

New manifest refusals, under `voice_invalid` with the rest (§9's own rule — a schema problem
does not get a code of its own here):

| code | message names | when |
|---|---|---|
| `voice_invalid` | *"states max_chars_basis ... and no max_chars"* | a repo manifest's arm carries a basis for a cap that is not there |
| `voice_invalid` | *"mem_fraction carries no note"* / *"context_length carries no note"* | a serving lever stated without the measurement that chose it |
| `voice_invalid` | *"states mem_fraction_note and no mem_fraction"* | the leftover of a number somebody deleted |
| `voice_invalid` | *"mem_fraction must be a fraction in (0, 1)"* | narrator's launcher refuses the same thing by name, and a refusal at the manifest beats a worker that exits 4 after a load |

## 10. Build order

1. **The source axis** (section 3) plus the basis on the `/v1/voices` row. Crucible only.
   Unblocked the deploy problem, which had been blocked since 2026-09-15. **IN 1.0.6.** What
   landed: the two source shapes and the refusals in §9; `source` and `identity_basis` on the
   `/v1/voices` row; `weights.installed` answering a local spec off the directory with no
   stamp; `pull` and `remove` refusing one by name; and a local voice kept OUT of
   `/v1/catalog`, because every row there offers a pull and a remove button and both would
   refuse. Tests in `test_voices.py` (the schema) and `test_voices_local_weights.py`
   (everything downstream), each seen to fail once with the check removed.
2. **Measurements statable as unknown** (section 4.2, manifest half). Crucible only.
   Precondition for a screening voice existing at all. **THE PACE HALF IS IN 1.0.6** — the
   three rates optional as a group, a partial triple refused by name, and the invented
   numbers stripped out of `higgs-default` and `zeroshot`. **The rest is on
   `feat/phase18-retake-flag`** with step 3, because the same files carry both: the
   `[voice.pace]` TABLE omissible, `max_chars` optional per backend end to end (manifest ->
   `/v1/voices` row -> narrator's voice document -> the resident record), and the repo
   schema's arm mirrored so a voice cannot be publishable and unloadable.
3. **The flag and the band on the render door** (section 4.0), the two take relaxations
   (section 5), the job result's `sampling` and `voice` (section 4.0.1), `params.width`
   (section 8), and `[voice.serving]`'s `mem_fraction` and `context_length` (section 11).
   Crucible plus one narrator change. **WRITTEN on `feat/phase18-retake-flag`, unmerged.**

   The narrator half is a BookForge branch and the tts env's pin has NOT moved
   (`crucible/envs/tts/*.txt`), so this branch is not runnable end to end until it does. The
   batch envelope Crucible sends is `{"action": "generate_batch", "language", "retake",
   "band"?, "width"?, "items": [...]}`, with `band` in narrator's camelCase
   (`paceCharsPerSec`, `maxCharsPerSec`, `minCharsPerSec`); narrator refuses the batch by the
   same two names Crucible refuses the request by, and Crucible refuses first so a
   well-formed request never reaches them. Retired rows gain `"capped": bool`, which this
   server already reads correctly and needs no change for.

(1) and (2) were the urgent half and they were small. (3) is an improvement to something that
already works — the ladder renders fine on its own path today — so it earned its way in on
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

**Still not done, and still owed.** Step (3) is written, but the key format has not changed
in the training repo and the screen is still parsing `_s` out of it. It lands with step (3)
being adopted there, or the substitution is not finished.

## 11. Open, and not decided here

**Memory fraction — DECIDED 2026-09-19, and it is a field.** This entry read "the cheapest
resolution is no new field at all", on the reasoning that the ladder scores on CPU and could
share the card at Crucible's unset default. The measurements that changed it: the default
`serve_higgs_sgl.sh` applies is **0.60** (line 59), the fraction is preallocated as KV on top
of ~7.7 GiB of weights *whatever the width is*, 0.55 measured 24.0-24.1 GB on a 24 GB card,
Crucible's own sigma serve sat at 23,561 MiB of 24,564 at the unset default the same day, and
WDDM pages to host RAM 4-10x slower with **no error at all**. 0.48 with width 4 is about
20 GB. Narrowing the batch does not lower that floor, which is exactly why it could not be
folded into `width`.

So `[voice.serving]` gains **`mem_fraction`** (`HIGGS_SGL_MEM_FRACTION`), and beside it
**`context_length`** (`HIGGS_CONTEXT_LENGTH`), which is a different wall for the same kind of
reason: SGLang-Omni's `HiggsTtsEngineBuilder.context_length` is a class attribute of 4096
(narrator records it at `engine/higgs/sgl_served.py:217-221`), 4096 tokens holds roughly
2,000 characters of prompt plus its frames, and the ladder's Third Reich bank tops out at
2,008 — so at the default the longest rungs truncate because the CONTEXT ran out and the run
records it as the VOICE's length wall. Plausible, wrong, silent. A screening voice states
8192.

Both are OPTIONAL and both owe a `<field>_note` when stated, as `max_num_seqs` does; absent
means the launcher's own number, which is a value in a file with an owner. Both go to BOTH
backends — Owen, 2026-09-19: *"we're going to want to configure darwin to work the same way.
context limits and such."* — and a knob narrator's MLX backend does not have is narrator's to
refuse by name at load, never Crucible's to drop.

**One thing owed on narrator's side, and it is stated rather than assumed:**
`HIGGS_CONTEXT_LENGTH` has **no reader** at bookforge HEAD. `sgl_served.py` says in as many
words that the 4096 "cannot be raised from here, from the launcher, or from a request", and
no such variable exists in that tree. Crucible sets it because the name is the agreed channel
narrator is growing the reader under; until the pin moves, it is set and read by nothing.

**Engine reuse across merges.** Not designed for. At the ladder, serve start is 67 s against a
42–54 min render — ~2%. At the *screen* it dominates: 5 checkpoints × (30 s merge + 67 s serve
+ 16 renders) is ~23 min, mostly not rendering. Worth revisiting only there, and noting that
merging is inherent — the v3 talker declares no `SupportsLoRA` and neither vLLM nor SGLang
exposes `--enable-lora`, so an adapter must be folded into full weights first.

**Whether a screening render should be guarded-but-reporting.** Not considered here. The
ladder wants no guard; a future caller might want the verdict without the intervention. That
is a third state and nobody has asked for it.
