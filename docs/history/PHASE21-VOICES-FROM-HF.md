# Phase 21 — A voice's facts travel with its weights: Crucible ships no voices

**Owen, 2026-09-19 (relayed by the training-PC session, confirmed by Owen: "lets rule on the
details and get ready to start on it, or start on it now"):** Crucible should never SHIP voices.
It downloads them from Hugging Face, and the safe band and the rest of the voice's config come down
WITH the weights. Deploying a checkpoint is one straightforward step.

This document is the CONTRACT (ARCHITECTURE.md R1). Sections 2–6 are the Crucible build; section
7 is the training side's; section 8 is the migration, in an order that cannot regress a voice.
Rulings marked **RULED** were taken on 2026-09-19 from the facts in section 1 and Owen's words;
the two in section 9 are still his.

> **BUILD STATE, 2026-09-19 (branch `feat/phase21-voices-from-hf`).** Sections 2, 3, 4, 5, 6
> and 10 are BUILT; section 8 step 1 is done and **steps 2, 3 and 4 are not**, which is the
> whole of what makes this build safe to ship: the five packaged manifests still ship and
> still WIN on their ids, `crucible/voices/pins.toml` is packaged EMPTY, and a pin is used
> only where a packaged manifest is absent.
>
> What is here: `crucible/voicerepo.py` (the `crucible-voice.toml` schema, the pins, the
> fetch, and `merge()`), `crucible/voicecard.py` (`card` and `export`),
> `crucible/engines/higgs-v3/base.toml` (2.6, the recommended ruling), `[tts.<engine>]` in
> `config.toml` written by `crucible init` through the single function
> `config.declared_tts_footprints` (2.3, the recommended ruling — turning it off is deleting
> one argument), the `pin` body on `PUT /v1/voices/{id}`, `crucible voices pin|check|card|export`,
> and `manifest` / `pace_basis` / `max_chars_basis` on the `/v1/voices` row.
>
> **Deviations from this contract, each with its reason:**
> 1. The row's `manifest` has a FOURTH word, `packaged`, for the five `crucible/voices/*.toml`
>    fine-tunes this build still ships. The contract's three (`repo`, `override`, `engine`)
>    are the three that survive step 8.3; `packaged` exists only while 8.1 is true and is
>    deleted with those files. A silence there would have made a packaged voice
>    indistinguishable from a repo one on the wire.
> 2. `voice_manifest_unreadable` is a refusal this contract does not name, beside
>    `voice_manifest_missing`. "This revision carries no manifest" is permanent and sends its
>    reader to commit one; "the Hub did not answer" is a minute. Reporting the first about the
>    second would send somebody to re-commit a file that is already there.
> 3. Section 3's *"`[voice.pace]` absent = uncertified with `guard: true` refused per §6.1"* is
>    built as far as the SCHEMA half: an absent pace parses, the voice loads with `pace` all
>    null and `pace_basis` null, and a test pins it. The `guard: true` REFUSAL is not built,
>    because this build has no `guard` field on the render door at all — PHASE18 §4's request
>    certificates (`max_chars`/`sampling`/`guard`) are unbuilt, and inventing the field here to
>    refuse it would be building §4 sideways. It is one small piece of work once §4 lands.
> 4. The `test_this_build_ships_…` group in `tests/test_voices.py` is KEPT rather than
>    replaced, and the fake-repo group is added beside it. Those eight tests are about the five
>    files this build still ships and still prefers; deleting them now would drop coverage of
>    the thing that currently owns the numbers. They go in step 8.3's commit, with the files.
> 5. The card's two new frontmatter keys are spelled `higgs_pace_basis`,
>    `higgs_max_chars_served_basis` and `higgs_max_chars_mlx_basis`, so each sits beside the
>    number it certifies and stays in the `higgs_*` family the audit scripts read. `max_chars`
>    is per-arm and could not be one key.
> 6. `crucible voices export` REFUSES to invent `basis` or `max_chars_basis` and asks for them
>    (`--pace-basis`, `--measured-from`, `--max-chars-basis`, `--uncertified`). The packaged
>    schema cannot state either, and a default would launder the two defects the fields exist
>    to expose. It also refuses a voice whose band would need `edges = "percentile"`, which
>    `Pace` does not carry across.
> 7. A `[voice.pace]` table that is PRESENT owes `basis` even when it states only a packing
>    target and no rates — the contract's "present ⇒ basis required", read literally. `basis`
>    certifies the table.
>
> **AMENDED 2026-09-19 (ruled after the training-PC session's argument, built here):** deviation 7's
> open question — whether `inherited` should owe prose — is closed the strict way. §2.1 now requires
> `inherited_from` on an inherited pace and refuses it on a measured one, and requires
> `measured_from` on a measured pace and refuses it on an inherited one: each basis owes exactly its
> own sentence, which is `estimate_basis`'s rule. `inherited_from` rides the `/v1/voices` row beside
> `pace_basis`, appears in the card's limits line, and `crucible voices export --inherited-from`
> refuses to invent it. Four refusals and the two carries, each seen to fail once against the code
> without the rule.

## 0. Why, in one paragraph

Deploying mistborn ckpt-4257 on 2026-09-19 took seven steps, and step seven was "the manifest
Crucible ships is STILL stale" (`crucible/voices/mistborn.toml`: pace 13.33, band 400–700,
revision e5bf8017 — every one a retired number). The card frontmatter on the HF repo already
carried the four numbers the manifest restates (`higgs_pace_chars_per_sec`,
`higgs_safe_min_chars`, `higgs_safe_max_chars`, `higgs_max_chars_served`), written by a second
step of the same deploy. A value and its description in two places drift, always — this is the
shape of deathstalker's 16.64 surviving onto weights that measured 15.91, and of thirdreich's
card carrying a `higgs_target_chars` ten days after that field was retired. The fix is a schema
fix: the manifest lives IN the repo, in the SAME commit as the weights it describes, and the local
side pins one thing.

## 1. What is true today (measured 2026-09-19; the section 2 design is built against this)

- Seven packaged manifests in `crucible/voices/*.toml`: five fine-tunes (`deathstalker`,
  `mistborn`, `owen`, `sigma`, `thirdreich`, all `owenmorgan/<id>-higgs-v3`) and two on Boson's
  base repo `bosonai/higgs-tts-3-4b` (`higgs-default`, kind `token`; `zeroshot`). Every file
  declares BOTH arms (`[voice.backends.cuda-linux]`, `[voice.backends.mlx-darwin]`) with the same
  repo and revision; **there are no per-arm files or branches on HF** — one set of bytes serves
  both arms.
- VOICE facts per file: `display kind narrator_engine language sample_rate`, `[voice.pace]`
  (triple + `safe_min/max_chars` or `target_chars`, optional `edges`), per arm `max_chars`,
  `sampling` (+`sampling_reason`), `clips`, and `[[voice.takes]]`.
- MACHINE facts per file: `memory_bytes_estimate` — **19_000_000_000 on every cuda arm,
  12_133_000_000 on every mlx arm**, `estimate_basis = "declared"` with a note naming the 3090 Ti's
  `--mem-fraction-static 0.60` reservation or the Mac Studio's MLX cap certificate; and
  `[voice.serving] max_num_seqs = 16` on every voice. Identical across all seven voices, which is
  the proof they are facts about a BOX and an ENGINE, not about a voice. `max_num_seqs` is already
  kept off the wire (`voices.py:239`, `jobs/tts/common.py:204`).
- The loader (`crucible/voices.py`) refuses everything as one `VoiceError` (PHASE18 §9); the
  `PUT /v1/voices/{id}` door surfaces it as `voice_invalid`. `[voice.pace]` must be PRESENT (empty
  allowed) — PHASE18 §4.1's "omissible in whole" is ruled and NOT built.
- `crucible voices pull` reads the manifest as the pin (no separate pin list), `snapshot_download`s
  the whole repo at the 40-char revision into `<home>/voices/<id>/<backend>/`, stamps
  `crucible-pull.json`; `installed()` reads the stamp's repo+revision against the manifest.
- `PUT /v1/voices/{id}` takes a whole manifest and writes the `<home>/voices` overlay, which wins
  on a shared id. Its ONLY caller anywhere is the training PC's campaign script `put_voice.py`
  — it is the promotion path today, which is what Owen said it should stop being.
- BookForge reads from a voice row only `id installed loadable reason takes`, plus cap and pace
  in `voice-band.ts`, all off `GET /v1/voices`. It never writes a voice.
- Nothing in Crucible reads or writes a model card. The card writers are campaign scripts on the
  training PC (`E:\training\_campaigns\2026-09-19-mistborn-deploy\hf_deploy_mistborn.py`,
  generated fresh per deploy from the 09-16 one), rewriting six frontmatter keys by regex and the
  NOTICE from a template.
- **The HF repos are not uniformly trustworthy today (training-PC audit, 2026-09-19):**
  `thirdreich` and `sigma` carry NO safe band in their cards; thirdreich's card says
  `higgs_max_chars_served: 1623` (a training-row length, not the served sweep), `higgs_max_chars_mlx:
  900` (a placeholder never measured) and `higgs_target_chars: 1000` (retired), and its repo has no
  LICENSE and no `merge_manifest.json`; `owen`'s pace 16.32 is INHERITED from the predecessor run,
  re-measurement owed; `thirdreich` (4 commits) and `sigma` (3) have revisions where the weights are
  newer than the card, because the deploy wrote them as separate commits before squashing.
  **The packaged manifests are the truth today; the cards are not.** Migration order follows.

## 2. The design — RULED

### 2.1 One file in the repo: `crucible-voice.toml`

At the ROOT of the HF repo, committed in the SAME commit as the weights it describes. Not the
card's frontmatter: a machine contract is not parsed out of a human README where an editorial
fix breaks a loader. The card is GENERATED from it (2.5), so there is one writer.

```toml
schema = 1                       # the manifest schema; a loader refuses a schema it does not read

[voice]
display         = "Mistborn"
kind            = "checkpoint"   # checkpoint | zeroshot | token
narrator_engine = "higgs-v3"
language        = "en"
sample_rate     = 24000

[voice.pace]                     # OMISSIBLE IN WHOLE: absent = an uncertified voice (PHASE18 4.1)
basis             = "measured"   # measured | inherited — REQUIRED when the table is present
pace_chars_per_sec = 13.76
max_chars_per_sec  = 17.89
min_chars_per_sec  = 10.58
safe_min_chars     = 500
safe_max_chars     = 800
# or target_chars = 600; or edges = "percentile" — exactly today's rules, unchanged
measured_from      = "mb_hp_rvcbed1 ckpt-4257, n=51 in the 500-800 band"   # prose, REQUIRED when basis = measured; refused when inherited
# inherited_from   = "ow_v8_rvcbed1 ckpt-966; these weights have no ladder yet, re-measurement owed"
#                                          prose, REQUIRED when basis = inherited; refused when measured

[voice.arms.cuda-linux]
max_chars       = 800
max_chars_basis = "measured"     # measured | placeholder — REQUIRED; a placeholder is served and SAID
sampling        = { temperature = 0.8, top_p = 0.95, top_k = 50 }
# sampling_reason, clips: exactly today's rules

[voice.arms.mlx-darwin]
max_chars       = 800
max_chars_basis = "measured"
sampling        = { temperature = 0.8, top_p = 0.95, top_k = 50 }

[[voice.takes]]                  # exactly today's ladder rules
[[voice.takes]]
temperature = 0.7
reason = "…"
```

What is NOT in it, by rule: `id` (a repo can be offered under any id; the pin names it),
`hf_repo`/`revision` (the file IS the revision), `memory_bytes_estimate`, `estimate_basis`,
`estimate_note`, `[voice.serving]`, any memory fraction. Those are section 2.3's.

`basis` on pace and `max_chars_basis` on each arm exist because of section 1: an inherited pace
and a placeholder cap are real states that shipped, and a schema that cannot say so ships them as
measured facts. Both are REFUSED when absent. `inherited` and `placeholder` are served exactly like
`measured` — they are certificates the voice states — and both ride the `/v1/voices` row
(`pace_basis`, `max_chars_basis`) so a person can see them.

**EACH BASIS OWES EXACTLY ITS OWN SENTENCE — RULED 2026-09-19**, after the training-PC session's
argument, and the shape is the loader's own `estimate_basis` rule: `declared` REQUIRES a note and
`measured` REFUSES one. So `basis = "measured"` requires `measured_from` and refuses
`inherited_from`; `basis = "inherited"` requires `inherited_from` — prose naming the run and
checkpoint the number came from and why it was not measured on these weights — and refuses
`measured_from`. Four refusals, each by name.

The reason the word alone is not enough: **an inherited pace from a sibling checkpoint of the same
corpus is near enough** — mistborn measured 13.29, 13.33 and 13.76 across three retrains — **and an
inherited pace from a different corpus two versions back is the deathstalker defect**, 16.64 carried
from `ds_v5_prod` onto weights that measured 15.91, 4.4% fast, enough to mis-size narrator's
duration guard from the first chunk. A reader deciding whether to trust a band has to be able to
tell those two apart, and only the sentence does it. `inherited_from` therefore rides the
`/v1/voices` row beside `pace_basis` (null when the pace is not inherited) and is printed in the
card's `## Measured limits` line for an inherited pace. `crucible voices export` gains
`--inherited-from` and refuses to invent it, exactly as it refuses to invent `basis`.

**The two arms stay per-arm tables in one file**, because there are no per-arm bytes on HF and the
numbers are not always equal (thirdreich 1623 vs 900 today). `arms` replaces the word `backends` in
the repo schema so the two schemas cannot be confused for each other; the internal model keeps
`backends`.

### 2.2 The local side pins ONE thing

`crucible/voices/pins.toml` (packaged: the repos this build may OFFER) plus `<home>/voices/pins.toml`
(the machine's own, wins per id), each:

```toml
[mistborn]
hf_repo  = "owenmorgan/mistborn-higgs-v3"
revision = "10b797e3e4f4a42fb356ff7c62f8365e9b24b1e6"
```

A sha still names one byte-state, so reproducibility is unchanged; what changes is that the
manifest at that sha cannot disagree with the weights at that sha. The packaged `voices/*.toml`
manifests for the five fine-tunes are DELETED at the end of section 8, and `voices_dir()` stops
refusing an empty directory.

**Loading a pinned voice** = the repo's `crucible-voice.toml` at that revision (fetched alone with
`hf_hub_download` when the weights are not pulled yet, so `GET /v1/voices` can list an uninstalled
voice with its real facts; read from the snapshot when they are) + the pin (id, repo, revision) +
the machine table (2.3) → the SAME internal `VoiceManifest` the engine reads today. One internal
model, two front doors; `jobs/tts/common.py`, `narratorvoices.py`, the guard and the takes ladder
do not change.

Refusals, by name, all `VoiceError` → `voice_invalid` as today: `voice_manifest_missing` (the
revision carries no `crucible-voice.toml`), `voice_manifest_schema` (an unknown `schema`), and every
existing schema refusal verbatim. A pinned repo whose manifest is missing is NOT served and is
NOT read from any other source — that refusal is what makes section 8's order safe.

### 2.3 Machine facts live with the machine

`config.toml` on each server gains, written by `crucible init` for its backend and rewritable by a
person or by a measurement:

```toml
[tts.higgs-v3]
memory_bytes_estimate = 19_000_000_000      # this box's serving footprint for this engine
estimate_basis        = "declared"          # declared | measured — today's pairing rules
estimate_note         = "3090 Ti, SGLang --mem-fraction-static 0.60 (the number every voice declared on 2026-09-19)"
max_num_seqs          = 16
max_num_seqs_note     = "…"
```

The values are the ones every packaged voice declares today (section 1), carried over by
`crucible init` per backend with the citation in the note; a Mac writes 12_133_000_000 and its
own note. `fits` and the row's `memory_bytes_estimate`/`estimate_basis` read from here. A server
whose config has no `[tts.<engine>]` table cannot serve that engine's voices and says so by name
(`engine_footprint_unset`); nothing is defaulted.

### 2.4 `PUT /v1/voices/{id}` goes back to being an override

Two bodies, told apart by shape:
- `{"pin": {"hf_repo": "…", "revision": "…" | null}}` — REPIN. Writes the home `pins.toml` row;
  `null` resolves the repo head as today (`resolve_revision`). This is what a deploy does per
  machine. Also the CLI: `crucible voices pin <id> <repo>@<sha>`.
- `{"voice": {…}}` — a full LOCAL manifest exactly as today (the PHASE18 `path`+`identity`
  directory arm lives here; so does a person overriding a voice on their own machine). The row
  says `manifest: "override"`.

The row gains `manifest: "repo" | "override" | "engine"` (2.6). `DELETE /v1/voices/{id}` removes
the home pin row or the override, whichever the id has, never weights, as today.

### 2.5 The card is rendered by Crucible, from the TOML

`crucible voices card <repo>@<sha> [--upload]`: reads `crucible-voice.toml` at that revision,
renders `README.md`'s frontmatter and its `## Measured limits` section from it, refuses RETIRED
keys by name (`higgs_target_chars` is the first; a table in the renderer, not a regex in a
campaign script), leaves every other line of the existing card byte-identical, and with
`--upload` commits it. The frontmatter keeps the six `higgs_*` names the audit scripts already
read, with `pace_basis` and per-arm `max_chars_basis` added. One renderer, in the repo with the
loader, tested as "what the loader reads is what the card says".

`crucible voices check <repo>@<sha>` (or a local file): parse and print the manifest exactly as
the loader would, or the refusal. The training side runs it before pushing.

### 2.6 The two base-model rows — RULED, with the alternative recorded

`higgs-default` (kind `token`) and `zeroshot` sit on `bosonai/higgs-tts-3-4b`, which is not ours
and cannot carry a `crucible-voice.toml`. They are not voices Owen trains; they are the ENGINE's
own base behaviour (the token default is narrator's own default on the mlx arm; zero-shot is
"clips from the request"). **They stay packaged, moved to `crucible/engines/higgs-v3/base.toml`
and reported with `manifest: "engine"`.** "Crucible ships no voices" is then exactly true of
voices. The alternative — a tiny repo of ours (`owenmorgan/higgs-base-voices`) carrying only a
manifest whose weights point at Boson's repo — makes the rule absolute at the cost of a second
HF repo and a `weights = {hf_repo, revision}` indirection every other manifest would then have
to allow-or-refuse. Section 9, ruling 1, if Owen wants absolute.

## 3. Tests (Crucible)

`tests/test_voices.py` keeps every schema test that is about the internal model; the eight tests
that pin "this build ships these manifests with these numbers" (the `test_this_build_ships_…`
group) are replaced by the same assertions against a FAKE repo fixture — a directory standing in
for an HF snapshot with a `crucible-voice.toml` — so the catalog numbers are still asserted, of the
thing that now owns them. New: the repo-schema parser (every refusal, `basis` and `max_chars_basis`
required, `[voice.pace]` absent = uncertified with `guard: true` refused per PHASE18 §6.1, unknown
`schema` refused); the pin list (packaged + home, home wins per id, a pin whose revision carries no
manifest is refused by name and NOT served); the merge (repo TOML + pin + machine table → the same
`VoiceManifest`, field by field, against a packaged manifest converted by the section 8 tool);
`PUT` both bodies; the row's `manifest` and `pace_basis`/`max_chars_basis`; the card renderer
(render → parse the frontmatter back → equals the TOML; a retired key refused; a foreign line
untouched); `engine_footprint_unset`. Every new test seen to fail once (the falsifiable rule).
`tests.sh --changed`.

## 4. The conversion tool — the bridge from today's files

`crucible voices export <id>` (packaged manifest → `crucible-voice.toml` + the machine rows it
drops, printed so nothing is lost silently). This is what section 8 runs against the five
fine-tunes so the HF manifests are written from the TRUTH (the packaged files), not from the
cards. It refuses to export a voice whose packaged pace is empty unless told it is uncertified.

## 5. What does NOT change

The engine's internal `VoiceManifest`, `narratorvoices.py`, the guard, the takes ladder, `fits`,
the render and stream doors, `crucible voices pull`'s snapshot + stamp, the PHASE18 path arm,
BookForge's reading of a row (it reads nothing this phase removes; it gains two optional fields
it may show). The pull still fetches the whole repo at the pin — the manifest rides in.

## 6. The wire, summarised

`GET /v1/voices` rows gain `manifest`, `pace_basis`, `max_chars_basis` (optional-by-vintage, never
by shape). `PUT /v1/voices/{id}` gains the `pin` body. Nothing else moves.

## 7. The training side (the training-PC session owns this; nothing here is Crucible's to build)

Writes `crucible-voice.toml` from the ladder's numbers; pushes weights + TOML in ONE commit
(`create_commit` with every operation, then squash); runs `crucible voices check` before and
`crucible voices card --upload` after; then `crucible voices pin` on each machine and `pull`. The
campaign scripts' FM_OWNED regexes and NOTICE template retire — the card is Crucible's; NOTICE stays
the deploy's (it is a licence notice, not a voice fact). Deploy is then: one commit, one pin per
machine.

## 8. Migration — in an order that cannot regress a voice

1. Crucible: sections 2–4 built, released. Loader accepts pins; the five packaged manifests STILL
   ship and still win (nothing has moved yet).
2. Training side, per fine-tune: **the DEPLOY writes `crucible-voice.toml`; the measurement
   database is the CROSS-CHECK, never the source.** Ruled this way after the training-PC session
   read `temper.sqlite` (`E:	raining\_campaigns\_reports	emper.sqlite`, 2026-09-19): its
   `ladders` table has NO pace column (`cert_json` is NULL on every row; mistborn's 13.76 was
   computed by hand from `scored.json`, a median over 57 clean in-band renders); its `band_min/max`
   is the LADDER'S computed band, not the RULED one (mistborn's row says 500–700, Owen ruled and
   shipped 500–800 the same day; deathstalker shipped 500–800 against the chart's own 400–1000 —
   prosody degrades before defects do, and `promote_voice --band` exists for exactly that); and
   thirdreich and sigma have NO rows in it at all. Two of the TOML's fields — the ruled band and
   WHICH checkpoint — are decisions, and decisions are made at deploy time by Owen.
   So: the deploy script writes the TOML from the ladder's numbers plus Owen's ruling; a
   disagreement with the database is a finding, not something to average. **The deploy also
   RECORDS the decision when it makes it**: `deploys` gains `ruled_band_min`, `ruled_band_max`,
   `pace`, `pace_basis`, written at deploy — from that change on, a deploy can be regenerated from
   the database; the ones before it cannot, and this sentence is where that is admitted.
   **thirdreich and sigma are transcribed ONCE, by hand, from the field notes** (thirdreich 500–800
   from 4n.50: 620 renders, n=32 per rung; sigma likewise) — a listed step, not a discovery. This is
   also where thirdreich's 1623/900/1000 are corrected and its LICENSE and `merge_manifest.json`
   added. `basis = "inherited"` is written only where nothing measured the SHIPPED weights (owen's
   16.32 today). `crucible voices export <id>` from the packaged manifest is the second cross-check.
   Commit the TOML + regenerated card as ONE commit on top of the current weights; squash.
   **The retrains are NOT on this phase's critical path.** deathstalker's mix is HELD pending
   Owen's green light (stopped 2026-09-19 12:03, lock released, partial mix deleted); owen's re-cut
   corpus is built and gated but its train has never started (`ow_v8_rvcbed1` trained and laddered
   2026-09-16, band 200–400, and is done, not in progress). When either does train, it goes through
   the new door as its first deploy — a nice first exercise, not a dependency.
3. Crucible: `pins.toml` gains the five ids at the new shas; the five packaged manifests are
   deleted in the SAME commit; `voices_dir()` stops requiring them. `test_voices` asserts the
   catalog against the pins' manifests (fetched into the fixture at build time from the test's own
   copies, never from the network in CI).
4. Each machine: `crucible voices pin` (or the app's future "update voices") + `pull`.
   `installed()` treats a stamp at the old revision as not installed, as today, so nothing serves
   old bytes under a new manifest.
A machine that skips step 4 keeps serving its old pin with its old manifest, consistently — the
property the whole phase buys.

## 9. Rulings still Owen's

1. **Base rows** — packaged as the engine's (2.6, recommended) or a manifest-only repo of ours.
2. **Where the machine footprint is first written** — `crucible init` copying today's declared
   numbers per backend (2.3, recommended), or left unset until `crucible capability` measures one
   (nothing serves until then).

## 10. Build

One Opus agent, branch `feat/phase21-voices-from-hf`, worktree, Crucible only, sections 2–4 and
the tests in 3, fakes only (no Hub calls in tests; `hf_hub_download` behind the same seam
`snapshot_download` already sits behind). No release cut; Owen tests, the training side runs
section 8 step 2, then 3 and 4. GPU untouched.
