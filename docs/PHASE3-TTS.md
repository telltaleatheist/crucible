# Phase 3b: the `tts` job type

Contract for the largest capability. Extends DESIGN.md; where the two disagree this file
wins for `tts`. Written 2026-09-13 from `docs/CLIENT-SURFACES.md` sections 3 and 10 (tier 3)
and Owen's two rulings — the division of knowledge (DESIGN.md section 3.1) and *voices come
from the server*.

Owen, 2026-09-12, on the second one, verbatim:

> we're going ot have to pass in the actual voice. maybe we retrieve available voices from
> crucible and we send in the order for deathstalker, the voice that was returned by the
> server. and the server will have to have a streaming or a rendering option for higgs. the
> bookforge browser extension uses streaming a lot. i use it every sunday

Those are the two doors, and the last sentence is why both are mandatory rather than one
being a later addition.

## 1. What moves, and what does not

**Moves to Crucible.** The engine process and its lifetime. The accelerator lease. Which
voice is resident. Sampling per (voice, backend). Cap certificates per (voice, backend). The
token-budget formula that turns characters into `max_new_tokens`. An engine's EOS levers. The
reference clips a zero-shot voice is conditioned on. The WSL spawn, the path rewriting and
the per-engine VRAM arithmetic — all of it deleted rather than ported.

**Stays in BookForge.** Chunking (`listen-chunks.ts`, the paragraph packer). Text
normalisation (`listen-text.ts`, `tts-punctuation.ts`, the number rules, the CAPS fold, the
glyph strip). The pace guard's *judgment* — which takes are good, which rows need a retake,
when a hole is a hole. The retake decision. Assembly. The session layout and the resume
rule. The ledger.

The line between those two paragraphs is the division of knowledge: **the server measures,
the client judges.** Crucible reports what a chunk actually did — its duration, its
characters per second, whether it hit the cap — and never decides what to do about it.

## 2. A voice is a manifest, and the server advertises it

`voices/<id>.toml` in this repo, one file per voice id, mirroring `models/<id>.toml`
exactly. The id is Crucible's and stable across backends; the weights and the numbers differ
per backend.

```toml
[voice]
id = "deathstalker"
display = "Deathstalker"
kind = "checkpoint"           # checkpoint | zeroshot | token
narrator_engine = "higgs-v3"  # which of narrator's engines serves it
language = "en"
sample_rate = 24000

# The band the chunk packer works to. Advertised so a client can pack to it; the
# client does the packing, the server states the shape. These are the numbers that
# live in BookForge's higgs-models.json voice document today.
[voice.pace]
pace_chars_per_sec = 16.64      # required
max_chars_per_sec = 21.63       # required
min_chars_per_sec = 12.80       # required; min < pace < max
safe_min_chars = 600            # this voice's packing shape: a band...
safe_max_chars = 800
# target_chars = 600            # ...or a single target. Never both; neither is
                                # also legal and means "pack to max_chars".

[voice.backends.cuda-linux]
hf_repo = "owenmorgan/deathstalker-higgs-v3"
revision = "<40 hex>"
memory_bytes_estimate = 19_000_000_000
estimate_basis = "declared"      # measured | declared
estimate_note = "SGLang's --mem-fraction-static 0.6 …"   # required when declared
max_chars = 800                  # THE cap certificate for (voice, cuda-linux)
sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }
# sampling_reason = "…"          # required when `sampling` is not the engine's

[voice.backends.mlx-darwin]
hf_repo = "owenmorgan/deathstalker-higgs-v3"
revision = "<40 hex>"
memory_bytes_estimate = 12_133_000_000
estimate_basis = "declared"
estimate_note = "deathstalker's MLX certificate, 11.3 GiB peak at 900 chars"
max_chars = 800
sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }
```

That schema is what `crucible/voices.py` loads, and it differs from this section's first
draft in four places. Each difference is there because the file as drafted could not be
filled in truthfully from BookForge's catalog; each is marked below.

Five things in that file are load-bearing.

**The cap is per backend and must stay per backend.** Every voice's two blocks carry
identical numbers today — 600/600, 800/800, 1000/1000, 1100/1100 (CLIENT-SURFACES.md section
3.1). That is a coincidence of the current catalog, not a property of the world, and
DESIGN.md section 3 already promises a cap certificate per (model, backend). One number
shared between two accelerators is how a cap measured on one card silently governs the
other.

**DIFFERENCE 1 — it is `max_chars`, and this document used to call it `cap_tokens`.**
Corrected 2026-09-13, while the loader was being written. Those four numbers are
**characters**: in BookForge they are `backends.<arm>.maxChars`, the length of text a voice
may be handed. The token cap is a different quantity and narrator derives it per chunk from
the text it is actually given — `HiggsBudget.cap_frames`, `int(len(text) / 15.0 * 25 * 1.8)
+ 100`, clamped against the stack's context window by `sgl_served.frame_cap`. Writing 800
into a field named `cap_tokens` would have handed the engine a frame ceiling roughly eight
times too small and cut every chunk mid-sentence while the request reported success. The
manifest carries the catalog's name and the catalog's meaning; the token budget stays where
it is computed. **Nothing in `tts` carries a token cap on the wire**, and the `chunk` event's
`capped` (section 6) is therefore about the frame cap narrator computed, not about this.

**DIFFERENCE 2 — the pace block is three required rates plus an OPTIONAL packing shape.**
As drafted it required all seven numbers, and no voice in the catalog declares all seven.
The three rates (`pace_chars_per_sec`, `max_chars_per_sec`, `min_chars_per_sec`) are
required of every voice and must satisfy `min < pace < max`, which is narrator's own rule in
`engine/higgs/config.py`'s `_length_band` — the band is the measured pace and the two edges
derived from it, and narrator keeps only the RATIOS and re-centres them on the book's own
running median. A voice with no measurement of its own carries the narrator engine's default
band, which is still a recorded number (Higgs v3: 15.0 / 20.0 / 14.5, read off
`HiggsDefaults` and `HiggsV3Defaults`). What the client packs to is then EITHER a
`safe_min_chars`/`safe_max_chars` band (what the five fine-tunes declare — their training
corpus's interquartile range, measured 2026-09-09) OR a single `target_chars` (what the
zero-shot voices declare), never both, and a voice declaring neither packs to the backend's
`max_chars`, which is what BookForge does today. The loader refuses a band whose ceiling
exceeds the arm's `max_chars`, the same rule BookForge and narrator both refuse on.

**DIFFERENCE 3 — `sample_rate` is required in `[voice]`.** The `/v1/voices` row carries it
and a client writing FLACs cannot be handed a null. It is 24000 for every voice in the
catalog, which is exactly the kind of coincidence that becomes a hard-coded constant if it is
not written down per voice.

**`sampling` is the engine-level default, and a deviation owes a reason.** Owen's rule, in
memory as `higgs-sampling-default-no-deviation`: 0.8 / 0.95 / 50 is *the boson default*, one
engine-level number, and a per-voice deviation requires a written reason. So a backend block
whose `sampling` differs from its engine's default must also carry `sampling_reason`, and
the loader refuses it if it does not. On SGLang, sampling is **mandatory per request** —
omitting `top_k` samples the untruncated 1026-way codebook tail, measured as one chunk
running to the cap with 80 s of silence — so there is no "leave it to the engine" option
here, and the manifest is where the numbers have to be.

**`memory_bytes_estimate` is measured or the block does not exist.** The same rule the model
manifests learned the hard way: a computed estimate came out 34% light on `mlx-darwin` and
6% light on `cuda-linux`. A voice that has not been measured on a backend carries no block
for that backend, and the server refuses to load it there by name rather than guessing.

**DIFFERENCE 4 — `estimate_basis`, because that rule made every voice unloadable.** Neither
of Owen's accelerators was free the night the manifests were written (section 10 says as
much), so under the rule above not one voice could carry a block and the whole job type would
have been untestable. So every block states where its number came from: `"measured"` — 
somebody watched the card — or `"declared"`, which means it came from the engine's own
configured reservation (Higgs v3 on SGLang runs at `--mem-fraction-static 0.6`, and the
catalog records that as "0.60 holds ~19 GB at 16 in flight") or from a sibling voice's
certificate. `"declared"` additionally requires `estimate_note`, and `"measured"` refuses
one — a note beside a measured number reads as an excuse for it. **The basis rides on the
`/v1/voices` row**, so nothing downstream can mistake one for the other, and every voice this
build ships says `declared`. The model manifests have the same problem and do **not** have
this field: `models/qwen3.5-9b.toml` carries the word MEASURED in a comment no protocol
reads. That asymmetry is deliberate for now — those numbers really were measured — but it is
why a reader finds provenance in two shapes.

**Zero-shot clips belong to the voice, not to the request.** A `kind = "zeroshot"` voice
carries its reference clips the way a checkpoint carries its weights:

```toml
[voice.backends.mlx-darwin]
hf_repo = "owenmorgan/higgs-zeroshot-refs"
revision = "<40 hex>"
clips = [
  { file = "stranger-01.wav", transcript = "...", seconds = 8.4 },
]
```

The transcript is the book-exact text the clip was cut from and is **never an ASR guess**
(`narrator/engine/protocol.py` refuses an empty one; the training-text doctrine is the same
law). There is one exception and it is named rather than general: a voice may declare
`clips = "from-request"`, which means this voice id exists precisely so an operator can hand
over a clip that is not published yet, and a job naming it must carry the clips in its
`inputs`. That is the only way audio ever crosses the wire in the render direction, and a
voice that does not declare it **refuses** clips in the request rather than ignoring them.

### `GET /v1/voices`

The rows are what `/v1/info`'s `tts` capability carries **verbatim**, produced by the same
function — the same rule, and for the same reason, as `llm`'s models (PHASE2-LLM.md section
5). One voice, one description; a client never reconciles two.

```json
{
  "id": "deathstalker",
  "display": "Deathstalker",
  "kind": "checkpoint",
  "language": "en",
  "narrator_engine": "higgs-v3",
  "backend_supported": true,
  "installed": true,
  "resident": false,
  "loadable": true,
  "reason": null,
  "revision": "<40 hex>",
  "fingerprint": "deathstalker@<40 hex>",
  "memory_bytes_estimate": 19000000000,
  "estimate_basis": "declared",
  "max_chars": 800,
  "sample_rate": 24000,
  "takes": 2,
  "needs_reference": false,
  "pace": {
    "pace_chars_per_sec": 16.64,
    "max_chars_per_sec": 21.63,
    "min_chars_per_sec": 12.8,
    "target_chars": null,
    "safe_min_chars": 600,
    "safe_max_chars": 800
  }
}
```

`revision`, `fingerprint`, `memory_bytes_estimate`, `estimate_basis` and `max_chars` are
`null` when `backend_supported` is false, because they live in the backend block this host
does not have — and `0` would read as "needs nothing".

**`takes` is how many rungs this voice's ladder has**, and it is on the row so a client can
ask before it submits. A `take` past the end is `unknown_take` and is never clamped, and a
client spreading N candidates across the ladder (BookForge's Correct Sentences does exactly
that) has to know where the ladder ends. What each rung MEANS is deliberately not published,
for the same reason `sampling` is not: the numbers are engine tuning and publishing them
invites a client to send them back. It is never below 1 — take 0 exists whether or not the
file declares it.

**`needs_reference` says whether a `load-voice` for this row must carry a clip** — true for a
`kind = "zeroshot"` voice and false for every other kind (section 5's amendment). On the row
so a picker can show its clip field before the load is refused, and derived from `kind` by
the server rather than left for a client to derive, because which kinds need one is the
server's rule. It is a fact about the KIND and not about the backend block, so it stays true
on a host this voice cannot be served on. **A client reads it the way it reads `route`**
(PHASE15-HOST.md section 3.3's client reading rule, asked once of the whole document): if no
voice row in the document carries `needs_reference` the server predates the field and every
voice is read as `false`, if some rows carry it and one does not the document is refused
`voices_needs_reference_missing` naming that row, and a value that is not a boolean is
refused `voices_needs_reference_unknown` — all or nothing, never a per-row default.

`pace` is the whole block and not the one key the draft showed: a client that is going to
pack has to see all of it, and the two shapes (a band, a target) are told apart by which keys
are null. `narrator_engine` is on the row because it is what decides which env a load needs
and therefore what a `reason` is talking about; `estimate_basis` is on it for the reason in
difference 4. Neither `estimate_note` nor `sampling_reason` is — they are prose for whoever
reads the manifest, and a row is not a place to argue.

**`sampling` is deliberately not on that row.** It is engine tuning, it is the server's, and
publishing it invites a client to send it back. The same goes for the EOS levers, the
`max_new_tokens` formula and the engine flags. What a client gets is the shape it must pack
to (`pace`, `max_chars`) and the identity it must record (`fingerprint`).

## 3. The take ladder is the server's steps and the client's judgment

`docs/PLAN.md`: *the ladder's steps are server config; the client asks for take N.*

```toml
[[voice.takes]]
# take 0 — the boson default. No deviation, so no reason is owed.

[[voice.takes]]
temperature = 0.7
reason = "measured 2026-09-11 over the same 88 chunks: 0.8 gave 4 guard fires / 3 holes / 0 drops, 0.7 gave 8 / 7 / 0. Not a better setting — a DIFFERENT one, which is what a retake of a chunk that failed at 0.8 needs (Owen's ruling 2026-09-14: a retake must not reuse the settings that produced the problem)."
```

A `tts` job carries `take: N`, an index into that list, and nothing else about sampling. The
client decides *that* a row needs another take and *which* take it keeps; the server decides
what take 1 means for this voice on this backend. A `take` past the end of the list is
refused by name (`unknown_take`) rather than clamped — a silent clamp is a retake ladder
that stops climbing without telling anyone.

This is the one place where the division of knowledge had a genuinely arguable alternative
(move the whole ladder, judgment included). It is written here so that changing it later is
a decision rather than a drift.

### RULED, and BUILT, 2026-09-14

Owen's question above is answered, and the answer keeps the ladder here. He was shown
BookForge's Correct Sentences spreading temperatures of its own (`computeTakeTemperatures`,
0.4 / 0.8 / 1.0 around an Orpheus-era 0.6) and ruled:

> *"we dont have to use temperature as the lever to get sentences to sound different, but the
> goal is to re-render sentences that dont sound quite right. prosody is bad, theres babbling
> or truncation, some other issue. thats why the feature exists. i was using temperature as a
> lever because it gives a different output. if we can get the same result without changing
> temperature then thats fine. i just know if a sentence/chunk was problematic before, itll
> likely be problematic again with the same settings used to originally generate it."*

So the requirement is not *a temperature*; it is that **a retake must not reuse the settings
that produced the problem**. Unseeded sampling alone gives a different output at the same
settings, which is the weak form; a different RUNG is the strong form, and a problematic
chunk needs the strong form at least once. **The spread IS the take ladder**, a temperature
is never on the app's wire, and a client spreads its N candidates across the rungs rather
than re-rolling take 0 N times.

Three things this build changed.

1. **The five fine-tunes declare rung 1** — `temperature = 0.7`, with the measurement above
   as its written reason. `higgs-default` and `zeroshot` deliberately do not: that
   measurement is 88 chunks of a fine-tune's output, a zero-shot voice's spread depends on a
   clip nobody has measured against, and a rung that is not measured is a number somebody
   will later mistake for one. They still have take 0, which every voice has whether or not
   its file says so.
2. **The rung reaches narrator PER ITEM, and A RUNG IS TWO FACTS: `(sampling deltas, seed
   offset)`.** `generate` and `generate_batch` items carry
   `sampling: {temperature?, topP?, topK?, repetitionPenalty?}` since
   `narrator/engine/item_sampling.py` and `take: N` since 2026-09-15 — section 4 has the
   details. Crucible resolves `take: N` against the voice's ladder and sends **only the keys
   the rung declares**: rung 1 is one line and means "take 0, but cooler", and each engine
   lays the item's keys over its resolved sampling key by key. Take 0 sends **no `sampling`
   key at all**, because absent means "the voice's loaded sampling", which is exactly what
   take 0 is — `{}` would be asking for a rung with nothing in it, which narrator refuses as
   `sampling_malformed`, correctly. The rung rides on each item and not on the request
   because that is where narrator's channel is, and because the streaming door legitimately
   mixes rungs in one batch.

   **`take` rides on EVERY item, 0 included**, and the asymmetry with `sampling` is
   deliberate: `{}` is not a sampling, but 0 *is* a take — narrator reads an absent key and
   an explicit 0 as the same number — so sending it makes the wire say which take produced
   each artifact.

   **THE SEED HALF, 2026-09-15.** Point 1 above was only half honoured for a day. narrator
   seeds chunk i at `config.seed + i` on both Higgs arms and the ladder never varied it, so
   two renders of one chunk at the same sampling were byte-identical however they were
   labelled: take 0 and take N whenever their numbers matched, and every pair of take-0
   re-rolls, always. Owen's requirement is *a retake must not reuse the settings that
   produced the problem*, and **a seed is a setting**. So a rung now also names a SEED
   OFFSET: narrator renders take N in that take's own lane,
   `seed + index + TAKE_SEED_STRIDE * take` (`engine/higgs/truncation.py`), and the stride
   is a whole number of the guard's re-roll lanes so no take's draw is any other take's or
   any re-roll's. **Two renders of the same chunk at the same take are byte-identical by
   design — that is what reproducibility means here — and a different take is a different
   draw even when its sampling equals take 0's.**

   The two halves are independent on narrator's wire: `take: 3` with no `sampling` is a
   legal item. Crucible's own `[[voice.takes]]` still refuses a rung above 0 that declares
   no numbers (`voices.py:_check_takes`: *"a rung that is the same sampling as the one below
   it is a different DRAW, which is what a re-roll is for"*) — a rule written when a
   different draw was the one thing a rung could not ask for. The seed lane makes such a
   rung expressible; **whether to allow one is an open ruling**, and nothing in this build
   depends on the answer.
3. **`sampling_not_wired` was DELETED on 2026-09-14 and CAME BACK on 2026-09-15 with a
   different subject.** Its original meaning — *this contract has no channel for a take
   above 0* — stopped being true and was not kept as a refusal nothing can raise. (Its
   take-0 half had already gone when the voices document started carrying sampling per load;
   the render door's went at the same time, and the streaming door's second one — a voice
   whose take-0 sampling deviates — went with it.) It now means **the narrator ON THIS WIRE
   has no rung channel**, which is a statement about a PROCESS and can never stop being
   possible: the tts env pins narrator by commit, and a pin is allowed to be old. It was.
   See the handshake below.

**THE HANDSHAKE: `itemTake` on `ready`.** narrator says `itemTake: true` before any engine
loads, and both doors refuse a take above 0 by name when it is absent rather than sending a
rung into silence. This is not hypothetical — on 2026-09-15 two render jobs on voice `owen`,
one 150-char sentence at take 0 and take 1, returned **byte-identical 264,174-byte
artifacts**, because the env's pinned narrator (bookforge `0eeb0267`) read `item['voice']`
and dropped the rest without a word. Crucible built the item correctly and reported a take
that never happened; the recipe's pin and this contract's belief about it were one fact with
two owners (`docs/ARCHITECTURE.md`), and the handshake is the comparison.

**The gate asks about the TAKE, not about the sampling.** What the caller asked for is take
N; what the handshake answers is "do you read a rung". `sampling is not None` stood in for
both and was equal to neither — it was held up only by the `_check_takes` rule above, in a
different file. (The key was `itemSampling` for one day, 2026-09-14 to 2026-09-15; it was
renamed rather than joined by a second key when the seed half landed, because a build has
both halves or neither. Nothing had shipped under the old name.) Engine-level support is a
separate and later fact and stays per row: `sampling_not_supported`, `take_not_supported`.

**narrator's own refusals travel back per row, by name.** A rung that is not a sampling is
`sampling_malformed`; one this engine has no lever for — the MLX arm has no repetition
penalty — is `sampling_not_supported`. Both arrive as that row's `message` and Crucible
carries them across as the row's error without interpreting them: one is a typo and the
other is the wrong backend, and the difference matters to whoever sent it. A per-item
refusal fails **its own row** and not the batch, which is narrator's rule and the reason
`_resolve_row` exists at all.

## 4. Engines: narrator is the managed subprocess

> **THE ENGINE LIST, 2026-09-14.** Owen: *"orpheus is deprecated too but hasnt been
> removed yet. higgs is the frontier"* — *"i guess we can remove it now."* This section
> described TWO narrator engines and Crucible shipped a recipe, an env key, a sampling
> row, a streaming width and a CLI choice for both. **There is now exactly one,
> `higgs-v3`**, and the paragraphs below that still reason about "the two engines" are
> kept because the REASON survives the engine: the per-engine env naming, the per-engine
> recipe and the per-engine tables exist so that the SECOND engine is an addition and
> never a rebuild of the first.
>
> What a new narrator engine has to add, one place per fact: a row in
> `voices.NARRATOR_ENGINE_SAMPLING` (the one list the `--narrator-engine` choices, the
> task door's refusal and `/v1/capability`'s `narrator_engines` all read), a row in
> `ttsstream.STREAM_BATCH_WIDTH` (a MEASURED width; `batch_width_for` refuses an engine
> nobody has measured and there is no default), a recipe per backend under `envs/tts/`,
> a row in `jobenv.CUDA_LINUX_SERVING_STACK` if it starts a server underneath narrator,
> and membership of `narratorvoices.DOCUMENT_READERS` if it resolves a voice by name.
> `tests/test_jobenv.py`'s drift guard compares the first of those against the recipes
> on disk in both directions, which is the check that was missing while an unservable
> engine sat in the list.

Crucible does not reimplement narrator's EOS surgery, guard or codec arithmetic. It runs
the code that already has them.

`python/narrator` in the BookForge repo is a proper installable Python package with exactly
the shape Crucible's engine layer wants: an engine registry keyed by id (of which
`higgs-v3` is the one Crucible names), a per-engine extras matrix in its `pyproject.toml`
that mirrors Crucible's
`envs/<type>/<backend>.txt` one for one, a resident server (`python -m narrator.serve`)
speaking newline-delimited JSON over stdin and stdout, and the EOS levers, caps, guards and
codec arithmetic that CLIENT-SURFACES.md row 15 calls "the single hardest thing in the `tts`
contract" already implemented and already measured.

So: **`narrator` is to `tts` what `vllm` is to `llm`.** A pinned dependency in the env
recipe, started and stopped by `crucible/engines/narrator.py` behind the same `Engine`
interface, SIGTERM only, its stdout and stderr in `~/.crucible/logs/engine-<voice>.log`.
narrator itself starts SGLang-Omni on `cuda-linux` and loads MLX in-process on
`mlx-darwin`, and tears its engine down when it exits — which is exactly the lifetime
Crucible's lease wants (CLIENT-SURFACES.md section 3.4).

Two consequences, both named rather than hidden:

- **The readiness probe is not HTTP.** Every other engine answers `/v1/models`;
  `narrator.serve` answers a `ready{device,backend}` line on stdout. `SubprocessEngine` has
  to grow a seam for that, rather than the narrator engine faking an HTTP server to fit the
  one that exists. **Done, 2026-09-13**: `ready()` now calls `announced_ready()`, which
  returns the message when the engine is up and `None` while it is not, and names what it was
  waiting for in `readiness_description()` so a timeout reads truthfully. The default is the
  `/v1/models` poll, byte for byte what vLLM and mlx-lm had before; neither overrides either
  method, and a test asserts that. `tests/test_engine_readiness.py` drives the seam with
  `tests/fake_narrator.py`.
- **`start()` needed a second seam. Done, 2026-09-13.** It gave every engine
  `stdin=DEVNULL` and sent stdout to the log file, which is right for an engine whose wire is
  HTTP and wrong for narrator, whose wire *is* those two pipes. The seam is `stdio()`,
  returning the Popen keyword arguments that describe how the three standard streams are
  wired — keyword arguments rather than three streams, because the text and buffering modes
  belong to the same decision: a pipe Crucible writes JSON lines to is a text-mode,
  line-buffered, **UTF-8** pipe (stated rather than inherited — the text on it is a book and
  the locale of whatever shell started the server has no business deciding how an em-dash
  crosses it), while a log file taking an engine's raw output is not. Two smaller hooks came
  with it: `attach(process)`, called once the process exists, where narrator starts reading;
  and `detach()`, called at the end of `stop()`, where the reader is joined and the pipes are
  closed. The default of all three is what vLLM and mlx-lm had, neither overrides any of
  them, and a test asserts that alongside the readiness one.
- **Crucible ends up depending on a package that lives in an app's repo.** narrator's
  `engine/` and `serve/` know nothing about audiobooks, but `compat/` and `assemble/` do,
  and the whole thing is versioned with BookForge. The env recipe pins it by git sha so a
  server is never surprised. **Extracting `narrator` into its own repo is an owed ruling for
  Owen**; until he makes it, this pin is the honest arrangement rather than a workaround.

`envs/tts/` holds one recipe per (narrator engine, backend) rather than one per backend, and
the reason is in narrator's dependency matrix rather than in Crucible's design: each engine
pins its own serving stack against its own torch — Higgs v3 needs `vllm-omni==0.28.0`
against torch 2.13, and narrator's own pyproject says installing any two into one env
"resolves torch twice and breaks whichever loses". So `cuda-linux` gets
`~/.crucible/envs/tts-<engine>/` and the voice manifest's `narrator_engine` picks which one
a load uses; today that is `~/.crucible/envs/tts-higgs-v3/` and nothing else. On
`mlx-darwin` the engines share one env, because on the Mac they genuinely do, and the
recipe there is named for the backend.

Two site-packages patches must be re-applied after any upgrade of the `higgs-v3-server`
group. pip cannot express that. `crucible doctor` checks for both and reports them by name,
as their own rows and their own problems rather than folded into the env row — an env whose
pins all match is otherwise reported ready, and a reader has no way to tell that from an env
that will render every chunk with 240 ms of garbage on the end.

**"Must be re-applied" named nobody until 2026-09-15, and the sentence was load-bearing.**
pip does not merely fail to express a patch to somebody else's package — it UNDOES one, by
writing the distribution's own file over the edit. MEASURED on owens-pc: `crucible install
tts --narrator-engine higgs-v3 --build --force` at 07:34 restored both files pristine, and
from 07:46 every Higgs load failed at narrator's sentinel proof —
`.log.sentinel.jsonl holds no records`, the report 0 bytes, because the code that writes a
record per invocation had just been uninstalled. Renders at 07:23 worked on the same pins,
the same recipe and the same narrator sha, and both logs read `vLLM server version 0.28.0`:
the pins were never the question. `crucible doctor` said `missing` for both the whole time,
from a command nobody runs after an install.

So `jobenv.install_env` now RUNS the two appliers after pip and before it stamps the env,
and `narratorpatches.apply` re-runs the checker over the result and refuses the install when
it disagrees — a script's exit code is its own idea of success, and the thing that has to be
true is the marker `crucible doctor` greps for tomorrow. An env that is stamped installed is
an env whose patches are in. Only for `tts`: the `llm` recipe pins `vllm` too, and the
recipe's own pins then select a second time, so `mlx-darwin` (neither distribution) runs
neither script and is not called broken for it.

The appliers themselves live in `crucible/envs/tts/patches/`, byte-identical copies of
BookForge's `electron/scripts/higgs/`, vendored for the same reason the marker table below
is a copy: narrator's wheel does not carry them (it ships `engine/higgs/launch/` and nothing
else). **The owed move is narrator shipping its own patches** — the engine that requires a
patched server is the honest owner of the patch — on the same ruling as extracting narrator
into its own repo. Until then `tests/test_narrator_patches.py` reads `REL`/`MARKER`/
`ABSENT_MARKER` out of each script and asserts they are the table's, so the two halves
cannot drift apart in silence.

**One of the two names in this document was stale, and building the checker found it.** The
second patch is `patch_sentinel_filter.py`, not `work/patch_tail_trim.py`; narrator's own
`pyproject.toml` still says the old name too. The difference is not cosmetic. The retired
script wrote the helper `_trim_trailing_sentinel_frames`, and so does the live one, so a
checker grepping for that helper would certify a band-aided env as patched. The markers
Crucible uses are BookForge's own measured ones, mirrored into
`crucible/narratorpatches.py` (the duplication is deliberate for the reason
`electron/tool-paths.ts` gives about its own: a Crucible server must not need a BookForge
checkout to answer "is this env sound"):

| patch | file | present | absent | what breaks without it |
|---|---|---|---|---|
| `vllm-negative-token-id` | `vllm/v1/engine/input_processor.py` | `min_input_id != -100` | — | every voice-clone request is HTTP 400, so only the default voice serves |
| `higgs-sentinel-filter` | `vllm_omni/.../higgs_audio_v3.py` | `_filter_sentinel_frames` | `[:, :-1]` | every chunk ends in ~240 ms of audible garbage |

`higgs-sentinel-filter` additionally reports **stale** rather than applied when the marker is
there but `final=%s, window=%d frames` is not: that is v1 of the patch, which substituted
sentinels before the identity trim so the trim found nothing, and it looks patched to any
marker grep.

### What `crucible/engines/narrator.py` is, and what it is not (2026-09-13)

It is one class for both narrator engines, because from Crucible's side they differ only in
which env the interpreter comes from and what `NARRATOR_ENGINE` says; what runs underneath is
narrator's business and Crucible learns which it got from the `ready` line. The argv is
`<tts env python> -m narrator.serve` and nothing else — the voice rides the `load` message
(for `higgs-v3` the weights are in the voices document Crucible writes, see below; an engine
that reads no document carries its weights directory on the message instead), and the port is
not used at all, so **`base_url` refuses** rather
than returning a port nothing is listening on.

Four things about it that are decisions rather than details:

- **`converse()` yields lines in arrival order and reorders nothing.** Rows retire out of
  order — a short row finishes while a long one is still going, and `tests/fake_narrator.py`
  retires a batch in reverse on purpose — so the row's identity is the `i` narrator echoes
  back, which is the *caller's* number. Buffering into caller order here would defeat both
  doors: the render door writes each FLAC as its row retires, overlapped with the next row's
  generation, and the streaming door needs `batch_chunk` lines out of the same iterator while
  a row is still generating. One reader thread, one iterator, and the consumer keys on `i`.
- **A line on stdout that is not a protocol message is a refusal naming the line.** The same
  rule, from the same incident, as `crucible/workers.py`: narrator's own aligner had a library
  log to stdout on a 401-chunk book and corrupt the result stream.
- **`stop()` sends `{"action": "quit"}` before it signals.** narrator's own docstring calls
  the stdin `quit` its primary teardown — it unwinds the stdin loop from inside the process
  and releases the GPU through the atexit hooks. SIGTERM reaches the same place through a
  handler that raises `SystemExit(143)` and is the backstop for a worker that has stopped
  reading its stdin. Neither is SIGKILL.
- **`load-voice` now means the weights are in memory, not that a process is up.** `ready` says
  narrator is listening; `loaded` says the engine underneath it has a voice. So the `load`
  message is part of the load: `Residency._start` grew a `confirm` argument (a model's engine
  has nothing there — a 200 from `/v1/models` means the weights are on the card), and a
  failure in it tears the engine down exactly as a readiness failure does.

**The sample rate is narrator's, and a disagreement is a refusal.** `/v1/voices` publishes
`sample_rate` off the manifest and a client writes FLACs at it; narrator reports on its
`loaded` line the rate the engine it actually built renders at. The two are compared at load
time and a mismatch names both numbers. It is deliberately not a resample: audio resampled to
match a manifest is audio that no longer matches the engine, and nothing downstream would say
so.

### The voices document: Crucible writes what narrator reads (2026-09-14)

**A Higgs v3 voice is a NAME, not a directory on the `load` message.** Crucible's first real
render found this on both arms in one night. On `cuda-linux` the launcher agent read
`Higgs v3 load carried modelDir='…'. The served model is the launch script's argument, not a
per-load field` off the engine log; on `mlx-darwin` the keeper got the same refusal from
`HiggsV3MlxEngine.resolve_load_voice` ("which weights the MLX backend loads comes from the
voice document … not from a per-load field"). Both arms then resolve `voice` by name in a
JSON document whose path is **`NARRATOR_HIGGS_VOICES`** (`engine/higgs/config.py`:
`voices_path` refuses an unset variable, `load_voice` refuses an absent voice, naming the
ones the file has). Crucible had never written one. Two defects, one cause: the document is
per-engine tuning — the merged directory the server starts on, the cap, the safe band, the
pace triple, the sampling — and under BookForge `electron/higgs-models.ts:higgsVoicesDocument`
writes it per spawn from `higgs-models.json`.

**Under Crucible the one owner of every one of those facts is the voice manifest**, and the
weights are where `crucible voices pull` put them. So `crucible/narratorvoices.py` is
`higgsVoicesDocument` for a server that has never heard of BookForge: `Residency.load_voice`
calls `write_document(home, manifest, spec, weights_dir)` **at every load**, before the engine
is built, and hands the result to `build_voice_engine(..., voices=)`; `NarratorEngine
.environment()` then carries `NARRATOR_HIGGS_VOICES=<path>`. One file per server —
`~/.crucible/narrator-higgs-voices.json` — holding **exactly the voice being loaded**, overwritten
each time. Per load and not per engine start or per install, because a Higgs v3 voice change
IS a worker restart, so a load is the one moment the document has to be true, and a document
listing every installed voice would be a list of claims about stamps nobody re-checked.

The entry, key by key — every one a key narrator's `load_voices` reads, spelled as it reads
it, and nothing it does not read:

| key | from | note |
|---|---|---|
| `kind` | `[voice].kind` | `checkpoint` → `checkpoint`; `token` → **`default`**, narrator's name for the model's own voice |
| `checkpointDir` | the pulled directory | a checkpoint's merged weights, or a **zeroshot voice's BASE weights** (2026-09-14 — both arms load it, which is how a clone renders on the bytes the pin names instead of on whatever base snapshot the HF cache holds). **`kind` is what says which**, and the two are one statement: narrator reads a `clips` voice's directory into `ClipsVoice.base_dir` and every other kind's into `checkpoint_dir`, then checks the required files at the load message (`v3_served.voice_serve_target`) — a merge's `generation_config.json`, base weights' bytes and nothing else. See "What base weights are not" below |
| `clips` | the `load-voice`'s `reference`, placed on disk | zeroshot voices only, and required of them: `[{path, transcript, seconds}]`, narrator's own three keys. `seconds` is MEASURED by Crucible off the wav header, never taken from the client |
| `maxChars` | `[voice.backends.<arm>].max_chars` | characters; the one key narrator refuses a checkpoint without |
| `targetChars` | `[voice.pace].target_chars` | when declared |
| `safeMinChars`, `safeMaxChars` | `[voice.pace].safe_*_chars` | when declared; the manifest loader has already refused a band above the cap |
| `sampling` | `[voice.backends.<arm>].sampling` | as `{temperature, topP, topK}`; `topK` a whole number because narrator refuses `50.0` |
| `paceCharsPerSec`, `maxCharsPerSec`, `minCharsPerSec` | `[voice.pace]` | the triple narrator's `_length_band` takes all-or-nothing |

Not written, each deliberately (the module docstring says why): `maxCharsSource`, `scene`,
`allowedControls`, `maxReferenceSeconds`, `clips`, `_overrideNote`.

**The `load` message per engine.** `higgs-v3`: `{"action": "load", "voice": <id>, "warm":
true}` — nothing else, on both arms; the engine refuses, before sending, a voice the document
does not carry and a `weights_dir` the document's `checkpointDir` disagrees with (two
statements of one fact, compared). An engine outside `DOCUMENT_READERS` takes
`{"action": "load", "voice", "modelDir", "warm"}` instead, byte for byte what narrator's wire
always had; it reads no document and is refused one. `tests/fake_narrator.py`
now makes narrator's own refusals under `--engine higgs-v3` — `modelDir` by name, an unset
variable, an absent voice — so a residency that stopped writing the document fails in the
suite rather than on a book.

**ONE (voice, arm) pair is refused by name before any engine starts**, in
`narratorvoices.voice_entry` — it was two until 2026-09-14, when the zero-shot clip got a
channel (section 5's amendment) and the first refusal stopped being true:

- ~~`kind = "zeroshot"`, either arm.~~ **Lifted.** It was refused because "a Crucible
  zeroshot voice's clips are either `from-request` or files in a refs repo nothing has laid
  out, and an entry naming files Crucible has not checked is a load that dies inside
  narrator". The load door now carries the clip, Crucible writes the wav itself beside the
  document, and the entry names a path this process just wrote. What replaced it is a pair of
  refusals about the CLIP rather than the kind: a zeroshot voice with none, and a clip on a
  voice that is not one.
- `kind = "token"` on `cuda-linux`. narrator's served arm exports `HIGGS_MODEL_DIR` only for a
  checkpoint voice and **unsets** it otherwise, and its launch script then serves "the base
  snapshot out of the HF cache" — not the directory Crucible pulled at the pin, so a server
  started that way would render under a fingerprint naming bytes it never read. On
  `mlx-darwin` the same voice loads: the base weights come from `NARRATOR_HIGGS3_MLX_MODEL`
  (`model_dir = checkpoint or model_dir_from_env()`), which the document sets to the pulled
  directory. **RULING OWED, narrator's side:** a way for the served arm to be told the base
  directory for a `default` voice. Until then `higgs-default` is a Mac-only smoke voice.

  **A lead on that ruling, found while wiring zeroshot and deliberately NOT acted on.**
  narrator's document reader passes `checkpointDir` into `DefaultVoice` exactly as it does
  into `ClipsVoice`, and the served arm exports whatever `checkpoint_dir` the config ends up
  with — so writing the pulled base directory as a `default` voice's `checkpointDir` would
  very likely make the served arm start on the bytes Crucible pinned, which is the whole of
  what this refusal is waiting for. That is the move this build now makes for `clips`. It is
  not made for `token`, because the two differ in what has been tested and in whose decision
  it is: a zero-shot load is a new door built to a written plan, and re-pointing
  `higgs-default` is a behaviour change to a shipped smoke voice on an arm nobody has run it
  on. Owen's ruling, with this lead in front of him.

### What base weights are not: the first zero-shot load, 2026-09-15

The first zero-shot load ever made through Crucible on the PC was refused by name at
00:11, before anything touched the card:

```
engine_failed: narrator (higgs-v3) refused the request: Higgs v3 voice 'zeroshot':
the merged checkpoint /home/telltale/.crucible/voices/zeroshot/cuda-linux does not
carry generation_config.json, which is a REQUIRED ...
```

**The pull was complete and the file does not exist.** `voices/zeroshot.toml` pins
`bosonai/higgs-tts-3-4b` at `239f63fb7b02b1aa085f98d9efae5e35cc5523e8`; that tree lists
thirteen files — `config.json`, `chat_template.jinja`, `model.safetensors` and its index,
the tokenizer pair, `.gitattributes`, `README.md` / `AGENTS.md` / `PROMPTING.md` /
`LICENSE`, one asset — and no `generation_config.json` under any name or subdirectory.
Every one of them is in `~/.crucible/voices/zeroshot/cuda-linux`, with
`crucible-pull.json` recording the same repo and revision, so `weights.pull` fetches the
whole snapshot and omitted nothing. Nothing on Crucible's side was wrong.

**The fault was a field with two meanings, and it is fixed in narrator.** `checkpointDir`
meant "the merged fine-tune this voice IS", whose own `generation_config.json` is the
sampling `vllm-omni serve` resolves from the model directory (`--generation-config auto`);
narrator requires that file of a merge because an empty request against a directory without
one gets a bare `SamplingParams()` — top_p 1.0, top_k DISABLED — which derails long chunks
into babble. Base weights have never carried it, which is why narrator has sent
`SERVER_DEFAULT_SAMPLING` explicitly for them since 2026-09-05. When this build started
writing the pulled BASE directory into that same key so a clone would render on pinned
bytes, the requirement came along with the name.

So narrator reads the KIND beside the key: a `clips` voice's directory lands in
`ClipsVoice.base_dir`, checked by `v3_served.require_base_weights_dir` (the bytes, no
file), and every other kind's in `checkpoint_dir` as before. Naming both is refused —
one server runs on one model. **Crucible's wire does not change**: it already writes
`kind` and `checkpointDir` together, and `tests/test_narrator_voices.py` now asserts the
pair rather than either half, because writing `checkpoint` there would resurrect the
refusal with no other symptom.

Two things that followed from the same conflation and are fixed with it: the config's
`served_sampling` went EMPTY for any voice naming a directory, so a zero-shot render on
base weights would have gone out with no sampling at all — the exact babble case — and
the prep's safe-band check refused any voice naming a directory as "a fine-tune that
declares no safe band", which a zero-shot voice is not and never declares.

### Sampling reaches narrator: take 0 through the document, a rung per item

**No `caps` are sent on the `load` message**, still. narrator's caps channel is
`register_voice_caps`, whose key vocabulary is narrator's older engine's — `temperature`, `topP`, `minP`,
`repPenalty`, the four `eos*` levers, `maxCharsPerSec` — with **no `topK` at all**, which
*raises* on a key it does not know, and which `higgs_v3_config_from_worker_kwargs` refuses
wholesale by name. **The document is the channel that exists**: its `sampling` key is read
onto the voice by `load_voices` and applied as the engine's override on both arms
(`v3_engine.higgs_v3_config_from_worker_kwargs` since the pinned 0eeb0267;
`mlx_backend.higgs_v3_mlx_config_from_worker_kwargs` since 2026-09-06). Every manifest in this
build states the boson default, so what the document asks for is what the engine would have
rendered at — **except** on a merged checkpoint whose own `generation_config.json` says
otherwise, where writing it is what makes take 0 the boson default rather than whatever the
merge script left in the file. A voice that deviates with its written reason renders at what
it declares; the take-0 half of the old `sampling_not_wired` refusal is gone because it had
become false.

**And the other half went on 2026-09-14, when narrator grew the per-item channel.** A rung is
per RENDER and the document is per LOAD, so until narrator's `generate` and `generate_batch`
took sampling there was nowhere for take 1 to go and a reload per take is not a ladder. There
is now: `narrator/engine/item_sampling.py`, one module, one spelling and one set of refusal
names — `sampling_malformed` / `sampling_not_supported` for the numbers, and, since
2026-09-15, `take_malformed` / `take_not_supported` for the item's `take`, which is the seed
offset half of the same rung (section 3, point 2). A build that has the module has both
halves and says so once, as `itemTake` on `ready`.

| | |
|---|---|
| the wire | an item may carry `sampling: {temperature?, topP?, topK?, repetitionPenalty?}` |
| the spelling | **the voices document's**, deliberately — `engine/higgs/config.py`'s `_SAMPLING_KEYS`, so the per-load channel and the per-item channel are one vocabulary and not two names for one lever |
| absent | the voice's loaded sampling, which IS take 0. Not a fallback: it is the documented meaning of "no rung" |
| an overlay | the rung is applied OVER the engine's resolved sampling KEY BY KEY, so a one-line rung keeps take 0's `top_p` and `top_k`. This is measured rather than tidy: on SGLang-Omni an unset `top_k` is the untruncated 1026-way codebook tail, one chunk to the cap with 80 s of silence |
| refused | `sampling_malformed` (not an object, empty, an unknown key, a value that is not a positive number — `topK` not a whole one) and `sampling_not_supported` (well formed, and this engine has no such lever, or has no per-item sampling at all). **Per row**, never per batch |

Crucible's half is `narratorvoices.take_sampling(manifest, take)`: the rung's own keys,
translated through the same `_SAMPLING_ON_THE_WIRE` map the document uses, and `None` at take
0. Both doors send it on every item — the render door one take per job, the streaming door one
per `say` — and both carry narrator's per-row refusal back to the client as that row's error,
by name.

## 5. Residency holds one thing, whatever kind it is

`crucible/jobs/llm/residency.py` held at most one resident *model*. The accelerator
does not care what kind of thing is on it, and a card holding a Higgs checkpoint has no room
for a 9B. So `Residency` generalises: **at most one resident engine, of either kind**, and
loading a voice unloads a model exactly as loading a model unloads a voice.

`GET /v1/health`'s `resident_models` keeps its name and its shape (a list of ids) and gains
`resident_kind: "llm" | "tts" | null`, so a client can tell which door to knock on.

**Done, 2026-09-13.** The holder moved to `crucible/residency.py` — it is no longer the
llm's, and a `tts` job reaching into another job type's package for the thing that owns the
card would make the one-at-a-time rule look like a courtesy between two modules rather than a
property of the server. `build_registry()` hands the SAME holder to every job type that
touches the card, which is what makes the rule true rather than aspirational, and a test
asserts it.

Two things fell out of the generalisation that the draft did not anticipate, and both are
about ids:

- **Model ids and voice ids are separate namespaces**, and nothing stops a voice being called
  `qwen3.5-9b`. So residency is asked `is_resident(kind, id)` and never `resident_id == id`,
  or a resident voice would light up a model's `/v1/models` row; and `unload-model` /
  `unload-voice` each check the kind before asking the holder, which unloads by id alone.
- **The OpenAI proxy asks for `resident_model`, not `resident`.** With a voice on the card
  there is no `base_url` to forward a chat request to, and narrator answers no OpenAI route,
  so that door's honest answer is the same `model_not_resident` it gives for an empty card.

Their weights are separate on disk too: `~/.crucible/voices/<id>/<backend>/` beside
`~/.crucible/models/<id>/<backend>/`, so a `crucible voices pull` can never overwrite a model
and leave a stamp that reads as installed to either.

New job types, mirroring the model pair exactly, and enabled by `[jobs] enable_tts` in
`config.toml` (`crucible init --enable-tts`) the way `enable_llm` enables the model pair:

| Type | Refusals, all before queuing |
|---|---|
| `load-voice` | `unknown_voice`, `voice_not_installed`, `backend_unsupported`, `env_missing`, `accelerator_busy`, `insufficient_memory`, and the reference trio below |
| `unload-voice` | `voice_not_resident` |

One note on the first of those: an unknown id is refused as **`unknown_model`** rather than
`unknown_voice`, because `jobs.resolve_model()` gets there first — it checks the requested
model against what the job type advertises, for every job type, before `preflight()` runs.
`unknown_voice` exists and is what `crucible/jobs/tts/` raises from its own lookup; it is
simply not the code a client sees on this path. Changing that means changing `resolve_model`
for every type, which is a decision about the whole API rather than about `tts`.

### AMENDMENT, 2026-09-14: `load-voice` carries the zero-shot reference clip

Owen, on what a zero-shot voice IS:

> *"zero shot uses a voice reference and the base model i believe. it should effectively be
> treated as a model, for all intents and purposes, except the route it takes to retrieve and
> return the audio."*

So it loads through this same door, with one extra field. The weights are the server's — the
base model, pulled at `voices/zeroshot.toml`'s pin, like any other voice's — and **the clip
is the client's**: BookForge keeps its four in `<userData>/runtime/higgs-models/refs/`, the
browser extension keeps its own in the browser, and neither is published anywhere a server
could pull from. A clip is a per-client CHOICE, like the voice pick itself, so it travels
with the load, which is the one moment it is needed.

```json
{
  "type": "load-voice",
  "model": "zeroshot",
  "params": {
    "reference": {
      "data": "<base64 of a RIFF/WAVE file, no data: prefix>",
      "transcript": "He had been walking for some time.",
      "name": "the stranger"
    }
  }
}
```

**`model`, not `voice`, and `params`, not the top level.** `POST /v1/jobs` is
`{type, model, params, inputs}` for every job type and the wire's word for the thing that
produces the bytes is `model` (section 6); a second envelope for this one door would be a
second shape to learn. The SDK's `loadVoice(voice, {reference})` spells it the readable way.

**The field names are narrator's, translated once.** narrator reads reference clips out of
the `NARRATOR_HIGGS_VOICES` document as `{path, transcript, seconds}` — where `path` is a
file on the server's own disk, which it checks with `os.path.isfile`, base64s itself into
vllm-omni's `references[].data` on the served arm, and hands to `encode_reference_audio` on
the MLX one. A client across a network has no such path, so the wire carries the bytes and
`crucible/voicereference.py` is the one place they become a file: written to
`~/.crucible/narrator-reference.wav`, beside the document and in the same breath, because
narrator checks the path the document names.

Two deliberate differences from that shape, each with its reason:

- **`seconds` is NOT on the wire.** narrator needs it — `reference_seconds` raises on a clip
  that has none, because the 30-second budget is checked before the request is built — but
  Crucible is holding the bytes and reads the duration out of the RIFF header. A duration the
  client states is a second owner of a fact the server already has, and the day the two
  disagree the refusal would name the honest one as the liar (`docs/ARCHITECTURE.md`, R1).
- **`transcript` IS required**, and BookForge's `docs/EXTENSION-TO-CRUCIBLE-PLAN.md` §4b —
  which sketched this field as `{data: <base64 wav>}` alone — is wrong about that. narrator
  refuses a `ReferenceClip` with an empty transcript AT CONSTRUCTION and says why in as many
  words: *"a zero-shot clone conditioned on a wrong or absent transcript is a whole book in a
  subtly wrong voice, reported as success"*. It is the book-exact text the clip was cut from,
  never an ASR guess — the same law the training corpora are held to. **The extension's clip
  picker therefore needs a transcript field beside its file input.**

`name` is optional and is a label for whoever reads the resident report; it is never derived
from anything, because a made-up name is one a client would then look for.

**The three refusals, all before the job is queued:**

| code | what it means |
|---|---|
| `reference_required` | the voice's kind is `zeroshot` and the load carries no `params.reference`. The base weights with no reference are the model's OWN voice — a different speaker at 12 % of the narrator ceiling — and a book rendered in it under this id would be reported as success |
| `reference_not_allowed` | a `checkpoint` or `token` voice carries one. A checkpoint's voice is in its weights and a token voice's is in the engine; narrator would clone from the clip and leave the weights this load names doing nothing, under their fingerprint |
| `reference_malformed` | `data` is not base64 (validated strictly — a `data:` prefix or a pasted newline is a refusal, not a silent skip), the bytes are not a readable WAV, the transcript is missing or blank, or the clip is over **30.0 s** (`v3_served.MAX_REFERENCE_SECONDS`; vllm-omni answers HTTP 400 "Reference audio too long" above it) or over a 32 MiB decoded ceiling, checked against the encoded length first so a gigabyte is never decoded |

**The resident report says which clip.** `zeroshot` is one voice id and any number of
recordings, so the id alone is two clients each assuming the resident one is theirs.
`GET /v1/activity`'s `resident` block gains `reference: {name, sha256, seconds}` — `null` for
every other kind and for a model, always present, and the sha256 is over the DECODED audio so
two clients sending the same wav agree and a client who re-encoded a different take does not.
The `load-voice` job's own `done` carries the same object. A `/v1/voices` row does **not**:
a row is a statement about the manifest and this host, not about the load, and it already
says `resident` and `needs_reference`.

**What this does NOT change.** The render door still refuses a zero-shot voice that is not
already resident (`voice_kind_unsupported`, section 6) — a render job loads its own voice and
has no `reference` field, and inventing a second clip channel there would be two doors owning
one fact. The streaming door stopped refusing the kind entirely: it never loads, so a
`zeroshot` session can only ever attach to a voice that was loaded with its clip.

## 6. The render door — job type `tts`

A normal job on the exclusive lane.

```json
{
  "type": "tts",
  "model": "deathstalker",
  "params": {
    "language": "en",
    "take": 0,
    "chunks": [
      { "index": 41, "text": "He had been walking for some time." },
      { "index": 42, "text": "The road did not appear to end." }
    ]
  }
}
```

**`model` is the voice id.** The wire's word for "the thing that produces the bytes" is
`model`, and for `tts` that thing is the voice — which for Higgs is not a pun but the
literal truth: a v3 voice *is* the merged checkpoint the engine was started on
(CLIENT-SURFACES.md section 3.3, "a Higgs voice change is a full worker restart"). So
`describe_models()` for this type returns the voices, `/info`'s `tts` capability rows are
voices, and provenance records `model: {id: "deathstalker", revision: "<checkpoint sha>"}`
with no new vocabulary invented for it.

**Artifacts: `<index>.flac`, one per chunk**, mono 24 kHz PCM_16 — byte for byte the format
BookForge's assembly and resume already expect, so nothing downstream changes. Each carries
its provenance sidecar, as every artifact does. The index is the **client's** and is never
assigned or renumbered here: it travels out as narrator's batch `i`, back on the retiring
row, and into the artifact's name.

**ffmpeg encodes the FLAC, and that was a real decision.** narrator hands back base64 PCM16
and something has to turn it into a file. Not `soundfile`: `libsndfile` is a compiled
dependency on every platform Crucible installs on, and growing the server's own interpreter a
compiled audio library so it can re-encode audio it did not decode is the wrong shape — this
process deliberately imports no torch, no vLLM and no narrator, and that would be the first
crack in it. ffmpeg is already a hard requirement (`asr` refuses `ffmpeg_missing` before it
queues a job), so `tts` refuses **the same way, by the same name, through the same probe**,
with its own reason in the message.

    ffmpeg -f s16le -ar <rate> -ac 1 -i pipe:0 -c:a flac <index>.flac

**`<rate>` is not a constant.** It is the rate narrator reported on its `loaded` line, which
the load already compared against the manifest and refused on a disagreement — so it is both
the engine's truth and the manifest's, which is the only state in which writing a header is
honest. Every retiring row also carries a `sampleRate`, and a row whose rate is not the
loaded one is a **failed row**, not a resample: the bytes would be at one rate and the header
would claim the other.

**The whole job is one `generate_batch`**, with no `stream` flag anywhere in the item list —
narrator's own docstring: "A generate_batch with no `stream` flag anywhere takes the
pre-existing code path, byte for byte." How many rows run at once is engine tuning and belongs
to the server, and "the server" in that sentence is narrator rather than Crucible:
`generate_batch` does its own scheduling, grouping consecutive rows on MLX and dispatching
singly on vLLM, and cutting the list up here would be second-guessing a read-ahead window
Crucible cannot see.

**A render job may load its voice; a stream may not.** This is the one asymmetry with `llm`,
and it is deliberate. A chat request is fine-grained and unattended — two clients alternating
would thrash the card, so the proxy never loads (PHASE2-LLM.md section 5). A render job is an
operator's explicit order, it owns the exclusive lane for its whole duration, and it is the
thing the queue was built to serialise. So if the wrong voice (or none) is resident when a
`tts` job reaches the front of the lane, the job loads it, emitting `warming` events exactly
as `load-voice` does. The streaming door, being a connection rather than a job, behaves like
chat: it refuses with `voice_not_resident` and names what is resident instead.

**Progress and measurement.** Alongside the standard `progress {fraction, message}` and
`artifact {name}`, a `tts` job emits one new event per chunk:

```
chunk {index, seconds, chars, chars_per_sec, tokens, capped, take, guard}
```

> **AMENDED by PHASE6-REMOTE-RENDER.md (Owen's ruling, 2026-09-13).** This section used to
> say that the seven fields above were "the whole guard interface", that the server measures
> and the client judges, and that BookForge's PaceTracker would read those numbers and
> decide. **The model judges, the server forwards, the client orders.** The seven fields
> survive, unchanged, as the server's own measurements; `guard` is the eighth, and it is the
> verdict narrator's engine already reached — forwarded verbatim, `null` when narrator sent
> none. Crucible reads nothing inside it. PHASE6 sections 3 and 4 are current; read them
> before writing a consumer.
>
> Why the old text could not stand: the PaceTracker was never in this path, and until
> narrator grew `render_many` the serve world — the door this job drives — had **no guard at
> all**. See PHASE6 section 0.

`capped` is true when generation stopped because it hit the frame cap rather than because
the model finished — the difference between "a long sentence" and "a runaway". It is now one
input to a verdict the engine has already reached rather than a number a client has to
reason from.

Three of the seven measured fields are the server's own arithmetic and three come off the
wire, and the split matters:

- `index` and `take` are the request's. `chars` is **Crucible's own count of the text it
  sent**, not a number read off the reply — the rule `crucible/workers.py` states about
  positional results: a number a subprocess echoes back is a number a subprocess can get
  wrong, and this one is already known exactly.
- `seconds` is measured from the PCM that actually arrived (`len(pcm) / 2 / sample_rate`) and
  then **compared** with the duration narrator reported. A disagreement of more than 50 ms —
  one Higgs frame is 40 ms at 25 fps, so this is not rounding — fails that row: a reply
  describing audio other than the audio attached to it is not a measurement.
  `chars_per_sec` is those two divided.
- `tokens` and `capped` come off the wire, and **narrator does not put them there.**

**`capped` and `tokens` are `null` against the pinned narrator, and `null` means "narrator
did not say".** `serve/worker.py` sends `{i, format, data, duration, sampleRate}` for a
retiring row and nothing else; the frame cap it computed (`HiggsBudget.cap_frames`, clamped
by `sgl_served.frame_cap`) never leaves the engine, and there is no token count on the wire
at all. Crucible cannot derive either — the cap is narrator's own arithmetic over the text,
and the server never sees a frame count. So both are nullable, and **`null` is never to be
read as `false`**: a runaway reported as "not capped" is exactly the failure the field exists
to prevent. `tests/fake_narrator.py` does send them, which is how the reporting path is
tested, and its docstring now says in as many words that a test asserting `capped is True` is
asserting about that file rather than about narrator.

**Owed on narrator's side:** `capped` and `tokens` on each `batch_item` (and on `done` for a
streamed row). It is a few lines where `cap_frames` is already in scope, and until it lands
BookForge's PaceTracker gets a duration and a `null` where it wants a flag.

`chunk` is an addition to DESIGN.md section 4's event vocabulary. It is additive and
`api_version` does not move: a client that does not know the kind still sees every
`progress`, `artifact` and `done` it saw before. The SDK yields unknown event kinds through
rather than dropping them, so that stays true for the next one too.

That last sentence was **false for the first day of its life**, and it is worth leaving the
correction visible. `readEvent` narrowed the name against a closed list and threw, so a
v0.2.0 client watching any job on a server that had learned `chunk` lost the whole stream at
the first one — while this paragraph was the stated reason `api_version` did not have to
move. It is now what it always claimed: an unknown kind arrives as
`{event: 'unknown', kind, data}`, is never terminal, and
`sdk/ts/test/unit-unknown-event.test.ts` fails if that stops being true. The same rule was
then needed one level out, for a capability whose ROWS a client cannot read — `info()` broke
the same way, and for the same reason.

**A failed chunk is reported and the run continues** — the same rule `align` and `rvc` have,
and the opposite of `asr`'s own, for a stated reason: a transcript with a fifteen-minute hole
in the middle is invisible in the output, while a missing `<index>.flac` is a file that is not
there and resume already knows how to ask for it again. So one bad sentence never sinks the
other 1,399. A failure is named in a `progress` line the moment it happens — so a client
watching the stream learns which index to re-ask for without waiting for the end — and again
in `done`, which carries `rendered`, `failed: [{index, message}]`, `take` and `sample_rate`.
No `chunk` event is emitted for a row that produced no audio, and no artifact is invented for
it.

A **short batch**, on the other hand, fails the job (`narrator_protocol`). One answer per row
is narrator's own guarantee — its comment on why is that "a row with no message hangs its
sentence until the 180s timeout taints the worker" — so a `batch_done` with rows unanswered
is a protocol failure and not a partial answer. So is a row answered twice, or a row answered
for an index nobody asked for.

**Refusals, all before the job is queued.** `unknown_model` (via `resolve_model`, section 5's
note), `invalid_params`, `ffmpeg_missing`, `backend_unsupported`, `env_missing`,
`voice_not_installed`, `accelerator_busy`, `insufficient_memory`, and four this door adds:

| code | what it means |
|---|---|
| `voice_kind_unsupported` | the voice is `kind = "zeroshot"` **and is not already resident**. A render job loads its own voice (below), a zero-shot load needs the reference clip only `load-voice` carries (`params.reference`, section 5's amendment), and this job's params are `language`, `take` and `chunks` — a second clip channel here would be two doors owning one fact. Load it first, then render. *Narrowed 2026-09-14; it used to refuse the KIND outright, on the true-at-the-time grounds that narrator's `load` message carried no clips at all.* |
| `sampling_not_wired` | the narrator ON THIS WIRE did not announce `itemTake` on its `ready` line, so it has no per-item rung channel and a take above 0 would come back as take 0 under take N's name. Asked of the live process, because the tts env pins narrator by commit and a pin may be older than the channel — on 2026-09-15 it was, and two takes of one sentence returned byte-identical audio. **Only above take 0**: take 0 asks for the numbers and the seed lane every narrator ever built already uses. *Its ORIGINAL meaning — "this contract has no channel" — was deleted on 2026-09-14 when `narrator/engine/item_sampling.py` made it false; the code and the name came back a day later with the subject above (sections 3 and 4).* |
| `unknown_take` | a take past the end of the ladder. Never clamped. |
| `chunk_too_long` | a chunk longer than the (voice, backend) `max_chars`. **Refused, not re-split**: chunking is the client's (section 1), and a server that quietly cut a chunk in half would return two files where one was asked for. |

`invalid_params` also covers two shapes worth naming: a blank `text` (narrator answers an
empty generate with a whole-request `error`, which would take the other rows with it) and two
chunks sharing an index (an index is an artifact name, so two would be one FLAC overwriting
another).

**Resume stays client-side.** BookForge already knows which `<index>.flac` files exist and
exceed 1024 bytes; it sends the chunks it still needs. The server has no session, no project
and no memory between jobs — DESIGN.md section 10, "no library, no project files, no
per-user state" — and resume is exactly the kind of state that would break it.

**The batch writer is the SDK's.** No shared mount, ever: `Z:` is invisible to WSL, which is
the whole reason whole-m4b alignment cannot run on this PC today. The SDK fetches each
artifact as its `artifact` event lands and writes `<index>.flac` where assembly and resume
look, overlapped with the next chunk's generation.

## 7. The streaming door — a session, an event stream, and posts

The Listen path, the in-app Play button, and the browser extension Owen uses every Sunday.
Its requirements are not the render door's with a smaller buffer; they are different in kind
(CLIENT-SURFACES.md row 18): sub-sentence audio emitted *while a row is still generating*,
rows retired out of order within a batch, and a cancel that aborts work in flight.

### Why this is not a WebSocket

It was, in the first draft of this file, and the draft was wrong for a measured reason.

**Node 20 has no global `WebSocket`** — it is behind `--experimental-websocket` there and
only becomes ordinary in 22. Electron 33, which is what BookForge ships, bundles Node 20.18,
and the SDK runs in the **main** process, where the renderer's browser `WebSocket` is not in
scope. Checked on this machine rather than assumed: `node -v` is v20.19.5 and
`typeof WebSocket` is `undefined`.

That leaves three ways to have a WebSocket and none of them is free. Raising the SDK's floor
to Node 22 does not help, because Electron's Node is Electron's. Adding `ws` breaks the one
rule the SDK has had since phase 1 — **zero runtime dependencies**, so it can be imported by
Node, bun and Electron without a resolution story. Writing an RFC 6455 client by hand is
about two hundred lines of masking, fragmentation, continuation frames, close codes and
UTF-8 validation, which is two hundred lines of subtle protocol in a client whose entire
job is to be boring.

So the door is built out of the two things this server already does well:

| | |
|---|---|
| `POST /v1/tts/stream` | opens a session → `{session_id, voice, fingerprint, sample_rate, backend}` |
| `GET /v1/tts/stream/{id}/events` | SSE: everything the server has to say, including the audio |
| `POST /v1/tts/stream/{id}` | one op: `say`, `cancel`, `cancel_all`, `close` |
| `DELETE /v1/tts/stream/{id}` | the same as `close`, for a client that only has verbs |

Every one of those is `fetch` and `ReadableStream`, which the SDK already uses for
`events()`. Nothing new is imported on either side.

The cost is that SSE is a text protocol, so PCM travels base64 — 33% over the wire. At
24 kHz mono PCM16 that is 48 KB/s of audio becoming 64 KB/s, which is nothing, and it is
**not a regression**: narrator already base64s its PCM over its own pipe, so the bytes
BookForge handles today are the same shape.

The gain is not just the dependency. `Last-Event-ID` **already works** on this server's SSE
streams, so a Listen connection that drops in a tunnel reattaches mid-sentence instead of
starting the row again — which a WebSocket would have needed its own machinery to do.

### The frames

Events on the stream, each with the usual strictly-increasing id:

```
ready   {voice, fingerprint, sample_rate, backend}
audio   {id, seq, pcm_base64, seconds}
restart {id, from_seq, reason}          # added while building — DIFFERENCE 1 below
done    {id, seconds, chars, chars_per_sec, capped, cancelled}
error   {id?, code, message}
closed  {reason}
```

Ops on the post:

```
{"op": "say",    "id": "r12", "text": "...", "take": 0}
{"op": "cancel", "id": "r12"}
{"op": "cancel_all"}
{"op": "close"}
```

Out-of-order retirement falls out of the shape: ids are the client's, `seq` counts within an
id, and `done` for one row may arrive while another is still emitting. There is no batching
parameter on the wire — how many rows the engine runs at once is engine tuning and belongs
to the server (Higgs measured worthless above width 1).

`say` returns **202 and the row's id**, not the audio. A client that wants the audio reads
the stream, and a client that never opened the stream is refused by name rather than
generating into nothing.

### Lifetime, and what a dropped connection means

A session holds the resident voice's attention, so it cannot outlive its client silently.

- The SSE stream sends a keepalive comment every 15 s, as the job streams already do.
- **A dropped stream does not cancel immediately.** It starts a 15-second grace window, and
  a reconnect with `Last-Event-ID` inside that window reattaches to the same session and
  replays what it missed. This is the one behaviour a WebSocket could not have given for
  free, and it is the difference between a tunnel costing a reconnect and costing a
  sentence.
- When the window closes, the session closes and every row still in flight is cancelled.
  Work nobody is waiting for is time stolen from the next job — the same rule as the `llm`
  proxy, for the same reason.
- A session is refused (`voice_not_resident`) if its voice is not the resident one. The
  streaming door never loads, exactly as chat never loads; only the render job does, and
  section 6 says why.

BookForge's own TTS WebSocket on 8766 stays exactly where it is and becomes a **relay** to
these three routes. The extension's protocol does not change, which is what keeps Sunday
working.

### What building it changed, 2026-09-13

Seven things this section did not know, each found by writing `crucible/ttsstream.py`
against `tests/fake_narrator.py` and then running it on a real socket.

**DIFFERENCE 1 — there is an eighth frame, `restart {id, from_seq, reason}`, and per-row
cancel is a lie without it.** The contract says `{"op": "cancel", "id": "r12"}` and narrator
has no such op: its `cancel` aborts **everything in flight**. Redefining the op to mean
"cancel all" would be a lie on the wire, and reporting one row as cancelled while its
neighbours silently died with it would be worse. So a row not yet handed to narrator is
dropped before it starts, a row in flight is stopped by aborting its batch, and the
**survivors of that batch are resubmitted**. A resubmitted row has already sent audio under
its id, and without a frame saying so a client concatenates the row's first seconds twice
with nothing on the wire to explain the stutter. `restart` is that frame: every `audio` for
that id below `from_seq` is void. `seq` never restarts across it, so "ids strictly increase
within a row" stays true and `from_seq` is where the good audio begins.

**DIFFERENCE 2 — a width is a MEASUREMENT, and there is a 25 ms coalescing window.**
This section's parenthetical above named a *ceiling*; the thing that dispatches is
BookForge's worker pool `flushBatch()`, which coalesces a 25 ms window into
`min(STREAM_RAMP_WIDTH = 8, streamBatchCeiling())` — so the number that matters is what the
ramp dispatches and not what the ceiling allows. `STREAM_BATCH_WIDTH` carries Higgs's 1 and,
since the ruling of 2026-09-14, nothing else; the second engine's row is a measurement
somebody takes, which is why `batch_width_for` refuses rather than defaulting. The window
came with it and was found
by a failing test rather than designed in: rows arrive one HTTP post at a time, so a worker
that dispatched the instant the first one landed put **every row in a batch of its own** —
the width would have been a number that never happened, and the whole cost of a per-row
cancel would have looked free right up until the day it was not. The wait is skipped
entirely when the width is 1 rather than added to the first syllable of every sentence.

**So the cost of a per-row cancel, stated plainly:** on `higgs-v3`, nothing — the width is 1,
the in-flight row IS the batch, there are no survivors and `restart` never fires. On an
engine with a width of N, up to N-1 other rows lose whatever they had generated and generate
it again. `higgs-v3` is the only engine this build names, so the survivor path is
**unreachable through a manifest today**; its test patches the width to reach it and says so.

**DIFFERENCE 3 — a session and the exclusive lane need a mutual exclusion, and this section
did not say so.** A streaming session is a connection rather than a job, so it does not queue
behind the lane — and narrator has one stdin and one stdout. Two conversations on that wire
do not collide loudly; they read each other's `batch_item` lines and deliver a row of audio
under another row's id, and a `load-voice` arriving mid-session would SIGTERM the engine out
from under a sentence. So the card now has a named owner: `Residency.claim(holder,
may_mutate=)`. A session claims it for its lifetime and promises never to load; the render
door claims it for its batch and may load its own voice, on the thread it claimed on;
`load`, `load_voice`, `load_aligner` and `unload` refuse **`engine_in_use`** to anybody else.
The card-touching job types make the same refusal in `preflight`, so a client is told
before its job is queued rather than watching it fail in the lane.

> **Amended 2026-09-13 (ARCHITECTURE.md section 3.1).** This said "all five", counting
> `load-model`, `unload-model`, `load-voice`, `unload-voice` and `tts`. It is **seven**:
> `align` and `unload-aligner` were added with the admission ruling, having been the third
> mutator of residency since phase 4 without ever asking — so an align job submitted under
> an open session was accepted and then failed a minute later at
> `_refuse_mutation_if_claimed`, which is the exact thing this paragraph says does not
> happen. `asr` and `rvc` still do not ask, deliberately: they never touch the resident
> engine, and their contention is memory, which `accelerator.guard` refuses by name.
>
> A count in prose is a fact with two owners (ARCHITECTURE.md rule R1), which is why it is
> now stated as a list rather than a number.

**DIFFERENCE 4 — the replay buffer is bounded by the window, not by a count, and a resume it
cannot serve is refused.** A frame is dropped when it is older than the grace window **and**
every attached reader has been handed it. A count would either be a number of seconds written
as a number of frames — wrong the moment the engine's chunk size changes — or big enough to
hold a book in memory at 64 KB/s of base64. A `Last-Event-ID` below what the session still
holds is refused as **`replay_unavailable`**, naming the oldest id it has, and never skipped
past: audio with a hole in it and nothing saying so is the exact failure this door exists to
prevent.

**DIFFERENCE 5 — `say` carries a required `take` and there is no default on the wire.** The
SDK's `say(id, text, take?)` defaults it to 0 in the caller's own code, which is a client
choosing; a default in the request body would be the server choosing, and now that the ladder
is wired that would be a render at a take nobody asked for. **Rows in one session may be at
different takes** — each `say` resolves its own rung and each item carries its own `sampling`
— which is what spreading candidates across the ladder looks like on this door. The refusals a `say` can make are the
render door's, one row at a time: `unknown_take`, `chunk_too_long`
(refused, never re-split), a blank text, plus `duplicate_row_id` — an id is what every frame
names its row by, so two rows sharing one would be two streams of audio under one name.

**DIFFERENCE 6 — the refusals this door adds.** `stream_session_open` (a second session, named
with the first's id), `stream_not_attached` (a `say` on a session whose event stream has never
been opened — the contract's "refused by name rather than generating into nothing", now with a
code), `unknown_session`, `unknown_row`, `stream_closing`, `replay_unavailable`,
`engine_in_use`, and `unknown_narrator_engine` (no measured batch width for an engine nobody
has measured one for; a guess is wrong in both directions — too low halves throughput, too
high multiplies a cancel's cost). `cancel` answers `{"outcome": "dropped" | "aborting_batch" | "already_finished"}`,
because those are three different costs and a client is entitled to know which it got —
`already_finished` in particular is the ordinary race on a live connection and not an error.
A row that fails on its own gets `error {id, code: "row_failed", message}` and is **never**
resubmitted; that is also what makes resubmission self-limiting, since a row that genuinely
fails comes back to a batch with no cancel in it and is reported there.

**DIFFERENCE 7 — `done`'s `chars_per_sec` is nullable.** A row that was cancelled before it
emitted anything has no rate, and 0.0 would read to a pace guard as an infinitely slow
narrator. `capped` is `null` against the pinned narrator for section 6's reason, and `null`
still never means `false`.

**One measurement about the drop itself, because the first version of its test was wrong.**
A real socket close is carried to the SSE generator by starlette's disconnect listener and
ends it in about **0.17 s**; the 15 s keepalive is the detector for the *other* kind of
departure, a tunnel that died without closing anything, where only a write that fails can
discover it. A shorter `is_disconnected()` tick was written, measured to change neither case,
and taken back out. The reason this had to be measured at all is that `httpx.Response.close()`
called from another thread while a reader is inside `iter_lines()` does **not** close the
socket, so the first reattach test was not testing a reattach — it was opening a second reader
beside a first that had never left. The test now does a real `shutdown(SHUT_RDWR)`.

## 8. API additions

| Route | Auth | Returns |
|---|---|---|
| `GET /v1/voices` | yes | the rows in section 2 |
| `POST /v1/jobs {type: "load-voice", model}` | yes | a job; `warming` while narrator starts, `done {resident: id}` |
| `POST /v1/jobs {type: "unload-voice", model}` | yes | a job; `done {resident: null}` |
| `POST /v1/jobs {type: "tts", model, params}` | yes | a job; `chunk` per chunk, `artifact` per FLAC |
| `POST /v1/tts/stream` | yes | **201** and `{session_id, voice, fingerprint, sample_rate, backend}`, section 7 |
| `GET /v1/tts/stream/{id}/events` | yes | SSE: `ready`, `audio`, `restart`, `done`, `error`, `closed` |
| `POST /v1/tts/stream/{id}` | yes | **202** and one op: `say`, `cancel`, `cancel_all`, `close` |
| `DELETE /v1/tts/stream/{id}` | yes | close the session; `{session_id, closed}` says whether the wire was down by the time it answered |

`GET /v1/info` gains a `tts` capability whose rows are `/v1/voices`' rows verbatim.
`GET /v1/health` gains `resident_kind`.

One consequence of the render door that `llm` does not have: **`tts` is both a capability name
and a job type name.** `llm` is only a capability name — the types you POST are `load-model`
and `unload-model` — so appending its capability could not collide with anything. The render
door's type is literally `tts`, so `/v1/info`'s registry loop already produces a `tts`
capability from `describe_models()`. The voice rows **replace** it rather than being appended
beside it: two capabilities under one name describing one set of voices in two shapes is
exactly the reconciliation this rule exists to prevent.

`crucible doctor` gains `narrator_patches` — one row per site-packages patch, with its
status, the file it edits and what breaks without it (section 4).

`GET /v1/voices` is refused with `job_type_disabled` when `[jobs] enable_tts` is false, the
same way `/v1/models` is for `llm`, and `/v1/info` simply carries no `tts` capability there.

The CLI gains `crucible voices list` and `crucible voices pull <id>` beside `crucible models`
(the same rows, answering "what is on this disk" rather than "what can this server be asked
for"), and `crucible install tts --narrator-engine <engine>`, whose choices are
`voices.NARRATOR_ENGINE_SAMPLING`. The flag is required for `tts` and refused for `llm`:
`cuda-linux` names one venv per narrator engine and `mlx-darwin` has one for every engine,
so the command may not pick for you — and it stays required with one engine in the list,
because the alternative is a default that silently becomes the wrong engine the day there
are two. `crucible doctor` reports
one env row per narrator engine under `tts_envs`.

## 9. SDK additions (`@crucible/client`)

- `voices()` → `VoiceInfo[]`, `loadVoice(id, {reference?})` / `unloadVoice(id)` → job ids.
  The row carries `takes` (how many rungs, so a client can ask before it submits) and
  `needsReference` (whether a load must carry a clip); `reference` is
  `{data, transcript, name?}` and is validated client-side for the two required fields
  before a round trip (section 5's amendment).
- `render({voice, language, take, chunks, signal})` → **a job id**, not a handle; `chunk` is
  in the event vocabulary, and `writeArtifactsTo(jobId, dir)` is the batch writer of section
  6. See "What the render client deviated from, and why" below for the handle.
- `stream({voice, language})` → a session: `say(id, text, take?)`, `cancel(id)`,
  `cancelAll()`, `close()`, and an `AsyncIterable` of `{id, seq, pcm: Int16Array}`
  interleaved with `{id, done}` (and, since the door was built, `{id, restart}`).
  Still zero runtime dependencies, and now genuinely so: it is `fetch` and the same SSE
  reader `events()` already uses, on a runtime that has no `WebSocket` (section 7).
- Typed refusals for every named code in this document.

### What is built, 2026-09-13

The lifecycle half, matching the server's: `voices()`, `loadVoice(id)`, `unloadVoice(id)`,
and the type `VoiceInfo` / `VoicePace` read strictly from the section 2 row. The SDK's unit
suite goes from 52 tests to 84, none of which needs a live server.

**`render()` and `stream()` are deliberately not built.** Sections 6 and 7 are still being
written in Python — `crucible/engines/narrator.py` does not exist and `load-voice` reports
`engine_not_implemented` — so their wire is not settled. A client written against a contract
that may still move is how the two halves end up disagreeing, and the disagreement would be
silent. They are a follow-up, and `chunk` is deliberately **not** in the SDK's event
vocabulary until then: a `chunk` frame today is a `CrucibleProtocolError` naming it, which is
the correct answer from a client that does not yet speak it.

### The render client, 2026-09-13 (later the same day)

The render door landed on `main`, so `render()`, the `chunk` event and the batch writer are
built against the bytes `crucible/jobs/tts/render.py` actually sends. The SDK's unit suite
goes from 84 tests to 123, still none of which needs a live server: a `node:http` fixture
answers the event stream and the artifact route, and a temporary directory takes the files.
`stream()` remains unbuilt and is another builder's.

Two of those tests are timing proofs rather than shape proofs, and they are the ones that
matter. "Fetches each artifact as its event lands" **gates the SSE stream on the artifact GET
arriving**, so a writer that waited for `done` deadlocks it rather than failing an assertion
— which is the honest shape of "this turned a streaming server back into a batch one". It
carries a timeout for exactly that reason. Both it and the concurrency ceiling were
mutation-tested: breaking the behaviour fails the test.

**What the render client deviated from, and why**

- **`render()` returns a job id, not "a job handle".** Every other queueing call in this
  client returns an id and `events()` is how a job is watched; a handle with its own iterator
  would be a second way to watch one job, and two clocks on one stream is how they disagree.
  `writeArtifactsTo(jobId, dir)` is a method beside `events()` rather than a method on a
  handle, and it *is* `events()` — it yields the job's own events unchanged, interleaved with
  the files it has written. That is the "without swallowing the job's own events" requirement
  answered in the type rather than in a second callback channel.
- **`submit()` grew an options bag** (`submit(request, {signal})`), because a `tts` body can
  be a whole book's text and the caller needs a handle on that POST. The signal aborts the
  submit and nothing else: once the server has answered with an id the job exists, and
  `cancel(id)` is what stops it.
- **`writeArtifactsTo` bounds its fetches** (default 4, `concurrency` to change it).
  `events()` replays a job's whole history before it follows live, so attaching to a
  nearly-finished 1,400-chunk render delivers 1,400 `artifact` frames in one burst;
  unbounded, that is 2,800 sockets in a tick, against the server that is still rendering.
- **It writes the provenance sidecar first, then the artifact.** DESIGN.md section 7 says a
  client must persist provenance beside the output; this order makes the existence of
  `<index>.flac` imply the existence of `<index>.flac.provenance.json`. The other order can
  leave a finished chunk that cannot say which voice, which revision or which server made it
  — and resume, which only looks at the FLAC, would never ask for it again. The sidecar's
  **bytes** are the server's own, not a re-serialisation: the document is meant to be
  persisted, and round-tripping it through this client's reader would rewrite whitespace it
  did not author. It is still parsed, so a sidecar that is not a provenance document is a
  protocol error and nothing lands.
- **Resuming with `lastEventId` writes only what arrives after it.** With the whole history
  replayed, `done`'s `artifacts` list is reconciled against what was seen, so a name there
  that produced no `artifact` frame is still written. With a `lastEventId`, it is not: the
  prefix the caller chose not to replay is what they already had when they recorded that id,
  and re-fetching a finished book's worth of FLACs on every reconnect is a worse bug than the
  one it would guard against. **This is a judgement call and it is the one worth arguing
  with.**
- **`node:fs/promises` and `node:path` are loaded from a specifier assembled at run time,**
  inside the one method that writes files. Node's builtins are not a dependency in the sense
  the zero-dependency rule means — nothing is installed to get them — but a *static* import
  would put fs into the module graph of `import {CrucibleClient}` itself, which a bundler
  targeting a browser-ish runtime resolves at build time and fails on. Hiding the specifier
  costs the import's types, so the four fs calls and the one path call the writer makes are
  declared as an interface and checked at the seam; a runtime without them gets a
  `CrucibleError` saying what is missing, not a `TypeError` about `undefined`.
- **`maxChars` is not re-checked client-side.** The cap is per (voice, backend), it rides on
  the voice row, and a copy of it here would be a second thing to drift — the same reasoning
  that keeps faster-whisper's language list out of `asr()`. What *is* checked client-side is
  what is a fact about the request rather than about the server: a duplicate index (an index
  is an artifact name, and two would also be two writes racing for one path in the caller's
  library), a blank chunk, a non-integer take.
- **`capped: boolean | null` rather than a wrapper that forces narrowing.** A shape like
  `{said: false} | {said: true, capped: boolean}` would make the compiler refuse
  `if (chunk.capped)`, and it was rejected: it deviates from the wire, and
  `AcceleratorHolder.bytes` already models the identical hazard — a null that must never be
  read as zero — as `number | null`. One hazard, one shape. What the type *does* enforce is
  the other half: the reader refuses a `chunk` frame that **omits** `capped`, because "narrator
  did not say" has to be something the server said rather than something the client inferred
  from an absence.

**One gap this found, and fixed.** `done` is built by the queue as
`{"artifacts": [...], **job.done_extra}`, and the client read the two keys it modelled and
**silently dropped the rest**. That lost `tts`'s `failed: [{index, message}]` — which section
6 says in as many words is how a client reading only the terminal event learns which indices
to ask for again, and it never arrived — along with `rendered`, `take`, `sample_rate`, and
`load-voice`'s `fingerprint` beside its `resident`. `DoneData.extra` now carries every other
key verbatim, exactly as `ProgressData.extra` already carried a progress frame's own
measurements, and `readRenderResult(done)` reads a render's terminal news strictly out of it.
It is the same class of bug as `info()` breaking on `enable_tts`: a shape the document
described and the client did not read.

**One contradiction found and left for a ruling.** Section 6 says "The SDK yields unknown
event kinds through rather than dropping them, so that stays true for the next one too." **It
does not, and never has.** `readEvent` narrows the event name against a closed list and
raises `CrucibleProtocolError` on anything else, and `sdk/ts/src/errors.ts` states the
opposite doctrine in as many words: "A new event kind is a breaking change and would come with
a new `api_version`, so this is never absorbed quietly." Both cannot be true, and the cost of
the code's version is concrete: `chunk` was added without moving `api_version`, so an
0.2.0 client watching any job on a server that has learned a new event kind loses the whole
stream — a bug waiting for the Mac, and for phase 4's `align` and `rvc`. It is deliberately
**not** changed here: it is a doctrine call, not a bug fix, and `readEvent` is shared with the
builder adding `stream()`.

### The streaming client, 2026-09-13 (later still)

`stream()`, `src/stream.ts` and `decodeBase64`. The SDK's unit suite goes from 128 tests to
146, still none of which needs a live server: a `node:http` fixture writes the frames and
then destroys the socket under the reader, which is the only way to prove the reattach.

The session object **is** the `AsyncIterable`, which is the one place this differs from
`render()`'s shape and for the opposite reason to it. A job is watched with `events(jobId)`
because a job outlives any watcher and two clocks on one stream would disagree; a session
*is* its stream — it has no life without one, and `say` is refused by name until one is open
— so an id plus a separate reader would be two halves of one object that cannot be used
apart.

Four decisions in it are worth arguing with:

- **It reattaches on its own, and `events()` does not.** That asymmetry is deliberate. A job
  goes on running whether anybody is watching and its event log is kept for the life of the
  job, so a client can resume whenever it likes; a session's grace window is fifteen seconds
  and missing it costs the session, the rows in flight and the listener's place in the
  paragraph. So a dropped stream is reattached with `Last-Event-ID` for as long as the window
  could still be open — which is the behaviour the server's SSE door was chosen for, and
  which every caller would otherwise write again and get wrong. A **refusal** is never
  retried: `unknown_session` and `replay_unavailable` are the server saying the window has
  closed or that the audio would have a hole in it, and trying again quietly is how a client
  ends up playing a sentence that is missing its middle.
- **A read that fails is sorted from a frame that cannot be read.** Anything the client
  raised — an unknown event kind, an id that does not follow, a session-wide `error` —
  travels straight back; a socket that died under the reader is the drop this door exists to
  survive and is reattached. A consumer breaking out of its `for await` lands in neither,
  because a return completion at a `yield` runs `finally` and skips `catch`.
- **`decodeBase64` refuses malformed input rather than skipping it**, unlike `atob` and most
  hand-rolled decoders. One character outside the alphabet would otherwise shift every sample
  after it, and a session's chunks are concatenated with their neighbours — so "quietly a few
  samples short" is a click in the middle of a sentence that nothing in the pipeline would
  explain.
- **PCM16 is read through a `DataView`, not `new Int16Array(bytes.buffer)`.** The wire is
  little-endian — narrator's own format, and what `-f s16le` tells ffmpeg on the render door
  — and the cast would read it in the host's byte order, which is right on x86 and arm64 and
  silently wrong anywhere else; it also throws outright on an odd byte offset, which a decoded
  base64 buffer is free to have.

`chunk` stays out of this door's vocabulary and `restart` is new to it, so the contradiction
above about unknown event kinds is untouched and still owed a ruling.

Four things this section did not say, each found by reading the bytes the server actually
sends rather than the prose:

- **`info()` was broken on any server with `enable_tts`.** Section 8 says the `tts`
  capability's rows are `/v1/voices`' rows verbatim, and the client was reading every
  capability but `llm` with DESIGN.md section 4's descriptor — which demands a `source` and
  a `vram_bytes` a voice row does not carry. `Capability` is now a three-way union with
  `isTtsCapability` beside `isLlmCapability`, on the same rule: the capability whose rows
  come from a listing route is read with that route's reader.
- **A voice row's `reason` is always present, `null` when loadable**, where a model row omits
  the key entirely (`voice_rows` writes `"reason": reason` unconditionally; `model_rows`
  writes it only `if reason is not None`). The client reads each as it is rather than tidying
  the difference away. The rule that does not differ is the one that matters: not loadable
  and no reason is a protocol error.
- **`kind` and `estimate_basis` are narrowed to their manifest vocabularies**
  (`checkpoint | zeroshot | token`, `measured | declared`), because the loader refuses any
  other word — so a third value is a contract change, not a value to pass through.
  `resident_kind` is **not** narrowed, for the opposite reason: section 8 says the set grows,
  and phase 4's aligner lands at the same key.
- **`sample_rate`, `takes` and `pace` survive a missing backend block.** They are facts about
  the voice, not about this host, and `voice_rows` sends them on an unsupported row too; only
  the five backend-block fields go null together.

One thing the server sends that this document does not describe, found the same way and left
for the Python side: **`GET /v1/accelerator`'s `resident` block hard-codes `"kind": "llm"`
and reads `resident.model_id`** (`crucible/api.py`), which `ResidentVoice` does not have. With
a voice on the card that route raises rather than reporting it. The probe's own contract
(PHASE4-AUDIO.md section 5) already says the kind is the family of the resident thing and
that voices land there, so this is the route not having caught up with section 5's
generalised residency.

## 10. Verification

No new number is written into a manifest without the accelerator it was measured on, and
both of Owen's cards were busy the night this was written. So the order is: build against a
**fake narrator** — the same shape as `tests/fake_engine.py`, a script that speaks the
JSON-lines protocol and emits a sine wave — prove every wire behaviour in pytest, and leave
exactly one owed item.

**Owed, and it must be done before BookForge renders a book through this:** on the PC, load
`deathstalker` on `cuda-linux`, render a chapter, measure the real VRAM and write it into
the manifest, and confirm the audio is identical in format to what
`narrator.compat.worker` produces today. On the Mac, the same through MLX. Then the
streaming door with the browser extension pointed at BookForge's relay — the Sunday test,
and the only one that matters to Owen personally.

### What is built, 2026-09-13

Sections 2, 5 and 8's lifecycle half: `crucible/voices.py`, `voices/*.toml`,
`crucible/residency.py`, `crucible/jobs/tts/`, `GET /v1/voices`, the `tts` capability,
`resident_kind`, and the CLI. 125 tests, every refusal path among them; `python -m pytest`
goes from 155 to 280. Sections 6 and 7 — the render door and the streaming door — are not
built, and `crucible/engines/narrator.py` does not exist: `build_voice_engine()` raises
`NotImplementedError` naming that file, `load-voice` reports it as
`engine_not_implemented`, and a test asserts that it refuses rather than pretending.

### What is built, later on 2026-09-13: the engine and the render door

`crucible/engines/narrator.py` (section 4), `crucible/jobs/tts/render.py` (section 6), the
`stdio()`/`attach()`/`detach()` seam in `SubprocessEngine.start()`, the three `envs/tts/`
recipes, `crucible/narratorpatches.py` and the `narrator_patches` row in `crucible doctor`.
The helpers `load-voice` and the render door share moved to `crucible/jobs/tts/common.py`;
nothing about their behaviour changed in the move. `python -m pytest` goes from 406 at the branch point to 469 — 61 of those are this
branch's, and `origin/main` gained the rest while it was being built.

Section 7, the streaming door, is still unbuilt. What the engine seam **gives** it: the
process lifetime, `send()` for an out-of-band op while a request is in flight, and
`converse()` yielding `batch_chunk` lines in arrival order as they land, which is exactly the
sub-sentence stream it needs. What the seam does **not** give it: any session, any id
allocation, any `Last-Event-ID` replay buffer, any `cancel` scoped to one row rather than the
whole engine (narrator's `cancel` aborts what is in flight, and there is no per-row cancel on
its wire), and no route — all four of those are section 7's own work.

`envs/tts/` also settled how a recipe pins something that is not on PyPI. narrator is a **PEP
508 direct reference** carrying a 40-character sha; `jobenv.recipe_pins` skips those (they
are exact pins, just of a commit) and `jobenv.recipe_direct_references` checks them against
the commit pip actually recorded in **PEP 610's `direct_url.json`**, because `pip list`
reports narrator's declared version and that does not move when the sha does. A direct
reference naming a branch instead of a sha is refused: a branch is not a pin. **The sha
itself is not restated here** — it lives in the three `envs/tts/*.txt` recipes, and
`test_every_tts_recipe_pins_the_same_narrator_commit` is what keeps those three copies one
fact. A fourth copy in prose is a copy nothing compares.

Three deliberate gaps, each refused by name rather than worked around, and each with what is
owed on narrator's side written beside it: sampling and the take ladder (section 4), zero-shot
clips (section 6), and `capped`/`tokens` on the `chunk` event (section 6).

**A second owed item, and it is about the pins rather than the numbers.** A voice manifest
pins the HuggingFace revision a `crucible voices pull` will actually fetch. For three of the
five fine-tunes that is NOT the merge BookForge's catalog measured its pace and band against,
because those merges live in the WSL guest and were never uploaded — deathstalker's repo
holds the 2026-09-06 `ds_ad4lm_prod_ckpt1080` while the PC serves `ds_v7_930_prod`;
mistborn's holds `mb_h2lm` from 2026-09-08 while the pace and band were measured on
`mb_full_rvc1_5947_prod` on 2026-09-12; owen's holds the 2026-09-07 upload while the arms
serve `ow_v7_prod`. `sigma` and `thirdreich` agree. The caps survive the gap because they are
rulings over the whole family rather than certificates bound to one directory; the **pace
bands do not**, and each manifest says so in a comment. Either push the current merges (the
catalog's notes say Owen has not given the green light) or re-measure against the pins.

### What is built, later still on 2026-09-13: the streaming door

`crucible/ttsstream.py`, the four routes in `crucible/api.py`, the exclusive claim in
`crucible/residency.py`, and `stream()` in the SDK. `python -m pytest` goes from 616 at the
branch point to 645 — 29 of those are this door's — and the SDK's unit suite from 128 to 146.

Three things that were owed a measurement and got one, on a real uvicorn and a real socket
(`tests/test_tts_stream.py` runs the whole server rather than the ASGI app, because a caller
who walks away from a stream is a state `TestClient` cannot reach):

- **The grace-window reattach works, and the proof is arithmetic.** A row is generating, the
  socket is killed with `shutdown(SHUT_RDWR)` mid-row, a new stream attaches carrying
  `Last-Event-ID`, and the two halves concatenate in `seq` order into exactly the row's whole
  audio — no gap, no repeat, no seq seen twice, and the frames that landed while nobody was
  listening are still there.
- **A real drop is carried to the SSE generator in about 0.17 s** by starlette's disconnect
  listener; the 15 s keepalive is the detector for a tunnel that died without closing
  anything. The grace window therefore starts when the client goes, not a keepalive later.
- **Per-row cancel costs nothing on every voice that ships.** All seven manifests declare
  `narrator_engine = "higgs-v3"`, whose measured width is 1, so the in-flight row is the
  batch and there are no survivors to resubmit. The cost on a wider engine — up to N-1
  rows regenerated — is proved by a test that patches the width, and it says so.

**What is still owed, and it is the only one Owen cares about personally:** the Sunday test.
The browser extension pointed at BookForge's relay, pointed at these four routes, on the real
narrator with `deathstalker` on the card. Nothing here has spoken to a GPU.

Two smaller owed items this door added. `GET /v1/health` says nothing about an open session,
so a client cannot ask "is somebody listening" without trying to open one and being refused;
and BookForge's relay on 8766 is unwritten, which is where the extension's protocol meets
these routes.

## 11. What this deliberately does not do

- **No assembly.** Crucible returns chunks; the m4b is BookForge's.
- **No chunking and no text normalisation.** Section 1.
- **No retake decision.** Section 3. The server owns what take N *means*; the client owns
  *that* a row needs another take and *which* take it keeps.
- **No voice creation from a clip.** A zero-shot load conditions the base weights on a
  recording for as long as that voice is resident (section 5's amendment); it does not
  publish, name, store or remember a clip. The next load overwrites the wav, exactly as it
  overwrites the voices document, and for the same reason.
- **No session.** Section 6.
- **No voice creation.** Training a checkpoint and publishing it to HuggingFace is
  `orpheus-finetune`'s job and stays there. Crucible pulls a published voice at a pinned
  revision, exactly as it pulls a model.
