# Phase 9: a server declares what it can actually do

**Owen, 2026-09-13:** *"if i run crucible on a 6 gb gpu, it should use quantized 9b to fit on
that card. thats something we can configure on crucible install — picking which models and how
quantized those models are for each server install"* — then, narrowing it:
*"higgs is tied to a certain size. we cant (or wont) quantize that. if higgs doesnt fit in a
card that crucible is installed on, it's disabled on that gpu."*

This phase is that ruling, what it costs, and the one thing that has to be measured before any
of it can be trusted.

---

## Status, 2026-09-13

**BUILT — the selection step (section 2) and the refusal (section 2.1).**

| what | where |
|---|---|
| the selection rule itself | `crucible/capability.py` — `CLASSES`, `decide`, `decide_all`, `job_type_enabled` |
| the record in config.toml | `crucible/config.py` — `CapabilityRow`, `CapabilityRecord`, `[capability]` + `[[capability.classes]]` |
| the refusal that names the number | `crucible/jobs/__init__.py` — `disabled_error`, used by `resolve`, `/v1/models`, `/v1/voices` and the streaming door |
| install's selection step | `crucible/cli.py` — `_capability_step`, called by both install branches |
| a door of its own | `crucible/cli.py` — `crucible capability [--write] [--json]` |
| doctor | `crucible/cli.py` — `_capability_report`: stale record, contradicted flag, a fitting type not yet installed |
| the tests | `tests/test_capability.py`, 32 of them, starting with the two answers Owen already runs |

Three decisions the build made that this document did not state, each argued at length in
`crucible/capability.py`'s module docstring:

1. **A capability class, not a job type, is the unit of selection.** `enable_llm` is one
   boolean and the ruling needs three — Owen's *"translation is binary per server"* cannot be
   said by a flag that also covers cleanup. So `llm` is `clean` + `translate` + `pages`, the
   flag is the disjunction, and the per-class rows carry the verdicts. Without this, a host
   that cleans and cannot translate has nowhere to say so, and "translate is disabled" would
   survive only as prose inside a reason string — which R4 forbids from being load-bearing.
2. **Candidates are ordered by declared size, descending.** Not by a new `precision` key in
   the manifests: `memory_bytes_estimate` already answers "how heavily is this quantized"
   within a family, and a second field saying it in other units is a second owner of one fact
   (R1). The order comes out right on both backends — 56.4 > 21.6 on cuda, 55.5 > 33.9 on mlx.
3. **There is no `margin` term.** Section 1.3 writes the test as `estimate + margin ≤ total −
   allowance` and section 1.2 answers what `margin` is: it is `desktop_allowance_bytes`. A
   second reserve on top of the first is a number nobody has measured, and on the 3090 Ti it
   would disable translate at 20.1 GiB against a 21.0 GiB budget — a card that has been
   translating for months. That is section 1.1's failure repeating itself, so the fit test is
   `estimate ≤ total − allowance` and the two known-good answers are asserted as tests.

**One thing in this document is WRONG and the build does not implement it.** Section 3 calls a
6 GB box *"an `llm` + `rvc` + `asr` server"*. Section 1.1 point 2 says the opposite and is
right: there is no 4-bit 9B in this build and no 27B small enough, so **a 6 GB card has no
`llm` capability at all** — `clean`, `translate` and `pages` all fail and the flag goes with
them. `test_a_six_gig_card_keeps_llm_only_if_something_behind_it_fits` is that correction.
`rvc` and `asr` survive, exactly as section 3 says.

**STILL OWED**

- Section 3's typed slot in BookForge's router, and a pin validated when it is made.
- Section 5's deletion of `electron/orpheus-memory.ts`'s tier table.
- Section 4's measurement is **DESCOPED, not deferred** — Owen, 2026-09-13: *"I don't think we
  need to measure estimates. I've been using this system the way it is for months and it works
  fine. Use the current settings for each."* Selection runs on the DECLARED estimates. There
  is deliberately no `measured` vs `declared` distinction in `capability.py`, because nothing
  would consume one.

**BUILT, 2026-09-14 — the local form (section 7).** Owen's ruling, via Foundry: these manifests
are the catalog of record for Foundry's LOCAL lineup too. A `[local]` table on a model
manifest, `crucible/lineup.py`, `scripts/gen-foundry-lineup.py [--check|--verbose]`,
`foundry-lineup.json` at the repo root (committed, generated, compared by content in CI and in
`tests/test_lineup.py`). One amendment to the bullet above: `[local]` DOES carry a
`measured` / `declared` distinction (`needs_basis`), because that table now has a consumer —
Foundry's picker draws the basis so it can err on the side it wants.

---

## 1. The ruling, and why it is smaller than it looks

**Owen, 2026-09-13, refining it:** *"translation is binary per server as well. it should use a
27b to translate. if 27b doesnt fit on the card then it cant translate. however, the mac can
handle 27b quantized to 8 bit i believe, and cuda can only handle it quantized to 4 bit… ideally
crucible will adapt to what the system can handle."*

An earlier draft of this section split job types into "binary" and "graded". **That was the
wrong axis**, and the ruling above is the right one. There are two decisions here, not one, and
they have different owners:

| Decision | Owner | Why |
|---|---|---|
| **which model FAMILY a task requires** — translate needs 27B-class, cleanup needs 9B-class | **the client** | a quality requirement about the work, not about the card |
| **which QUANTIZATION of that family this host runs** — 4-bit here, 8-bit there, bf16 elsewhere | **Crucible** | how the engine realizes the model; `crucible-division-of-knowledge`'s "tuning is Crucible config, never a wire field" |
| **no quantization of the required family fits** | **Crucible** | the job type is disabled, with the number that disabled it |

So every job type is "binary in its family, graded in its quantization". `tts` looks binary only
because Higgs has exactly one quantization — Owen: *"higgs is tied to a certain size. we cant
(or wont) quantize that."* It is the same rule with a family of one.

**The interface that falls out of it:** the client asks for a **capability class** and the
server answers with what it has. BookForge says "a translate-class model"; Crucible answers
`qwen3.8-27b-4bit`, or refuses with the number. Today BookForge names a model id outright
(`--model qwen3.5-9b`) and Crucible checks residency — so asking by class is an ADDITION to the
wire, not a replacement, and nothing that names an id breaks.

### 1.1 The matrix, as the manifests actually stand

Read off `models/*.toml`, not estimated:

| manifest | `cuda-linux` | `mlx-darwin` |
|---|---|---|
| `qwen3.8-27b` (bf16) | 56.4 GB | 55.5 GB |
| `qwen3.8-27b-4bit` | 21.6 GB | **33.9 GB** |
| `qwen3.5-9b` (bf16) | 20.9 GB | 20.4 GB |
| `qwen3.8-27b-8bit` | **does not exist** | **does not exist** |
| `qwen3.5-9b-4bit` | **does not exist** | **does not exist** |

Three things this table says that the ruling assumed otherwise:

1. **No 8-bit 27B is needed. Owen, 2026-09-13:** *"quantized to 4 bit on the mac is fine."* So
   the Mac translates on `qwen3.8-27b-4bit`'s mlx block — 33.9 GB, comfortable in 64 GB — and
   the `qwen3.8-27b-8bit` manifest owed since phase 2 is **struck, not deferred**. Worth keeping
   the reason it was never obviously worth building: MLX's 4-bit 27B is **33.9 GB against CUDA's
   21.6**, so an 8-bit would have landed near bf16's 55.5 and bought very little.
2. **`qwen3.5-9b-4bit` does not exist either**, and a 4-bit 9B is ~5-6 GB of weights before KV
   cache. A 6 GB card is therefore disabled for cleanup as well as translate — not a
   configuration to build, a refusal to word well.
3. **The probe needs nothing new.** An earlier draft of this document proposed adding a
   compute-capability / tensor-core field, on the strength of an example Owen used in passing.
   He corrected it: *"tensor cores was an example. we dont need to measure that. its really about
   what will fit and how well. we have to resize some things and disable others to make it fit
   on whatever server it's running on."*

   That is the whole rule, and `accelerator.py` already answers it — `total_bytes` is the
   question. **Do not grow the probe for this phase.** A capability axis nothing selects on is a
   field that will drift, and the reason to add one has to be a rule that needs it, not an
   example that mentioned it.

### 1.3 The selection rule

For each job type, on this backend:

1. Take the model family the task requires.
2. Walk its variants **best-precision first** — bf16, then 8-bit, then 4-bit.
3. **Take the first that fits.** Fits means
   `memory_bytes_estimate + margin ≤ total_bytes − desktop_allowance_bytes`.
4. **If none fits, the job type is disabled**, and the refusal records the best candidate and
   the shortfall.

"Resize some things and disable others" is exactly steps 3 and 4. Two things the order is
deliberate about:

- **Best-first, not smallest-that-fits.** Quantization is not free — a 4-bit translate is a
  worse translation than an 8-bit one, and this rule is about the best output the host can
  actually hold, never the most that can be crammed onto it.
- **`margin` is the thing that stops this being arithmetic on paper.** An estimate that exactly
  equals available memory does not fit; it OOMs on the first long context. Section 1.2 is that
  case in the concrete, on the machine this was ruled on.

#### What "resize" means, and what it does NOT mean

**Owen, 2026-09-13:** *"translate works just fine on the pc right now. we just have to have a
smaller kv cache. which doesnt matter, because we translate one block at a time. basically one
paragraph at a time."*

A draft of this section treated a model's context ceiling as a capability and concluded that
moving translate to Crucible on the PC would be a **6x regression**, because Owen's Ollama tag
`qwen3.8:27b-24g` carries `num_ctx 98304` while vLLM on the same card measures a ceiling of
**18,811 tokens** (KV at 86,251 B/token against a 1.51 GiB pool, from the 2026-09-12 probes).
That conclusion was wrong, and the way it was wrong is the useful part:

> **A context ceiling is not a capability. What matters is the context the WORKLOAD sends.**

Translation is paragraph-granular by design — `foundry/src/translate/run.ts`: *"paragraph-sized
inputs translate well, and page fragments are catastrophic… its rows are whole paragraphs."* One
paragraph against an 18,811-token window is not a constraint. The 98304 on the Ollama tag is
what the Modelfile says, not what the work uses.

**And Foundry already does the right thing about it.** `src/translate/vllm.ts` reads
`max_model_len` back from the server and sizes each request into it (`room = maxModelLen -
prompt`); it deliberately does not invent a `num_ctx`, because under vLLM the window is fixed at
start. So a client pointed at a 16384-token server asks for 16384-token work. Nothing has to be
told; it adapts.

So **resizing the context is a real lever and a cheap one**, and the bar it has to clear is the
task's own unit of work, not the checkpoint's `max_position_embeddings`. Steps 3 and 4 of the
rule below take `context` as *what this task sends*, and a host that cannot hold even that is
the disabled case.

**One live defect this exposes, and it is cleanup rather than translate.** `foundry/src/clean/
runner.ts:85` is `CTX_MAX = 16384` and `qwen3.8-27b-4bit`'s cuda block is `context_default =
16384` — **exactly equal, with no room between them** — while BookForge's `text-server.ts:
273-276` asserts *"Foundry pins num_ctx 12288…; 16384 covers that with headroom."* There is no
headroom: a clean prompt at Foundry's own ceiling is an HTTP 400 from the server BookForge
launched for it. Three files, one number, and the only one that is wrong is the one claiming
margin.

#### The rule as stated picks WRONG on the Mac, and that is a real bug

Owen ruled that the Mac should translate at 4-bit. Run the walk against today's config and it
does not:

```
Mac unified                          64.0 GB
− desktop_allowance_bytes (3 GiB)     3.2 GB
= "available"                        60.8 GB
  qwen3.8-27b bf16 (mlx)             55.5 GB   ← best-first takes THIS
  qwen3.8-27b-4bit (mlx)             33.9 GB   ← what Owen ruled
```

bf16 "fits" 60.8 GB, so a best-first walk selects it and leaves **8.5 GB for macOS, the
compositor, the browser and every other app** — on a machine where the model shares one pool
with all of them.

The defect is **`desktop_allowance_bytes`, which defaults to 3 GiB on BOTH backends**
(`config.py:32-40`, `cli.py:929` — applied after detection has already established which backend
this is). On `cuda-linux` the card is a separate pool and 3 GiB is a defensible desktop reserve.
On `mlx-darwin` there is no separate pool: the allowance has to cover the entire operating
system, and 3 GiB is nowhere near it.

**So the allowance is per-backend, and it is a ruling rather than a default** — on unified memory
it is the larger part of the ruling, because it is the only thing standing between "the model
fits" and "the machine still works".

> **RULED AND BUILT, 2026-09-13.** `config.default_desktop_allowance_bytes(backend_kind,
> total_bytes)`, resolved by `crucible init` **after** detection rather than by argparse before
> it, and still overridden by an explicit `--desktop-allowance-bytes`.
>
> - `cuda-linux` — **3 GiB flat**, unchanged. A compositor and a browser want about the same
>   VRAM on a 12 GB card as on a 24 GB one, so a constant is the honest shape, and Owen's 3090 Ti
>   has run this number for months.
> - `mlx-darwin` — **25% of unified memory**. The reserve has to cover the entire OS out of the
>   pool the model allocates from, and that scales with the machine. 25% is the complement of
>   Metal's own `recommendedMaxWorkingSetSize` (~75% of physical on Apple Silicon), so it is a
>   number with a source rather than a guess.
>
> It did not need the measurement of section 4, because a better check was already available:
> **Owen's own configuration, which predates the rule.** At 25% the 64 GB Studio shows 48 GB
> available, bf16 27B (55.5 GB) is refused and the 4-bit (33.9 GB) is selected — which is what he
> already runs. `test_the_mac_reserve_selects_the_4bit_27b_owen_already_runs` in
> `tests/test_accelerator.py` is that check, kept.

This is worth stating plainly because of how it was found: the rule was written, then checked
against a decision Owen had already made, and it disagreed with him. **The disagreement was the
rule's, not his.** A selection rule that is never run against a known-good answer is a rule
nobody has tested.

### 1.2 `desktop_allowance_bytes` is what decides whether translate exists

The arithmetic on the machine this was ruled on:

```
3090 Ti total                      25.8 GB
− desktop_allowance_bytes (3 GiB)   3.2 GB
= available to a job               22.5 GB
  qwen3.8-27b-4bit needs           21.6 GB
= margin                            0.9 GB
```

It nominally fits and practically does not — that is before a browser or a compositor. So the
desktop-reserve question that section 5 deletes from BookForge **returns here as Crucible's**,
and translate is the job that makes it bite: too low an allowance and translate is enabled and
OOMs; too high and it is disabled on a card that could have done it.

**Owen ruled on this directly (2026-09-13): the estimates are not to be measured.** *"I've been
using this system the way it is for months and it works fine. Use the current settings for
each."* The 3 GiB reserve and the declared per-model sizes are therefore the numbers this build
selects on, and the 0.9 GB margin above is a margin that has been holding in production rather
than one nobody has tried. Section 4's measurement is **descoped**, not deferred: what it was
for — deciding whether translate exists on the 3090 Ti — has already been answered by a year of
the machine answering it.

**Almost none of this is new machinery.** Crucible already has:

- `config.py:73-78` — `enable_echo / enable_llm / enable_asr / enable_tts / enable_align /
  enable_rvc`, per job type, whose own comment (`:122`) says *"ABSENT means off and that is not
  a fallback… A capability flag is a different animal."*
- `jobs/__init__.py:144` — a disabled type is refused by name, `400 job_type_disabled`.
- `api.py` — `/v1/info` answers `job_types`, and says so: *"What you can POST is answered by
  `job_types`."*
- `GET /v1/accelerator` — the card probe, already opt-in and already shaped for this.
- The precedent: `asr` and `align` are **already** disabled by capability on `mlx-darwin`, with
  refusals that name the reason rather than the flag.

So the ruling adds **one axis** — card size — to a mechanism that already reasons about
capability. What is missing is only that a human sets the flag today and a measurement should.

## 2. What install does — BUILT

`crucible install` gains a selection step, between building the env and finishing:

1. **Probe** the card (`accelerator`), on the backend this host actually has.
2. **For each job type**, compare the card against what that type needs on this backend.
3. **Write the `enable_*` flags** — and, next to each, **the reason**.
4. **Refuse nothing silently.** A type turned off records the number that turned it off.

The reason is not decoration. It is the difference between a server that is configured and a
server that is broken in a way nobody can see.

**As built**, with three things the four steps above leave open:

- **The step runs after the env, and writes before it refuses.** Not before the env, because a
  flag saying "this server offers tts" must not be written by a run whose pip install then
  failed. Not skipped when the card is too small, because the env on disk is still right — the
  card is what is wrong, and a bigger GPU or a smaller allowance makes the same env usable
  without a rebuild. And the record is written *and then* the command exits 1 with the named
  reason, because R6 says partial work survives failure and here the partial work is the only
  durable answer to "why is tts off on this box".
- **`crucible capability` is a verb of its own**, and a dry run by default. The decision
  depends on three things that move independently of the envs — the card (Owen swaps GPUs),
  `desktop_allowance_bytes`, and the manifests (a new quantization ships with a release). If
  the only door to re-deciding were `crucible install`, re-deciding would mean rebuilding a
  multi-gigabyte venv to answer a question about arithmetic, and nobody would ever do it.
- **`--write` may only turn a type OFF.** A flag means "this server offers this type", which
  needs the card to fit *and* the env to exist, and only `install` knows the second. Turning a
  flag off because the model no longer fits is safe in the direction that matters; turning one
  on because the arithmetic works would advertise a job type with no env behind it.

### 2.1 The refusal message must change with it — BUILT

Today a disabled type says:

> *job type 'tts' is not enabled on this server (set `[jobs] enable_tts = true` in config.toml)*

On a 6 GB card that sentence is **actively harmful**: it tells the operator to flip a flag that
will then OOM on the first render. When install disabled a type for a measured reason, the
refusal has to carry it:

> *tts is disabled on this server: Higgs v3 needs ~19 GB and this card has 6 GB. Higgs is not
> quantized, so this is not a tuning choice.*

**A refusal that suggests a fix which cannot work is worse than one that just says no.** It is
the no-band-aids rule applied to an error message, and it is the whole reason step 3 records a
reason rather than a boolean.

**As built** (`crucible/jobs/__init__.py`, `disabled_error`), there turned out to be **three**
reasons a type is off and not two, and a reader has to be able to tell them apart:

1. **Nothing has been decided here** — no `[capability]` record, a config `crucible init`
   wrote and nothing probed. It must not invent a reason and must not imply the flag is safe:
   *"no capability selection has been recorded here — nothing knows whether this host can hold
   the models it needs. Run `crucible capability` to find out before turning [jobs] enable_tts
   on; on a card that is too small, turning it on buys an OOM instead of a server."*
2. **The card cannot hold it** — the section 2.1 case. Every class behind the flag is recorded
   disabled, the message carries each one's reason and shortfall, and it ends *"Turning [jobs]
   enable_tts on would not change any of those numbers; it would only move the failure to the
   first request."* The shortfall is also on `details.shortfall_bytes` as a NUMBER, because a
   sentence is never load-bearing (R4).
3. **The card can hold it and the env was never built** — recorded enabled, flag off. This is
   the one case with an action that works, so it is the one case that gets one: *"Install it
   with `crucible install tts`."*

The old sentence had **four** owners — `resolve`, `/v1/models`, `/v1/voices` and the streaming
door each wrote their own copy. They are now one function, because three of the four would
otherwise have been left behind saying the harmful thing.

## 3. What it costs PHASE7: a slot becomes typed

This is the part that is cheap now and expensive later.

PHASE7 assumes every registered server has a GPU slot that can take **any** GPU step. Under this
ruling that is false: a 6 GB server is an `llm` + `rvc` + `asr` server and not a `tts` server.
So:

- **`any` means "any server that can do THIS step"**, never "any server". The router filters by
  the server's advertised `job_types` before it considers rank or slots.
- **A pin is validated when it is made, not when it runs.** Pinning a book to a server that
  cannot do TTS must be refused at the moment of pinning — the user is right there, and the
  alternative is a queue that accepts work it will fail hours later.
- **ONE BOOK = ONE GPU survives intact**, and gains a precondition: the server a book is pinned
  to must be able to do *every step in that book's chain*, because the book does not move.

## 4. The prerequisite: the estimates are declared, not measured — DESCOPED

> **RULED 2026-09-13, and the build follows it.** Owen: *"I don't think we need to measure
> estimates. I've been using this system the way it is for months and it works fine. Use the
> current settings for each."* Selection runs on the DECLARED estimates in the manifests, and
> `crucible/capability.py` carries no `measured` / `declared` distinction — a field nothing
> consumes is a field that drifts. The section below is kept as the record of why the
> distinction looked necessary and of what a real measurement would still be worth; it is not
> a blocker on anything in section 2, which is built.

**This phase cannot be trusted until the numbers it reads are real**, and today they are not:

- All 14 voice blocks carry `estimate_basis = "declared"`. Every one claims the same
  19,000,000,000 bytes — a figure derived from SGLang's `--mem-fraction-static 0.6`, never once
  watched on a card.
- The whisper, aligner and RVC manifests do not have the field at all. Each says *"COMPUTED, NOT
  MEASURED"*. The distinction is not merely absent there, it is unrepresentable — the loader
  refuses unknown keys.

That number would now be load-bearing in **three** places at once:

1. whether a job is admitted — `residency.py:531` hands it to `accelerator.py:390`, which adds
   it to **measured** free VRAM and refuses below `need_bytes`;
2. whether a job type is **installed at all** on this server;
3. what a client is told this server can do.

A wrong declared number therefore disables a server that could have rendered the book, or
installs a type that OOMs on first use — and in neither case does anything say why.

**The gate on this phase is `scripts/keeper-tts-live.sh` on a real card, both backends**, with
the printed lines pasted into the manifests and `estimate_basis` moved to `measured`. One
blocker inside the blocker: that script gates its whole memory measurement on
`command -v nvidia-smi` (`:211`, `:362`), so the `mlx-darwin` half cannot run as written.
`scripts/measure-llm-memory.sh:10-16` already reads unified memory on Darwin; that branch has to
be carried across first.

Until the numbers are measured, **install may record what it would have decided and refuse
nothing** — a dry run is honest; a decision on a guessed number is not.

## 5. What this deletes

BookForge's `electron/orpheus-memory.ts` tier table — `extreme / fast / moderate / light`, plus
`auto`.

**Owen, 2026-09-13:** *"the extreme/moderate/fast/etc settings are irrelevant now. it uses what
is available on the system. the user doesnt set those. crucible does."*

He is right that nothing sets them: there is no settings row in `creamsicle-desktop/src` or in
`dist/renderer`, and the `orpheus-memory:get` / `:set` IPC handlers in `main.ts` are dead
endpoints nobody calls. `auto` is the only reachable mode.

But the table is **not** inert, and deleting it is a real change with two live call sites:

- `parallel-tts-bridge.ts:7456` — `resolveConcreteOrpheusTier(free, total, …)` →
  `fitOrpheusTier` → `orpheusMemoryProfile` → `computeSafeGpuUtil(profile.capMB,
  profile.marginMB, …)`, which sizes vLLM.
- `higgs-spawn.ts:435` — darwin-only, and **it is Higgs**: the tier sets
  `NARRATOR_HIGGS3_MLX_BATCH` and `NARRATOR_HIGGS3_MLX_MEM_BUDGET_GB` on the Mac.

The thing to see is that **`auto` already IS this ruling**: it probes free and total VRAM and
picks. It is not a user preference that outlived its UI — it is the right behaviour implemented
in the wrong repo, owned by the wrong app, named after the wrong engine.

Deleting it also settles an open R1 defect rather than reconciling it: the desktop VRAM reserve
currently has **three** owners — `gpu-arbiter.ts:285` (3072), `orpheus-memory.ts:170` (10240),
and the per-tier `marginMB` of 1024/2048 which is the one actually subtracted — with
`orpheus-memory.ts:412` admitting in prose that the `extreme` tier *can never satisfy* the
10240 guarantee and letting `auto` pick it anyway behind a fourth number. None of those three
numbers needs to win. They go with the table.

## 6. Order

1. ~~**Carry `measure-llm-memory.sh`'s Darwin branch into `keeper-tts-live.sh`**~~ — DESCOPED
   with section 4. Still worth doing on its own merits; no longer on this phase's path.
2. ~~**Measure**, both backends~~ — DESCOPED. See section 4's ruling block.
3. **Install's selection step** (section 2), with the reason recorded and the refusal
   rewritten — **DONE, 2026-09-13.** See the status block at the top for where each piece is.
4. **The typed slot** in the client router (section 3) — `any` filtered by advertised
   `job_types`, and a pin validated when it is made. **OWED**, and it is the one that touches
   BookForge's queue.
5. **Delete the tier table** (section 5) now that Crucible is the thing sizing. **OWED**, and
   it goes last because it is a deletion in the other repo.

Steps 4 and 5 are what is left. Step 3 went first rather than third because steps 1 and 2 were
descoped out from under it, and it turned out to be the step the other two depend on anyway:
step 4 filters on what a server advertises, and nothing advertised a per-type verdict until
step 3 wrote one down.

## 7. The local form: one catalog, two doors — BUILT 2026-09-14

**Owen, 2026-09-13, via Foundry:** Crucible's model manifests are the catalog of record for
Foundry's **local** lineup too — the Ollama / llama.cpp fallback its app runs when no Crucible
is present — so that "what can this machine run" has one owner. Foundry's app vendors a
generated JSON and compares it by content.

### 7.1 Why one owner

Before this, the same three models were described in three places and nothing compared them:

| the fact | Crucible | Foundry |
|---|---|---|
| the 9B and the 27B a machine can run | `models/*.toml`, sizes measured on the card | `app/electron/llm-catalog.ts` — a table of Ollama tags with sizes *"read off ollama.com/library/qwen3.5/tags, 2026-08-26"* |
| the page reader's weights | `models/dots-ocr.toml` — `dots-studio/dots.ocr` for vLLM | `app/electron/page-reader.ts` — `ggml-org/dots.ocr-GGUF`, a Q8_0 pair for llama-server |
| what translate may run on at all | `capability.py` — the class is disabled below a 27B | the wizard's "largest that fits" over its own table |

That is ARCHITECTURE.md section 1's shape exactly: one fact, two owners, nothing comparing
them. The ruling collapses it to one owner and **two doors**: `/v1/models` for a machine that
has a Crucible, and `foundry-lineup.json` for one that does not. Both are read off the same
file.

### 7.2 The `[local]` table

Optional, one per model manifest, validated as strictly as every other table (unknown key
refused by name; required keys per kind; a wrong type named; a pin that is not a pin refused):

```toml
[local]
kind = "ollama"                      # or "gguf"
tag = "qwen3.5:9b-bf16"              # ollama: the exact tag — what `ollama pull` gets, what --model is
# hf_repo  = "<owner>/<name>"        # gguf: the repo
# revision = "<40-char sha>"         # gguf: the pin; a branch name is refused
# file     = "<name>.gguf"           # gguf: the text tower
# mmproj   = "<name>.gguf"           # gguf: the vision projector — REQUIRED when [model]
#                                    #   modalities carries "image", REFUSED when it does not
download_bytes = 19_321_189_044      # what a pull fetches
needs_bytes = 20_821_189_044         # what it takes to RUN: weights resident plus working room
needs_basis = "declared"             # or "measured" — the same discipline as estimate_basis
minimum_for = ["translate"]          # optional: the classes this model is the FLOOR for
```

Two rules cross tables, and both are refusals at load rather than surprises at a screen:

- **A `[local]` table requires `[model] display` and `[model] description`.** The lineup is
  drawn as tiles, and a tile with no label is a tile somebody downstream would invent a label
  for. `display` is the name every other catalog in this repo already uses (voices, rvc,
  denoise); `description` is new and optional on a model without a local form. Both now travel
  on `/v1/models`, `null` when unstated.
- **A page reader needs its projector.** llama-server serves a vision model as a text tower
  plus an `--mmproj`; started without the projector it loads, answers `/v1/models`, and
  refuses every page — a broken page rather than a missing file. So `image` in `modalities`
  without `mmproj` is refused, and `mmproj` on a text-only model is refused too.

`minimum_for` is **Owen's tile rule**: the smallest model a class may run on *at all*. Each
entry must be a name in `capability.CLASSES` (the manifest may not invent a class), and the
generator further refuses a floor for a class the model does not serve.

### 7.3 The file

`scripts/gen-foundry-lineup.py` writes `foundry-lineup.json` at the repo root. It re-parses no
TOML: every row comes through `crucible/manifests.py`, and every `classes` list through
`capability.classes_for_model`, which walks the same `CLASSES` table install selects on — so a
model's classes have one owner and no `classes = [...]` key exists in any manifest.

```json
{ "generated_from": "<crucible git sha>", "schema": 1, "models": [
  { "id": "qwen3.5-9b", "classes": ["clean"], "label": "Qwen 3.5 · 9B", "description": "…",
    "local": { "kind": "ollama", "tag": "qwen3.5:9b-bf16",
               "downloadGB": 19.32, "needsGB": { "value": 20.82, "basis": "declared" } },
    "minimum": false, "minimumFor": [] } ] }
```

A `gguf` row's `local` is `{kind, hf_repo, revision, file, mmproj | null, downloadGB,
needsGB}`. Gigabytes are decimal at two places — Ollama's own unit (`ollama list` prints
19_321_189_044 B as "19 GB"). Rows are in id order. **A model without a `[local]` table is
omitted**, not emitted with `local: null`: a machine without Crucible cannot run it, and
`--verbose` says which. `minimum` is true when the row is in `minimum_for` for any of its
classes.

`generated_from` is the commit the generator ran on — always one behind the commit carrying
the file — so **`--check` and `tests/test_lineup.py` compare everything but that key**; a
check that included it would be red on every commit, and a red guard is a broken guard (R2).
CI runs `--check` on every push.

### 7.4 What the three rows say, and the two things writing them found

All three `needs_bytes` are **DECLARED**: the download plus 1_500_000_000 — Foundry's own
`OVERHEAD_GB` (`app/electron/llm-catalog.ts`: KV at an ordinary context plus the runner's
buffers, chosen to err small). None has been watched on a card, and the basis says so all the
way to the picker. Each manifest's `[local]` comment records where every byte came from and
the date it was read.

| model | local form | download | needs |
|---|---|---|---|
| `qwen3.5-9b` | `qwen3.5:9b-bf16` — bf16 because that is the clean-text ruling | 19.32 GB (`/api/tags`, 2026-09-14) | 20.82 GB |
| `qwen3.8-27b-4bit` | `qwen3.8:27b-24g` (Q4_K_M); `minimum_for = translate, simplify, analysis` | 17.74 GB | 19.24 GB |
| `dots-ocr` | `anthonym21/dots.ocr-GGUF` @ `42ab3102…`: `Dots.Ocr-1.8B-Q8_0.gguf` + `mmproj-Dots.Ocr-F16.gguf` | 4.42 GB (tree API, LFS sizes) | 5.92 GB |

**Found 1 — the 27B tag is not a published tag.** `qwen3.8:27b-24g` is Owen's own Modelfile
over the library's `qwen3.8:27b` (num_ctx 98304 and his sampling baked in; `ollama show` says
`parent_model: qwen3.8:27b`). `ollama.com/library/qwen3.8:27b-24g` answers **404**;
`qwen3.8:27b` answers 200. So `ollama pull qwen3.8:27b-24g` fails on every machine but his,
which contradicts the key's own definition. It is written as ruled and **labelled a stopgap in
the manifest**; the ruling owed is one of two — the catalog names the published parent and
Foundry keeps pinning `num_ctx` per book on its ollama door (its SLOTS.md says it already
does), or the Modelfile is published under a namespace and the tag gains that prefix. The
bytes are the same either way: the model and projector blobs are the parent's; a Modelfile
adds a 132-byte params layer.

**Found 2 — Foundry's page reader pins a different pair.** `page-reader.ts` names
`ggml-org/dots.ocr-GGUF` with a **Q8_0** projector (1.34 GB); the ruling names anthonym21's
**F16** projector (2.52 GB, regenerated 2026-03-23 from a corrected converter per that repo's
README). Same model, two repos, two projectors — the exact defect this section exists to
close, and the first vendoring will say so rather than let the two drift in silence.

### 7.5 What it is not

Not on the wire. A Crucible never runs Ollama, so `/v1/models` does not carry `[local]`; the
JSON file is that table's one door. `display` and `description` do travel, because they are
facts about the model rather than about a machine that lacks a server.
