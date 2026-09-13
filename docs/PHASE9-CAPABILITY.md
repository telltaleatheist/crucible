# Phase 9: a server declares what it can actually do

**Owen, 2026-09-13:** *"if i run crucible on a 6 gb gpu, it should use quantized 9b to fit on
that card. thats something we can configure on crucible install — picking which models and how
quantized those models are for each server install"* — then, narrowing it:
*"higgs is tied to a certain size. we cant (or wont) quantize that. if higgs doesnt fit in a
card that crucible is installed on, it's disabled on that gpu."*

This phase is that ruling, what it costs, and the one thing that has to be measured before any
of it can be trusted.

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
fits" and "the machine still works". Section 4's measurement has to produce it for both backends,
and until it does, the honest behaviour is the dry run: record what would have been selected and
select nothing.

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
OOMs; too high and it is disabled on a card that could have done it. This number is now a
ruling, not a default, and it belongs in the measurement of section 4.

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

## 2. What install does

`crucible install` gains a selection step, between building the env and finishing:

1. **Probe** the card (`accelerator`), on the backend this host actually has.
2. **For each job type**, compare the card against what that type needs on this backend.
3. **Write the `enable_*` flags** — and, next to each, **the reason**.
4. **Refuse nothing silently.** A type turned off records the number that turned it off.

The reason is not decoration. It is the difference between a server that is configured and a
server that is broken in a way nobody can see.

### 2.1 The refusal message must change with it

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

## 4. The prerequisite: the estimates are declared, not measured

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

1. **Carry `measure-llm-memory.sh`'s Darwin branch into `keeper-tts-live.sh`** so the Mac half
   can run at all. Cheap, and it unblocks everything below.
2. **Measure**, both backends, and move `estimate_basis` to `measured` where it now exists —
   adding the field to the asr / align / rvc loaders where it does not.
3. **Install's selection step** (section 2), with the reason recorded and the refusal rewritten.
4. **The typed slot** in the client router (section 3) — `any` filtered by advertised
   `job_types`, and a pin validated when it is made.
5. **Delete the tier table** (section 5) once Crucible is the thing sizing.

Steps 1 and 2 are a card and an afternoon. Step 3 is small because the flags already exist.
Step 4 is the one that touches BookForge's queue. Step 5 is a deletion and goes last, because
until step 3 ships the tier table is still the thing doing the job.
