# Engines, capability and the card

How Crucible decides what a host can run, what it puts on the card, and how each
engine is started and stopped. Modules: `backend`, the capability modules
listed under "Where each part lives", `memorybudget`,
`precision`, `servingplan`, `ttsplan`, `asrplan`, `accelerator`, `vram`,
`ladder`, `residency`, `engines/*`, `decide`,
`sampling`, `pages`, `llamacpp`, `ollamastore`, `interpreter`, `lineup`.

## Backends

| backend | host | engines |
|---|---|---|
| `cuda-linux` | Linux with an NVIDIA card (on Windows: the server inside WSL2) | vLLM for text and pages; `llama-server` (llama.cpp) for a block that is one GGUF |
| `mlx-darwin` | Apple Silicon | `mlx-lm` for text, Crucible's own `mlx_vlm_serve` for pages and for image decisions |
| `llama-windows` | Windows, natively | `llama-server` (llama.cpp) on GGUF |

- One engine per (backend, class family); `manifests.BACKEND_ENGINES` owns the
  pairing, `engines.build_engine` only maps a name to a class and refuses unknown
  names. The one exception is the weights' form: on `cuda-linux` a block that names
  a GGUF `file` runs on `llama-server` (`manifests.block_engine`, `GGUF_BACKENDS`).
  Owen, 2026-10-09: B-Sides sends one request at a time, so its models gain nothing
  from vLLM's batching and paid its compile and graph capture (3 min 22 s on the
  first load, about a minute after) on every album's swap; they are GGUF on
  llama-server, and the text verbs stay on vLLM.
- `llama-windows` is a full backend, not a relay (Owen, 2026-09-14: *"the windows
  side should still host GPU jobs even if WSL isnt present/workable"*). The WSL
  engine is still better where it runs: batching, parallel page reading, and the
  five Python job types (narrator, whisper, aligner, urvc, separator) that never
  run natively on Windows.
- `backend.detect_windows` never refuses (Owen: *"a crucible server will run on
  absolutely anything"*): no NVIDIA driver means the llama.cpp CPU build and a
  pool of system RAM, and the capability row says "slow" instead of disabling
  the class. The accelerator guard is stricter: a driver that is present but
  will not answer is `accelerator_unreadable`, never "use the RAM".
- **A backend runs where its engine runs and nowhere else.** vLLM and SGLang
  have no win32 build, so a `cuda-linux` config found on a Windows host is
  refused, and `llama-windows` off win32 is refused the same way
  (`backend_not_here`). The ASR and align backends are the ones that run their
  engines (`asrmodels.ASR_BACKEND_ENGINES`, `alignmodels.ALIGN_BACKEND_ENGINES`);
  Windows is never one of them. On `llama-windows` the Python job classes answer
  `enabled: false` with the one sentence `capability.NEEDS_WSL_REASON`, the same
  for every class so an app shows it once, and `crucible install` refuses them
  `needs_wsl`.
- WSL2's `nvidia-smi` lives at `/usr/lib/wsl/lib/nvidia-smi` and is not always
  on PATH in a non-login shell.
- On Apple Silicon the "card" is unified memory: `vram_bytes` is the machine's RAM.
- `physical_memory_figures` (Windows, `GlobalMemoryStatusEx`) returns total and
  available from one call: capability sizes against total, the guard against
  available.

## What a card can do

`backend.CardFacts` has two halves. DECLARED facts come from the compute
capability. MEASURED facts come from the ladder's record. A fact nobody knows
is `None` (unknown), never `False`, and **unknown refuses nothing**.

Floors, read from the pinned engine source (vLLM 0.29.0, torch 2.13.0+cu130):

| feature | floor | source |
|---|---|---|
| bf16 | sm_80 | vLLM `platforms/cuda.py` `supported_dtypes` / `check_if_supports_dtype`; torch emulates below 8.0 |
| FlashAttention 2 | sm_80 | vLLM `flash_attn.py` |
| fp8 | sm_89 | vLLM `supports_fp8` |

- FA2 and fp8 are reported but nothing in the catalog needs them (vLLM falls
  back to Triton attention on any capability).
- Tensor cores arrive at sm_70, **except the Turing GTX 16 family**
  (TU116/TU117: GTX 1650/1660/1660 SUPER/1660 Ti), which is sm_75 without
  tensor cores. It is detected by the `GTX 16` name prefix. Tensor cores are
  reported and never required.
- `probe_compute_capability` sends its own `nvidia-smi` query. A driver that
  predates `compute_cap` rejects the whole query, and that must read as
  "unknown generation", not "no backend".
- Measured by the ladder: `cuda_graphs` (False means vLLM starts
  `--enforce-eager`, which is slower but still runs) and `vllm` (False refuses
  every vLLM block on this card and quotes the first error line).

## Capability selection

Ruling (Owen, 2026-09-13): *"it uses what is available on the system. the user
doesnt set those. crucible does."* The client picks the capability class and
therefore the model family. Crucible picks the quantization.

1. **A goal, then the most parameters, then the highest precision**
   (docs/VERB-SIZING.md rules 2 and 3, built 2026-10-09). A text class carries a
   `Goal` (`CapabilityClass.goal`, in `params_b`): 27B for `generate`, `translate`,
   `simplify` and `analysis`, 9B for `decide` and `clean`. The automatic pick
   (`CapabilityClass.pick_order`) takes, among the candidates at or below the goal
   that fit, the most `[model] params_b`, then the most bits (each backend block's
   stated `bits`), then a model's own form before its `weights_of` alias; the
   catalog's order (`memory_bytes_estimate` descending, then id) settles the rest.
   It never takes one above the goal; Settings can. A class with no goal (the media
   classes) keeps the old rule: the first that fits by declared size. The record
   says which: *"qwen3.5-9b (goal 9B; bf16 fits with 2.2 GiB to spare)"*, or below
   the goal *"(goal 27B; the largest that fits this card)"*.
2. **The bar is total memory minus the desktop allowance, not free memory.**
   A capability describes the host, and free VRAM only describes this second.
   "Is there room right now" belongs to the accelerator guard at load time.
3. **There is no margin term.** The desktop allowance is the margin. Owen,
   2026-09-13: *"I've been using this system the way it is for months and it
   works fine. Use the current settings for each."*
4. **Known-good checks** (in `tests/test_capability.py` and
   `tests/test_verb_goals.py`): a 3090 Ti (24 GiB, 3 GiB reserve) translates on
   `qwen3.8-27b-4bit` and decides on `qwen3.5-9b`. A 64 GiB Studio (25% reserve)
   translates on the 8-bit 27B and decides on `qwen3.5-9b`. An 8 GiB card runs every
   text verb on `qwen3.5-0.8b`.
5. **Generation before memory** (fresh-install #48). First, the precision
   the card can run the candidate at. Second, whether it can start the
   candidate at all.

### Precision (`precision`, Owen 2026-09-26: *"we can quantize if we need to. no less than 4."*)

- A stated bf16 on a card without bf16 runs as **fp16**
  (`enginespec.bf16_fallback` / `card_args` add `--dtype float16`). It uses
  the same bytes, keeps the same place in the walk, and the verdict names the
  precision (`capabilitywords.precision_note`). vLLM itself picks fp16 for `auto` on such a card.
- The stated dtype is `enginespec.declared_dtype`, the one rule `capability`,
  `precision` and `engines.vllm` share; `enginespec` imports only `backend`, so
  neither `capability` nor `precision` loads an engine adapter.
- **Floor: 4 bits** (`precision.MIN_WEIGHT_BITS`). Nothing below it is ever
  a candidate, and a GGUF that names one is refused at load
  (`manifests._gguf_name`).
- Bits are derived from what the engine loads, never from a manifest key: the
  GGUF file's quant tag, then a quantization in the repo name (`-4bit`, `AWQ`,
  `INT4`, ...), then a stated dtype. A block that states none (faster-whisper,
  RVC, separator) is unknown and is not refused.
- Only a measured fact can bar a candidate at every precision: today that is
  `vllm` false (`enginespec.card_needs`). A class left with only barred
  candidates is refused with that fact (`Decision.lacking_features`).
- **No quality check on quantization.** A lower precision is chosen from
  memory and card facts alone. Crucible never measures whether the result
  sounds or reads acceptably. For Higgs a person listens to a quantized
  checkpoint before one exists to be offered.

### Width before bits (`ttsplan`, Owen 2026-09-26)

*"no less than 4 covers higgs as well"*, *"we should drop batches to 1 at a time
before we quantize. id rather it go slow than sound worse"*. **Precision beats
parallelism.** A Higgs v3 voice on `cuda-linux` is offered, in order:

1. bf16 at the voice's declared width (`max_num_seqs`, 16),
2. bf16 at narrower widths down to one passage at a time,
3. 8-bit, one at a time,
4. 4-bit, one at a time.

All figures are declared, not measured. Width 16 uses the voice's own
estimate (SGLang-Omni's `--mem-fraction-static 0.60` reservation on the
3090 Ti). One passage at bf16 is ~10 GB, 8-bit ~5.8 GB, 4-bit ~4.2 GB. Each
extra passage is ~2,000 tokens × 147,456 B/token (Qwen3-4B backbone:
36 layers × 8 KV heads × 128 × 2 × 2 B). The quantized rows are `pending`
because no quantized Higgs exists. Capability says one would fit but never
offers it for pulling. A narrower width reaches narrator as
`HIGGS_MAX_NUM_SEQS`. SGLang's memory fraction is **not** rescaled, so a
narrow plan on a card smaller than the 3090 Ti is a declared fit that still
needs a load test. The mlx arm sizes its batch from `MLX_TIERS` and keeps a
single estimate. Qwen3-ASR uses the same order (Owen: *"yes, fewer at once
before quantizing for asr too"*): fewer pieces at once at full precision
first, then a smaller model.

### Classes

- `llm` is split into several classes because the ruling is about the work, not
  the job type. A 12 GB card that cleans but cannot translate has to be able to
  say so. `simplify` and `analysis` stay separate from `translate` (Owen,
  2026-09-13: *"they can't lie to the user and say a translate job is running
  when it's actually a simplify job"*).
- **No floor** (REVERSED 2026-10-09, docs/VERB-SIZING.md section 5). The 9B floor
  for clean/translate/simplify/analysis and `min_params_b` are gone: every text
  verb's lineup runs down to the 0.8B, so a small card gets a smaller model, never
  "off". Only a card that cannot hold even the smallest is refused, by name. The
  goal (above) caps the automatic pick from the other side.
- `generate` is the one client-sized class (Owen, 2026-09-23: *"make it one
  class and give it the ability to set the context limit"*). The default is
  8192 tokens. A larger request is checked against the host ceiling and refused
  with `context_over_limit`. Crucible never clamps it. Other classes refuse
  `?context_tokens=` by name.
- Routable classes are declared (`routable`), never derived from job type.
  `pages` is not routable (it sends page images) and neither is `decide`
  (no upstream returns logprobs).
- Aliases (`[model] weights_of`) are always the dearer served form, so they
  are skipped unless the class wants that form (`decide` may carry images).
  An asr alias is a different engine, so it stays a candidate.
- `pages` concurrency comes from `pages.PAGE_CONCURRENCY` (12). Both apps
  send twelve, and `models/dots-ocr.toml`'s KV note is sized against that
  number.
- asr order is Qwen3-ASR, whisper turbo, whisper tiny. This is also the
  catalog's size order on both machines, and `tests/test_asr_lineup.py`
  holds it.
- `denoise` is its own class. It shares the rvc env, but a 913 MB separator and a
  2.5 GiB urvc stack are different arithmetic.
- `image` is its own class and job type (image.md). Its candidates' estimate is the peak of
  the largest of the model's three stages at the largest size a block admits, not the sum of
  the download, because the workers hold one stage at a time.
- An app's model choice (`chosen`) is honoured or refused with numbers. It is
  never silently replaced. `decide_all` requires `chosen`, `decide` requires
  `gpu_vendor`, and `record` requires `routes`, all without defaults, so a
  caller that forgets one cannot silently un-choose, mis-size or un-route.
- `Decision.reason` is written for the operator (backend and block names).
  `Decision.summary` is a subjectless verb phrase for anyone, and the caller
  supplies the subject.
- A request-sized `GET /v1/capability` row is decided live for that caller
  and never written back to the record.

### Context ceilings

`engine_total = weights + overhead + kv_bytes_per_token × context × concurrency`.
The context and concurrency belong to the class's work, not to the model's
`context_default`. `Candidate.context_ceiling` is the single ceiling function:
the smaller of *served* (the block's `max_context`) and *memory* (what
`available_bytes` affords). An `overhead_bytes` taken from a measured peak must
have that peak's KV subtracted, or the KV is counted twice (this held
`qwen3.8-27b-8bit` near 64k until 2026-09-23). A host that cannot hold the
weights at all is a disabled class or an `insufficient_memory` load, never a
too-long request. `MIN_LOAD_CONTEXT` is a stated floor, not a measured one.

Figures in capability messages are GiB so they match the guard's refusals.

### Where each part lives

There is no `capability` module; each concern has one module and callers import from it.

| module | owns |
|---|---|
| `memorybudget` | `GIB`, `MIB`, `gib_text`, `available_bytes` (total less the allowance) and `engine_budget_bytes` (that, capped by free memory). Capability, the guard, `vram` and `ttsplan` all subtract the allowance here. |
| `fit` | `WorkingContext`, `Candidate`, `ContextCeiling`, `CatalogCandidates` and the manifest cache |
| `capabilityclasses` | the class table `CLASSES`, `BY_NAME`, and which models a class serves |
| `capabilitywords` | the phrases a verdict is written in (needs, too old, serving width, card description) |
| `verdict` | `Decision`, `decide_capabilities`, `decide_all`, `record`, `pool_name`. `decide_capabilities` is one small function per outcome, and every `Decision` is built by `_decision`. `verdict.decide` is an alias, kept apart in name from the decision door (`crucible/decide.py`). |
| `contextceiling` | `context_ceilings`, `check_ceiling`, `check_load_context` and the `context_over_limit` refusal |
| `capabilityquery` | the `GET /v1/capability` query: `?class=`, `?context_tokens=`, `?concurrency=` and their 400/503 refusals |
| `installplan` | the install and download confirmation text (`/v1/capability/plan`) |
| `servingplan` | `ServingVariant`, the one row type of both serving ladders (`ttsplan`, `asrplan`) |

**Manifest cache.** A `CatalogCandidates` whose loader has a known directory
(models, asr, align, rvc, denoise) parses that directory once and answers from
memory until any `*.toml` in it changes. The key is the directory's resolved
path and mtime, plus each file's name, mtime and size. So an edited, added or
removed manifest is read on the next call, and nothing has to be restarted.
Voices are not cached. `load_all_voices` merges pins, the engine's own voices
and several directories, and no single directory stands for them. On a Windows
dev box, a sized `GET /v1/capability?class=generate` went from about 35 ms
(every model TOML parsed) to about 4 ms (only the files are stat'ed).

## The desktop reserve

`[accelerator] desktop_allowance_bytes` is what the host keeps for itself.
Every capability fit and every vLLM budget subtracts it.

- **Measured** (`ladder.measure_desktop_reserve`, NVIDIA only): `memory.used`
  is sampled once a second for `DESKTOP_SAMPLES` (5) seconds with nothing of
  Crucible's loaded, and then
  `allowance = min(peak + max(peak, 1 GiB), 3 GiB)`.
  The headroom covers a desktop somebody is using, not the idle one that was
  sampled. The 1 GiB is `accelerator.FOREIGN_PROCESS_FLOOR_BYTES`, the existing
  line between "desktop" and "somebody's job", so the reserve leaves room for
  one more such process. Doubling scales with screen size. 3 GiB is what
  owens-pc (a streaming multi-monitor desktop) has always used, and it caps
  what a bad sample can cost. Example: a GTX 1660 SUPER holding 0.3 GiB gets
  1.3 GiB.
- The sample is device-wide `memory.used`, so it counts desktop memory even
  where the driver names no processes (Windows `[N/A]`; WSL2 lists none). A
  sample inside WSL2 sees the Windows desktop (Windows' and WSL's nvidia-smi
  agree, 2026-09-18) but cannot tell whose memory it is. The 3 GiB ceiling
  bounds that error.
- Sampling is refused, by name, when a Crucible server holds something (or
  answers but cannot be asked), when a `llama-server` is on the card, or when
  any compute app holds 1 GiB or more.
- **Basis** (`[accelerator] desktop_allowance_basis`): `measured` (sampled, with
  a note saying what and when), `declared` (Crucible's default: 3 GiB on an
  NVIDIA card it could not sample, 25% of unified memory on a Mac, which is the
  complement of Metal's ~75% `recommendedMaxWorkingSetSize`), or `stated` (a
  person set it). **Nothing ever changes a stated reserve on its own** (Owen,
  2026-09-26). Only `crucible capability --measure-desktop` replaces one, and it
  prints the old and new values. The reserve is always shown together with its
  basis (`config.desktop_reserve_words`).

## The accelerator guard (`accelerator`)

Crucible refuses by name and never evicts anybody else's process.

- `cuda-linux`: a foreign compute app over 1 GiB is `accelerator_busy`. Free
  memory below the estimate is `insufficient_memory`, and the message names both
  numbers. Under WSL2 the compute-app list is **empty** even while a process in
  the VM holds 17 GB (measured 2026-09-12), but `memory.free` is accurate. So
  `unattributed_bytes` counts VRAM in use that no listed app accounts for,
  beyond the allowance, and refuses it. It is clamped at zero, and Crucible's
  own resident engine counts as reclaimable.
- The child of Crucible's child is still Crucible's. narrator's launcher starts
  a serving child. On native Linux the owned set is expanded through `/proc` to
  every process whose **session or process-group leader** is an owned pid.
  Leaders only: engines and workers start with `start_new_session=True`, so
  expanding through a non-leader would claim siblings such as a trainer started
  from the same shell. A `/proc` that cannot be parsed refuses, as an
  unreadable driver does. `accelerator.proc_entries` is the one `/proc` walker;
  narrator's `processes_launched_by` reads `environ` through it.
- The guard runs four checks in order: foreign holders, unattributed bytes
  (`cuda-linux` only), then room (Mac: total less the allowance; everywhere
  else: free plus what Crucible's own resident engine would give back).
- **`accelerator_busy` is weather; the rest is not** (2026-10-02, Owen's rule
  that a transient fault is waited on with a sentence). The first two checks
  (`_refuse_holders`, `_refuse_stray`, including a holder an earlier Crucible
  left) refuse `accelerator_busy`, listed in `accelerator.WAITS_FOR_THE_CARD`
  and so in `admission.KEEPS_WAITING`. Every card-loading job type runs the
  guard in `preflight`, so admission is where it is met: a request that may
  wait keeps its place in the line with the refusal recorded on the item
  (`WaitingLine.not_yet` → `Waiting.card_wait`), and the pump checks again only
  every `line.CARD_RECHECK_S` (5 s). `refuse_if_larger_than_host`,
  `refuse_if_card_lacks`, the room checks (`insufficient_memory`) and
  `accelerator_unreadable` stay failures by name. The guard's second run inside
  `run` is not changed: a job that has started is never put back in the line.
- `mlx-darwin`: a unified pool is **sized, not sampled** (Owen, 2026-09-22: *"it
  shouldnt put a gate on like that. theres actually plenty of memory
  available"*). The check uses total minus the allowance, the same as
  capability. The `vm_stat` free figure is reported only, because macOS pages
  apps out for Metal.
- `llama-windows`: the card is shared by design. dwm, explorer and the browser
  appear as compute apps, so the rule is **room, not solitude**. Foreign apps are
  listed in `details.processes` but never cause a refusal. One exception is a
  `llama-server` Crucible did not start (a crashed run's orphan), which is found
  **by image name** because its pid cannot be known. No unattributed-bytes
  check runs here, because the desktop always holds memory that belongs to no
  compute app.
- `refuse_if_larger_than_host` and `refuse_if_card_lacks` run **before** the
  env and weights checks. Nobody should be told to download 55 GB for a model
  that can never start on the card.
- A probe that cannot answer raises `ProbeError`, which never means "free".

## vLLM memory planning (`vram`)

Inside WSL2, **CUDA cannot see the Windows desktop**: `torch.cuda.mem_get_info`
reports a constant while nvidia-smi's free figure moves. vLLM sizes everything
from `mem_get_info` and profiles a whole-card delta, so desktop growth during a
long load (for example a cold `torch.compile`) is charged to the KV pool. On
2026-09-17 this failed a load at `-0.19 GiB` with the same fraction that worked
the next night.

- Crucible passes `--kv-cache-memory-bytes`, which bypasses the utilisation
  fraction and the profiling window, and derives `--gpu-memory-utilization`
  only as vLLM's startup gate.
- Budget = **min(nvidia-smi free + reclaimable, total − allowance)**. There is
  no third term. The pool is capped at
  `kv_bytes_per_token × context × max_num_seqs`, because bytes beyond that
  cannot be used. `max_num_seqs` reads the flag with `enginespec.flag_value`,
  so the last one given wins, as it does in vLLM. A value that is not a whole
  number is a `ManifestError` that names the model and its manifest file.
- `fits` is vLLM's own one-full-context-request check, asked before the engine
  starts.
- Not planned: `dots-ocr` on cuda-linux (its 0.5× estimate is a deliberate
  budget) and every `llama-windows` block.
- Capability must never use this module. It works from free memory, and
  capability works from total memory.

## The ladder (`ladder`, `crucible ladder`, `<home>/ladder/card.json`)

Rungs in order: `card` (nvidia-smi only: memory, desktop, disk, driver, declared
features), `env` (torch initialises in each installed env, and the matmul speed
of each dtype; bf16 below 8.0 is emulated), `cuda_graphs` (capture and replay in
the llm env), `vllm` (the smallest installed model starts and answers once,
started exactly as `load-model` would).

- A failed rung is a measurement and records its first error line. A rung
  that could not run cleanly is `waiting` (the guard preflight said busy or
  short) or `interrupted` (a foreign compute app appeared while `Watch`
  sampled at 1 Hz). Neither is ever read as a failure.
- Consumed: `cuda_graphs` and `vllm`. Recorded only: speed (it never refuses) and
  memory (declared estimates still decide fit).
- The record is keyed on card name, compute capability, total memory and
  Crucible version. A different key means no record: nothing measured,
  nothing refused, and `doctor` reports it as stale. The file is rewritten
  whole and atomically after each rung. A file that will not parse also counts
  as no record.
- `ladder.card_for` is the one card every decision reads.

## Residency (`residency`)

- **At most one resident thing of any kind** (`llm`, `tts`, `align`,
  `denoise`, `image`). A card holding a Higgs checkpoint has no room for a 9B. Aligner
  and separator are `workers.WorkerSession`s held open, not engines, and have
  their own slots.
- Only the exclusive job lane mutates residency. An engine is published only
  after it proves it is up, and unpublished before it is signalled.
- **Claims.** A streaming session holds narrator's single stdin/stdout, so
  the card has a named owner: `claim(may_mutate=...)`. A render claims with
  `may_mutate=True` (it loads its own voice). A stream claims with `False`.
  Mutators refuse `engine_in_use` while somebody else holds the claim. Claims
  never block.
- `refuse_if_claimed` is asked by load/unload-model, load/unload-voice,
  unload-aligner, `tts` and `align`. `asr` and `rvc` deliberately do not
  ask: they contend for memory, which the guard answers, not for narrator's
  wire. `echo` needs no accelerator.
- **The dying slot.** `unload` moves the handle to `DyingResident` (with a pid
  snapshot) before signalling. The slot clears only when the stop returns.
  Until then `owned_pids` still counts the process, and every load, claim and
  stream refuses `engine_still_stopping`. Crucible never SIGKILLs, so a human
  ends this state. `shutdown` retries the stop once.
- **Clearances.** A settlement that is clearing the card is not a foreign
  holder. Every door waits a clearance out (`settled_for`, with a budget of
  `CLEARANCE_TIMEOUT_SECONDS = STOP_TIMEOUT_SECONDS + 30`) and then records
  its hold **under `_claim_lock`**. The settlement checks and claims under the
  same lock (`claim_to_clear`), so no door can record a hold the settlement
  missed. An unload of the subject being cleared counts as the same intent and
  reports `done`. A clearance that outlives the budget is raised as a wedge.
- `Residency` is lifecycle only: claims, evict, warming, dying, reclaim, the
  resident record and `unload`. It knows no engine and no job. A load is
  `Residency.occupy(kind, subject_id, start, say=...)`: occupy refuses (a
  claim on another thread, a stop still running), evicts, marks warming,
  calls `start()` and publishes the `Occupant` it returns (the `Resident`
  record plus the engine or worker session whose pids and `stop()` it owns).
  The body of each load lives with its job package: `jobs.llm.occupy_model`,
  `jobs.tts.common.occupy_voice`, `jobs.align.occupy_aligner`,
  `jobs.denoise.occupy_separator`. An occupant whose kind or id is not the one
  asked for is stopped and refused.
- `engines.start_engine` tears down a half-started
  engine on **any** `BaseException`. An orphan there sits in no slot, so the
  guard would call it foreign.
- A voice load is not finished at `ready`. narrator's `load`/`loaded` exchange
  is part of it, and the sample rate on `loaded` must match the manifest.
  Crucible refuses a mismatch and never resamples.
- `occupy_model` requires `context` and `plan` (no defaults): an unsized pool
  on a shared card is the 2026-09-17 failure. Reloading the same id is a full
  restart.
- Engine-specific argv and served names are class methods on the engine,
  reached through the `engines.ENGINES` table, never by comparing names:
  `load_args(spec, weights_dir, context, plan, card_flags=, source=)` and
  `served_name(weights_dir, model_id)`. The defaults are the manifest's args
  plus the plan's flags, and the model id. `engines.engine_load_args` and
  `engine_model_name` look the class up by `spec.engine`.
- `jobs.llm` and `jobs.tts.common` call `engines.build_engine`, `engines.engine_model_name`
  and `engines.build_voice_engine` through the module, so a test's fake engine patches
  `crucible.engines`.
- Argv order: the manifest's args, then `card_args`, then the KV plan's flags.
  argparse uses the last spelling, so later flags override. `VllmEngine`
  adds `--max-model-len` and the decide flags (mlx-lm has no such flag, and
  its `max_model_len` is admission). `LlamaServerEngine` composes `-m`,
  `--mmproj` and `-c`, because only the server knows where the weights are.
  `MlxLmEngine` serves under the resolved weights path, `MlxVlmEngine` under
  the path as given.

## Engines

### Common (`engines/base`)

- Engines bind 127.0.0.1 on a free port. stop is SIGTERM to the process
  group and **never SIGKILL**, because a killed CUDA process wedges WSL2 until
  Windows reboots. On win32 stop sends `CTRL_BREAK_EVENT` and then terminates
  (`procgroup`). Each engine gets its own process group or session.
- **One stop routine.** `procgroup.stop_gracefully(process, what, timeout,
  log)` is the only ask/wait/refuse sequence: engines (`SubprocessEngine.stop`),
  workers (`workers._terminate`, `WorkerSession.stop`) and nothing else. A
  process still alive after the wait is an error naming its pid, `kill <pid>`
  (never `-9`) and the log. `STOP_TIMEOUT_SECONDS` (180 s) and
  `LOG_TAIL_LINES` live in `procgroup`. Scripts that run inside a job env
  cannot import Crucible and repeat the rule by hand: the ladder's smoke
  scripts get SIGTERM and 180 s and are then reported by pid; the rvc worker
  recycles urvc with SIGTERM and 180 s and writes the pid to stderr.
- **Stop budget.** `SubprocessEngine.sigterm_wait_seconds` is the SIGTERM wait
  (180 s; llama-server 30 s). `stop_budget_seconds` is the worst case of a
  whole `stop()`: the SIGTERM wait, plus on win32 two `KILL_WAIT_SECONDS`
  (taskkill, then the wait after it). narrator overrides it with
  `QUIT_GRACE_SECONDS` (210) + the reader join (2) + the base budget (180) +
  2 x (`LAUNCHED_SERVER_GRACE_SECONDS` 180 + one 1 s poll) = 754 s on Linux.
  Whoever waits on a stop from outside (residency's clearance) must wait
  longer than `stop_budget_seconds`, or it reports a wedge that is only a slow
  stop. Inner waits compose: since narrator 72069b5b its quit stops the
  server it launched, and `QUIT_GRACE_SECONDS` (210) is the time that stop
  takes, so the SIGTERM after it is a fallback, not the usual path. The rvc
  worker's 180 s urvc stop happens after `ready`, where the worker exchange
  has no silence clock, so nothing outside it gives up first.
- **Next steps in refusals.** A missing executable names the install:
  `crucible install llm` (vLLM, mlx-lm, mlx-vlm, and llama-server, whose
  install pulls the pinned llama.cpp build) or `crucible install tts`
  (narrator). A missing weights directory names `crucible models pull <id>`
  (narrator: `crucible voices pull <id>`). A port another server answers, or
  a bind failure in the log (`address already in use`, any engine that binds
  a port), is `port_in_use`: Crucible picks a fresh port on every start, so
  the next step is to run the load again.
- Logs are `~/.crucible/logs/engine-<id>.log`, **appended** and never truncated
  or rotated. A reload used to erase the log of the hang being investigated.
  `log_tail` reads only the current run (up to its header), so a dead run's
  fatal line cannot refuse a new start.
- Chat admission is **the engine's own concurrency + 1**. The +1 is the next
  request, ready when a slot frees. An engine that states no concurrency is
  not bounded. A serial engine with an unbounded queue starved a request past
  its deadline (Foundry on mlx-lm, 2026-09-20).
- A decision needs top logprobs, which is a fact read from the pinned engine
  source and required together with its basis (`decide_reading`).

### vLLM

- Environment: `VLLM_NO_USAGE_STATS`, `DO_NOT_TRACK`,
  `VLLM_WSL2_ENABLE_PIN_MEMORY=1` (the V2 runner needs a UVA buffer; without it
  no model loads under WSL2, and pinning measured fine there),
  `VLLM_USE_FLASHINFER_SAMPLER=0` (the llm env has no nvcc and FlashInfer
  JIT-builds its sampler during warm-up). Remove the last one only if the recipe
  gains a CUDA compiler.
- Always set: `--served-model-name <id>`, `--max-model-len`, and for
  decisions `--max-logprobs 32`, `--logprobs-mode raw_logprobs` (distribution
  before temperature) and `--enable-prompt-tokens-details`.
- `start` refuses an argv without `--max-num-seqs`, because the chat door reads
  its admission from that flag.

### mlx-lm (`mlx-darwin` text)

- `/v1/models` reports the **resolved** weights directory, so the proxy
  rewrites `model` after checking the Crucible id. A request's `model` would
  otherwise be loaded, and the 409 gate prevents that.
- `/v1/models` answers before the weights load, so `confirm()` sends a
  one-token completion.
- Blocks must state `--decode-concurrency`, `--prompt-concurrency` and
  `--prompt-cache-size` (`REQUIRED_FLAGS`). mlx-lm 0.31.3 batches continuously
  up to `--decode-concurrency`, and the door admits that + 1.
- A closed socket does not stop a non-streamed request. Cancelling stops
  further sends, and the settlement's SIGTERM stops the rest.
- Top logprobs are capped at 11 upstream. `patch_mlx_lm_top_logprobs` raises
  that to 40, and `start` refuses an unpatched env (`llm_env_unpatched`).
  Logprobs are computed after logits processors, so a decision sends its own
  sampling and applies no manifest defaults.

### mlx-vlm server (`mlx_vlm.py`, `mlx_vlm_serve.py`)

- `python -m mlx_vlm server` never put the image in the prompt for dots.ocr
  (216 prompt tokens against 3,464; measured 2026-09-14). Crucible runs the
  in-process path behind `/v1/models` and `/v1/chat/completions`. The file is
  standalone (the env has no `crucible`), and it removes its own directory from
  `sys.path` so `import mlx_vlm` finds the library.
- Only **static micro-batches** are used: every row is inserted before the
  first `next()`, matching `_generate_batch`. Continuous insertion produced
  garbage. Rows are keyed on the processor's `image_grid_thw` (equal grid means
  equal prompt length). Mixed lengths in one batch produce garbage, and so does
  `prefill_batch_size` below the row count.
- The vision tower runs **one image at a time**. Twelve 200-dpi pages in one
  call trip the macOS Metal watchdog. The cost is ~15% on small pages.
- The server loads before it binds, so a 200 from `/v1/models` means ready.
  Images are converted to RGB. `--width` comes from the manifest and is
  required.
- A **page** is a body with no `logprobs`, `top_logprobs` or
  `chat_template_kwargs` and exactly one user message carrying an `image_url`
  part (`is_page_request`). It refuses temperature ≠ 0, top_p ≠ 1, n ≠ 1,
  streaming, unknown fields, any message shape other than one image plus one
  text part, and a wrong model. Private upstream names are checked at start
  (`_check_upstream`). They are safe only because the mlx-vlm version is
  pinned exactly. The page path was checked byte for byte against the
  shipped server on the Mac Studio (dots.ocr, 2026-09-28: same text, usage
  and time).
- Every other body is a **question** (`parse_question`): the decision door's
  system + user turns, 0 to 8 `data:` images in the user turn (in order),
  greedy, `max_tokens` honoured, `chat_template_kwargs` limited to
  `enable_thinking`. A question runs alone (`Asked`), never batched with pages
  or other questions: its prompt length is its own, and mixed lengths in one
  batch produce garbage. Eight ~640 px frames in one call are ~3 MP, far below
  the pages that tripped the Metal watchdog.
- `logprobs: true` returns `choices[0].logprobs.content[]`, one entry per
  generated step (a stop token included), each `{token, logprob, bytes,
  top_logprobs: [{token, logprob, bytes}]}`, the OpenAI chat shape the door
  reads. mlx-vlm 0.7.1's `BatchGenerator(compute_logprobs=True,
  top_logprobs_k=k)` argsorts the whole vocabulary and caps nothing; the
  server caps `top_logprobs` at `MAX_TOP_LOGPROBS` = 40 (mlx-lm's number) and
  refuses more as `too_many_top_logprobs`. `MlxVlmEngine.max_logprobs` is
  that constant.
- mlx-vlm normalises in the logits' dtype (bf16 for bf16 weights), which is
  the error `patch_mlx_lm_fp32_logprobs` fixes for mlx-lm. Here the server
  passes `logits_in_float32` as the row's logits processor, so the log-sum-exp
  runs in float32; greedy argmax is unchanged. Measured on the 9B-vl
  (2026-09-28): the top 40 sum to 1.000.
- Questions over the same images reuse the vision tower's output
  (`VisionFeatureCache`, 16 entries, keyed by the sha256 of the data URIs) on
  model types whose `get_input_embeddings` reads it (`VISION_CACHED_MODEL_TYPES`:
  `qwen3_5`). A decision's prime and questions all carry the same images.
- Measured on the Mac Studio, 9B-vl bf16, 2026-09-28: a question about one
  640×360 frame is ~322 prompt tokens and 0.65–0.77 s; three frames ~770
  tokens and 1.2–1.6 s; the first request after load 4 s.

### llama-server (`llama-windows`, and GGUF blocks on `cuda-linux`)

- `--alias <id>` makes the served name equal the Crucible id, so the proxy
  forwards `model` verbatim.
- Nothing is ever adopted. A taken port is `port_in_use`, and the fix is to retry.
- Stop sends 30 s of graceful `CTRL_BREAK_EVENT`
  (`sigterm_wait_seconds = GRACEFUL_STOP_SECONDS`) and then terminates the
  tree through the shared stop. This deliberately departs from never-SIGKILL,
  which applies only inside WSL2. On `cuda-linux` the same shared stop sends
  SIGTERM for those 30 s and never SIGKILLs: a llama-server there holds CUDA
  inside WSL2 like any other engine.
- Fatal log lines (CUDA OOM, missing CUDA runtime DLL, unreadable GGUF) end
  the readiness wait immediately as `pages_engine_failed`.
- Every block runs `--parallel 1`, so the chat door admits 2
  (`tests/test_chat_admission.py` enforces the flag). Decisions are served with
  no small logprob cap (pre-sampling probabilities, b10970).
- On `cuda-linux` the binary is Crucible's own build of the same tag
  (`scripts/build-llama-server-linux.sh`; ggml-org publishes CUDA builds for
  Windows only), pinned in `hosttools.LLAMA_SERVER_BUILDS` on our `tools`
  release and placed at `<home>/tools/bin/llama-server` by `crucible install llm`.
  Only the binary is ours: it is built against CUDA 13.0 and links cudart and
  cuBLAS dynamically, and the engine starts it with `LD_LIBRARY_PATH` led by the
  llm env's `nvidia/cu13/lib` (`llamacpp.cuda_linux_engine`), the PyPI wheels
  vLLM's torch already pins. The driver's `libcuda.so.1` is the host's. The llm
  install on cuda-linux is whole only with both (`llm_engine_status(config,
  backend)`); a GGUF load without the binary is `env_missing` on the llm env, so
  install-on-submit runs `crucible install llm`, which keeps the env and places
  the binary. Every block names `--n-gpu-layers all`, so a model that does not
  fit fails its load by name instead of spilling layers to the CPU.
- Structured output: `response_format` `json_schema` reaches llama-server
  unchanged and is compiled to its own GBNF grammar (b10970
  `common/json-schema-to-grammar.cpp`), whose capped string repeats a character
  rule that includes the escapes, so a `maxLength` string keeps its newlines
  (the defect that moved vLLM to llguidance does not exist here).
- On `llama-windows` the binary is a pinned **engine subject** (`llamacpp`): `LLAMA_CPP_RELEASE`
  is never read from a listing. The CUDA build is two zips (the build plus
  `cudart`) unpacked into one directory. Every digest is checked before
  anything is placed, and a zip member that escapes the target is refused.
  A pull stages the download beside the engine, verifies and unpacks it there,
  writes the stamp, and only then swaps it in. A forced re-pull that fails
  anywhere (a digest, a missing `llama-server.exe`, a directory held open by a
  running server: `engine_replace_failed`) leaves the engine that was there
  untouched and serving.
- `ollamastore` lets `llama-windows` serve GGUFs already in Ollama's store
  (`OLLAMA_MODELS` is honoured). These are different bytes from the manifest's
  pin, so they carry their own provenance, `ollama:<tag>@sha256:<digest>`,
  and are never filed under the manifest fingerprint. The checks are: a model
  layer exists, the blob exists, and the blob size matches. A tag whose model
  reads images must also carry a projector layer: half a vision model loads
  and then cannot see. Blobs are not
  hashed (up to 19 GB on the load path).

## The decision door (`decide`)

- Labels are bare capitals `A`–`Z` (one token everywhere, and the same string in
  decoded and raw-BPE form). Limits: 26 options, 10 score levels, 8 images,
  and a margin of 4 extra logprobs.
- The state goes in the **system** message and the question in the user turn.
  mlx-lm reuses a hybrid model's cache only from an exact prefix saved at a
  segment end, llama-server checkpoints at the last user message, and vLLM
  caches a plain prefix. Images open the user turn, because templates refuse
  images in system messages.
- The prime's user text is fixed and non-empty; mlx-lm finds the system
  segment end by diffing against an empty user turn.
- Every sampling knob is stated and no manifest `[defaults]` apply, since a
  repetition penalty would move the letters.
- base64 is strict (llama-server silently truncates at the first bad character).
- The wrap serializer has **no return annotation**. An annotation replaces the
  answer schema in OpenAPI with `{}`.
- Whether a model answers images **here** is its manifest's backend block
  (`serves`), never the weights' modalities (`model_text_only` otherwise).
  The refusal's `image_models` detail lists the `decide` candidates whose
  block serves images on this backend, largest first, and the message names
  the `load-model` job for the first. A decision never loads anything.
- On `mlx-darwin` images are answered by `qwen3.5-9b-vl` through mlx-vlm. Its
  block shares `qwen3.5-9b`'s download (`mlx-community/Qwen3.5-9B-bf16`, an
  mlx-vlm conversion that carries the vision tower; mlx-lm drops the tower when
  it loads the same folder). Its memory terms are `computed`: weights are the
  four safetensors files (18,819,722,691 bytes); overhead is the 9B text
  form's 2,069,045,094 plus an image reserve of 362,496,000 = 8 images × 1,600
  patches (640×640 at patch 16) × (1,536 fp32 pixel values × 4 bytes + 10,064
  bf16 activations × 2 bytes: qkv 3×1,152, MLP 4,304, two residuals of 1,152)
  plus 8 × 400 merged tokens × 4,096 × 2 bytes; the vision attention is fused
  SDPA per image, so no score matrix is held. Each 640 px image costs one
  context token per 32×32 px (patch 16, merge 2): 400 for a square, 220 for
  640×360, plus 2 delimiters, so eight are ~3,216 tokens inside `decide`'s
  8,192-token working context, and `kv_bytes_per_token` (32,768: 8 full-attention
  layers × K and V × 4 heads × 256 × bf16) covers them like any other token.
- The 0.8B/2B/4B keep their text-only mlx-lm blocks: a model id has one
  block per backend, and those blocks serve chat and text decisions with
  mlx-lm's continuous batching and prefix cache. Their mlx-community repos are
  mlx-vlm conversions too, so an image form would be a `-vl` alias per size
  (`weights_of` the base, same pin, `engine = "mlx-vlm"`, `serves = ["text",
  "image"]`, `--width 1`), the same shape as the 9B.
- A Mac whose largest fitting decide candidate is a 9B now selects
  `qwen3.5-9b-vl` for `decide` (candidates sort by memory, and the vision form
  is larger), as the PC and Windows lineups already do. The Mac Studio's
  64 GB still selects `qwen3.8-27b-8bit`.
- On vLLM the prefix cache of a hybrid (attention plus mamba) model works in whole
  attention blocks, and vLLM sizes the block per model so its page is at least
  the mamba page: 544 tokens on the 0.8B (measured 2026-09-23: a 121-token prompt
  sent three times cached nothing, a 1,345-token one cached 1,088 from the second
  send), 528 on the 9B and 784 on the 27B 4-bit (each engine log's "Setting
  attention block size to N tokens", owens-pc, 2026-10-09). The prime buys reuse
  only in whole blocks, which is also how `decide`'s working context is sized in
  `capabilityclasses`. A state shorter than one block caches nothing at all.
- Refuse mode carries no `missing_labels` key. Report mode nulls a missing
  label and never invents a number.
- The items form (api.md "The items form") reads a list of items about one
  state. Whether an engine does it in one request is `decide_items_batched`,
  stated with `decide_items_basis` and read by `engines.decide_items_reading`.
  mlx-lm and mlx-vlm answer `POST /v1/crucible/items` with
  `engines/items_forward.py`: every item's lone-question prompt is tokenized
  through the chat template, the common token prefix runs once (in
  2,048-token chunks), and the items' tails are read as rows of batched
  forwards over that cache repeated per row (`read_rows`: right-padded, each
  row read at its own last token, longest tails grouped first; rows per
  forward bounded by `ROW_BYTES`, 512 MiB of repeated cache, and by
  `CHUNK_TOKENS`); the head is applied to those positions only, in float32,
  top-k by argsort. Why rows and not one forward per item, measured on the Mac
  Studio M1 Ultra 9B bf16 2026-10-01: MLX's bf16 matmul leaves its
  matrix-vector kernel past one row (0.8 GB of weights: 1.6 ms at 1 row, 6.0 ms
  at 2-64 rows, 11.7 at 128), so a 9B forward of 2-64 tokens costs ~115-150 ms
  whatever its length and a forward of B rows x T tokens costs what B*T tokens
  in one row cost (3x36: 243 ms; 1x108: 231 ms). Three ~36-token items in turn
  were ~400 ms, as rows ~240 ms. (An earlier note measured batched tails at
  0.207-0.631 s per item and kept them in turn; that was before rows were
  bounded by the bytes of the repeated cache.) The engine also keeps the cache
  of the last 4 STATES it read (`StateCache`, 1 GiB at most, least recently
  used out; a state over the budget is read and not kept), cut where the state
  ends — what every item shares with the open user turn left empty — so the
  next decision about the same state with other questions skips the prefill
  (a held state is used only as a PREFIX of the new one: a recurrent layer
  cannot be trimmed back). The reply's `cached_tokens` says how many state
  tokens came from it. A single item reads everything past the state in its
  own row (no forward of its own for a prefix nothing else shares). On mlx-vlm
  (sequential `read_items`, unchanged) the images are embedded once with the
  shared prefix (`get_input_embeddings` over the shared part plus the longest
  tail, so the rope positions are the lone question's), and a tail is text
  only. vLLM and llama-server answer one prompt per request, so the items go
  through the questions machinery, prime first; vLLM batches them and reuses
  the prefix.
- On mlx-lm the QUESTION form rides the same route (`decide_questions_batched`,
  read by `decide_items_reading(...).questions`, `decide_items.
  decide_questions_on_items`): the open user turn and one question block per
  question are exactly the prompts the chat path sends (same distributions:
  identical to 1e-16 on a repeated state, otherwise within one bf16 logit
  step, 0.125 at |logit| 16-32), one request, no prime; each question's
  `timing_ms.per_question` is that request, `cached_tokens` the state tokens
  the engine reused. Through the chat path a question on mlx-lm was three
  forwards of the ~115 ms floor (mlx-lm prefills the system, user and
  thinking-tail segments apart), plus its token, plus a pipelined token
  nobody reads, plus a fresh detokenizer table (below). Measured 2026-10-01,
  ~300-token state, door logic against the engine: 1/2/3 questions on a
  state the engine holds 155/263/386 ms (chat path 460/1204/1521 before,
  329/788/1021 with the detokenizer patch); on a new state 656/756/864 ms
  (931/1746/1965 before). The chat path is still faster for an EXACT repeat
  of one question (74 ms with the patch): mlx-lm's own prompt cache then holds
  all but its last token.
- `mlx-lm-detokenizer-tokenmap` (self-applied): stock mlx-lm 0.31.3 builds a
  streaming detokenizer's id-to-token table from `tokenizer.vocab` for every
  request, on the one generation thread, and a fast tokenizer's `vocab` is a
  fresh 248k-entry dict on each read: ~150 ms per chat request on Qwen3.5,
  serialising the start of concurrent ones. The patch builds the table once per
  tokenizer (`_crucible_tokenmap`, memoised on the wrapper; only ever read).
  A one-question chat decision on the 9B went 221 -> 73 ms with nothing else
  changed and identical answers.
- mlx-lm gets the route from two env patches: `mlx-lm-decide-items` edits
  `mlx_lm/server.py` (the `/v1/crucible/items` branch at the top of
  `do_POST`, and a branch in `_generate` that drains an active batch and then
  runs the items job on the generation thread, where MLX's stream is), and
  `mlx-lm-decide-items-helper` copies `engines/items_forward.py` in as
  `mlx_lm/_crucible_items.py` (`creates=True`: a missing file is `missing`,
  not "no such package"). Both are `SELF_APPLIED_LLM_PATCHES`: `MlxLmEngine`
  applies them itself at start when they are missing, with the env's own
  python, after the other four are checked as before, so a Mac that upgrades
  needs no `crucible env patch llm` for them (`mlx-lm-detokenizer-tokenmap` is
  self-applied the same way). The helper's marker is
  `ITEMS_VERSION = <n>`: change `ITEMS_VERSION` with every change to
  `items_forward.py`, or an env keeps the older copy. An engine process
  started before its env had the route answers 404, which the door reports as
  `503 decide_not_served` naming the `load-model` job that fixes it.
- An items job holds mlx-lm's generation thread for its whole run (81.5 s for
  250 Briefcase items on the 9B); chat requests queue behind it.
- **Likelihood questions** (api.md "Likelihood questions") read the log-probability of
  every token of a free-text candidate instead of one label token. How an engine does
  it is `decide_likelihood_route` (with `decide_likelihood_basis`, read by
  `engines.likelihood_reading`): `items` (mlx-lm, mlx-vlm: the items route reads a
  `candidates` body, ITEMS_VERSION 3), `prompt-logprobs` (vLLM) or None (llama-server,
  refused `400 likelihood_unsupported_on_engine` before anything waits; b10970 returns
  no prompt-token log-probability: `n_probs` covers generated tokens only and
  `/v1/completions` refuses `echo`). `decide_likelihood_images` says whether images may
  ride along (mlx-vlm only; elsewhere `400 likelihood_images_unsupported_on_engine`).
- A candidate's prompt is the chat template's **open assistant reply**: the question's
  turns plus `{"role": "assistant", "content": <candidate>}` rendered with
  `continue_final_message` (thinking off), so it ends on the candidate's last
  character. Its context is the same turns with the generation prompt. For Qwen3.5's
  template the two agree byte for byte (the open reply renders `<think>\n\n</think>\n\n`
  before the content, exactly as the generation prompt does), so the candidate is
  scored as the first words of the reply. `continue_final_message` cuts the render at
  the stripped content, which is why a candidate may not start or end with whitespace.
- **The boundary** (`items_forward.likelihood_split`, the one owner, used by the Mac
  engines and by the door for vLLM): every prompt is tokenized whole, and a question's
  boundary is the token prefix its context shares with all its candidates. A candidate
  whose first characters merge with the context's last token re-reads that token
  (`boundary_tokens` 1, `BOUNDARY_SLACK`); more than one is `400
  candidate_not_a_reply` (the template does not continue the reply it opened). All of
  a question's candidates are scored from one boundary, so their totals compare the
  same thing.
- **Mac**: one items request for every likelihood question of a decision. The state
  runs once (and is kept in `StateCache` as for items), each candidate is a row of a
  batched forward (`read_likelihood_rows`; mlx-vlm one forward per candidate,
  `read_likelihood`), and the head runs on the candidate's positions only, 64 at a
  time (`SCORE_CHUNK`), float32 log-softmax, the target token gathered. The shared part
  stops one token before the earliest boundary, because a row must read the hidden
  state that predicts its first scored token.
- **vLLM**: `/tokenize` renders the context (`add_generation_prompt`) and every
  candidate (`continue_final_message`) first, on the CPU, so every refusal is made
  before a forward pass; then one `/v1/chat/completions` per candidate with
  `prompt_logprobs: 0`, `return_token_ids: true`, `max_tokens: 1` under the engine's
  admission. The reply's `prompt_token_ids` must equal what `/tokenize` said
  (`engine_error` otherwise), and each scored position's dict is read at the prompt's
  own token id. vLLM 0.29.0 sets `skip_reading_prefix_cache` for any request with
  `prompt_logprobs` (`sampling_params.py` L540-543): the cached positions would have no
  logprobs. So each candidate prefills its whole prompt (state included); vLLM batches
  them, but a long state costs N prefills. vLLM writes -inf as -9999.0
  (`clamp_prompt_logprobs`); the door refuses that as `engine_error` rather than
  reporting a number the model never gave.
- **Raw-text continuation is not built.** Every use so far has a natural request
  (spell this sentence, read this line, title this chapter, say this word), and an
  instruct model scored off its template is a distribution nobody tuned. If one turns
  up: a `mode: "text"` whose context is the state verbatim, `/v1/completions` with
  `prompt_logprobs` on vLLM, `tokenizer.encode` on the Mac, the same boundary code.
- `mlx_vlm_serve.py` compares realpaths when it drops its own directory from
  `sys.path`: run from `/tmp` on macOS (`/private/tmp`), the abspath test missed
  and `import mlx_vlm` found `engines/mlx_vlm.py`.

## Chat defaults (`sampling`)

The request wins. A key the request omits is filled from the manifest. If
neither states it, nothing is sent. A key present in the request counts as
stated even when it is `null`. Every response carries `X-Crucible-Sampling`
naming the source of each key. It is a header because the proxy passes bodies
through verbatim. `thinking` travels in `chat_template_kwargs`, and Crucible
owns only that one key there.

## Prefix reuse on the chat door

Every engine reuses a chat's prompt prefix with nothing asked of the client, and
every one reports it the same way, which the proxy passes through untouched:
`usage.prompt_tokens_details.cached_tokens` (the SDK's `ChatUsage.cachedTokens`).

- **vLLM**: prefix caching is on for every block (vLLM 0.29.0's default; no
  manifest turns it off, and Qwen3.5 then runs mamba cache mode `align`).
  Crucible starts it with `--enable-prompt-tokens-details`, so the count is in
  every reply. Reuse is in whole blocks of 528 tokens (9B) or 784 (27B 4-bit):
  a prompt shares nothing until its common prefix passes a block boundary.
  vLLM logs a rolling `Prefix cache hit rate` every 10 s in the engine log.
- **mlx-lm**: its `LRUPromptCache` (`--prompt-cache-size` sequences) keeps a
  hybrid model's cache only at a segment end: the end of the system messages,
  the end of the user turn, the thinking tail. So a chat reuses exactly its
  system messages, whole, and nothing inside them. Eviction takes assistant
  and user entries before the system one. It logs `Prompt Cache: N sequences`
  per type. (`patch_mlx_lm_cache_counters` is not a counter of hits: it forces
  the caches' bookkeeping arrays to evaluate so a long decode does not exhaust
  Metal's buffer count.)
- **llama-server**: one slot (`--parallel 1`), `cache_prompt` on by default; it
  checkpoints the recurrent state at the start of the last user message, so a
  chat reuses everything before its last user turn.

What a client does to be reused: everything that is the same across requests
first and byte-identical (system prompt, rules, examples), the part that changes
last, in the last user message. On vLLM, make the shared part longer than a
block or it buys nothing.

## Prefill (`prefill`)

A chat body may carry `"prefill": "<text>"`: the answer begins with that text
and the model writes on from it. The reply's `content` is what it wrote after
the prefill; the client joins the two. Crucible takes the member out and sends
the engine the messages plus `{"role": "assistant", "content": <prefill>}` with
`continue_final_message: true` and `add_generation_prompt: false`
(`crucible/prefill.py`). It is a field of its own because the engines read a
bare trailing assistant message three ways: llama-server b10970 continues it,
vLLM 0.29.0 closes it and answers after it unless told, and mlx-lm 0.31.3
always closes it. Whether an engine can is `chat_prefill` with its
`chat_prefill_basis`, read by `engines.chat_prefill_reading`. Refused by name:

- `prefill_not_served` (400): mlx-lm and mlx-vlm (no way to continue a message),
  and every upstream model. A queued chat is refused from its manifest before
  it waits or loads anything, and again from the resident engine.
- `prefill_with_thinking` (400): thinking not resolved off (request or manifest
  `[defaults]`). Qwen3.5's template closes an empty `<think></think>` before a
  continued message whether thinking is on or off, so a prefilled answer never
  thinks; the request has to say that is what it wants.
- `prefill_with_grammar` (400): `response_format` (any type but `text`),
  `structured_outputs`, `guided_*`, `grammar` or `json_schema`. The grammar
  constrains the answer from its first generated token, not from the end of
  the prefill, so the engine would write a whole new document after it.
- `prefill_conflict` (400): the body also ends in an assistant message, or
  states `continue_final_message` / `add_generation_prompt` itself.
- `invalid_request` (400): not a non-empty string, or it begins or ends with
  whitespace: Qwen3.5's template trims an assistant message, so `{"answer": `
  is continued as `{"answer":` (rendered with the 9B's tokenizer, 2026-10-10).

Prefill and prefix reuse agree: the prefill comes after everything else, so a
prompt's shared prefix is the same with or without it. On mlx-lm (were it
served) a final assistant message also turns off its segment split, which is
part of why it is refused there rather than patched in.

## Page requests (`pages`)

`GET /v1/info` `pages_engine.request` is the single definition. The engine
must not be detectable from the answer. dpi 200, `max_pixels` 11,289,600,
`max_tokens` 8192 as a ceiling (the longest accepted page over 18,202 pages
was 7,677), temperature 0 (measured as good as 0.1 over 516 pages, and
deterministic), the model-card prompt byte for byte, image first then prompt,
PNG data URI only, concurrency 12. A page that returns
`finish_reason: "length"` must be re-read at the full ceiling.

## Interpreters (`interpreter`)

CPython comes from astral-sh python-build-standalone `install_only` at a pinned
release, verified against that release's `SHA256SUMS` digest, and stamped so it
is fetched only once. It never comes from PATH. The Windows tree has no `bin/`
(`interpreter_python`). The `-shared` infix is retired upstream. The 3.12 row
exists for the Higgs recipe (sglang-omni 0.1.4). The server runs 3.11.
Downloads are unpacked beside the destination and moved last, so a failure
leaves the machine unchanged. The install progress line is a declared JSON
wire, not a log scrape.

## Foundry lineup (`lineup`)

`foundry-lineup.json` is read from the manifests (`[model]`, `[local]`) and
`capability.classes_for_model`. Rows without `[local]` are omitted. Schema 3
(2026-10-09) dropped `floors` and each row's `minimum` and `minimumFor` with the
manifests' `[local] minimum_for`: no class has a floor (docs/VERB-SIZING.md
section 7). `--check` ignores `generated_from`, which always trails by one
commit.
