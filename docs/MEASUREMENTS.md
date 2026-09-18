# Measurements — what was watched on a card, and when

Every number in `crucible/models/*.toml` is one of three things: **measured** on a
machine, **computed** from a checkpoint's `config.json`, or **declared** because
nobody has looked yet. The manifests say which, per term, in a `basis` field.
This file is the other half of that: the runs themselves — what was done, on what
hardware, what came out, and what the run could NOT answer.

**Why it exists.** Twice now a number has been re-derived from scratch because
the measurement behind it was in a comment in one manifest and a session's
scrollback, and the next session could not tell a measured number from a
plausible one. A constant with no citation is an invented constant
(`verification-must-be-falsifiable`). This is where the citation lives.

**What goes in here.** Anything read off a card: a load that succeeded, a load
that failed, an A/B of a flag, a throughput run. **Failures are measurements** —
the −0.19 GiB below is the most useful number on this page, because it is the one
that disproved a thing the manifests asserted.

**What does NOT go in here.** Design (that is `FITS-AND-THE-CARD.md`), rulings
(`MODEL-CHOICE.md`, `PHASE*.md`), or the numbers themselves as the source of
truth — the manifests own those. This is the provenance, not the value.

## The machines

| host | accelerator | pool | backend | notes |
|---|---|---|---|---|
| owens-pc-wsl | RTX 3090 Ti | 24_564 MiB (23.99 GiB) | `cuda-linux` | inside WSL2; **shares the card with the Windows desktop** |
| owens-mac | M1 Ultra, Mac Studio | 64 GiB unified | `mlx-darwin` | no other tenant of consequence |
| owens-pc | RTX 3090 Ti | as above | `llama-windows` | **nothing has ever been measured here** |

The first row's last column is the whole of section 5 of `FITS-AND-THE-CARD.md`
and most of this page. The desktop is a live tenant whose size Crucible does not
control and, until the work below, did not consult.

## How to take one

* `scripts/measure-llm-memory.sh <model>` — the engine's **share** of the
  accelerator, at rest and under a request that fills `context_default`. Works on
  both backends. This is where `memory_bytes_estimate` comes from.
* `scripts/calibrate-kv.sh <model> <util> [context]` — two-point calibration
  (`FITS-AND-THE-CARD.md` section 4): the KV slope and the intercept, per model
  per host, read off vLLM's own DEBUG memory profiler. cuda-linux only, because
  the whole arithmetic it measures is vLLM's. Results in the 2026-09-18 section.

Both want an idle machine. On the PC, check `pgrep -af train_lora.py` first and
refuse if training is up — WSL pytest and training share 13 GB.

---

## 2026-09-12 — the founding measurements, PC and Mac

vLLM 0.29.0 / torch 2.13.0+cu130 inside WSL2; mlx-lm 0.31.3 / mlx 0.32.2 on the
Mac. `nvidia-smi` sampled every 2 s. These are the runs every `memory` table in
`models/` is quoted from.

**`qwen3.5-9b`, cuda-linux, util 0.84, context 16384**

| | |
|---|---|
| card before the engine started | 1_192 MiB |
| weights on the card | 17.66 GiB |
| non-KV demand | 19.02 GiB |
| KV pool | 1.09 GiB = 27,443 tokens |
| card at rest, resident | 20_986 MiB |
| card PEAK, 11,622-token completion | 21_172 MiB (3_140 MiB free) |
| **engine's share at that peak** | **19_980 MiB = 19.52 GiB** |

**`qwen3.8-27b-4bit`, cuda-linux, util 0.86, context 16384**

| | |
|---|---|
| card before the engine started | 1_188 MiB |
| weights on the card | 17.68 GiB |
| non-KV demand | 19.12 GiB |
| KV pool | 1.51 GiB = 18,811 tokens |
| card at rest, resident | 21_502 MiB |
| card PEAK, 15,380-token completion | 21_819 MiB (2_493 MiB free) |
| **engine's share at that peak** | **20_631 MiB = 20.15 GiB** |

**The one thing this pair proved that nothing else has.** KV on the 27B really
costs **86_251 B/token** where `config.json` arithmetic says 65_536 — **24%
light**, because vLLM pads the attention page up to the linear layers' recurrent
state. That is the single result behind `FITS-AND-THE-CARD.md` section 4's
insistence on measuring the slope instead of computing it, and it is why
`qwen3.5-9b`'s `kv_bytes_per_token = 32_768` is marked `computed` rather than
trusted.

**Mac, same day, via mlx's own allocator**

| model | after load | peak under a full-context completion |
|---|---|---|
| `qwen3.5-9b` | active 17.91 GB | 20.38 GB at 12,198 tokens |
| `qwen3.8-27b-4bit` | active 14.09 GiB | 31.55 GiB at 98,220 tokens |

The 27B was also measured from outside by `measure-llm-memory.sh` at +32_116 MiB
— two methods, 0.6% apart. **Computed would have said 22.5 GB, 34% under**, and
the gap is neither weights nor KV: mlx-lm prefills in 2048-token steps and takes
logits over the whole step against a 248,320-token vocab, so one step's logits
are 2.03 GB. A long-context MLX model cannot have this number computed.

**Two failures worth keeping from that day**, both now prevented in code:

* `UVA is not available` — vLLM 0.29's V2 runner needs pinned host memory and
  WSL defaults `VLLM_WSL2_ENABLE_PIN_MEMORY` to 0. Died 45 s in.
* FlashInfer's sampler JIT-builds a kernel and the `llm` env has no nvcc. Died
  two minutes in, **after** the KV cache was allocated.

Both are set by `crucible/engines/vllm.py:environment()`. They cost this page's
own harness a wasted run on 2026-09-18 when it spawned vLLM without them.

---

## 2026-09-16 — the same argv, two answers

`qwen3.8-27b-4bit`, unchanged `engine_args`, minutes apart on the PC:

    Available KV cache memory: -0.08 GiB    -> engine_failed
    Available KV cache memory:  2.44 GiB    -> clean load

**Nothing about the model or the flags moved. The desktop did.** This is the
observation that opened `FITS-AND-THE-CARD.md` section 0a, and it is the reason
the utilisation numbers in the manifests — each chosen as "the smallest that
starts" — were never as stable as they read.

Also measured that day, **how Ollama handles the same question** (section 6.1),
kept here because it is the shape Crucible must not copy:

* `num_ctx: 1_000_000` on a 262,144-token checkpoint → **HTTP 200**, silently
  clamped to 262144, no field on the response saying so.
* ~5,628 tokens of prompt at `num_ctx: 512` → **HTTP 200**, `prompt_eval_count`
  1026, the instruction at the front thrown away, and a confident wrong answer.

Never an error. An OOM would have been more honest.

---

## 2026-09-17 — `qwen3.5-9b` will not load at its own manifest's number

Owen, from BookForge: *"`crucible@owens-pc-wsl` could not load qwen3.5-9b: vllm
exited 1 before it was ready."*

    --gpu-memory-utilization 0.84  --max-model-len 16384  (the manifest's own)
    Model loading took 16.8 GiB memory
    Graph capturing finished in 5 secs, took 0.09 GiB
    Available KV cache memory: -0.19 GiB
    ValueError: No available memory for the cache blocks

Card state around it: total 24_564 MiB, desktop holding ~3_330 MiB, so ~21_234
MiB free. `available_bytes(total, allowance)` answered **21_492 MiB** — a budget
**258 MiB larger than the card had**.

Note the direction of the surprise: weights were **16.8 GiB**, 0.85 GiB *less*
than the 17.66 GiB of 2026-09-12, because `--language-model-only` now keeps the
vision tower off the card. The load got cheaper and still failed. Whatever
consumed the difference is not the model.

**Two things this run establishes and one it does not.**

Establishes: the manifest utilisations are calibrated against a **quiet desktop**
— both founding runs started from ~1_190 MiB — and Owen's desktop is now 3_000
to 3_300 MiB. And `--language-model-only` landing did not buy headroom, exactly
as `qwen3.5-9b.toml` predicted in prose ("it buys DEPTH, not headroom").

Does not establish: **why**. Read from the vLLM 0.29.0 source that produced the
number (`v1/worker/gpu_worker.py`, `v1/worker/utils.py:request_memory`,
`utils/mem_utils.py:memory_profiling`):

    requested      = total x util                       # the desktop is CHECKED, never subtracted
    total_consumed = free_at_init - free_after_profile  # a WHOLE-CARD delta
    non_kv         = total_consumed + transient_peak_headroom
    available_kv   = requested - non_kv - cudagraph_estimate

The baseline desktop cancels out of `total_consumed`, so on this arithmetic a
desktop that is merely *large* should not matter — only one that **grows while
vLLM profiles**, which `total_consumed` charges to the KV pool 1:1. That is a
hypothesis until measured, and measuring it is the 2026-09-18 section.

---

## 2026-09-18 — two-point calibration on the PC

*Owen authorised the card: "both cards are free", then "go ahead and do the test
and fix this the right way".*

**Harness.** `scripts/calibrate-kv.sh <model> <util> [context]` — runs
the engine's real argv with `VLLM_LOGGING_LEVEL=DEBUG`, which makes
`MemoryProfilingResult.__repr__` print every term of the arithmetic above, and
samples `nvidia-smi` at 1 Hz throughout so the desktop's own movement during the
load is on the record. Engine args and engine env are both **asked for** —
`load_manifest(...).backends['cuda-linux'].engine_args` and
`VllmEngine.environment()` — never retyped, so the calibration measures the
engine Crucible actually runs.

`GPU KV cache size: N tokens` beside `Available KV cache memory: X GiB` gives
**bytes per token by division** — measured, on this card, with no architectural
knowledge at all. That is section 4's slope, and it needs no second context:
varying the utilisation moves the pool and holds everything else, which is a
cleaner second point than varying the context.

### Finding 1 — inside WSL2, CUDA cannot see the Windows desktop

Three samples, seconds apart, same card, nothing running:

| | `nvidia-smi --query-gpu=memory.free` | `torch.cuda.mem_get_info()` free |
|---|---|---|
| 1 | 21_254 MiB | 23_285 MiB |
| 2 | 21_352 MiB | 23_285 MiB |
| 3 | 21_336 MiB | 23_285 MiB |

nvidia-smi's figure moves with the desktop. **CUDA's does not move at all** —
24_564 − 23_285 = 1_279 MiB is a fixed driver reservation, not a live reading.
Windows' own nvidia-smi agrees with WSL's nvidia-smi (3_286 vs 3_319 MiB used),
so this is not a Windows-vs-Linux disagreement: it is that the guest's **CUDA
runtime is blind to the host compositor**.

**Why this is the root of it.** vLLM's entire budget is built from
`mem_get_info` — `request_memory`'s `free >= total x util` gate, and
`total_consumed`'s before/after delta. So on this host:

* the startup gate can **never** refuse on account of the desktop; it is
  comparing against a number that does not include it;
* and `total_consumed` is blind to the desktop's *baseline* but NOT to whatever
  the driver surfaces while the profile runs — which is charged to the KV pool
  1:1.

`crucible/accelerator.py` reads `nvidia-smi`, and its own comment says
*"`memory.free` under WSL2 **is** accurate for the whole card"*. That is true,
and it is the right probe. What was missing is that **vLLM never sees it.** The
engine cannot discover the desktop for itself, so Crucible has to tell it.

### Finding 2 — the two points, and the slope

`qwen3.5-9b`, cuda-linux, context 16384, `--max-num-seqs 16`, manifest args
otherwise. The compile cache was warm, so each load is ~60 s.

| | util 0.84 | util 0.90 |
|---|---|---|
| initial free (CUDA) | 22.74 GiB | 22.74 GiB |
| requested = total x util | 20.15 GiB | 21.59 GiB |
| weights | 16.8 GiB | 16.8 GiB |
| total consumed (mem_get_info) | 18.01 GiB | 17.93 GiB |
| torch peak increase | 0.21 GiB | 0.21 GiB |
| **non-KV (the intercept)** | **18.21 GiB** | **18.14 GiB** |
| **available KV** | **1.94 GiB** | **3.37 GiB** |
| **GPU KV cache size** | **51,738 tokens** | **89,680 tokens** |

**Slope**, the two-point form of `FITS-AND-THE-CARD.md` section 4:

    (3.37 - 1.94) GiB / (89,680 - 51,738) tokens = 40,470 B/token

and each point taken alone agrees — 40,259 and 40,346 B/token. The manifest's
`kv_bytes_per_token = 32_768`, computed from `config.json`, is **23% light**.
That is the same error, in the same direction, and within a point of the same
size as the 24% the 27B measured on 2026-09-12. **Computed KV slopes are light
on this engine, on both of the models anyone has checked.**

**Intercept**: 18.14–18.21 GiB, so 18.175 ± 0.035 GiB — stable to 0.4%
run-to-run. The manifest's terms say 19.02 GiB (17.66 weights + 1.36 overhead).
The gap is **0.845 GiB**, and `--language-model-only` removed a vision tower the
manifest itself measures at **0.85 GiB**. The intercept fell by exactly the
tower, and the manifest was never updated. Both of its `cuda-linux` memory terms
are stale, in the *safe* direction for fitting and the *wrong* direction for
knowing what is going on.

### Finding 3 — why 2026-09-17 failed and tonight's identical argv did not

Same model, same flags, same `0.84`. Last night `-0.19 GiB`; tonight
`+1.94 GiB`. nvidia-smi says the desktop was about the same size both times
(~3.0–3.3 GiB), so the desktop's **size** is not the discriminator.

What differs is the **length of the profiling window**. Last night's load was a
cold `torch.compile`: *"Initial profiling/warmup run took 65.00 s"*, better than
two minutes end to end, at 23:08 with Owen working. Tonight the compile cache
was warm and `Memory profiling takes 3.51 seconds`. `total_consumed` is a
whole-card delta across that window, so **the longer the load, the more of
someone else's allocation is charged to our KV pool.** Last night that was
~2.1 GiB, and 2.1 GiB is the whole difference between the two runs.

This also explains 2026-09-16's identical-argv flip on the 27B, which needed no
explanation of its own after all.

**So the defect is not "the fraction is too small".** Raising it would buy a
bigger pool on a quiet machine and fail again on a busy one, because the term
that moves is not in the fraction at all.

### Finding 4 — `--kv-cache-memory-bytes` removes the mechanism

vLLM 0.29 prints the suggestion itself on every load:

    Replace gpu_memory_utilization config with `--kv-cache-memory=1969924260`
    (1.83 GiB) to fit into requested memory, or `--kv-cache-memory=4750391296`
    (4.42 GiB) to fully utilize gpu memory.

and `config/cache.py` states the semantics: `kv_cache_memory_bytes` **"(when
not-None) ignores gpu_memory_utilization"**, and `gpu_worker.py:527` **skips
memory profiling entirely** when it is set.

That is the whole defect removed rather than tuned:

* no `requested = total x util`, so the fraction-of-a-shared-card is gone;
* no profiling window, so **nobody else's allocation can be charged to our
  pool** — the exact mechanism of Finding 3.

The cost is that Crucible must now know the intercept and the slope itself,
which is precisely what the `[memory]` tables hold and what section 4 measures.
The two halves were built for each other.

> `--gpu-memory-utilization` still has one job: `request_memory()` runs in
> `init_device` regardless and raises if `free < total x util`. It stays a
> **gate**, derived from the same budget, and stops being the sizing knob.

### Finding 5 — the pool stated in bytes, confirmed

    --kv-cache-memory-bytes 2800000000
    -> "reserved 2.61 GiB memory for KV Cache as specified … skipped memory profiling"
    -> GPU KV cache size: 69,416 tokens
    -> Maximum concurrency for 16,384 tokens per request: 4.24x

2_800_000_000 / 69_416 = **40,336 B/token**, against a prediction of 69,200 to
69,550 tokens from the two-point slope. It landed in the middle. The flag does
what its documentation says, and it is the only one of the four derivations
whose numerator is exact rather than a two-decimal GiB, so **40,337 B/token is
what went into the manifest**.

A fourth point, taken later the same night when the harness was re-run from its
repo home to prove that copy works: `--kv-cache-memory-bytes 2_500_000_000` ->
62,086 tokens = **40,266 B/token**. Four derivations now span 40,259 to 40,470,
and the manifest carries 40,337.

### What these runs changed

**`models/qwen3.5-9b.toml`, `[backends.cuda-linux.memory]`** — all three terms,
`basis` from `computed` to `measured`:

| term | was | now | why |
|---|---|---|---|
| `weights_bytes` | 18_962_006_507 (17.66 GiB) | 18_038_862_643 (16.80 GiB) | `--language-model-only` landed 2026-09-15 and these were never re-taken |
| `overhead_bytes` | 1_460_288_880 | 1_476_395_008 | the rest of the measured 18.175 GiB intercept |
| `kv_bytes_per_token` | 32_768 (computed) | 40_337 (measured) | the computed figure is 23.1% light |

`memory_bytes_estimate` is deliberately **not** re-taken: it feeds the
accelerator guard's "does this fit", where being 0.7 GiB conservative is the
safe direction, and lowering a guard in the same change that lands a calibration
is two changes wearing one hat. The terms now come to 3.7% under it, inside the
parser's 5% tolerance, and the manifest says why in place.

**`crucible/vram.py`** (new) — sizes the pool against the card:

    budget = min(total − desktop_allowance, free)        # nvidia-smi's free
    pool   = budget − weights − overhead
    pool   = min(pool, kv_per_token x context x max_num_seqs)
    refuse if pool < kv_per_token x context

No third term, per `capability.py`'s ruling 3 — the allowance IS the margin.
`capability.decide()` is untouched and stays on TOTAL, per its ruling 2: a
capability is a fact about the host, and a browser open during `crucible
install` must not disable TTS for good.

A block with **no `[memory]` terms is left exactly as its manifest states** —
`dots-ocr`'s `0.5` is a deliberate budget so page reading can share the machine,
not a calibration to be improved.

### Finding 6 — the fix loaded, with its own numbers

Not a re-run of the calibration: `plan_vllm_memory` was called against the live
card through the real code path, and the engine was started with exactly the
argv it composed.

    live card : 24_564 MiB total, 21_567 MiB free
    plan      : KV pool 2.81 GiB of a 20.99 GiB budget (18.17 GiB weights+overhead),
                74_887 tokens at 40_337 B/token (measured) x 16 in flight
    flags     : --kv-cache-memory-bytes 3020737741 --gpu-memory-utilization 0.8749

and vLLM answered:

    reserved 2.81 GiB memory for KV Cache … skipped memory profiling
    GPU KV cache size: 75,021 tokens
    Maximum concurrency for 16,384 tokens per request: 4.58x

**Predicted 74,887 tokens, got 75,021 — 0.18% out.** The desktop was under its
allowance here (2_745 MiB against 3_072), so the budget came from the allowance
and not the measurement, which is the branch that holds room open for the
desktop to grow back into. Both branches are covered in `tests/test_vram.py`.

**And the budget was respected.** nvidia-smi sampled at 1 Hz across the load
peaked at **23_471 MiB of 24_564**, so 1_093 MiB stayed free with the desktop
on the card. Engine share at that peak is 23_471 − 2_745 = **20_726 MiB =
20.24 GiB**, against the 20.99 GiB it was given. The number Crucible handed the
engine is a number the engine stayed inside.

---

## What is still unmeasured

| what | where it bites | why not yet |
|---|---|---|
| **every `llama-windows` block** | all three manifests | no load has ever been taken on a Windows card; `basis = "declared"`, overhead is Foundry's 1.5 GB `OVERHEAD_GB` |
| `qwen3.5-9b` KV slope, cuda-linux | `kv_bytes_per_token = 32_768` | `computed`; the 27B proves computed slopes run 24% light |
| `qwen3.5-9b` KV slope, mlx-darwin | same constant | same, plus the MLX prefill residual the 27B block warns about |
| `dots-ocr`, cuda-linux | `memory_bytes_estimate` **is a budget, not a sum** | 0.5 x the card is a deliberate choice so pages can share the machine; it has no terms and cannot be split until measured |
| `qwen3.8-27b-4bit`, mlx-darwin | no `[memory]` table at all | one point cannot separate an intercept from a slope, and the 2.03 GB/step prefill residual may be either |
| `qwen3.8-27b-8bit`, both arms | whole manifest | written 2026-09-17, never loaded |
| the Mac's 9B peak | `basis = "computed"` | the block asks for its own re-take; wants an idle Mac |
