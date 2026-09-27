# Proposal: a measurement ladder that finds out what a card can do

**STATUS: PROPOSAL, 2026-09-26. Nothing here is built.** Owen decides before any of it is.

Fresh-install #48. kylies-pc's GTX 1660 SUPER is a 6 GB Turing card (sm_75). Owen:

> *"this gpu is also not capable of most things, so itll be useful to figure out how we can
> know what its capable of without direct measurements. or maybe we could run a measurement
> ladder that tests what the gpu is capable of upon install"*

His standing rule decides the shape of the answer: **Crucible must be idiot proof.** Nobody finds
out what a card can do by a job failing on it; Crucible knows first and says so plainly.

---

## 1. What the declared half already answers, and what it cannot

The declared half is built in the same change as this proposal (package G). `crucible capability`
now asks two questions of every candidate, in this order:

| question | answered by | example on sm_75 |
|---|---|---|
| **can this card START it at all?** | `backend.card_features` (the probe's `compute_cap` against floors read in the pinned vLLM and torch) and `engines.vllm.card_needs` (what the block's stated dtype demands) | `qwen3.5-9b` and both Qwen3-ASR blocks run `bfloat16` under vLLM, and vLLM 0.29.0 refuses bf16 below 8.0 at its first line. Refused: *"needs bf16 (compute capability 8.0 or newer), and this card is sm_75 (7.5)"* |
| **can it HOLD it?** | the memory walk, total bytes against declared estimates | `whisper-tiny` 1.6 GiB of 3.0 GiB available: yes |

Both are arithmetic on facts nobody had to measure. What they **cannot** say, and why a ladder is
the only honest source for each:

1. **How fast.** A card that can start and hold a model can still be too slow to be worth
   offering. Nothing in any manifest says what a GTX 1660 SUPER does per second, and nothing in
   the repo could be read to find out. The one number there is, is measured: 60 s of 48 kHz
   audio through `sigma` in **24.6 s wall with a cold load, about 2.4x real time** on that card
   (FRESH-INSTALL-KYLIES log, "The first real job").
2. **Whether an engine that does not refuse still works.** Three live cases:
   - the torch workers (the aligner, Qwen's own `qwen-asr` package) **emulate** bf16 below 8.0
     rather than refusing (`torch.cuda.is_bf16_supported`, `including_emulation`, torch
     `cuda/__init__.py` L244-251). They will run; how slowly is unknown.
   - vLLM with `dtype auto` (`dots-ocr`, `qwen3.8-27b-4bit`) **falls back to float16** with a
     warning (`vllm/config/model.py` L2268-2309). Same memory; whether a bf16-trained checkpoint
     reads pages or translates as well in float16 has never been looked at.
   - Higgs v3 under SGLang-Omni: its stages default to `dtype="bfloat16"`
     (`sglang_omni/models/higgs_tts/stages.py` L384, L506), and the env carries flash-attn-4 and
     flashinfer, which vLLM itself floors at 8.0 on Turing (`flashinfer.py` L506-510: "currently
     broken on SM75"). Whether narration starts on a 24 GB Turing card (Titan RTX, Quadro RTX
     6000) is unknown, and the declared half deliberately does not guess.
3. **What this card's desktop really holds.** `desktop_allowance_bytes` is 3 GiB by declaration.
   On a 6 GB card that is half the card, and it decides most of kylies-pc's answer: with it,
   `align` is refused at 5.5 GiB against 3.0 available; the card may well have more.
4. **Whether the manifest's memory estimates hold on this card.** Every figure outside Owen's
   3090 Ti and Mac Studio is carried from them. PHASE9 section 4 records Owen's ruling that the
   estimates are not to be measured (*"Use the current settings for each"*), and this proposal
   does not overturn it; see call 4 below.

## 2. The ladder

**Rungs, cheapest first. Each rung runs only if the one before it passed, and a rung that fails
is a measurement, not an error** (MEASUREMENTS.md: "Failures are measurements"). Every rung runs
the real code path: the installed env, the engine Crucible starts, the argv it composes, never a
harness that re-types them (the 2026-09-18 calibration lost a run to exactly that).

| rung | what it runs | what it answers | what it costs, where known |
|---|---|---|---|
| **0. the card** | nvidia-smi only: name, memory.total, compute_cap, driver, uuid; `memory.used` sampled at 1 Hz for a few seconds with nothing of ours on the card | the facts; the desktop's real size on this machine, as `calibrate-kv.sh` already samples it | seconds; no GPU work |
| **1. the env** | in each INSTALLED env's python: `torch.cuda` initialises, allocates, and runs one small matmul in each dtype the env's engines use (float16, float32, and bfloat16 where the card has it or torch emulates it) | the env works on this card at all (a torch wheel built without this arch fails here, not in a job); the emulated-bf16 penalty on this card, which is the aligner's question | seconds per env; no model |
| **2. one real job per installed type** | the model the capability walk SELECTED for each enabled class, on one short fixed input shipped with the release (a spoken clip for `asr`/`align`, a 60 s clip for `rvc`/`denoise`, one page for `pages`, one paragraph for the text classes, one sentence for `tts`), through the ordinary job door | starts, finishes, output well-formed; load seconds; peak engine share (nvidia-smi at 1 Hz, as `calibrate-kv.sh`); speed as the type's own unit (realtime factor, tokens/s, pages/min) | dominated by the cold load. Measured: rvc 24.6 s wall cold on the 1660 SUPER; the 9B warm ~60 s and cold `torch.compile` 65 s of profiling alone on the 3090 Ti (MEASUREMENTS 2026-09-17/18); narrator ~110 s to healthy (`residency.DEFAULT_READY_TIMEOUT_SECONDS`' note) |
| **3. on demand only** | for a vLLM class: two-point KV calibration at the class's working context (`scripts/calibrate-kv.sh`, FITS-AND-THE-CARD section 4) | this card's real intercept and slope | minutes (two loads) |

**Not in the ladder:** anything the capability walk refused (a refused model is not tried to see
if the refusal was right; the declared rule is the engine's own and cannot be argued with by a
run), anything bigger than the selected model, training, and stress.

**How long it takes.** Rungs 0 and 1 are seconds. Rung 2 is one cold load per installed type
plus a short job, so it scales with what was installed: on kylies-pc today (`asr`, `rvc`,
`denoise` enabled) that is three small models, and the one measured figure among them is rvc's
24.6 s. A total is not stated here because it would be an invented number; the first ladder run
states it, and that run is the one to set a budget from (call 2).

## 3. What it records

One file per card, `<CRUCIBLE_HOME>/ladder/<gpu uuid>.json`, rewritten whole by each run:

- **the key**: gpu uuid, name, compute_cap, driver version, Crucible version, and each env's
  provenance (the fingerprint `crucible doctor` already prints). A change to any of them makes
  the record STALE, and `doctor` says so the way it says `capability_stale` today, by comparing
  values rather than dates.
- **per rung, per subject**: `basis = "measured"`, `measured_at`, `outcome` (`passed`, `failed`,
  `interrupted`, `skipped`), the numbers the rung answers, the input's sha256, the card's
  contention during the run (foreign `memory.used`, `utilization.gpu` samples), and on a failure
  the engine's first error line and a pointer to the full log.
- `basis` is MEASUREMENTS.md's word and means the same thing: somebody (here, Crucible) watched
  the card. A number that came off a contended run is marked `contended` and never used as if it
  were clean.

For the reference machines (Owen's PC, the Studio, kylies-pc), a run's summary is pasted into
MEASUREMENTS.md by hand, as every measurement there is. The ladder file is the host's own record;
MEASUREMENTS.md stays the repo's provenance.

## 4. How it feeds `capability`

The rule is **a measurement may only add a refusal or confirm a yes; it never argues with a
declared refusal.**

| the ladder says | capability does |
|---|---|
| rung failed for a reason that is the card's (env will not initialise; engine died loading; job failed) | the class is disabled with `measured:` in front of the reason, the date, and the first error line. The person reads *"cannot transcribe — this card failed Crucible's own test run of whisper-tiny on 2026-09-26: …"* instead of finding out from a job |
| rung passed | the row says so: *"measured on this card: 2.4x real time"*. `summary` carries the speed in the person's words |
| rung passed, but slower than the type's floor | reported, and refused only if Owen sets a floor per type (call 3). No floor is invented here |
| rung interrupted (the card got busy) | nothing changes; the rung is retried later |
| a declared refusal (bf16 on sm_75) | stands. The ladder never runs a refused model |
| a stale record | ignored for decisions and reported by `doctor`, until the next run |

Memory numbers from rung 2 are **recorded and not consumed** unless Owen reverses PHASE9's
ruling (call 4). The declared estimates keep deciding fit.

## 5. On a card somebody is using

This is the part that decides whether the ladder is safe to run at install at all. Everything in
it is an existing rule of this repo applied to one more caller.

1. **Never compete, never evict.** Before each rung, the accelerator guard's own test
   (`accelerator.guard`): a foreign compute app over the 1 GiB floor, or unattributed VRAM past
   the desktop allowance under WSL2, and the rung does not start. The ladder records `waiting:
   the card is in use (pid …)` and tries again later. Crucible never evicts anything, and a
   measurement is no exception.
2. **It is a job in Crucible's own queue, at the lowest priority.** It never runs beside a
   client's job, it yields the moment one arrives (the current rung is `interrupted`, not
   `failed`), and a person can cancel it from the tray.
3. **Watch the card while it runs.** nvidia-smi at 1 Hz, as `calibrate-kv.sh` does. If somebody
   else's memory grows during a rung, the rung is `interrupted` and its numbers are thrown away.
   MEASUREMENTS.md 2026-09-18 Finding 3 is why: a growing neighbour is charged to our numbers 1:1
   and would make the card look worse than it is. Loads use `--kv-cache-memory-bytes` where the
   engine takes it, which removes vLLM's profiling window, the mechanism that charged it.
4. **The desktop is measured, not assumed, under WSL2.** CUDA inside the guest cannot see the
   Windows desktop (MEASUREMENTS Finding 1); rung 0 reads nvidia-smi, which can.
5. **A game at full tilt makes speed numbers wrong, not the card.** `utilization.gpu` is sampled
   with every number; a run under another tenant's load is marked `contended` and not trusted for
   speed.
6. **Where the person is.** At install the ladder runs after the env is built and says what it is
   doing in one line ("testing what this card can do with transcription, about a minute"). On a
   machine that is somebody's desktop, like kylies-pc, it may be better run the first time the
   card is idle rather than immediately (call 1).

## 6. Where it runs from

- `crucible install <type>`: rungs 0-2 for that type, after the env and before the capability
  step writes, so the first record already carries measured verdicts.
- `crucible ladder [--rung N] [--type T]`: on demand, and what `doctor` names when the record is
  stale (a swapped card, a new driver, a new release).
- The tray: a "Test this card" item, and the one-line result per type.
- Never on a schedule by itself unless Owen asks for it.

## 7. Calls for Owen

1. **When**: at the end of `crucible install` (the person is waiting anyway), or deferred to the
   card's first idle window (nobody's game is interrupted)? Or both: rungs 0-1 at install, rung 2
   when idle.
2. **Budget**: the first run on kylies-pc, the 3090 Ti and the Studio sets it. Until then, should
   rung 2 stop after a fixed time per type, and what time?
3. **Speed floors**: should a measured speed ever REFUSE a class ("slower than real time: not
   offered"), or only be reported? If refuse, the floor is Owen's per type, not a constant here.
4. **Memory**: rung 2 measures each selected model's peak share on this card. Record only (today's
   ruling stands), or let a measured figure replace the declared estimate for this host, in the
   safe direction only?
5. **The inputs**: a clip, a page and a paragraph shipped with every release (a few MB), or pulled
   on first run like the weights?
6. **Quality under fallback**: rung 2 checks that output is well-formed, not that it is good. Does
   the ladder compare output against a reference (a known transcript, a known page) so that a
   bf16 checkpoint running in float16 can be caught reading worse, or is that out of scope?
