# Phase 4: `align`, `asr`, `rvc`, and the accelerator probe

Contract for the three remaining audio job types and the one server feature that retires
most of BookForge's GPU plumbing. Extends DESIGN.md. Written 2026-09-13 from
`docs/CLIENT-SURFACES.md` sections 4 and 10 (tier 4).

One thing the audit had to correct in BookForge's own plan documents is worth restating
here, because it decides the shape of section 3: **`align` is not WhisperX.** The app's
aligner is Qwen3-ForcedAligner-0.6B, everywhere, with no fallback — DESIGN.md's table
already says so. WhisperX survives only as a CPU env supplying faster-whisper for the
rough-transcript stage of whole-m4b alignment. That stage is a different model doing a
different thing, which is why `asr` below is a job type of its own and not a mode of
`align`.

## 0. How a job runs code that is not in the server's interpreter

`llm` never had this problem: vLLM and mlx-lm are *servers*, so Crucible starts one and
talks HTTP to it. `tts` does not have it either, because narrator is a server too. The
three types here are **libraries** — faster-whisper, ultimate-rvc and the Qwen3 aligner are
imported, not connected to — and they live in their own venvs, because each one's torch pin
is incompatible with the others and with the server's.

Crucible cannot import them. It runs them, the same way it runs everything else:

**Each job type ships a worker script, and the server spawns it with that type's
interpreter.** `crucible/jobs/<type>/worker.py` is a **standalone module** — stdlib plus
the one library its env has, importing nothing from `crucible` — invoked as
`<env python> <path to worker.py>` and speaking newline-delimited JSON on stdout, one
object per line, with the job's parameters handed over on stdin. It is the narrator wire
with a different vocabulary, and it is that on purpose: one protocol shape for every
subprocess this server owns.

Three rules fall out of that, and each of them is a bug that has already happened
somewhere in BookForge:

- **The worker imports nothing from `crucible`.** Its env has no `crucible` installed and
  never will; an `from ...errors import` in a worker is an ImportError at the first real
  job and a green test suite right up until then. The test for each worker runs it as a
  subprocess, exactly as the server does, rather than importing it.
- **Results go on a stream nothing else writes to.** narrator's aligner learned this the
  expensive way: a library's logger wrote to stdout and corrupted the result stream on a
  401-chunk book, and the fix was to dup fd 1 for results and point the original at stderr
  (`align/worker.py:56`). Crucible's workers do the same — **fd 1 is results, stderr is
  everything else, and a library that prints is expected to print.**
- **A result is matched to its job by position, never by an index the worker echoes back.**
  Same source, same reason: an index a worker reports is an index a worker can get wrong.

`crucible install <type>` builds the env from `envs/<type>/<backend>.txt`, `crucible doctor`
reports its presence and its pins, and a job whose env is missing is refused by name
(`env_missing`) before it is queued, exactly as `llm` already does.

### 0.1 The envelope, as built

`asr` landed first and the plumbing is shared, so the vocabulary below is every phase 4
type's. It lives in **`crucible/workers.py`** (spawn, read, marshal) and
**`crucible/workerenv.py`** (`~/.crucible/envs/<type>/`, the recipe, the stamp). Both are
`llmenv`'s and `manifests.py`'s ideas parameterised by job type; both say in their own
docstrings that the duplication is deliberate for now and that merging is a follow-up
rather than something to do while three builders are in the tree.

The worker speaks five message kinds on fd 1, and **a line that is not one of them is a
refusal naming the line**, never something skipped:

```
{"type": "ready",    ...}   exactly once, before any result. It is also what tells the
                            server HOW MANY results to expect, so the count is checkable.
{"type": "progress", ...}   whatever this type measures.
{"type": "result",   ...}   one per unit of work, matched by POSITION, carrying no index.
{"type": "failed",   "message"}   the whole run; the server stops the worker.
{"type": "done"}            exactly once, last. Exiting 0 without it is an unfinished
                            answer, not a short one, and is refused.
```

Two things the contract did not say and that the first build had to decide:

- **The wait for `ready` is a *silence* timeout, not a deadline.** Any message resets it.
  A worker decoding an eighteen-hour book cannot say how much work there is for twenty
  minutes, but it is noisy about bytes the whole time; a deadline would kill real work and
  a silence timeout only fires on a worker that has genuinely wedged. After `ready` there
  is no timeout of any kind — what ends a long run early is a cancel.
- **A result may carry `{"error": ...}` instead of its payload.** The worker keeps going,
  so one run finds every bad unit rather than one per re-run. What the *server* does with
  the holes is each type's ruling, and `asr`'s is in section 3.

Stopping a worker is SIGTERM, a wait, and a named refusal if it will not go. Crucible does
not SIGKILL a process that may hold CUDA; that wedges WSL2 until Windows reboots.

## 1. One rule decides three designs: no shared mount, ever

`tts` batch, `rvc` and whole-m4b `align` all read and write the same thing today — a
session's per-sentence FLACs on a filesystem both halves can see. CLIENT-SURFACES.md row 17
calls this "the design decision phase 3 has to make first", and phase 3b made it: **bytes
cross the wire, the SDK writes the files.**

It is decided the same way here, for the same reason and one more. The reason is that a
shared mount is not available: the library is on `Z:` (`\\TITAN\iO`), WSL cannot mount a
network drive, and that single fact is why whole-m4b alignment **cannot run on Owen's PC at
all today** (`whisperx-align-bridge.ts:642` refuses by name when the qwen env is `viaWsl`).
A server that requires a shared mount would ship that bug forward into the thing meant to
fix it. The extra reason is that a Crucible is expected to be on another machine — the Mac
across the room, a droplet on the tailnet — where there is no mount to share even in
principle.

The price is bandwidth: about 100 KB per sentence FLAC, so roughly 140 MB each way for a
1,400-sentence book. On a LAN or a loopback that is seconds, and it is the whole cost of the
feature working on every machine instead of one.

## 2. `align` — Qwen3-ForcedAligner

The easiest job type in the list, and it unblocks a feature that cannot run at all today.

**Model:** `Qwen/Qwen3-ForcedAligner-0.6B`, about 1.2 GB, `bfloat16` on an accelerator and
`float32` on CPU. It gets a manifest like any other model, with `job_type = "align"`.

**Residency:** the model is resident **across a whole book** — hundreds of chunks, one load.
That is what the generalised `Residency` (PHASE3-TTS.md section 5) is for; an aligner is a
third kind of resident thing and needs no new mechanism.

**The job:**

```json
{
  "type": "align",
  "model": "qwen3-aligner",
  "params": {
    "language": "en",
    "chunks": [{ "index": 41, "text": "He had been walking for some time." }]
  },
  "inputs": { "41.flac": { "blob_id": "..." } }
}
```

One input per chunk, named `<index>.<ext>`, matched to its text by index. Audio is decoded
to **16 kHz mono float32** by the server (ffmpeg), because the sample rate is engine
knowledge and the client should not have to know it. A chunk longer than
**`QWEN3_MAX_AUDIO_S = 300`** is refused by name rather than chunked — the model's limit,
not a policy, and splitting it would silently change the alignment.

`language` must be one of the eleven codes the aligner supports; an unknown one is refused
before the job is queued.

**Out:** one `alignment.json` artifact, plus a `cue` event per chunk as it lands
(`cue {index, items: [...]}`), so a killed run costs the chunks it had not reached and not
the ones it had.

**What stays in BookForge, and it is most of the value:** the item-to-word mapping, the
`_normalized` letter-sequence equality check that refuses a model which rewrote the text
(`aligner.py:681`), every derived score, and the coverage report. Crucible returns one
timestamped item per *its own* tokenization and asserts nothing about words.

**No retries and no other backend, ever.** A failed chunk is reported, the run continues,
and no second aligner is tried. `--backend qwen3` is hardcoded in the app today for exactly
this reason; here it is the manifest's engine and there is nothing to fall back to.

## 3. `asr` — faster-whisper

A new job type, and a real gap rather than an oversight: `generate-sentences` is a GPU queue
step, it is the only ASR site in the app that takes the arbiter lease, and the whole-m4b
align door needs a rough transcript before the aligner runs.

**Models:** six, each its own manifest, 75 MB to 3,090 MB. **There is no default in the app
and there is none here**: a job names its model or is refused. An ASR pass at the wrong size
is a transcript that looks fine and is worse, with nothing in the output to say so.

The manifests live in **`asr/<id>.toml`** with their own loader,
`crucible/asrmodels.py` — the same strictness as `crucible/manifests.py` and, frankly, the
same two hundred lines; the two should be one loader parameterised by directory and
required keys, and that merge is a follow-up. Every revision below is the repo's `main` sha
read from the HuggingFace API on 2026-09-13:

| Crucible id | HF repo | revision | params |
|---|---|---|---|
| `faster-whisper-tiny` | `Systran/faster-whisper-tiny` | `d90ca5fe2602…` | 39M |
| `faster-whisper-base` | `Systran/faster-whisper-base` | `ebe41f70d5b6…` | 74M |
| `faster-whisper-small` | `Systran/faster-whisper-small` | `536b0662742c…` | 244M |
| `faster-whisper-medium` | `Systran/faster-whisper-medium` | `08e178d48790…` | 769M |
| `faster-whisper-large-v3` | `Systran/faster-whisper-large-v3` | `edaa852ec7e1…` | 1550M |
| `faster-whisper-distil-large-v3` | `Systran/faster-distil-whisper-large-v3` | `c3058b475261…` | 756M |

Two corrections to what this section used to say. The distilled model is **not** under
`Systran/faster-whisper-*`; Systran publishes it as `faster-distil-whisper-large-v3`, and
the Crucible id keeps the family prefix so all six sort and read as one family. And the
`memory_bytes_estimate` in every one of these six is **COMPUTED, not measured** — the exact
`model.bin` byte count plus a declared 1.5 GiB for the CUDA context, the cuBLAS/cuDNN
workspaces and a 30-second window's activations. Nobody has run faster-whisper under
Crucible on a card yet (the 3090 Ti was on an overnight fine-tune), every manifest says so
in the file, and the first real book is what replaces those lines with measurements.

**There is no `mlx-darwin` block, for any of the six, and it is not an oversight.**
faster-whisper is CTranslate2, and **CTranslate2 has no Metal backend** — on Apple Silicon
it builds against Accelerate and runs on the CPU cores, and `device="mps"` is a `ValueError`
(SYSTRAN/faster-whisper#515 and #911, both still true as of 2026-09). Since this section
already refuses the CPU road on its own terms, the Mac gets `backend_unsupported` by name
and `crucible install asr` on the Mac refuses naming the recipes that do ship. The Mac's ASR
story is `mlx-whisper`: different weights, a different library, so a second engine behind
the same job type rather than a second recipe — new manifests, a `mlx-whisper` entry in
`ASR_BACKEND_ENGINES`, a second worker. It is written down in `envs/asr/mlx-darwin.md`,
where somebody looking for the missing `mlx-darwin.txt` will find it, and it is not in this
phase.

**The job:**

```json
{
  "type": "asr",
  "model": "faster-whisper-base",
  "params": { "language": "en", "vad_filter": true, "word_timestamps": true },
  "inputs": { "audio.m4b": { "blob_id": "..." } }
}
```

All three params are **required**. `vad_filter` and `word_timestamps` default to true in the
app and to nothing here, because a default would mean a transcript quietly produced under
different rules than the caller assumed. `language` is a faster-whisper code, checked
against the tokenizer's own list before the job is queued, or the literal `"auto"` — which
is a *value* meaning "detect it" and not an absence. Exactly one input, of any container
ffmpeg can read; more than one is refused.

**`compute_type` is not a wire parameter.** It is `float16` on an accelerator and `int8` on
CPU — engine knowledge, decided by the server from its own backend, exactly as the division
of knowledge says. Since there is no CPU backend it is `float16`, always. Note that the app
has a **one-shot CPU fallback** here today (`transcribe-bridge.ts`); Crucible does not, and
will not: it is not a fallback at all but a silent substitution — the run still produces a
transcript, it is a different transcript, and nothing in the output says which one you got.

**Windowing is the server's too:** 900-second windows, each reaching **15 seconds past its
own boundary** so a sentence straddling the cut is spoken in full inside it, with the next
window starting at the boundary. (The old wording, "15-second back-overlap", describes the
same arrangement from the next window's side.) A client sends one file.

**ffmpeg is required and is refused by name.** The worker decodes through ffmpeg, not
through faster-whisper's PyAV decoder, which silently TRUNCATES some assembled m4b files —
one real 18-hour book decoded to six hours and the transcript stopped dead mid-book with no
error. A host with no ffmpeg on PATH gets `ffmpeg_missing` at submit time.

**Out:** `transcript.json` — whisper's own segments in absolute book time with the window
overlaps removed, plus the model, the pinned revision, the language asked for and the
language detected, the duration, and the windowing that produced it. Sentence-cue grouping
and the WebVTT stay in BookForge, for the reason section 2 gives about `align`.

One deviation from the app worth knowing: BookForge groups words into sentence cues *first*
and dedupes the cues; the grouping does not exist here, so the same rule — sort by start,
drop anything beginning inside a kept span, 0.1 s tolerance — is applied to the *segments*.
Boundary behaviour is therefore close but not identical.

**Progress** is `progress {fraction, message, stage, processed_s, total_s, cues}`. The three
extra keys are the wire change this section needs and are exactly what BookForge's existing
parser reads off its own `DECODE` and `PROGRESS` lines, so an 18-hour book shows a moving
position while the percentage is still rounding to zero. `stage` is `decoding` or
`transcribing`, and **the decode drives no fraction**: it is real work with a real position,
but none of the transcript exists yet and a bar that counts it as progress towards the
transcript is a bar that lies.

**A failed window fails the job, and publishes nothing.** The worker keeps going after one,
so a run finds every bad stretch instead of one per re-run; then the job fails naming the
windows and their offsets. A fifteen-minute hole in the middle of a transcript looks exactly
like a transcript without one — the same argument as the one against a default model, and it
gets the same answer.

`asr` is off unless `[jobs] enable_asr` says otherwise (`crucible init --enable-asr`), its
env is `crucible install asr`, and its weights come through the one `crucible models pull`,
which now covers both manifest directories as a single namespace of ids — a collision
between them is refused rather than settled by which directory was read first.

## 4. `rvc` — ultimate-rvc

**Model identity becomes a manifest.** Today an RVC model is a *folder name* under
`<userData>/runtime/rvc-models/rvc/voice_models/<Name>`, discovered by looking for a `.pth`,
with `forceIndexRate0` derived from the **absence** of a `.index` file. That is a filesystem
convention standing in for an identity, and it does not survive the trip to another machine.
`rvc/<id>.toml` names an HF repo and a revision the way every other manifest does —
`owenmorgan/deathstalker_rvc_v1` is already published, so this is a translation and not an
invention — and declares `has_index`, which is what `forceIndexRate0` was inferring.

**The job:**

```json
{
  "type": "rvc",
  "model": "deathstalker-rvc-v1",
  "params": {
    "index_rate": 0.3, "protect_rate": 0.1, "n_semitones": -2, "f0_method": "rmvpe"
  },
  "inputs": { "41.flac": {"blob_id": "..."}, "42.flac": {"blob_id": "..."} }
}
```

One artifact per input, same name. **Every input must produce an output** — a missing one is
a failed job, not a short answer.

Three details that are not obvious and are all measured:

- **`protect_rate`'s scale is inverted.** Lower protects more, 0.5 is off, and it is a no-op
  at index rate 0. The parameter name is ultimate-rvc's and stays; the inversion goes in a
  comment at the one place it is validated, because a reviewer will otherwise "fix" a bound.
- **An absent `f0_method` or `hop_length` means the flag is omitted**, so urvc keeps its own
  default. It does not mean a value Crucible chose. This is the one place where "absent" is
  a meaningful wire value rather than a refusal, and the reason is that urvc's defaults are
  the tuned ones.
- **Batching is a memory bound, not a throughput choice.** The app recycles a worker process
  every 96 files, proven necessary on a 64 GB Mac. That is engine knowledge: the server does
  it, the client never sees it, and the model reload it costs is the server's problem to
  reduce later.

`URVC_SKIP_INIT=1`, `HF_HUB_OFFLINE=1`, `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1`
go in the engine's `environment()` with the reason that earned each one — three bundled
OpenMP runtimes SIGSEGV without the third.

**Never `urvc.exe`**: pip's Windows console script bakes a stale shebang. Crucible runs
`python -m ultimate_rvc.cli.main`, which is what the app already does, and the comment says
why so nobody simplifies it back.

## 5. The accelerator probe — `GET /v1/accelerator`

The single highest-value item in the audit, and the smallest. BookForge has **three
incompatible arbitration schemes**: the queue's single GPU slot plus `gpuAdmission`; an
in-process mutex whose `timeoutMs` **proceeds without the lock** rather than failing
(`gpu-arbiter.ts:88`); and, for the hosted page reader, nothing at all. On top sits
`external-gpu-job.lock`, a Windows-only convention with **no producer inside the app**,
whose deletion nothing watches — hence a 15-second admission recheck.

```json
{
  "backend": "cuda-linux",
  "gpu": { "vendor": "nvidia", "name": "NVIDIA GeForce RTX 3090 Ti", "total_bytes": 25757220864 },
  "free_bytes": 24297504768,
  "used_bytes": 1459716096,
  "desktop_allowance_bytes": 3221225472,
  "unattributed_bytes": -3011411968,
  "resident": { "kind": "llm", "id": "qwen3.5-9b", "since": "...", "memory_bytes_estimate": 20950548480 },
  "holders": [
    { "pid": 44503, "name": "python", "bytes": 1249902592, "owned_by_crucible": false }
  ],
  "detail": "22.6 GiB free of 24.0 GiB, 1 compute app(s), desktop allowance 3.0 GiB"
}
```

`crucible/accelerator.py` already queries `nvidia-smi --query-compute-apps` for the load
guard; this route is that query with the resident set attached, and it is unauthenticated by
nothing — it needs the bearer token like every other private route.

Four fields the sketch above did not have, each of which a caller needs to avoid drawing the
wrong conclusion:

- **`unattributed_bytes`** — VRAM in use that no listed compute app accounts for, past the
  declared desktop allowance. Under WSL2 the driver shim answers the compute-app query with
  an **empty list** even while a process inside that same VM holds 17 GB (measured on Owen's
  PC, 2026-09-12). On that host — which is the host BookForge runs on — `holders` is
  misleadingly empty and this is the only honest report that the card is busy. It is `null`,
  not `0`, on `mlx-darwin`, where "used unified memory" is the OS doing its job and
  attributing it to compute processes is not a question `vm_stat` can answer.
- **`used_bytes` and `desktop_allowance_bytes`** — so a client can do the same arithmetic
  the guard does rather than inferring the host's declared facts.
- **`holders[].bytes` may be `null`**, where the driver will not say (WDDM, permissions).
  That is a refusal to answer and it is not zero.
- **`resident.kind`** is the family of the resident thing, not the job type that put it
  there. Today the only resident thing is an LLM engine, so it is `"llm"`; phase 3's
  generalised residency adds tts voices and phase 4's aligner beside it, at the same key.

A probe that cannot answer is **`503 accelerator_unreadable`**, never zeroes and never "the
card is free". A client polling for a free GPU must read it as "ask again".

The injection seam on the client side **already exists and is already wired**:
`setGpuHolderProbe` (`queue-engine.ts:1275`, wired at `queue-ipc.ts:56`). One call into this
route retires the lock file, the 44-second reload trade, the flat-utilisation reservations,
and most of `wsl-lifecycle.ts`'s GPU-teardown ladder.

**No eviction, ever.** The probe reports; it never asks anyone to leave. That rule is already
in PHASE2-LLM.md section 4 and it does not soften because more job types now depend on it.

## 6. One operational request from the audit, worth honouring everywhere

*"Return the server-side traceback in error bodies."* `recentServerLog()` is currently the
only way a mid-run engine-core crash is diagnosable in BookForge, because the API stays up
answering 500s whose body says "see stack trace (above)".

Crucible already tails the engine log into `EngineError` on a failed start. Extend the same
courtesy to a failed *job*: when a job fails because its engine died, the `failed {error}`
event carries the last lines of that engine's log in `details`, not a pointer to a file on a
machine the client may not be able to read.

## 7. What is deliberately left out, and why

**Denoise, vocal separation and Resemble Enhance** (CLIENT-SURFACES.md row 23) are three
more GPU models in the same envs and they are not in this phase. The reason is specific
rather than a shrug: their contract is **sample-exact** — 44.1 kHz stereo in, an identical
frame count out, or the offsets manifest cannot re-cut about 1,400 sentences
(`denoise-bridge.ts:305`) — and Resemble needs deterministic per-seed RNG because production
renders five seeds and takes the per-frame spectral median. Neither property can be asserted
against a fake engine; both need real audio on a real card, measured against Owen's existing
output. Building them blind would produce code that looks right and silently shifts a book
by a few samples, which is the exact failure this project exists to stop making.

They get a phase of their own when a card is free.
