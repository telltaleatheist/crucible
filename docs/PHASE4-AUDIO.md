# Phase 4: `align`, `asr`, `rvc`, and the accelerator probe

Contract for the three remaining audio job types and the one server feature that retires
most of BookForge's GPU plumbing. Extends DESIGN.md. Written 2026-09-13 from
`docs/CLIENT-SURFACES.md` sections 4 and 10 (tier 4).

A correction to DESIGN.md section 3 first, because it is load-bearing and the table is
wrong: **`align` is not WhisperX.** The app's aligner is Qwen3-ForcedAligner-0.6B,
everywhere, with no fallback. WhisperX survives only as a CPU env supplying faster-whisper
for the rough-transcript stage of whole-m4b alignment, which is why `asr` is a separate job
type below and not a mode of `align`.

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

**Models:** six, each its own manifest — `tiny`, `base`, `small`, `medium`, `large-v3`,
`distil-large-v3`, from `Systran/faster-whisper-*`, 75 MB to 3,090 MB. **There is no default
in the app and there is none here**: a job names its model or is refused. An ASR pass at the
wrong size is a transcript that looks fine and is worse, with nothing in the output to say
so.

**The job:**

```json
{
  "type": "asr",
  "model": "faster-whisper-base",
  "params": { "language": "en", "vad_filter": true, "word_timestamps": true },
  "inputs": { "audio.m4b": { "blob_id": "..." } }
}
```

**`compute_type` is not a wire parameter.** It is `float16` on an accelerator and `int8` on
CPU — engine knowledge, decided by the server from its own backend, exactly as the division
of knowledge says. Note that the app has a **one-shot CPU fallback** here today
(`transcribe-bridge.ts`); Crucible does not, and will not: there is no CPU backend, and a
transcript that quietly ran at `int8` on a CPU is a different transcript.

**Windowing is the server's too:** 900-second windows with 15-second back-overlap, the
numbers the app measured. A client sends one file.

**Out:** `transcript.json`, with `progress {fraction, message}` carrying processed seconds
and cue count, so the app's existing progress parser has the same information it has now.

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
  "resident": { "kind": "tts", "id": "deathstalker", "since": "..." },
  "holders": [
    { "pid": 44503, "name": "python", "bytes": 1249902592, "owned_by_crucible": false }
  ]
}
```

`crucible/accelerator.py` already queries `nvidia-smi --query-compute-apps` for the load
guard; this route is that query with the resident set attached, and it is unauthenticated by
nothing — it needs the bearer token like every other private route.

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
