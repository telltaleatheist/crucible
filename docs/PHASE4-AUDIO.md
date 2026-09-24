# Phase 4: `align`, `asr`, `rvc`, and the accelerator probe

Contract for the three remaining audio job types and the one server feature that retires
most of BookForge's GPU plumbing. Extends DESIGN.md. Written 2026-09-13 from
`docs/CLIENT-SURFACES.md` sections 4 and 10 (tier 4).

**All three are built.** `asr` landed first and established the worker envelope; `align` and
`rvc` followed and are what sections 0.2, 2 and 4 now describe. Everywhere this document
turned out to be wrong about something real — the aligner's size, where the RVC models are
actually published, which fork of ultimate-rvc has the command this depends on — the
correction is written in place and says what it corrects, because the next person to read
this will be reading it *instead of* the code.

Each type is off unless its flag says otherwise: `[jobs] enable_asr`, `enable_align`,
`enable_rvc`, set by `crucible init --enable-asr --enable-align --enable-rvc`.

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

**On win32 (a Windows host running a job natively), since 2026-09-23:** the polite signal
is `CTRL_BREAK_EVENT` to the worker's own process group (`CREATE_NEW_PROCESS_GROUP` at
spawn), the same wait, and then the tree is terminated (`taskkill /T /F`) — the WSL2 reason
does not exist for a native Windows process, which is the deviation `llama_server.py` already
made. All three halves live in `crucible/procgroup.py`, which refuses an unknown platform by
name. Until then every stop path called `os.killpg`, which win32 does not have: a cancelled
`align` job on Windows failed with `AttributeError` and left its worker running.

### 0.2 The worker that outlives a job, as built

`workers.py`'s docstring promised the split to make was `start`/`send`/`stop` with
`run_worker` as the three in a row, and that is what `align` needed and what is now there.
An **exchange** is one request in, one `ready`, N results, one `done`; `_Conversation` runs
one over a process's two pipes; `run_worker` is one exchange with stdin closed afterwards,
and `WorkerSession` is a process that outlives them. The only difference between the two is
what happens to stdin: a one-shot worker gets it CLOSED, because a process blocked on a read
it will never satisfy is a hang with no error, and a session's stays open because the next
request goes down it.

`stop()` is **close stdin, wait, then SIGTERM** — the polite door first, so the worker's own
read loop sees EOF and releases CUDA the way its own code expects to. There is still no
third step.

**`rvc` deliberately does NOT take a session**, and that is a correction to what this
section implied. Its 96-file recycle is a memory bound that wants the process to *die* so
the OS reclaims everything it leaked; a worker held open across jobs would be the one thing
that bound exists to prevent. The recycling happens one level down, inside
`jobs/rvc/worker.py`, where each batch is its own urvc process. So `rvc` is a `run_worker`
type exactly like `asr`, and `align` is the only session.

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

**Model:** `Qwen/Qwen3-ForcedAligner-0.6B`, `bfloat16` on an accelerator and `float32` on
CPU. Two corrections from the build:

- It is **1.7 GB, not 1.2**: `model.safetensors` is 1,835,544,544 bytes exactly, read from
  the HuggingFace API on 2026-09-13 (about 918M parameters including the audio tower; the
  name counts the text side). The pinned revision is `c7cbfc2048c4…`.
- **There is no `job_type` key and there cannot be one.** The manifest lives in
  `align/<id>.toml` with its own loader, `crucible/alignmodels.py`, because
  `crucible/manifests.py` requires `params_b`, `context_default` and `modalities` on every
  `[model]` table and permits only `vllm` and `mlx-lm` as engines — those manifests describe
  what an OpenAI-compatible engine serves. A forced aligner has none of those. **The
  DIRECTORY is the job type**, which is how `asr/` already works and is the better
  arrangement anyway: nothing can declare `job_type = "llm"` inside `align/` and be
  half-believed by two loaders. The weights still pull through `crucible models pull`, so
  there is one command to learn rather than three.

**There is no `mlx-darwin` block, and the reason is NOT `asr`'s.** `asr` cannot have one:
CTranslate2 has no Metal backend. This one could — the aligner is plain torch, torch has an
MPS backend, and BookForge's own aligner already accepts `mps` as a device. It is missing
because **nobody has measured it**: the bake-off that chose this aligner ran in WSL2 on the
3090 Ti, `bfloat16` on MPS is a different numerical path, and a recipe is not the place to
assert a result nobody has. `envs/align/mlx-darwin.md` says so at length, in the place
somebody looking for the missing `.txt` will find it, and says exactly what would settle it:
align a chapter on a Mac that has already been aligned on the PC and compare the timestamps.

**Residency:** the model is resident **across a whole book** — hundreds of chunks, one load.
That is what the generalised `Residency` (PHASE3-TTS.md section 5) is for, and what it
needed to hold a third kind was:

- a third `KIND_ALIGN` and a `ResidentAligner` row (no `base_url`, no `engine`, and `device`
  / `dtype` / `max_audio_s` on it instead, because those are what it was actually loaded
  with);
- a **second holder slot**, `_session`, beside `_engine`. An LLM and a voice are servers
  behind `SubprocessEngine`; the aligner is a `workers.WorkerSession`. They are stopped
  differently and raise different errors, so one `_held` of a union type would have put a
  `hasattr` in charge of which. `owned_pids()` unions both slots — reading only the engine
  slot was the bug waiting to happen the moment a second shape existed — and `unload()`
  stops whichever is there;
- `KIND_NOUNS`, because `describe_resident` had a two-way conditional that would have called
  an aligner "the resident model" and sent its reader to `unload-model`.

**The load happens inside the `align` job; the unload is a job of its own.** `llm` and `tts`
are loaded by an explicit job because a client chooses *when* to spend 200 s of warm-up
against what else is queued. An aligner load is seconds and is always immediately followed
by the work it was loaded for, so making a client send two jobs to align one book would be
ceremony. Taking it OFF the card is a decision about somebody else's next job, so
`unload-aligner` exists — without it an aligner could only be evicted by loading something
else, which would make "one card, one thing" a rule you can only obey by breaking it.

The session's first exchange is a `{"op": "load"}` request the worker answers with `ready`
once the checkpoint is on the device. A load is a real exchange and not a bare spawn on
purpose: a process that has started has proved only that python runs.

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

One input per chunk means **both directions are checked**: a chunk with no audio and an
input with no chunk are each a named `invalid_inputs` refusal, because a book aligned with
1,399 of its 1,400 chunks reads as a complete answer. The client's `index` never reaches the
worker — a chunk on the wire to it carries `{audio, text}` and nothing else, and a result
carries no index either, so position is the identity in both directions.

`language` crosses the wire as the ISO code and reaches the model as its own English NAME
("English", "Cantonese"), which is what `model.align` takes. The mapping is the server's.

**Out:** one `alignment.json` artifact, plus a `cue` event per chunk as it lands
(`cue {index, items: [...]}`), so a killed run costs the chunks it had not reached and not
the ones it had. `JobContext.cue` is new and takes a dict rather than `**keys`, because the
payload is a *row of the answer* and its shape is the job type's.

Two details the contract did not settle:

- **A failed chunk gets a cue too**, carrying `error` instead of `items`. A client watching
  the stream should learn about a failure at the same moment as the successes around it.
- **A failed chunk does not fail the job.** It is named, in the artifact, per chunk, and the
  `done` event carries `failed: [indexes]`. This is the opposite of `asr`'s ruling and the
  difference is not inconsistency: a hole in a transcript is *invisible* in the transcript,
  while a chunk that carries `error` instead of `items` is visible in every consumer of the
  document. Section 2 already said "a failed chunk is reported, the run continues", and this
  is what that costs to keep true.

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
| `faster-whisper-large-v3-turbo` | `dropbox-dash/faster-whisper-large-v3-turbo` | `0a363e9161cb…` | 809M |

**`faster-whisper-large-v3-turbo` joined on 2026-09-23** (for ContentStudio), and it is the
one row not from Systran, because Systran publishes no turbo conversion. It is
`dropbox-dash/faster-whisper-large-v3-turbo` — the repo faster-whisper 1.2.1's own
`_MODELS` table names for `"large-v3-turbo"` under its former name
`mobiuslabsgmbh/…`, converted from `openai/whisper-large-v3-turbo` at `float16`, and
byte-identical in `model.bin` to the next most-downloaded conversion. A conversion on the
hub pinned by commit is inside DESIGN.md section 5's source rule, the same standing the
`mlx-community` conversions have; the manifest carries the survey, and the check that the
config and front end are OpenAI's turbo (128 mels, a four-layer decoder by its alignment
heads). Its estimate is computed the same way as the six above.

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

**`initial_prompt` (added 2026-09-23, for ContentStudio) is the one optional param.** A
string whisper is primed with before it hears the audio, as if it were the transcript so far
— the way to tell it how a title and the proper nouns in it are spelled. Both engines take
it under that name in `transcribe()` (faster-whisper 1.2.1 and mlx-whisper 0.4.3, the
versions the `asr` recipes pin, read in their source), and both encode `" " + prompt.strip()`
at the head of the token history the first 30-second segment is conditioned on.

- **Optional, `null` meaning none.** Every other param is required because it changes the
  transcript, and so does this one; but BookForge's pinned `@crucible/client` 1.0.23 sends
  exactly the three keys above, and a fourth required key would make every one of its
  transcripts a 400 on deploy day. Absent is what that client means — no prompt — and
  `transcript.json` records `initial_prompt: null` either way, so the document still names
  its rule. The SDK's `asr({initialPrompt})` sends the key whenever its caller states it.
- **A non-string is refused, and so is a blank string** (`invalid_params`, naming the
  field): `""` beside `null` would be two spellings of none, and faster-whisper would
  encode `""` as a lone space token.
- **Applied to every 900-second window, not only the first.** Each window is its own
  `transcribe()` call and each call starts its token history empty, so
  `condition_on_previous_text` carries nothing across a window boundary; a prompt given to
  window 0 alone would prime fifteen minutes of an eighteen-hour book. Inside a window it
  is not permanent either — the history is cut to its last 223 tokens and reset by a
  temperature fallback above 0.5 — so what the caller gets is the start of every window
  primed, which is the most either engine offers.
- **Longer than whisper keeps is a failed job, by name.** Both libraries keep only the
  last `max_length // 2 - 1` = 223 prompt tokens, so a longer prompt would silently lose
  its *beginning*. The count needs the model's own tokenizer, which exists only in the
  worker's env, so each worker counts after loading and fails the run before decoding
  any audio.

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
language detected, the `initial_prompt` (or `null`), the duration, and the windowing that
produced it. Sentence-cue grouping
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
`rvc/<id>.toml` fixes that, and declares `has_index`, which is what `forceIndexRate0` was
inferring.

**Correction: there is no `owenmorgan/deathstalker_rvc_v1` repo.** Every RVC model Owen has
published is a `.tar.gz` under `rvc/` in ONE repo, `owenmorgan/owen-morgan-bookforge`,
alongside six others and the XTTS weights (`electron/data/rvc-voice-assets.json`). So this
manifest cannot be "an HF repo and a revision the way every other manifest does", and it has
three keys no other manifest in the repo has:

- **`archive`** — the path inside the repo. `snapshot_download` would fetch about 800 MB to
  get at 80, so `weights.pull_archive` fetches that one file with `hf_hub_download` and
  unpacks it. The tarballs unpack to `rvc/voice_models/<name>/`, which is a whole
  `URVC_MODELS_DIR` root — verified against every published tarball on 2026-09-13, not
  assumed.
- **`archive_sha256`** — a snapshot download is verified by the hub client against the
  revision; a single file fetched by path deserves the same. The app's catalog already
  carries the digest for all seven, so it is a translation. The digest is checked **before
  anything is unpacked** and a mismatch refuses without writing a byte, and the unpack
  refuses any member that would land outside the target or that is a link (python 3.11 has
  no `filter="data"`, so the rule is written out).
- **`model_name`** — the folder inside the archive, which is the argument urvc is given. It
  is not derivable from the id: `sigma` is `Sigma Male Narrator`, `us-female-1` is
  `US_Female_1`, `owen-morgan` is `Owen Morgan`.

`rvc` gets its **own weights family and its own command** (`crucible rvc list` / `crucible
rvc pull`, under `~/.crucible/rvc/`) rather than joining `crucible models pull`. Two
reasons: `models pull` snapshot-downloads a repo and could not serve an archive at all, and
`sigma` is *also* a narrator voice id — one tree for both is how one pull overwrites the
other and leaves a stamp that reads as installed to either.

**Seven manifests ship, and all seven have a `.index`** — so `forceIndexRate0` fires for
none of them. It is still declared rather than assumed, and a non-zero `index_rate` against
a model that says `has_index = false` is refused by name (`model_has_no_index`) rather than
clamped: BookForge clamps, and a caller who asked for 0.5 and silently got 0 has an output
that sounds wrong for a reason nothing in it explains.

**The models with no manifest**, because they have never been published anywhere a manifest
could point at: `deathstalker_rvc_v2`, `mistborn_rvc_v2`, `mistborn_rvc_v3_aol`, and the
training-only checkpoints (`mistborn_rvc_v3_refinegan`, `_rg32`, `_rg40`,
`owen_morgan_rvc_v1`, `third_reich_rvc_v1`). A manifest naming a repo path that does not
exist would be worse than no manifest: it would list as a model and refuse at pull time.

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

### 4.1 Three things the build had to decide

**The engine is Owen's FORK, pinned to a commit.** `generate convert-dir` — the warm-model
batch command this whole job type is built on — exists only in
`telltaleatheist/ultimate-rvc@bookforge` and not in the `ultimate-rvc` on PyPI, which has
`generate convert`, one file per process, i.e. 1,400 model loads for a book. Both call
themselves **version 0.5.11**, so the version is not an identity here and the commit is. The
recipes therefore carry the repo's first PEP 508 direct reference,
`ultimate-rvc @ git+…@05cc3f1ba921…`, and `workerenv` gained `recipe_direct_refs` /
`installed_direct_refs` to check it against `pip freeze` — `pip list` reports the declared
version and says nothing about the commit, so checking it there would compare a sha against
`0.5.11` and call every correctly built env broken. `crucible doctor` names the commit, not
the version, for the same reason.

**The base assets are pulled from the engine's own repo — PLAN.md's owed ruling 3, now
discharged.** urvc needs a contentvec embedder and a pitch predictor before it can convert
anything; they are the engine's rather than any model's, and this section used to say
Crucible could not fetch them, because the only source written down anywhere was a 388 MB
tarball on a **GitHub release** in BookForge, which DESIGN.md section 5 refuses as a
weights source.

That was wrong about the world rather than about the rule. Read out of the installed fork
on 2026-09-13 — `ultimate_rvc/rvc/lib/tools/prerequisites_download.py`, which **is** the
first-run downloader `URVC_SKIP_INIT=1` turns off:

```python
url_base = "https://huggingface.co/JackismyShephard/ultimate-rvc/resolve/main/Resources"
models_list    = [("predictors/", ["rmvpe.pt", "fcpe.pt"])]
embedders_list = [("embedders/contentvec/", ["pytorch_model.bin", "config.json"]), ...]
```

So the engine's own upstream is HuggingFace, which DESIGN.md allows, and what Crucible
pulls are exactly the bytes urvc would have fetched for itself, from the repo it would
have fetched them from — at a **pinned revision** instead of `main`, with a sha256 per
file. (The ancestral upstreams a general RVC tutorial names — `lengyue233/content-vec-best`,
`lj1995/VoiceConversionWebUI` — are *not* what this fork downloads, and pinning them would
be pinning a different provenance than the engine's own.)

`rvcbase/ultimate-rvc.toml` is the declaration and `crucible/rvcbase.py` the loader.
**`crucible rvc pull-base`** places all four files under `~/.crucible/rvc-base/`, where
the job has always looked, verifying every digest **before placing any file** — a
half-placed base tree is one urvc will start against and fail inside, hours later, in
somebody's book. A job without them is still refused by name
(`rvc_base_models_missing`), and the refusal now names a command that works.

Two things this fixed on the way past. **`fcpe.pt` is pulled too**, because `f0_method` is
a job parameter and a client may legitimately ask for it — urvc's own prerequisite list
fetches both predictors, and so does this. And **`config.json` is pulled beside the
embedder's weights**: the job's old hardcoded check named two files and missed it, without
which `transformers` will not load the embedder directory at all. Which files are needed
is now `rvcbase`'s to say and the job reads it (R1), so those two lists cannot drift again.

### Ruling owed

- **What else does a conversion fetch at run time?** `torchcrepe` downloads its own weights
  on first use, so `f0_method: "crepe"` may still reach the network inside a job. Not
  measured, and not pulled here: the crepe predictors are that library's rather than
  urvc's, and every BookForge recipe uses `rmvpe`. The first `crepe` run settles it.

Each job composes its own `URVC_MODELS_DIR` out of symlinks — the shared base assets plus
**one** model — under the job's scratch. One model and not all seven, because urvc resolves
a model by NAME and a root holding seven is a root where a name can resolve to the wrong one.

**One extension per job.** `convert-dir` takes a single `--input-glob` and a single
`--output-ext`, and "one artifact per input, same name" is only true when the output keeps
the input's extension. A mixed-format job is refused by name rather than half converted.

## 4.2 `denoise` — audio-separator, in the rvc env

The pass BookForge runs over a session's rendered sentences before it assembles them.
Fine-tuned narration voices are trained on a deliberate ~-65 dBFS room-hiss bed —
**load-bearing for reliable end-of-audio**, so it is not a defect in the training
data — and the consequence
is that every raw render carries a hiss during speech that cuts out at the digitally
silent assembly gaps. One mel-band roformer pass removes it.

It is a job type and not a mode of `rvc`, and it is **not** an env of its own.

### It shares the `rvc` env, and that is the first time anything does

audio-separator is torch, `envs/rvc/<backend>.txt` already pins the exact torch it wants,
and BookForge runs both out of one env today (`electron/denoise-bridge.ts` reaches for the
RVC env's python). A second venv would be a second 3 GB torch on disk to drive the same
card. So:

- `envs/rvc/cuda-linux.txt` and `envs/rvc/mlx-darwin.txt` pin **`audio-separator==0.31.1`**
  — the version BookForge's own resident worker was written and verified against, having
  read that release's `cli.py` to confirm every separation parameter it leaves alone is an
  argparse default identical to the constructor default. Checked against this recipe's
  other pins from PyPI metadata on 2026-09-13 and compatible on every one (torch>=2.3 here
  2.7.0, numpy>=2 here 2.2.5, librosa>=0.10 here 0.10.2,
  rotary-embedding-torch>=0.6.1,<0.7.0 here 0.6.5). **The pin is compatible, not resolved**
  — nobody has run pip against these files, which is what their existing "NOT YET A
  RESOLVED SET" notice already says, and audio-separator additionally pulls `onnx-weekly`
  and `onnx2torch-py313` unpinned.
- `crucible install rvc` builds the env and decides **both** capability flags, because it
  is the door that has just built the thing they share.
  `workerenv.JOB_TYPES_SERVED_BY_ENV` is the one owner of "denoise lives in the rvc env",
  and `crucible doctor`'s "you could turn this on" note reads it so the command it
  suggests is one that exists.
- `[jobs] enable_denoise` is still its **own** flag. A host may have the env and the RVC
  models and no separator checkpoint, or the other way round, and one flag for both would
  advertise a job type whose first request refuses.
- The capability class is its own too: a 913 MB separator and a 2.5 GiB urvc stack are
  different arithmetic against the same card.

### The job

```json
{
  "type": "denoise",
  "model": "denoise-roformer",
  "params": {},
  "inputs": { "block_00.wav": {"blob_id": "..."} }
}
```

**One audio input, at the model's native rate.** Every stem the separator writes comes back
as an artifact, and `done` names which of them is the primary one.

**Two separators ship** (the second added 2026-09-23, for ContentStudio), and the model id
is the only thing that differs on the wire:

| id | checkpoint (audio-separator's name) | `primary_stem` |
|---|---|---|
| `denoise-roformer` | `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt` | `dry` |
| `vocals-roformer` | `vocals_mel_band_roformer.ckpt` (Kimberley Jensen) | `vocals` |

Nothing in the job type names a stem: the one it publishes is the manifest's
`primary_stem`, matched as `(<stem>)` in the file audio-separator writes, and the
invariants below are checked against that one. `vocals-roformer` comes from the same
`Politrees/UVR_resources` snapshot as the denoiser; its checkpoint's sha256 is the one
Kimberley Jensen's own `KimberleyJSN/melbandroformer` upload carries, and its mirror config
parses equal to audio-separator 0.31.1's own `vocals_mel_band_roformer.yaml`. Its
`memory_bytes_estimate` is computed the same way, and says so. Both separators are the
same resident kind, so a job for one while the other is on the card loads its own
checkpoint in the other's place.

**`params` is empty, and that is the contract.** Every knob audio-separator takes is an
engine default BookForge measured and left alone; `use_autocast` is CUDA-only by the
library's own documentation, so the server reads it off the *backend* rather than off a
manifest or a request. DESIGN.md section 3.1's rule is that a knob crosses the seam one at
a time, with a reason, and none has one yet. The params object is still validated with
`extra="forbid"`: a client asking for something is told no by name rather than answered
with something else.

**Blocking stays in the client**, the same ruling that put chunking there for `tts`. The
app concatenates a book's sentences into ~22-minute blocks, denoises each, and slices the
stems back at recorded offsets. Crucible denoises one thing at a time.

**And the separator is RESIDENT** (Owen's ruling, 2026-09-15). Those two sentences are not
in tension: the WIRE is still one block per job and the blocking is still the client's;
what changed is that the checkpoint stays on the card BETWEEN jobs, as `KIND_DENOISE` —
the fourth resident kind, the same `workers.WorkerSession` shape as the aligner in
section 2.

This reverses a ruling in this document, and the reason is a measurement Crucible had no
way to see. The job type argued that "a separator loads once per job either way", which is
true of one job and false of a pass: the client sends **~44 jobs for a 15-hour book**, so
the load was paid ~44 times. BookForge had already made the identical mistake on its own
side and already fixed it — `electron/scripts/separator_worker.py` (bookforge `019afa52`)
replaced a per-block spawn with a resident worker because the fixed cost was *"being paid
44 times (10-25 s each) for ~85 s of real work per block"*, which
`electron/denoise-bridge.ts:27-32` records as *"roughly a third of the pass. One load now
serves the whole book."* The warm figure from that commit is 7.4 s of an 11.0 s one-shot.
A job type reasons about one job; a pass is a property of the client, and nothing here
could see it.

What the ruling needed, beyond the kind itself:

- `unload-denoiser`, for `unload-aligner`'s reason exactly — without it a separator could
  only be evicted by loading something else. There is no `load-denoiser`, for
  `load-aligner`'s reason: a load is seconds and is always immediately followed by the
  block it was loaded for.
- **A LEASE at the client's door**, and this half is not optional. `crucible/settle.py`
  clears the card the moment the last holder lets go, so an unleased run of ~44 blocks
  would reload per block *and be right to* — "one more block is coming" is the client's
  fact and nothing here can see it. `electron/crucible/denoise.ts` takes one after the
  first block (a lease names what is already resident) and releases it in `dispose()`.
- `done.extra.load_seconds` is the load time on the block that paid for it and `0.0` on
  every block after. A pass whose blocks all report a load is a pass that has lost the
  residency — which is how this hid the first time, with every job succeeding and every
  log clean.

**Only the primary stem is published.** The client slices the `(dry)` stem and discards the
rest, so shipping the others meant downloading a second ~233 MB copy of each block's noise
— about 10 GB across a 15-hour book — in order to delete it. Every stem is still measured
and still named in `done.extra.stems`; `done.artifacts` is what was published and
`done.extra.stems` is what was produced, and they differ on purpose.

### Three invariants, each one BookForge's and each one measured

- **The input must already be at 44.1 kHz.** The model is 44.1 kHz native and its librosa
  front-end crashes on other rates. Crucible does **not** resample: a stem returned at a
  rate the caller did not send is a stem whose sample offsets no longer mean anything, and
  the client is the one that knows what rate it wants back. The worker refuses before the
  model loads. (It is the worker's check and not the server's because the server's
  interpreter deliberately imports no audio library at all.)
- **Exactly one output names the primary stem** (`(dry)`, declared in the manifest). Zero
  means the model produced something other than what the manifest says it produces; two
  means nothing can say which one is the denoised audio. The app asserts the same thing.
- **The primary stem comes back the same length, sample for sample.** That invariant is
  what makes the client's offset slicing safe. It is enforced on the primary stem only,
  because that is the one it was measured on; the other stems' figures are reported and
  not enforced, rather than enforced on an assumption nobody has tested.

### The checkpoint has a real upstream, and `crucible denoise pull` fetches it

`denoise/denoise-roformer.toml` names both halves of the model's identity, because they
disagree: audio-separator resolves a model by **filename** inside its `model_file_dir`
(`denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt` and its `..._config.yaml`, read from
the library's own `models.json`), while the bytes come from paths that are named
differently. Neither is derived from the other.

The upstream is **`Politrees/UVR_resources` at `929e057b…`** — verified against the
HuggingFace API on 2026-09-13, with the checkpoint's LFS oid as its sha256 (913,097,300
bytes) and the config's digest computed from the bytes the API served. That matters
because audio-separator's *own* downloader pulls from a GitHub release, which DESIGN.md
section 5 refuses as a source of weights; the HF mirror is a source it allows.

**`crucible denoise list` and `crucible denoise pull <id> [--force]` are that command**,
and the ruling this section used to owe is discharged. They are `rvc list` / `rvc
pull-base`'s shape with nothing new invented: `weights.pull_files` does the fetching,
every digest is verified before either file is placed, the stamp is the same stamp, and
a pull that finished while leaving an expected file absent is **said out loud rather than
reported as success**. `crucible doctor`'s denoise row and `/v1/info`'s model row then
report what is here. Four refusals, each by name: an unknown id (naming what this build
ships), a model with no block for this backend, a digest mismatch (`weights`', which
places nothing at all), and the post-pull absence above.

Three things this had to decide, each of them a consequence of the *flat* directory:

- **The layout has one owner and it is `crucible/denoisemodels.py`.**
  `denoise_models_root(home)` and `model_files(manifest, spec)` are what the puller
  writes by and what `crucible/jobs/denoise/__init__.py` looks by, because the drift this
  prevents already happened once next door: `rvc`'s base assets were two lists and the
  job's was one file shorter, which showed up inside transformers hours later
  (ARCHITECTURE.md R1).
- **One stamp per model, not one per directory.** `~/.crucible/denoise-models` is flat
  because audio-separator resolves a model by filename inside one `model_file_dir`, so a
  single `crucible-pull.json` there would be overwritten by the second model's pull and
  would then report the first as never installed. `weights.pull_files` gained a
  `stamp_name` for exactly this, and the stamp is `crucible-pull-<id>.json`. For the same
  reason `--force` does not empty the directory — it replaces this model's two files and
  leaves anybody else's alone.
- **`config_bytes` joined the manifest**, beside the `model_bytes` that was already there.
  It is 1,621, read from the HuggingFace API at the pinned revision, and it is what lets
  `denoise list` say what a pull will cost before it runs. The `FileSource` shape
  `pull_files` takes wants a size per file, and a size nobody declared is a size nobody
  checked.

The job's refusal is unchanged in kind and better in content: `denoise_model_missing`
still names the repo, the revision, the two source paths and the two target names, and
now also names the command that places them — strictly more than `rvc`'s base-asset
refusal could say before it had one.

### Ruling owed

- **Does audio-separator reach the network even when both files are present?**
  `list_supported_model_files` fetches `download_checks.json`, and `load_model` fetches
  `mdx_model_data.json` / `vr_model_data.json`, each skipped only when the file is already
  in `model_file_dir`. So a job on a host with no route out may fail for a reason that has
  nothing to do with the model. **Not verified**, because verifying it means installing the
  env, and the first real `crucible install rvc` is what settles it. If it does, those
  three files join the two the refusal already names.
- **`use_autocast` on `mlx-darwin`.** Off, because audio-separator documents it as
  CUDA-only. Nobody has measured what the Mac does without it; BookForge's own denoise runs
  on the PC.

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
  "unattributed_bytes": 0,
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
  attributing it to compute processes is not a question `vm_stat` can answer. It is **never
  negative**: the subtraction goes below zero whenever the allowance already covers
  everything the driver can see — an idle 3090 Ti holding 1.7 GiB against a 3.0 GiB
  allowance read as −1.5 GiB until 2026-09-13 — and a client sizing a load against a
  negative is reading headroom that is not there. The example above shows the zero an idle
  card now reports.
- **`used_bytes` and `desktop_allowance_bytes`** — so a client can do the same arithmetic
  the guard does rather than inferring the host's declared facts.
- **`holders[].bytes` may be `null`**, where the driver will not say (WDDM, permissions).
  That is a refusal to answer and it is not zero.
- **`resident.kind`** is the family of the resident thing, not the job type that put it
  there. It was **the literal string `"llm"` beside `resident.model_id`** until phase 4 — an
  AttributeError the moment a voice or an aligner was resident, since neither has a
  `model_id`, and unreachable until `align` gave the route a third kind to meet. Both are
  now read off the resident itself (`resident.kind`, `resident.id`), so nothing here has to
  be remembered when a fourth kind lands.

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

**As built, the tail is in the error MESSAGE and not in a `details` field**, and that is
`asr`'s shape rather than a phase 4 decision: `workers._log_tail` appends the last 40 lines
to every `WorkerError`, and the job types turn that into a `JobError` whose message carries
it. A client reading `error.message` gets the engine's own last words with no second
request; a client that wanted them structured does not. Splitting them out is a change to
the error envelope and therefore to every job type at once, so it is a follow-up rather than
something `align` and `rvc` should do alone. The obligation the audit actually asked for —
*do not hand back a pointer to a file on another machine* — is met.

One addition phase 4 needed: a **session's** worker that dies mid-request is reported the
same way, naming the exit code and quoting the log, and the residency stops advertising it
rather than letting the next job write into a closed pipe.

## 6a. SDK additions (`@crucible/client`)

Written 2026-09-13, after sections 3 and 5 landed on the server.

- **`accelerator()` → `AcceleratorState`.** The probe, typed. The section-5 route shipped
  with **no SDK method at all**, and the reasoning was sound: the SDK's tests needed a live
  server, so the method could not be proved without one, and "every behaviour gets a test"
  outranked the convenience. That premise is now false — see the unit-test note below — and
  the method is the thing BookForge's `setGpuHolderProbe` seam has been waiting for.

  `holders[].bytes` is `number | null` and the null travels all the way to the caller.
  `unattributedBytes` is `number | null` and is not defaulted to zero on `mlx-darwin`:
  "nobody unaccounted for" and "unanswerable" are different answers. `resident.kind` is a
  plain string, not a union, because this section says the set grows.

- **503 `accelerator_unreadable` is its own type**, `CrucibleAcceleratorUnreadable`. It
  extends `CrucibleServerError` — the status really is a 5xx, and every phase-2 handler that
  catches one still catches this — with a narrower name for the callers that must act
  differently. That distinction is the entire point of the route: an unreadable probe is
  *ask again*, never *the card is free*, and a caller that cannot tell the two apart gets no
  more from this route than it had from the lock file.

- **`asr(options) → job id.`** `{model, audio, filename, language, vadFilter,
  wordTimestamps}`, every one required and none supplied by the client. `language` is sent
  as given: faster-whisper's own code list is the server's authority, it refuses naming the
  code before the job is queued, and a second copy of a hundred codes in the SDK is a second
  thing to drift. `filename` is a parameter and not a constant because the input's name
  becomes the file's name on disk and ffmpeg reads the container off the extension.

  It returns a job id, like every other submit; the transcript is the artifact
  `transcript.json`. There is no "job handle" type in this SDK and inventing one for `asr`
  alone would be a second way to watch a job.

- **`progress` now carries `extra`.** Section 3's three additional keys were reaching a
  typed client and being silently dropped, because `ProgressData` modelled `{fraction,
  message}` and nothing else. `JobContext.progress(fraction, message, **extra)` is open by
  design, so the SDK carries everything else on the frame verbatim — server spelling, server
  types — in `ProgressData.extra`, and `{}` where there was none. Modelling `stage` /
  `processed_s` / `total_s` / `cues` as named fields would have put one job type's vocabulary
  in the API's type, where the next job type's measurements collide with it.

- **`ModelInfo.modalities`** (phase 3c), **`Health.residentKind`** (PHASE3-TTS.md section 8)
  and **`ServerInfo.jobTypes`** (`fix(info)`, 2026-09-13) were all being sent and all being
  ignored. All three are read strictly now. `modalities` is never null on any host, unlike
  every other field on that row; `jobTypes` is what to POST and is deliberately not the
  capability list, so a client reads "what can I ask for" from it rather than discovering an
  unknown job type by being refused one.

- **`unattributedBytes` is surfaced exactly as it arrives, negative or not.** The server
  clamps it at zero as of `fix(accelerator)`, and a negative from an older one is that
  server's bug. Clamping it in the client too would put the correction in the wrong
  repository and make the real fault unfindable — the same argument as the one against the
  CPU fallback in section 3.

**The unit suite no longer needs a server.** It answers each route from an in-process
`node:http` fixture — the harness phase 1 already had, extended — which is how a test asserts
that `/v1/accelerator` is called with the bearer token and the version header, that a holder
row missing `bytes` is a `CrucibleProtocolError` rather than a zero, that a 503
`accelerator_unreadable` never resolves to a state, and that `asr()` refuses each missing
option by name before it sends anything. 52 tests to 84. `scripts/e2e.sh` still proves the
live seam, and gained two cheap cases that need no weights: the probe answers a state or the
named refusal, and `voices()` is refused `job_type_disabled` on the echo-only server it
starts.

**Not built, deliberately: `render()` and `stream()`** (PHASE3-TTS.md sections 6 and 7).
Their wire is still being written in Python, and a client for a contract that may move is how
the two ends come to disagree silently.

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
