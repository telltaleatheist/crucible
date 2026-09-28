# Resumable jobs: the journal (2026-09-27)

A Crucible job can run for hours, and a book can take days. Until this, a job
that failed at hour three of four kept nothing but what it had already
published: a Qwen3-ASR transcription of *The Coming of the Third Reich* (3,015
pieces) that died in the aligner had two hours of decoding and nothing to show
for it. This document is the contract for keeping that work.

## 1. The rulings

Owen, 2026-09-27, verbatim:

> "we should definitely be writing work to disk, so if something fails, we dont
> lose everything. preferably writing to disk often. the nature of crucible
> necessitates very long jobs. sometimes stretched over days. one failure would
> lose a lot of work."

> "im thinking resuming can be a specific flag. if the user doesnt send the
> resume flag then it starts fresh. if they do send a resume flag, it continues
> from where they left off. maybe we could even have a call that shows what's
> available to resume?"

And on who retries: Crucible does not queue. *"It grants and releases leases.
That's it."* So the server never resumes anything by itself. It keeps the
journal; the app decides whether and when to send `resume`.

## 2. What a client sees

- **Every submit of a journaled job answers `resume_id`** beside `job_id`
  (`POST /v1/jobs` → `{"job_id", "resume_id"}`; `null` for a job type that
  keeps no journal). `GET /v1/jobs/{id}` carries it too, with `resumed`.
- **Without `resume`, a job starts fresh and writes a NEW journal.** It never
  reads an old one. Earlier journals are kept, never overwritten, so forgetting
  the flag destroys nothing.
- **With `params.resume: "<resume_id>"`** the job continues that journal after
  the server checks it is the same work. The check runs before the job exists
  and before any upload is moved into it, so a refused resume costs nothing:

  | code | status | when |
  | --- | --- | --- |
  | `resume_mismatch` | 409 | the job type, the model or its exact revision, an output-affecting param, an input's sha256, or the type's format version differs. The sentence names the first difference, e.g. *"the input 'book.m4b' is not the same file: sha256 2f00… (34 bytes) in the journal vs cdba… (12 bytes) now"* |
  | `unknown_resume_id` | 404 | this server has no such journal |
  | `invalid_resume_id` | 400 | not 32 hex digits |
  | `resume_expired` | 410 | the journal was collected (`retention_days`) or discarded; the id is remembered by a tombstone, so it is never called unknown |
  | `resume_in_use` | 409 | a queued or running job is writing it |
  | `resume_unsupported` | 400 | the job type (or, for `asr`, the whisper engine) keeps no journal yet |

- **The events say what was resumed.** A resumed job's stream carries a `note`
  whose message starts `resumed:`, e.g. *"resumed: 0 of 40 pieces done (20
  decoded, 0 aligned), from journal d932… (last saved …)"*, with `resume_id`,
  `resumed_from`, `units_done` and `units_total` beside it. The job type adds
  its own notes as it skips work (`20 of 40 piece(s) already decoded in journal
  …; decoding 20`).
- **A plan that changed underneath is refused, not stitched.** Some checks can
  only run once the work starts (Qwen's piece plan needs the audio decoded).
  Those fail the job by name (`resume_plan_mismatch`, `resume_mismatch`)
  before any unit is redone.

### The routes

| route | what |
| --- | --- |
| `GET /v1/resumable` | `{"resumable": [row, ...]}`, newest first by `last_saved` |
| `GET /v1/resumable/{resume_id}` | one row |
| `DELETE /v1/resumable/{resume_id}` | discard it: `{"resume_id", "discarded": true, "units_done", "units_total"}`; refused `resume_in_use` while a job writes it |

A row:

```json
{
  "resume_id": "46ab37b3285e42558106a79d81353a53",
  "job_type": "asr",
  "model": {"id": "qwen3-asr-1.7b", "revision": "<the pinned revision>"},
  "format_version": 1,
  "inputs": [{"name": "book.m4b", "sha256": "2f00…", "bytes": 1203338113}],
  "params": {"engine": "vllm", "dtype": "bfloat16", "language": "en", "...": "..."},
  "units_done": 2400,
  "units_total": 3015,
  "progress": "2,400 of 3,015 pieces done (3,015 decoded, 2,400 aligned)",
  "created": "2026-09-27T16:19:22+00:00",
  "last_saved": "2026-09-27T18:02:11+00:00",
  "expires_at": "2026-10-04T18:02:11+00:00",
  "job_id": "<the job that started it>",
  "last_job_id": "<the job that last wrote it>",
  "state": "interrupted",
  "jobs": [{"job_id": "…", "state": "interrupted", "resumed": false}]
}
```

`state` is how the last writer ended: `queued`, `running`, `done`, `failed`,
`cancelled` or `interrupted`. A writer recorded as running whose server has
since stopped reads `interrupted`. `params` is the resolved, output-affecting
params, so an app can rebuild the resume submit from the row alone (plus its
own copy of the audio).

CLI: `crucible api resumable list|get|discard`, and `crucible api job submit
… --resume <resume_id>`. TS SDK: `client.resumable()`,
`client.resumableEntry(id)`, `client.discardResumable(id)`,
`JobStatus.resumeId` / `resumed`, `AsrOptions.resume`.

## 3. The journal on disk

`crucible/journal.py`. Under the Crucible home, NOT under `jobs/`, because
`JobStore.reap` deletes a job's directory once its artifacts are fetched, and a
journal must outlive every job that writes it.

```
<CRUCIBLE_HOME>/journals/<resume_id>/manifest.json     what the work is, and how far it got
<CRUCIBLE_HOME>/journals/<resume_id>/units/<key>.json  one finished unit each
<CRUCIBLE_HOME>/journals/<resume_id>/units/<key>.bin   the bytes of a file unit (put_file)
<CRUCIBLE_HOME>/journals/_gone/<resume_id>.json        tombstone of a collected or discarded journal
```

**Every file is written whole or not at all:** a temporary name in the same
directory (dot-prefixed, so no reader opens it), flush, fsync, `os.replace`
into place, then fsync the directory where the platform allows (not Windows,
which has no directory handle to sync). A crash mid-write leaves a dot-named
temporary, never a half unit that reads as finished. `os.replace` is retried
briefly on Windows' `PermissionError` (a reader holding the manifest).

`manifest.json`:

| key | what |
| --- | --- |
| `journal_format` | this container's own version (1) |
| `resume_id` | 32 hex digits |
| `job_type` | e.g. `asr` |
| `model` | `{"id", "revision"}`, the exact pinned revision |
| `format_version` | the job type's unit format (section 5) |
| `params`, `params_sha256` | the output-affecting params, resolved, canonical JSON (sorted keys) |
| `inputs` | `[{"name", "sha256", "bytes"}]` |
| `units_done`, `units_total`, `progress` | the job type's own count, and a sentence |
| `created`, `last_saved`, `expires_at` | ISO-8601 UTC; `expires_at = last_saved + retention_days` |
| `job_id` | the job that started it |
| `writer`, `jobs` | the job writing it now and how each writer ended |

A unit file is `{"key", "saved_at", "data"}`. Units are written the moment each
exists. The manifest's counts are rewritten at most every 2 s while a job runs
(`PROGRESS_WRITE_SECONDS`) and always when it ends: the units are the truth,
and a count a second stale costs nothing a restart would notice.

**Retention.** The existing `[jobs] retention_days` collector takes journals
too: `JobStore.reap` calls `Journals.reap` on the same idle-lane tick, by the
same window, measured from `last_saved`. A journal whose writer is queued or
running is never taken, however old. There is no second collector.

**Input digests.** An upload's sha256 is the one `POST /v1/uploads` already
wrote beside it, so a four-hour book is not read twice; inline bytes are hashed
in memory; an artifact input (`{"artifact": {job_id, name}}`) is hashed from
disk at submit, the one case that reads the file.

## 4. The reference: Qwen3-ASR (`crucible/jobs/asr/qwen.py`)

`JOURNAL_FORMAT_VERSION = 2` (2 since the loop guard recognises a context echo). A unit is keyed by the piece it belongs to:
`piece_key` = `L<level>.<first sample>-<last sample>` of the piece's core on the
worker's own 16 kHz timeline (integers, never a float's spelling).

| unit | written | holds |
| --- | --- | --- |
| `plan.L<level>.<all\|region>` | after each split | window, overlap, duration, samples, the `speech_only` kept table, every piece's `[core start, core duration, audio start, audio duration]` |
| `text.<piece>` | as each result lands off the worker (the worker sends a batch's results when the batch is decoded) | `text`, `tokens`, `hit_token_limit` |
| `words.<piece>` | after each aligner batch of 16 | the aligner's `items`; a piece the aligner failed is NOT written, so a resume after `asr_align_failed` re-aligns only the failures |
| `verdict.<piece>` | when the piece's fate is decided | `landed` (with its owned text and word count), `silent`, `context_echo` (span and echoed word count; the piece is done and left empty), or `redecode` (the loop guard's `redecoded` entry verbatim) |

**Identity** (`AsrJobType.journal_identity`): the model and revision, the
engine and the dtype it runs at on this card, `language`, `context`,
`word_timestamps`, `vad_filter`, `piece_s` and `overlap_s` RESOLVED (so `null`
and `30` are the same work), `speech_only` with its three settings and the
detector's sha256, and the aligner's id and revision. Not in it: `resume`
itself, and the serving width (how many pieces go at once, not what any says).

**The plan is deterministic from audio + params, and checked.** Every cut is
the centre of the quietest 100 ms window in the search span before the nominal
cut, or with `speech_only` the latest join in that span
(`qwen_worker.split_points`), over samples ffmpeg decoded, with no randomness;
Silero is numpy on the CPU. A resume recomputes the plan and compares it with
the journal's before any piece is decoded; a different plan (another ffmpeg
decoding the container differently, a changed cutter) is
`resume_plan_mismatch`, because the journal's text belongs to other audio. Each
loop-guard re-cut's plan is journaled and checked the same way.

**On resume** every piece with a `text` unit is not sent to the engine, every
piece with a `words` unit is not sent to the aligner, and each verdict is
recomputed from the journaled text and word times and must equal the recorded
one (else `resume_mismatch`: the loop guard or the ownership rule changed
without the format version moving). Everything after the units is a pure
function of them, so the transcript comes out the same. A resumed run loads the
aligner only if some piece still needs word times; it still loads the ASR
engine, because the worker decodes the audio to recompute the plan.

**Whisper** (`asr/__init__.py`, faster-whisper and mlx-whisper) keeps no
journal yet and refuses `resume` as `resume_unsupported`.

### The check (2026-09-27, stub workers, no GPU)

A real server (`create_app`, uvicorn, the real lane, journal, queue and
`QwenAsrRun`) on a scratch `CRUCIBLE_HOME`, with the ASR worker replaced by a
stub that speaks `qwen_worker.py`'s wire (batched results, a per-piece delay, a
switch to die mid-request) and `tests/fake_align_worker.py` as the aligner. The
input: 1,200 s in 30 s pieces with word timestamps, one silent piece, one
token-limit loop and one aligner collapse, each re-decoded at 15 s: 42 pieces,
44 decodes, 42 alignments, 133 unit files.

| interruption | resumed `transcript.json` and `transcript.text.json` |
| --- | --- |
| the ASR worker died after 20 results (job `failed`, `worker_failed`) | byte-identical |
| the same journal resumed again after the job directory was reaped (`job_reaped`) | byte-identical |
| the server process hard-exited in the middle of writing unit 23 (a text unit; half a temp file left) | byte-identical; the half-written temp was ignored, the job read `interrupted` after restart |
| the server hard-exited mid-write of unit 75 (a words unit, 30 aligned) | byte-identical |
| `taskkill /F /T` of the server tree mid-align (16 of 38 aligned) | byte-identical; *"16 of 38 piece(s) already aligned …; aligning 22"* |
| the job cancelled mid-decode | byte-identical |

The only differences are in the provenance SIDECAR, never the transcript:
`job_id`, `started`, `finished` (a different job, at a different time), and
`params` / `params_sha256` (the resumed job's params carry `resume`). The
refusals came back as the table in section 2 says, including a changed plan
(`STUB_SPLIT_SHIFT`: *"piece 0 is 0.000+31.500 s now and 0.000+30.000 s in the
journal"*). Retention was checked on `Journals.reap` with the clock moved: kept
at +6 days, collected at +8, a running writer's journal kept regardless, and the
id then answering `resume_expired`.

**What the stub cannot show.** vLLM's greedy decoding is not guaranteed
batch-invariant: the same piece decoded beside different neighbours can, rarely,
come out differently. A resume after a mid-batch failure pairs the remaining
pieces differently from the uninterrupted run, so on the PC a resumed
transcript is a correct transcript of the same plan but is not guaranteed
byte-identical to a run that was never interrupted. At level 0 the worker
publishes a batch's results together, so a resume usually starts on the
original batch boundary and pairs pieces the same way. The Mac engines
(mlx-audio, qwen-asr) decode one piece per call and are unaffected.

## 5. The contract for the next job types (rvc, denoise, whisper, align-longform)

To bring a job type onto the journal:

1. **Say what a unit is.** The smallest piece of work whose result can be kept
   and trusted by itself, and whose loss would be worth avoiding. It needs a
   stable KEY computed from the request and the plan, never from a position
   in a list that can shift (`[A-Za-z0-9][A-Za-z0-9._-]*`, at most 200
   characters).
2. **Declare the identity.** Implement `journal_identity(model, params) ->
   journal.Identity | None` on the job type (optional; `POST /v1/jobs` looks for
   it). It returns `job_type`, `model`, the exact `revision`, the type's
   `format_version`, and `params`: every param that changes a unit's content,
   RESOLVED to the value the run will use (defaults filled in), and nothing
   else (never `resume`; not speed knobs). Return None for a model or mode that
   keeps no journal, and refuse `resume` for it with `resume_unsupported` in
   `preflight`. A type without the method gets `resume_unsupported` from the
   submit door for free. Add `resume: StrictStr | None = None` to the params
   model.
3. **Keep a `JOURNAL_FORMAT_VERSION` constant** beside the job type and bump
   it whenever a unit written by one build would mean something else to the
   next: the unit's keys or values, the key scheme, the plan's shape, or any
   rule whose answer is recorded.
4. **Write each unit the moment it exists**: `ctx.journal.put(key, data)` for
   JSON; `ctx.journal.put_file(key, path, data)` for bytes (a converted file, a
   stem), which writes `<key>.bin` durably before the `<key>.json` that says
   the unit is finished. Call `ctx.journal.progress(done, total, sentence)` as
   units land (it throttles itself); the sentence is what `GET /v1/resumable`
   shows a person.
5. **Read back only on resume.** `ctx.resumed` is true only for a job sent
   `resume`; a fresh job never reads its journal. Skip every unit present,
   `ctx.journal.get(key)` / `ctx.journal.file(key)`, and redo the rest.
6. **Record any plan and check it.** If the units depend on a plan computed
   at run time (Qwen's pieces, whisper's windows, align-longform's chunks),
   journal the plan and refuse a resume whose recomputed plan differs
   (`resume_plan_mismatch`). Never stitch units onto a plan they were not
   made for.
7. **Record decisions and check them.** If the job decides something from a
   unit (Qwen's loop guard), journal the decision and refuse a resume that
   decides differently from the same units (`resume_mismatch`).
8. **Prove it byte-identical.** With stub workers on a scratch
   `CRUCIBLE_HOME`: an uninterrupted run, then the same job interrupted each
   way (worker dies, server hard-exits mid-write, cancel), resumed, and every
   artifact compared byte for byte. Only the provenance sidecar's `job_id`,
   `started`, `finished`, `params` and `params_sha256` may differ. Write the
   result into this document, with anything the stub cannot show.

Suggested units, for the agents taking these on:

| type | a unit | notes |
| --- | --- | --- |
| `rvc` | one input file converted (`put_file`) | key from the file's input name and index; identity includes the voice, its pitch and index settings |
| `denoise` | one separated file (`put_file`) | key from the input name; identity includes the separator and its settings |
| `asr` whisper | one 900 s window's segments | the worker is one-shot today (`run_worker`); it needs `on_result` per window and a request field naming windows to skip. The window plan (duration, `speech_only` kept table) is the plan to record |
| `align-longform` | each stage's output: the coarse ASR pieces, then each chunk's alignment | its stages already drive the asr and align workers directly (`alignlongform/stages.py`), so the Qwen reference's per-piece units carry over for the coarse pass |
