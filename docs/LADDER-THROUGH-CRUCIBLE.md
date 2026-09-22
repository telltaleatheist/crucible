# The ladder through Crucible, for good

Owen, 2026-09-21 (relayed by the training-PC session): *"we'll just use our normal
ladder script tonight but I'd like to get it worked out."* This is the requirement
list from the ladder's side, checked item by item against Crucible `main` (1.0.21)
and narrator at the pin (bookforge `52807873`), with what is already there, what is
left, and which of the leftovers is a ruling rather than code.

The one-line summary: **the ladder box is on a stale narrator env.** Items 1, 3 and
the lease half of 5 are built at the pins above; the box has not seen them. Items 2
and 4 are half-built and need a keeper and two fields. Item 5's lifecycle half is a
ruling. Item 7 is the acceptance run and needs the card.

## 1. Judging and batching are separate levers — BUILT, not on the box

`retake: false` no longer means "render one at a time". Since 2026-09-20 an unjudged
Higgs batch goes through the engine's own batched driver with the batch's `width`
(narrator `serve/worker.py`, the `_can_guard_its_own_batch` branch: *"unjudged batches
routed here 2026-09-20"*), with `tracker=None` building an `UnjudgedPlan` — one take per
row, exactly as sent, no verdict. The sequential comprehension in
`_generate_audio_batch` is reached only by engines with no batched driver (Orpheus on
the Listen door, `_warmup`). The measurement the box made — `#running-req` pinned at 1
across 684 scheduler lines at width 4, 12.9 s/chunk against 4.16 s direct — is the
pre-2026-09-20 path and is the reason the branch exists.

On `mlx-darwin` the width comes from the MLX `BatchGenerator` tiers, the same branch.

**Left:** nothing to write. The acceptance run (7) is what proves `#running-req`
reaches the voice's `max_num_seqs`, and it needs the reinstalled env.

## 2. Seed lanes survive batching — RULE HOLDS, KEEPER OWED

The seed is `seed + index`, shifted by the take (`HiggsV3Engine._seed_for` and
`_request_seed`; the fake engine restates the rule at `serve/fake_engine.py:549-558`).
Batch position, batch size and completion order do not enter it, and the unjudged plan
pins `attempt` to 0, so cell `(index, take)` draws the same seed on every checkpoint.

**Left:** one explicit keeper, on the batched driver, that renders the same rows in two
batch orders and at two widths and asserts every chunk's recorded `seed` is identical.
The fake engine already records `seed` per render (`_record_render`), so the test is
cheap. Without it the invariant is a comment.

## 3. Each chunk lands on disk as it completes — BUILT

The render job publishes every chunk the moment it finishes:
`jobs/tts/render.py:1397-1401` writes `<index>.flac` to scratch and calls
`ctx.artifact(..., index=chunk.index)`, and `jobs/base.py:411` copies it into the job's
`artifacts/` directory and writes its provenance sidecar **immediately**, then records
the index in `chunks_done`. A clean `systemd stop` mid-job keeps every chunk that had
finished. The "128 chunks held in memory until the end" the box saw is not this door at
this version; before believing it, read `artifacts/` during a run on 1.0.21.

**Left:** nothing to write. The ladder client's skip-if-artifact-exists already makes an
interrupted job resumable against `chunks_done`.

## 4. Per-chunk progress in the job record — HALF

`GET /v1/jobs/<id>` carries `chunks_done` (sorted indexes, `api.py:3995`) and the
`progress` fraction and message. The events stream carries a `chunk` event per chunk.

**Left:** two fields on the job read — `chunks_total` (the request's row count, so
done/total is one read and not a client's memory of what it sent) and `chunk_at`,
the wall-clock stamp of the last chunk event, so pace is measurable from the record
without tailing an engine log that dies with the engine.

## 5. Lifecycle — ONE RULING, ONE ALREADY-BUILT

**The lease across takes is built.** `POST /v1/models/{id}/lease {act, ttl_seconds}`
takes whatever is resident — there is no `kind` on the body; the server reads it off
the resident — and the `load-voice` job accepts `lease: {act, ttl_seconds}` on the
load itself (`jobs/tts/__init__.py:126`, the 1.0.13 lease-on-load path). A ladder that
opens a lease at its first load and heartbeats it between takes is protected by the
deploy guard for the whole ladder, not only while a job is on the lane. That is the
recommended shape; nothing new is needed.

**The register/unregister race is a ruling.** Today `DELETE /v1/voices/{id}` answers
409 `voice_in_use` while the voice is resident and 404 `voice_not_custom` when there is
no home manifest to remove (`api.py:2415`). A DELETE that raced a restart got neither a
204 nor a retry, and `ladder-screen` stayed registered for three hours. Proposed:

- DELETE is idempotent on the *reached* state: a voice that is simply absent answers
  204 (the state asked for is the state there is). 404 stays for a **packaged** voice,
  which cannot be removed here and should say so.
- 409 `voice_in_use` stays. A ladder that wants the voice gone unloads first, or holds
  the lease and lets its own release settle the card.
- A voice registered with a `path` backend and no lease, no job and no residency after a
  restart is an orphan: reported by name in `GET /v1/voices` (`orphan: true` with the
  reason), never deleted on the server's own judgment. Garbage collection is a person's
  or the ladder's `DELETE`, which is now safe to repeat.

Needs Owen's yes on the third bullet before it is code: it is the only one that changes
what a read says about somebody else's voice.

## 6. Keep what works — UNCHANGED

`context_length` honored on `cuda-linux` (8192 read back, no silent 4096 truncation of
the 2,008-char rung); `[voice.pace]` omissible; `[voice.serving]` required with
`max_num_seqs`; the deploy guard (`4a7992d`). None of the work above touches them, and
the acceptance run reads `context_length` back as its first check.

## 7. Acceptance — NEEDS THE CARD

One run over `bank_ds_len.json` (128 prompts × takes 0–3) against
`~/higgs_v3_merged/ds_v9_recut1_3510` and `_3393`, same card, through Crucible:

- 512/512 per checkpoint;
- whole-run average within ~1.3× of the direct path's measured 4.16 s and 5.49 s per
  render;
- `artifacts/<index>.flac` appearing on disk during the run, and `chunks_done` growing
  with them;
- the per-cell seeds identical to the direct-path renders the box holds.

**Order of work:** reinstall the narrator env on the ladder box (the pin, not a copy) →
the seed keeper (2) → `chunks_total` / `chunk_at` (4) → the DELETE ruling (5) → the
acceptance run (7) in the GPU slot Owen names. The GPU is training `ds_v10` tonight; the
slot is tomorrow or later and is asked for, never taken.
