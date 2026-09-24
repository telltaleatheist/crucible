# `envs/asr/mlx-darwin.txt` exists now — and it is a different engine, not this recipe

This file used to explain why the Mac had no `asr` recipe. It has one, beside
this note, since 2026-09-14. What is worth keeping here is the part that has
not changed: **the Mac's whisper is not `cuda-linux.txt` with the CUDA lines
taken out.**

## Why `faster-whisper` is still not on this backend, and never will be

`asr` on `cuda-linux` is faster-whisper, and faster-whisper is CTranslate2.
**CTranslate2 has no Metal backend.** It supports CUDA and CPU; on Apple Silicon
it builds against Accelerate and runs on the CPU cores, and a request for
`device="mps"` is a `ValueError: unsupported device mps`
(SYSTRAN/faster-whisper#515 and #911, both still open and still true as of
2026-09).

Running it on the Mac's CPU is not the compromise it looks like.
PHASE4-AUDIO.md section 3 rules on the same question in the other direction:
`compute_type` is `float16` on an accelerator and `int8` on CPU, and a
transcript that quietly ran at `int8` on a CPU is a *different transcript* with
nothing in the output to say so. Crucible has no CPU backend and does not want
one by the back door.

## What landed instead

`mlx-whisper==0.4.3`, as a **second engine** behind one job type:

- `envs/asr/mlx-darwin.txt` — `pip freeze` from a python 3.11.16 env built from
  that one pin on the Mac Studio on 2026-09-14. That env is where the seven
  manifests' memory figures were measured, and it was deleted afterwards.
- `mlx-darwin: "mlx-whisper"` in `ASR_BACKEND_ENGINES`
  (`crucible/asrmodels.py`), with a second table — `ASR_ENGINE_ID_PREFIX` — the
  loader enforces.
- **Seven new model ids**, all `mlx-whisper-*`: tiny, base, small, medium,
  large-v3, large-v3-turbo, distil-large-v3, from `mlx-community`'s own
  conversions, each pinned to a full commit sha verified against the hub API.
- A second worker, `crucible/jobs/asr/mlx_worker.py`, speaking the identical
  JSON-lines wire — so `transcript.json` is one document whichever machine made
  it, and BookForge's align and transcript readers change nothing.
- `asr/mlx-darwin` in `.github/workflows/envpacks.yml`.

## The two rules that keep this honest

> **2026-09-24: rule 1 was replaced.** Owen's asr lineup ruling kept three
> models — `whisper-large-v3-turbo`, `qwen3-asr-1.7b`, `whisper-tiny` — and made
> each ONE id on both backends: the Mac's turbo and tiny are the
> `[backends.mlx-darwin]` blocks of `asr/whisper-large-v3-turbo.toml` and
> `asr/whisper-tiny.toml`, at the same pins as before. The other five mlx sizes
> (base, small, medium, large-v3, distil-large-v3) were removed. A whisper id is
> now two conversions, and a transcript's provenance sidecar says which one —
> backend, engine, repo, revision — which is what the id used to have to say.
> The loader binds engines to a FAMILY now (`ASR_ENGINE_FAMILY`), not to an id
> prefix. Rule 2 stands. The measurement table below is the record of
> 2026-09-14; its two surviving rows are the manifests' figures today.

**1. The ids do not cross.** *(History — see the note above.)* `faster-whisper-large-v3` and
`mlx-whisper-large-v3` are different conversions of the same original
checkpoint, at a different quantisation, by a different library, and they will
disagree about a hard passage. `transcript.json` records the model id and
nothing else about the bytes, so an operator comparing two transcripts has to
be able to tell from the id which engine produced each. The loader refuses a
manifest whose id does not begin with its engine's prefix — a convention
nobody can forget.

**2. `vad_filter: true` is a REFUSAL here, not a silent no-op.** faster-whisper
ships Silero VAD; mlx-whisper has no voice-activity detector at all. Its
`no_speech_threshold` is the model's own per-segment judgement, which is a
different mechanism on different evidence and not a substitute. So `crucible/
jobs/asr` answers `400 vad_unsupported_by_engine` before the job is queued —
the CPU argument above, one layer up. `vad_filter: false` runs perfectly well.

One thing the wire GAINS rather than loses: `language_probability`.
`mlx_whisper.transcribe()` does not report it, so the worker runs whisper's own
`detect_language()` on the window's first 30 seconds — exactly what transcribe
does internally when no language is given — and reports that probability, then
passes the detected code in so the detection is not run twice. Measured on the
M1 Ultra, 2026-09-14: `en` at 0.9946824908256531.

## What was measured, and what was not

Every one of the seven `memory_bytes_estimate` figures is
`mx.get_peak_memory()` over ONE 900-second window — `WINDOW_SECONDS`, the unit
this job type actually cuts a book into — on the Mac Studio on 2026-09-14, with
`word_timestamps=True`. Each manifest carries its own run. Summary, with the
realtime factor from the same runs:

| id | peak bytes | 900 s took | realtime |
|---|---|---|---|
| `mlx-whisper-tiny` | 549,418,642 | 11.95 s | 75.3x |
| `mlx-whisper-base` | 877,017,662 | 13.11 s | 68.6x |
| `mlx-whisper-small` | 1,540,273,318 | 31.11 s | 28.9x |
| `mlx-whisper-medium` | 2,607,243,002 | 36.39 s | 24.7x |
| `mlx-whisper-large-v3` | 4,153,379,610 | 142.01 s | 6.3x |
| `mlx-whisper-large-v3-turbo` | 2,654,916,970 | 35.43 s | 25.4x |
| `mlx-whisper-distil-large-v3` | 2,549,972,298 | 25.84 s | 34.8x |

**`large-v3` is four times slower than `turbo` on this backend** for the same
128-mel encoder, because it decodes with thirty-two layers against turbo's
four. An 18-hour book is about three hours of Mac on `large-v3` and about
forty minutes on `turbo`. That is a real trade and both ids exist to offer it.

**Not measured, and it matters: ACCURACY.** Nobody has run mlx-whisper and
faster-whisper over the same book and compared the transcripts. The two are not
interchangeable at one id — which is what the id rule above is for — and
nothing here says which is better on Owen's material. The memory and the speed
are watched numbers; the quality is an open question, and the honest place to
answer it is one book through both.
