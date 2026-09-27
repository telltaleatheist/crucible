# ASR and alignment internals

Covers `crucible/jobs/asr/`, `crucible/jobs/align/`, `crucible/jobs/alignlongform/`,
`crucible/asrmodels.py`, `crucible/alignmodels.py`, `crucible/asrplan.py` and the manifests
in `crucible/asr/*.toml` and `crucible/align/*.toml`.

The Qwen3-ASR plan was [history/PHASE25-QWEN-ASR.md](../history/PHASE25-QWEN-ASR.md) (not maintained); the journal and
`resume` are [RESUMABLE-JOBS.md](../RESUMABLE-JOBS.md). The original design is
[history/PHASE4-AUDIO.md](../history/PHASE4-AUDIO.md) sections 2 and 3. This file holds what the code alone
does not say.

## Standing principles

- **No silent substitution.** A param an engine cannot honour is a 400 naming it, never a
  no-op: `vad_filter: true` on an engine with no VAD (mlx-whisper, both Qwen engines),
  `initial_prompt` on Qwen, `context` on whisper, `resume` on whisper
  (`resume_unsupported`), `piece_s`/`overlap_s` on whisper, overlap without word
  timestamps, a speech knob without `speech_only`. A transcript made under rules the caller
  did not ask for has nothing in it to say so.
- **No default model, no CPU fallback.** A job names its model or `resolve_model` refuses
  it. BookForge retries a failed CUDA load on CPU at int8; Crucible never does. It has no CPU
  backend, and an int8 CPU transcript is a different transcript.
- **A hole is a failure.** A failed whisper window, a Qwen piece the aligner failed, or a
  piece that loops past the last rung fails the job and publishes no `transcript.json`.
  Workers keep going after a window or chunk fails so one run finds every bad stretch.
- **Position is identity.** No worker reports an index. The n-th result is the n-th
  window/chunk/piece sent. (Narrator's aligner got an index wrong on a 401-chunk book.)
- **fd 1 is the protocol only.** Every worker dups fd 1 away and points it at stderr before
  importing anything that logs. A whisperx `StreamHandler(sys.stdout)` warning landing
  between two result lines once failed a whole book with "Extra data".
- **Workers are standalone.** `worker.py`, `mlx_worker.py`, `qwen_worker.py`,
  `align/worker.py` and `speechonly.py` import the standard library, numpy, their engine and
  `workerio`, never `crucible`: their envs do not have it. `speechonly.py` imports numpy
  inside its functions so the server can use its constants and `Timeline` without numpy.
- **Nothing reaches a worker as a library default.** Every request key is required; a null
  is an explicit value (`language: null` = detect, `initial_prompt: null` = no prompt,
  `speech: null` = off). A producer that stopped sending a key would silently change what
  runs.
- **Never evicts.** An `asr` job does not unload a resident model to make room
  (no `reclaimable_bytes`). An `align` job may load over a resident voice, as an `llm` load
  may, but `preflight` refuses while a streaming session holds it.

## The model lineup and manifests

Owen, 2026-09-24: the caller picks from a fixed lineup, each ONE id across both backends:
`whisper-large-v3-turbo`, `whisper-tiny`, `qwen3-asr-1.7b`, `qwen3-asr-0.6b`, plus the
Mac-only MLX ports `qwen3-asr-1.7b-mlx` and `qwen3-asr-0.6b-mlx` (*"if i want speed i can get
it via mlx"*). `asrmodels.ASR_LINEUP` states it for tests; the directory is the lineup.

Rules `asrmodels._parse` enforces:

- **Family binds engine.** A block's engine must belong to `[model] family`
  (`ASR_ENGINE_FAMILY`) and the id must start with the family, so no id is whisper on one
  machine and Qwen on the other.
- **One checkpoint where engines can share one.** Qwen blocks must pin the same repo and
  revision on every backend (`ONE_CHECKPOINT_FAMILIES`), so a community conversion cannot
  slip in under the official id. Whisper is exempt: CTranslate2 and MLX cannot read each
  other's format, so a whisper id is two conversions, told apart by the provenance sidecar
  (`model_provenance` records backend, engine, repo, revision).
- **Full precision.** Qwen blocks state `bfloat16` (`QWEN_ASR_DTYPES`); never the 8-bit MLX
  build. On a card below compute capability 8.0 vLLM refuses bf16 and the engine starts in
  float16 (`engines.vllm.run_dtype`, `qwen.run_dtype_on`): same two bytes a parameter, so the
  memory figures hold. Owen, 2026-09-26: *"we can quantize if we need to. no less than 4."*
- **Qwen blocks state everything** (`QWEN_BACKEND_REQUIRED`, `VLLM_BACKEND_REQUIRED`):
  `max_batch`, `max_new_tokens`, and on vLLM `max_model_len`, `kv_cache_memory_bytes`,
  `kv_bytes_per_token`. `qwen_asr`'s default batch of 32 aborted an MPS process
  ("too large for kernel"); batch 4 used 41 GB.
- **Mac engines take one piece per call** (`max_batch = 1`, `_check_qwen_block`), so the loop
  guard reads each piece's own token count.
- **`weights_of`**: the `-mlx` ids are the same bytes as their official sibling, so they
  share its download folder (one copy on disk, Owen 2026-09-24). The base must be a
  non-alias of the same family pinning the same repo and revision.
- Old per-engine ids (`faster-whisper-*`, `mlx-whisper-*`) are gone, not aliases:
  `unknown_model`. Stranded weights are reported by `crucible doctor`, never deleted.

Engines: `faster-whisper` (cuda-linux, asr env), `mlx-whisper` (mlx-darwin, asr env),
`vllm` (cuda-linux, llm env), `mlx-audio` (mlx-darwin, llm env), `qwen-asr` (Qwen's own
package on torch MPS, mlx-darwin, **align env**: it already pins qwen-asr 0.0.6,
transformers 4.57.6, torch 2.14.0, ContentStudio's measured versions). `qwen3-asr-1.7b` on
the Mac runs `qwen-asr`: on identical pieces it heard a few more fillers than the MLX port
(11-13 vs 9-10 on a 10-minute window) at 2.5x the time.

Device names are each library's own: `cuda` for CTranslate2, `metal` for MLX, `mps` for
torch (the aligner and `qwen-asr`). Workers refuse the other's spelling. There is no CPU
entry and no default device.

`asrmodels.py` and `alignmodels.py` duplicate most of `manifests.py`'s strictness; merging
them into one loader parameterised by (directory, required keys, engines) is an open
follow-up. The aligner is not in `models/` because `manifests.py` requires `params_b`,
`context_default` and `modalities`, which mean nothing for an aligner. The directory is
the job type.

### Manifest figures

Qwen3-ASR (both sizes): 28 layers, 8 KV heads, head_dim 128, so KV per token is
2 x 28 x 8 x 128 x 2 B = **114,688 B**. Audio tokens: 13 per second of audio (vLLM
`_get_feat_extract_output_lengths`), so a 180 s piece is 2,340 tokens.

- `max_model_len = 8192`: 2,340 audio + 1,024 context (`QWEN_CONTEXT_MAX_TOKENS`) + 64
  scaffolding (`QWEN_PROMPT_SCAFFOLD_TOKENS`) + 4,096 new = 7,524, rounded to a power of two. The
  loader refuses less than 7,524.
- `kv_cache_memory_bytes = 3,758,096,384`: 114,688 x 4,096 tokens x 8 pieces. Stated so
  vLLM skips its own profiling (and its 0.92-of-the-card default); that is also why the
  audio tower's activation reserve has to be a stated term.
- vLLM on cuda-linux `memory_bytes_estimate` (COMPUTED, not measured):
  weights + 1,476,395,008 non-KV overhead (the 9B's measured figure, an upper bound) +
  1 GiB declared audio-tower activations + the KV pool. 1.7B: 11,006,754,728 B;
  0.6B: 8,184,324,920 B.
- Mac figures are MEASURED on the M1 Ultra, 2026-09-24: `qwen-asr` 1.7B 9,126,805,504 B
  (`torch.mps.driver_allocated_memory()` peak, process also held the aligner, so an upper
  bound); 0.6B 6,232,440,832 B. `mlx-audio` 1.7B 7,726,809,000 B computed and consistent
  with the run; 0.6B 3,966,967,600 B measured.
- The estimate is the ASR engine alone. A word-timestamped job adds the aligner's own
  manifest figure (`qwen.need_bytes`), never a copy of it.

Whisper: faster-whisper blocks are weights + a DECLARED 1.5 GiB (CUDA context,
cuBLAS/cuDNN workspaces, one 30 s window's activations), not yet measured on a card.
mlx-whisper blocks are MEASURED `mx.get_peak_memory()` over one 900 s window on the M1 Ultra
(turbo 2,654,916,970 B at 25.4x realtime; tiny 549,418,642 B at 75.3x).

`whisper-large-v3-turbo` on cuda-linux pins `dropbox-dash/faster-whisper-large-v3-turbo`
(the canonical name behind faster-whisper 1.2.1's own `_MODELS` entry
`mobiuslabsgmbh/...`, which redirects there). Systran publishes no turbo conversion. It is
byte-identical to `deepdml`'s conversion and verified as turbo (128 mel bins, alignment
heads all on decoder layers 0-3, float16 by size).

`qwen3-aligner`: cuda-linux 5,905,580,032 B MEASURED on the 3090 Ti 2026-09-26 over a
300 s, 742-word window (peak reserved 4.55 GiB + 0.95 GiB CUDA context). It is also the cap
`workers.torch_memory_cap` puts on the worker, so an under-estimate becomes an OOM.
mlx-darwin 5,885,296,640 B is MPS **driver**-allocated memory (the caching allocator keeps
blocks while the session is resident), measured over three 300 s chunks, flat after the
second. Still owed: comparing MPS timestamps against CUDA's for the same revision
(`envs/align/mlx-darwin.md`). Bake-off (2026-09-08): 229x realtime and 51/61 cues exact vs
WhisperX's 18x and 39. The 0.6B is pinned by name; a larger sibling is a new measurement.

## Whisper runs (`jobs/asr/__init__.py`, `worker.py`, `mlx_worker.py`)

- **900 s windows, each extended 15 s past its boundary** (`WINDOW_SECONDS`,
  `OVERLAP_SECONDS`, BookForge's measured numbers). Handing `model.transcribe()` an 18-hour
  file frames it into ~19 GiB of float64 and OOMs. mlx-whisper would not OOM, but it windows
  too so both engines produce the same document (`window_s`, `overlap_s`, `windows`).
- **Dedup across windows**: sort segments by start, drop any that begin inside a kept span
  (0.1 s tolerance, `OVERLAP_TOLERANCE_SECONDS`). BookForge dedupes sentence cues instead, so
  boundary behaviour is close but not identical.
- The server shifts window n by `n * WINDOW_SECONDS`; the worker reports no index.
- **`initial_prompt` goes to every window.** Each window is its own `transcribe()` call and
  starts with an empty token history (faster-whisper 1.2.1, mlx-whisper 0.4.3); inside a call
  the prompt is the head of a history cut to its last 223 tokens and scrolls out after a few
  segments. The worker counts the prompt with the model's own tokenizer after load and
  refuses one longer than `max_length // 2 - 1` (it would lose its beginning silently). It
  is optional on the wire because fleet clients send only three keys; `transcript.json`
  records `initial_prompt: null`. Blank is refused (`""` and `None` would be two spellings;
  faster-whisper encodes `""` as `" "`).
- **Decode through ffmpeg**, not faster-whisper's PyAV `decode_audio`, which silently
  truncates some assembled m4b files (an 18-hour book decoded to six hours). Hence
  `_require_ffmpeg` is a 409 at submit time.
- **Language**: codes are checked against faster-whisper's `_LANGUAGE_CODES` before queueing
  (whisper only raises after the model is on the card). `auto` is an explicit value. With
  detection, the first window's answer is the document's. mlx-whisper reports no language
  probability, so `mlx_worker.detect_language` runs whisper's own detection on the first 30 s
  and passes the code on (one detection, not two); a named language reports probability 1.0.
- `serialise` forwards only text, span and words. Engine diagnostics (`avg_logprob`,
  `no_speech_prob`, tokens) are not published: a published field must stay published.
- The asr worker needs `workers.worker_environment`: ctranslate2 resolves cuBLAS at the first
  matmul, so without pip's CUDA libraries on the loader path the model loads and the first
  window fails with "libcublas.so.12 is not found".
- `READY_SILENCE_TIMEOUT_SECONDS = 900`: a watchdog on a worker that says nothing at all (covers a
  cold 3 GB load), reset by every message. Not a run deadline.

## Qwen3-ASR runs (`jobs/asr/qwen.py`, `qwen_worker.py`, `loopguard.py`)

Two per-job sessions: the ASR worker (`qwen_worker.py`) and the aligner (the `align` worker,
only with `word_timestamps`), both started by the job and stopped in its `finally`. Not
resident because the job needs both models on the card at once and `Residency` holds one
thing; see `docs/history/PHASE25-QWEN-ASR.md` section 3. vLLM runs in-process (not `vllm serve`) because the HTTP door
needs vLLM's `[audio]` extras the llm env lacks, and with `VLLM_ENABLE_V1_MULTIPROCESSING=0`
so the engine core is not a grandchild holding the card.

### Pieces and overlap

- Defaults `DEFAULT_PIECE_S = 30`, `DEFAULT_OVERLAP_S = 0.4` with word timestamps, 0
  without. Caller-settable (Owen, 2026-09-26): `piece_s` 5 to 180
  (`QWEN_PIECE_MAX_SECONDS`), `overlap_s` 0 to 5 and under half a piece.
- **Why 30 s, not 180 s**: a cut at a pause lands a piece's first word at sample zero and the
  model skips it (327 of 6,532 cues on *The Coming of the Third Reich* lost their opening
  word; 1.5 s of lead-in fixed it). antirez's port measured 120 s pieces repeating ~20% and
  180 s pieces looping. 0.4 s is faster-whisper's `speech_pad_ms`. 180 s is the hard ceiling
  because `qwen_asr`'s aligner is not trusted past it (`MAX_FORCE_ALIGN_INPUT_SECONDS`), and
  every budget (`max_model_len`, token budget) was computed from it. An uncut 600 s decode
  found zero fillers.
- **Cut points** (`qwen_worker.split_points`, ported from `qwen_asr`'s
  `split_audio_into_chunks`): search the last 10 s before each nominal boundary (or half a
  piece) for the quietest 100 ms window and cut at its **centre**. The search is on the near
  side only, so no piece exceeds `max_piece_s` (`qwen_asr` searches 5 s either side and
  overshoots). With `speech_only`, the latest join in the span wins over the quietest window.
- Each wav holds the core plus `overlap_s` of real neighbouring audio each side. Pieces under
  `MIN_PIECE_SECONDS` are zero-padded at the tail; reported durations stay real.
- **Ownership** (`qwen.own_words`): a word is kept by the piece whose core `[start, end)`
  holds its midpoint; the last core is closed at its end. Cores tile the source, so every
  word is kept exactly once. That needs word times, hence overlap requires
  `word_timestamps`. The piece's text is sliced to its kept words by matching aligner items
  on **letters and digits only** (`_word_spans`): the aligner returns its own normalisation
  ("life-changing" -> `lifechanging`), and verbatim matching failed whole jobs on the first
  hyphen. An item that cannot be found fails the job by name.
- Every Qwen job is always told the language, which must be one of the aligner's eleven
  (`auto` refused): auto-detection costs the 1.7B time, and the aligner cannot detect.

### Context

`context` is the system-turn instruction and vocabulary (e.g. ContentStudio's filler prompt,
which took a 10-minute window from 9 fillers to 19). Not `initial_prompt` under another name.
Up to 1,024 of the model's tokens, counted by the worker after load; the server first refuses
more than 8,192 characters. Chat control tokens (`<|im_end|>` etc.) are refused rather than
stripped, as vLLM's own transcription door silently does. On vLLM the worker renders the
repo's own `chat_template.json` (always a system turn) plus `language {Name}<asr_text>`, as
`qwen_asr` 0.0.6 does. mlx-audio's prompt differs by one `\n` after the context; measured as
no difference (8 vs 9 fillers). Neither engine applies `qwen_asr`'s
`detect_and_fix_repetitions`: loops are the server's to detect, not the worker's to tidy.

### Loop guard, token limit and re-decode

See `docs/history/PHASE25-QWEN-ASR.md` section 5 for the full rationale. The failure: a 180 s piece came back as one
line repeated ~60 times until `max_new_tokens`, and the aligner stamped all 2,388 words at one
instant, silently erasing three minutes.

Signals (`loopguard.text_signal`, `alignment_signal`):
1. `hit_token_limit`: the decode used its whole budget. `token_budget` is 4096 tokens per
   180 s (~22.8/s, ContentStudio's setting) scaled to the piece and floored.
2. More than 8 words/s (`WORDS_PER_SECOND_CEILING`) past `RATE_MIN_WORDS`. The observed
   loop was 13.3/s; brisk speech is ~3.
3. A 1-to-N-word phrase repeated back-to-back at least 8 times (`REPEAT_MIN_COUNT`).
4. Aligner collapse: 12 consecutive zero-length items (`COLLAPSE_RUN_ITEMS`); the aligner's
   resolution is 80 ms, spans are rounded to 1 ms.

Thresholds are decisions from one observed loop and speech rates, not a corpus. Rules 1-3
count whitespace words, so they under-fire on Chinese/Japanese/Cantonese; rule 4 still
catches those.

A flagged piece is re-cut **from the original source** (so its overlap is real neighbouring
audio, not the looping piece's) at the next rung of `window_ladder`: the piece length, then
halves (30 -> 15 -> 7.5 at the default). Greedy decoding (temperature 0) is deterministic,
so only different input changes the outcome. A piece that loops at the last rung fails the
job as `asr_decode_loop` with its time range. A repetition penalty was rejected: it pushes
against the fillers and verbatim repeats the model was chosen to keep.

An empty decode (or punctuation only) is silence, not a hole. A piece whose every word was
in its overlap is also silence.

### Aligner batching and `transcript.text.json`

- `_publish_text` writes `transcript.text.json` **before alignment** (Owen, 2026-09-27): an
  aligner failure must cost only the alignment (job 928bdf54 lost two hours of
  transcription). Rows are **pieces**, each with its owned span (`start`/`end`) and the heard
  span (`audio_start`/`audio_end`); the text covers the heard span, since ownership needs
  word times. A client can send each row to `align` as-is. `transcript.json` remains the
  finished artifact.
- It is published again before every later round of alignment, so a stretch that was
  re-decoded is covered by its re-decoded pieces and never by the looping piece they replace
  (`_text_pieces`, keyed by `piece_key`; a re-decode drops its piece). Each publish replaces
  the artifact atomically (`atomicjson.write_json` into the artifacts directory), so a client
  reading it mid-job gets the old document or the new one, never half of either.
- `_align_pieces` sends `ALIGN_BATCH = 16` pieces per request (8 minutes of audio at 30 s)
  with a progress event after each. One request for 3,015 pieces left the stream silent for
  over ten minutes and the client's went-quiet guard cancelled the job. Batches also let a
  cancel land between them.
- A fresh run loads the aligner up front, so a broken aligner is found before hours of
  decoding; a resumed run loads it only when a piece still needs word times.

### Journal and resume

Units: `plan.L<level>.<region>`, `text.<piece>`, `words.<piece>` (per aligner batch),
`verdict.<piece>`. `piece_key` is level plus core in integer samples: a re-cut of a piece
already shorter than the next window yields the same core, and without the level the child
would read its looping parent's text. A resume recomputes the plan (deterministic from the
ffmpeg samples and params) and refuses `resume_plan_mismatch` if it differs; recomputed
verdicts must match (`resume_mismatch`). A piece the aligner failed is not journaled, so a
resume re-aligns only failures. The identity (`journal_identity`) holds every param resolved
to its effective value, the engine and dtype, and the aligner's id and revision, but not
`resume` or the serving width. Bump `JOURNAL_FORMAT_VERSION` whenever a unit's meaning, the
piece key, the plan shape, ownership, the loop guard or the worker's output for the same
audio changes. Everything else is in RESUMABLE-JOBS.md.

### Width ladder (`asrplan.py`)

Owen, 2026-09-26: *"yes, fewer at once before quantizing for asr too."* On a card that cannot
hold `max_batch` pieces at full precision, the engine starts at the widest narrower width that
fits, down to one, before the capability walk moves to a smaller model (never under 4 bits;
no quantized Qwen3-ASR is pinned). Only vLLM has a width to narrow.

    need(width)    = memory_bytes_estimate - kv_cache_memory_bytes + kv_pool(width)
    kv_pool(width) = max(width x kv_cache_memory_bytes / max_batch,
                         max_model_len x kv_bytes_per_token)

The weights, overhead and 1 GiB audio-tower reserve are held fixed (upper bound). The floor is
vLLM's refusal to start with a pool smaller than one `max_model_len` sequence
(8,192 x 114,688 = 896 MiB, two pieces' worth, so width 1 needs what width 2 does). A
word-timestamped job counts its aligner too, so it may narrow further than the capability
verdict (which is about the transcriber alone). Refusals use `qwen.floor_bytes`, so "never on this
host" is only said below one piece. `gpu_memory_utilization` is only vLLM's startup gate here,
set to the estimate over the card rounded up.

## Speech only (`speechonly.py`)

Owen, 2026-09-27: *"Sending silences through an asr model produces hallucination and
nonsense."* Off by default until measured against the Deathstalker book's 133 known dropped
openings. Not combinable with `vad_filter: true`. Full contract: `docs/history/PHASE25-QWEN-ASR.md` section 11.

- **Detector**: Silero VAD 6.2.1, run as a numpy port of `silero_vad/tinygrad_model.py` on the
  CPU, over weights read directly from the protobuf of `silero_vad_op18_ifless.onnx`
  (sha256-checked; `hosttools.SILERO_VAD`). numpy is the only runtime in all five worker envs;
  adding onnxruntime or torch to a recipe would mark every installed env stale. Checked
  against onnxruntime's default model: at most 8.6e-6 apart. The wheel's
  `silero_vad_16k.safetensors` holds OLDER weights (up to 0.42 apart); do not pin it.
- **Generous by design**: `threshold` 0.3 (Silero's is 0.5), no minimum speech length, `pad_s`
  0.3 s each side, and only gaps of at least `min_gap_s` (2 s) after padding are removed, at
  the file's ends too. Caller bounds: `speech_threshold` 0.1-0.7 (below 0.1 nothing is
  removed, above 0.7 quiet speech is lost), `speech_pad_s` 0.1-2 (tighter is the
  dropped-opening bug again), `speech_min_gap_s` 1-60.
- A source with no speech detected fails the job with a sentence, not an empty transcript.
- `detect` compacts the decoded buffer in place (an 18-hour book is 4 GB of float32); the
  caller must not read `wav` afterwards.
- **Timeline exactness.** The worker cuts and transcribes the shortened signal and reports
  `kept` (integer source-sample spans) and `samples`. `Timeline` maps back sample-exactly.
  At a join a start belongs to the region after and an end to the region before, so a span
  ending at a join does not reach across removed audio. Segment ends map independently (a
  segment may span a removed stretch, which is true). A word maps both ends into the region
  holding its midpoint and is clamped there, never stretched over removed audio.
- Whisper: windows are cut from the shortened signal, the window shift and dedup run on that
  timeline, then times move to the source. Qwen: pieces, ownership, the loop guard and every
  re-cut all run on the shortened timeline, and only `_document` moves times to the source.
  The worker's split cache is keyed by source and speech settings so re-cuts read the same
  signal.
- `speech`/`removed` are null when off; `removed: []` means the detector ran and removed
  nothing.

## Align (`jobs/align/`)

- The aligner is the first worker that outlives a job: a `WorkerSession` held in `Residency`
  as its own kind (not an engine with a `base_url`). Loaded implicitly by the first `align`
  job (a 20 s load always followed by its work); unloaded only by `unload-aligner` or by
  loading something else. `unload-aligner` checks the resident is an aligner before
  unloading, since the holder unloads by id alone.
- `QWEN3_MAX_AUDIO_S = 300` is a **refusal, not a split**: the model card supports timestamps
  "within up to 5 minutes", and splitting would move timestamps silently.
- Eleven languages (`QWEN3_LANGUAGES`, ISO code -> the English name `model.align` takes), checked
  before queueing: the model places words badly in a language it was not trained on.
- No retries and no second backend (Owen, 2026-09-05).
- Results are the model's own tokens (665 items for a 668-word window), not the caller's
  words. Word mapping, the letter-sequence check and scoring stay in the client.
- A `cue` event goes out as each chunk lands (since 2026-09-25), so a killed run costs only
  the chunks it had not reached.
- If the session dies or a cancel interrupts it, the resident row is dropped so `/v1/health`
  does not advertise a dead aligner. A failed `stop()` is reported as a log line and a job
  `note`, never replacing the original error (`_forget`).
- `start_aligner_session` is the one start path (resident aligner, asr word times,
  align-longform); it caps CUDA memory at the manifest's figure so the aligner can sit beside
  vLLM, and treats a load that answers with results as broken.
- The worker writes a 16 kHz PCM_16 temp wav per chunk because `model.align` takes a path; the
  Qwen ASR worker writes 16-bit wavs too, so both models hear the same samples.

## Align-longform (`jobs/alignlongform/`)

A whole audiobook (the m4b as-is) plus the book's sentences in, a VTT and a report out.
Stages: `transcribe` (faster-whisper rough pass) -> `coarse-align` (pure) -> `align` (Qwen3
aligner on the card) -> `write` (pure). `STAGES` names are matched verbatim by BookForge's
progress bars. The job charges one slot for all four, CPU stages included (Owen's TTS ruling:
*"the entire tts step goes to the other system ... even if it's cpu"*). `model` is the aligner;
the rough whisper size is the `rough_model` param.

- **Workers, not nested jobs.** A job waiting on another job would deadlock on the lane it
  holds. The rough pass uses `run_worker` (exits before the aligner starts, so the two are
  never on the card together, and `vram_estimate` is the larger, not the sum). The aligner is
  a per-job session stopped in `finally`, not the resident one: borrowing the resident holder
  would evict the client's model.
- **The rough pass**: `word_timestamps` true (coarse-align walks word positions),
  `vad_filter` false (a dropped stretch would read as skipped text), no prompt, and
  **`speech: None`**. A speech-only cut would put every time on the shortened timeline, and
  this stage's `index * WINDOW_SECONDS` shift does not apply the `kept` table; if a worker
  ever reports one the stage fails. The coarse DTW handles wordless stretches itself.
  `WINDOW_SECONDS`/`OVERLAP_SECONDS` are imported from the asr job (ints: the worker refuses
  floats). A failed window fails the stage: coarse-align would read it as skipped narration.
- **coarse.py** is a faithful port of BookForge's `align_audiobook.py:coarse_align`; keep the
  original's constants and oddities. Pass 1 anchors sentence-opening trigrams (one miss
  tolerated) and keeps the longest increasing subsequence; pass 2 walks between anchors, so
  drift is bounded by anchor spacing (the whole-book walk flatlined at 8.6% matches). `_rate`
  uses only sentence-adjacent pairs and clamps: a recap once measured 212 tok/s. Unmatched
  interior runs are interpolated over the words heard in the gap, or kept `None` when either
  the time test or the word test says the text cannot fit (unnarrated front matter once
  dragged chapter 1 85 s late); interior-trigram matches rescue misheard openings. The word
  test is skipped next to a failed transcribe slice. `direct` marks a sentence whose own
  opening was found, and it is trusted over the aligner when they disagree.
  In code: `_trigram_index`, `_anchor_candidates`, `_anchor_chain` (pass 1),
  `_walk_around_anchors` (pass 2, recording through `_Placement.place`), `_fill_interior_runs`
  (each gap a `_Run`: `_rescue_run` when it is overfull, else `_spread_run`), then
  `_never_backwards`. `_enough` is the one "got >= max(3, need - 1)" test.
- **jobtype.py**: `run` resolves ffmpeg and the weights into a `_Book`, then `_run_stages` runs
  transcribe, place sentences, plan windows, align windows and write the transcript in order;
  `StageFailed` and `NoCues` become `JobError` in `run` alone.
- **plan.py**: chunk pads `PAD_HEAD, PAD_TAIL = 4.0, 20.0` are the original's, not tunables
  (the tail covers the last sentence up to the next rough start). Spans are capped at
  `2 * chunk_s` (aligner memory is quadratic in span), and capped ranges are reported by
  audio time. `chunk_s` must fit the aligner's 300 s limit.
- **cues.py**: `snap_boundaries` (contiguous mode only) moves each shared seam to the middle of
  an overlapping silence clipped to the window, bounded by neighbours with `min_gap`. A run
  with no cues raises `NoCues`: a bare `WEBVTT` is not a transcript. A failed aligner window
  keeps its coarse timings.
- Artifacts must be registered, not just written to `ctx.scratch`, or the job reports `done`
  with nothing.
