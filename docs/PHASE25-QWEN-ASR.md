# Phase 25: Qwen3-ASR under `asr`

**Status: built, not yet run on a card. No GPU was used to write it.** Every number
in this document is sourced, and each one says whether it was computed, declared,
or measured by somebody else. Section 8 lists the live runs that would turn the
computed ones into measurements; they need Owen's go.

## 0. The ruling, and what it does and does not change

Owen, 2026-09-24: *"we're fully switching over to qwen for transcribing and
aligning. it seems flawless. 1.7b - the biggest one"*. Then: *"the 41 gb memory use
wont be a problem since we'll be using vllm or sglang instead"*. Then, on precision:
full precision (bf16, unquantised) on both machines, and not the 8-bit MLX build.

The evidence is ContentStudio's live test on the Mac Studio that day (M1 Ultra, MPS,
bf16, `qwen-asr` 0.0.6, transformers 4.57.6, torch 2.14):

- whisper-large-v3-turbo transcribed a 3.9-hour stream and found 76 fillers.
- Qwen3-ASR-1.7B with a context prompt found about 70 ums and 34 uhs.
- A 10-minute window had 9 fillers without the context and 19 with it.
- Repeats were kept verbatim.
- Owen called cuts made at the aligner's word boundaries "flawless".
- Speed was about 12x realtime alone, alignment included: 13,938 s in 1,301 s
  while the GPU was shared.

What this phase changes, and what it leaves alone:

- **It is one new model under an existing job type.** `qwen3-asr-1.7b` sits under
  `asr`: one audio file in, `transcript.json` out. There is no new verb, because
  nothing about the request is new; Crucible is a task-agnostic GPU orchestrator.
- **The whisper manifests stay.** Apps switch first, and whisper is retired in a
  later cut.
- **The aligner stays as `align` runs it.** It is the same worker and the same
  `align/qwen3-aligner.toml` (section 4).

## 1. The engines, and why these ones (research of 2026-09-24)

| backend | engine | package | pinned where |
|---|---|---|---|
| `cuda-linux` | `vllm` | vLLM 0.29.0 | `envs/llm/cuda-linux.txt` (already) |
| `mlx-darwin` | `mlx-audio` | mlx-audio 0.5.5 | `envs/llm/mlx-darwin.txt` (already, as mlx-vlm's dependency) |

**No new env on either machine.** Both engines run from the `llm` env that each
host already has.

### vLLM 0.29.0 supports Qwen3-ASR natively

At tag `v0.29.0`, `vllm/model_executor/models/registry.py` registers two relevant
models:

- **`Qwen3ASRForConditionalGeneration`** (L565). It is built from the Qwen3-Omni
  audio tower plus `Qwen3ForCausalLM` (`models/qwen3_asr.py`).
- **`Qwen3ASRForcedAlignerForTokenClassification`** (L303). This is the aligner,
  run as a pooling model (`models/qwen3_asr_forced_aligner.py`, `runner="pooling"`,
  `pooling_task="token_classify"`).

Neither needs the `qwen-asr` package. That matters, because `qwen-asr` 0.0.6's
`[vllm]` extra pins `vllm==0.14.0`, and 0.14.0 is not the llm env's vLLM.

**How the audio gets in.** Qwen3-ASR's processor is called with `truncation=False`
(`transformers_utils/processors/qwen3_asr.py` L116), so a 180 s piece is encoded
whole and not cut at the feature extractor's `chunk_length` of 30 s.

**Why no new packages are needed for audio.** The worker passes audio to vLLM as
`(samples, 16000)`:

- At a matching rate, vLLM resamples nothing (`AudioResampler.resample` returns
  its input, `multimodal/audio.py`). vLLM's `[audio]` extras (`av`, `scipy`,
  `soundfile`, `soxr`, `setup.py` L1513) are only needed to decode or resample,
  and the llm env does not carry them.
- That is also why the engine runs in-process and not as `vllm serve`
  (section 3). The HTTP transcription door decodes uploads with those extras.

### SGLang supports Qwen3-ASR too narrowly to use

SGLang added Qwen3-ASR in PR #22073 (merged 2026-04-07), first released in
**v0.5.11**. The latest release is v0.5.20 (2026-09-18), and the model file there
is `python/sglang/srt/models/qwen3_asr.py`.

At v0.5.20 its transcription door cannot serve this contract:

- **No context.** `TranscriptionRequest` has no `prompt` field, and the default
  ASR prompt has no system turn.
- **No forced language.** The processor carries `# TODO: support force_language`.
- **Output capped at 256 tokens.** `max_new_tokens: 256` is hardcoded in the
  adapter; a 180 s piece of speech is about 700 tokens.
- **No timestamps.** `segments=[]`, with a TODO that names the ForcedAligner.
- **No ForcedAligner in `srt/models/` at all.**

Chat completions with audio may carry a system turn, but that was not verified.

**SGLang is Owen's long-term preference (PHASE24), and it is not the first cut.**
When SGLang's door takes a context, a forced language and an output budget, an
`sglang` engine is a new row in `ASR_BACKEND_ENGINES` and a new branch in the
worker. Nothing else changes.

### The Mac: mlx-audio 0.5.5, reading the official checkpoint

mlx-audio first had `stt/models/qwen3_asr` and `qwen3_forced_aligner` in 0.3.1
(2026-01-29). `system_prompt` and `batch_size` arrived in 0.4.0. Version 0.5.5 is
the latest release (2026-09-21) and is already in the Mac's llm env.

**It loads the official repo directly.** `Qwen3ASRModel.sanitize` does three
things:

- strips the `thinker.` prefix;
- drops the tied `lm_head.weight`;
- transposes the conv kernels.

The pinned revision holds `thinker.audio_tower.*` (397 tensors),
`thinker.model.*` (310), and `thinker.lm_head.weight`, and nothing else. The
worker loads it with `strict=True`, so a renamed layer is a refusal and not a
freshly initialised one.

The worker also checks that every floating-point parameter loaded as bfloat16,
and refuses otherwise (section 7).

**So the Mac runs the same bytes as the PC.** Neither of these community
conversions is used:

- `mlx-community/Qwen3-ASR-1.7B-bf16` @ `e1f6c266`, 4,076,186,653 B. It is the
  same tensors less the tied `lm_head`, but a different repo, so pinning it would
  need a second id.
- `Alkd/Qwen3-ASR-1.7B-MLX-8bit` @ `b8522483`. This is byte-identical to
  mlx-community's 8-bit file (LFS oid `bf304b00…`, the same as `aufklarer/`'s).
  It lacks the preprocessor, chat-template and generation configs that mlx-audio's
  `post_load_hook` reads. It is 8-bit, and Owen ruled 8-bit out.

`moona3k/mlx-qwen3-asr` 0.4.4 is a standalone package with its own MLX aligner. It
is the credible alternative if mlx-audio's prompt handling proves limiting
(section 8, item 5). Its source has not been reviewed.

### transformers-on-CUDA was not chosen

It is what ContentStudio ran on MPS. It computes logits over the whole prompt, and
it would need a new env pinning `transformers==4.57.6`, which `qwen-asr` requires
exactly, beside the llm env's 5.17.0. vLLM needs neither.

## 2. The wire

`POST /v1/jobs` with `type: asr`, `model: qwen3-asr-1.7b` and one audio input.

| param | whisper | Qwen3-ASR |
|---|---|---|
| `language` | a code or `auto` | **required, and one of the aligner's eleven:** en de fr es it pt ru ja ko zh yue. `auto` is refused (`language_unsupported_by_engine`). |
| `vad_filter` | faster-whisper only | **must be `false`** (`vad_unsupported_by_engine`); neither engine has a VAD |
| `word_timestamps` | whisper's own | `true` runs the Qwen3 aligner; `false` returns piece-level segments only |
| `initial_prompt` | optional | **refused** (`initial_prompt_unsupported_by_engine`) |
| `context` | **refused** (`context_unsupported_by_engine`) | optional (details below) |

`context` is Qwen's system-turn instruction:

- A blank string is refused.
- Anything over 8,192 characters is refused before the job is queued.
- The chat template's own control tokens (`<|…|>`, `<asr_text>`) are refused. The
  context is placed verbatim inside the system turn, where one of them would end
  the turn. vLLM's own door strips these silently; Crucible refuses them.
- The worker counts the context's tokens with the model's tokenizer after loading.
  Past 1,024 tokens (`QWEN_CONTEXT_MAX_TOKENS`), the job fails by name.

**Why `context` is a new field and not `initial_prompt` reused.** Whisper reads
`initial_prompt` as the transcript so far: 223 tokens that scroll out after a few
segments. Qwen reads `context` as an instruction, whole, before every piece. One
field would carry two limits and two meanings under one name. Each engine refuses
the other's field by name, and the refusal tells the caller which field to send.

**Why the language is always stated.** ContentStudio measured that auto-detection
costs the 1.7B time, and the aligner has no detection at all: it takes a language
name. Qwen3-ASR transcribes thirty languages. The eleven here are the ones the
aligner places words in, so a word-timestamped transcript exists for every one of
them.

### What a client sends (ContentStudio)

```json
{
  "type": "asr",
  "model": "qwen3-asr-1.7b",
  "params": {
    "language": "en",
    "vad_filter": false,
    "word_timestamps": true,
    "context": "Verbatim transcript of a livestream. Transcribe every disfluency exactly as spoken, including filler sounds: um, uh, ah, er, hmm, and false starts and repeated words."
  },
  "inputs": {"stream.m4a": {"blob_id": "…"}}
}
```

In the SDK, that is `client.asr({model: 'qwen3-asr-1.7b', audio, filename:
'stream.m4a', language: 'en', vadFilter: false, wordTimestamps: true, context})`.

### What comes back: `transcript.json`

The fields a whisper consumer reads have the same keys and types: `model`,
`revision`, `hf_repo`, `language`, `language_probability`, `language_requested`,
`vad_filter`, `word_timestamps`, `initial_prompt`, `duration_s`, and `segments`
with `start`, `end`, `text` and `words`.

A word is whisper's four keys: `{start, end, word, probability}`. `probability` is
always `null`, because the aligner places words and does not score them. Times are
absolute seconds into the input.

The document differs from a whisper transcript in these places:

- **A segment is one piece** (section 3), up to 180 s long. It is not a
  whisper-sized sentence, and Crucible does not invent sentence units.
- **A word is the aligner's own item.** It has no punctuation; the segment's `text`
  has it.
- **There are no `window_s`, `overlap_s` or `windows` fields.** Pieces are cut at
  quiet points with no overlap, so there is nothing to deduplicate.
- **These fields are added:**
  - `engine`
  - `dtype`
  - `aligner` (`{model, revision, hf_repo}`, or null when `word_timestamps` is
    false)
  - `context`
  - `piece_max_s` (180)
  - `pieces`
  - `silent_pieces`, the pieces the model heard nothing in. A silent piece gets no
    segment, which is how whisper treats silence too.
  - `redecoded`, one row per piece the loop guard re-cut (section 5)
- **`language_probability` is always `1.0`.** It is the caller's assertion and not
  a detection, which is the mlx-whisper worker's rule for a named language.

Progress events carry `{stage, processed_s, total_s, cues}`, as the whisper engines'
do. The stages are `decoding`, `transcribing` and `aligning`. `note` events announce
each re-decode.

## 3. How a job runs, and why it is per-job and not resident

`crucible/jobs/asr/qwen.py` holds **two sessions for the life of one job**:

1. **ASR.** The session is `qwen_worker.py` in the llm env. It decodes the input
   once with ffmpeg to 16 kHz mono and cuts it into pieces (the cut is described
   below). It writes each piece as a 16-bit wav, and then decodes pieces in
   batches of the manifest's `max_batch`.
2. **Aligner** (only when `word_timestamps` is true). The session is the `align`
   job type's own worker in the align env, loading `qwen3-aligner` exactly as
   `align` does.

**How the input is cut.** Pieces are at most 180 s long: `qwen_asr`'s
`MAX_FORCE_ALIGN_INPUT_SECONDS`, the length the official SDK cuts to when
timestamps are on. Each cut is made at the quietest 100 ms in the 10 s before the
nominal boundary. The official SDK searches 5 s either side, so its pieces run up
to 5 s over the limit; this search never goes past it.

Each round decodes the pending pieces, reads each for a loop, aligns the clean
ones, and reads each alignment for a collapse. Anything flagged is re-cut and goes
into the next round (section 5). Both sessions are stopped in the job's `finally`.

**Per-job, not resident.** The job owns every hold: it started them and it stops
them, so there is no orphan for a reconciler to find. There are three reasons:

- **A word-timestamped job needs two models live at once.** A collapse found by the
  aligner sends the piece back to the ASR model. `Residency` holds ONE thing, so a
  resident ASR engine would be evicted by the aligner its own job loads.
- **The unit of work is hours of audio.** A cold vLLM start is a minute or two
  (weights, then a CUDA graph per batch size up to 8) against the twenty-odd
  minutes of a 3.9-hour stream.
- **It never evicts anybody.** As with every `asr` job, the guard refuses it by
  name, before it is queued, when the card will not hold it beside what is already
  resident.

**Why vLLM runs in-process, not as `vllm serve`.** There are two reasons:

- The job's two-model lifecycle above.
- The audio extras described in section 1.

`VLLM_ENABLE_V1_MULTIPROCESSING=0` (`vllm/envs.py` L1386) keeps the engine core in
the worker's own process. The one process holding CUDA is then the one that is
asked to exit.

The worker's environment is otherwise the resident engine's own, from
`crucible/engines/vllm.py` `ENVIRONMENT`. That has one owner and covers the
WSL2 pinned-memory setting and the FlashInfer sampler.

## 4. The aligner

The aligner is `align/qwen3-aligner.toml`, unchanged. It is named by the ASR
manifest's `aligner` key, and its memory is that manifest's figure: the job's guard
adds the aligner's figure to the ASR estimate. The aligner's figure is never copied
into the ASR manifest.

vLLM 0.29.0 can serve this aligner as a pooling model. The pre- and
post-processing that turns classifier logits into timestamps lives in `qwen_asr`'s
`Qwen3ForcedAligner`, though: the text tokenization with `nagisa`/`soynlp` for
ja/ko, and the monotonic fix. Moving the aligner into vLLM would mean porting that
code. It is not done here.

## 5. The repetition-loop guard (`crucible/jobs/asr/loopguard.py`)

**The failure.** Once in 78 pieces on ContentStudio's run, a 180 s piece repeated
one line about 60 times, 2,388 words in all, until it hit `max_new_tokens`. The
aligner then stamped every word at one instant, and the real speech was lost with
nothing to say so.

**The four signals.** Each threshold is a DECISION, set from that one loop and
from speech rates:

| signal | fires when | why |
|---|---|---|
| `token_limit` | the decode hit its budget: `min(max_new_tokens, max(256, ceil(duration × 4096/180)))` | speech ends on its own; 4096/180 s is ContentStudio's own setting |
| `words_per_second` | more than 8.0 words per second of audio, counted only past 24 words | brisk speech is about 3 a second; the loop was 13.3 |
| `repeated_phrase` | a 4–40 word phrase repeated back to back 8 or more times | verbatim repeats are 1–2 words a few times; the loop was about 60 |
| `aligner_collapse` | 12 or more consecutive zero-length aligner items | the aligner's step is 80 ms, so a real word spans at least one step |

**The budget.** A flagged piece is re-cut at quiet points to at most **60 s**, and
decoded again. If one of those still loops, it is re-cut to at most **20 s**.
Greedy decoding is deterministic, so decoding the same input again reproduces the
loop; a smaller window changes the input.

If a 20 s piece still loops, the job fails as **`asr_decode_loop`**. The error
names the piece's range (for example `200.0-220.0s (0:03:20.0-0:03:40.0)`), the
ladder that was tried, and the signal. No artifact is published.

Each re-decode is a `note` event on the job's stream and a row in `redecoded`.

**Repetition penalties are refused.** vLLM's and mlx-audio's penalties scale down
every token already in the prompt or output. The prompt is the context, which is
where "um, uh, ah, er, hmm" live. A penalty would work against the very fillers
this model was chosen to keep. vLLM 0.29.0's `SamplingParams` has no
`no_repeat_ngram_size`.

Neither engine applies `qwen_asr`'s `detect_and_fix_repetitions`. That function
silently collapses repeats inside the official SDK's `parse_asr_output`.

**A known limit.** The text rules count whitespace-separated words, so they
under-fire on zh, ja and yue. The `token_limit` and `aligner_collapse` rules still
hold in those languages.

## 6. Envs

**Nothing is installed or changed.**

- `envs/llm/cuda-linux.txt` already pins `vllm==0.29.0`, `transformers==5.17.0`
  and `numpy==2.3.5`.
- `envs/llm/mlx-darwin.txt` already pins `mlx-audio==0.5.5` (declared
  `mlx>=0.31.1`, `transformers>=5.14.0`), `mlx==0.32.2` and `transformers==5.17.0`.
- The aligner's `envs/align/*.txt` is unchanged: `qwen-asr==0.0.6`, `torch==2.14.0`
  and `transformers==4.57.6`. These are ContentStudio's versions exactly.

A host that transcribes with Qwen needs these, and `crucible doctor` and the `asr`
type's `check` report each env separately:

- the **llm** env;
- the **align** env, for word timestamps;
- the weights `qwen3-asr-1.7b` and `qwen3-aligner`, via `crucible models pull`.

## 7. Memory

`asr/qwen3-asr-1.7b.toml` states the terms of each figure; the totals are:

| | ASR engine | + aligner | job need |
|---|---|---|---|
| cuda-linux | 11,006,754,728 (computed) | 3,446,157,280 (computed) | 14,452,912,008 = 13.46 GiB |
| mlx-darwin | 7,726,809,000 (computed) | 5,885,296,640 (measured 2026-09-14) | 13,612,105,640 |

**cuda-linux**, 11,006,754,728 in total:

| term | bytes | basis |
|---|---|---|
| weights | 4,698,521,512 | the file sizes; an upper bound, because the 622 MB tied `lm_head` is not loaded |
| non-KV overhead | 1,476,395,008 | the 9B's measured figure, carried as an upper bound |
| audio-tower activations | 1,073,741,824 | declared |
| KV pool | 3,758,096,384 | computed: 114,688 B/token × 4,096 × 8 |

The KV pool is passed to vLLM as `kv_cache_memory_bytes`, so vLLM does not claim
its default 92% of the card. `gpu_memory_utilization` is the estimate over the card,
rounded up: 0.43 on the 3090 Ti. With the pool size stated, vLLM reads it only as
the start gate, "free memory must be at least this".

**mlx-darwin**, 7,726,809,000 in total:

| term | bytes | basis |
|---|---|---|
| weights | 4,698,521,512 | the file sizes, an upper bound |
| KV for one worst-case piece | 880,803,840 | computed: 7,680 tokens × 114,688 B |
| activations and MLX buffer cache | 2,147,483,648 | declared at 2 GiB |

The worker calls `mx.clear_cache()` between pieces.

**ContentStudio's 41 GB is not this budget.** That figure was transformers on MPS:
full-prompt logits for batch 4 (151,936 × 2,460 × 4 in bf16), plus torch's MPS
caching allocator, which never returns memory. Neither exists on either engine here.

**The batch.** The batch is set in the manifest and is never the library's value:

- On cuda-linux it is 8 (`max_num_seqs`).
- On mlx-darwin it is 1. mlx-audio batches only the chunks it cut out of one input,
  and Crucible gives it one piece per call so that the guard sees each piece's own
  token count.
- The loader refuses any other value on the Mac.
- `qwen_asr`'s own default of 32 is what aborted the MPS process on
  ContentStudio's run.

## 8. What must be measured on the first live run (Owen's go, per machine)

**PC (WSL2, 3090 Ti).** The card must be free of a resident 9B. Install nothing new.

1. `crucible models pull qwen3-asr-1.7b`, and `crucible models pull qwen3-aligner`
   if it is not already pulled.
2. Run one asr job on a 10-minute clip with a known filler count and
   `word_timestamps: true`. Record:
   - that vLLM starts with `kv_cache_memory_bytes` and CUDA graphs captured;
   - the peak from `nvidia-smi --query-gpu=memory.used` sampled at 1 Hz, which
     replaces the computed 11.0 GB;
   - the realtime factor.
3. Run the same job on ContentStudio's 3.9-hour stream. Record:
   - the realtime factor;
   - the count of each loop signal;
   - the `redecoded` rows, if any;
   - the ums and uhs, compared with ContentStudio's 70 and 34.
4. Check that the job's `finally` leaves the card empty: `nvidia-smi` shows no
   process after `done`.

**Mac Studio.** Nothing is resident.

5. The same two jobs. Record the peak `mx.get_peak_memory()` or the process
   footprint, which replaces the computed 7.7 GB.
   - **Compare fillers against the PC on the same clip.** mlx-audio's prompt writes
     `{context}\n` before `<|im_end|>`. That is one character the official template
     does not have, and whether it moves filler recall is unmeasured.
6. Check that the worker's bf16 assertion passed. It refuses by name otherwise.

**Both.**

7. Count zero-length aligner items per piece on clean speech, to set
   `COLLAPSE_RUN_ITEMS` against the real baseline rather than a guess.

## 9. What would change the shape

A streaming-transcription door, or many short jobs, would make residency worth
having. That needs `Residency` to hold an ASR engine and its aligner as one resident
thing, or the aligner moved into vLLM as a pooling model (section 4).

SGLang, when its door carries a context and a forced language (section 1), is one
row and one worker branch.
