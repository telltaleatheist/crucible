# Client surfaces — every place BookForge and Foundry talk to a model

A read-only audit, 2026-09-12, taken so phases 3 and 4 are planned against what the two
apps actually do rather than what they are remembered to do. Nothing in either app was
changed; every claim below was read out of source at these commits:

| Repo | Path | HEAD at audit |
|---|---|---|
| Crucible | `C:\Users\tellt\Projects\crucible` | `6cb9b97` |
| BookForge | `C:\Users\tellt\Projects\bookforge` | `d6e63634` |
| Foundry | `C:\Users\tellt\Projects\foundry` | `3b13392` |
| Foundry, vendored into BookForge | `bookforge/foundry-app/` | `3b13392` — byte-identical on the model seam (verified file by file, CRLF aside) |

What Crucible offers today, for reference: `llm` behind an OpenAI-compatible proxy
(`POST /v1/openai/chat/completions`, verbatim in both directions except `model`, with
`chat_template_kwargs` forwarded), `load-model` / `unload-model` jobs, one resident model
at a time, an accelerator guard, and `echo`. `vlm-pages`, `tts`, `align` and `rvc` are
named but unbuilt.

> **Added 2026-09-23 (PHASE22, `PHASE22-DECIDE.md`):** a second model door beside the chat
> proxy, **`POST /v1/decide`** — a state and questions with fixed answer sets in, a probability
> distribution per question out, from one forward pass at the resident model. The SDK's
> `decide()` and `crucible api decide` are its client surfaces. Its first consumers are
> Foundry's Categorize tile (which ran snap beside Crucible until then) and BookForge's block
> kinds, transcript boundaries and TTS-cleanup gating; the closed-question door of section 6
> (`askConstrained`) is the same shape of question and a candidate, not a ruling.

Read the last section first if you only read one. Everything above it is the evidence.

---

## 0. The two machines, as they are actually configured

Read from the live settings files, not from source defaults.

| Setting | owens-pc (Windows) | owens-mac-studio |
|---|---|---|
| `app-settings.json` `cleanTextModel` | `qwen3.5:9b-bf16` | `qwen3.5:9b-mlx-bf16` |
| `llmServer` | **`vllm`** | *(absent → `ollama`)* |
| `vllmUrl` | `http://localhost:8300/v1` | *(absent)* |
| `vllmModel` | `""` — meaningful: "whatever it is serving" | *(absent)* |
| `keepServerWarmMinutes` | *(absent → 0, stop on drain)* | *(absent → 0)* |
| `defaultLlmModel` | *(absent → `qwen3.8:27b`)* | *(absent → `qwen3.8:27b`)* |
| `tts-engine.json` | `{engine: orpheus, voices: {orpheus: tara}}` | `{engine: higgs, voices: {orpheus: deathstalker, higgs: deathstalker}}` |
| `tool-paths.json` | `useWsl2ForOrpheus: true`, `useWsl2ForHiggs: true`, `wslHiggsCondaEnv: higgs3`, `useWsl2ForVlm: true`, `wslVlmCondaEnv: dots`, `wslVlmModel: rednote-hilab/dots.ocr`, `qwenAlignEnv: qwen-align`, `wslDistro: Ubuntu`, an HF token | only `ffmpegPath` |
| `ttsNumberNormalizerModel` | **not set** → `qwen3.5:9b-q8_0` | not set |
| `crucible-servers.json` | `{servers: []}` — empty | *(not present)* |
| Renderer `localStorage['bookforge-settings']` | `aiConfig.provider: ollama`, `aiConfig.ollama.model: cogito:14b`, `pipelineDefaults.*Model: cogito:8b`-class, a stored Anthropic API key | not read |

Live at audit time:

- **PC Ollama** serves 9 tags (`qwen3.5:9b-bf16`, `9b-q8_0`, `9b`, `qwen3.8:27b-24g`, `27b`,
  `cogito:8b/14b/32b`, `qwen3:32b`).
- **Mac Ollama** serves 31 tags, including `qwen3.5:9b-mlx-bf16`, the `headline-14b-*`
  family, `blocks-v5-06b-*`, and **eleven `foundry-footnotes-*` / `foundry-ocr-*` LoRA
  adapter builds**. Those are Owen's own fine-tunes and exist only as Ollama models.
- **PC WSL** (`Ubuntu`) has conda envs `crucible`, `dots`, `higgs3`, `sglomni`,
  `orpheus_tts`, `qwen-align`, `whisperx-cuda`, `separator`, `headline27b`; `~/models/`
  holds `Qwen3.5-9B` and `Qwen3.8-27B-AWQ-INT4`.
- **PC port 8200** was answering with `higgs-v3-ds` from `sgl-omni` (a training job held
  `external-gpu-job.lock`). Port 8300 was down.

Two facts follow immediately. First, **the PC is already pointed at a vLLM on 8300 that
BookForge itself starts** — the machine is one settings key away from the shape Crucible
wants. Second, **the Mac's model zoo is not reproducible from a manifest**: a Crucible that
cannot serve an Ollama-built LoRA adapter cannot take Foundry's footnote and OCR passes.

---

## 1. BookForge — the LLM / text call sites

All of these are in the Electron main process. Four transport primitives carry almost
everything; each feature below says which it uses.

| Primitive | file:line | Wire |
|---|---|---|
| `cleanChunk` (Ollama) | `electron/ai-bridge.ts:2803`, fetch at `:2846` | `POST http://localhost:11434/api/generate` |
| `cleanChunkWithClaude` | `electron/ai-bridge.ts:2441`, fetch at `:2459` | `POST https://api.anthropic.com/v1/messages` |
| `cleanChunkWithOpenAI` | `electron/ai-bridge.ts:2533`, fetch at `:2551` | `POST https://api.openai.com/v1/chat/completions` |
| `cleanChunkWithLocal` | `electron/ai-bridge.ts:2605` → `llamaBridge.generate` | `POST http://127.0.0.1:8769/v1/chat/completions` |

Two dispatchers sit above them: `cleanChunkWithProvider` (`:2626`, full-rewrite, with the
safeguard cascade) and `callProviderExtracted` (`:3085`, thin, answer-extracted).

### 1.1 AI cleanup — the edit-list pass (the primary cleanup product)

1. `cleanChunkEditList`, `electron/ai-bridge.ts:3308`; model call at `:3343`; driven from
   `cleanupEpub` via `processOneChunk` at `:5003`.
2. Ollama `POST /api/generate`. The base URL is the module constant
   `OLLAMA_BASE_URL = 'http://localhost:11434'` (`:292`) — **`config.ollama.baseUrl` is
   carried and deliberately ignored on this path** (stated at `:3074`).
3. Model from `AIProviderConfig.ollama.model` (`:164`), which the renderer fills from
   `localStorage['bookforge-settings'].aiConfig` — on this PC, `cogito:14b`.
   `DEFAULT_MODEL = 'cogito:14b'` (`:293`) is used only for context sizing.
4. `{model, prompt: <chunk>, system: <editlist prompt + few-shot + custom instructions>,
   stream: true, think?: false, options: {temperature, num_predict, num_ctx},
   keep_alive: '5m'}` (`:2850-2866`). `temperature` default **0.1**;
   `num_predict = EDITLIST_NUM_PREDICT = 6144` (`:2985`); `num_ctx` pinned **once per job**
   (`:4952`) by `estimateNumCtxForBudget`, bucketed to 4096 and capped by
   `numCtxMaxForModel` (16384 for ≤15B, else 12288, `:146`). `think: false` only when
   `/api/show` reports the capability.
   **Not sent:** `top_p`, `top_k`, `min_p`, `repeat_penalty`, `stop`, `seed`, `format`/JSON
   mode, grammar, schema. JSON is requested in prose.
5. Streamed NDJSON; `extractAnswer` (`:2941`) pulls `<answer>…</answer>` and throws
   `REASONING_OVERRUN` on a second block, an unclosed tag, or a surviving `<think`. Then
   `firstJsonObject` → `applyEditList`. A parse failure keeps the original and records
   `edit-parse-fail` with **no retry**; only transport errors retry (3 attempts,
   `attempt × 2000 ms`). No absolute timeout — an **inactivity** timer,
   `OLLAMA_INACTIVITY_TIMEOUT_MS = 300000` (`:369`), reset on every chunk read. Job
   `AbortController` chained per request (`:2823`).
6. Sequential for Ollama and `local` (`useParallel` forced off, `:5021`). `keep_alive: '5m'`
   keeps the model hot between chunks; `releaseCleanupModel` (`:3943`) issues an Ollama
   unload at job end. No GPU lock on the Ollama path (the arbiter cannot see an external
   process, `gpu-arbiter.ts:315`); the `local` provider does take one.
7. Prose-only chunker `chunkChapterProse` (`:694`), default **2000 chars** for cleanup and
   4000 for simplify (`:4012`). One chunk per request.

**For Crucible:** the body is `/api/generate`-shaped, not chat-shaped, and the only knobs
that cross are `temperature`, `num_predict`, `num_ctx`, `think`, `keep_alive`. Mapped to
OpenAI that is `temperature` + `max_tokens` + `chat_template_kwargs.enable_thinking` —
**the `llm` proxy already suffices**, once the prompt/system pair is expressed as two
messages and `num_ctx` is dropped (the server owns the context window). `keep_alive`
becomes nothing to do: the server owns residency.

### 1.2 AI cleanup — the legacy full-rewrite path

`cleanChunkWithProvider`, `ai-bridge.ts:2626`, dispatch at `:2656`. Same four backends.
System prompt from `getOcrCleanupSystemPrompt` (`:1380`) → `prompts/tts-cleanup.txt` or
`tts-cleanup-neutral.txt`.

- Ollama: as 1.1 but `num_predict = text.length * 2` (`:2858`).
- Claude: `{model, max_tokens, system, messages:[user]}` (`:2466`) — **no temperature, no
  top_p, no stop, no stream**.
- OpenAI: `{model, max_tokens, temperature: 0.1, messages}` (`:2557`).
- Local: `llamaBridge.generate` → `{model, messages, max_tokens (4096, doubling to 16384 on
  an empty answer with `finish_reason: 'length'`), temperature: 0.1, stream: false,
  chat_template_kwargs: {enable_thinking: false}}` (`llama-bridge.ts:706`).

Response handling is the richest in either app: `applyOutputSafeguards` (`:1009`) —
skip-marker and conversational detection (`SKIP_MARKERS` `:509`, `AI_ASSISTANT_PATTERNS`
`:512`); **output under 70 % of input length** → copyright-refusal keyword check → one
reminder retry → split in half → keep the original; a repetition guard (`detectRepetition`,
`:413`) retries once then falls back; HTTP 400 / "context" / "too long" → split and recurse
(`:2743`). A job aborts at `MAX_FALLBACK_COUNT = 10` (`:370`).
Cloud providers may run parallel: `workerCount = min(parallelWorkers ?? 3, totalChunks)`
(`:5021`).

### 1.3 Simplify — block groups

`makeSimplifyBlockCall`, `ai-bridge.ts:3431`, call at `:3444`. Prompt
`prompts/simplify-{dejargon,destiffen,learner}.txt` prefixed with
`THINKING_TRIGGER = 'Enable deep thinking subroutine.'` (`:1240`).
`num_predict = max(4096, payload.length * 2)` (`:1879`). The answer must be `<answer>`
wrapping exactly N `<block id="N">` tags; `parseBlockAnswer` (`:1766`) **throws
`MALFORMED_BLOCK_ANSWER`** otherwise and the group degrades to one call per block — content
failures are never re-rolled. Groups: ≤ 8 consecutive non-heading blocks, ≤ 4000 chars,
sent only at ≥ 120 chars (`:1621`). Sequential only.

### 1.4 Hyphen line-break arbitration

`planHyphenJoins`, `ai-bridge.ts:3254`, call at `:3275`. Live, invoked from `cleanupEpub`
at `:4500`. **The pairs live in the system prompt; the user turn is the literal string
`'Adjudicate every item above.'`** `temperature` hardcoded 0.3, `num_predict =
max(4096, batch.length * 80)`. 100 pairs per batch. A failed batch is skipped whole — no
retry, and the pairs stay unresolved, which is the conservative direction.

### 1.5 TTS number normalization

`createOllamaNormalizerRunner`, `electron/tts-number-normalizer-runner.ts:57`, call at
`:77` → `generateEditListWithOllama` (`ai-bridge.ts:3065`). **Ollama only, by design**
(`:3057`). Model key `ttsNumberNormalizerModel` in `tool-paths.json`
(`electron/tool-paths.ts:253`) — **unset on this PC**, so the default
`DEFAULT_NORMALIZER_MODEL = 'qwen3.5:9b-q8_0'` (`tts-number-normalizer.ts:144`) is what
runs. System prompt `prompts/tts-number-normalize.txt`; user turn is
`PREVIOUS / TARGET / NEXT`. `temperature: 0`, `num_predict: 2048`, `num_ctx` pinned once.
Two retries only — one transport, one parse at identical settings. Over **10 % parse
failures aborts the pass** (`MAX_PARSE_FAIL_SHARE = 0.10`, `:158`). One request per block,
never batched; `runner.release()` in a `finally` issues an explicit Ollama unload, because
the GPU is about to go to the TTS engine.

> The EPUB book path (`normalizeNarrationNumbers`, `:2254`) has **no production caller
> left** — `prepareNarrationInput` (`parallel-tts-bridge.ts:7769`) reads the OPF stamp and
> makes no model call for `.epub`; only `.txt` reaches the model.

### 1.6 Translate — the ledger pass

`translateParagraphBatch`, `electron/mono-translation-job.ts:375`, call at `:412` →
`callAI` (`electron/text-ai.ts:253`).

- Ollama: `{model, prompt, stream: false, think?, keep_alive: '5m', options: {temperature:
  0.3, num_predict: max(4096, prompt.length*3), num_ctx}, system?}` (`text-ai.ts:81`).
- Claude: `{model, max_tokens: 4096, messages}` — **no `system` field at all**; the system
  prompt is prepended into the user message (`:150`). No temperature.
- OpenAI: `{model, messages, temperature: 0.3}` — **no `max_tokens`** (`:217`).

Batches of ≤ 10 paragraphs / ≤ 5000 chars (`mono-translation-job.ts:340`) marked with
`<<<N>>>`; a missing paragraph is re-requested individually, then falls back to the source
text. `HOSTED_API_TIMEOUT_MS = 180000` on Claude/OpenAI only — **the Ollama call at
`text-ai.ts:103` has no timeout and no AbortSignal.**

`electron/bilingual-processor.ts` no longer exists; it became `text-ai.ts` when the
bilingual feature was deleted on 2026-09-05 (`text-ai.ts:3`).

### 1.7 Translate EPUB — the wizard/queue bridge

`translateChunkWithProvider`, `electron/translation-bridge.ts:431`; four backends at `:241`,
`:298`, `:363`, `:412`. Same shapes, with **`keep_alive: '10m'`** (`:244`) — the only site
that differs. System prompt is the inline `TRANSLATION_PROMPT` constant (`:102`). Chunks of
`DEFAULT_CHUNK_SIZE = 2500` chars (`:86`). Retries 3× on transport; `TIMEOUT_MS = 180000`
per chunk on all four; job aborts at `MAX_FAILED_CHUNK_COUNT = 10`.

### 1.8 Book analysis and audiobook analysis

`analyzeChunkWithProvider`, `electron/book-analysis.ts:394`; Ollama `:435`, Claude `:495`,
OpenAI `:567`. **No `local` provider** — it throws (`:414`). `temperature: 0.1`,
`num_predict: 4096`, `keep_alive: '5m'`. Expects a JSON array, asked for in prose.
`strictResponse` mode throws by name on refusal, on `stop_reason: 'max_tokens'`, and on an
empty answer, with a sanitized Claude response logged (`:329`, `:355`) — this is the only
place in either app that inspects Anthropic refusal blocks. `CHUNK_SIZE = 6000` (`:136`).
`TIMEOUT_MS = 180000` on the cloud paths; **the Ollama call has no timeout**.

### 1.9 Ollama capability probe and eviction

- `probeThinkingCapability`, `electron/ollama-capabilities.ts:30` — `POST /api/show` with
  `{model}`, cached per `(baseUrl, model)` for the process lifetime, **throws on non-200**.
- `loadedOllamaModels` → `GET /api/ps` (`gpu-arbiter.ts:329`).
- `unloadOllamaModel` → `POST /api/generate` with `{model, keep_alive: 0}` and no prompt
  (`gpu-arbiter.ts:347`).
- `OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434'`
  (`gpu-arbiter.ts:322`) — **the only site in the codebase that honours the env var.**
- `verifyOllamaGenerate` (`ai-bridge.ts:1929`) runs before every Ollama cleanup job and
  doubles as a warm-up.

### 1.10 Dead code, flagged so it is not ported

`LlamaModelServer` (`electron/llama-model-server.ts`, whole file — nothing constructs it);
`cleanupChapterStreaming` (`ai-bridge.ts:3789`); `planFootnoteRemoval` (`:3190`, its only
caller commented out at `:4439`); `loadNarrationTextPrompt` and
`prompts/tts-narration-text.txt`; `prompts/ll-cleanup.txt` and `ll-simplify.txt`;
`normalizeNarrationNumbers`.

---

## 2. BookForge already runs inference servers. This is the finding that matters most.

Five model servers can be alive on this machine, started by three different owners:

| Port | Server | Started by | Arbitrated? |
|---|---|---|---|
| 8300 | vLLM, `Qwen3.5-9B-bf16` / `Qwen3.8-27B-AWQ-INT4` | `electron/text-server.ts` | yes — `vllm:text`, low-priority, yields |
| 8077 | vLLM, `dots.ocr` | `electron/vlm-page-server.ts` | yes — `vlm-page-reader`, **no yield**, by design |
| 8000 | vLLM, `dots.ocr` | `foundry-app/electron/vllm-server.ts` | **no — nothing at all** |
| 8200 | SGLang-Omni, `higgs-v3-ds` | **narrator**, inside the render worker | only via the render's `tts:<jobId>` lease |
| 8769 | llama.cpp, a Cogito GGUF | `electron/llama-bridge.ts` | yes — `llama:cleanup`, yields |

Plus Ollama on 11434, which nothing in the app starts or stops.

### 2.1 `electron/text-server.ts` — a vLLM lifecycle manager that is a small Crucible

1364 lines. It serves `foundry clean-text` / `translate` / `simplify` on this PC today.

- **Port 8300**, `TEXT_SERVER_URL = http://localhost:8300/v1` (`:113`, `:116`).
- Spawns `wsl.exe … bash electron/scripts/vllm/serve_text_vllm.sh` (`buildTextServerCommand`,
  `:926`; spawn at `:1025`), which `exec`s
  `python -m vllm.entrypoints.openai.api_server --model … --served-model-name … --port 8300
  --dtype … --max-model-len … --max-num-seqs … --gpu-memory-utilization …
  --enable-prefix-caching --mamba-cache-dtype … --kv-cache-dtype …
  --limit-mm-per-prompt '{"image":0,"video":0}'`.
- **Two pinned profiles** (`TEXT_MODEL_PROFILES`, `:256`):

  | id | served name | act | repo @ revision | dtype | max-num-seqs | max-model-len | gpu-mem-util |
  |---|---|---|---|---|---|---|---|
  | `qwen35-9b-bf16` | `Qwen3.5-9B-bf16` | clean | `Qwen/Qwen3.5-9B` @ `c2022362…` | bfloat16 | 16 | 16384 | 0.90 |
  | `qwen38-27b-awq-int4` | `Qwen3.8-27B-AWQ-INT4` | translate / simplify / analysis | `cyankiwi/Qwen3.8-27B-AWQ-INT4` @ `63768c10…` | auto | 16 | 16384 | 0.90 |

- `profileForKind(kind)` (`:333`) maps the act to the model; `servedModelForRequest`
  (`:367`) writes the profile's served id onto the request and **refuses by name** if the
  request asks for a different one.
- `assertProfileTable()` (`:391`) enforces three invariants at import: the served name must
  match `/^qwen3(\.|:|-|$)/i` or Foundry will not send `enable_thinking: false`; the
  revision must be a 40-hex commit sha, never `main`; the id must match its key.
- **Staging**: absent weights are downloaded rather than refused — `snapshot_download` at
  the pinned revision into the guest's ext4, resumable, with a JSON progress line every two
  seconds (`electron/scripts/vllm/text_model_download.py`). "Staged" means `config.json` +
  a `*.safetensors` + **no `.incomplete` blob** (`:653`).
- **Adoption**: a server already answering on 8300 with this profile's served name is used
  and never stopped (`:1330`); any other name is refused by name and left alone (`:1339`).
- **GPU**: acquires `GPU_OWNER_TEXT = 'vllm:text'` (`:160`) as the **low-priority** holder —
  it registers an `onYield` and steps off when a render asks for the card (`:1015`). It
  reads `external-gpu-job.lock` and refuses to start while it exists (`:1002`).
- **Shutdown**: `wslPkillGraceful` on
  `[v]llm\.entrypoints\.openai\.api_server.*--served-model-name <name>.*--port 8300`
  (`:139`) — SIGTERM, poll, **never SIGKILL**, because a killed CUDA process wedges the WSL
  VM. An `alive` outcome is logged and left (`:1113`).
- **Keep-warm**: `noteTextQueueIdle(minutes)` (`:1208`) — 0 stops on drain (the default);
  a window always has an end; ceiling 240.
- **Measured** (`docs/TEXT-SERVER.md`): ~110 s to start, ~95 s to swap profiles; 458
  blocks/min against Ollama's 110; 7 requests decoding at util 0.90 out of Foundry's 12.

**This file is a line-for-line rehearsal of Crucible's phase 2.** Manifests with pinned
revisions, an env, a managed subprocess, a readiness probe on `/v1/models`, one resident
model at a time, refusal by name, an accelerator guard, a `/v1/openai` passthrough. The
differences are named in the final section.

### 2.2 `electron/vlm-page-server.ts` — the page reader

- Starts `wsl.exe -d <distro> -e bash -lc "<conda> run -n <env> --no-capture-output vllm
  serve <model> --trust-remote-code --host 0.0.0.0 --port 8077 --gpu-memory-utilization <u>
  --max-model-len 32768"` (`:229`).
- Port fixed at **8077** (`:87`) because a port chosen inside the guest is undiscoverable
  from Windows. Model from `tool-paths.json` `wslVlmModel` — on this PC
  `rednote-hilab/dots.ocr`. No default; absent is a named refusal (`wslVlmRefusal`, `:166`).
- Reservation sizing: `util = clamp(min(freeMB − 1500, 12288) / totalMB, 0.05, 0.85)`
  (`:204`), capped at 12 GiB after a ~20 GiB reservation OOM-killed neighbours on
  2026-08-11.
- **Registers no `onYield`, deliberately** (`:56`): yielding mid-conversion fails a
  ninety-minute run, so other GPU tenants queue behind it. This is the opposite posture to
  the text server, and both are correct for their job length.
- Never adopts a foreign server. Reference-counted holders, no idle keep-warm, immediate
  teardown when the last holder releases. Startup budget 15 min (a first serve downloads
  ~5.7 GB); shutdown `pkill -TERM` inside the guest, left alive rather than SIGKILLed after
  60 s.
- Sole consumer `electron/vlm-convert.ts:700`; a user-configured endpoint wins and nothing
  is started.

### 2.3 `electron/llama-bridge.ts` — the bundled llama.cpp (`local` provider)

Live but confined. Starts `llama-server -m <gguf> --port 8769 -c <8192|16384> -ngl <n>
--threads 4` (`:566`). Models are a hardcoded four-entry `COGITO_MODELS` catalog (`:55`),
bartowski Q4_K_M; the active one is named by `activeModelId` in
`<userData>/llama-models/active-model.json` (`:214`) — **no `app-settings.json` key is
involved**, and `AIProviderConfig.local.model` is explicitly informational only
(`ai-bridge.ts:178`). Request is OpenAI-shaped with
`chat_template_kwargs: {enable_thinking: false}`. Holds `GPU_OWNER_LLAMA` with an `onYield`
that defers until the in-flight generation finishes; 5-minute idle shutdown.
`electron/llama-model-server.ts` is a generic reusable variant that **nothing imports**.

---

## 3. BookForge — TTS

There is **one Python package** behind every TTS call site: `python/narrator`, in the
BookForge repo. BookForge never speaks HTTP to a TTS model. It does one of two things.

| Door | Transport | Module |
|---|---|---|
| **Batch** (library render, retake, CLI) | `spawn()` per phase; stdout text lines + a final JSON line; audio lands as **FLAC files on a shared filesystem** | `narrator.compat.app` / `narrator.compat.worker` |
| **Streaming** (Listen, extension, in-app Play) | one resident `spawn()`; **JSON-lines over stdin/stdout**; audio returns as **base64 PCM16 in the response** | `narrator.serve` |

The HTTP inference server (SGLang-Omni for Higgs) is started and owned **inside narrator**.
No `.ts` file anywhere touches port 8200 or `/v1/audio/speech`.

### 3.1 The engines as configured

**PC — Higgs v3 on SGLang-Omni inside WSL.** Confirmed from
`electron/scripts/higgs/serve_higgs_sgl.sh:158`:

```
exec "$HIGGS_SGL_ENV/bin/sgl-omni" serve \
  --model-path "$MODEL" --model-name higgs-v3-ds \
  --host "$HIGGS_SGL_HOST" --port "$HIGGS_SGL_PORT" \
  --mem-fraction-static "$HIGGS_SGL_MEM_FRACTION" \
  --tts_engine.factory.max_running_requests "$HIGGS_MAX_NUM_SEQS" \
  --tts_engine.factory.cuda_graph_max_bs "$HIGGS_SGL_CUDA_GRAPH_MAX_BS" \
  --tts_engine.factory.max_new_tokens "$HIGGS_SGL_MAX_NEW_TOKENS"
```

Catalog (`electron/data/higgs-models.json`): `stack: sglang-omni`, port **8200**, endpoint
`/v1/audio/speech`, health `/health`, served name `higgs-v3-ds`, `maxRunningRequests: 16`,
`memFractionStatic: 0.6`, `contextTokens: 4096`, `maxNewTokens: 7500`, cold start 110 s.
Sampling is **one engine-level block** at the top of the catalog —
`{temperature: 0.8, topP: 0.95, topK: 50}` — refused if absent (`higgs-models.ts:1111`); a
per-backend override requires a written `_samplingNote` stating the reason.
**On SGLang, sampling is mandatory per request**: omitting it disables top-k and samples the
untruncated 1026-way codebook tail (measured: one chunk ran to the cap with 80 s of
silence).

**Mac — Orpheus on MLX** (`narrator-spawn.ts:561` routes both engines on darwin to the
`narrator-mlx` conda env), and Higgs also has an MLX arm. Orpheus engine defaults
(`python/narrator/engine/orpheus/config.py`): temperature **0.6**, top_p **0.8**, min_p
**0.0**, repetition_penalty **1.1**, max_tokens **3700**, stop `[128258]`, `eosBoost 0.0`
(off), `eosBoostStart 1.2 × expected`, `eosFloor 0.0`, `eosFloorRate 15.0`,
`maxCharsPerSec 19.0`. **No `top_k` and no `seed` are passed at all** in the
`SamplingParams` call (`engine/orpheus/sampling.py:233`).

**Backend-specific caps.** `higgsVoiceCapsForModel(model, arm)` (`higgs-models.ts:1912`)
selects `backends.served` or `backends.mlx` per arm and writes that arm's cap into the
voice document. The mechanism is real and load-bearing — a `null` mlx cap is refused by
name. But **every voice's two blocks carry identical numbers today**: 600/600 (default and
the four `zeroshot-*`), 800/800 (deathstalker, mistborn, owen), 1000/1000 (thirdreich),
1100/1100 (sigma). The "900 MLX vs 1200 served" split appears only in a superseded note for
a deleted checkpoint. **A `tts` job contract must still carry per-backend caps**; it simply
isn't exercising the difference right now.

### 3.2 Library render — prep, worker, retake, assembly

- **Prep** — `parallel-tts-bridge.ts:3350`, spawn at `:3557`, module `narrator.compat.app`.
  argv `--headless --ebook --session --session_dir --language --tts_engine <orpheus|higgs-v3>
  --device --prep_only` plus `--higgs_voice <catalogId>` or `pushVoiceArgs`. **No sampling
  flags** — the XTTS-era `--temperature/--top_p/--top_k/--repetition_penalty/--speed` are
  parsed and ignored (`compat/flags.py:138`). Writes `session-state.json`. Stall timeout 10
  min of stdout silence.
- **Worker** — `:4172`, spawn at `:4473`, module `narrator.compat.worker`. **Chunk text is
  not sent** — it is read from `session-state.json` by global index
  (`render/worker.py:670`). Sampling reaches the worker **as environment variables**
  (`:4315-4463`), all Orpheus-only: `ORPHEUS_TEMPERATURE`, `ORPHEUS_TOP_P`,
  `ORPHEUS_MIN_P`, `ORPHEUS_REP_PENALTY`, `ORPHEUS_EOS_BOOST`, `ORPHEUS_EOS_BOOST_START`,
  `ORPHEUS_EOS_FLOOR`, `ORPHEUS_EOS_FLOOR_RATE`, `ORPHEUS_MAX_CHARS_PER_SEC`,
  `ORPHEUS_SENTENCE_GAP`, `ORPHEUS_BATCH_SIZE`, `ORPHEUS_GPU_MEM_UTIL`,
  `ORPHEUS_MLX_CACHE_LIMIT_GB`, `ORPHEUS_VLLM_DTYPE`.
  Higgs sends none of those: its whole certificate travels in a **voice document JSON file**
  (`higgs-models.ts:2411`, named by `NARRATOR_HIGGS_VOICES`) carrying
  `{kind, checkpointDir | clips[{path,transcript,seconds}], maxChars, targetChars,
  safeMinChars, safeMaxChars, allowedControls, maxReferenceSeconds,
  sampling{temperature,topP,topK}, paceCharsPerSec, maxCharsPerSec, minCharsPerSec}`.
- **Wire protocol back**: progress `Converting sentence <i>/<total> (<pct>%)`
  (`PROGRESS_LINE_RE`, `:2676`); MLX batch heartbeats; guard fires as
  `[HIGGS3][HIGGS_GUARD_EVENT] {json}` / `[ORPHEUS][ORPHEUS_GUARD_EVENT] {json}` (`:116`);
  a final compact JSON result line (`compat/app.py:189`). **Errors go to stdout, not
  stderr** (`:2699`).
- **Artifacts**: `<sentences_dir>/<globalIndex>.flac`, mono 24 kHz PCM_16 FLAC. Resume test
  is "the file exists and exceeds 1024 bytes" (`:2484`).
- **Timeouts**: startup 10 min, no-heartbeat 12 min, watchdog every 30 s;
  `MAX_WORKER_RETRIES = 2`.
- **Reference clips are base64 inline in the POST body** on SGLang
  (`sgl_served.py:344`: `{data: <raw b64>, media_type: 'audio/wav', text: <transcript>}`);
  vLLM-Omni uses a `data:` URI instead.
- **Retake** — `:3859`, spawn at `:4030`. argv adds `--sentence_indices i,j,k` and
  `--num_takes N` | `--take_temperatures t1,t2,…`. **Only the temperature varies per take**;
  Python ranks nothing, selection is BookForge's.
- **Assembly** — `:5941`. No engine, no model; runs natively in the generic bundled tools
  env on every platform (`narrator-spawn.ts:297`).

### 3.3 Streaming — the resident server

`electron/orpheus-worker-pool.ts:832`, spawn at `:1053`: `python -u -m narrator.serve`,
engine chosen by `NARRATOR_ENGINE`. Newline-delimited JSON both ways.

Requests: `{action:'load', id, voice, modelDir?, adapterDir?, baseDir?, caps{}, warm}`
(`:1495`); `{action:'generate', text, language, stream: true, voice?}` (`:1882`);
`{action:'generate_batch', items:[{i, text, voice?, stream?}]}` (`:1708`);
`cancel`, `stop`, `quit`.
Responses: `ready{device,backend}` (the startup sentinel), `status`, `loaded{voice, backend,
engine, sampleRate, pads, edgeFadeMs{in,out}}`, `audio{format:'pcm16', data:<b64>, duration,
sampleRate}`, `chunk{seq,…}`, `done`, `batch_item{i,…}`, `batch_chunk{i,seq,data}`,
`batch_done`, `error`, `stopped`. **No heartbeat of any kind.**

**Sampling on the wire: none.** `PlaySettings.temperature/topP/repetitionPenalty` are
XTTS-era wire compatibility and are ignored (`:68`). Orpheus sampling arrives once per voice
in `load`'s `caps{}`; Higgs sampling arrives in the voice document written before the spawn.

Batching: `flushBatch()` (`:1647`) coalesces a 25 ms window into one `generate_batch` at
`min(STREAM_RAMP_WIDTH = 8, streamBatchCeiling())`; the ceiling is 16 for Orpheus but
**`HIGGS_STREAM_BATCH_WIDTH = 1`** (`:249`) — Higgs Listen dispatches one row at a time
because width measured worthless for it at 2.0× realtime.
Residency: one process kept warm across chunks and sessions, torn down by a 15-minute idle
timer. **A Higgs voice change is a full worker restart** (`:1351`) — a v3 voice *is* the
merged checkpoint the server was started on.

Chunking is BookForge's: `packListenChunks` (`listen-chunks.ts:105`) with a 300-char opener
ramp up to the voice's safe band for Higgs; one sentence capped at 450 chars for Orpheus.
Text is normalised first by `speakableListenText` (`listen-text.ts:179`).

### 3.4 Residency and the GPU lease — the one real defect found

The SGLang server and the MLX process each live **exactly as long as the worker process**
(`v3_engine.py:606` starts it, `:653` stops it, atexit backstops) and stay warm across every
chunk of a book. One server per render. That maps cleanly onto Crucible's exclusive lease.

But BookForge's own lease does not match. `acquireGpuForJob` (`parallel-tts-bridge.ts:7220`)
takes the in-process mutex with a 10-minute timeout, then does full VRAM-tier sizing **only
for Orpheus** — the `if (engine === 'orpheus')` branch returns at `:7411` and Higgs falls
through to a best-effort 4500 MB wait that never fails the job. Worse, if the device
resolves to `CPU` (the `cuda-tts` pack absent and the user on `auto`), `:7229` returns
**before acquiring the mutex at all**, so a Higgs render can run holding no lease. Moving
residency to Crucible deletes this whole class of bug.

`detectRecommendedWorkerCount()` (`:3286`) is **1 worker on Windows/Linux**, up to 4 on
darwin — so on the PC a single process renders the whole book, and the concurrency that
matters is the 16 in-flight POSTs inside it.

---

## 4. BookForge — alignment, ASR, RVC, and the rest of the GPU work

**Correction to the plan documents: `align` is no longer WhisperX.** The app's aligner is
**Qwen3-ForcedAligner-0.6B**, everywhere, with no whisperx fallback
(`electron/coverage-align-job.ts:26`, `:418`). WhisperX survives only as a CPU env
supplying faster-whisper for the rough-transcript stage of whole-m4b alignment.

### 4.1 Post-render chunk alignment — the one that runs on every book

`runPostRenderAlignment`, `electron/parallel-tts-bridge.ts:4833`, called from
`checkAllWorkersComplete` at `:5038`. It is the **final phase of the `tts-conversion`
step**, not a queue row, and it runs *before* the session is copied out of WSL, because the
qwen env is inside WSL.

- `runCoverageAlign(stepId, {processDir, language, device: 'gpu', chapterGap}, null)`
  (`:4893`) → `electron/coverage-align-job.ts:462` → argv (`:410`):
  `align --session-dir <p> --report <p>/coverage.json --language <l> --backend qwen3
  --device <cpu|cuda|mps> --python <alignEnv python> --workers 1 --chapter-gap <s>`.
  **`--backend qwen3` is hardcoded** (`:418`); narrator's own default is still whisperx.
- Spawned through `buildNarratorSpawn` → `wsl.exe -d Ubuntu bash -c "export … && cd ~ &&
  '<wslCondaPath>' run --no-capture-output -n '<qwenAlignEnv>' python -u -m narrator.cli …"`
  (`narrator-spawn.ts:483`). `qwenAlignEnv` is `qwen-align` on this PC. Every argv element
  and env *value* is path-translated by `toGuestPath` (`:595`).
- Inside narrator the model lives in a **grandchild**: `align/env.py:348` spawns
  `[<alignEnv python>, '-m', 'narrator.align.worker']`.
- **Model: `Qwen/Qwen3-ForcedAligner-0.6B`**, pinned by name at
  `python/narrator/align/aligner.py:562`; `dtype = float32 on cpu else bfloat16` (`:620`);
  cached per `(backend, language, device)` in `_MODEL_CACHE` (`:519`) — **one load per
  worker process, reused for the whole book**. ~1.2 GB into
  `HF_HOME = <userData>/runtime/qwen-align-cache`.
- Per chunk: `{index, audioPath, text, language, backend, device, ffmpeg,
  paceCharsPerSecond}`; audio decoded by ffmpeg to **16 kHz mono float32**
  (`SAMPLE_RATE = 16000`, `:97`). Hard cap `QWEN3_MAX_AUDIO_S = 300.0` (`:569`) — longer is
  refused, not chunked. Language must be one of 11 codes (`:575`).
- Results come back as JSON-lines on a **reserved fd 1** (`align/worker.py:56` — fd 1 is
  dup'd for results and pointed at stderr, after whisperx's logger corrupted the stream on a
  401-chunk book). Result `k` answers job `k` by deal position, never by the record's own
  index (`env.py:317`).
- **No timeout anywhere. No retries.** A failed chunk is reported and the run continues; no
  other backend is ever tried. Abort is `taskkill /F /T /PID` — the whole tree, because the
  model is a grandchild (`coverage-align-job.ts:718`).
- **Takes no lease of its own** — it runs inside the TTS step, which already owns the single
  GPU slot and the `tts:<jobId>` lease. `--workers 1` always on a GPU device.

The standalone `align` queue row (`queue-steps/align.ts:131`) is the same call with
`resource: config.device === 'gpu' ? 'gpu' : 'cpu'` (`:147`) — the only config-dependent
lane in the app. `'gpu'` resolves through `systemProbe` to `mps`/`cuda`, or **refuses; never
a silent CPU downgrade** (`coverage-align-job.ts:368`). Nothing in the app composes this row
any more; the live producer is `cli/coverage-align.js:111`, which always says `cpu`.

### 4.2 Whole-m4b EPUB→audio alignment — and why it cannot run on this PC

`runEpubAlignOnFiles`, `electron/whisperx-align-bridge.ts:589`, spawn at `:767`. **Two
interpreters, one script** (`electron/scripts/align_audiobook.py`): the rough transcript is
**faster-whisper `base`** in the CPU-only `whisperx-env` (600 s slices, `vad_filter`,
`word_timestamps`), and the fine alignment is Qwen3-ForcedAligner in the qwen-align env.

**Windows is refused by name on this door** (`:642`): if the qwen env is `viaWsl`, it
throws — the library is on `Z:` (`\\TITAN\iO`), a network drive WSL cannot mount. So this
feature currently **cannot run on owens-pc at all**. Unblocking it today needs a native
Windows CUDA qwen env; unblocking it through Crucible needs nothing but a server that can
see the audio.

Protocol is `STAGE <name>` / `SUBPROGRESS <name> <n>` / `RESULT <json>` / `ERROR <msg>`
over five weighted stages (`:476`). No timeout, no retries. Takes no arbiter lease, but the
`generate-sentences` step is `resource: () => 'gpu'` (`queue-steps/generate-sentences.ts:47`),
so it burns the single GPU slot for the whole run including the CPU-only rough pass.

### 4.3 Whisper transcription

`transcribeAudiobook`, `electron/transcribe-bridge.ts:84`, spawn at `:119` —
`electron/scripts/transcribe_audiobook.py` in the **bundled tools env, native, never WSL**.
Catalog `WHISPER_MODELS` (`electron/whisper-models.ts:47`): `tiny`/`base`/`small`/`medium`/
`large-v3`/`distil-large-v3`, from `Systran/faster-whisper-*`, 75–3090 MB. **No default in
code** — the picker chooses; an absent model is downloaded inside the job.
`compute_type = 'float16' if cuda else 'int8'`, with a one-shot CPU fallback. The file is
decoded once to 16 kHz mono float32 and transcribed in **900 s windows with 15 s
back-overlap** (`CHUNK_SEC`/`OVERLAP_SEC`, `:46`), `vad_filter=True, word_timestamps=True`.
Protocol: `PROGRESS <frac> [processedSec totalSec cues]`, `DECODE`, `STAGE`, `DEVICE`, then
a final JSON line.
**This is the only ASR site that takes the arbiter lease**:
`acquireGpu('whisper-transcribe:<name>', {timeoutMs: 600000})` (`:100`), released in a
`finally` (`:201`).

### 4.4 RVC voice conversion

`runRvcEnhancement`, `electron/rvc-job.ts:167` → `rvc-bridge.ts:151` → `runUrvcConvertDir`
(`:232`), spawn at `:241`:
`python -m ultimate_rvc.cli.main generate convert-dir <in> <out> <modelName>
--index-rate <r> --protect-rate <p> --input-glob '*.flac' --output-ext flac
[--f0-method …] [--hop-length …] [--n-semitones …]`.
**Never `urvc.exe`** — pip's Windows console script bakes a stale shebang (`:53`).
Env from `BOOKFORGE_RVC_ENV` or `componentManager.resolveEntry('rvc-env')`; `cuda-rvc`
overlays torch 2.7.1+cu126 into it. Env hardening at `:209`: `URVC_SKIP_INIT=1`,
`HF_HUB_OFFLINE=1`, `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1` (three bundled OpenMP
runtimes SIGSEGV otherwise).

- **Model identity is a folder name**, not a repo: `<userData>/runtime/rvc-models/rvc/
  voice_models/<Name>` (`rvc-models.ts:9`). Catalog `electron/data/rvc-voice-assets.json`,
  plus user sources and **auto-discovered local drop-ins** (any folder with a `.pth`;
  `forceIndexRate0` is derived from the *absence* of a `.index`).
- Parameter precedence (`rvc-models.ts:261`): `forceIndexRate0 → 0`, else the user's
  explicit value, else the catalog default, else `RVC_STOCK_INDEX_RATE = 0.5`.
  **`protectRate`'s scale is inverted** — lower protects more, 0.5 is off, and it is a no-op
  at index rate 0 (`rvc-bridge.ts:91`). An absent `f0Method`/`hopLength` means the flag is
  **omitted** so urvc keeps its own default.
- **Batching is a memory bound, not a throughput choice**: `batchSize ?? 96` files per
  recycled worker process, each batch hardlinked into a fresh mkdtemp and converted by its
  **own process that exits** (`:356`, proven on a 64 GB Mac 2026-07-17). **The model reloads
  once per 96 sentences.**
- Progress is one regex, `/^\[RVC\]\s+(\d+)\/(\d+)/` (`:257`). No timeout, no retries.
  Force-killing mid-CUDA-init can wedge the env's torch import until reboot (`:277`).
- Lease `rvc:job:<jobId>` (`rvc-job.ts:316`), and **the reuse check runs before the lease is
  taken** (`:291`) so a cached derived set never queues behind a nine-hour narration.

### 4.5 Denoise and the Enhance tab — three more GPU models

- **Denoise is a real GPU model, not DSP.**
  `DENOISE_MODEL = 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt'` — **hardcoded at
  `electron/denoise-bridge.ts:58`**, no settings key, no component id, downloaded by
  audio-separator on first use. Runs in the rvc-env via
  `electron/scripts/separator_worker.py` (spawn `:711`), **resident across ~22-minute
  blocks**, `use_autocast=True` (20.1x → 29.2x realtime). Protocol is `@@BFSEP@@`-sentinel
  JSON lines parsed by `indexOf` + `JSON.parse`, not a regex (`:807`).
  **The sample-exact post-condition is load-bearing**: one stem, rate still 44100, and
  **frames exactly equal in and out** (`:305`) — the offsets manifest re-cuts ~1,400
  sentences from it.
- **Enhance tab** runs three models under one lease (`enhance-bridge.ts:1364`):
  `vocals_mel_band_roformer.ckpt` (hardcoded `:92`), **Resemble Enhance** in its own conda
  env with weights bundled (`DEFAULT_ENHANCE_PARAMS` at `:106` = `{nfe: 64, tau: 0.75,
  lambd: 0.1, solver: 'midpoint', smartChunk: true, seeds: 5, anchor: true}` — and
  `tool-paths.json` on this PC sets exactly those), then RVC. Resemble's `resolve_device()`
  is cuda → mps → **`sys.exit`**; it refuses CPU. It is called twice per run
  (`--denoise-only`, then full), so **the model loads twice**. Progress for the roformer is
  a **tqdm percentage scrape** (`:637`).

### 4.6 The component/env resolution mechanism

Two functions. `componentManager.resolveEntry(id)`
(`electron/components/component-manager.ts:1137`) returns the installed record's
`entryPath` if it exists on disk, else `null` — covering a managed download and a
user-pointed external path identically. `getEnvPathForEngine(engine)`
(`electron/narrator-paths.ts:257`) routes Orpheus/Higgs to a WSL marker or a component, and
**no engine → `toolsEnvPath()`** (`:319`); `getPythonInvocation(engine)` (`:403`) wraps it
into `conda run -p <env> python`.

Catalog ids (`components/component-catalog.ts:236`): `calibre`, `tesseract`, `orpheus`,
`llama-cuda`, `cuda-tts`, `cuda-rvc`, `whisper`, `rvc-env`, `resemble-env`, `whisperx-env`,
`qwen-align-env` (**darwin-arm64 only** — on Windows there is no component, only the
`qwenAlignEnv` WSL env-name setting), `rvc-voice-*`, `whisper-model-*`, `foundry-cli`.
`higgs-env` and `narrator-mlx` are referenced but **not in the catalog**.

Installed on owens-pc today: `rvc-env` and `whisperx-env` (managed), `resemble-env`
(external, `C:\Users\tellt\miniconda3\envs\resemble-enhance`), `foundry-cli`, `whisper`
(a pip overlay marker inside the old e2a python env), five `whisper-model-*`, nine
`rvc-voice-*`/`rvc-local-*`, and one orphan: **`blocks-model-foundry-blocks-v1-4b`**, a GGUF
at `%LOCALAPPDATA%\foundry\models\` that **no code in either repo references** — the residue
of the retired "blocks" classifier, the same feature whose `blocks-server.ts` consumer
`llama-model-server.ts` still names but which does not exist. **`qwen-align-env` is not
installed**, confirming that alignment on this PC goes through the WSL `qwen-align` env by
name.

### 4.7 The four GPU model backends, side by side

Each of these is a candidate job type. Seven points each; the detail is in the sections
above.

| | **dots.ocr** (§7) | **Qwen3-ForcedAligner** (§4.1) | **faster-whisper / WhisperX** (§4.2-4.3) | **urvc** (§4.4) |
|---|---|---|---|---|
| **1. Feature, site** | PDF → EPUB page reading. `foundry/src/vlm/endpoint.ts:148`; BookForge drives it at `electron/vlm-convert.ts:756` | post-render chunk alignment (`parallel-tts-bridge.ts:4833`) and the standalone row (`queue-steps/align.ts:131`) | "Generate sentences" — ASR (`transcribe-bridge.ts:84`) and the rough pass of whole-m4b align (`whisperx-align-bridge.ts:589`) | RVC enhancement (`rvc-job.ts:167`) and whole-file convert (`rvc-bridge.ts:640`) |
| **2. Runner** | **vLLM** serving an HF repo, OpenAI `/v1/chat/completions`. MLX in-process on darwin. Never transformers | **transformers in-process**, `Qwen3ForcedAligner.from_pretrained` (`aligner.py:621`), in a grandchild python | **faster-whisper** (CTranslate2) in-process | `python -m ultimate_rvc.cli.main generate convert-dir` — **never `urvc.exe`** (stale baked shebang, `rvc-bridge.ts:53`) |
| **3. Model id, weights** | `rednote-hilab/dots.ocr` (endpoint) / `mlx-community/dots.ocr-4bit` (MLX), `foundry/src/vlm/models.ts:160`. Weights in the WSL HF cache; `maxTokens 8192`, `maxPixels 11289600` | `Qwen/Qwen3-ForcedAligner-0.6B`, pinned at `aligner.py:562`. ~1.2 GB into `<userData>/runtime/qwen-align-cache` | `Systran/faster-whisper-{tiny,base,small,medium,large-v3}` + `faster-distil-whisper-large-v3` (`whisper-models.ts:47`), 75–3090 MB into `<userData>/runtime/whisper-models/<id>/model.bin`. **No default — the picker chooses** | a **folder name**, not a repo: `<userData>/runtime/rvc-models/rvc/voice_models/<Name>`. Catalog `electron/data/rvc-voice-assets.json` + auto-discovered drop-ins. Base assets: contentvec + `rmvpe.pt` |
| **4. Per-call input** | one page, **PNG at a pinned 200 dpi** (`VLM_DPI`, `foundry/src/vlm/read.ts:67`; 1300×2112 for a 468×760 pt page), **base64 data URI** in an `image_url` content part + the card's `layout-all` text prompt (`DOTS_PROMPT`, `models.ts:130`, its sha256 pinned by `test/vlm/dots.test.ts`), `temperature: 0`, `max_tokens ≤ 8192`. **No JSON schema** — the dialect is parsed from prose | one chunk: 16 kHz mono float32 audio **≤ 300 s** (`QWEN3_MAX_AUDIO_S`, `:569`) + the exact word text + an ISO language from an 11-code table. Longer audio is **refused, not chunked** | the whole file decoded once to 16 kHz mono float32, then **900 s windows with 15 s back-overlap**; `vad_filter=True`, `word_timestamps=True`; `compute_type = float16 on cuda else int8` | a **directory** of short FLACs; `--index-rate`, `--protect-rate` (**inverted scale** — lower protects more, 0.5 is off), `--n-semitones`, `--f0-method` (`rmvpe`), `--hop-length`. An absent f0/hop flag is **omitted**, not defaulted |
| **5. Output, protocol** | `choices[0].message.content` + `finish_reason` + `usage.completion_tokens`. Appended to a **resumable bank** (below). Progress on **stderr** | JSON-lines on a **reserved fd 1** (`align/worker.py:56`); result *k* answers job *k* by deal position. Products: `coverage.json` + `<stem>.sentences.vtt` | `PROGRESS <frac> [sec total cues]` / `DECODE` / `STAGE` / `DEVICE` then a final JSON line; five weighted `STAGE`/`SUBPROGRESS`/`RESULT` stages on the align door | one regex, `/^\[RVC\]\s+(\d+)\/(\d+)/`. Every input must produce an output, verified before advancing |
| **6. GPU, residency, batching** | **vLLM continuous batching, 12 pages in flight** (`DEFAULT_VLM_CONCURRENCY`, `endpoint.ts:130`). Server resident for the run. Lease: `vlm-page-reader` with **no `onYield`** on the WSL route; **none at all** on the hosted port-8000 route | GPU, **one model load per worker process, reused for a whole book**. `--workers 1` on a GPU device. **Takes no lease of its own** — it rides inside the TTS step's | GPU when available. **The only ASR site that takes the lease** (`transcribe-bridge.ts:100`). No batching — sequential windows. The rough pass runs in a **CPU-only** `whisperx-env` | GPU via `cuda-rvc`'s torch 2.7.1+cu126. **The model reloads once per 96 files** — `batchSize ?? 96` per recycled process is a **memory bound, not throughput** (`rvc-bridge.ts:356`). Lease `rvc:job:<id>`, taken **after** the reuse check |
| **7. Where it runs today** | **WSL** (`vllm serve` in env `dots`, port 8000 hosted or 8077 standalone). MLX natively on the Mac | **WSL** (`conda run -n qwen-align`). On the Mac, the `qwen-align-env` component. **Windows native: not installed** | **Windows native**, bundled tools env — never WSL. The `whisperx-env` is CPU-only by design | **Windows native**, the managed `rvc-env`; MPS-native torch on the Mac. Source fork at `C:\Users\tellt\Projects\ultimate-rvc` |

**Resumability, which `vlm-pages` must not lose.** BookForge banks page answers at
`~/Documents/BookForge/foundry-runs/vlm-<pdfSha256[0:16]>/readings.jsonl`
(`vlmReadingsPath`, `electron/vlm-convert.ts:192`) — so **the run directory is keyed by the
PDF's sha256**, while the bank inside it is **keyed by page number and belongs to that one
PDF** (`foundry/src/vlm/readings.ts`). Every answer is appended and **fsynced the moment it
exists**; a kill costs only the page in flight. A finished run drops `completed.json` beside
it, which is how a resume is told apart from a replay; a re-read writes into a **pending
bank and swaps by one rename** only when it has actually finished, so a dead re-read leaves
the finished bank exactly where it was. It is a cache of **answers**, not of books —
everything downstream still re-renders, re-parses and re-assembles, so a dialect fix is free
over answers that cost an hour.

That behaviour is app policy and should stay in the app. What Crucible must not do is make
it impossible: a `vlm-pages` job that only reports at the end of a whole PDF would throw
away the per-page fsync that makes a killed ninety-minute read cost one page.

---

## 5. BookForge — the queue, the lanes, and the lock

- A step declares a `StepResource` of `gpu` | `cpu` | `wait`
  (`shared/queue/engine-types.ts:179`), with `RESOURCE_SLOTS = {gpu: 1, cpu: 2, wait: 32}`
  (`:182`). **One GPU step at a time, whatever it is.**
- Which steps are GPU steps: `tts-conversion`, `final-denoise`, `rvc-enhancement`,
  `generate-sentences`, `vlm-convert`, `video-assembly` (all `resource: () => 'gpu'`);
  `align` is `config.device === 'gpu' ? 'gpu' : 'cpu'` (`queue-steps/align.ts:147`);
  `foundry-job` is `gpu` for `read`/`translate`/`simplify`/`clean` and `cpu` otherwise
  (`queue-steps/foundry-job.ts:101`); and **`resourceForProvider`
  (`queue-steps/runtime.ts:37`) routes an AI pass to `cpu` when the provider is `claude` or
  `openai` and to `gpu` when it is `ollama` or `local`** — "a hosted API is network latency
  and nothing else".
- A GPU step additionally passes **admission** (`queue-engine.ts:44`): it holds while
  `%APPDATA%\BookForge\external-gpu-job.lock` exists, and while the `gpu-arbiter` reports a
  holder that is not one of ours. The engine **checks** the arbiter rather than acquiring
  it, because the bridges acquire it themselves.
- `shared/gpu/external-job-lock.ts:31` — the lock is **Windows-only** and is a convention
  with the Windows-side training tooling. Nothing inside the app creates it; an unreadable
  or empty lock still counts. It was held at audit time by a `higgs-v3-finetune` agent.
- `gpu-arbiter.ts` is an **in-process mutex** (`:57`), not a system-wide one: one holder,
  waiters nudge the holder's `onYield`, and `acquireGpu`'s `timeoutMs` lets a waiter
  **proceed without the lock** rather than deadlock (`:88-99`). `waitForFreeVram` (`:122`)
  is a best-effort `nvidia-smi` preflight for processes the mutex cannot see, and it never
  fails a job.

Taken together: BookForge's arbitration is honest about being local and in-process, and it
says so. Crucible's queue is the first thing that could make it true across machines.

---

## 6. Foundry — the text acts

Foundry runs on **Bun** (`package.json` `engines.bun >= 1.1.0`, shebang at `src/cli.ts:1`).
Everything text-side funnels through two transports and one chooser, with no
Ollama/vLLM branching above that seam.

| Module | Role |
|---|---|
| `src/translate/ollama.ts` | the Ollama dialect (`/api/chat`, `/api/tags`) |
| `src/translate/vllm.ts` | the OpenAI dialect (`/v1/chat/completions`, `/v1/models`) |
| `src/translate/model-server.ts:52` | the only place that picks — `--server ollama\|vllm`, declared, never sniffed |
| `src/analyze/verify.ts:314` `askConstrained` | the closed-question door, which branches to Ollama **`/api/generate`** (not `/api/chat`) or vLLM chat |

### 6.1 The prose transport

**Ollama** (`ollama.ts:285`, body at `:203`):
`{model, stream: false, messages: [system, user], options: {temperature, num_ctx,
num_predict}, think: false}`. `think` is added **only** when `/^qwen3(\.|:|-|$)/i` matches
the tag (`:129`) — qwen2.5 400s on the field.
**vLLM** (`vllm.ts:401`, body at `:237`):
`{model, stream: false, messages: [system, user], temperature,
max_tokens: capFor(system, user, wanted, maxModelLen),
chat_template_kwargs: {enable_thinking: false}}`. `num_ctx` is deliberately dropped
(`:244`). `capFor` (`:208`) is
`maxModelLen − (⌈(len(sys)+len(user))/2.5⌉ + 256)`, floor 128 — **and if the server did not
report `max_model_len` on `/v1/models`, there is no clamp at all.**

**Never sent anywhere in the repo**: `top_p`, `top_k`, `min_p`, `repeat_penalty`,
presence/frequency penalties, `seed`, `stop`, tools, images. Every body is
`stream: false`. The only deadline is a **300 s whole-request** `AbortController`
(`ollama.ts:89`), and **`Transport` has no abort member** — a pool cannot cancel in-flight
requests. The vLLM path has no unload: `releaseModel` returns `'not-ours'`
(`model-server.ts:205`) by ruling, because BookForge's arbiter owns that process.

### 6.2 The constrained transport — the one place a schema is used

`askConstrained` (`verify.ts:314`) is used by analyze verdicts and both tag calls.

- vLLM (`vllm.ts:292`): `{model, stream: false, messages: [one user turn, no system],
  temperature: 0, max_tokens, response_format: {type: 'json_schema', json_schema: {name,
  schema, strict: true}}, chat_template_kwargs: {enable_thinking: false}}`.
- Ollama (`verify.ts:209`): `{model, prompt, stream: false, format: <schema object>,
  options: {num_ctx, num_predict, temperature: 0}, think: false}`.
- **The thinking-model trap** (`verify.ts:283`): when `response` is empty and `thinking` is
  a non-empty string, `thinking` is read as the answer — a JSON grammar on a thinking model
  routes the object into the reasoning channel.
- Truncation: Ollama `done_reason === 'length'`, vLLM `finish_reason === 'length'`
  (`:387`). Both become a **degradation, not a throw**. **No retries at all** on this door.

### 6.3 The call sites

| Act | file:line | Model + default | Sampling | Concurrency | Chunking |
|---|---|---|---|---|---|
| translate / `--rewrite` simplify | `src/translate/run.ts:2672` (`askOne`), `:2709` (`askGroup`) | `--model`, default `DEFAULT_TRANSLATE_MODEL = 'qwen3.8:27b'` (`:200`); app key `defaultLlmModel` | temperature **0.2**, `num_ctx` 8192 (Ollama), `num_predict = max(128, ⌈len×4/2.5⌉)` | `--concurrency`, default **4** Ollama / **12** vLLM (`model-server.ts:103`) | one paragraph per request; batching only within one list/quote/table under `CHUNK_CHARS = 2000` (`:268`) |
| clean-text (book) | `src/clean/runner.ts:176`; loop `tts-number-normalizer.ts:2012` | `--model`, default `qwen3.5:9b-q8_0` (`:144`); app key **`cleanTextModel`** | temperature **0**, `num_predict` **2048**, `num_ctx` pinned once per book, clamped [4096, 16384] (`runner.ts:81`) — **not sent under vLLM** | 4 Ollama / 12 vLLM | **one block per request, never batched**; neighbours as read-only context |
| clean-text (bare EPUB) | `src/clean/epub.ts:564`, `:573` | same | same | same | text nodes of stamped EPUB elements |
| analyze verdicts | `src/analyze/run.ts:493` → `verify.ts:407` | `--model`, default `qwen3.8:27b` | temperature **0**, `max_tokens` **128**, JSON schema | **1** Ollama (`run.ts:136`) / 12 vLLM | one request per (window, category) |
| tag — aboutness | `src/tag/run.ts:428` → `ask.ts:163` | `--model`, default `qwen3.8:27b` | temperature **0**, `num_predict` **128**, schema | **serial `for` loop**, no pool (`run.ts:425`) | 5 passages per tag (`evidence.ts:30`) |
| tag — suggestions | `src/tag/run.ts:458` → `ask.ts:214` | same | `num_predict` **512**, schema | one call per document | 12 passages / 6000 chars |
| NLI ranker (**not an LLM**) | `src/analyze/nli-bridge.ts:427` | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (`:68`) | — | resident subprocess, NDJSON over stdin/stdout | `texts × hypotheses` in one round trip |

**`foundry tag` is Ollama-only today** — `openModelServer({kind: 'ollama', …})` is
hard-written at `tag/run.ts:269` and the command has no `--server` flag. The transport
underneath already speaks both; adding the flag is the whole change.

### 6.4 Response handling and failure policy

- translate: `checkAnswer` (`run.ts:1066`) refuses an answer under **25 %** of the source
  length (a summary) or over **3×** (a ramble). **3 attempts at identical settings**
  (`ATTEMPTS = 3`, `:230`); after that the block is **left in the source language**, not
  failed. **25 refusals ends the run** (`REFUSAL_LIMIT`, `:246`).
- clean-text: `firstJsonObject` → `JSON.parse` → `edits` must be an array. **One** transport
  re-roll, **one** parse retry at the same settings (temperature 0, so a second identical
  answer is the real answer). Then a validation wall: ≤ 24 edits/block, `find` ≤ 200 chars,
  ≤ 25 % of the block's characters edited, and every accepted edit must be re-placeable in
  the original or the pass **throws**. **Over 10 % parse-fail share ends the run.**
- analyze/tag: any failure, truncation or unreadable answer is a **degradation** — recorded
  as skip + warning, never a flag, and deliberately not cached. Every tag call degraded →
  the run throws.

### 6.5 Settings and the served-model-id-as-cache-key problem

Foundry has **two** settings files, deliberately:

- **`<configDir>/settings.json`** (`src/backend/settings.ts:53`) — the *engine's*, governing
  the **VLM reading path only**: `backend.mode` (`auto|endpoint|mlx`), `endpointUrl`,
  `endpointModel`, `wslDistro`, `vllmPython`, `python`. It carries no text-act key by
  design.
- **`<userData>/app-settings.json`** (`app/electron/app-settings.ts:215`) — the *app's*:
  `defaultLlmModel` (default `qwen3.8:27b`), `cleanTextModel` (default `qwen3.5:9b-q8_0`),
  `ollamaUrl`, `llmServer`, `vllmUrl` (default `http://localhost:8000/v1`), `vllmModel`
  (empty means "ask the server"), `keepServerWarmMinutes`. **BookForge reads this same file
  through a mirrored clamp set** (`electron/narration-clean-text.ts:223`) so the two doors
  cannot dial different servers.

> Foundry's declared default `vllmUrl` is `http://localhost:8000/v1`, and **port 8000 on
> this machine is Foundry's own reading server serving dots.ocr**. A machine that flips
> `llmServer` to `vllm` and leaves the URL alone points a text cleanup at a vision model.
> `textServerRoute` (`text-server.ts:458`) says so in a sentence rather than starting
> anything. This is exactly the class of mistake a capability-advertising server removes.

**The served-model id is a cache key.** It is hashed into `cleanKey`
(`src/clean/run.ts:148`), the translate bank, and the narration stamp written into the
EPUB's OPF. Changing the name Crucible reports for the same weights **re-asks every block of
every book**.

`app/electron/job-queue.ts` `modelArgs` (`:2536`) emits `--model` only if non-empty and
`--server vllm` only when not the default; `argsFor` is exported specifically so
BookForge's `cli/clean-step.js --dry-run` prints the same line.

**Foundry sends no images outside `src/vlm/`.** Verified by grep for `base64`, `image_url`,
`images`, `toDataURL` across `src/` minus `src/vlm/`.

**There is no shared prompt module.** `src/clean/prompt.ts:27` is the closest — it only
concatenates two embedded `.txt` files in a fixed order, and that order and separator are
part of a cross-repo artifact BookForge and orpheus-finetune both check.

---

## 7. Foundry — VLM page reading

### 7.1 The request, which is an ordinary chat completion

`foundry/src/vlm/endpoint.ts:133`, body at `:163`:

```
POST <endpoint>/chat/completions
{ "model": <model>, "temperature": 0, "max_tokens": <adaptive, ≤ 8192>,
  "messages": [{ "role": "user", "content": [
    { "type": "image_url", "image_url": { "url": "data:image/png;base64,<page png>" } },
    { "type": "text", "text": <prompt> } ] }] }
```

- **`temperature: 0` always** (`endpoint.ts:163`, and the reason is stated at `:22` — "a layout answer is a measurement of a page"). No `top_p`, `top_k`, `seed`, `stop`, or
  `response_format`.
- **One page per request**, `concurrency` workers over a shared queue (`:132`), default
  **12 in flight** (`DEFAULT_VLM_CONCURRENCY`, `:130`; mirrored at
  `bookforge/shared/vlm/conversion.ts:355`). BookForge hands the WSL reader
  `concurrency: 0` (`electron/vlm-convert.ts:704`), i.e. it takes the default 12.
- Response read as `choices[0].message.content` + `finish_reason` +
  `usage.completion_tokens` (`:196`).
- **Nothing is retried** — a failed page names itself and the run stops. `--readings` is
  what makes the re-run cheap (`:32`). No timeout on `vlm-convert` at all, deliberately
  (`bookforge/electron/foundry-bridge.ts:398`).

### 7.2 Rasterisation is local, pinned, and not a wire parameter

**`export const VLM_DPI = 200`** — `foundry/src/vlm/read.ts:67`. A 468×760 pt page becomes
**1300×2112 PNG**. The pixel budget on the endpoint route is the model's own `maxPixels`
(11,289,600 ≈ no resize at 200 dpi); on MLX it is `MLX_MAX_PIXELS` (`read.ts:81`).
**BookForge passes no DPI flag anywhere**, and `--python <interpreter with PyMuPDF>` is
**mandatory on every route** (`electron/vlm-convert.ts:110`) because rasterisation is
always local. Page selection is exclusion-only: `--skip-pages <1-based csv>`
(`shared/vlm/conversion.ts:450`).

So **Crucible receives pictures, never PDFs.** DESIGN.md's "send the PDF; the server
rasterises" is a larger, different job than what either app does today.

### 7.3 The model

Registry entry `dots-ocr`, `foundry/src/vlm/models.ts:160`: MLX repo
`mlx-community/dots.ocr-4bit`, `endpointModel: 'rednote-hilab/dots.ocr'` (`:183`),
`maxTokens: 8192` (`:181`), `maxPixels: 11289600` (`:182`), and the prompt is the model
card's `layout-all` prompt **pinned by a sha256 test** (`:129`).
`DEFAULT_VLM_MODEL_ID = 'dots-ocr'` (`:257`).

### 7.4 Three routes, and a fourth server nobody arbitrates

`resolveVlmRoute` (`bookforge/shared/vlm/conversion.ts:535`) picks one of three, and **no
route is ever a fallback for another**:

| route | when | who serves |
|---|---|---|
| `endpoint` | a user-typed URL — **always wins** | whatever is there |
| `mlx-local` | darwin/arm64 only | in-process MLX |
| `wsl-server` | Windows | `electron/vlm-page-server.ts` on port 8077 (see 2.2) |

**And there is a fourth.** The hosted path — `foundry-job` with `request.kind === 'read'`
(`queue-steps/foundry-job.ts:110`) — runs through the vendored Foundry, whose
`foundry-app/electron/vllm-server.ts` starts **its own vLLM on port 8000** with a flat
`GPU_UTIL = 0.5` (`:65`, `:94`) and `MODEL = 'rednote-hilab/dots.ocr'` hardcoded (`:80`).
**It has no gpu-arbiter import, and `foundry-job.ts:201` brackets only the *text* server —
a `read` gets no lease at all.** This is the one real residency hole in the app.

Live config confirms this is the active path on this PC.
`%APPDATA%\foundry\settings.json` (the *engine's* file, separate from the app's):

```json
{ "backend": { "mode": "endpoint", "endpointUrl": "http://localhost:8000/v1",
               "wslDistro": "Ubuntu",
               "vllmPython": "/home/telltale/anaconda3/envs/dots/bin/python" } }
```

so the reading server's launch line here is
`conda run -n dots --no-capture-output vllm serve rednote-hilab/dots.ocr
--trust-remote-code --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.5
--max-model-len 32768`. The env is **`dots`**, not the `foundry-vllm` that
`backend-setup.ts:45` builds — this machine was set up by hand. No `endpointModel` is set,
so the registry's `rednote-hilab/dots.ocr` is what every request carries. With
`keepServerWarmMinutes` absent (0), dots.ocr is torn down the moment the queue drains.

BookForge itself issues exactly one VLM HTTP request: the Test button,
`GET <base>/models` with a 10 s timeout (`electron/vlm-convert.ts:1029`). Everything else is
`foundry vlm-convert --pdf … --readings … --python … [--vlm-endpoint <url>
[--vlm-endpoint-model <m>] [--vlm-concurrency <n>]]` (`shared/vlm/conversion.ts:420`),
spawned as a **native Windows .exe** (`foundry-bridge.ts:420`) — **there is no WSL route for
the Foundry CLI itself**; grep for `wsl.exe` / `windowsToWslPath` / `/mnt/c` across the
Foundry bridge files returns zero.

Progress is on **stderr**, parsed by `parseVlmProgressLine` (`conversion.ts:649`) in a
deliberate order: `/^page\s+(\d+)\/(\d+):\s+rendered$/` first (it has no prefix), then a
`startsWith('vlm-convert:')` gate, then the endpoint form, then the MLX form.

### 7.5 The vendored copy is not stale on the model seam

The full `git ls-tree` of `foundry@3b13392:app/` against `bookforge:foundry-app/` is **132
source blobs, all identical**; the only extra paths are BookForge's own `VENDORED.md` and an
`IPC-CHANNELS.md` byte-identical to Foundry's. Across all of
`foundry-app/{electron,shared,src}` there are **zero occurrences** of `temperature`,
`top_p`, `max_tokens`, `num_ctx`, `num_predict` or `repetition_penalty` in code, no prompt
text, and the complete `fetch(` inventory is four calls: `GET /v1/models` readiness
(`vllm-server.ts:294`), Ollama `GET /api/version` and `/api/tags` (`ollama.ts:76`), Ollama
`POST /api/pull` (`ollama.ts:328`), and the `foundry-file://` protocol handler. **No
`/chat/completions` anywhere in the vendored tree** — every inference call is in
`foundry/src`.

Its built `dist/` **is** stale (built 2026-09-08; four sources refreshed 2026-09-11), but
the 28 argv flags composed by `dist/electron/job-queue.js` are identical to source, so the
model seam is unaffected — only the queue-pump ordering fix is missing from what runs.

> **One thing to design around:** a stored queue row in a project's `project.json` **freezes
> its own `model` / `server` / `ollama` / `concurrency` at enqueue time**. A resumed job asks
> for what the row recorded, not what the settings file says now. Pointing a machine at
> Crucible does not repoint rows already queued.

---

## 8. The CLI adapters — BookForge's headless mirror

Every adapter is spawned by `cli/bookforge-tts.py` as
`node --require cli/electron-stub.js cli/<adapter>.js …` and `require`s a **compiled**
`dist/electron/*.js` module, calling the exact symbol the app's queue step calls;
`tools/test-cli-parity.js` enforces it. So the CLI is not a second implementation — it is a
second **caller**, which is why it belongs in this audit: whatever Crucible offers must
satisfy both.

| CLI door | Compiled door | Model/server flags |
|---|---|---|
| `--ai-clean` (`cli/ai-clean.js:153`) | `aiBridge.cleanupEpub` | `--provider`, `--model`, `--ollama-url`, `--temperature` (0.1), `--chunk-size`, `--stages`, `--parallel-workers`. Key only via env |
| `--clean` (`cli/clean-step.js:387`) | vendored `foundry-app` `job-queue.runJob`, arbiter via `dist/electron/text-server.js` | `--server ollama\|vllm`, `--model`, `--ollama`, `--concurrency`, `--keep-model`, `--keep-server` |
| `--clean-lines` (`cli/clean-lines-step.js:278`) | `narration-clean-text` + `foundry-bridge.runFoundry` + 5 `text-server` doors | `--keep-model`, `--keep-server`; **`--model` explicitly ignored** |
| `--narration-text` (`cli/narration-text-step.js:110`) | `narration-clean-text.cleanTextEpub` | none — **`--model` is refused with a printed note** |
| `--pass` (`cli/processing-pass-step.js:75`) | `processing-passes.runProcessingPass` | `--provider`, `--model` (required), `--ollama-url`, `--mode`, `--source-lang`/`--target-lang` |
| `--narration-prep` | `parallel-tts-bridge.prepareNarrationInput` | none; model is `ttsNumberNormalizerModel` |
| `--tts` (`cli/orpheus-batch-render.js:448`) | `parallel-tts-bridge.renderRangeHeadless` | `--engine`, `--voice`, `--model-dir`, `--higgs-override <json>` |
| `--audiobook` | + `denoise-job.runFinalDenoise`, `reassembly-bridge.startReassembly` | as above plus `--sentence-gap`, `--chapter-gap`, `--sentences-dir` |
| `--rvc` / `--rvc-enhance` | `rvc-bridge.convertFileRvcChunked` / `rvc-job.runRvcEnhancement` | `--model`/`--voice-id`, `--index-rate`, `--protect-rate`, `--f0-method`, `--n-semitones` |
| `--generate-sentences` | `whisperx-align-bridge.runEpubAlignOnFiles`, `transcribe-bridge.transcribeAudiobook` | `--whisper-model` (default `small`), `--language`, `--device` |
| `--coverage-align` | `coverage-align-job.runCoverageAlign` | `--language` required; **device hard-coded `'cpu'`** |
| `--crucible-*` (`cli/crucible.js:155`) | `dist/electron/crucible/servers.js` | see 8.2 |

### 8.1 Two divergences between the CLI and the app

1. **`--keep-server` and `--server` are unreachable from the Python wrapper.** `cmd_clean`
   composes only `--project/--foundry-project/--model/--ollama/--concurrency/--keep-model/
   --foundry-dist/--dry-run` (`cli/bookforge-tts.py:1075`); there is no `--keep-server`
   argparse argument. Both flags are node-adapter-only.
2. **The CLI stops the text server directly; the app hands it to the drain policy.** The
   app calls `noteTextQueueIdle(keepWarmMinutes)` in a `finally`
   (`queue-steps/foundry-job.ts:297`, `narration-clean-text.ts:590`); the CLI calls
   `stopTextServer(...)` (`clean-step.js:404`, `clean-lines-step.js:297`). So a user's
   `keepServerWarmMinutes` is ignored on the CLI path, and the `noteTextQueueBusy` bracket
   is asymmetric.

Also: `--keep-model` is an *Ollama* word (don't `keep_alive: 0` the weights) while
`--keep-server` is *BookForge's own text server* (don't pay the 110 s restart). Under
Crucible **both collapse into the server's residency policy** and neither needs to be a
client flag.

`cli/orpheus-render.js` is the one adapter that is **not** the app's path — it calls
`orpheus-worker-pool` per sentence and skips the scheduler and the batch ladder (its own
header says so).

### 8.2 The Crucible seam that already exists

`electron/crucible/servers.ts` — registry at `<userData>/crucible-servers.json`,
`{servers: [{name, url, token, added}]}`, written temp-and-rename at mode 0600.
A file that exists and does not parse is a **`corrupt_registry`** refusal telling you to
repair it by hand rather than replacing it (`:148`). `listServers()` returns a **different
type that structurally cannot carry a plaintext token** (`:84`). `addServer` refuses
`invalid_name`, `invalid_url` (including a URL ending in `/v1`, which the SDK appends
itself), `empty_token`, `duplicate_server`. `crucibleClientFor(name, clientName)` requires a
`clientName` so the shared server's log says which app queued the job.

**Nothing in the app calls it yet** — no UI, no IPC, no settings row (stated at `:18` and
confirmed by grep). `cli/crucible.js` is the only consumer, using `ping`, `info`, `health`,
`submit({type: 'echo', inputs: {…: {inline}}})`, `events`, `artifact`, `provenance`; it
exits 0 only if the bytes round-trip identically, and each of the SDK's eight error types
gets its own one-line message. `@crucible/client` is pinned in `package.json:132` to the
`v0.1.0` release tarball. `cli/README.md:1256` records a measured PC → Mac echo over the
tailnet.

---

## 9. Where the two apps do the same thing differently

The server should not have to know about any of this. It is listed so nobody tries to make
it.

| Thing | BookForge | Foundry |
|---|---|---|
| **Cleanup prompt** | `prompts/tts-cleanup-editlist.txt` + a per-chunk few-shot block built from `scanDamagedWords` | `src/clean/prompt.ts:27` — `tts-number-normalize.txt` + `"\n\n"` + `tts-narration-text.txt`, a fixed order that is itself a checked artifact |
| **Cleanup unit** | a 2000-char prose chunk, headings excluded | one block, with PREVIOUS/NEXT as read-only context |
| **Cleanup answer** | `<answer>` wrapping a JSON edit list, found by `firstJsonObject` | a JSON edit list, found by `firstJsonObject` — same idea, different envelope |
| **Translate unit** | 10 paragraphs / 5000 chars, `<<<N>>>` markers | one paragraph; batching only within one list/quote/table under 2000 chars |
| **Translate temperature** | 0.3 | 0.2 |
| **Thinking off** | Ollama `think: false`, gated on a `/api/show` capability probe | Ollama `think: false` / vLLM `chat_template_kwargs`, gated on a **regex over the model name** (`/^qwen3(\.|:|-|$)/i`) |
| **Truncation detection** | output < 70 % of input length, then a reminder retry, then split | `finish_reason`/`done_reason === 'length'`, treated as a degradation |
| **Retry on a bad answer** | reminder retry, then split, then keep the original | 3 identical attempts (translate) or 1 (clean); none at all on constrained calls |
| **JSON mode** | never used — JSON asked for in prose | `response_format: json_schema, strict: true` (vLLM) / `format: <schema>` (Ollama) on every closed question |
| **Abort** | every path chains an `AbortSignal` | **no abort member on `Transport` at all** |
| **Per-request deadline** | 180 s on cloud paths, a 300 s *inactivity* timer on Ollama streaming, none on several Ollama calls | a flat 300 s whole-request deadline everywhere |
| **Who owns the vLLM** | BookForge starts and stops it (`text-server.ts`) | Foundry never starts or stops one; `releaseModel` returns `'not-ours'` by ruling |
| **Context** | `num_ctx` estimated, bucketed to 4096, capped per model size | `num_ctx` 8192 (translate) or pinned once per book (clean); dropped entirely under vLLM |

Two prompts, two chunkers, two truncation heuristics, two retry ladders — and that is
correct. Crucible's job is to make the *transport and the residency* one thing, not the
policy.

---

## 10. What the server must be able to connect to

Ranked by what unblocks the most work for the least server change. Each row names the job
type and the field it maps to.

### Tier 0 — moves today, with zero server change

1. **BookForge clean-text and Foundry's three language acts, on the PC.** `llmServer` is
   already `vllm` and `vllmUrl` is already `http://localhost:8300/v1`. Foundry's vLLM
   transport (`src/translate/vllm.ts:401`) sends exactly what Crucible's proxy forwards:
   `model`, `messages`, `temperature`, `max_tokens`, `stream: false`, and
   `chat_template_kwargs: {enable_thinking: false}` — the one field PHASE2-LLM.md already
   promises to pass through by name. Point `vllmUrl` at Crucible, register the server, and
   `clean` / `translate` / `simplify` run. **Job type: `llm`. No new field.**
   Two preconditions, both already in the spec: `GET /v1/models` must report
   **`max_model_len`**, or Foundry's `capFor` (`vllm.ts:208`) does not clamp and an
   over-long request becomes a 400; and the served name must keep matching
   `/^qwen3(\.|:|-|$)/i` or Foundry stops sending `enable_thinking: false` and the model
   reasons before every block.
2. **BookForge's own `crucible` provider for cleanup / simplify / translate.** The seam is
   built: `AIProviderConfig` (`ai-bridge.ts:164`) and `providerConfigOf`
   (`queue-steps/ai-provider.ts:27`) are the two places a fifth provider is added, and the
   registry (`electron/crucible/servers.ts`) already resolves a named server to a client.
   The bodies need `temperature` + `max_tokens` + `enable_thinking` and nothing else.
   **Job type: `llm`. No new field.**
3. **`load-model` / `unload-model` replacing `--keep-model` and `--keep-server`.** Both
   apps already have the call site, the log line and the flag wired
   (`foundry/src/clean/runner.ts:181`, `run.ts:1450`, `cli/clean-step.js:401`). Foundry's
   `releaseModel` returns `'not-ours'` under vLLM by *ruling*, not by inability — a job that
   can actually load and unload is what that ruling was waiting for.
   **Nothing to do: the server owns residency.** The client flags should be deleted, not
   mapped.

### Tier 1 — small, named additions to `llm`

4. **`response_format: {type: 'json_schema', strict: true}` must survive the proxy.**
   Foundry's analyze verdicts and both tag calls depend on it (`vllm.ts:292`), and it is the
   only structured-output mechanism either app uses. vLLM supports it; the proxy is
   verbatim, so this should already work — but it is untested and it is the difference
   between a verdict and a degradation. **Field: `response_format`, passed through.**
5. **`/v1/models` must report `max_model_len` per model.** Named separately from row 1
   because the current `/v1/models` row shape in PHASE2-LLM.md §5 lists
   `context_default` and `memory_bytes_estimate` but not `max_model_len` — and Foundry reads
   the OpenAI field name, not Crucible's. **Field: add `max_model_len` to the OpenAI-shaped
   rows, or make `context_default` surface under that name in `/v1/openai/models`.**
6. **An honest `finish_reason`.** Analyze and tag turn `'length'` into a degradation rather
   than a wrong answer (`vllm.ts:367`, `verify.ts:349`); BookForge's audiobook analysis
   throws by name on it (`book-analysis.ts:539`). A proxy that normalises or drops it turns
   a caught truncation into silent corruption. **Nothing to add — a rule for the proxy: do
   not touch it.**
7. **A capability probe that answers "does this model think?"** BookForge's
   `probeThinkingCapability` (`ollama-capabilities.ts:30`) does `POST /api/show` and
   **throws on non-200**, so every Ollama-shaped call fails loudly against a server that
   does not answer it. Under Crucible this is not needed — the client sends
   `chat_template_kwargs` explicitly, as Foundry already does — but the BookForge Ollama
   paths must be changed to *not* probe when the provider is `crucible`. **Client-side work,
   named here so it is not discovered late.**
8. **Serving Owen's Ollama-built LoRA adapters.** The Mac has eleven
   `foundry-footnotes-*` / `foundry-ocr-*` / `headline-14b-*` / `blocks-v5-*` models that
   exist only as Ollama builds. Nothing in either app calls them from the paths audited
   here, but the footnote and rubric work in the memory notes does. A manifest-and-HF model
   story has no answer for "an adapter someone built with `ollama create`".
   **Open question for the model manifest, not a blocker for phases 3–4.**

### Tier 2 — `vlm-pages` (dots.ocr), and it is closer than it looks

9. **`vlm-pages` ≈ the `llm` proxy plus image content parts — and it drops in today with
   zero client change.** `resolveVlmRoute` (`shared/vlm/conversion.ts:535`) picks
   `endpoint` over every other route, so a Crucible exposing an OpenAI-compatible `/v1`
   serving dots.ocr is reached by typing a URL. Foundry already sends an ordinary chat
   completion whose first content part is
   `{type: 'image_url', image_url: {url: 'data:image/png;base64,…'}}`
   (`src/vlm/endpoint.ts:168`), one page per request, `temperature: 0`, `max_tokens ≤ 8192`.
   **The requirements are exact**: `/v1/models` must list precisely the id sent as
   `--vlm-endpoint-model` (`rednote-hilab/dots.ocr`); **≥ 12 concurrent pages**, because
   that is Foundry's ungated default and BookForge passes `concurrency: 0` to take it;
   32k context; and ~11.3 MP PNGs at 200 dpi per request.
   **Rasterisation stays local on every route** (PyMuPDF, `--python` mandatory), so
   Crucible receives pictures, never PDFs — and DPI and page range are **not** wire
   parameters. DESIGN.md's "send the PDF; the server rasterises" is a *different and
   larger* job type; it should not be confused with the one that unblocks Foundry now.
   **Job type: `llm` with image content parts, or `vlm-pages` defined as exactly that.**
10. **Per-page streaming results, so the bank survives.** See 4.7: a killed read must cost
    one page, not a book. Whatever shape `vlm-pages` takes, an answer must be deliverable
    per page as it lands.
11. **`vlm-page-server.ts` deletion, and the port-8000 hole with it.** 479 lines of WSL
    spawn, VRAM arithmetic, guest `pkill` and refusal messages exist only because the
    server is a local child process. Once a Crucible serves dots.ocr, this file,
    `wslVlmRefusal`, `useWsl2ForVlm`, `wslVlmCondaEnv`, `wslVlmModel`, the `wsl-server` arm
    of `resolveVlmRoute` and `foundry-app/electron/vllm-server.ts` all go — **including the
    one path in the app that reserves half the card with no arbitration at all**. Keep
    `mlx-local`: it is the Mac's only route. Also note `stopVlmPageServer`
    (`vlm-page-server.ts:469`) currently **has no caller** — the app's `before-quit` stops
    the hosted Foundry and the text server but not this one.

### Tier 3 — `tts`, the largest and the least like the rest

12. **A per-chunk `tts` request must carry three different voice shapes**: a prompt token
    (Orpheus stock/adapter), a catalog id resolving to a merged checkpoint directory
    (Higgs fine-tune), and **inline `{base64 wav, transcript, seconds}` reference clips**
    (Higgs zero-shot). The last one means the job's `inputs` must accept audio, not just
    parameters.
13. **Sampling is per-backend and not a superset.** Orpheus sends no `top_k` and no `seed`;
    Higgs on SGLang **requires** `top_k` (omitting it samples the untruncated 1026-way
    codebook tail) and seeds every chunk at `base + index`. A single `sampling` object with
    optional fields is wrong; the contract needs the backend to declare which knobs it
    honours, the way the caps already do.
14. **`max_new_tokens` is computed per chunk by an engine-specific model.** Orpheus:
    `chars / rate × 84 × 1.4`. Higgs: `int(chars/15 × 25 × 2) + 150`, clamped on SGLang to
    `4095 − promptBound`. Either the client keeps computing it (and the server must accept
    it) or the server must know the engine's chars→frames model. **Field:
    `max_new_tokens`, client-computed.**
15. **Orpheus's EOS controls are logit surgery inside the sampler, not post-hoc filters.**
    `eosBoost`, `eosBoostStart`, `eosFloor`, `eosFloorRate` (`engine/orpheus/config.py:329`)
    must be implemented by whatever serves Orpheus, or every runaway guard is lost. This is
    the single hardest thing in the `tts` contract.
16. **Per-backend cap certificates in the job contract.** `backends.served` and
    `backends.mlx` are separate blocks per voice and `higgsVoiceCapsForModel` picks by arm.
    The numbers happen to be equal today (600/800/1000/1100), which is a coincidence of the
    current catalog, not a property. A `tts` job must be able to say "this cap is for this
    (model, backend) pair" — which DESIGN.md already promises ("cap certificate is per
    (model, backend)"). **Keep that promise; it is load-bearing.**
17. **The batch door is filesystem-coupled and the streaming door is not.** The worker
    writes `<sentences_dir>/<globalIndex>.flac` directly into the session, and resume is
    "the file exists and exceeds 1024 bytes". A `tts` job returning FLAC-per-chunk over HTTP
    fits the **streaming** door almost exactly; the **batch** door needs either a shared
    mount or a new local writer that lands bytes at the paths assembly and resume already
    expect. **This is the design decision phase 3 has to make first.**
18. **Streaming needs more than per-chunk progress.** The Listen door wants sub-sentence
    PCM16 emitted *while a row is still generating* (fast start), per-row retirement out of
    order within a batch, and a `cancel` that aborts an in-flight batch. The current
    `progress {fraction, message}` + `artifact {name}` event vocabulary does not express
    any of that.
19. **The GPU-lease defect disappears.** `parallel-tts-bridge.ts:7229` can return before
    acquiring the mutex when the device resolves to `CPU`, and `:7411` gives Higgs no VRAM
    sizing at all. Moving residency to a server that owns the accelerator deletes this class
    of bug rather than fixing an instance of it.

### Tier 4 — `align`, ASR, and `rvc`

20. **`align` (Qwen3-ForcedAligner-0.6B) — the easiest job type in the list, and it
    unblocks a feature that cannot run on this PC at all.** Per chunk: 16 kHz mono float32
    audio ≤ 300 s + the exact word text + an ISO language from the 11-code table; back, one
    timestamped item per *its* tokenization. The item→word mapping, the `_normalized`
    letter-sequence equality check that refuses a model which rewrote the text
    (`aligner.py:681`), and every derived score stay in BookForge. Requirements: bf16 on
    GPU, **the model resident across a whole book** (hundreds of chunks), per-chunk failures
    reported with the run continuing, no retries, progress at chunk granularity.
    The win is not speed — it is deleting the `viaWsl` fork that **blocks whole-m4b
    alignment on Windows entirely** (`whisperx-align-bridge.ts:642`, because the library
    lives on `Z:` and WSL cannot mount it) and retiring the darwin-only `qwen-align-env`
    component alongside the Windows-only `qwenAlignEnv` env-name setting.
21. **ASR (faster-whisper) — Crucible has no job type for it, and it needs one.** This is a
    real gap, not an oversight to wave through: `generate-sentences` is a GPU queue step,
    it is the only ASR site that takes the arbiter lease, and the whole-m4b align door
    needs a rough transcript before the aligner runs. The contract: six model sizes, a
    device + `compute_type` pair (`float16` cuda / `int8` cpu), 900 s windows with 15 s
    overlap, `vad_filter` and `word_timestamps`, streaming position and cue-count progress.
    Absorbing it retires the `whisper` pip overlay, `<userData>/runtime/whisper-models`, and
    the CPU-only `whisperx-env` whose only remaining job is that rough pass.
    **Job type: new — call it `asr`, or fold it into `align` as a mode.**
22. **`rvc` (ultimate-rvc).** Inputs are a **model folder name** (not a repo), `indexRate`,
    `protectRate` (inverted scale, no-op at index rate 0), `nSemitones`, `f0Method`,
    `hopLength`, over a directory of short FLACs; `[RVC] done/total` progress; every input
    must produce an output. It runs **per sentence** in the enhancement step and **on the
    assembled file** in `convertFileRvcChunked` — both, and they are different jobs with
    differently-spelled flags on purpose. Moving it deletes the 96-file process-recycling
    wrapper (`rvc-bridge.ts:356`), which exists purely as a memory bound and costs a model
    load every 96 sentences. It has the **same filesystem coupling as `tts` batch** (it
    reads and writes a session's per-sentence FLACs), so it should be decided at the same
    time as row 17, not separately.
23. **Roformer separation and Resemble Enhance, if the `rvc` type generalises.** Denoise
    (`denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt`, hardcoded), vocals separation, and
    Resemble Enhance are three more GPU models in the same envs. **The sample-exact contract
    is load-bearing**: 44.1 kHz stereo in, identical frame count out, or the offsets manifest
    cannot re-cut ~1,400 sentences (`denoise-bridge.ts:305`). Resemble needs deterministic
    per-seed RNG, because production renders 5 seeds and takes the per-frame spectral median.

### The residency contract is where most of the value is

Rows 11, 17, 19, 20 and 22 are all instances of one thing. BookForge has **three incompatible
arbitration schemes** today: the queue's single GPU slot plus `gpuAdmission`; the in-process
arbiter mutex whose `timeoutMs` **proceeds without the lock** rather than failing
(`gpu-arbiter.ts:88`); and, for the hosted `read` path, **nothing at all** — a vLLM
reserving half the card on port 8000 with no coordination. On top of that sits
`external-gpu-job.lock`, a Windows-only convention with **no producer inside the app**,
whose deletion nothing watches (hence the 15 s admission recheck).

Crucible should expose a **"who holds the accelerator" probe**. The injection seam already
exists and is already wired: `setGpuHolderProbe` (`queue-engine.ts:1275`, wired at
`queue-ipc.ts:56`). That one call, plus load/unload on demand, retires the lock file, the
44 s reload trade, the flat-utilisation reservations, and most of `wsl-lifecycle.ts`'s
GPU-teardown ladder.

One operational request while doing it: **return the server-side traceback in error
bodies.** `recentServerLog()` (`vlm-page-server.ts:272`) is currently the only way a mid-run
engine-core crash is diagnosable, because the API stays up answering 500s whose body says
"see stack trace (above)".

### What does **not** move, in either app

Chunking (`listen-chunks.ts`, the paragraph packer, Foundry's `CHUNK_CHARS`), text
normalisation (`listen-text.ts`, `tts-punctuation.ts`, number rules, CAPS fold, glyph
strip), prompts, the edit-list validation wall, the retake ladder, assembly, the session
layout and resume rule, the ledger, the OPF stamp, and the served-model-id-as-cache-key
contract. DESIGN.md §3 already says so; this audit found nothing that argues against it.

### The two things to check before phase 3 starts

- **The served name is a cache key in Foundry** (`src/clean/run.ts:148`, and the narration
  stamp in the EPUB's OPF). Crucible substitutes its own id on the way out of the proxy by
  design (PHASE2-LLM.md §5). Decide deliberately whether `qwen3.5-9b` or `Qwen3.5-9B-bf16`
  is the name that lands in a book's stamp — **and note that a served name must say its
  dtype**, because a server cannot report one and two books cleaned at two precisions would
  otherwise be indistinguishable in their records (`text-server.ts:183`).
- **Nothing in either app streams or cancels an LLM request.** Foundry's `Transport` has no
  abort member at all; BookForge chains an `AbortSignal` everywhere. If Crucible's job model
  is the path forward for text, the cancel semantics are already specified
  (`DELETE /jobs/{id}`) — but the OpenAI proxy is the path both apps actually take, and
  there the only cancel is dropping the fetch.
