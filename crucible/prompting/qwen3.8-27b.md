# Qwen 3.8 27B (4-bit, 4-bit with vision, 8-bit): how to prompt it

Crucible ids `qwen3.8-27b-4bit`, `qwen3.8-27b-4bit-vl` and `qwen3.8-27b-8bit`. The base model is `Qwen/Qwen3.8-27B` @ `1d4bf0f2`; Crucible does not pin this repo, it serves quantized copies made from it:

| backend | repo @ revision | ids |
|---|---|---|
| cuda-linux (vLLM) | `avyukth/Qwen3.8-27B-AWQ-INT4` @ `5a2ee524` | 4bit, 4bit-vl |
| mlx-darwin (mlx-lm) | `mlx-community/Qwen3.8-27B-4bit` @ `3e6447f0` | 4bit |
| mlx-darwin (mlx-lm) | `mlx-community/Qwen3.8-27B-8bit` @ `815b83c0` | 8bit, Mac only |
| llama-windows (llama-server) | `unsloth/Qwen3.8-27B-GGUF` @ `4ca72078` (`UD-Q4_K_M` + `mmproj-F16`) | 4bit, 4bit-vl |

They serve the `llm` chat door and are the goal model (27B) for `generate`, `translate`, `simplify` and `analysis`.

Read on 2026-10-15. The Crucible code and docs are at `5a579e7a` (2026-10-13).

---

## Sources

Labels are as in `qwen3.5.md`: OFFICIAL is the Qwen team, COMMUNITY is third parties, ENGINE is source code of the engine version Crucible runs, and CRUCIBLE / MEMORY are Owen's records.

| # | Label | Source | Revision / date read |
|---|---|---|---|
| T1 | OFFICIAL | Qwen/Qwen3.8-27B model card, https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md | `1d4bf0f2` (HEAD; created 2026-08-05, lastModified 2026-08-14) |
| T2 | OFFICIAL | `chat_template.jinja` @ `1d4bf0f2` | Byte-identical in the AWQ @ `5a2ee524`, MLX 4-bit @ `3e6447f0` and MLX 8-bit @ `815b83c0` repos (diffed) |
| T3 | OFFICIAL | `generation_config.json` @ `1d4bf0f2`: `temperature 1.0, top_p 0.95, top_k 20`, EOS `[248046, 248044]` | Same values in the AWQ and MLX repos |
| T4 | OFFICIAL | `preprocessor_config.json` @ `1d4bf0f2` | Identical to Qwen3.5-9B's |
| T5 | COMMUNITY | `avyukth/Qwen3.8-27B-AWQ-INT4` README + `recipe.yaml`; its `tokenizer.json` checked for `truncation` (null) | `5a2ee524` |
| T6 | COMMUNITY | `mlx-community/Qwen3.8-27B-4bit` and `-8bit` README + `config.json` (`quantization: {group_size 64, bits 4/8, mode affine}`, converted with mlx-vlm 0.6.8) | `3e6447f0` / `815b83c0`. The 4-bit repo's HEAD is now `10c35caa`, so Crucible pins an older revision |
| T7 | COMMUNITY | `unsloth/Qwen3.8-27B-GGUF` README + embedded chat template (HF API `?expand[]=gguf`) | `4ca72078` |
| T8 | COMMUNITY | Unsloth, "Qwen3.8 – How to Run", https://unsloth.ai/docs/models/qwen3.8 | read 2026-10-15 |
| T9 | COMMUNITY | vLLM recipe, https://recipes.vllm.ai/Qwen/Qwen3.8-27B | page "Updated 2026-10-15" |
| T10 | COMMUNITY | HF discussion #178, "Qwen 3.8 get stuck in massive 20k–50k token thinking loops", https://huggingface.co/Qwen/Qwen3.8-27B/discussions/178 | opened Aug 27, latest post early Oct 2026 |
| T11 | COMMUNITY | HF discussion #76, "Why Qwen3.8-27B overthinks?", https://huggingface.co/Qwen/Qwen3.8-27B/discussions/76 | mid-Aug 2026 |
| T12 | COMMUNITY | QwenLM/Qwen3.8 issue #217, template rejects `reasoning_effort="high"` → HTTP 500 via vLLM, https://github.com/QwenLM/Qwen3.8/issues/217 | opened 2026-08-19, closed "not planned" 2026-10-03 |
| T13 | COMMUNITY | Rapid-MLX issue #2474, `chat_template_kwargs.reasoning_effort` dropped, https://github.com/raullenchai/Rapid-MLX/issues/2474 | 2026-08-27, fixed 08-29 |
| T14 | COMMUNITY | QwenLM/Qwen3.8 issue #145, reasoning loops incl. Qwen3.6-27B-FP8, https://github.com/QwenLM/Qwen3.8/issues/145 | closed "not planned" 2026-08-06 |
| T15 | ENGINE | vLLM 0.29.0 (`~/.crucible/envs/llm` in WSL): `chat_completion/protocol.py` L240-252, L565-590; `config/model.py` L326-332, L1680-1760; `sampling_params.py` L240-262 | read 2026-10-15 |
| T16 | ENGINE | mlx-lm `v0.31.3` `mlx_lm/server.py` L540-547, L1170-1186, L1818-1845 | tag v0.31.3 |
| C1 | CRUCIBLE | `docs/internals/engines-and-capability.md` ("Chat defaults", "Prefix reuse", "Prefill", "Structured output") | `5a579e7a` |
| C2 | CRUCIBLE | `docs/API.md` chat completions (incl. `reasoning_effort` row); `crucible/sampling.py`; `crucible/manifests.py` `DEFAULTS_KEYS` | `5a579e7a` |
| C3 | CRUCIBLE | `sdk/ts/README.md` "Thinking on or off, per call" | `5a579e7a` |
| C4 | CRUCIBLE | `crucible/models/qwen3.8-27b-*.toml` | `5a579e7a` |
| C5 | CRUCIBLE | `docs/MODEL-CHOICE.md` §8; `docs/VERB-SIZING.md`; `docs/MEASUREMENTS.md`; `docs/CHANGELOG.md` (1.0.40; the Mac prefill-step entry) | `5a579e7a` |
| M1 | MEMORY | `crucible-inference-server.md` (the 27B manifest entry, 2026-09-13) | |
| M2 | MEMORY | `crucible-state-2026-10-10-pm.md`, `crucible-state-2026-10-01.md` (Mac 27B stutter) | |
| M3 | MEMORY | `crucible-state-2026-09-25.md` (the 27B thinking-on title/scrub calls) | |

---

## What the model was trained to see

**Chat format.** Qwen3.8 is the Qwen3.5 architecture (`model_type: qwen3_5`) [T5]. It uses the same ChatML turns, `<|im_end|>` EOS, and tool, system and vision rules as Qwen3.5 [T2]:
- no default system prompt;
- the system message must come first;
- no images in the system message;
- system content is trimmed.

The differences from 3.5 are in thinking, below.

**Thinking: on by default, with three template variables** [T1, T2]
- **`enable_thinking`**
  - Unset or `true`: the assistant turn opens `<think>\n`.
  - `false`: the template pre-fills an empty `<think>\n\n</think>\n\n`.
  - In the card's words: "Qwen3.8 models operate in thinking mode by default."
- **`reasoning_effort`**
  - Accepted values are `xhigh` (the default), `medium` and `low`. Any other value raises "Unexpected reasoning effort …" [T2].
  - It works by **writing an instruction into the system turn**, and only when thinking is on:
    - `xhigh` prepends "Reasoning effort is set to xhigh. Please think carefully…";
    - `low` prepends "…Keep your thinking brief and focused…";
    - `medium` adds **nothing**.
  - If the app sends no system message, the template creates one holding just that instruction [T2].
  - So effort is a prompt, not a budget, and it changes the start of the prompt.
- **`preserve_thinking`**
  - Unset or `true` by default.
  - It renders the `reasoning_content` of **every** past assistant turn as `<think>…</think>`, not just those after the last user query [T1, T2].
  - The card says this improves "KV cache utilization" and agent consistency.
  - `false` restores the 3.5 behaviour of keeping only the thinking after the last user message [T1].

**History handling changed from 3.5** [T2, diffed against the 3.5 template]
- The 3.8 template **no longer splits a `<think>…</think>` found inside `content`**. It reads thinking only from `reasoning_content`.
- The card's own example appends the answer as `content` and the thinking as `reasoning_content` [T1].

**How Crucible maps this**
- **No `[defaults]`.** None of the three Qwen3.8 manifests has a `[defaults]` table, so **thinking is on unless the app states it off**, at `xhigh` effort [C4, T2].
- **`reasoning_effort` on cuda-linux.** OpenAI's top-level `reasoning_effort` is forwarded to the local engine as sent [C2]. vLLM 0.29.0 passes it into the chat template as `reasoning_effort`. If the app did not put `enable_thinking` in `chat_template_kwargs`, vLLM also sets `enable_thinking = (reasoning_effort != "none")` [T15 protocol.py L575-588]. So against Qwen3.8 on cuda-linux:
  - `low` or `medium` works;
  - `none` turns thinking off;
  - `high`, `minimal` or `max` makes the template raise, and vLLM returns an error [T12].
- **`reasoning_effort` on mlx-darwin.** mlx-lm 0.31.3 reads `chat_template_kwargs` and passes every key to the template [T16 L540-547], but has no top-level `reasoning_effort` field [T16 L1170-1192]. Send it as `chat_template_kwargs: {"reasoning_effort": "low"}`. This is not measured.
- **No reasoning parser.** Crucible starts vLLM without `--reasoning-parser` [C1]. On cuda-linux, thinking therefore comes back inside `content`, and **no `reasoning_content` is returned**. The template will not strip that inline thinking from later turns (above). An app holding a multi-turn chat on the PC must strip it itself, or send it back as `reasoning_content`.
- **Thinking budget.** `max_tokens` is the only one [C3].

---

## Writing the prompt

**Do**
- **Decide thinking per call and state it.** For extraction, cleanup, translation of a sentence, or a classification, send `thinking: false`. Thinking on means at least the `xhigh` system instruction and an unbounded trace (see "Measured in Crucible").
- **If you want thinking but shorter, send `reasoning_effort: "low"`.** On the Mac send it inside `chat_template_kwargs`. Accept that it is a prompt hint, not a cap; budget `max_tokens` anyway.
  - The card warns that in multi-turn agentic work, lower effort "may" lead to insufficient analysis and retries [T1].
  - The community reports it failing to bound runaway thinking on large prompts [T10].
- **Keep the effort level constant across calls that should share a prefix cache.** It is written into the system turn, so changing it changes the prefix. `medium` writes nothing.
  - This follows from T2 and C1 together; it is not official guidance.
- **Put fixed material first and the changing input last, in the final user message.** On vLLM the 27B's cache blocks are 784 tokens, so a shared prefix shorter than that buys nothing [C1].
- **Images go in the user message, before the text question,** as in the card's examples [T1]. See "Vision input".

**Don't**
- **Don't send `reasoning_effort` of `high`, `max` or `minimal`.** The upstream template raises [T2, T12]. Unsloth's GGUF template maps `high` to `xhigh` [T7]; the upstream copies served on cuda-linux and the Mac do not.
- **Don't rely on `/think` or `/nothink`.** These are not documented for 3.8. The 3.5 cards explicitly drop them [qwen3.5.md S1].
- **Don't send inline `<think>` text back in `content`** (see above).
- **Don't put images in the system message** [T2].

---

## Settings

### Official sampling [T1 Quickstart tip and Best Practices; T3]

| mode | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|---|---|---|---|---|---|---|
| Thinking | 1.0 | 0.95 | 20 | 0.0 | **0.0** | 1.0 |
| **Non-thinking** | **0.7** | **0.80** | **20** | 0.0 | **1.5** | 1.0 |

- **Change from 3.5.** In thinking mode `presence_penalty` is **0.0**; 3.5 recommends 1.5 [T1 vs qwen3.5.md]. The card still allows 0–2 "to reduce endless repetition", with the same language-mixing caveat [T1].
- **The repo's `generation_config.json`** carries only the thinking-mode trio (`1.0 / 0.95 / 20`) [T3].

### Output length [T1]

The card suggests up to 262,144 tokens of reasoning and 131,072 of final response, "for frameworks that support separate token limits". It frames these as agentic-task budgets "within the 1M context length". Neither applies to Crucible: its 27B contexts are 16K–98K and it has no separate thinking budget [C3, C4].

### Mapped to Crucible

- **Manifest defaults.** None are set [C4]. `presence_penalty` and `min_p` cannot be manifest defaults at all [C2]; send them per request.
- **cuda-linux (vLLM, AWQ).**
  - vLLM loads the repo's `generation_config.json` by default [T15 model.py L326]. The AWQ repo has one [T3].
  - So an omitted `temperature/top_p/top_k` becomes **1.0 / 0.95 / 20**, the official **thinking** trio, even when the app has turned thinking off.
  - `presence_penalty` defaults to 0 [T15].
- **mlx-darwin.** mlx-lm 0.31.3's server does not read `generation_config.json` for request defaults. It uses its flags, which Crucible leaves at the defaults [T16; C4]:
  - **T 0.0 (greedy)**, top_p 1.0, top_k 0;
  - **`max_tokens` 512** when unsent;
  - `presence_penalty` over a 20-token window only.
- **llama-windows.** Not read; see "Open questions".
- **Checking what was applied.** `X-Crucible-Sampling` names the source of each value [C1].

---

## Vision input (`qwen3.8-27b-4bit-vl`)

- **Same vision path as Qwen3.5** [T4]: identical `preprocessor_config.json` (16-px patches, 2×2 merge, so about 1,024 px per token; 64–16,384 tokens per image).
- **What the card shows** [T1]: diagram maths, place recognition, and video at `fps=2`. It claims understanding "from STEM diagrams and documents to hour-scale videos".
- **The vision tower is not quantized in the AWQ copy.** It stays bf16 (0.92 GB) because its intermediate size does not divide into 128-groups [T5]. So image *encoding* is full precision on the PC, while the language model reading the encoded image is INT4.
- **The AWQ README's vision check.** It reports "read rendered text: exact" and "identify shape / colour / position: correct". These are spot checks, not a benchmark [T5].
- **Crucible limits** [C4]:
  - 8 images, no video, per prompt on cuda-linux (`--limit-mm-per-prompt`);
  - no mlx-darwin vision block for the 27B;
  - "A 24 GB card cannot hold its PC form at all; a Windows machine without WSL can" (llama-windows with `mmproj-F16`).

---

## Structured output

**Official.** The card says nothing about schema-constrained decoding [T1].

**What Crucible enforces** [C1] is the same as for Qwen3.5:
- llguidance on vLLM and on mlx-lm (patched);
- GBNF on llama-windows;
- `json_whitespace: compact` everywhere except llama-windows.

**Thinking matters more here**, because thinking is the 27B's default:
- On vLLM and mlx-lm a grammar binds from the first token, so a thinking-on JSON request simply produces JSON with no reasoning. The template has already opened `<think>\n` and, when thinking is on, injected the `xhigh` system instruction. Whether that degrades the JSON is unmeasured.
- On llama-server (cuda-linux) the same combination is refused (`structured_output_with_thinking`).
- **State `thinking: false` with every schema.** This is a recommendation built on the C1 facts, not official guidance.

---

## Tool calling

**Official** [T2]. When a request carries `tools`, the template opens the system turn with:
- `# Tools`, then each tool's JSON inside `<tools>…</tools>`.
- An instruction to reply ONLY in the XML call form when it calls a function:
  `<tool_call>
<function=NAME>
<parameter=P>
V
</parameter>
</function>
</tool_call>`.
- These rules: required parameters must be given, any reasoning comes BEFORE the call and never after it, and the model answers normally when no tool fits.

The app's own system message is appended after that block. A `role: "tool"` result is rendered into a user turn, inside `<tool_response>…</tool_response>`. This is the Qwen3-Coder XML form, not Qwen 2.5/3's Hermes JSON.

**What Crucible does** [C1 "Tool calling"]
- The manifest declares `[tools] format = "qwen3-xml"`.
- Each engine reads the model's calls back as OpenAI `tool_calls`:
  - vLLM through `qwen3_coder`;
  - llama-server through `--jinja`;
  - mlx-lm through its tokenizer's parser.
- Send OpenAI's shape and read OpenAI's shape back. Never write `<tool_call>` into a prompt yourself.
- `tools` beside a `response_format` is refused (`tools_with_response_format`).

**Practical guidance.** None of this is official.
- Write each tool's `description`, and each parameter's, as instructions to the model. The schema is all it sees of the tool.
- Mark the required parameters `required`. The template tells the model they must be given.
- Validate `arguments` in the app before running anything. It is the model's JSON, and the model can get it wrong.

---

## Size and quantization notes

All three forms are the same 27B. Their differences are quantization, engine and cache behaviour.

### `qwen3.8-27b-4bit`, cuda-linux: AWQ INT4 (`avyukth`) [T5]

- **How it was quantized**:
  - W4A16, symmetric, group size 128;
  - `lm_head` and the MTP head kept bf16;
  - calibrated on 128 × 512 tokens of `ultrachat_200k` (general chat).
- **Caveat from the quantizer**: "48 of 64 layers received INT4 without activation-aware scaling". The Gated DeltaNet (linear-attention) layers could not be AWQ-replayed, so they are round-to-nearest. Only the 16 full-attention layers and all MLPs got AWQ scaling.
- **Their evidence**: GSM8K at temperature 0 was 39/40 on **40 problems only**. "No bf16 side-by-side was run, so the exact delta from the base model is unmeasured."
- **Tokenizer**: the quantizer warns that a calibration tokenizer can bake in `truncation: max_length 512`. This repo's `tokenizer.json` has `truncation: null` (checked 2026-10-15), so prompts are not clipped.
- **Crucible's form** [C4]: 16,384-token default context, 32,768 max, text-only (`--language-model-only`). Weights are about 18.06 GB measured.
- **Choice history**: a different AWQ (cyankiwi's, 21 GB) did not fit 24 GB [M1].

### `qwen3.8-27b-4bit`, mlx-darwin: MLX 4-bit (`mlx-community`) [T6]

- Affine quantization, group size 64, 4 bits, converted with mlx-vlm 0.6.8. No quality figures are published.
- Crucible's context default is 98,304 [C4].
- The pinned revision `3e6447f0` is behind the repo's HEAD `10c35caa`. What changed upstream was not read.

### `qwen3.8-27b-8bit`, mlx-darwin only: MLX 8-bit (`mlx-community`) [T6]

- Affine, group size 64, 8 bits.
- Described as "the best translation a 64 GB Mac can hold" [C4]. Its weights are about 29.5 GB, so it is Mac only; the FP8 cuda arm was removed on Owen's ruling [C5 MODEL-CHOICE §8].
- MEASUREMENTS lists its whole manifest as "written 2026-09-17, never loaded" [C5]. Later stutter work ran it [M2], so that line is stale.
- Context default is 12,288 [C4].

### llama-windows: Unsloth `UD-Q4_K_M` GGUF [T7, T8]

- Unsloth's dynamic quant. Their own claim: ">10% top-1% better accuracy" than other providers, not independently checked.
- Its **embedded template differs from upstream**:
  - it adds a `developer` role;
  - it merges consecutive leading system/developer messages;
  - it maps `reasoning_effort: "high"` to `xhigh`;
  - it adds stricter tool-call argument checks.

---

## Community findings

- **Runaway thinking** [T10]
  - Reported on SGLang bf16, Unsloth Q4_K_XL GGUF and MLX 4-bit: 20k–55k+ thinking tokens with no answer, and hangs of 45–80 minutes.
  - Reported as **working**: a hard thinking budget (2,048–16,384) together with T 0.6 / top_p 0.95; and making sure `top_k=20` is actually applied (SGLang's default is off).
  - Reported as **failing**: `reasoning_effort: "low"` on large prompts.
  - No Qwen team reply in the thread.
- **Unsloth** [T8]
  - Same sampling table as the card.
  - Its tip to disable thinking in llama.cpp with `--chat-template-kwargs '{"reasoning_effort":"none"}'` **conflicts with its own GGUF template**, which raises on any effort outside `xhigh/medium/low/high` while thinking is on [T7].
  - It recommends 4-bit GGUFs (16–19 GB) for 24 GB cards.
- **vLLM recipe** [T9]
  - `--reasoning-parser qwen3`, described as "required in practice".
  - `enable_thinking` and `reasoning_effort` set through `chat_template_kwargs`, or server-wide with `--default-chat-template-kwargs`.
  - It lists official FP8 (`Qwen/Qwen3.8-27B-FP8`) and NVFP4/MXFP4 variants.
- **`reasoning_effort` plumbing** [T12, T13]
  - Clients that send `high` (Claude Code via vLLM's Anthropic endpoint) get a 500 from the upstream template. The Qwen team did not change the template; the issue was closed "not planned".
  - One MLX server (Rapid-MLX) dropped `chat_template_kwargs.reasoning_effort`, which pinned the model to `xhigh`. Fixed there; it is not the engine Crucible runs.
- **Overthinking blamed on weights** [T11]
  - A user claims scale misalignment in SSM conv layers 52–62 and published a "fixed" bf16.
  - No benchmarks or Qwen response. **Unverified; do not act on it.**
- **Loops in the related 3.5/3.6 models** [T14]
  - Includes Qwen3.6-27B-FP8.
  - A QwenLM contributor called framework-independent loops "not normal" and suggested changing the inference framework. Closed "not planned".

---

## Measured in Crucible

Only facts from Crucible's docs and changelog and from Owen's memory notes.

- **Thinking-on run-out on the Mac** [C3]
  - ContentStudio, 2026-09-25: `qwen3.8-27b-4bit`, `maxTokens` 8192.
  - 4 of 9 chapter-title calls ran out after 405–412 s with **0 characters of content**.
  - The 5 that finished returned 325–351 characters with no think text.
  - Hence: fail on `finish_reason === 'length'`, never on empty content.
- **Long thinking crashed the Mac's 27B** with a `metal::malloc` resource limit. Fixed by an mlx-lm patch in Crucible 1.0.40 [C5 CHANGELOG].
- **Desktop stutter on the Mac 8-bit** [C5 CHANGELOG; M2]
  - mlx-lm read prompts 2,048 tokens per GPU step, about 12–13 s each on the 27B-8bit, and froze the window server for up to 13.2 s.
  - The prefill step is now derived from the model size (27B: 256).
  - This is serving, not prompting, but long prompts are what trigger it.
- **Memory at long context** [M1; C5 MEASUREMENTS]
  - On the Mac, the 27B 4-bit peaked at 31.55 GiB at 98,220 tokens, against a computed 22.5 GB. The computed figure was 34% light because mlx-lm's prefill logits cover a 248k vocabulary.
  - So a very long prompt costs real memory beyond the KV.
- **The 27B's thinking-on title/scrub calls** were given a 16k budget [M3].
- **Verb sizing** [C5 VERB-SIZING]: the 27B is the goal for generate, translate, simplify and analysis. `decide` moved off the 27B to the 9B on 2026-10-09.

---

## Conflicts and open questions

**Conflicts between official guidance and what Crucible does now**

1. **Thinking default.**
   - Crucible's Qwen3.8 manifests set no `thinking`, so the 27B **thinks by default at `xhigh`** for every verb whose app does not state otherwise.
   - The Qwen3.5 manifests all state `thinking = false`.
   - Upstream behaviour is honoured, but the two families behave oppositely through the same door, and translate/simplify on the 27B pay a thinking trace unless the app opts out. [C4, T1, T2]
2. **Default sampling.**
   - Official non-thinking: 0.7 / 0.8 / 20 / presence 1.5.
   - What an app that sends nothing gets:
     - **vLLM**: 1.0 / 0.95 / 20 / presence 0, the thinking set, from `generation_config.json`, **even with thinking off**;
     - **mlx-lm**: **greedy** with **`max_tokens` 512**.
   - PC and Mac therefore differ, and neither matches non-thinking guidance. `presence_penalty` cannot be a manifest default.
   - [T1, T3, T15, T16, C2, C4]
3. **Reasoning parser.** Official and community serving use `--reasoning-parser qwen3` [T1 via the recipe, T9]. Crucible runs without it, so on the PC thinking lands in `content`, and the 3.8 template no longer strips inline thinking from history [T2, C1].
4. **Thinking budget.** The card assumes separate reasoning and answer limits (262k / 131k) [T1]. Crucible's only budget is `max_tokens`, and the community's one effective fix for runaways is a hard thinking budget [T10, C3].

**Conflicts within the sources**

5. Unsloth's "`reasoning_effort: none`" advice versus its own template, which raises [T7, T8].
6. Unsloth's GGUF template maps `high` to `xhigh`; the upstream template served on cuda-linux and the Mac raises on `high` [T2, T7]. **The same request therefore succeeds on llama-windows and fails on the PC or Mac.**

**Open questions**

- On cuda-linux, when an app sends a top-level `reasoning_effort` but no `chat_template_kwargs`, Crucible adds nothing for Qwen3.8 (no manifest `thinking`), so vLLM's mapping applies. Has anyone sent `reasoning_effort: "low"` through Crucible to a 27B? It is not measured.
- Does a JSON-constrained call with thinking left on (`xhigh` injected) degrade answers compared with thinking stated off? It is unmeasured on either engine.
- Quality of the AWQ INT4 (RTN on 48 of 64 layers) against MLX 4-bit and the GGUF on our verbs: no side-by-side is recorded.
- llama-server b10970's default sampling and `n_predict` for llama-windows were not read.
- What changed in `mlx-community/Qwen3.8-27B-4bit` between `3e6447f0` (pinned) and `10c35caa` (HEAD)? It was not diffed.
- Should the Qwen3.8 manifests state `thinking = false`, or carry the non-thinking sampling? This is Owen's call; the data is above.
