# Qwen 3.5 (0.8B, 2B, 4B, 4B 8-bit, 9B, 9B with vision): how to prompt it

Crucible ids `qwen3.5-0.8b`, `qwen3.5-2b`, `qwen3.5-4b`, `qwen3.5-4b-8bit`, `qwen3.5-9b` and `qwen3.5-9b-vl`. They come from `Qwen/Qwen3.5-0.8B` @ `2fc06364`, `Qwen/Qwen3.5-2B` @ `15852e8c`, `Qwen/Qwen3.5-4B` @ `851bf6e8` and `Qwen/Qwen3.5-9B` @ `c2022362`. The 4B 8-bit is `owenmorgan/qwen3.5-4b-w8a16` @ `84a25838`, and the 9B with vision shares the 9B's weights. The mlx-darwin and llama-windows mirrors are listed in [S8] and [S9]. They serve the `llm` chat door and the text verbs `generate` and `decide` (`clean`, `translate`, `simplify` and `analysis` were retired to the apps on 2026-10-25). `-9b-vl` and the cuda-linux and llama-windows blocks of the 0.8B, 2B and 4B also read images.

Read on 2026-10-15. The Crucible code and docs are at `5a579e7a` (2026-10-13). `qwen3.5-4b-bside` is out of scope.

---

## Sources

Labels used below:
- **OFFICIAL**: Qwen team artifacts.
- **COMMUNITY**: third parties.
- **ENGINE**: source code of the engine version Crucible runs. This is neither model guidance nor community opinion. It records what the engine does with what the app sends.
- **CRUCIBLE / MEMORY**: Owen's own records.

| # | Label | Source | Revision / date read |
|---|---|---|---|
| S1 | OFFICIAL | Qwen/Qwen3.5-9B model card, https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md | `c2022362` (repo HEAD; lastModified 2026-03-02) |
| S2 | OFFICIAL | Qwen/Qwen3.5-4B model card | `851bf6e8` (HEAD; lastModified 2026-03-02). Diffed against S1: the Quickstart and Best Practices are identical apart from the model name |
| S3 | OFFICIAL | Qwen/Qwen3.5-0.8B model card | `2fc06364` |
| S4 | OFFICIAL | Qwen/Qwen3.5-2B model card | `15852e8c`. Diffed against S3: identical apart from the name and one extra image example |
| S5 | OFFICIAL | `chat_template.jinja` of each repo at the pins above | 9B and 4B byte-identical; 0.8B and 2B byte-identical to each other |
| S6 | OFFICIAL | `preprocessor_config.json`, 9B and 0.8B | at the pins |
| S7 | OFFICIAL | HF file listing of `Qwen/Qwen3.5-9B` @ `c2022362` (API `tree`) | No `generation_config.json` in the repo |
| S8 | COMMUNITY | `mlx-community/Qwen3.5-{0.8B,4B,9B}-bf16` `chat_template.jinja` | `30675851`, `491fdc7c`, `27cdb77d`. Byte-identical to upstream |
| S9 | COMMUNITY | Chat templates embedded in `unsloth/Qwen3.5-{0.8B,2B,4B,9B}-GGUF` (HF API `?expand[]=gguf`) | `6ab46149`, `f6d5376b`, `e87f1764`, `3885219b` (all equal to the manifest pins) |
| S10 | COMMUNITY | Unsloth, "Qwen3.5 – How to Run Locally", https://unsloth.ai/docs/models/qwen3.5 | read 2026-10-15; the page notes updates of Mar 5 and Mar 17 |
| S11 | COMMUNITY | vLLM recipe, https://recipes.vllm.ai/Qwen/Qwen3.5-9B | page dated 2026-09-14 |
| S12 | COMMUNITY | QwenLM/Qwen3.8 issue #145, "Qwen3.5 series models easily fall into infinite loops…", https://github.com/QwenLM/Qwen3.8/issues/145 | closed "not planned" 2026-08-06 |
| S13 | COMMUNITY | Tylogi/TyloQuant issue #12, Qwen3.5-0.8B thinking loops, https://github.com/Tylogi/TyloQuant/issues/12 | opened 2026-09-18, open |
| S14 | COMMUNITY (ours) | `owenmorgan/qwen3.5-4b-w8a16` README + `generation_config.json` | `84a25838` |
| S15 | ENGINE | vLLM 0.29.0, installed at `~/.crucible/envs/llm` in WSL: `entrypoints/openai/chat_completion/protocol.py` L240-252, L565-590; `config/model.py` L326-332, L1680-1760; `sampling_params.py` L240-262; `transformers_utils/config.py` L1078-1104 | read 2026-10-15 |
| S16 | ENGINE | mlx-lm `v0.31.3` `mlx_lm/server.py`, https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/server.py, L540-547, L1170-1186, L1818-1845 | tag v0.31.3 |
| S17 | OFFICIAL (Qwen3, the predecessor) | Qwen docs, vLLM deployment, https://qwen.readthedocs.io/en/latest/deployment/vllm.html | read 2026-10-15; covers Qwen3, not 3.5 |
| C1 | CRUCIBLE | `docs/internals/engines-and-capability.md`: "Chat defaults", "Prefix reuse", "Prefill", "Structured output" | `5a579e7a` |
| C2 | CRUCIBLE | `docs/API.md` `POST /openai/v1/chat/completions`; `crucible/sampling.py`; `crucible/manifests.py` `DEFAULTS_KEYS` (L82-97) | `5a579e7a` |
| C3 | CRUCIBLE | `sdk/ts/README.md`, "Thinking on or off, per call" | `5a579e7a` |
| C4 | CRUCIBLE | `crucible/models/qwen3.5-*.toml` | `5a579e7a` |
| C5 | CRUCIBLE | `docs/VERB-SIZING.md` §7-8; `docs/history/PHASE22-DECIDE.md` §0, §2.4-2.5; `docs/RETRIEVAL.md`; `docs/CHANGELOG.md` | `5a579e7a` |
| M1 | MEMORY | `crucible-inference-server.md` | Owen's project notes, not published |
| M2 | MEMORY | `cleanup-n16-state.md` | |
| M3 | MEMORY | `number-normalization-model-pass.md` | |
| M4 | MEMORY | `crucible-state-2026-09-25.md` (2026-09-28 entry) | |
| M5 | MEMORY | `clean-text-model-bf16.md` | |

---

## What the model was trained to see

**Chat format (ChatML-style).** [S5] Every turn is written as `<|im_start|>{role}\n{content}<|im_end|>\n`, and the tokenizer's `eos_token` is `<|im_end|>`. The template accepts the roles `system`, `user`, `assistant` and `tool`. Any other role raises an error. Unsloth's GGUF template for 3.5 does **not** add a `developer` role; it does so only for 3.8 [S9].

**System prompt.** [S5]
- There is **no default system prompt**. If the app sends none, the template writes none.
- A system message must be the **first** message. A later one raises "System message must be at the beginning."
- A system message cannot carry images or video. The template raises an error.
- System content is `|trim`med, so leading and trailing whitespace is dropped. PHASE22 relies on this [C5].
- When `tools` are given, the template writes its own tool preamble into the system turn first, and appends the app's system text after it.

**Thinking.** [S1–S5]
- The model emits `<think>\n…</think>\n\n` and then the answer.
- The switch is the template variable `enable_thinking`. On vLLM and SGLang it travels as `chat_template_kwargs: {"enable_thinking": false}` [S1].
- **Defaults differ by size:**
  - **4B and 9B** think by default: "Qwen3.5 models operate in thinking mode by default" [S1, S2]. With `enable_thinking` unset, their template opens `<think>\n` [S5].
  - **0.8B and 2B** do not: "Qwen3.5-0.8B operates in non-thinking mode by default" [S3, S4]. Their template opens a thinking block only when `enable_thinking` is explicitly `true` [S5].
- With thinking off, every size's template pre-fills an empty `<think>\n\n</think>\n\n` after `<|im_start|>assistant\n` [S5].
- **No soft switch.** "Qwen3.5 does not officially support the soft switch of Qwen3, i.e., `/think` and `/nothink`" [S1, S3]. Do not put those tokens in prompts.
- **History.** The final output alone belongs in the history, not the thinking. The template does this itself: it keeps `reasoning_content` only for assistant turns after the last real user query. It also splits a `<think>…</think>` found inline in `content` [S1 Best Practices item 4; S5]. On Crucible, send an assistant turn back with its `content` and, within a tool loop, its `reasoning` (vLLM hands the template `reasoning` as `reasoning_content`).

**How Crucible maps this.** [C1, C2, C3, C4]
- `thinking` in the SDK becomes `chat_template_kwargs.enable_thinking` on the wire. Crucible owns only that one key in `chat_template_kwargs`.
- All six manifests set `[defaults] thinking = false`. A chat that does not state it is sent `enable_thinking: false`, and the `X-Crucible-Sampling` header reports `"thinking":"manifest"`.
- `decide` always states thinking off itself and takes no manifest defaults [C5, PHASE22 §2.4].
- Every Qwen3.5 manifest declares `[reasoning] format = "qwen3-think"`, so Crucible starts vLLM **with** `--reasoning-parser qwen3` (since the reasoning-parser change, 2026-10-29) [C1, "Reasoning"; C4]. When thinking is on, the thinking comes back as `message.reasoning` (`delta.reasoning` streamed) and `content` is the answer alone, on cuda-linux as on mlx-darwin (mlx-lm splits it itself). llama-server writes it as `reasoning_content`, which the door renames `reasoning` both ways, and Crucible's mlx-vlm server (9b-vl on the Mac) splits it as mlx-lm does [C1]. The cuda-linux split is read from vLLM 0.29.0's source, not yet measured.
- The parser takes its mode from `enable_thinking` and assumes **on** when unstated, while the 0.8B/2B templates assume off. The manifests' `thinking = false` default keeps the two agreeing; an app never sees the difference because Crucible always states it [C4].
- `thinking_token_budget` (vLLM's request field) now works on cuda-linux: Crucible passes it through, and vLLM ends the thinking after that many tokens. It is not a field on mlx-darwin or llama-server. `max_tokens` still caps thinking plus answer together; a thinking-on run that spends it all comes back with `reasoning` and no `content` on both backends [C3].

**Template mirrors differ.** Unsloth's GGUF template, used by the `llama-windows` blocks, flips the default for 4B and 9B to **non-thinking** (`enable_thinking` must be `true` to think). The same is stated in prose: "Thinking is disabled by default for all small models (0.8B, 2B, 4B, 9B)" [S9, S10]. The MLX mirrors are byte-identical to upstream [S8]. Because Crucible states `thinking` on every Qwen3.5 chat, the two defaults never decide anything in practice. An app that bypasses the manifest default could still see different behaviour per backend.

---

## Writing the prompt

**Do**
- **State thinking explicitly** for each call, even though the manifest fills it. The 0.8B/2B and 4B/9B defaults are opposite upstream [S3, S5]. Use `thinking: true` only where the task needs reasoning and you have budgeted `max_tokens` for it.
- **Put fixed material first and variable material last**: system prompt, rules and examples byte-identical across requests, and the changing input in the final user message. Crucible's engines reuse prefixes this way. On vLLM the shared part must be longer than one cache block (528 tokens on the 9B, 544 on the 0.8B) or it saves nothing. mlx-lm reuses the system segment whole. llama-server reuses everything before the last user turn [C1, "Prefix reuse"; C5 VERB-SIZING §7].
- **Images go inside the user message**, as `image_url` parts. The official examples put the image **before** the text question [S1, S3]. See "Vision input".
- **To standardise the output format, say so in the prompt.** The card's own examples are prompt sentences: a `\boxed{}` instruction for maths, and an `"answer": "C"` JSON field for multiple choice [S1 Best Practices item 3]. For guaranteed shape, use `response_format` (see "Structured output").
- **Use `prefill`** (Crucible's field) to start the answer with fixed text. It needs thinking stated off and no grammar. It must not begin or end with whitespace, because the template trims an assistant message [C1, "Prefill"].

**Don't**
- **Don't send `/think` or `/nothink`.** They are unsupported in 3.5 [S1].
- **Don't put images or video in the system message.** The template raises an error [S5].
- **Don't send a system message anywhere but first** [S5].
- **Don't feed thinking text back** in the history [S1 item 4]. On cuda-linux, where the thinking lands in `content`, strip `<think>…</think>` (and an unclosed `<think>`) before reusing a reply [C3].
- **Don't judge a thinking-on reply by empty `content`.** Judge it by `finish_reason == "length"`. On mlx a run-out gives empty `content`; on vLLM it gives partial thinking [C3].
- **Don't stack heavy few-shot blocks on the 9B in non-thinking mode.** It can reason out loud anyway (see "Measured in Crucible", M1).

---

## Settings

### Official sampling, per size and mode

All values are OFFICIAL from the model cards. `min_p=0.0` and `repetition_penalty=1.0` are the same in every row.

**4B and 9B** [S1, S2; Quickstart tip and Best Practices]

| mode / task | temperature | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| Thinking, general | 1.0 | 0.95 | 20 | 1.5 |
| Thinking, precise coding (WebDev) | 0.6 | 0.95 | 20 | 0.0 |
| **Non-thinking, general** | **0.7** | **0.8** | **20** | **1.5** |
| Non-thinking, reasoning tasks | 1.0 | 0.95 **or** 1.0 | 20 **or** 40 | 1.5 **or** 2.0 |

The card disagrees with itself on the last row. Its Quickstart tip says 1.0/0.95/20/1.5, and its Best Practices section says 1.0/1.0/40/2.0 (see "Conflicts").

**0.8B and 2B** [S3, S4]

| mode / task | temperature | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| **Non-thinking, text** | **1.0** | **1.0** | **20** | **2.0** |
| Non-thinking, vision | 0.7 | 0.8 | 20 | 1.5 |
| Thinking, text | 1.0 | 0.95 | 20 | 1.5 |
| Thinking, vision or precise coding | 0.6 | 0.95 | 20 | 0.0 |

About `presence_penalty`, the cards say to tune it "between 0 and 2 to reduce endless repetitions". They also say a higher value "may occasionally result in language mixing and a slight decrease in model performance" [S1, S3].

### Output length

- The cards recommend 32,768 output tokens "for most queries", and 81,920 for competition maths and code [S1, S3].
- The non-thinking example uses `max_tokens=32768` [S1].
- On context: "we advise maintaining a context length of at least 128K tokens to preserve thinking capabilities" [S1]. This applies to thinking mode.

### Mapped to Crucible's parameters

- **Manifest defaults.** A manifest `[defaults]` can hold only `temperature`, `top_p`, `top_k`, `max_tokens`, `repetition_penalty` and `thinking` [C2 `DEFAULTS_KEYS`]. `presence_penalty` and `min_p` are **not** manifest keys. An app that wants the official `presence_penalty` must send it per request (OpenAI's `presence_penalty`; `top_k` and `min_p` as extra body members). The Qwen3.5 manifests set only `thinking = false` [C4].
- **What fills an omitted value.** Engine defaults, not Qwen's:
  - **cuda-linux (vLLM 0.29).** vLLM loads `generation_config.json` from the repo by default (`--generation-config auto`) [S15]. The Qwen3.5 repos have none [S7]. The 4B-8bit's carries only `eos_token_id` [S14]. So vLLM's neutral defaults apply: `temperature 1.0, top_p 1.0, top_k 0 (off), min_p 0, presence_penalty 0` [S15 `sampling_params.py`].
  - **mlx-darwin (mlx-lm 0.31.3).** The server defaults are `--temp 0.0` (greedy), `--top-p 1.0`, `--top-k 0`, `--min-p 0.0`, `--max-tokens 512`. Crucible passes none of these flags [S16; C4 `engine_args`]. Also, mlx-lm applies `presence_penalty` over a `presence_context_size` of **20** tokens by default, not over the whole sequence as vLLM does [S16 L1182-1183]. The same number means a weaker penalty there.
  - **llama-windows (llama-server b10970).** Not read for this guide. See "Open questions".
- **Checking what was applied.** Read `X-Crucible-Sampling` on every reply. It names `request`, `manifest` or `engine` as the source of each key [C1].

---

## Vision input (`qwen3.5-9b-vl`, and the image-serving blocks of the 0.8B, 2B and 4B)

**What the model accepts** [S1, S3, S5]
- `image_url` or `image` parts inside a **user** message, with the text in the same message. The official examples place the image first.
- Several images per message are allowed. With the template variable `add_vision_id` set, the template labels them "Picture 1:", "Picture 2:" and so on. It is off by default.
- Video is accepted by the model, but Crucible sets `--limit-mm-per-prompt '{"image": 8, "video": 0}'`. So it serves at most **8 images and no video** per prompt on cuda-linux [C4].

**Resolution** [S6]
- Images are cut into 16-px patches with a 2×2 merge (`patch_size` 16, `merge_size` 2). One visual token therefore covers 32×32 = 1,024 pixels. This is derived from the config.
- `shortest_edge` 65,536 and `longest_edge` 16,777,216 are pixel-area bounds, about 64 to 16,384 tokens per image. This is derived from the config.
- Crucible's 9b-vl manifest records that a 640-px image is about 400 tokens on the Mac, which matches the arithmetic [C4].
- Large images are expensive. Downscale to what the task needs. This is inference, not official guidance.

**What it does well** [S1]
- The card claims early-fusion multimodal training that "outperforms Qwen3-VL models across reasoning, coding, agents, and visual understanding".
- Its examples are a geometry problem from a diagram, place recognition ("Where is this?"), and video summarisation.
- For vision, the small cards recommend lower-temperature sampling (0.7/0.8 non-thinking; 0.6/0.95 thinking) [S3].

**Crucible routing** [C4, C5]
- The 0.8B, 2B and 4B serve images on cuda-linux and llama-windows only. On mlx-darwin they are text-only, so an image decision on a Mac needs `qwen3.5-9b-vl` (mlx-vlm).
- mlx-vlm serves one request at a time and enforces **no** structured output.
- On a 24 GB PC card, `-9b-vl` leaves about 1,700 tokens of KV, so decide-with-images picks `qwen3.5-4b` there [C5 VERB-SIZING §8a].
- `decide` switches to the vision form by itself when a request carries images.

---

## Structured output

**Official.** The Qwen3.5 cards give no guidance on schema-constrained decoding. Their only format advice is a prompt sentence for benchmark answers [S1 item 3]. The Qwen3-era docs (the predecessor) point to the engine's JSON guidance *and* say to instruct the model through the system message [S17].

**What Crucible enforces** [C1 "Structured output"; C2]
- `response_format` (`json_schema` / `json_object`) is compiled with llguidance:
  - on vLLM;
  - on mlx-lm, through Crucible's patch;
  - on Crucible's llama-server build on cuda-linux;
  - on llama-windows it uses GBNF instead.
- mlx-vlm (`-9b-vl` on the Mac) enforces nothing and is refused `structured_output_not_served`.
- `"json_whitespace": "compact"` keeps whitespace out of the JSON. It is refused on llama-windows and mlx-vlm.
- **Thinking and a grammar.**
  - On vLLM and mlx-lm the grammar binds from the first generated token, so a thinking-on request just answers JSON at once and no thinking happens.
  - On llama-server (cuda-linux) a JSON constraint with thinking on is refused (`structured_output_with_thinking`).
  - In every case, **state thinking off when you constrain**.
- `prefill` and a grammar cannot be combined (`prefill_with_grammar`).

**Practical guidance.** These points combine the official notes above with the measured facts below; none of them is official.
- Describe the fields in the prompt as well as in the schema. The model sees only the prompt; the schema only masks tokens.
- Prefer `json_schema` to `json_object`.
- Keep strings capped (`maxLength`) where the content can run on.
- On the Mac with images there is no enforcement, so validate the JSON in the app.

---

## Tool calling

**Official** [S5]. When a request carries `tools`, the template opens the system turn with:
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

## Size notes

### `qwen3.5-0.8b`
- Upstream defaults to non-thinking [S3]. The card says it is intended for "prototyping, task-specific fine-tuning, and other research or development purposes" [S3].
- In thinking mode the card warns it "is more prone to entering thinking loops compared to other Qwen3.5 models". It recommends tuning sampling and streaming so a runaway can be detected and stopped [S3].
- A community report reproduces endless "Wait, …" self-correction loops with thinking on at the default sampling; non-thinking answers normally [S13].
- Official non-thinking text sampling is unusually hot: T 1.0, top_p 1.0, presence 2.0 [S3].
- In Crucible it is the floor of every text verb's lineup on an 8 GiB card without the 4B-8bit [C5 VERB-SIZING §7]. It reads images on cuda-linux (`--limit-mm-per-prompt` image 8).

### `qwen3.5-2b`
- Same guidance as the 0.8B, including the thinking-loop warning [S4].
- Measured on Foundry cleanup triage (a yes/no question per sentence with worked examples) [M2]: the **2B was near-random, AUC 0.55–0.82**, against the 9B's 0.89–0.98. Use it for decisions only where nothing larger fits.

### `qwen3.5-4b`
- Thinks by default upstream; same card guidance as the 9B [S2].
- The decide-with-images pick on a 24 GB PC card [C5].

### `qwen3.5-4b-8bit`
- Our own W8A16 quant of the 4B @ `851bf6e8`. Round-to-nearest, no calibration data. The vision tower is kept bf16 but not loaded (`--language-model-only`), so it is text only [S14; C4].
- The README says "Serve with thinking off; the chat template is the base's" [S14]. Its `generation_config.json` holds no sampling values [S14].
- Crucible's context ceiling for it is 32,768 tokens and `--max-num-seqs 4` [C4].
- **No quality measurement against bf16 is recorded.** Treat its output quality as unmeasured.

### `qwen3.5-9b` / `qwen3.5-9b-vl`
- Thinks by default upstream [S1].
- The goal model for `decide` [C5 VERB-SIZING §7]. It is Owen's ruled cleanup model (bf16) [M5].
- Measured strengths: triage AUC 0.89–0.98 [M2]. It separates relevant from junk tags as a reranker (0.87 vs 0.50) where the dedicated reranker could not [C5 RETRIEVAL].
- Measured weaknesses: see M1 and M3 below.
- `-9b-vl` shares the weights. Every note here applies, plus "Vision input".

---

## Community findings

- **Unsloth** [S10]:
  - Toggle thinking in llama-server with `--chat-template-kwargs '{"enable_thinking":true}'`.
  - Their sampling table copies the card's Quickstart tip, including 1.0/0.95/20/1.5 for "non-thinking, reasoning".
  - "Gibberish" output may mean too short a context; they suggest `--cache-type-k bf16 --cache-type-v bf16`.
  - Their GGUF templates carry tool-calling fixes. The embedded template only iterates `tool_call.arguments` when it is a mapping [S9].
- **vLLM recipe** [S11]:
  - Disable thinking server-wide with `--default-chat-template-kwargs '{"enable_thinking": false}'`.
  - Use `--reasoning-parser qwen3` to separate thinking.
  - MTP speculative decoding is available.
  - "Reduce `--max-cudagraph-capture-size`" if Mamba cache errors appear.
  - Crucible does none of these. It states thinking per request through the manifest instead.
- **Reasoning loops on larger 3.5 and 3.6 models** [S12]:
  - Users report circular thinking at the recommended sampling.
  - A QwenLM contributor (jklj077) replied that loops occurring regardless of prompt suggest trying a different inference framework, since "this behavior is not normal".
  - Closed "not planned"; no fix recorded.
- **0.8B** [S13]: thinking-mode "Wait, I should not use any markdown"-style loops until `finish_reason=length` at T 1.0 / top_p 0.95 / top_k 20. The reporter's suggestions were a lower temperature, a larger budget, or repetition detection. Open, unresolved.

---

## Measured in Crucible

Only facts from Crucible's docs and changelog and from Owen's memory notes.

- **Thinking must be stated off for mechanical work** [M1 L244].
  - The record: "Qwen3.5 is a REASONING model: send `chat_template_kwargs.enable_thinking=false` (SDK `thinking:false`) for cleanup or 64 tokens all go to reasoning."
  - This is why every Qwen3.5 manifest has `thinking = false` [C4].
- **Thinking out loud with thinking off** [M1 L255-257].
  - The setup: `qwen3.5-9b` with the BookForge edit-list prompt, which carries a few-shot block.
  - The result: it "runs away thinking OUT LOUD as `content` even with thinking:false" — 6,144 tokens in 214 s, against 15 tokens in 3 s without the few-shot. It was caught by `finish_reason=length`.
  - Recorded as "NOT a production cleanup pairing yet".
- **Thinking-on run-out shape** [C3].
  - mlx-darwin: a run-out leaves `content` empty (measured on the 27B; see the 3.8 guide).
  - cuda-linux: expected to leave partial thinking in `content` (read from source, not measured).
  - Fail on `finish_reason`.
- **2B vs 9B as a yes/no judge** [M2]: 2B AUC 0.55–0.82 (near-random) vs 9B AUC 0.89–0.98 on cleanup triage.
- **9B on number normalisation** [M3].
  - First real run, `qwen3.5:9b` Q4_K_M via Ollama, not through Crucible. The 9B was wrong on chapter:verse references and said "hyphen"/"colon" literally.
  - This led to deterministic rules for those classes. A later `qwen3.5:9b-q8_0` run had one wrong model edit out of 14.
- **Decide: one prompt with many slots was rejected** [M4]. Asking many items in one prompt gave 25–60% argmax agreement, and the model copied filler. The shipped `items` form reads the shared state once and continues each item alone.
- **Decide (snap) on the 9B Q8_0** [C5 PHASE22 §0]: about 40 ms of prefill for a 110-token state, and "does this document say X" traps held (p ≈ 0.003). Measured 2026-09-22.
- **Prefix-reuse block size** [C1, C5]: 528 tokens (9B) and 544 (0.8B) on vLLM, measured.
- **Template trims an assistant message** [C1 "Prefill"]: `{"answer": ` is continued as `{"answer":`. Rendered with the 9B tokenizer, 2026-10-10.

---

## Conflicts and open questions

**Conflicts between official guidance and what Crucible does now**

1. **Default sampling.**
   - Official non-thinking (4B/9B): T 0.7, top_p 0.8, top_k 20, presence 1.5. Official non-thinking (0.8B/2B): T 1.0, top_p 1.0, top_k 20, presence 2.0.
   - Crucible's Qwen3.5 manifests set no sampling, so an app that omits it gets the engine's defaults:
     - **vLLM**: T 1.0, top_p 1.0, **no top_k**, no presence penalty (no `generation_config.json` in the repos).
     - **mlx-lm**: **greedy (T 0)** and **`max_tokens` 512**.
   - So the same request samples differently on the PC and the Mac, and neither matches Qwen. `presence_penalty` cannot be set as a manifest default at all.
   - [S1, S3, S7, S15, S16, C2, C4]
2. **Output length.**
   - Official: 32,768 tokens for most queries.
   - Crucible: no `max_tokens` default. Unsent, vLLM's limit is the remaining context, and mlx-lm stops at 512.
   - [S1, S16, C4]
3. **Context for thinking.**
   - Official: at least 128K to preserve thinking capability.
   - Crucible: `context_default` 8,192–16,384, `max_context` 32,768–131,072 [S1, C4].
   - This matters only when an app turns thinking on.
4. **Reasoning parser.**
   - Official serving examples (4B/9B) start vLLM and SGLang with `--reasoning-parser qwen3`.
   - Crucible now runs vLLM with it too (`[reasoning]` in the manifest), so thinking comes back as `reasoning` and `thinking_token_budget` works [S1, C1, C3].
   - One consequence differs from mlx-darwin: with thinking on, a `response_format` grammar waits for `</think>` on vLLM, so the model thinks and then answers the JSON in `content`; mlx-lm binds the grammar from the first token and answers at once. Thinking off, both answer at once.

**Conflicts within the sources**

5. **The 4B/9B card contradicts itself** on "non-thinking, reasoning tasks". The Quickstart tip says 1.0/0.95/20/1.5; Best Practices says 1.0/1.0/40/2.0. Unsloth repeats the tip version [S1, S10].
6. **Thinking defaults.**
   - Upstream: 4B/9B think by default. Unsloth's GGUF template and guide: all four sizes are non-thinking by default [S5, S9, S10].
   - Crucible always states it, so this is moot unless a chat bypasses the manifest.
7. **The 0.8B/2B card's thinking example** puts `"enable_thinking": True` at the top level of `extra_body`. Its own note says vLLM and SGLang need it inside `chat_template_kwargs`; the top-level form is Alibaba Cloud's API [S3].

**Open questions**

- llama-server b10970's default sampling and `n_predict` for `llama-windows` were not read for this guide. Check `tools/server` at b10970 before stating them.
- Does the 4B-8bit (RTN W8A16, no calibration) lose measurable quality against bf16 on our verbs? Nothing recorded.
- Is mlx-lm's 20-token `presence_context_size` window what Qwen's `presence_penalty=1.5` assumes? Qwen's numbers are presumably tuned on vLLM/SGLang semantics. This is unverified.
- Should Crucible's manifests carry Qwen's non-thinking `temperature`/`top_p`/`top_k`? This is Owen's call. The data is above.
- Does the 9B's out-loud reasoning under a few-shot prompt [M1] also happen at the official non-thinking sampling? It was measured at whatever that run sent, which is not recorded.
