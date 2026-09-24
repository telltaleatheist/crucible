# PHASE22 — the decision door: `POST /v1/decide`

> *"lets work snap into crucible as another service"* (Owen, 2026-09-23). And, from the
> Foundry tile that ran snap beside Crucible on 2026-09-22: *"AN EXPERIMENT, run outside
> Crucible on purpose … If it earns its place it becomes a Crucible job."* It did, and this
> is that.

**This document is the CONTRACT.** Everything below is what the door does, stated before it
is built; where a ruling is still Owen's it says so in section 9.

> **BUILD STATE (2026-09-23): SERVER BUILT on `feat/decide-door`, NOT CUT, NOT RUN ON A
> CARD.** `crucible/decide.py` (the reading), `POST /v1/decide` in `crucible/api.py`, the
> engine record (`decide_logprobs` / `max_logprobs` / `decide_basis` on each engine class,
> read through `engines.decide_reading`), vLLM's three flags composed in
> `Residency._engine_args`, `LlamaServerEngine.chat_concurrency = 1`, and
> `tests/fake_engine.py` answering chat logprobs in vLLM 0.29.0's shape. Tests:
> `tests/test_decide_core.py` (48, snap's pure tests ported), `tests/test_decide_api.py`
> (23, the door end to end through the fake), 6 new in `tests/test_chat_admission.py`; a
> mutation run (letters matched loosely, the fan-out ungated, the prime skipped) fails 7 of
> them. No GPU was used: every test runs against the fake. The SDK/CLI/docs half is a
> separate agent's. The live pass on the card and the Mac is owed and listed in section 8.

## 0. Why

snap (`C:\Users\tellt\Projects\snap`, its README) is a "System One" decision model in the
style of TypeSafe AI's Jev: a state (text, or images) and a set of questions with fixed
answer sets go in, a probability distribution over each answer set comes out, from ONE
forward pass of an ordinary instruct model. No decoding: the prompt ends where the answer
begins, the engine reports the next-token distribution at that position, and the letters
`A`..`Z` that tag the options are read off it by token, renormalised, and returned. Measured
2026-09-22 on an RTX 3090 Ti with Qwen3.5 9B Q8_0: ~40 ms of prefill for a 110-token state,
every accuracy check green, a 96k-token book read once and then ten questions answered at
100–140 ms each, and "does this document say X" traps held (people of the era absent from the
book at p ≈ 0.003).

Today it is a separate server. Foundry's Categorize tile spawned `snap serve` as a child,
first against snap's own llama-server and then (2026-09-23, `92488d4`) against Crucible's
chat door, with the model loaded and leased through Crucible and snap reading the letter
logprobs the chat door already forwards verbatim. That works, and it is the wrong shape by
this project's own rules:

- **A GPU feature is designed as a Crucible job type**, not as another app-side spawn with
  its own port and its own engine knowledge (bookforge `CLAUDE.md`, "Crucible").
- **The client knows the order and the server; the server knows the engine** (DESIGN.md
  §3.1). Which engine reports logprobs how, which flags make them available, whether a
  prefix cache exists to prime, whether the model is hybrid and can only rewind to a
  checkpoint — every one of those is engine knowledge, and snap-beside-Crucible put it in
  the app's process.
- **One fact, one owner** (ARCHITECTURE.md R1). A decision's prompt is a fact about the
  weights it is read from — `pages.py` is the precedent: the page prompt lives in Crucible
  because the model it addresses does.

So the decision becomes a door of Crucible's, served by whatever engine Crucible runs for
the model, and snap's process goes away for the apps.

## 1. What is true today (measured, 2026-09-22/23)

- **The chat door forwards `logprobs`, `top_logprobs` and `chat_template_kwargs` untouched**
  (PHASE2-LLM.md §5; `tests/test_llm_api.py:1495-1528` asserts the bytes). The SDK's
  `chat()` cannot send them and drops them from the reply (`sdk/ts/src/client.ts:1160-1221`,
  `:3195-3218`), which is why the tile spawned snap instead of calling the SDK.
- **vLLM 0.29.0** (cuda-linux), read from the source at tag `v0.29.0` (commit `98dff2a`) on
  2026-09-23: `/v1/chat/completions` returns `choices[0].logprobs.content[].top_logprobs`
  as `ChatCompletionLogProb {token, logprob, bytes}` inside `ChatCompletionLogProbsContent`
  (`vllm/entrypoints/openai/chat_completion/protocol.py` L81-95); `top_logprobs` is the
  request field (L220) and must come with `logprobs: true` (L840-852). `--max-logprobs`
  defaults to 20, `-1` meaning uncapped (`vllm/config/model.py` L250-254), and a request
  past it is `400 "Requested sample logprobs of N, which is greater than max allowed: M"`
  (`vllm/sampling_params.py` L812-827). `--logprobs-mode` takes `raw_logits`,
  `raw_logprobs`, `processed_logits`, `processed_logprobs` and defaults to `raw_logprobs`
  (`vllm/config/model.py` L101-107, L255-262; "raw" = before any logits processor). Both
  flags are spelled `--max-logprobs` / `--logprobs-mode` (`vllm/engine/arg_utils.py`
  L934-935). `--enable-prompt-tokens-details` is a `FrontendArgs` bool, default False
  (`vllm/entrypoints/launchers/cli_args.py` L132, registered as `--enable-prompt-tokens-
  details` by L242-244 with `BooleanOptionalAction`, `arg_utils.py` L387-389); without it
  `_make_prompt_tokens_details` returns None (`chat_completion/serving.py` L90-108), and
  WITH it `cached_tokens` may still be null (`PromptTokenUsageInfo`,
  `vllm/entrypoints/serve/engine/protocol.py` L96-99). Also found: a request field
  `logprob_token_ids` (protocol.py L286-296) that returns logprobs for named vocab ids — the
  exact tool for a fixed label set, not used here because it needs token ids and every
  other engine reports strings; noted for when a vLLM-only reader is worth writing.
- **llama-server b10970** (llama-windows): the stock GGUF is the one snap measured
  (`unsloth/Qwen3.5-9B-GGUF`, `Qwen3.5-9B-Q8_0.gguf`). Its `/v1/chat/completions` takes
  `chat_template_kwargs` and `image_url` data-URI parts (server README at b10964 §"POST
  /v1/chat/completions"). **VERIFIED from the source at tag `b10970` (commit `bfdc321`),
  2026-09-23: the chat route DOES honour `logprobs`/`top_logprobs`, in the OpenAI shape.**
  `oaicompat_chat_params_parse` maps `logprobs: true` to `n_probs = top_logprobs` (default
  20) and refuses `top_logprobs` without `logprobs` (`tools/server/server-common.cpp`
  L1403-1412). The probabilities are PRE-sampling unless the body says
  `post_sampling_probs` (default false, `server-task.h` L76; `server-context.cpp` L1790,
  L1964-2019), taken from a softmax over the raw logits (`get_token_probabilities`,
  `server-common.cpp` L1524-1554). The non-streamed chat reply writes
  `choices[0].logprobs = {content: [{id, token, bytes, logprob, top_logprobs: [{id, token,
  bytes, logprob}]}]}` (`server-task.cpp` L264-300, L434-437) — vLLM's path with an extra
  `id` — and `usage.prompt_tokens_details.cached_tokens` is ALWAYS present (L365-371).
  `n_probs` has no cap below the vocabulary. Two caveats read in the same source:
  `logprobs` is omitted entirely when no token was produced (`probs_output.size() > 0`,
  L434 — the snap prime incident's shape, and why a prime is never read), and a live run
  on the card is still owed (section 8). So ONE reader serves it and snap's `/completion`
  reader stays in snap. Crucible starts it with `--parallel 1`, and until this phase
  declared no `chat_concurrency`, so the chat door did not bound it (fixed, §2.6).
- **Qwen3.5 is hybrid** (DeltaNet + full attention every 4th block). On llama-server a prompt
  prefix is reused only from a *context checkpoint* (~50 MiB, ~80 ms each), created at batch
  boundaries; a state sent ALONE first ("priming") leaves the checkpoint at its end and every
  question then prefills only its own ~48 tokens (snap's `/completion` reader; through the
  chat route that holds only for the layout of §2.5.1). Crucible's llama-windows launch does not
  set `--ctx-checkpoints` (engine default 32). vLLM's prefix cache is block-granular and
  needs no priming to share a prefix, but a prime still keeps N concurrent questions from
  each prefilling the same state at once.
- **mlx-lm 0.31.3** (mlx-darwin text), READ in the installed
  `~/.crucible/envs/llm/lib/python3.11/site-packages/mlx_lm/server.py` on the Mac Studio on
  2026-09-23 (no model run): the chat route takes `logprobs: bool` and `top_logprobs: int`
  (L191-192, read from the body at L1189-1190) and validates `top_logprobs` to **at most 11**
  (`min_val=0, max_val=11, whitelist=[-1]`, L1245); `_format_top_logprobs` emits
  `{id, token, logprob}` (L426-435) and the reply is
  `choices[0].logprobs.content = [dict(top[0], top_logprobs=top), …]` (L1317-1321) — the
  same path as vLLM's (content[0] is the TOP token's entry rather than the sampled one's,
  which the reader never reads), no `bytes`. Token strings are RAW pieces
  (`convert_ids_to_tokens`), which for a bare capital letter is the same string. The
  logprobs are taken AFTER the logits processors (`mlx_lm/generate.py` L409-420), which is
  why a decision states its own sampling and takes no manifest default (§2.4). Usage
  carries `prompt_tokens_details.cached_tokens` when the prompt cache reports a count
  (L1339-1347). The Mac PAGES engine
  (`engines/mlx_vlm_serve.py`) refuses any field but its `KNOWN_FIELDS` and calls
  `batch_generate(compute_logprobs=False)`: it cannot serve a decision as it stands.
- **`qwen3.5-9b` is declared `modalities = ["text"]` on every backend**: vLLM runs it
  `--language-model-only`, llama-windows has no `mmproj`. An image decision on it is a
  manifest change and a memory re-measurement, not a door change (§2.7).
- **A decision holds nothing.** Like a chat, it takes no lane and makes no job row; the
  client that wants the model to stay across a book holds a lease
  (`POST /v1/models/{id}/lease`), exactly as the Categorize tile did.

## 2. The design, RULED

### 2.1 A synchronous door, sibling of the chat door

`POST /v1/decide` on the private router: Bearer auth, `X-Crucible-Api` major 1, `X-Crucible-
Act` read before any work (an unknown act is a 400, as on chat). The request names a `model`;
if it is not the resident model the door answers **`409 model_not_resident`** with the same
body the chat door gives — *Crucible never loads a model to answer a decision*. An upstream
model id (`anthropic/…`) is refused **`400 decide_needs_logprobs`**: the door reads a
distribution, and no upstream returns one.

Each decision is recorded in `InFlight` (so `GET /v1/activity` shows it under `chat.rows`
with its act and client — a decision IS a completion of one token), bounded by
`chat_admission(resident.engine)` (**`503 chat_queue_full`** with the measured `Retry-After`
when the engine's admission is full), and closed by `_chat_over` so the card settles after
it exactly as after a chat. Nothing new is invented for admission, settlement or activity:
a decision walks through the chat door's machinery with a different body in and out.

### 2.2 The wire

Request:

```json
{
  "model": "qwen3.5-9b",
  "state": "<string, or any JSON value — non-strings are serialised as compact JSON>",
  "images": ["<base64 image file>", "..."],
  "questions": {
    "team":   {"type": "choice", "instructions": "Which team should handle this?",
               "options": {"billing": "Payment and invoice issues", "technical": "Bugs and errors"}},
    "anger":  {"type": "score",  "instructions": "How frustrated is the customer?",
               "levels": ["Calm", "Frustrated but civil", "Very angry"]},
    "urgent": {"type": "yesno",  "instructions": "The message conveys urgency"}
  },
  "missing": "refuse"
}
```

`images` is optional (at most 8, `400 too_many_images`); `state` may be `""` only when
images are given. `missing` is optional, `"refuse"` (the default) or `"report"`, anything
else `400 invalid_request` naming it — see *Missing labels* below. Unknown keys are refused (`extra="forbid"`, as every Crucible params model).
`choice` takes 2–26 options in insertion order; `score` 2–10 unique ordered levels; a
question name is a single path member. Validation refusals are **`400 invalid_request`**
naming the field, Crucible's own code for the caller's mistake; `too_many_options` and
`too_many_images` keep their names because a client can act on them.

Response `200`:

```json
{
  "model": {"id": "qwen3.5-9b", "revision": "…", "fingerprint": "…"},
  "engine": "vllm",
  "answers": {
    "team":   {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.91, "technical": 0.09},
               "logprobs": {"billing": -0.0943, "technical": -2.4079},
               "confidence": 0.91, "label_mass": 0.998},
    "anger":  {"type": "score", "score": 1.4, "level": "Calm",
               "probabilities": {"Calm": 0.62, "Frustrated but civil": 0.36, "Very angry": 0.02},
               "logprobs": {"Calm": -0.478, "Frustrated but civil": -1.0217, "Very angry": -3.912},
               "confidence": 0.62, "label_mass": 0.997},
    "urgent": {"type": "yesno", "p": 0.83, "logprob": -0.1863, "label_mass": 0.99}
  },
  "timing_ms": {"total": 84.0,
                "per_question": {"team": {"wall_ms": 21.3, "prompt_tokens": 136, "cached_tokens": 64}},
                "prime": {"wall_ms": 31.0, "prompt_tokens": 75, "cached_tokens": 0}},
  "tokens": {"per_question": {"team": 136}, "images": 0}
}
```

The answer shapes, `label_mass`, `score` = Σ(1-based level index × p), `confidence` = the
largest renormalised probability, and `p` = renormalised P(Yes) are snap's, unchanged
(`snap/docs/CONTRACT.md`). What Crucible adds: `model` is the provenance triple every
artifact sidecar already carries (PHASE2-LLM.md §5), so a decision names the weights that
made it; `engine` is the engine kind that answered. `timing_ms` is Crucible's wall clock per
request (the OpenAI reply carries no prefill time), `prompt_tokens` is `usage.prompt_tokens`,
and `cached_tokens` is `usage.prompt_tokens_details.cached_tokens` **or `null` when the
engine did not say — never 0** (a number nobody measured is not a measurement).
`tokens.per_question` is the same `usage.prompt_tokens`, kept because snap's clients read
it there.

> *Corrected by the build (2026-09-23):* the example above used to show the team question
> at `prompt_tokens: 61, cached_tokens: 75` beside `tokens: 136` — snap's openai-chat split
> (prompt minus cached), which contradicts the sentence it sits over. The sentence is the
> contract: `prompt_tokens` is the whole prompt, so it is 136, and a `cached_tokens` of
> `null` on the prime beside a number on a question could not come from one engine.
>
> Two more the build settled. A request with no `model` is **`400 invalid_request`**
> naming `model`, not the chat door's `model_required`: this door has a schema (it is what
> `docs/API.md` is generated from), and the schema's refusal is the one every other field
> gets. And a request that never got an answer on the wire — refused, reset, timed out — is
> the chat door's own **`502 engine_unreachable`** (`_engine_unreachable`, after the same
> one-retry budget); `engine_error` is an answer that came back and could not be read.

Engine-side faults: **`502 engine_error`** naming the engine and its status/body — a
non-200, a body that is not JSON, a reply missing `usage` or `logprobs` or naming a letter
twice — and **`502 label_not_in_probs`** naming the question and
the letter when the engine's top-K did not contain a label — that one keeps its own name
because it is the one a caller repairs by shortening the option list (or by asking for
`missing: "report"`, below).

**Log-probabilities** (ruled by Owen 2026-09-23, asked for by the Briefcase session that
runs Viterbi over the answers). Every `choice` and `score` answer carries
`"logprobs": {<option>: ln p, …}` in OPTION ORDER — the natural log of the renormalised
probability beside it — and a `yesno` carries `"logprob": ln p`. Multiply by `label_mass`
(add `ln label_mass`) for the un-renormalised mass. **The values are NOT calibrated**: they
are one forward pass's reading, not measured frequencies. `-Infinity` is not JSON, so a
probability of exactly 0 has a `null` log-probability; the only way the door produces one
is the report-mode `yesno` below (a double underflowing on a letter the engine did return
would too, and is not a case anyone has seen).

**Missing labels** (same ruling). `missing` is `"refuse"` by default — a label outside the
engine's top-K is `502 label_not_in_probs`, exactly as before. With `"report"` the door
**never invents a number**: a label outside the top-K gets `null` in `probabilities` and in
`logprobs`, is named in the answer's `"missing_labels": ["<option>", …]` (option order;
`Yes`/`No` for a yesno), and the renormalisation, `confidence`, `score` (the expected value
over the levels returned, at their own 1-based indices) and `label_mass` all run over the
letters the engine actually returned; `choice`/`level` is the argmax over those. In report
mode `missing_labels` is on EVERY answer, `[]` when nothing was missing; **in refuse mode
the key is absent**, not empty (the SDK demands it when it asked for `report` — the caller
asked for it, so it is load-bearing — and refuses it as a protocol error otherwise). A question whose EVERY label is missing is still
`502 label_not_in_probs` in both modes — there is no answer to report
(`details.letter` is then `null`). The prime is unchanged, and no engine body carries the
mode: it is Crucible's reading, never the engine's.

```json
"anger":  {"type": "score", "score": 1.4, "level": "Calm",
           "probabilities": {"Calm": 0.6, "Frustrated but civil": 0.4, "Very angry": null},
           "logprobs": {"Calm": -0.5108, "Frustrated but civil": -0.9163, "Very angry": null},
           "confidence": 0.6, "label_mass": 0.5, "missing_labels": ["Very angry"]},
"urgent": {"type": "yesno", "p": 0.0, "logprob": null, "label_mass": 0.2,
           "missing_labels": ["Yes"]}
```

A `yesno` in report mode with `A` or `B` missing has `p` = the returned one renormalised
alone: `1.0` with `No` missing, `0.0` (and `logprob: null`) with `Yes` missing, the other
named in `missing_labels`. That is honest and useless — the whole signal is in
`label_mass`, which is exactly the one letter's raw probability — so a caller gates on
`label_mass` (or on `missing_labels` being non-empty) before it believes `p`.

### 2.3 The reading is Crucible's, the order is the client's

`crucible/decide.py` owns everything snap's `prompt.py`, `labels.py` and the pure half of
`decide.py` own today, ported with their unit tests: the system prompt, the messages (since
the release after 1.0.24 the system message is snap's frame, `\n\nState:\n` and the state text; the user
message is the images as content parts, then the question block with the lettered legend
and "Answer with the letter only." — §2.5.1 says why the state moved out of the user
message), the letter assignment (A.. in option
order; `yesno` is A = Yes, B = No), the renormalisation over the letters and `label_mass`,
the expected-value score. The client sends none of that; it sends the state, the questions
and the options, which is the ORDER. "Prompts stay in the app" (DESIGN.md §3) is not
contradicted: the decision prompt is not the app's prompt about its book, it is the frame
that makes an instruct model report a distribution, and it is a fact about the weights.

### 2.4 One reader, the engine's OpenAI chat route

Every engine Crucible runs is reached through its own `/v1/chat/completions` at
`resident.base_url`, the route the chat door already proxies. Per question the door sends:

```json
{"model": "<engine's served name>", "messages": [system, user],
 "max_tokens": 1, "logprobs": true, "top_logprobs": <K>, "temperature": 0,
 "chat_template_kwargs": {"enable_thinking": false}, "stream": false}
```

and reads `choices[0].logprobs.content[0].top_logprobs`, matching a label by the token
STRING equal to its letter (the engines report token strings, not ids — decoded on vLLM and
llama-server, raw BPE pieces on mlx-lm, and a bare capital letter is the same string in
both). `K` is the number of labels in the question plus 4 (a margin for the tokens that
outrank a letter when the model wanted to say something else — which `label_mass` then
reports) and never more than the engine's stated maximum (§2.6). A question with MORE
labels than that maximum is refused before anything is sent (`503 decide_not_served`
naming the engine, its cap and the question) — on mlx-lm's 11 that is any choice past K.

A label that is not among the K entries that came back is the request's `missing` mode's
to handle (§2.2): refused as `label_not_in_probs`, or reported as `null` and named in
`missing_labels`. The reader never widens K or asks the engine again to find it, and in
neither mode does it put a number where the engine gave none. The log-probabilities it
returns are ln of the renormalised probabilities it computed, not the engine's raw
`logprob` field — the engine's is for the whole vocabulary, the answer's for the labels.

*Corrected by the build:* this paragraph said `enable_thinking: false` "goes through the
same `chat_template_kwargs` merge the chat door's `apply_defaults` does". It does not go
through `apply_defaults` at all, and on purpose: **a decision states every knob a reading
depends on and takes no manifest `[defaults]`**. mlx-lm computes its logprobs AFTER its
logits processors (`mlx_lm/generate.py` L409-420), so a manifest `repetition_penalty` would
move the letters — and the letters are in the legend, exactly the context a repetition
penalty punishes. `enable_thinking: false` is stated in the body, and a stated key is one
no manifest default can override, so a manifest's `[defaults] thinking` still cannot turn
a decision into a reasoning trace. The prime is sent with no `logprobs` and no
`top_logprobs` at all (its reply is never read for letters, §2.5).

The engine renders its own chat template around the messages (which is why there is no
template to read and prove, unlike snap's llama path) and the first generated token is the
first token of the assistant content. That holds for vLLM, llama-server and mlx-lm; the
reader does not know or care which it is talking to, and reads only `token` and `logprob`
of each entry (vLLM adds `bytes`, llama-server `id` and `bytes`, mlx-lm `id`).

Images travel as `{"type": "image_url", "image_url": {"url": "data:image/<fmt>;base64,…"}}`
content parts at the start of the USER message, ahead of the question's text part, the
encoding `pages.py` already uses, on an engine whose manifest declares `image` (§2.7). They
cannot go in the system message with the state text (Qwen3.5's template raises "System
message cannot contain images."), so the system message says where they are
(`decide.IMAGES_NOTE`) — §2.5.1.

**Where an engine cannot do this, the door refuses by name, `503 decide_not_served`,
saying which engine and why** (today: the `mlx-vlm` page server, which computes no
logprobs; and any engine asked for more options than its cap). Each engine class states
`decide_logprobs`, `max_logprobs` and `decide_basis` — the source reading the numbers came
from — and `engines.decide_reading()` refuses a class that states one without the others,
as `chat_admission` does. No second reader is written for an engine until it is measured
to need one — snap's `/completion` reader for llama-server (ids from `/tokenize`, a proven
assistant prefix, `multimodal_data`) stays in snap as the bench; b10970's chat route was
verified to carry logprobs (§1), so it is not needed here.

### 2.5 Priming and concurrency

With more than one question the door first sends the shared prefix — system + state (+
images) — as its own one-token completion, then the questions **concurrently, up to the
engine's admission** (vLLM batches them; llama-server runs them one at a time behind
`--parallel 1`; the door's own concurrency ceiling is `chat_admission`'s number when the
engine states one and 16 otherwise — `decide.UNSTATED_ENGINE_CONCURRENCY` — and both are
Crucible's numbers, never the wire's). A decision is ONE `InFlight` row however many of its
questions are on the wire, so it is admitted as one completion; its own fan-out is what the
ceiling bounds. On
llama-server the prime is what leaves the hybrid model's checkpoint at the end of the
state; on vLLM it fills the prefix cache before N questions ask for the same blocks. Either
way a prime's reply is never read for letters, and a prime that returns no logprobs is not
refused (snap `3509bc5`'s rule: a prime is not an answer).

*Corrected by §2.5.1:* the sentence above about llama-server was not true of the layout
it was written for — the old prime did not leave a checkpoint at the end of the state.

#### 2.5.1 The state lives in the SYSTEM message (fixed after 1.0.24, Briefcase's Mac smoke)

**The finding.** Briefcase's live smoke on 1.0.24 (Mac Studio, `qwen3.5-9b` on mlx-lm
0.31.3; one transcript state of ~3,000 tokens, 100–700 choice questions of 26 options, 64
per `/v1/decide` call): accuracy identical to direct llama-server, but **4.55 s per
question against 0.68 s — 6.7× slower**, and the engine log showed mlx-lm re-prefilling the
whole state for EVERY question. The prime was never reused.

**The cause, read in mlx-lm 0.31.3's installed source on the Mac Studio
(`~/.crucible/envs/llm/lib/python3.11/site-packages/mlx_lm/`), 2026-09-23:**

- Qwen3.5 is hybrid: `models/qwen3_5.py` L304-305 makes an `ArraysCache` (recurrent state)
  for every linear layer. `can_trim_prompt_cache` (`models/cache.py` L88-92) is therefore
  False, so `LRUPromptCache.fetch_nearest_cache` (L1674-1693) can reuse an entry only when
  its token key is the EXACT prompt or an exact PREFIX of it (`result.shorter`, L1690); the
  branch that trims a longer entry back (L1683) is closed to it.
- Entries are saved only at SEGMENT ends. `server.py` `_tokenize` (L516-624) cuts a chat
  prompt whose last message is the user's (L577: otherwise no segments at all) into at most
  three: the system segment, ending where `apply_chat_template(system + [user ""])` first
  differs from the prompt (L583-603) — i.e. just after `<|im_start|>user\n` — then the user
  segment, then the thinking tail. The batch path (both KV and Arrays caches have `merge`,
  so Qwen3.5 is batchable) fetches at L753, saves a cache at each `end_of_segment and not
  end_of_prompt` (L864-880, typed `system`/`user`), and at the end one more under prompt +
  generated tokens (L902-908, `cache_type="assistant"`).
- The door's layout through 1.0.24 was `[system: frame, user: "State:\n" + state (+
  "\n\n" + question)]`. The prime's user-segment entry ended in `<|im_end|>\n<|im_start|>
  assistant\n` and its final entry in the generated token — neither a prefix of any
  question. The one boundary prime and question shared was the end of the FRAME, ~43
  tokens. Measured with the Mac's own tokenizer and mlx-lm's own `_tokenize` (no model
  load): old layout, a 535-token prime and three questions — every question's reusable
  prefix 43 tokens.

**The fix: one layout, every engine.** `[system: SYSTEM_PROMPT + "\n\nState:\n" + state
(+ "\n\n" + IMAGES_NOTE when there are images), user: (images +) question block]`; the
prime is the same system message and a fixed user turn, `decide.PRIME_USER_TEXT` ("The
questions follow."). The shared prefix is now the WHOLE system turn, and it ends exactly at
mlx-lm's system-segment boundary. Same probe, new layout: the system segment ends at token
527 for the prime and for every question (a question whose block begins with the prime's
first word included — the boundary is found against the `user ""` render, not between two
prompts), and the prime's saved system entry is a prefix of every question. The prime's
user turn must never be EMPTY: `system + [user ""]` would then be a prefix of the prompt,
no index would differ, and no system segment would be saved.

**mlx-lm, and whether to serialise.** The door awaits the prime's reply before any question
leaves (`_decide_on_engine`), and the system entry is inserted during the prime's PREFILL,
before its token is generated — so every question fetches it, deep-copies it
(`copy.deepcopy`, L1678/L1692), and prefills only `user` + block + tail. The door's fan-out
on the Mac is `chat_admission`'s 2 (`chat_concurrency = 1`); the two in flight each copy
the same entry and neither can spoil it. Serialising to 1 buys nothing. The entry also
survives the stock `--prompt-cache-size 10`: `CacheOrder.pop` (L1649) evicts `assistant`
then `user` entries before the single `system` one, and each question adds one of each.

**vLLM 0.29.0.** Token-level prefix caching in whole blocks — 544 tokens on this hybrid
(§8a). The old layout already shared frame + state up to the question's `\n\n`; the new one
shares frame + state + `<|im_end|>\n<|im_start|>user\n`. Same blocks, give or take one
boundary: nothing gained and nothing lost. Images sit at the start of the user turn in
every request, identical, so they remain part of the shared prefix.

**llama-server b10970** (read at tag `b10970`, 2026-09-23). The chat route renders Qwen3.5
through the specialised Qwen3-Coder handler (`common/chat.cpp` L1204-1210), which declares
message delimiters including USER `<|im_start|>user` (`common/parsers/qwen3-coder.cpp`
L31-37). `tools/server/server-context.cpp` breaks a prompt batch at the start of the LAST
user message (L3550-3556) and lays a context checkpoint before decoding the batch that
starts there (L3588-3634, `is_last_user_message` exempt from `--checkpoint-min-step`), plus
two near the end, at 4 + n_ubatch and 4 tokens before it (L3559-3579; n_ubatch 512 by
default, `common/common.h` L452, and Crucible does not set it). A later request restores the
newest checkpoint at or before its common prefix (L3350-3377). **Honestly: the old layout
was not aligned here either.** Its last user message began after the frame, BEFORE the
state; its checkpoint 4 tokens from the end sat in the template tail after the state
(`<|im_end|>…`), past where a question diverges; the one that could serve was
~516 tokens from the end, so every question re-prefilled roughly the last 500 tokens of
the state plus its own block (and a state under ~516 tokens reused only the frame). snap's
"~48 tokens a question" was its `/completion` reader, whose prime ended exactly at the
state. §2.5's sentence was carried over from snap without that difference. **With the new
layout the last user message starts exactly where the shared prefix ends**, in the prime
and in every question, so the prime's user-start checkpoint IS the shared prefix and each
question prefills only its own user turn — and question N leaves the same checkpoint for
question N+1. Images get no reuse on llama-server in either layout: the checkpoint is laid
before the user turn, and no checkpoint follows an image chunk (L3621-3622). `qwen3.5-9b`
on llama-windows is text-only today, so this is a note, not a loss.

**The prompt text changed, so the answers may move.** SYSTEM_PROMPT's words and the
question block are unchanged; only where the state sits moved (and Qwen's template `|trim`s
system content, so a state's trailing whitespace no longer reaches the model). **Owed
before the layout is trusted:** Briefcase's YTSeg smoke re-run against it — F1@±1 1.000,
Pk 0.058 on `u25JYe8E3RA`, 16/16 sanity on 1.0.24 — for accuracy AND speed, with the mlx-lm
log showing questions 2..N prefilling only their own tokens (`cached_tokens` = the system
segment on every question after the prime). The same check on llama-server (a restored
checkpoint at the user start) and on vLLM (cached tokens unchanged from 1.0.24) belongs to
the §8 live pass.

### 2.6 Engine flags the door needs, composed by Crucible, never on the wire

Composed in `Residency._engine_args` beside `--max-model-len` (they are facts about what
the DOOR needs from the engine, not about the model, so they do not belong in a manifest's
`engine_args`):

- **vLLM:** `--max-logprobs 32` (26 letters + the margin), `--logprobs-mode raw_logprobs`
  (stated, not defaulted: a build that defaulted to processed logprobs would answer a
  one-hot distribution at temperature 0 and every decision would read as certain),
  `--enable-prompt-tokens-details` (so `cached_tokens` is a number). All three spellings
  exist at 0.29.0 (§1). They are ONE tuple, `engines/vllm.py`'s `DECIDE_ARGS`, appended
  right after `--max-model-len` and before the sized KV pool; `VllmEngine.max_logprobs` is
  the same constant, so the flag and the reader's clamp cannot disagree.
- **llama-server:** nothing new for the reading (`n_probs` has no small cap). The one change
  is a declared **`chat_concurrency = --parallel`** (1) for `llama_server.py`, so
  `chat_admission` bounds its chat door at 2 the way it bounds mlx-lm — the same defect
  Crucible 1.0.10 fixed for the Mac, present on Windows today, and a decision fan-out is
  exactly the load that would expose it.
- **mlx-lm:** no flag exists and none is needed: the server caps `top_logprobs` at 11 on its
  own (§1), `MlxLmEngine.max_logprobs = 11`, and its `chat_concurrency = 1` already bounds
  the fan-out at 2.

#### 2.6.1 The Mac's cap is raised to 40 by a patch (Owen: "go ahead", 2026-09-23)

**Supersedes the mlx-lm bullet above.** No flag exists, so the ceiling is edited where it
lives: mlx-lm 0.31.3's `mlx_lm/server.py` validates `top_logprobs` with
`self._validate("top_logprobs", int, min_val=0, max_val=11, whitelist=[-1])` (L1245 on the
Mac Studio, read 2026-09-23), and that validator is the only ceiling — `_format_top_logprobs`
takes any `top_n`, and both generation paths pass `args.top_logprobs` through. At 11, K =
labels + 4 meant no question with more than 7 options could be read on the Mac; Briefcase
needs 11 and 26. 40 covers 26 letters plus the margin, with room.

- **The patch.** `crucible/envs/llm/patches/patch_mlx_lm_top_logprobs.py` replaces that one
  line with `max_val=40` (plus a comment naming itself). Anchor = the stock line, byte-exact
  and required exactly once; marker = the patched `_validate(... max_val=40 ...)` line;
  absent marker = the stock line. Idempotent (`ALREADY_PATCHED` by the marker), patched from
  the LIVE file with a `.orig` snapshot, and `ANCHOR_NOT_FOUND` (exit 2) when a newer mlx-lm
  moved or reworded it — re-derived by a person, never skipped.
- **One registry, per env type.** `crucible/envpatches.py` maps a job type to its patch table
  and appliers: `tts` is `narratorpatches.NARRATOR_PATCHES` unchanged, `llm` is this patch.
  Selection is still by the recipe's pins: it runs on `mlx-darwin` (pins `mlx-lm`), is
  `not_applicable` on `cuda-linux` (vLLM) and on `llama-windows` (no llm recipe at all).
- **Installed before the stamp.** `jobenv.install_env` applies each env type's table after
  pip and before the stamp, then checks it; an env whose patch is not in has no stamp.
- **The cap is tied to the check at ENGINE START.** `MlxLmEngine.max_logprobs = 40`, and
  `MlxLmEngine.start()` runs the patch's own check against the env its interpreter lives in
  and refuses `llm_env_unpatched` by name unless it is `applied`. The door's reading
  (`engines.decide_reading`) is class-level, so the guarantee is that no mlx-lm is ever
  RESIDENT unpatched: the door can never be told 40 by an engine that would answer 400.
- **Doctor.** `crucible doctor` reports `llm_patches` rows (`llm patch (...)`); `missing` or
  `stale` is a problem naming `crucible env patch llm`, `no_env` is left to the env row.
- **Every install and upgrade applies it, before the service starts.** An upgrade never runs
  `crucible install`, so the install step list (`sdk/bootstrap/src/steps.ts`, generated into
  `install.sh`/`install.ps1`) carries `env-patch-llm`: the NEW wheel's `crucible env patch llm`,
  run after the wheel is in and before `service-install`, while the server is down. It
  applies, checks, and fails the install by name; no llm env is "nothing to patch" (exit 0),
  and vLLM / llama.cpp hosts read `not_applicable`. A deploy runs the release's own
  `install.sh`, so it gets the step with no deploy-side copy.

### 2.7 Images: a manifest ruling, not a door feature

The door carries `images` from day one and the reader sends them as content parts. Whether a
given model ANSWERS them is the manifest's `modalities`: on a `text` model the door refuses
`400 model_text_only` naming the model. `qwen3.5-9b` is text-only on every backend today, so
image decisions ship when a manifest declares an image-capable row — the natural candidate
is `qwen3.5-9b` with `modalities = ["text", "image"]` on cuda-linux (dropping
`--language-model-only`, measuring memory again) and `mmproj = mmproj-F16.gguf` on
llama-windows (snap measured that exact projector: a rendered 800×1100 page is ~850 tokens,
page-type at 0.996–0.999 in 445–628 ms). That is Owen's ruling (§7.3), because it changes
what `pages` and `analysis` load.

### 2.8 Client surfaces

- **SDK** (`sdk/ts/src/client.ts`): `decide(request, {act?}) → DecideResponse`, typed
  request and answer unions, the same `CrucibleRefused` codes. The ANSWER is demanded — an
  answer per question of the type asked, its `choice`/`level`/`score`/`p`, its
  `probabilities` and `label_mass`. Everything that describes it — `confidence`,
  `logprobs`/`logprob`, the timings (`cached_tokens` included), the token counts, the
  model's pins and the engine — reads as `null` when a server does not state it. *Amended
  2026-09-24:* this line said "Lockstep: the SDK demands every field it knows" until Owen
  ruled that any Crucible that answers works (INTENT.md); a field present with the wrong
  type is still refused.
- **CLI** (`crucible/apiclient.py`): `crucible api decide --model <id> --state <text|@file>
  [--image <file>]… --choice name "instructions" opt=desc… --yesno name "…" --score name "…"
  l1,l2,… [--act <name>]`, printing the response JSON — snap's own CLI grammar, so a person
  moving from `snap decide` types the same thing.
- **Docs:** `API.md` regenerates from the request model's docstrings (`release.sh` refuses a
  cut otherwise); `API-CLI.md` gains the verb; DESIGN.md §3's table gains the row
  (`decide` | state + questions | distributions | one forward pass at the resident model,
  `PHASE22-DECIDE.md`) and DESIGN.md §4 the route; CLIENT-SURFACES.md names the door.
- **Capability / act:** a decision declares its act like a chat. Whether `decide` is its own
  capability class in the AI page (a model chosen for decisions) or rides on `analysis` is
  §7.2.

### 2.9 The lineup (built 2026-09-23)

Owen, 2026-09-23: *"we should configure snap to work with either the 27b, the 9b, a 3b/4b, or
a 0.8b depending on what the user passes in/requests, and depending on what system it's
running on and how much ram is available … we should use the latest available. qwen 3.8
ideally. if thats not available, the 3.5 models … the models should be pulled from the
official model repository"*, and *"vision should be integrated into crucible's functionality
as well. and batching."* Three pieces, built together:

**A `decide` capability class** (§7.2, answered: its own class). `job_type = "llm"`, a
valid `X-Crucible-Act` because `ACT_NAMES` is derived from `CLASSES`. Candidates: every
`qwen3.8` and `qwen3.5` manifest, best-first by `memory_bytes_estimate` like every class —
so a card that holds a 27B decides on it and a laptop still decides — with **no size
floor**. Not routable, although the brief proposed routable: the door refuses every
upstream id (`decide_needs_logprobs`, §2.1), so a routable class would let an operator send
the act somewhere that refuses all of it, and its refusals would carry the "add an API key"
offer that cannot help. Its working context is **8192 tokens × 2**, not × 16: the questions
EXTEND one primed state (§2.5) and share its KV through the prefix cache, and the 16
questions' tails at one 544-token vLLM block each (§8a) come to about one more state. Sized
as sixteen independent 8192-token states it would refuse the 9B on the 3090 Ti's cuda-linux
(24.8 GB of need against 22.5 GB of budget) — the very card and model snap measured
decisions on. `tests/test_decide_lineup.py` pins that.

**The floors became numbers.** `clean`, `translate`, `simplify` and `analysis` now carry
`min_params_b = 9` (`capability.NINE_B_FLOOR`), compared against `[model] params_b` — which
became a number (`params_b = 0.8` is a float) rather than an int. Before, the floor was a
side effect of the family filter and of the 9B being the smallest thing `models/` held; the
day a 4B arrived it would have put the 4B under cleanup and translation with nobody
deciding it. MODEL-CHOICE.md's 2026-09-23 addendum records it. Their candidate lists did not
move, and are asserted exactly per backend.

**Two tiers, from the official repos.** `qwen3.5-4b` (Owen's "3b/4b"; Qwen publishes a 2B
and a 4B, no 3B) and `qwen3.5-0.8b`. Qwen3.8 has no small tiers — the hub lists 27B,
Flash-Next and 2.4T-A95B — so "3.8 ideally" is the 27B pair already here, and 3.5 is the
latest that exists at these sizes. cuda-linux pins `Qwen/Qwen3.5-*` (Qwen's own org);
mlx-darwin pins `mlx-community/Qwen3.5-*-bf16` (the org the 9B already pins); llama-windows
pins `unsloth/Qwen3.5-*-GGUF` Q8_0 + `mmproj-F16.gguf`, because **Qwen publishes no GGUF for
3.5 or 3.8**. Every `memory_bytes_estimate` is COMPUTED (`basis = "computed"`, or
`"declared"` on llama-windows as every block there is), with each term's source written in
the manifest: weights from the safetensors headers at the pin, the 0.8B's text half MEASURED
in §8a's own log (1.53 GiB weights, 0.75 GiB overhead, 18,023 B/token of KV), the 4B's slope
carried EXACTLY from the 9B's measured 40,337 (every config field KV reads is identical), and
the image reserve carried from the 9B's measured 1.90 GiB A/B as an upper bound.

**What a backend SERVES is the block's; what the weights ACCEPT is the model's.** A new
optional `serves` on each `[backends.<kind>]` block, defaulting to `[model] modalities` and
refused by name unless it is a non-empty subset of it (`serves_not_subset`). The engine
choice (`engine_for`, via `class_family`), the image-pairing rules (`mmproj` required when
the block serves images and now REFUSED when it does not; `--skip-mm-profiling` and
`--language-model-only` refused beside a served image) and this door's `model_text_only` all
read the block's set. `model_text_only` now names `backend` and `serves` beside the model's
`modalities`, and `GET /v1/models` rows carry `serves` (null where the host has no block).
The small tiers declare `modalities = ["text", "image"]` and serve images on cuda-linux (vLLM
loads the tower; `--limit-mm-per-prompt {"image": 8, "video": 0}`, 8 being `MAX_IMAGES`) and
llama-windows (the projector), and `["text"]` on mlx-darwin — because the Mac's image engine
is Crucible's dots-specific page server, which computes no logprobs, and a model-wide
`image` would have moved these weights onto it.

**One copy on disk, two fit rows: `weights_of` (built 2026-09-23).** Owen: *"One copy on
disk, two fit rows in the catalog — i think this is a fine way to do it."* The 9B's and the
27Bs' rows are measured text-only and clean/translate depend on those numbers, so their
vision forms are their own ids; `[model] weights_of = "<base id>"` makes such an id an
ALIAS, whose weights are the base's download and nothing else.

- **The store** (`crucible/weights.py`). `subject_dir(alias, backend)` is the base's folder
  (`~/.crucible/models/<base>/<backend>`); every read, pull and removal asks it. Pulling the
  alias pulls the base as the base (same stamp, same folder, never twice) and then only
  `extra_files` — what the alias's block names beyond the base's: the llama-windows
  `mmproj-F16.gguf`, nothing on a whole-repo backend. The alias's `installed` is the base's
  stamp AND its own files present, so a base-only pull leaves it `installed: false` and the
  catalog row's `missing_files` names the projector. A pulled alias writes
  `crucible-alias-<id>.json` beside the base's stamp: that record, with the alias installed,
  is what "an alias exists here" means — on cuda-linux it owns no file, and without the
  record the base could never be refused there, nor the refusal ended.
- **The loader** (`crucible/manifests.py`, `resolve_weights_of`) refuses by name:
  `weights_of_unknown` (no such base beside it), `weights_of_chain` (an alias of an alias, a
  base that is itself aliased elsewhere, or an alias of itself), `weights_of_backend_missing`
  (a backend the base does not declare: there is no download there to share),
  `weights_of_pin_mismatch` (`hf_repo`, `revision` or `file` differ on a shared backend),
  `weights_of_fact_mismatch` (`family`, `params_b`, `trained_context`, `[defaults]` — facts
  about the weights) and `weights_of_local` (the local form, and so the Ollama-store reuse,
  is the base's: Ollama's blob has no projector for an alias). `modalities`, `serves`,
  `context_default`, `display`, `description`, `engine_args` and memory are the alias's own.
- **Removal.** Removing a base an alias holds is `weights_shared`, naming the alias — `409`
  at `DELETE /v1/catalog/model/<base>` with `details.aliases`, the same sentence from
  `crucible remove`, and the same refusal for a forced re-pull of the base (it empties the
  folder). Removing the alias takes its extra files and its record, never a byte of the
  base's. The three HOLDS (resident, lease, running task) are read through every id that
  reads the folder (`catalog.ids_reading`): an alias on the card holds its base
  `subject_in_use` whether or not it was ever pulled as itself. The Windows→WSL migration meets `weights_shared` on the base first (it sorts
  first) and retries it the next round, after the alias has gone.
- **Rows.** `/v1/models` and `/v1/info`'s llm rows carry `weights_of` (null for a base).
  `/v1/catalog` rows carry `shares_weights_of` and `missing_files` (null on a non-alias);
  the download's bytes are the base's row's, and the alias's `expected_bytes` /
  `installed_bytes` count only its own files — 0 where it has none, `expected_bytes` null
  where it has a projector no manifest states the size of.
- **Classes.** `decide` lists the aliases (family match, `aliases=True`); `clean`,
  `translate`, `simplify` and `analysis` do not — an alias is its base served dearer, and a
  best-first walk would otherwise put `qwen3.5-9b-vl` ahead of `qwen3.5-9b` for text.
- **The two aliases**: `qwen3.5-9b-vl` and `qwen3.8-27b-4bit-vl`, each serving
  `["text", "image"]` on cuda-linux (vLLM without the two text-only flags, with
  `--limit-mm-per-prompt {"image": 8, "video": 0}`) and llama-windows (the base's GGUF +
  `mmproj-F16.gguf`); no mlx-darwin block. Memory in section 7.3. A third,
  `qwen3.8-27b-8bit-vl`, shipped on cuda-linux alone and was **removed 2026-09-23** with its
  base's cuda-linux arm — Owen: *"we shouldnt have an 8 bit 27b on here. waste of space, wont
  fit in the gpu"*. The FP8 weights alone are 28.75 GiB against a 24 GB card, so that arm
  could only be downloaded and refused; the 8-bit 27B is mlx-darwin only, and an alias with
  no backend is not a model (nor could it keep a cuda-linux block its base lacks:
  `weights_of_backend_missing`).

**Windows, Owen 2026-09-23 (verbatim):** *"the original intent with windows was that
everything would go through WSL if it exists and nothing would exist on windows. no models.
it will all be managed by the engine in WSL, and windows points/orchestrates to WSL. if there
is no WSL, it all exists in windows crucible. everything is downloaded and managed there"*.
So every `llama-windows` row, and every projector an alias adds there, serves only a machine
with no WSL; on a machine with WSL the store and the engine are the guest's.

**The one real cost, and the picker's rule.** A base and its alias are two engine
configurations over the same files, so switching between them is a full engine reload (~20 s
on the 9B). An app therefore picks ONE form per server session: **the vision form when it
fits; otherwise the base for text and `qwen3.5-4b` for image decisions.** That rule is the
app's, stated here for the picker's author; Crucible offers both rows and the fit table says
which one this card can hold.

## 3. Tests (no GPU)

- `tests/fake_engine.py` learns to answer `logprobs` on `/v1/chat/completions`: a test
  installs a `probs_for(messages) -> {token: p}` and the fake emits `top_logprobs` with those
  entries plus filler mass, in vLLM's exact shape (`{token, logprob, bytes}`), honouring
  `top_logprobs` as a cap and refusing `top_logprobs > max_logprobs` with vLLM's 400 body.
- `tests/test_decide_api.py`: the contract's worked example end to end (exact answers,
  `score` = 1.4, `label_mass`, `null` cached_tokens); `model_not_resident`; the act header;
  `chat_queue_full` at the bound; a prime sent first for two questions and not for one, the
  questions' system message byte-identical to the prime's (the state in it and only in it,
  the question block only in the user message; §2.5.1), and the fake's `segments` prefix rule
  (mlx-lm 0.31.3's exact-prefix reuse) reporting the state cached on every question; `label_not_in_probs` when a
  letter is outside the top-K; `too_many_options` / `too_many_images` / `model_text_only` /
  `decide_needs_logprobs` refused before any request reaches the fake (assert zero
  requests); `decide_not_served` on an engine class that states no logprobs; provenance's
  `model` triple equals the load's; `/v1/activity` shows the decision in flight.
- `tests/test_decide_core.py`: snap's pure-function tests ported (legend, letters, yesno
  A/B, renormalisation, score, images-before-text message layout, 27 options refused).
- `tests/test_chat_admission.py` (there is no `tests/test_engines.py`; the admission tests
  live here): llama-server now states `chat_concurrency`; `chat_admission("llama-server")`
  is 2 with the basis naming `--parallel 1`, and every llama-windows manifest block is read
  to prove it says `--parallel 1` (the fact has two owners, so a check keeps them one);
  each engine's `decide_reading`; vLLM's line carries the three flags with the cap the
  reader clamps to. `tests/test_llm_api.py`'s two exact-engine-line assertions gained
  `DECIDE_ARGS`.
- `tests/test_api_client.py`: `crucible api decide` against `live_server` with the fake
  engine (echo never touches the card and neither does this).
- SDK: `sdk/ts` unit test for `decide()`'s request body, the demanded answer fields, and
  the informational ones reading as `null` when absent (2026-09-24).

## 4. What does NOT change

The chat door, its defaults, its bounding and its streaming; `load-model` / leases /
settlement; every job type; the manifests' model rows (until §7.3); the Mac's page engine;
`api_version` (a new route is an addition, a client that does not know it sees nothing
different).

## 5. What the apps do afterwards

Foundry's Categorize tile (`92488d4`, parked at `acf63c2`) comes back as `sdk.decide()` calls
inside its existing placement (`placeJob('analysis')` → load with the lease riding → groups
of 24 blocks with 12 of context → one `choice` per block), and its "start `snap serve`"
path is deleted. BookForge's uses (block kinds, transcript boundaries, TTS-cleanup gating)
are new consumers of the same call. snap the repo stays as the bench: the llama-server
`/completion` reader, the checkpoint measurements, the live fixtures — and its README points
here for the product door.

## 6. Build order

1. `crucible/decide.py` (port + tests) and the fake engine's logprobs.
2. The door in `api.py`, the engine record's `max_logprobs`, the composed flags, llama-
   server's `chat_concurrency`, the errors.
3. SDK + CLI + generated docs + DESIGN/PLAN/API-CLI/CLIENT-SURFACES rows.
4. Cut, deploy the fleet (standing-authorised), repin both apps.
5. Live pass (§8), then Foundry's tile.

## 7. Rulings still Owen's

1. **snap's own server retires for the apps** once the door ships; snap stays as the bench
   and the reference tests. (Proposed yes.)
2. **A `decide` capability class of its own**, so the AI page picks the model that answers
   decisions per server, or decisions ride on `analysis` (the tile used `analysis`).
   (Proposed: its own class; a 9B answering decisions while a 27B does analysis is the
   likely lineup.) **Built as its own class, 2026-09-23 (§2.9)**, on Owen's lineup
   request that day — no size floor, not routable.
3. **Images on `qwen3.5-9b`**: declare `image` on cuda-linux and llama-windows and re-measure
   memory, or leave the 9B text-only and name a separate image-capable model. (Proposed:
   declare it, measure, since the projector costs ~1 GB and BookForge's page-kind use case
   is the reason the door exists.)

   **RULED 2026-09-23 (Owen):** *"lets build in the functionality. even if it cant do it on
   this card with this model specifically, crucible is designed to be cross-platform. it
   should be capable of it. we can use the highest model we can for it."* Vision is served
   for every tier. **Built:** `serves` (§2.9) and image-serving `qwen3.5-4b` / `qwen3.5-0.8b`
   on cuda-linux and llama-windows. **Not built, and why:** the 9B and 27B rows' memory is
   MEASURED text-only and clean/translate on the 3090 Ti depend on it, so their vision form
   has to be its own id (`qwen3.5-9b-vl`, `qwen3.8-27b-4bit-vl`, `qwen3.8-27b-8bit-vl`, the
   way the 27B already has two ids for two forms). But the weights store keys a download by
   MODEL ID — `weights_dir()` is `~/.crucible/models/<id>/<backend>` and `pull()`
   snapshot-downloads into it — so a `-vl` id pinning the same repo and revision as its text
   id would download the same 18–31 GB a second time (Owen: *"i dont want to have 16 copies
   of giant models"*). **Ruling owed** on how two ids share one pin (e.g. a `[model]
   weights_of = "<id>"` whose pin must match, or a store keyed by repo@revision) before the
   `-vl` ids are added. Their arithmetic is ready: the 27B's tower is the 9B's (depth 27,
   hidden 1152, read from `Qwen/Qwen3.8-27B` config.json @ 1d4bf0f2), so the 9B's 1.90 GiB
   reserve and ~0.85 GiB of tower carry across; on the 3090 Ti the 9B-vl comes to 18.18 +
   0.85 + 1.90 = 20.93 GiB of non-KV against 21.0 GiB of budget — about 2,000 tokens of KV,
   i.e. it does not usefully fit there, which the fit table would say by itself.

   **RULED AND BUILT 2026-09-23** (Owen: *"One copy on disk, two fit rows in the catalog"*):
   `[model] weights_of`, section 2.9. The aliases' memory, COMPUTED as the base's number
   plus the tower plus the 1.90 GiB (2_040_109_466 B) reserve — the tower term turning out to
   be ZERO on both 27Bs, because their base figures already hold it:

   | alias | block | estimate | how |
   |---|---|---|---|
   | `qwen3.5-9b-vl` | cuda-linux | 22_990_657_946 | base 20_950_548_480 (measured 09-12, tower resident) + 0 + reserve. Terms: weights 18_038_862_643 (calibrated 09-18, text-only) + tower 912_020_960 = 18_950_883_603; overhead 1_476_395_008 + reserve = 3_516_504_474; 40_337 B/token; 23_128_269_485 at 16384 (0.6% over) |
   | | llama-windows | 11_945_668_128 | base 11_027_502_048 + `mmproj-F16.gguf` 918_166_080 (tree API @ 3885219b), declared |
   | `qwen3.8-27b-4bit-vl` | cuda-linux | 23_673_280_922 | base 21_633_171_456 + 0 (its measured 17.68 GiB of card weights exceed the file's whole 17.29 GiB — tower resident, pre-`--language-model-only`) + reserve. Terms: 18_983_441_367 / 1_546_188_226 + reserve = 3_586_297_692 / 86_251; 23_982_875_443 at 16384 (1.3% over) |
   | | llama-windows | 18_892_047_712 | base 17_964_440_224 + `mmproj-F16.gguf` 927_607_488 (tree API @ 4ca72078), declared |
   | ~~`qwen3.8-27b-8bit-vl`~~ | ~~cuda-linux~~ | — | **Removed 2026-09-23** with the 8-bit 27B's cuda-linux arm (Owen: *"we shouldnt have an 8 bit 27b on here. waste of space, wont fit in the gpu"*): its FP8 weights alone, 28.75 GiB, exceed the card. It read 50_725_919_915 (base 48_685_810_449 + 0 + reserve). The 8-bit 27B is mlx-darwin only, where mlx-lm serves text, so it has no vision form. |

   The towers, re-read 2026-09-23: `Qwen/Qwen3.8-27B` @ 1d4bf0f2, `avyukth/Qwen3.8-27B-AWQ-INT4`
   @ 5a2ee524 and `Qwen/Qwen3.8-27B-FP8` @ 017b9c7a all state vision_config depth 27, hidden
   1152, intermediate 4304, 16 heads — the 9B's, bar `out_hidden_size` 5120 against 4096 — and
   both 27B repos' shard headers sum `model.visual.*` to 921_460_192 B (the 9B: 912_020_960).
   On the 3090 Ti's fit budget (25_757_220_864 − 3 GiB = 22_535_995_392 B) the 9B-vl's
   intercept 22_467_388_077 leaves **1,700 tokens** of KV (the "~2,000" above, to the byte);
   its `context_default` stays 16384, so the fit table refuses it there rather than a row
   serving a context no client could use. The 27B-4bit-vl's intercept exceeds that budget
   outright (as the retired 8-bit-vl's did).
4. **The Mac**: if mlx-lm 0.31.3 cannot return top logprobs, is `decide_not_served` on
   mlx-darwin acceptable for this phase, or does Crucible's own page server grow a logprobs
   path (as it grew batching)? (Proposed: refuse by name now; grow it when a Mac consumer
   exists.)

## 8a. The live pass so far (2026-09-23, Owen's go for "a basic test with a small model")

Not through the door — the running server is 1.0.23 — but through the reading the door
calls, against a REAL vLLM: `Qwen/Qwen3.5-0.8B` on vLLM 0.29.0 in Crucible's own WSL llm
env, launched by hand with the engine's environment (`VLLM_WSL2_ENABLE_PIN_MEMORY=1`,
`VLLM_USE_FLASHINFER_SAMPLER=0`; without the first it dies `UVA is not available`, exactly
as `engines/vllm.py` says) and this phase's flags, `--gpu-memory-utilization 0.15`, port
8500, with Premiere holding the rest of the card. `tools/decide_probe.py` is the driver.

- **`--logprobs-mode raw_logprobs` is what runs, and it is a distribution.** The anger
  question at temperature 0: B 0.389 / C 0.343 / A 0.267, `<think>` at 0.0002 (thinking
  is off), `label_mass` 0.999. Not one-hot. `usage.prompt_tokens_details.cached_tokens` is
  reported (the flag works).
- **The port's reader parses vLLM's reply and answers the worked example**: billing 0.694,
  anger 2.08 "Frustrated but civil", urgent 0.893, churn 0.562 — identical, to four places,
  to snap's `openai-chat` path run sequentially against the same engine. Per question
  ~57–60 ms wall on the 0.8B for ~120-token prompts (the first, 195 ms, is warm-up).
- **Batching moves the numbers.** snap at `--concurrency 16` (four questions batched by
  vLLM) gave anger 0.406 where sequential gave 0.389; team/urgent/churn unchanged. That is
  bf16 batch-composition noise, ~0.02 on this size, and a calibration built later must
  tolerate it. Not a defect of either reader.
- **vLLM's prefix cache on this hybrid model is 544-token blocks.** The engine sets
  `attention block size to 544 tokens to ensure that attention page size >= mamba page
  size` (log), so identical 121-token prompts sent three times cached 0 tokens every
  time, while a 1,345-token prompt sent three times cached 1,088 (= 2 × 544) from the
  second send (184 → 64 ms). So on vLLM the prime buys reuse in 544-token units: nothing
  for a ticket-sized state, the full blocks of a long one. The door keeps priming (one
  ~60 ms request); the win is where the state is long, which is where it matters.

Still owed: the door itself on a card (needs a cut or a branch install), host mode
(llama-server), the Mac (mlx-lm, cap 11), and Foundry's tile.

## 8. Owed after the build (the live pass)

- The card: the worked example and snap's live fixtures through `crucible api decide`
  against `qwen3.5-9b` on vLLM, with the numbers beside snap's llama-server ones; assert the
  distributions match snap's (anger 0.69 / 0.30 / 0.02 on the ticket) — the check that
  `--logprobs-mode raw_logprobs` is what runs.
- Windows host mode: the same through llama-server b10970. Its chat route's logprobs are
  VERIFIED FROM SOURCE (§1), not yet from a run: the live check is that the letters come
  back pre-sampling (a real distribution at temperature 0, not one-hot) and that a prime
  followed by questions shows the checkpoint reuse snap measured.
- **The layout of §2.5.1**: Briefcase's YTSeg smoke re-run (accuracy and speed; the mlx-lm
  log showing questions 2..N prefilling only their own tokens), and on the card and llama-server
  the same `cached_tokens` check.
- The Mac: mlx-lm 0.31.3 returns `top_logprobs` capped at 11, read from its installed
  source (§1), not yet from a run. That makes §7.4 moot for every question of 11 labels or
  fewer (yes/no, scores, choices up to K); what is left of it is only whether a 12-to-26
  option choice on the Mac stays `decide_not_served` or is served another way. Owed: one
  run, and the check that the raw-piece token strings match the bare letters there.
- Foundry's tile on the door, ~1,170 questions into a book, timed.
