# `crucible api` — driving a server from a shell

Built 2026-09-16, because there was no way to run a job from a command line. The
rest of the command line (the `crucible/cli/` package) is operator and lifecycle —
`init`, `install`, `capability`, `models pull`, `serve`, `service`,
`doctor`, `token`, `uninstall` — and **not one of them submits a job**. A
fine-tuning script that wanted a retake ladder, a cleanup pass, a translation or
a batch of renders had to write its own HTTP client. This is that client, and it
lives in `crucible/apiclient.py`.

## The two halves of one binary

| | operator verbs | `crucible api …` |
|---|---|---|
| subject | **this machine's installation** | **a server**, anywhere |
| reads | `config.toml`, units, env packs, the disk | HTTP, with a bearer token when the server asks for one |
| address | there is nothing to point them at | `--url`, `--token`, `--pairing` |
| examples | `crucible install tts`, `crucible service start` | `crucible api job submit …` |

The distinction is real and it is a namespace, not a second program. Ollama is
one binary for `ollama serve` and `ollama run` and nobody finds it confusing
(docs/INTENT.md: Crucible should be like Ollama); `crucible/launcher.py` writes
ONE shim onto PATH and `crucible/uninstall.py` removes that one, so a second
binary would be a second PATH entry an upgrade can leave stale — which
`cli_launcher_conflict` has already cost twice.

**Why the word is `api`**: `crucible models` and `crucible voices` are already
taken, by verbs that list the manifests *this build* carries. What a *server*
holds is a different fact with a different owner, and one word for both is the
shape ARCHITECTURE.md R1 exists to stop. `crucible models list` reads the build;
`crucible api models` asks a server.

## Connecting

Exactly one of three, and never a fallback chain:

```bash
# 1. this machine's installed engine — nothing to pass
crucible api info

# 2. a server anywhere: its address, and --token only if it asks for one
crucible api info --url http://192.168.68.20:7100
crucible api info --url http://192.168.68.20:7100 --token <token>

# 3. one pairing line, which carries the address AND the token (if any)
crucible api info --pairing "$(crucible token --url | head -1)"
```

A server with `[auth] mode = "none"` (docs/DESIGN.md section 9; what a new `crucible
init` writes) asks for no token, like Ollama, so `--url` alone reaches every verb.
`crucible api ping --url <url>` says which kind a server is (`auth`). `--url` without
`--token` sends **no** `Authorization` header and never borrows the local engine's
bearer: sending this machine's credential to an address somebody typed is a leak with a
one-character typo as its cause. Against a server that does ask, that is
`unauthorized`, and the message says to pass `--token`, pair, or run `crucible auth none`
on it.

Without the CLI, the same calls are plain curl. `X-Crucible-Api: 1` is the one header
every `/v1` route but `ping` needs, whatever the mode (without it: `426
api_version_required`, which names it):

```bash
curl -s http://192.168.68.20:7100/v1/ping            # {"crucible": true, ..., "auth": "none"}
curl -s -H 'X-Crucible-Api: 1' http://192.168.68.20:7100/v1/info
curl -s -H 'X-Crucible-Api: 1' -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Say hi"}]}' \
  http://192.168.68.20:7100/v1/openai/chat/completions
```

`crucible auth` prints this machine's mode; `crucible auth none` and `crucible auth token`
set it, and the running server takes it up on its next request.

There is no new config file and no new environment variable. The local case goes
through `crucible/local.py:connection`, which already owns "where is the engine
and what is its token" and already knows that on Windows the answer is the WSL
guest's pairing file rather than `config.toml`. The pairing line is parsed by
`crucible/pairing.py`.

## Output, and exit codes

**JSON on stdout, always. There is no `--json` flag.**

* one request → one indented JSON document;
* following a stream → one compact JSON object per line (JSONL), as each event
  arrives, then the subject's final state as a last line;
* `crucible api job artifact` is the one exception and writes bytes, because an
  artifact is a FLAC.

| exit | meaning |
|---|---|
| 0 | it worked |
| 1 | refused — the server's `{"error": {...}}` on stderr **verbatim, with its code**, or this CLI's own named refusal, or the address did not answer (`server_unreachable`) |
| 2 | argparse usage |

A followed job that ends `failed` or `cancelled` exits **1**, even though every
HTTP request in it succeeded. A script must not read a failed render as success.

Refusals are never paraphrased. `retake_without_band`, `band_malformed`,
`width_over_serving`, `stream_session_open`, `server_busy`, `not_resident` —
those strings are what a person greps this repo for, and a client that rewrote
them into nicer English would have deleted the only handle on them.
(`chunk_too_long` and `unknown_take` were two of them until 2026-09-19 and are
retired: the server no longer refuses a chunk by length, and a take past the
ladder is a seed lane rather than an error.)

## The commands

### Reads

```
crucible api ping              is this a Crucible, and which one (no token needed)
crucible api health            the lane, the queue, each job type's readiness
crucible api info              identity, host, job types, every capability row
crucible api setup             the pairing lines an app's connect door takes
crucible api capability        what this host can hold and why  [--accelerator-probe]
                               [--class generate --context-tokens N --concurrency N]
crucible api resolve           which model a request would get, nothing loaded
                               [--verb generate|decide|embed|rerank|image ...] [--model ID]
                               [--max-params-b B] [--form NAME] [--images N]
crucible api accelerator       what is on the card right now, probed
crucible api activity          what this server is doing, and who asked
crucible api models            the llm models this SERVER holds
crucible api voices            the tts voices this SERVER holds
crucible api catalog           every subject this backend can hold, installed or not
crucible api openai-models     the resident model plus routed upstreams, OpenAI-shaped
```

### Writes to the server itself

```
crucible api catalog-remove <kind> <id>        delete an installed subject's files
crucible api settings                          GET the settings document
crucible api settings --patch @patch.json      PUT a partial patch  [--act <class>]
crucible api upstream-test <name> [--body …]   ask an upstream what it serves
crucible api low-vram on|off|auto              set [audio] low_vram (auto: Crucible decides)
crucible api pairing-requests                  connect requests awaiting a decision
crucible api pairing-decide --id … --code … --allow|--deny
```

### Jobs

```
crucible api job submit --type <t> [--model <m>] [--params <json|@file>]
                        [--input NAME=PATH]... [--input-blob NAME=BLOB]...
                        [--resume RESUME_ID] [--max-wait SECONDS | --no-queue]
                        [--follow] [--artifacts-dir DIR]
crucible api job get <job-id>
crucible api job events <job-id> [--since <event-id>]
crucible api job cancel <job-id>
crucible api job hold <job-id>                 keep its artifacts for a later job's inputs
crucible api job release <job-id>              the chain is done; remove the job now
crucible api job artifact <job-id> <name> [--out PATH|-]
crucible api upload <path>                     → {"blob_id", "bytes", "sha256"}
```

`--params` is JSON, and that is deliberate. Every job type owns a pydantic model
with `extra="forbid"` (`TtsParams`, `AsrParams`, `RvcParams`, …) and those models
are the contract; argparse flags mirroring them would be a second owner of every
field and would refuse bodies the server would have taken. A new job type needs
no change to the CLI.

`--input NAME=PATH` uploads the file through `POST /v1/uploads` and gives the job
the blob. `inline_base64` exists on the wire and is not offered: it puts the
bytes through the request body to save one round trip, and every real input here
is audio.

`--artifacts-dir` **requires `--follow`** and is refused without it
(`artifacts_need_follow`). Downloading artifacts means waiting, and a flag that
silently decides whether a command blocks for twenty minutes is a surprise.

A busy server holds the job in its queue (docs/QUEUE.md), up to an hour;
`--max-wait SECONDS` changes that, and `--no-queue` asks to be refused `server_busy`
at once instead. `chat`, `decide` and `align` take the same two flags.

### The queue (2026-09-30)

```
crucible api queue list                        → {"items": [...], "depth", "limits"}, in order
crucible api queue remove <job-id>             the job ends removed, reason operator
crucible api queue heartbeat <job-id>          the client that queued it is still here
crucible api queue events                      a snapshot, then every change; Ctrl-C ends it
```

### Every change on the server (2026-10-01)

```
crucible api events [--topics job,card] [--since <event-id>]
                                               a snapshot, then every change; Ctrl-C ends it
```

One line per event, from GET /v1/events. The names, payloads and topics are in
docs/EVENTS.md.

### Resumable jobs — the journals (2026-09-27)

```
crucible api resumable list                    → {"resumable": [...]}, newest first
crucible api resumable get <resume-id>
crucible api resumable discard <resume-id>     refused resume_in_use while a job writes it
```

A job type that keeps a journal (`asr` on Qwen3-ASR, for now) answers
`resume_id` beside `job_id`. If the job fails, is cancelled or the server stops
under it, submit the SAME job again with `--resume <resume-id>` (it is
`params.resume`): the finished pieces are read back instead of redone. Without
`--resume` a job always starts fresh. docs/RESUMABLE-JOBS.md is the contract.

### Tasks — work done *to* the server

```
crucible api task submit --type pull --kind model --id qwen3.5-9b [--follow]
crucible api task submit --type install --job-type tts [--narrator-engine higgs-v3]
crucible api task submit --type module --module @module.json
crucible api task submit --type engine --target <target>
crucible api task submit --type engine-restart
crucible api task list | get <id> | events <id> [--since N] | cancel <id>
```

### Serialized TTS — a session, one row at a time

```
crucible api stream open --voice sigma --language en        → {"session_id", …}
crucible api stream say <sid> --row r12 --text "…" --take 0
crucible api stream say <sid> --row r12 --text-file row.txt --take 1
crucible api stream events <sid> [--since N] [--until r12] [--audio-dir DIR]
crucible api stream cancel <sid> --row r12
crucible api stream cancel-all <sid>
crucible api stream close <sid>
```

The session id survives the process, which is what makes six separate commands a
usable shape. `--take` has no default, exactly as the wire has none: a default
would be a render at a rung nobody chose.

`--until <row>` returns as soon as that row is `done` or `error`, which is the
moment a ladder script has something to judge; without it the follow runs to
`closed`, which for a session you are still feeding never comes.

`--audio-dir` writes `<row>.wav` at **the rate on the session's `ready` frame** —
the engine's own, which the load already checked against the manifest. Resuming
with `--since` past that frame is refused rather than guessing a rate
(`audio_needs_the_ready_frame`); decode `pcm_base64` yourself in that case.

### Chat — the chat door

```
crucible api chat --model qwen3.5-9b --message "…"           one-shot shorthand
crucible api chat --max-params-b 4 --message "…"             the server's chat model, at most 4B
crucible api chat --verb generate --message "…"              the server's chat model (the default verb)
crucible api chat --body @request.json [--stream] [--act clean]
crucible api session open --act generate --max-params-b 4    open on the server's chat model, at most 4B
```

The body is forwarded **verbatim** in both directions except `model`, so
`chat_template_kwargs`, `max_tokens` and everything else are the engine's
vocabulary and belong in `--body`, not in flags this CLI would have to keep up
with. Thinking is `"chat_template_kwargs": {"enable_thinking": false}`; a
top-level `"thinking"` member is in no engine's request and is refused
`400 thinking_field_unknown` before anything loads (for `anthropic/<id>`,
`reasoning_effort` is the knob). A thinking model's thinking comes back as
`message.reasoning` (`delta.reasoning` streamed), apart from `content`; on
cuda-linux that is vLLM's reasoning parser, derived from the manifest's
`[reasoning]` (docs/internals/engines-and-capability.md, "Reasoning").

> **The route is `POST /v1/openai/chat/completions`** (also mounted at
> `/openai/v1/chat/completions` for OpenAI clients). There is **no bare
> `POST /v1/chat/completions`** on this server — checked against `crucible/api.py`
> and against the live engine on 2026-09-16.

### The anthropic upstream

A chat whose `model` is `anthropic/<id>` is not forwarded verbatim: the OpenAI body
is translated to Anthropic's Messages API (`POST /v1/messages`, `anthropic-version:
2023-06-01`, the key the server holds) and the answer is translated back. Crucible's
part is serving, so the rule is: **translate faithfully, refuse by name what cannot be
honoured, never drop a member silently.** Everything is checked before anything is
sent, so a refusal costs nothing.

```
crucible api chat --body @request.json [--stream]      # with "model": "anthropic/claude-opus-5-5"
python scripts/check-anthropic-live.py --url http://127.0.0.1:7100 --model claude-opus-5-5
```

What crosses, and as what (`crucible/upstreams.py`, `_anthropic_document`):

| OpenAI member | Messages API | Notes |
|---|---|---|
| leading `system` / `developer` messages | `system` | joined by a blank line; text parts only |
| `user` / `assistant` messages | `messages` | string content, or `text` and `image_url` parts |
| `image_url` `data:image/...;base64,` | `image` block, `source.type: base64` | the media type is the URL's |
| `image_url` `http(s)://` | `image` block, `source.type: url` | Anthropic fetches it; `detail` other than `auto` is refused |
| `max_tokens` / `max_completion_tokens` | `max_tokens` | one, not both; a positive integer; neither sent: **4096**, said as `"max_tokens": "upstream default 4096"` in `X-Crucible-Sampling` |
| `stop` | `stop_sequences` | |
| `temperature`, `top_p`, `top_k` | the same | only on models that take them (table below) |
| `reasoning_effort` | `output_config.effort` | `low`, `medium`, `high`, `xhigh`, `max`, each as itself |
| `response_format` `json_schema` | `output_config.format` `{type: json_schema, schema}` | structured output, not a forced tool; `name` and `strict` are read, the answer is the text block |
| `response_format` `text` | nothing | the default |
| `user` | `metadata.user_id` | |
| `tools` (`type: function`) | `tools` `{name, description, input_schema, strict}` | `parameters` is `input_schema`; a function with none is sent `{type: object, properties: {}}`; `strict` crosses as itself |
| `tool_choice` `auto` / `none` / `required` / `{type: function, function: {name}}` | `tool_choice` `{type: auto}` / `{type: none}` / `{type: any}` / `{type: tool, name}` | `required` and a named function only on models that keep forced tool use (table below); with no tools, `auto` and `none` send nothing |
| an assistant message's `tool_calls` | `tool_use` blocks `{id, name, input}` after its text block | `arguments` (a JSON string) is parsed and must be an object; empty or `null` content sends no text block |
| `tool` messages `{tool_call_id, content}` | ONE `user` turn of `tool_result` blocks `{tool_use_id, content}` | consecutive results merge into one turn (the Messages API answers every `tool_use` in the next turn); string content as is, text parts as text blocks; the id must name a call an earlier assistant message made |
| `n: 1` | nothing | one answer is all the Messages API writes |
| `stream`, `stream_options.include_usage` | `stream` | usage is a chunk of its own (below) |
| `chat_template_kwargs.enable_thinking` | nothing | dropped and said: `"thinking": "dropped"` in `X-Crucible-Sampling` |
| nothing | `thinking` `{type: adaptive, display: summarized}` | sent to the models that think by default (the table's Summaries column), so their thinking comes back as a summary; never `{type: disabled}` (a 400 on Opus 5.5 and Sonnet 5.5) |
| an assistant message's `content_blocks` | the turn's `content`, verbatim | what a reply carried, sent back unchanged: the turn as Claude wrote it, its thinking, text and `tool_use` blocks in order. Its text blocks joined must be the message's `content` and its `tool_use` blocks (id, name, input, in order) its `tool_calls`, or it is refused `400 content_blocks_mismatch` (`details.differs_in`), never rebuilt; a block of another type is `400 invalid_request` |
| an assistant message's `reasoning` | nothing | the readable summary; `content_blocks` carry the thinking |
| `context_tokens` | nothing | dropped and said: `X-Crucible-Context` source `dropped` |

Refused, `400 upstream_field_unsupported` with `details.fields` (the paths) and
`details.carried` (every member read): any member not in the table (`seed`,
`frequency_penalty`, `presence_penalty`, `logit_bias`, `logprobs`,
`parallel_tool_calls`, `functions`, `repetition_penalty`, …); `n` above 1; a `system`
or `developer` message after the first user or assistant turn; `function` messages and a
message's `name` (a tool message's too); a tool other than `type: function`, and a
member of a tool, function or tool call outside the table; a content part other than
`text` and
`image_url`; an image `detail` other than `auto`; a `json_schema.description` (it would
be lost: put it in the prompt); a `chat_template_kwargs` key other than
`enable_thinking`; a `stream_options` key other than `include_usage`.

Refused, `400 upstream_param_unsupported` with `details.param`, `details.model`,
`details.rule` (the table row that refused it) and, for sampling, `details.value`, for
effort, `details.takes`:

- a sampling member on a model that takes none. The one value those models accept,
  `temperature: 1` (Anthropic's default), is forwarded; any other temperature, and any
  `top_p` or `top_k`, is refused;
- a `reasoning_effort` the model does not take, and `none` or `minimal` on any model
  (the Messages API's lowest is `low`, and effort does not turn thinking off);
- `response_format` `json_object`: the Messages API's structured output takes a schema
  and has no schema-less JSON mode. Send `json_schema`, or ask for JSON in the prompt;
- `response_format` `json_schema` on a model with no structured output (no current row);
- `tool_choice` `required` or a named function on a model that removed forced tool use
  (Fable 5.1, Mythos 5.1, Opus 5.5, Sonnet 5.5: Anthropic answers `any` and `tool` with
  a 400). Send `auto` and say in the prompt which tool to call.

Malformed tool calling is `400 invalid_request` naming the field: a `tool_choice` that
is none of the four forms or names no declared tool, `required` with no tools, two tools
with one name, a tool name outside `[A-Za-z0-9_-]{1,64}`, `arguments` that are not a
JSON object, a `tool_call_id` no earlier assistant call made, `tool_calls` on a message
that is not the assistant's.

The model table (`ANTHROPIC_MODEL_RULES`), keyed by model-id prefix, the longest
matching row winning (`claude-opus-4-5-20251101` is the `claude-opus-4-5` row). It was
read from Anthropic's documentation on 2026-10-12 — [effort](https://platform.claude.com/docs/en/build-with-claude/effort.md),
[structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs.md)
and the [Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide.md),
[Sonnet 5.5](https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide.md),
[Haiku 5.5](https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide.md) and
[Fable 5.1](https://platform.claude.com/docs/en/models/fable-5-1/migration-guide.md)
migration guides. A new model is one row.

| Row | Sampling | Structured output | Effort levels | Forced tool use | Summaries |
|---|---|---|---|---|---|
| `claude-fable-5-1`, `claude-mythos-5-1` | no | yes | low, medium, high, xhigh, max | no | yes |
| `claude-fable-5`, `claude-mythos-5` | no | yes | low, medium, high, xhigh, max | yes | yes |
| `claude-opus-5-5` | no | yes | low, medium, high, xhigh, max | no | yes |
| `claude-opus-5` | no | yes | low, medium, high, xhigh, max | yes | yes |
| `claude-opus-4-8`, `claude-opus-4-7` | no | yes | low, medium, high, xhigh, max | yes | no |
| `claude-sonnet-5-5` | no | yes | low, medium, high, xhigh, max | no | yes |
| `claude-sonnet-5`, `claude-haiku-5-5` | no | yes | low, medium, high, xhigh, max | yes | yes |
| `claude-opus-4-6`, `claude-sonnet-4-6` | yes | yes | low, medium, high, max | yes | no |
| `claude-opus-4-5` | yes | yes | low, medium, high | yes | no |
| `claude-sonnet-4-5`, `claude-haiku-4-5` | yes | yes | none | yes | no |

Summaries (`adaptive_by_default`): the model runs adaptive thinking when a request names
none, so Crucible sends `thinking: {type: adaptive, display: summarized}` — the same
thinking, with a readable summary instead of the default empty text (`display:
omitted`), billed the same. The other rows think only when asked (Opus 4.6–4.8,
Sonnet 4.6) or only by `budget_tokens` (the 4.5s), and are sent no `thinking`.

**A model no row names is forwarded as asked**, and Anthropic answers for it: what it
refuses comes back as `502 upstream_rejected` in Anthropic's own words. Sending
`reasoning_effort` with no row still checks the level is one of the five.

Effort defaults to the model's own when unsent (`high`; `medium` on Opus 5.5 and
Haiku 5.5). Thinking is not turned off: the models in the Summaries column think
adaptively by default and are asked for their summary, and a thinking model spends part
of `max_tokens` thinking — send a `max_tokens` that leaves room, because 4096 can end
in `finish_reason: "length"` before any text.

The answer: the text blocks joined are `content`; `stop_reason` `end_turn` and
`stop_sequence` are `stop`, `max_tokens` and `model_context_window_exceeded` are
`length`, `refusal` is `content_filter`, `tool_use` is `tool_calls`, and any other
reason is relayed as Anthropic said it. Each `tool_use` block is one of
`message.tool_calls` (`{id, type: function, function: {name, arguments}}`, `arguments`
its `input` as a JSON string), in order; an answer that only called tools has
`content: null`. The `thinking` blocks' text, joined by a blank line, is
`message.reasoning` (absent when it is empty, as without summaries). A turn that
thought also carries `message.content_blocks`: its `thinking`, `redacted_thinking`,
`text` and `tool_use` blocks, in Claude's order and exactly as it wrote them
(signatures and all). `usage.prompt_tokens` is every prompt token (Anthropic's
`input_tokens` plus its cache writes and reads), `usage.prompt_tokens_details.cached_tokens`
the cache reads.

Streaming: text deltas become content chunks; a `tool_use` block's
`content_block_start` is a `delta.tool_calls` entry `{index, id, type: function,
function: {name, arguments: ""}}` and each `input_json_delta` one more `{index,
function: {arguments: <piece>}}` (a call with no input gets `"{}"` at its
`content_block_stop`), `index` counting the calls from 0; each `thinking_delta` is a
`delta.reasoning` chunk; a `signature_delta` is never shown, it completes its block;
the finish chunk (the one carrying `finish_reason`) carries the turn's
`delta.content_blocks`, assembled from the stream in Claude's order (thinking with its
signature, text, each call's input parsed at its `content_block_stop`), when the turn
thought; a call whose input did not arrive as JSON leaves the turn without them; `ping`
and event types this server does not know are skipped.

Tool calling holds no state: Crucible stores no tools and remembers nothing between
rounds. The app sends its tools and the whole conversation every round (its question,
the assistant's `tool_calls`, its own `tool` results), runs the tools itself and decides
when the loop ends. To keep Claude's thinking across the loop, send the assistant
message back as its reply carried it: its `content`, `tool_calls` and the
`content_blocks`, unchanged. The Messages API needs a thinking turn back exactly as it
was (history append-only; Opus 5.5 and Sonnet 5.5 write thinking blocks between their
tool calls, and a moved block invalidates them), so Crucible sends `content_blocks` as
the turn's content verbatim, in their order, after checking them against `content` and
`tool_calls`. `content_blocks` is taken off every message of a chat to any other model.

The `ollama` upstream carries the same tool calling in Ollama's native form: `tools` as
OpenAI's function shape (`parameters` defaulted as above; `strict: true` is refused
`upstream_field_unsupported`, Ollama has no strict schemas), an assistant's
`tool_calls` as `{function: {name, arguments: <object>}}`, a `tool` message as
`{role: tool, content, tool_name}` (the name of the call it answers). Ollama has no
`tool_choice`: `auto` (or none) is accepted, `none`, `required` and a named function are
refused `400 upstream_param_unsupported` with `details.upstream: "ollama"`. Its calls
come back as `message.tool_calls` with `arguments` a JSON string and Ollama's id, or
`call_<n>` (the call's place in that answer) when it gave none; streamed, each call
arrives whole as one delta; `finish_reason` is `tool_calls` when the answer called a
tool (Ollama says `stop`). With `stream_options.include_usage: true` the
stream ends, after the chunk carrying `finish_reason` and before `[DONE]`, with one
chunk whose `choices` is empty and whose `usage` is counted as above from
`message_start` and `message_delta` (cumulative). Without it no usage is sent, as
OpenAI does. An `error` event is relayed as `data: {"error": {type, message, upstream:
"anthropic"}}`, and a stream that ends before `message_stop` gets an error frame
saying it is truncated; neither gets `[DONE]`.

Billing and retries: a chat to `anthropic` or `openai` is sent again only when the
connection was never made (`ConnectError`, `ConnectTimeout`: five attempts over ~7.5 s),
so no byte reached the provider. Once the request may have arrived, a lost connection is
`502 upstream_unreachable` saying it was not sent again; the caller decides. A `429` is
passed back with `Retry-After` and never resent.

What only a live run confirms: `scripts/check-anthropic-live.py` sends one plain chat,
one streamed chat with usage and one JSON-schema chat, each with a small `max_tokens`,
through a server that already holds the key.

### Embed and rerank — the retrieval verbs

```
crucible api embed --input "The capital of China is Beijing." --input @notes.txt [--input-type document]
crucible api embed --input-type query --input "What is the capital of China?"   [--instruction "Given a question, find passages that answer it"] [--dimensions 1024]   [--encoding-format base64_float16] [--fingerprint <a stored fingerprint>]
crucible api rerank --query "What is the capital of China?"   --document "The capital of China is Beijing." --document "Gravity pulls." [--model qwen3.5-9b]
```

`POST /v1/embed` and `POST /v1/rerank`, the optional retrieval package's verbs
(`crucible install retrieval`; docs/RETRIEVAL.md is the guide for app authors). `--input` and
`--document` repeat and take text or `@file`. `rerank --model <a decide model>` reranks with
Crucible's general template and needs no package. The OpenAI- and Cohere-shaped twins
(`/v1/openai/embeddings`, `/v1/openai/rerank`) answer the same handlers for existing client
libraries; this CLI sends the native routes.

### Decide — the decision door

```
crucible api decide --model qwen3.5-9b --state "I was charged twice for March, please refund one." \
  --choice team "Which team should handle this?" billing="Payment and invoice issues" technical="Bugs and errors" \
  --score anger "How frustrated is the customer?" "Calm,Frustrated but civil,Very angry" \
  --yesno urgent "The message conveys urgency" [--act analysis] [--missing report]
crucible api decide --model qwen3.5-9b --state @ticket.txt --yesno refund "The customer asks for a refund"
crucible api decide --model <image-capable id> --image page.png --choice kind "What kind of page is this?" chapter="A chapter opening" body="Running prose"
```

`POST /v1/decide` (history/PHASE22-DECIDE.md, 2026-09-23): a probability distribution over
each question's fixed answers, read off one forward pass of the resident model. The
grammar is **snap's own** (`snap decide`), so a person moving from snap types the
same thing against a different address. `--choice`, `--score` and `--yesno` repeat
and keep the order they were typed in; `--image` repeats and is sent base64-encoded;
`--state` may be left out when an image is given, and the state is then `""`.
`--model` may be left out: the server then decides on the model it registered for
`decide`, or with `--image`, the one it registered for a decision with images
(`crucible api capability`, the decide row's `with_images`).

The prompt, the letters and the renormalisation are the server's — this verb sends
the order and prints the answer unchanged. Like chat it never loads a model
(`model_not_resident`), and the counts (2–26 options, 2–10 levels, 8 images) are the
server's to refuse (`invalid_request`, `too_many_options`, `too_many_images`), so
they are not copied here.

`--missing report` sends `"missing": "report"`: a label outside the engine's top-K is
then `null` in `probabilities`/`logprobs` and named in the answer's `missing_labels`,
and the rest renormalise over the letters returned, where the default (`refuse`, the
server's, so the flag is not sent when left out) makes the decision
`502 label_not_in_probs`. The word travels unchecked; any other is the server's
`400 invalid_request` naming `missing`.

### Align — known words, placed in time

```
crucible api align --model qwen3-aligner --language en \
  --window 0 @window-000.txt window-000.flac \
  --window 1 @window-001.txt window-001.flac --follow --out alignment.json
```

The `align` job (history/PHASE4-AUDIO.md section 2) behind a verb of its own (Owen,
2026-09-24): transcription's sibling, where the caller supplies the words. Every
`--window` is uploaded and named `<index>.<ext>` and they all go in ONE job, so the
aligner loads once and a window that fails is reported alone. A window is at most
300 s; cutting a long recording into windows is the caller's logic (typically from an
`asr` transcript's word times). `--out` saves `alignment.json` (per window, `items`
of `{text, start, end}` in seconds from that window's start, or `error`) and needs
`--follow`.

### Queue sessions

```
crucible api session open --act clean [--model qwen3.5-9b] [--idle 600] [--max-wait 3600]
crucible api session get <session-id>
crucible api session events <session-id> [--since EVENT_ID]
crucible api session touch <session-id>
crucible api session close <session-id>
```

`open` answers the ticket at once (`status` `open`, or `queued` with a `position`); `events`
follows the session until `opened`, then `closed` (or `removed`). While it is open every
other `crucible api` verb this client sends is one of its items. docs/QUEUE.md has the rules.

## What a fine-tuning script calls

### A batch of renders (the render door, docs/history/PHASE3-TTS.md section 6)

`params.json`:

```json
{
  "language": "en",
  "take": 0,
  "chunks": [
    { "index": 41, "text": "He had been walking for some time." },
    { "index": 42, "text": "The road did not appear to end." }
  ]
}
```

```bash
crucible api job submit --type tts --model deathstalker \
  --params @params.json --follow --artifacts-dir ./out
# ./out/41.flac, ./out/42.flac — mono 24 kHz PCM_16, plus provenance sidecars
```

`--model` is the **voice id**: for Higgs a voice *is* the merged checkpoint, so
the wire's word for "the thing that makes the bytes" is `model` and no new
vocabulary was invented.

### The retake ladder

A take is a rung the server declares (`takes` on `GET /v1/voices`) and the
judgment is the client's. Serially:

```bash
sid=$(crucible api stream open --voice sigma --language en | jq -r .session_id)
for take in 0 1; do
  crucible api stream say "$sid" --row row-1 --take "$take" --text "$SENTENCE"
  crucible api stream events "$sid" --until row-1 --audio-dir ./takes/$take
  # judge ./takes/$take/row-1.wav; break when it is good
done
crucible api stream close "$sid"
```

Or in batch — re-submit only the failing indices at the next take, which keeps
`<index>.flac` lined up with the client's own numbering:

```bash
crucible api job submit --type tts --model sigma \
  --params @retakes-take-1.json --follow --artifacts-dir ./out
```

A take past the end of a voice's ladder renders at the voice's own sampling in
that take's seed lane (2026-09-19) and is never clamped — take 4 is never take
2's numbers under take 4's name.

`params` may also carry `retake` (which ARM renders the batch), `band` (the
three rates the guarded arm measures against — required when `retake` is true,
`retake_without_band` otherwise) and `width` (how many chunks are in flight).
A `width` you do not send is **not sent on to narrator either**, and the engine
then renders at the width it was STARTED at — `HIGGS_MAX_NUM_SEQS` from
`[voice.serving].max_num_seqs` on a `cuda-linux` server, `NARRATOR_HIGGS3_MLX_BATCH`
off the measured tier table on a Mac. On `cuda-linux` a `width` above
`max_num_seqs` is `width_over_serving` from this server; on `mlx-darwin` the
stated width travels and narrator answers for the width it actually has.
The job's `done` reports the width **you stated**, and `null` when you stated
none — never a number this server invented. The `done` also names the FULL
sampling triple the engine applied and the weights it ran on, so a screening
record is self-describing.

### AI cleanup and translation

The model must already be resident — this CLI never loads one implicitly, and
neither does the chat door:

```bash
crucible api job submit --type load-model --model qwen3.5-9b --follow
crucible api chat --act clean --body @cleanup-request.json
crucible api chat --act translate --body @translate-request.json
crucible api job submit --type unload-model --model qwen3.5-9b --follow
```

A load may start the engine at a longer context than the model's default, up to
this host's ceiling (`crucible api capability`'s `generate` row lists it per
model); above it the submit is refused `context_over_limit` before anything is
evicted. Loading the resident model again at a new context is a reload:

```bash
crucible api job submit --type load-model --model qwen3.5-9b --params '{"context": 65536}' --follow
```

Send `"chat_template_kwargs": {"enable_thinking": false}` in the body (the SDK's
`thinking: false`) for a cleanup pass: Qwen3.5 otherwise spends a bounded budget
entirely on reasoning (measured; the same rule BookForge's `crucible` provider
follows). A top-level `"thinking": false` is refused `thinking_field_unknown`.

### The other job types, as bodies

| type | `--model` | `--params` | inputs |
|---|---|---|---|
| `echo` | — | `{"delay_ms": 25}` | any; each is copied to an artifact of the same name |
| `tts` | voice id | `{"language","take","chunks":[{"index","text"}]}` | none |
| `load-voice` | voice id | `{"timeout_s": …}`, plus `reference` for a `zeroshot` voice | none |
| `unload-voice` / `unload-model` / `unload-aligner` / `unload-denoiser` / `unload-image` / `unload-audio` / `unload-segment` / `unload-video` / `unload-score` | id | `{}` | none |
| `load-model` | model id | `{"timeout_s": …, "context": …}` — every key optional; `context` is refused above this host's ceiling (`context_over_limit`) | none |
| `asr` | asr id | `{"language","vad_filter","word_timestamps"[, "initial_prompt","context","piece_s","overlap_s","speech_only","speech_threshold","speech_pad_s","speech_min_gap_s"]}` | exactly one audio file; `"auto"` is a language (whisper only); `initial_prompt` is whisper's, `context`/`piece_s`/`overlap_s` Qwen3-ASR's; `speech_only` (default false, 2026-09-27) takes stretches without speech out first and lists them in the transcript's `removed`; its three knobs default to 0.3 / 0.3 s / 2 s and are refused without it |
| `align` | aligner id | `{"language","chunks":[{"index","text"}]}` | one per chunk, named `<index>.<ext>` |
| `align-longform` | aligner id | `{"language","sentences":[{"index","text","kind"}],"rough_model","chunk_s",…}` | exactly one audio file — the whole audiobook |
| `rvc` | rvc voice id | `{"index_rate","protect_rate","n_semitones"[, "f0_method","hop_length","piece_s","overlap_s","crossfade_s","output_rate","output_channels"]}` | many, of any length and format; names need no extension |
| `denoise` | separator id | `{}`, or `{"stems": "all"}` to get every stem back (`vocals-roformer`: the vocals AND the instrumental) instead of only the primary, `"format"`: `"wav"` (default), `"flac"` or `"mp3"` for every stem, and `"hop_s"`: seconds between the separator's windows, 0.5 up to its default of 8 (smaller is slower and smoother; `denoise_hop_out_of_range` outside) — every other separation knob is the engine's | exactly one audio file in anything ffmpeg decodes, at any rate (resampled to the separator's 44.1 kHz and back); a held job's artifact by reference works |
| `image` | image model id | `{"prompt"[, "negative_prompt","width","height","seed","steps","guidance","image_strength","mask","mask_blur"]}` — see docs/IMAGE.md | none; one image with `image_strength`; or, for inpainting and outpainting, the image and a mask, `mask` naming the mask input |
| `load-image` | image model id | `{}` — warms the model up before the first prompt | none |
| `audio` | audio model id | `{"prompt"` (sound effects, music) or `"tags","lyrics"` (songs), or `"tags","instrumental": true[, "planning_lyrics"]` (an instrumental song, planned from words it never sings: the client's, or a pool set picked by the seed), or a cover: `"tags","score","instrumental": true` or `"tags","score","lyrics"` (YuE2 renders the given score: played, or sung with the client's words)`[, "chords","duration_s","steps","cfg","negative_tags","negative_lyrics","key","meter","bpm","seed","format"]}` — which optional params a model takes is in its manifest; `negative_tags`/`negative_lyrics` (YuE2, cfg above 1 only, else `audio_param_conflict`) write the CFG negative branch the song steers away from, the client's own words (`done.audio.negative`, `negative_prefix_tokens`); `key` (`"Em"`, `"F#m"`, `"Bb"`: ABC's 30 major and minor keys), `meter` (`"3/4"`, `"6/8"`) and `bpm` (40-240) (YuE2, a planned song only; beside `score` `audio_param_conflict`) fix the score's `K:`, `M:` and `Q:1/4=` lines in its one scoring pass (`done.audio.header`: as fixed, and under `score` what the score states); see docs/AUDIO.md "Covers", "The negative branch" and "Key, meter and tempo" | none, except a cover: exactly the one input `score` names — uploaded, or `{"artifact": {"job_id","name"}}` of a held job (e.g. the `score` job's `score.abc`) |
| `load-audio` | audio model id | `{}` — warms the model up | none |
| `segment` | segment model id | `{}` for `birefnet` (the subject, by itself); `{"points": [{"x","y","label"}], "box": [x0,y0,x1,y1]}` (either or both) for `sam2.1-hiera-large` — see docs/SEGMENT.md | exactly one PNG, JPEG or WebP; out come `mask.png` and `cutout.png` at its size |
| `load-segment` | segment model id | `{}` — warms the model up | none |
| `video` | video model id | `{"prompt"[, "width","height","duration_s" or "num_frames","fps","seed","steps","audio"]}` — every limit is refused by name; see docs/VIDEO.md | none (text-to-video) or exactly one PNG/JPEG/WebP (image-to-video, its first frame) |
| `load-video` | video model id | `{}` — warms the model up | none |
| `score` | score model id (`sheetsage2`) | `{}`, or any of `{"keep_harmony": true, "start_s": …, "max_seconds": …}` — `keep_harmony` keeps the chords (SheetSage2's `full` task; default `melody-full`); `start_s`/`max_seconds` crop; see docs/SCORE.md | exactly one recording in any format ffmpeg decodes, at most 360 s after the crop; out come `score.abc` (YuE2-native), `instrumental.abc`, `sheetsage2.abc` (SheetSage2's own), `transcription.mid` and `score.json` |
| `load-score` | score model id | `{}` — puts SheetSage2 on the card | none |

**`denoise` keeps the input's timeline.** A stem has the input's sample rate and exactly its
frame count, whatever was sent, because a client slices the stem at the input's sample offsets.
A WAV or FLAC at the separator's own rate (44.1 kHz, every book block) goes to it as sent and its
stems are the separator's own WAV, untouched. Anything else (a YuE2 song at 48 kHz in FLAC or mp3,
an m4a, a video's sound) is decoded with the server's ffmpeg, resampled to 44.1 kHz with soxr
(VHQ), separated, and every stem resampled back to the input's rate and trimmed or padded by the
resampler's rounding (a frame or two) to the input's frames. The `done` event's `input` and
`separated` say which happened (`separated.said` in a sentence). The separator always writes
lossless WAV (`crucible/jobs/denoise`, `SEPARATOR_FORMAT`) and the worker checks its extension;
`format` then encodes each stem once, as an audio job encodes a song: `wav` (the default, so
every client that sends no `format` gets what it always got), `flac` (24-bit) or `mp3` (192 kbps
CBR). A stem the separator finds silent (a clean instrumental's vocals: audio-separator logs
"near-silent" and writes no file) comes back as digital silence on the input's timeline and is
named in `separated.silent_stems`; a stem missing for any other reason still fails the job by
name (`denoise_primary_stem_missing`). (Through 1.0.38 a freshly loaded separator could write FLAC despite the request, because
audio-separator's model instance kept its load-time format; fixed 2026-09-25.)

**Chaining a song into denoise (B-Sides).** Render with `"hold": true`, then name the song as
denoise's input by reference — the bytes never leave the server — and release the hold when
the stems are fetched:

```json
{"type": "audio", "model": "yue2-3b", "hold": true,
 "params": {"tags": "...", "lyrics": "...", "instrumental": true, "format": "flac"}}

{"type": "denoise", "model": "vocals-roformer",
 "params": {"stems": "all", "format": "mp3"},
 "inputs": {"song.flac": {"artifact": {"job_id": "<the audio job>", "name": "audio.flac"}}}}
```

The stems are named after the input's name: `song_(vocals)_vocals_mel_band_roformer.mp3` (the
primary) and `song_(other)_vocals_mel_band_roformer.mp3` (the instrumental). Then
`DELETE /v1/jobs/<the audio job>/hold`. Without the hold, fetching every one of the song's
artifacts lets it go, and a reference after that is `409 artifact_expired`
(`tests/test_denoise_chain.py`).

**`hop_s`: the windows, and what they cost (2026-10-11).** A separator runs its model on one
window at a time (11 s; 8 s for `denoise-roformer` and `bleed-suppressor-roformer`) and starts
one every `hop_s` seconds, averaging the overlaps with a Hamming weight. At the default 8 an
11 s model covers each sample about 1.4 times, so a seam or a burst of bleed lands every 8 s
and is heard there. A smaller hop averages more windows over every sample: `hop_s: 2` covers
it 5.5 times and takes about 4x as long (8 s of separating for a 224 s song at the default on
the 3090 Ti; 98 s against 50 s at 4 against 8 on the Mac). It runs from 0.5 to the
separator's own default, which is the largest because nothing past it is offered; outside
that range it is `400 denoise_hop_out_of_range` with `details` `{hop_s, min, max, model}`. The
hop is the job's, not the load's: one resident separator takes jobs at different hops with
no reload. The `done` event's `separated` carries `hop_s` (what ran), `hop_s_default`,
`window_s` (the model's, from its config) and `windows_said`. There is no test-time
augmentation for these models in audio-separator 0.31.1, so the hop is the knob.

**The separators and their stems.** `primary_is` and `secondary_is` in the `done` event say
what each stem is, from the manifest; the names are audio-separator's, from each model's
config (`<input>_(<stem>)_<checkpoint>.<ext>`):

| separator | primary (kept by `"stems": "primary"`) | the other stem | window |
|---|---|---|---|
| `vocals-roformer` (Kimberley Jensen) | `(vocals)`, the vocals | `(other)`, the instrumental | 11 s |
| `instrumental-roformer-becruily` | `(Instrumental)`, the instrumental (its target) | `(Vocals)`, the mix minus the instrumental | 11 s |
| `instrumental-roformer-unwa-v1e` | `(other)`, the instrumental (its target, which its config calls `other`) | `(vocals)`, the mix minus the instrumental | 11 s |
| `bleed-suppressor-roformer` (unwa, 97chris) | `(Instrumental)`, the instrumental cleaned | `(Bleed)`, the voice it took out | 8 s |
| `denoise-roformer` (aufr33) | `(dry)`, the dry speech | `(other)`, what was taken out | 8 s |

An instrumental-target model is the one to use when the instrumental is what is kept: its
model was trained on the instrumental, where `vocals-roformer`'s instrumental is the mix minus
its vocals estimate. The bleed suppressor is a second pass: send it the instrumental another
separator made, by reference to that denoise job's stem (`"hold": true` on the first job), and
keep its `(Instrumental)`. Which stem is primary is checked on the LAST parenthesised label of
each name, so an input already named `song_(Instrumental)_…` does not count twice. Mixing two
separators' instrumentals, or gating a stem back against the mix, is the client's.

```json
{"type": "denoise", "model": "instrumental-roformer-becruily", "hold": true,
 "params": {"hop_s": 2, "format": "flac"},
 "inputs": {"song.flac": {"artifact": {"job_id": "<the audio job>", "name": "audio.flac"}}}}

{"type": "denoise", "model": "bleed-suppressor-roformer",
 "params": {"hop_s": 2, "format": "flac"},
 "inputs": {"song.flac": {"artifact": {"job_id": "<the becruily job>",
   "name": "song_(Instrumental)_mel_band_roformer_instrumental_becruily.flac"}}}}
```

**Licences: Owen must rule before an app ships these.** The three separators added 2026-10-11
(`instrumental-roformer-becruily`, `instrumental-roformer-unwa-v1e`,
`bleed-suppressor-roformer`) are pinned from `Politrees/UVR_resources` at `929e057b`, the
mirror `vocals-roformer` and `denoise-roformer` come from. Their authors' own repos
(`becruily/mel-band-roformer-instrumental`, `pcunwa/Mel-Band-Roformer-Inst`; no HuggingFace
repo of the bleed suppressor's authors was found) state NO licence; the mirror is MIT-tagged
as a whole, which is not the authors granting one. Each manifest's `licence` says exactly
that, and `crucible denoise list --json` shows it. The becruily and v1e checkpoints are
byte-identical to their authors' (the same LFS sha256), and their configs have the same git
object id. `vocals-roformer`'s author repo is tagged MIT; `denoise-roformer`'s upstream was not
checked.

`protect_rate`'s scale is **inverted**: lower protects more and 0.5 turns
protection off. The tuned deathstalker→Sigma recipe is `index_rate 0.3`,
`protect_rate 0.1`, `n_semitones -2`, `f0_method rmvpe`.

**`rvc` returns each input's own format and duration, at no less than the model's rate, in
the input's channels.** Send a 12-hour
master whole: the job cuts it at quiet points into pieces of at most `piece_s` (default 60,
10 to 600), converts each with `overlap_s` of real audio on both sides (default 0.5, 0 to 5,
under half a piece), and joins them with a `crossfade_s` fade at each seam (default 0.02,
0 to 1, at most twice `overlap_s`), so memory is bounded by a piece. The artifact has the
input's container and sample format (24-bit FLAC in, 24-bit FLAC out; a WAV past 4 GiB comes
back RF64) and **exactly its duration**; urvc's own output (16-bit at the model's rate, about
20 ms short per minute) is resampled and trimmed or padded per piece to make it so. The
converted signal's resolution is urvc's 16 bits whatever the container says.

- **Sample rate** (Owen, 2026-09-26): `output_rate: "native"`, the default, is
  `max(input rate, the rate urvc writes)`, read from urvc's output (48 kHz for the published
  models), so a 24 kHz input comes back at 48 kHz with the model's band above 12 kHz kept.
  `"input"` keeps the input's rate. Frames: `out_frames = round(in_frames × out_rate / in_rate)`,
  half up, in integers — the same on every host. `done`'s `outputs` gives each artifact's
  `frames`, `sample_rate` and `channels`.
- **Channels** (Owen, 2026-09-26): `output_channels: "input"`, the default, gives the input's
  channel count. **It is not a per-channel conversion**: RVC converts a mono mix and makes one
  voice, and a stereo input gets that one voice, identical, in both channels. `"mono"` writes
  one channel.
- The last ~20 ms per minute of an input's final piece may be zero-padded silence (urvc's
  shortfall where there is no overlap to absorb it). Owen accepted this, 2026-09-26.
The format is read from the bytes, so `c000` and `c000.flac` are the same input to this job;
an input that is not WAV, FLAC, OGG, MP3 or AIFF is refused before anything is converted.

> **There is no `llm` job type.** LLM work is the chat proxy; the only `llm`-class
> job types are `load-model` and `unload-model`. `GET /v1/info`'s `job_types` is
> the list to ask rather than to remember.

## Following, and resuming

`GET /v1/jobs/{id}/events`, `/v1/tasks/{id}/events` and
`/v1/tts/stream/{id}/events` are SSE with strictly increasing ids, and every
`--follow` / `events` command reads them live rather than polling.

`--since N` sends `Last-Event-ID: N`, which is the server's own resume: what was
missed is replayed and the follow continues from there. **Nothing reconnects by
itself.** A follow that dies has had something happen to it, and quietly
restarting would turn an outage into a pause nobody sees — the ids are printed,
so resuming is re-running with `--since <the last one>`.

A closed pipe is success: `crucible api job events … | head` exits 0.

## Not covered, and why

| route | why not |
|---|---|
| `POST /v1/pairing/start`, `/poll` | the REQUESTING app's half of the connect dance; a CLI that already holds a token has no use for it, and `crucible token --url` is how a credential is handed out |
| `GET /v1/peer`, `POST`/`DELETE /v1/peer/claim` | PHASE17: the orchestrator's relation to its engine, authenticated under its own names. A client borrowing that door would be a second claimant |

`tests/test_api_client.py` enumerates the server's own OpenAPI document and
fails if a route has neither a verb nor a row in that table — so this list
cannot quietly go stale.

## Verification

`tests/test_api_client.py`, 37 tests, run under
`~/anaconda3/envs/crucible/bin/python -m pytest tests/test_api_client.py`.

Most of them drive `cli.main` with the argv a person would type, against a REAL
server on a REAL loopback socket (`tests/live_server.py`) — not a `TestClient`,
because the one thing this module can get wrong is reading a streamed body off a
socket a line at a time, and a `TestClient` would fake exactly that out from
under it. The `echo` job type carries the end-to-end cases; nothing here touches
the card.

Every assertion was watched fail: thirty-one mutations were applied to
`crucible/apiclient.py` one at a time — the bearer removed, `Last-Event-ID`
dropped, the WAV rate hard-coded, a failed job made to exit 0, a refusal
paraphrased — and each was confirmed to turn the intended test red before being
put back. Three findings came out of that pass and are recorded in the tests
that changed because of them:

1. asserting `unauthorized` proved nothing — a client sending **no**
   Authorization header at all gets the same code. The test now asserts the
   message, which differs between "missing" and "wrong".
2. `app.routes` finds nothing on this FastAPI (`_IncludedRouter` hides the real
   routes), and the first draft of the coverage test passed while measuring an
   empty set. It reads `app.openapi()` now.
3. `TaskCreate` tests `is not None`, so an explicit `"job_type": null` and an
   absent `job_type` are one request to it. `cmd_task_submit` still omits unset
   fields — stating something nobody asked for is wrong even where it is
   invisible — but no test claims to prove it.

The whole path was then run against the live `cuda-linux` engine on
127.0.0.1:7100 (0.6.3): submit an `echo` job with an uploaded input, follow six
SSE events to `done`, download the artifact, `diff` it against the source, and
watch `unknown_job_type` come back verbatim from a bad `--type`.
