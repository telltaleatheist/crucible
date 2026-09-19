# `crucible api` — driving a server from a shell

Built 2026-09-16, because there was no way to run a job from a command line. The
3,100 lines already in `crucible/cli.py` are entirely operator and lifecycle —
`init`, `install`, `capability`, `models pull`, `serve`, `service`,
`doctor`, `token`, `uninstall` — and **not one of them submits a job**. A
fine-tuning script that wanted a retake ladder, a cleanup pass, a translation or
a batch of renders had to write its own HTTP client. This is that client, and it
lives in `crucible/apiclient.py`.

## The two halves of one binary

| | operator verbs | `crucible api …` |
|---|---|---|
| subject | **this machine's installation** | **a server**, anywhere |
| reads | `config.toml`, units, env packs, the disk | HTTP, with a bearer token |
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

# 2. a server anywhere. BOTH flags or neither.
crucible api info --url http://192.168.68.20:7100 --token <token>

# 3. one pairing line, which carries the address AND the token
crucible api info --pairing "$(crucible token --url | head -1)"
```

`--url` without `--token` is **refused by name** (`token_required`) and never
borrows the local engine's bearer: sending this machine's credential to an
address somebody typed is a leak with a one-character typo as its cause.

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

Refusals are never paraphrased. `chunk_too_long`, `unknown_take`,
`stream_session_open`, `server_busy`, `not_resident` — those strings are what a
person greps this repo for, and a client that rewrote them into nicer English
would have deleted the only handle on them.

## The commands

### Reads

```
crucible api ping              is this a Crucible, and which one (no token needed)
crucible api health            the lane, the queue, each job type's readiness
crucible api info              identity, host, job types, every capability row
crucible api setup             the pairing lines an app's connect door takes
crucible api capability        what this host can hold and why  [--accelerator-probe]
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
crucible api pairing-requests                  connect requests awaiting a decision
crucible api pairing-decide --id … --code … --allow|--deny
```

### Jobs

```
crucible api job submit --type <t> [--model <m>] [--params <json|@file>]
                        [--input NAME=PATH]... [--input-blob NAME=BLOB]...
                        [--follow] [--artifacts-dir DIR]
crucible api job get <job-id>
crucible api job events <job-id> [--since <event-id>]
crucible api job cancel <job-id>
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

### Chat — the cleanup and translation door

```
crucible api chat --model qwen3.5-9b --message "…"           one-shot shorthand
crucible api chat --body @request.json [--stream] [--act clean]
```

The body is forwarded **verbatim** in both directions except `model`, so
`thinking`, `chat_template_kwargs`, `max_tokens` and everything else are the
engine's vocabulary and belong in `--body`, not in flags this CLI would have to
keep up with.

> **The route is `POST /v1/openai/chat/completions`** (also mounted at
> `/openai/v1/chat/completions` for OpenAI clients). There is **no bare
> `POST /v1/chat/completions`** on this server — checked against `crucible/api.py`
> and against the live engine on 2026-09-16.

### Leases

```
crucible api lease open <subject-id> --act clean --ttl 600
crucible api lease heartbeat <lease-id>
crucible api lease release <lease-id>
```

The path says `models` and the id may name a model, a voice or an aligner: the
server reads the kind off `Residency.resident`, so there is nothing to
disambiguate.

## What a fine-tuning script calls

### A batch of renders (the render door, PHASE3-TTS section 6)

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

A take past the end of a voice's ladder is `unknown_take` and is never clamped.

### AI cleanup and translation

The model must already be resident — this CLI never loads one implicitly, and
neither does the chat door:

```bash
crucible api job submit --type load-model --model qwen3.5-9b --follow
crucible api chat --act clean --body @cleanup-request.json
crucible api chat --act translate --body @translate-request.json
crucible api job submit --type unload-model --model qwen3.5-9b --follow
```

Send `"thinking": false` in the body for a cleanup pass: Qwen3.5 otherwise
spends a bounded budget entirely on reasoning (measured; the same rule
BookForge's `crucible` provider follows).

### The other job types, as bodies

| type | `--model` | `--params` | inputs |
|---|---|---|---|
| `echo` | — | `{"delay_ms": 25}` | any; each is copied to an artifact of the same name |
| `tts` | voice id | `{"language","take","chunks":[{"index","text"}]}` | none |
| `load-voice` | voice id | `{"timeout_s": …}`, plus `reference` for a `zeroshot` voice | none |
| `unload-voice` / `unload-model` / `unload-aligner` / `unload-denoiser` | id | `{}` | none |
| `load-model` | model id | `{"timeout_s": …}` | none |
| `asr` | whisper id | `{"language","vad_filter","word_timestamps"}` | exactly one audio file; `"auto"` is a language |
| `align` | aligner id | `{"language","chunks":[{"index","text"}]}` | one per chunk, named `<index>.<ext>` |
| `align-longform` | aligner id | `{"language","sentences":[{"index","text","kind"}],"rough_model","chunk_s",…}` | exactly one audio file — the whole audiobook |
| `rvc` | rvc voice id | `{"index_rate","protect_rate","n_semitones"[, "f0_method","hop_length"]}` | many, all the same extension |
| `denoise` | separator id | `{}` — and that is the contract | exactly one audio file |

`protect_rate`'s scale is **inverted**: lower protects more and 0.5 turns
protection off. The tuned deathstalker→Sigma recipe is `index_rate 0.3`,
`protect_rate 0.1`, `n_semitones -2`, `f0_method rmvpe`.

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
