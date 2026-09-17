# The Crucible API

**GENERATED — do not edit.** `python scripts/gen-api-docs.py` writes this file
from the FastAPI app the server actually runs, and `scripts/release.sh` refuses a
cut when it is stale. Change a request model and regenerate; never edit here.

Every route is under `/v1` unless it says otherwise. Protected routes need
`Authorization: Bearer <token>` **and** `X-Crucible-Api: 1`, checked in that order.
An error is always a JSON body under an `error` key holding `code`, `message` and
sometimes `details`. Crucible refuses by name and with numbers: branch on `code`,
show a person the `message`.

The prose for WHY a field exists lives in the phase docs (`docs/PHASE*.md`); what
is here is what you may send and what comes back.

## Discovery

Answered without a token. How a client finds a server and learns its api version.

### `GET /v1/ping`

Unauthenticated. Lets a client tell "wrong token" from "not a Crucible".

*Door:* open

*Answers:* `200`

## Pairing

The token exchange. Start and poll need only the version header; approval is authenticated, because approving is the act that grants access.

### `POST /v1/pairing/decision`

Decide Pairing

*Door:* token + `X-Crucible-Api: 1`

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `id` | string | yes | — |  |
| `user_code` | string | yes | — |  |
| `allow` | boolean | yes | — |  |

*Answers:* `200`, `422`

### `POST /v1/pairing/poll`

Poll Pairing

*Door:* `X-Crucible-Api: 1` only

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `id` | string | yes | — |  |
| `device_code` | string | yes | — |  |

*Answers:* `200`, `422`

### `GET /v1/pairing/requests`

Pairing Requests

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `POST /v1/pairing/start`

Start Pairing

*Door:* `X-Crucible-Api: 1` only

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `client_name` | string | yes | — |  |

*Answers:* `200`, `422`

## Setup

The operator page's own door: it hands out a token and the pairing line.

### `GET /v1/setup`

Everything an app needs to be pointed at this server, in one read. PHASE13-OPERATOR.md section 3.1. Owen, 2026-09-14: *"crucible has its own ui. and it provides the token or whatever else we need to set it up on foundry or bookforge."* This is "whatever else we need". **It returns the token, and that reveals nothing.** Every `/v1/*` route is behind the bearer token, so the only caller who can read this is one who already has it. What it buys is that nobody types a secret twice: the operator page fetches this and draws a copyable pairing line, and the person pasting that line into BookForge has not seen a token at all. `urls` and `pairing` are the same list read two ways, and both are derived rather than stored — the bind address this process actually holds, made dialable (`crucible/pairing.py`). A wildcard bind becomes one entry per non-loopback IPv4 interface; a concrete bind becomes exactly one. Never a hostname lookup: an interface is a fact about this host, a name is a fact about somebody else's resolver. `job_types` repeats `/v1/info`'s list rather than making the page read twice, and it repeats it from the same producer — `store.registry` — so the two cannot disagree. After an install task's reload (3.4) both answer the new list in the same tick.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

## What this server is

Identity, backend, and every model this build ships with — including the ones this card cannot hold, and why.

### `GET /v1/info`

Info

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

## The card

What the accelerator is and what is free on it right now.

### `GET /v1/accelerator`

What is on the card right now, and which of it is Crucible's. PHASE4-AUDIO.md section 5. This is the same `nvidia-smi --query-compute-apps` the load guard runs, plus the free/total figures, plus the resident set, plus a flag saying which holders are this server's own processes — and it exists because BookForge arbitrates the GPU three incompatible ways at once (a queue slot, an in-process mutex whose timeout *proceeds without the lock*, and nothing at all for the hosted page reader), on top of a lock file with no producer inside the app. One call here answers the question all three were guessing at. **It never evicts anybody, ever.** It reports, and that is the whole of it. The rule is PHASE2-LLM.md section 4's and it does not soften because more job types now depend on the answer. It is private like every other route here: the bearer token and the version header, in that order. A probe of somebody's hardware is not public information, and `GET /v1/ping` already exists for "is this a Crucible".

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

## What fits

Crucible's own verdict about this host: which classes are enabled, which model each runs, and the arithmetic behind a refusal.

### `GET /v1/capability`

What this server can hold, per capability class, and why not. The read a client needs before it decides what to ask for. PHASE 9 made the act-to-model mapping a PER-HOST fact — `crucible install` probes the card and picks the largest candidate that fits, so a 24 GB box serves `translate` with a 4-bit 27B, a bigger one serves it with something else, and a 12 GB box does not serve it at all. A client that was handed a model id by configuration would be carrying a model this server may have refused. WHY A CLASS AND NOT A JOB TYPE. `enable_llm` is one boolean and Owen ruled translation binary per server, so `clean` and `translate` have to be able to disagree. They are separate classes here for that reason and no other; `simplify` and `analysis` are NOT classes, because they select the same model `translate` does and a capability axis that nothing selects on is a field that will drift (Owen, 2026-09-13). `enabled: false` IS AN ANSWER, not an error. A server that cannot translate says so with the number that decided it, and a client should be able to render "this machine cannot do that" without it looking like a fault. THIS IS A RECORD, NOT AN AUTHORITY. `[jobs] enable_*` remains the single owner of what this server offers; this says what the numbers were when somebody decided. `total_bytes` is the card the decision was made on, so a reader can tell a stale record from a current one — which is how a swapped GPU is noticed without anybody writing down a date. `job_types` IS NOT PART OF THE RECORD, and that is why it is added here rather than in `CapabilityRecord.to_dict()`. PHASE13-OPERATOR.md section 4 draws the operator page's Job types section from this one read, and to draw it the page needs three things the stored record cannot carry: which job type each class feeds (`capability.CLASSES`), which command builds that type's env (`cli.INSTALLER_FOR` — `denoise` shares `rvc`'s), and which narrator engines a `tts` install may name (`voices.NARRATOR_ENGINE_SAMPLING`). All three are THIS BUILD's tables, read live; a record written months ago must not be able to answer them, because they are facts about the code, not about the card. Put in the record they would be a second copy that goes stale the day an engine is added — which is the shape R1 exists to forbid. The page holding its own copy is the same defect one layer out, and is what section 4 means by "never a hard-coded list".

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

## Settings

The one door apps configure Crucible through. Crucible is set-and-forget; everything an app wants changed is written here.

### `GET /v1/settings`

Where each class's work runs, and which upstreams are configured. PHASE15-HOST.md section 3.1. Owen, 2026-09-14: *"Settings live in the engine and nowhere else."* An app draws this document and writes through `PUT`; it holds no key, no route and no model list of its own. **A key is never in this answer.** `key_hint` is its last four characters, which is enough to recognise WHICH key is there — the question a person with two accounts asks — and nothing else. There is no route on this server that returns one.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `PUT /v1/settings`

A partial patch, applied whole or not at all, live without a restart. PHASE15-HOST.md section 3.2. The order inside one request is the contract's — upstreams, then routes, then the whole validated — which is what lets an app configure an upstream AND route a class to it in one call, the way section 5.2 tells it to. **A refusal applies nothing.** `settings.resolve` builds the candidate document in memory and raises before `settings.apply` writes a byte, so a request refused for its routes does not leave a key behind on a server whose operator believes it failed. The answer is the whole `GET /v1/settings` document AFTER the write, so a window never has to guess what took.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `POST /v1/settings/upstreams/{name}/test`

Ask an upstream what it serves, with a key that may not be saved yet. PHASE15-HOST.md section 3.2. The body is optional and carries `{"key": …}` or `{"url": …}` to test BEFORE saving, which is the order a person actually works in: paste, check it works, then save. With no body the stored record is used. **Unbilled, and never cached.** The answer is somebody else's and changes without telling us; a stale list shown beside a key the operator pasted ten seconds ago is exactly the moment they would believe it. `POST` and not `GET` because it takes a body carrying a secret, and a secret in a query string is a secret in a log.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `name` | path | yes | string |  |

*Answers:* `200`, `422`

## Models

What is installed, what is resident, and what a pull would cost.

### `GET /v1/models`

Every model this build has a manifest for, and where it stands here.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `POST /v1/models/{subject_id}/lease`

Take the one lease this server holds at a time, on ANY resident kind. The order of the checks is their specificity, which is the job door's rule: a bad ttl and an unknown act are true of the request whatever this server is doing, so a client with a typo is told about the typo rather than about somebody else's lease. Residency comes next, because leasing a thing that is not here is a different mistake from being too late for one that is. **The id may name a model, a voice or an aligner** (PHASE7-LANES.md section 5.2, extended 2026-09-14). The route keeps its `/models/` path and its one route family, because the question it asks does not change with the kind: *is this the thing on the card?* The card holds ONE thing, so the kind is read off the residency rather than sent — and the namespaces being separate (a voice may be called `qwen3.5-9b`) cannot produce an ambiguity here, since only one of two colliding ids can be resident at a time and a lease is only ever on the resident one. Without this a book rendered chapter by chapter paid a narrator load per chapter and a book aligned chapter by chapter paid an aligner load per chapter, because the unload ruling clears the card the moment nothing holds it and the lease — the one thing that can hold it — could only name a model.

*Door:* open

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `subject_id` | path | yes | string |  |

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `act` | string | yes | — |  |
| `ttl_seconds` | integer | yes | — |  |

*Answers:* `201`, `422`

## Jobs

The work. Every job type is created, polled and cancelled through the same routes.

### `POST /v1/jobs`

Admit one job, or refuse with the facts about the one already here. **This door refuses when the lane is busy (ARCHITECTURE.md section 3).** It used to queue, which made Crucible answer the same question two ways: the streaming door has always refused with `409 stream_session_open` naming the holder, while this one accepted and appended. Same server, same card, two policies. Now both refuse and both name who has it. The order of the checks is the order of their cost and their specificity, and it is deliberate. The type and model are resolved first, because `unknown_job_type` is true whether or not anything is running and a client with a typo should be told about the typo rather than about somebody else's render. Admission comes next, before `preflight` — preflight shells out (`ffmpeg -version`), reads manifests and probes the card with `nvidia-smi`, and spending that on a request that cannot be admitted is work done for a 409. It also comes before `store.create`, so a refused submission never makes a directory, and before the inputs are materialised, so it never writes a client's megabytes to disk to delete them again. **A lease is refused ahead of both** (PHASE7-LANES.md section 5.2). A chat completion holds nothing, so a server mid-way through a two-thousand-block translation looks idle between two blocks; a client that says it intends a run takes a lease, and while one is open this door refuses the jobs that would move the leased thing off the card. It does not refuse anything else — a lease is not a reservation, and the lane is still free for work that leaves the card alone, INCLUDING the work the lease was taken for: a `tts` render of the leased voice and an `align` on the leased aligner are admitted, because they run against what is already resident rather than loading it again.

*Door:* token + `X-Crucible-Api: 1`

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `type` | string | yes | — |  |
| `model` | string or null | no | — |  |
| `params` | Params | no | — |  |
| `inputs` | Inputs | no | — |  |

*Answers:* `202`, `422`

### `GET /v1/jobs/{job_id}`

Get Job

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | path | yes | string |  |

*Answers:* `200`, `422`

### `DELETE /v1/jobs/{job_id}`

Cancel Job

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | path | yes | string |  |

*Answers:* `200`, `422`

### `GET /v1/jobs/{job_id}/artifacts/{name}`

Job Artifact

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | path | yes | string |  |
| `name` | path | yes | string |  |

*Answers:* `200`, `422`

### `GET /v1/jobs/{job_id}/events`

Job Events

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | path | yes | string |  |

*Answers:* `200`, `422`

## Tasks

Long host-side work — installs, pulls, env packs — that is not a job because no model runs.

### `GET /v1/tasks`

The last few tasks, newest first. In memory; a restart forgets them.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `POST /v1/tasks`

Admit one operator task, or refuse by name. Every refusal is made here, before the 202, and in the order the job door uses: what is wrong with the REQUEST first (`unknown_subject`, `unknown_job_type`, `narrator_engine_required`, `invalid_module`), then what is already true (`already_installed`, `job_type_installed`), then what this server is doing (`task_busy`, and for anything that reloads the registry, `server_busy`). A client with a misspelled id told "busy" would come back in ten minutes to be told about the typo.

*Door:* token + `X-Crucible-Api: 1`

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `type` | string | yes | — |  |
| `kind` | string or null | no | — |  |
| `id` | string or null | no | — |  |
| `job_type` | string or null | no | — |  |
| `narrator_engine` | string or null | no | — |  |
| `module` | object or null | no | — |  |
| `target` | string or null | no | — |  |

*Answers:* `202`, `422`

### `GET /v1/tasks/{task_id}`

Get Task

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `task_id` | path | yes | string |  |

*Answers:* `200`, `422`

### `DELETE /v1/tasks/{task_id}`

Cancel. A pull stops at its next chunk and its partial bytes go. `cancelling` and not `cancelled`, exactly as the job door answers: the flag is set here and the runner ends when it sees it, which for a pull is the next progress callback and for an install is the SIGTERM landing. Watch the stream for the `cancelled` event — telling a caller "cancelled" before the download thread has stopped would be the ambiguous answer R3 forbids.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `task_id` | path | yes | string |  |

*Answers:* `200`, `422`

### `GET /v1/tasks/{task_id}/events`

Task Events

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `task_id` | path | yes | string |  |

*Answers:* `200`, `422`

## Activity

What the server is doing right now, in one read.

### `GET /v1/activity`

What is on this server and how far along — one read, no job id. PHASE7-LANES.md section 5. Owen, 2026-09-13: *"Crucible will have to have an api endpoint that will report what's on it and its progress so Bookforge can hit that endpoint and fill that gpu slot with that data."* WHY THIS IS A POLL AND NOT THE SSE IT ALREADY HAS. Per-job events are push, fine-grained and exactly right for the step that owns a job. This answers a different question, asked by a bench widget that owns no job and may never own one: *what is this machine doing?* Opening a stream per job per server to render one line of text is the wrong shape. The two do not compete — the step reads the stream, the bench reads this. IT REPORTS AND NOTHING ELSE. It does not admit, reserve, claim or lock. A client that reads "free" and submits is racing every other client, and that race is settled at the door: `POST /v1/jobs` admits one and refuses the other `server_busy`, naming the winner (ARCHITECTURE.md section 3). The loser has lost nothing but a round trip, because it never gave up ownership of its own queue — which is the point of the ruling. A reservation here would be a second place to arbitrate, and a stale one. **So this route is a bench display and a preflight, never admission.** It is the honest answer to "how long until that finishes"; it is not permission to submit, and a client must be able to be refused after reading it. Only `POST /v1/jobs` can say yes. THE PROBE IS OPT-IN, and that is the one design decision in this route. `nvidia-smi` is a subprocess costing tens of milliseconds, and a bench polling three servers every few seconds would spawn one per server per tick forever to render a number nobody is reading. `resident` below already says what is loaded and roughly what it costs, in memory, for free. A caller that genuinely wants the live figure asks for it with `?accelerator_probe=true` and pays for it; `GET /v1/accelerator` remains the full answer.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `accelerator_probe` | query | no | boolean |  |

*Answers:* `200`, `422`

## Health

Is this process alive. Cheaper than /v1/activity and says less.

### `GET /v1/health`

Health

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

## Catalog

Everything this build can serve, of every kind, and what is on disk. Removing a subject here is how weights are reclaimed.

### `GET /v1/catalog`

Every subject this backend can hold, installed or not. PHASE13-OPERATOR.md section 3.2. Every field is derived from something this server already owns and no row is authored here — see `crucible/catalog.py`, which is the whole of it.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `DELETE /v1/catalog/{kind}/{subject_id}`

Delete an installed subject's files. PHASE15-HOST.md 3.5a. THE DOOR THE WEIGHTS RULE NEEDS. 3.5: a subject is never stored twice on one machine, so when the guest has its own copy the Windows one goes — and the host must never reach into `crucible/weights.py`'s layout from outside to do it, because a layout with two owners is the shape ARCHITECTURE.md R1 is about. So the server that owns the disk owns the deletion, and this is how it is asked. THE ORDER OF THE REFUSALS IS THE JOB DOOR'S: what is wrong with the REQUEST first (an unknown kind or id is true whatever this server is doing), then what is wrong with this server's STATE. A caller who misspelled a subject id and was told "it is in use" would fix the wrong thing.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `kind` | path | yes | string |  |
| `subject_id` | path | yes | string |  |

*Answers:* `204`, `422`

## Voices

The narration voices this build ships, and what each one costs.

### `GET /v1/voices`

Every voice this build has a manifest for, and where it stands here.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

## Streaming narration

A long-lived session that takes text and gives audio back over SSE, instead of one render per request (PHASE3-TTS.md section 7).

### `POST /v1/tts/stream`

Open the one streaming session this server will hold at a time.

*Door:* token + `X-Crucible-Api: 1`

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `voice` | string | yes | — |  |
| `language` | string | yes | — |  |

*Answers:* `201`, `422`

### `POST /v1/tts/stream/{session_id}`

One op: `say`, `cancel`, `cancel_all` or `close`. **`say` answers with the row's id and not the audio.** A client that wants the audio reads the stream; a client that never opened one is refused by name rather than generating into nothing.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `session_id` | path | yes | string |  |

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `op` | string | yes | — |  |
| `id` | string or null | no | — |  |
| `text` | string or null | no | — |  |
| `take` | integer or null | no | — |  |

*Answers:* `202`, `422`

### `DELETE /v1/tts/stream/{session_id}`

The same as `{"op": "close"}`, for a client that only has verbs.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `session_id` | path | yes | string |  |

*Answers:* `200`, `422`

### `GET /v1/tts/stream/{session_id}/events`

The session's SSE stream — everything it has to say, audio included. `Last-Event-ID` is the reattach: a connection that dropped in a tunnel comes back here inside the grace window, is replayed what it missed and follows live from there. It is the one behaviour a WebSocket could not have given for free, which is why this door is not one.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `session_id` | path | yes | string |  |

*Answers:* `200`, `422`

## Uploads

Bytes too big for a request body. An upload answers with a `blob_id` a job input then names.

### `POST /v1/uploads`

Upload

*Door:* token + `X-Crucible-Api: 1`

**Body** (`multipart/form-data`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `file` | string | yes | — |  |

*Answers:* `201`, `422`

## Leases

A client saying it intends a run, so the card is not taken out from under it mid-chapter.

### `DELETE /v1/leases/{lease_id}`

Give the card back before the ttl does it for you. The usual end of a lease, and the one that matters: expiry is the backstop for a client that died, not the way a finished run ends. A run that releases frees the next client immediately instead of after up to an hour of nothing happening.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `lease_id` | path | yes | string |  |

*Answers:* `204`, `422`

### `POST /v1/leases/{lease_id}/heartbeat`

I am still here. Pushes the deadline out by the lease's own ttl. A 404 here is not an error to log and continue past: it means this client's run is no longer protected, and the card may move under it at any moment. The body says whether the lease was released or expired, which is the difference between "somebody took it from me" and "I stopped talking for too long".

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `lease_id` | path | yes | string |  |

*Answers:* `200`, `422`

## OpenAI-compatible

A chat surface shaped like OpenAI's, for clients that already speak it.

### `POST /openai/v1/chat/completions`

Proxied to the resident engine, or forwarded to an upstream. PHASE2-LLM.md section 5 is the local half and is unchanged in every respect. PHASE15-HOST.md section 3.4 is the other: a `model` of the form `<upstream>/<id>` goes to that upstream on the operator's account. **The slash is the whole of the test**, and it works because a local model id can never contain one — refused at manifest load, `manifest_model_id_slash`. One character, one owner, no table.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `GET /openai/v1/models`

The resident model in OpenAI's list shape, plus every routed upstream one. `resident_model` rather than `resident`: with a voice on the card there is no model to list, and narrator answers no OpenAI route. **The upstream rows are the ROUTED ones and not a catalog** (PHASE15-HOST.md section 3.4). This route answers *"what may I send as `model`"*, and the answer is the resident thing plus whatever the operator routed to — the upstream's whole catalog is a different question with a different door, `POST /v1/settings/upstreams/{name}/test`, and putting it here would make a client believe this server had agreed to serve any of them.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `POST /v1/openai/chat/completions`

Proxied to the resident engine, or forwarded to an upstream. PHASE2-LLM.md section 5 is the local half and is unchanged in every respect. PHASE15-HOST.md section 3.4 is the other: a `model` of the form `<upstream>/<id>` goes to that upstream on the operator's account. **The slash is the whole of the test**, and it works because a local model id can never contain one — refused at manifest load, `manifest_model_id_slash`. One character, one owner, no table.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `GET /v1/openai/models`

The resident model in OpenAI's list shape, plus every routed upstream one. `resident_model` rather than `resident`: with a voice on the card there is no model to list, and narrator answers no OpenAI route. **The upstream rows are the ROUTED ones and not a catalog** (PHASE15-HOST.md section 3.4). This route answers *"what may I send as `model`"*, and the answer is the resident thing plus whatever the operator routed to — the upstream's whole catalog is a different question with a different door, `POST /v1/settings/upstreams/{name}/test`, and putting it here would make a client believe this server had agreed to serve any of them.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

## Peers

Orchestrator and engine talking to each other (PHASE17-ORCHESTRATOR.md). Not an app-facing surface.

### `GET /v1/peer`

`GET /v1/peer` — who manages this engine, and how old this process is. PHASE17 2.4: health flows ONE way. The orchestrator polls this and `/v1/ping`; the engine calls nothing back. An engine that phoned home would need to know its orchestrator's address, keep it fresh across restarts, and behave when it is wrong — three facts to own for a push a 15-second poll already delivers.

*Door:* open

*Answers:* `200`

### `POST /v1/peer/claim`

`POST /v1/peer/claim` — an orchestrator says it manages this engine. A STATEMENT OF FACT, not a grant of permission: nothing on this server consults `managed_by` before doing anything, because there is nothing an orchestrator asks an engine to do that an app may not also ask (`crucible/peer.py`'s preamble). What it buys is that `/v1/info` can answer "who manages this".

*Door:* open

*Answers:* `200`

### `DELETE /v1/peer/claim`

`DELETE /v1/peer/claim` — the orchestrator's Quit (PHASE17 2.2). Nothing claimed is NOT a refusal: "there is no claim" is the state the caller asked for. Somebody else's claim is refused, because releasing one by accident is how an engine ends up unmanaged with a tray still watching it.

*Door:* open

*Answers:* `200`

## Request models in full

Every schema the routes above refer to, for a reader following a nested field.

### `Body_upload_v1_uploads_post`

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `file` | string | yes | — |  |

### `DecidePairing`

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `id` | string | yes | — |  |
| `user_code` | string | yes | — |  |
| `allow` | boolean | yes | — |  |

### `HTTPValidationError`

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `detail` | array of ValidationError | no | — |  |

### `JobCreate`

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `type` | string | yes | — |  |
| `model` | string or null | no | — |  |
| `params` | Params | no | — |  |
| `inputs` | Inputs | no | — |  |

### `JobInput`

One named input: either an uploaded blob or bytes inline in the request.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `blob_id` | string or null | no | — |  |
| `inline_base64` | string or null | no | — |  |

### `LeaseOpen`

`POST /v1/models/{id}/lease` — a client saying it intends a run. Both fields are required and neither has a default, for the streaming door's reason. A default `act` would put a name nobody chose on a bench, which is the thing `X-Crucible-Act` is refused for; a default `ttl_seconds` would be this server picking how long somebody else's run is, which is the one number only the client knows. **There is no `kind`.** The id in the path is the resident thing's, of whatever kind, and the card holds one thing — so the server reads the kind off `Residency.resident` and a client has nothing to disambiguate. A `kind` on the body would be a second owner of `resident.kind`, able to disagree with it (R1), and would let a client be refused for spelling a fact it was never asked to know.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `act` | string | yes | — |  |
| `ttl_seconds` | integer | yes | — |  |

### `PollPairing`

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `id` | string | yes | — |  |
| `device_code` | string | yes | — |  |

### `StartPairing`

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `client_name` | string | yes | — |  |

### `StreamOp`

`POST /v1/tts/stream/{id}` — one op. {"op": "say", "id": "r12", "text": "...", "take": 0} {"op": "cancel", "id": "r12"} {"op": "cancel_all"} {"op": "close"} `take` is **required** on `say` and has no default here. The SDK's `say(id, text, take?)` defaults it to 0 in the caller's own code, which is a client choosing; a default on the wire would be the server choosing, and now that the five fine-tunes declare a second rung that would be a render at a take nobody asked for. A take past the end of the voice's ladder is `unknown_take` and is never clamped.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `op` | string | yes | — |  |
| `id` | string or null | no | — |  |
| `text` | string or null | no | — |  |
| `take` | integer or null | no | — |  |

### `StreamOpen`

`POST /v1/tts/stream` — PHASE3-TTS.md section 7. Nothing has a default, for the render door's reason: a session opened in the wrong language, or on a voice the client did not choose, is a silent substitution and a whole afternoon of listening in the wrong accent.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `voice` | string | yes | — |  |
| `language` | string | yes | — |  |

### `TaskCreate`

`POST /v1/tasks` — one operator operation. PHASE13-OPERATOR.md 3.3. One model for three request shapes rather than three routes, because there is one lane and one refusal (`task_busy`) governing all of them, and a client that had to pick a path before it could be told "busy" would have to know which of three doors to retry. The validator is `StreamOp`'s in spirit: the `type` word decides which fields are required and which are REFUSED. A `narrator_engine` sent with a `pull`, or an `id` sent with an `install`, is a client that has confused two requests, and accepting it silently would run the wrong one.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `type` | string | yes | — |  |
| `kind` | string or null | no | — |  |
| `id` | string or null | no | — |  |
| `job_type` | string or null | no | — |  |
| `narrator_engine` | string or null | no | — |  |
| `module` | object or null | no | — |  |
| `target` | string or null | no | — |  |

### `ValidationError`

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `loc` | array of string or integer | yes | — |  |
| `msg` | string | yes | — |  |
| `type` | string | yes | — |  |
