# The Crucible API

**GENERATED — do not edit.** `python scripts/gen-api-docs.py` writes this file
from the FastAPI app the server actually runs, and `scripts/release.sh` refuses a
cut when it is stale. Change a request or answer model and regenerate; never edit here.

Every route is under `/v1` unless it says otherwise. Protected routes need
`Authorization: Bearer <token>` **and** `X-Crucible-Api: 1`, checked in that order.
An error is always a JSON body under an `error` key holding `code`, `message` and
sometimes `details`. Crucible refuses by name and with numbers: branch on `code`,
show a person the `message`.

The prose for WHY a field exists lives in the internals docs (`docs/internals/*.md`); what
is here is what you may send and what comes back.

## Discovery

Answered without a token. How a client finds a server and learns its api version.

### `GET /v1/ping`

Unauthenticated. Lets a client tell "wrong token" from "not a Crucible".

*Door:* open

*Answers:* `200` Ping

**Answer `200`** (`application/json`), the body:

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `crucible` | `True` | yes | — |  |
| `name` | string | yes | — |  |
| `api_version` | integer | yes | — |  |
| `pairing_version` | integer | yes | — |  |

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

*Answers:* `200`, `422` HTTPValidationError

### `POST /v1/pairing/poll`

Poll Pairing

*Door:* `X-Crucible-Api: 1` only

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `id` | string | yes | — |  |
| `device_code` | string | yes | — |  |

*Answers:* `200`, `422` HTTPValidationError

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

*Answers:* `200`, `422` HTTPValidationError

## Setup

The operator page's own door: it hands out a token and the pairing line.

### `GET /v1/setup`

Everything an app needs to be pointed at this server in one read, including its token and pairing lines.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

## What this server is

Identity, backend, and every model this build ships with — including the ones this card cannot hold, and why.

### `GET /v1/info`

Who this server is, what it runs on, what it serves, and the few fixed tables (terminal states, voice sources, service commands) a console shows.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200` Info

**Answer `200`** (`application/json`), the body:

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `server` | object | yes | — |  |
| `role` | string | yes | — |  |
| `managed_by` | object of string or null | yes | — |  |
| `host` | object | yes | — |  |
| `job_types` | array of string | yes | — |  |
| `capabilities` | array of object | yes | — |  |
| `pages_engine` | object | yes | — |  |
| `terminal_states` | TerminalStates | yes | — | The states after which a job or a task never changes again. |
| `voice_sources` | object of VoiceSourceLabel | yes | — |  |
| `service_commands` | array of ServiceCommand | yes | — |  |

## The card

What the accelerator is and what is free on it right now.

### `GET /v1/accelerator`

What is on the card right now and which holders are Crucible's own processes. Reports only; it never evicts anything.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

## What fits

Crucible's own verdict about this host: which classes are enabled, which model each runs, and the arithmetic behind a refusal.

### `GET /v1/capability`

What this server can hold, per capability class, and why not; `enabled: false` is an answer, not an error. A client-sized class may be sized with `?class=&context_tokens=&concurrency=`.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `GET /v1/capability/plan`

What an install (`?job_type=`) or a pull (`?subject=`) would give this card, decided live and writing nothing.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

## Settings

The one door apps configure Crucible through. Crucible is set-and-forget; everything an app wants changed is written here.

### `GET /v1/settings`

Where each class's work runs and which upstreams are configured. A key is never returned; `key_hint` shows its last four characters.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `PUT /v1/settings`

Apply a partial settings patch, whole or not at all, live without a restart. Answers the full settings document after the write.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `POST /v1/settings/upstreams/{name}/test`

List what an upstream serves, using the body's `key` or `url` when given, else the stored record. Never cached.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `name` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

## Models

What is installed, what is resident, and what a pull would cost.

### `GET /v1/models`

Every model this build has a manifest for, and where it stands here.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `POST /v1/models/{subject_id}/lease`

Hold whatever is resident (model, voice or aligner) on the card for a run; jobs that would move it are refused `409 leased`. A lease never loads anything.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `subject_id` | path | yes | string |  |

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `act` | string | yes | — |  |
| `ttl_seconds` | integer | yes | — |  |

*Answers:* `201`, `422` HTTPValidationError

## Jobs

The work. Every job type is created, polled and cancelled through the same routes. The `image` job's params, its result and how to prompt it are in docs/IMAGE.md; the `audio` job's (sound effects, music and songs) are in docs/AUDIO.md; the `segment` job's (subject cutouts and point-and-box selections, as masks) are in docs/SEGMENT.md; the `video` job's (clips with sound from words or a start picture) are in docs/VIDEO.md.

### `POST /v1/jobs`

Admit one job, or refuse by name: a busy lane is `409 server_busy`, and a missing environment or model is installed while the job is refused `409 installing`. With `queue` a busy lane queues the job instead: it waits with status `queued` and its events say where it stands. `params.resume` set to a `resume_id` continues a journaled job; without it the job starts fresh.

*Door:* token + `X-Crucible-Api: 1`

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `type` | string | yes | — |  |
| `model` | string or null | no | — |  |
| `params` | object | no | — |  |
| `inputs` | object of JobInput | no | — |  |
| `client_ref` | string or null | no | — | The client's own name for this work, echoed on the job record and never read by the server. |
| `hold` | boolean | no | `False` | Hold the job from creation, as `POST /v1/jobs/{id}/hold` would, so its artifacts outlive being fetched. |
| `queue` | QueueRequest or null | no | — | Opt in to the server's queue: while the lane is busy the job waits (status `queued`) instead of being refused `409 server_busy`. Without it, a busy server refuses as it always has. |

*Answers:* `202`, `409` ServerBusy, `422` HTTPValidationError

### `GET /v1/jobs/{job_id}`

Get Job

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | path | yes | string |  |

*Answers:* `200` JobStatus, `404` ErrorEnvelope, `422` HTTPValidationError

**Answer `200`** (`application/json`), the body:

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | string | yes | — |  |
| `type` | string | yes | — |  |
| `model` | string or null | yes | — |  |
| `status` | string | yes | — |  |
| `progress` | integer or number | yes | — |  |
| `position` | integer or null | yes | — |  |
| `error` | JobFailure or null | yes | — |  |
| `artifacts` | array of string | yes | — |  |
| `created` | string | yes | — |  |
| `started` | string or null | yes | — |  |
| `finished` | string or null | yes | — |  |
| `client_ref` | string or null | yes | — |  |
| `interrupted_at` | string or null | yes | — |  |
| `held_by` | string or null | yes | — |  |
| `held_since` | string or null | yes | — |  |
| `chunks_done` | array of integer | yes | — |  |
| `chunks_total` | integer or null | yes | — |  |
| `chunk_at` | string or null | yes | — |  |
| `resume_id` | string or null | yes | — |  |
| `resumed` | boolean | yes | — |  |
| `lease_id` | string or null | no | — |  |
| `sampling` | object or null | no | — |  |
| `removal` | JobRemoval or null | no | — |  |

### `DELETE /v1/jobs/{job_id}`

Cancel a job; a job still waiting in the queue is removed (reason `client`) and answers `status: removed`.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

### `GET /v1/jobs/{job_id}/artifacts/{name}`

Job Artifact

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | path | yes | string |  |
| `name` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

### `GET /v1/jobs/{job_id}/events`

Job Events

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

### `POST /v1/jobs/{job_id}/hold`

Keep this job's artifacts for a later job's `{"artifact": {job_id, name}}` inputs until the hold is released or `retention_days` collects it. Idempotent.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

### `DELETE /v1/jobs/{job_id}/hold`

The chain is complete: release the hold and remove the job now.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | path | yes | string |  |

*Answers:* `204`, `422` HTTPValidationError

## Queue

Jobs submitted with `queue` while the lane is busy wait here, in order: list them, remove one, keep one alive, or follow every change. How an app should use it is docs/QUEUE.md.

### `GET /v1/queue`

The jobs waiting for the lane, in the order they will be offered to it: the lease holder's first while its lease is open, then first come, first served.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200` QueueList

**Answer `200`** (`application/json`), the body:

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `items` | array of QueueItem | yes | — |  |
| `depth` | integer | yes | — |  |
| `limits` | object | yes | — |  |

### `GET /v1/queue/events`

Every change to the queue, for a dashboard: a `snapshot` first, then `added`, `moved`, `started` and `removed` as they happen.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `DELETE /v1/queue/{job_id}`

Take a waiting job out of the queue. It ends `removed` with reason `operator`; a job that has started is cancelled with DELETE /v1/jobs/{id}.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | path | yes | string |  |

*Answers:* `200` QueueRemoved, `404` ErrorEnvelope, `422` HTTPValidationError

**Answer `200`** (`application/json`), the body:

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | string | yes | — |  |
| `status` | `'removed'` | yes | — |  |
| `reason` | `'operator'` | yes | — |  |

### `POST /v1/queue/{job_id}/heartbeat`

Say the client that queued this job is still there. Only needed by a client that neither follows the job's events nor polls it.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | path | yes | string |  |

*Answers:* `200`, `404` ErrorEnvelope, `422` HTTPValidationError

## Resumable jobs

The resume journals: every job type that keeps one writes its finished work to disk as it lands, and a job sent `params.resume` continues it (docs/RESUMABLE-JOBS.md).

### `GET /v1/resumable`

Every resume journal this server keeps, newest first, with progress, inputs and expiry. Send a row's `resume_id` as `params.resume` to continue it.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `GET /v1/resumable/{resume_id}`

One journal, as `GET /v1/resumable` lists it.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `resume_id` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

### `DELETE /v1/resumable/{resume_id}`

Discard a journal now; refused `resume_in_use` while a job writes it.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `resume_id` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

## Tasks

Long host-side work — installs, pulls, env packs — that is not a job because no model runs.

### `GET /v1/tasks`

The last few tasks, newest first. In memory; a restart forgets them.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `POST /v1/tasks`

Admit one operator task (pull, install, module, engine or engine-restart), or refuse by name.

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

*Answers:* `202`, `409` ServerBusy, `422` HTTPValidationError

### `GET /v1/tasks/{task_id}`

Get Task

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `task_id` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

### `DELETE /v1/tasks/{task_id}`

Cancel a task. Answers `cancelling`; the stream's `cancelled` event says when it has stopped.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `task_id` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

### `GET /v1/tasks/{task_id}/events`

Task Events

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `task_id` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

## Activity

What the server is doing right now, in one read.

### `GET /v1/activity`

What this server is doing and how far along, in one read with no job id. A display and a preflight, never admission; `?accelerator_probe=true` adds a live card probe.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `accelerator_probe` | query | no | boolean |  |

*Answers:* `200` Activity, `422` HTTPValidationError

**Answer `200`** (`application/json`), the body:

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `server` | ActivityServer | yes | — | Who answered `GET /v1/activity`. |
| `resident` | ActivityResident or null | yes | — |  |
| `stopping` | object or null | yes | — |  |
| `warming` | string or null | yes | — |  |
| `claim` | object or null | yes | — |  |
| `streaming` | object or null | yes | — |  |
| `chat` | ActivityChat | yes | — | Chat completions in flight and the engine's admission limit. |
| `settings` | ActivitySettings | yes | — | Recent writes through `PUT /v1/settings`. |
| `catalog` | ActivityCatalog | yes | — | Recent removals through `DELETE /v1/catalog/{kind}/{id}`. |
| `lease` | ActivityLease or null | yes | — |  |
| `slots` | ActivitySlots | yes | — | Every lane this server admits work through. |
| `running` | array of ActivityJob | yes | — |  |
| `queued` | array of ActivityJob | yes | — |  |
| `accelerator` | object or null | no | — |  |

## Health

Is this process alive. Cheaper than /v1/activity and says less.

### `GET /v1/health`

Health

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

## Catalog

Everything this build can serve, of every kind, and what is on disk. Removing a subject here is how weights are reclaimed.

### `GET /v1/catalog`

Every subject this backend can hold, installed or not.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `DELETE /v1/catalog/{kind}/{subject_id}`

Delete an installed subject's files. Refused while the subject is resident, leased or named by a running task.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `kind` | path | yes | string |  |
| `subject_id` | path | yes | string |  |

*Answers:* `204`, `422` HTTPValidationError

## Voices

The narration voices this build ships, and what each one costs.

### `GET /v1/voices`

Every voice this build has a manifest for, and where it stands here.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200` array of VoiceInfo

**Answer `200`** (`application/json`), an array; each item:

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `id` | string | yes | — |  |
| `display` | string | yes | — |  |
| `kind` | string or null | yes | — |  |
| `language` | string or null | yes | — |  |
| `narrator_engine` | string or null | yes | — |  |
| `backend_supported` | boolean | yes | — |  |
| `installed` | boolean | yes | — |  |
| `resident` | boolean | yes | — |  |
| `orphan` | boolean or null | yes | — |  |
| `loadable` | boolean | yes | — |  |
| `reason` | string or null | yes | — |  |
| `revision` | string or null | yes | — |  |
| `fingerprint` | string or null | yes | — |  |
| `source` | string or null | yes | — |  |
| `identity_basis` | string or null | yes | — |  |
| `memory_bytes_estimate` | integer or number or null | yes | — |  |
| `estimate_basis` | string or null | yes | — |  |
| `serving` | object or null | yes | — |  |
| `max_chars` | integer or null | yes | — |  |
| `max_chars_basis` | string or null | yes | — |  |
| `pace_basis` | string or null | yes | — |  |
| `inherited_from` | string or null | yes | — |  |
| `manifest` | string or null | yes | — |  |
| `sample_rate` | integer or null | yes | — |  |
| `takes` | integer | yes | — |  |
| `needs_reference` | boolean | yes | — |  |
| `pace` | object or null | yes | — |  |
| `ref` | string or null | no | — | The tag this voice follows on its repo (`crucible`); null for an exact-sha pin. |
| `latest_revision` | string or null | no | — | The commit the tag named at the last explicit check (`POST /v1/voices/updates`, `crucible voices check-updates`, or a pull); never looked up by this GET. |
| `update_available` | boolean | no | `False` | A pull would move this voice from `revision` to `latest_revision`. |
| `update_checked_at` | string or null | no | — | When the tag was last looked up. |
| `update_error` | string or null | no | — | Why the last look-up failed; the voice stays on the revision it has. |
| `sampling` | object of integer or number or null | no | — | `{temperature, top_p, top_k}` this backend's arm renders take 0 with: the sampling Crucible writes into narrator's voice document. |
| `edge_fade_ms` | object of integer or number or null | no | — | `{in, out}`: raised-cosine fades, in milliseconds, at each chunk edge on this arm. |
| `chunk_gap` | object or null | no | — | The silence to add after each chunk: `inject_s` (net of the model's own tail), `target_join_s`, `model_self_tail_s`, optional `reader_sentence_gap_s` and `model_internal_gap_s`, and `rule`, `method`, `source`, `measured_on`. |
| `reference_seconds_cap` | integer or number or null | no | — | The most reference-clip audio this arm takes, in seconds. |
| `allowed_controls` | array of string or null | no | — | The inline control tokens (`<\|group:name\|>`) this arm allows; `[]` allows none. |

### `POST /v1/voices/updates`

Look up, on the Hub, the tag every voice here follows, and say which ones a pull would move. The only request that resolves a tag; GET /v1/voices never does.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `PUT /v1/voices/{voice_id}`

Pin a voice to a repo revision (`{"pin": ...}`) or write a local manifest override (`{"voice": ...}`), and return its `/v1/voices` row. A missing revision is resolved to the repo's head.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `voice_id` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

### `DELETE /v1/voices/{voice_id}`

Remove this machine's pin or override for a voice; a shipped voice reverts to its packaged manifest. Never deletes weights; an unknown id answers 204.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `voice_id` | path | yes | string |  |

*Answers:* `204`, `422` HTTPValidationError

### `GET /v1/voices/{voice_id}/manifest`

One voice's settings as a whole local manifest document, ready to edit and send back with `PUT`. `not_carried` names what the local schema cannot hold.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `voice_id` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

## Streaming narration

A long-lived session that takes text and gives audio back over SSE, instead of one render per request.

### `POST /v1/tts/stream`

Open the one streaming session this server will hold at a time.

*Door:* token + `X-Crucible-Api: 1`

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `voice` | string | yes | — |  |
| `language` | string | yes | — |  |

*Answers:* `201`, `422` HTTPValidationError

### `POST /v1/tts/stream/{session_id}`

One op: `say`, `cancel`, `cancel_all` or `close`. `say` answers with the row id; the audio arrives on the stream.

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

*Answers:* `202`, `422` HTTPValidationError

### `DELETE /v1/tts/stream/{session_id}`

The same as `{"op": "close"}`, for a client that only has verbs.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `session_id` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

### `GET /v1/tts/stream/{session_id}/events`

The session's SSE stream, audio included. Reattach with `Last-Event-ID` within the grace window to replay what was missed.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `session_id` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

## Uploads

Bytes too big for a request body. An upload answers with a `blob_id` a job input then names.

### `POST /v1/uploads`

Upload

*Door:* token + `X-Crucible-Api: 1`

**Body** (`multipart/form-data`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `file` | string | yes | — |  |

*Answers:* `201`, `422` HTTPValidationError

## Leases

A client saying it intends a run, so the card is not taken out from under it mid-chapter.

### `DELETE /v1/leases/{lease_id}`

Release the lease; if nothing else holds the card, it is cleared before this answers.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `lease_id` | path | yes | string |  |

*Answers:* `204`, `422` HTTPValidationError

### `POST /v1/leases/{lease_id}/heartbeat`

Push the lease's deadline out by its own ttl. A 404 means the lease was released or expired and the run is no longer protected.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `lease_id` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

## OpenAI-compatible

A chat surface shaped like OpenAI's, for clients that already speak it.

### `POST /openai/v1/chat/completions`

An OpenAI chat completion, proxied to the resident engine or, for a `<upstream>/<id>` model, forwarded to that upstream.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `GET /openai/v1/models`

The resident model in OpenAI's list shape, plus every upstream model a route names.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `POST /v1/openai/chat/completions`

An OpenAI chat completion, proxied to the resident engine or, for a `<upstream>/<id>` model, forwarded to that upstream.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `GET /v1/openai/models`

The resident model in OpenAI's list shape, plus every upstream model a route names.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

## Peers

Orchestrator and engine talking to each other (docs/internals/host-and-platform.md). Not an app-facing surface.

### `GET /v1/peer`

Who manages this engine, and this process's uptime.

*Door:* open

*Answers:* `200`

### `POST /v1/peer/claim`

An orchestrator states that it manages this engine; `force` takes it from another orchestrator.

*Door:* open

*Answers:* `200`

### `DELETE /v1/peer/claim`

The orchestrator releases its claim; releasing when nothing is claimed succeeds.

*Door:* open

*Answers:* `200`

## Everything else

### `POST /v1/decide`

One answer distribution per question, read off the resident model's next-token logprobs; with `items`, one choice answer per item in one request. Every refusal a caller can cause is made before anything is decided.

*Door:* token + `X-Crucible-Api: 1`

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `model` | string | yes | — | The Crucible model id, which must already be resident (`409 model_not_resident` otherwise). An upstream id (`<upstream>/<id>`) is refused `400 decide_needs_logprobs`: no upstream returns a distribution. |
| `state` | State | yes | — | What the questions are about: a string, used verbatim, or any other JSON value, serialised as compact JSON. Required and never null; may be `""` only when `images` carry the state. |
| `questions` | object of ChoiceQuestion or ScoreQuestion or YesNoQuestion or null | no | — | Question name to question. Names are single path members (no `/`, `\`, leading dot) and key the answers. Answers come back in this order. Exactly one of `questions` and `items` is sent. |
| `instructions` | string or null | no | — | The items form's ask, written under every item's text: "Which of the categories listed above does the speaker do in this passage?". Each item's question is `text`, a newline, then this; absent, `text` alone. Refused with `questions`. |
| `options` | object of string or null | no | — | The items form's shared options (name to one-line description, letters A, B, C… in the order given, 2 to 26). An item without its own `options` uses these. |
| `items` | array of DecideItem or null | no | — | The items form: an ordered list of choice questions about ONE state, each answered exactly as a lone choice question would be (it sees the state and its own question, never another item), in one request: on the Mac the shared state runs once and every item continues from its cache. Answers come back as a list in this order. At most 512 (`too_many_items`); token caps in docs/internals/api.md. |
| `images` | array of string or null | no | — | Base64 image files (PNG, JPEG, GIF or WebP; standard alphabet, padded, no whitespace, no `data:` prefix), read as part of the state, after its text. At most 8 (`too_many_images`), and only on a model whose manifest declares `image` (`400 model_text_only` otherwise). `[]` is the same as none. |
| `missing` | `'refuse'` or `'report'` | no | `'refuse'` | What to do when a label is not among the top tokens the engine returned. `refuse` (the default): the decision is `502 label_not_in_probs` naming the question and the letter. `report`: the door never invents a number — that option's probability and log-probability are null, it is named in the answer's `missing_labels`, and the renormalisation, `confidence`, `score` and `label_mass` run over the letters actually returned. A question whose EVERY label is missing is refused in both modes: there is no answer to report. |
| `queue` | QueueRequest or null | no | — | Wait for the model instead of being refused: `{}` or `{"max_wait_s": N}`. While the model is not resident, or every slot on its engine is taken, the request is held open in the server's queue (docs/QUEUE.md) and the model is loaded for it when its turn comes. Absent: refused as before. |

*Answers:* `200`, `422` HTTPValidationError

## Models in full

Every request and answer schema the routes above refer to, for a reader following a nested field.

### `Activity`

`GET /v1/activity`: what this server is doing, in one read.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `server` | ActivityServer | yes | — | Who answered `GET /v1/activity`. |
| `resident` | ActivityResident or null | yes | — |  |
| `stopping` | object or null | yes | — |  |
| `warming` | string or null | yes | — |  |
| `claim` | object or null | yes | — |  |
| `streaming` | object or null | yes | — |  |
| `chat` | ActivityChat | yes | — | Chat completions in flight and the engine's admission limit. |
| `settings` | ActivitySettings | yes | — | Recent writes through `PUT /v1/settings`. |
| `catalog` | ActivityCatalog | yes | — | Recent removals through `DELETE /v1/catalog/{kind}/{id}`. |
| `lease` | ActivityLease or null | yes | — |  |
| `slots` | ActivitySlots | yes | — | Every lane this server admits work through. |
| `running` | array of ActivityJob | yes | — |  |
| `queued` | array of ActivityJob | yes | — |  |
| `accelerator` | object or null | no | — |  |

### `ActivityCatalog`

Recent removals through `DELETE /v1/catalog/{kind}/{id}`.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `removals` | array of object | yes | — |  |

### `ActivityChat`

Chat completions in flight and the engine's admission limit.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `in_flight` | integer | yes | — |  |
| `max_in_flight` | integer or null | yes | — |  |
| `max_in_flight_basis` | string or null | yes | — |  |
| `rows` | array of ActivityChatRow | yes | — |  |

### `ActivityChatRow`

One chat completion in flight.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `id` | integer | yes | — |  |
| `act` | string or null | yes | — |  |
| `model` | string | yes | — |  |
| `client` | string or null | yes | — |  |
| `since` | string | yes | — |  |

### `ActivityHeld`

What keeps the resident subject on the card.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `fact` | string | yes | — |  |
| `who` | string | yes | — |  |
| `details` | object | yes | — |  |

### `ActivityJob`

A running or queued job, as `GET /v1/activity` shows it.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | string | yes | — |  |
| `type` | string | yes | — |  |
| `model` | string or null | yes | — |  |
| `status` | string | yes | — |  |
| `position` | integer or null | yes | — |  |
| `progress` | integer or number | yes | — |  |
| `message` | string or null | yes | — |  |
| `created` | string | yes | — |  |
| `started` | string or null | yes | — |  |
| `client` | string or null | yes | — |  |
| `waited_s` | integer or number or null | no | — |  |
| `max_wait_s` | integer or null | no | — |  |

### `ActivityLease`

The open lease.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `lease_id` | string | yes | — |  |
| `kind` | string | yes | — |  |
| `client` | string or null | yes | — |  |
| `act` | string | yes | — |  |
| `since` | string | yes | — |  |
| `expires_at` | string | yes | — |  |

### `ActivityResident`

What is on the card.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `kind` | string | yes | — |  |
| `id` | string | yes | — |  |
| `since` | string | yes | — |  |
| `memory_bytes_estimate` | integer or number or null | yes | — |  |
| `engine_exit_code` | integer or null | yes | — |  |
| `reference` | Reference | no | — |  |
| `held_by` | ActivityHeld or null | yes | — |  |
| `unclaimed_since` | string or null | yes | — |  |

### `ActivityServer`

Who answered `GET /v1/activity`.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `name` | string | yes | — |  |
| `version` | string | yes | — |  |
| `api_version` | integer | yes | — |  |
| `backend` | string | yes | — |  |
| `uptime_s` | integer or number | yes | — |  |

### `ActivitySettings`

Recent writes through `PUT /v1/settings`.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `writes` | array of object | yes | — |  |

### `ActivitySlot`

The one accelerated lane. `queue_depth` counts the job on the lane (admitted or running) plus every job waiting in the server's queue.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `busy` | integer | yes | — |  |
| `of` | integer | yes | — |  |
| `queue_depth` | integer | yes | — |  |
| `accepts_work` | boolean | yes | — |  |

### `ActivitySlots`

Every lane this server admits work through.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `accelerated` | ActivitySlot | yes | — | The one accelerated lane. `queue_depth` counts the job on the lane (admitted or running) plus every job waiting in the server's queue. |

### `ArtifactRef`

An artifact of a previous job on this server, taken as an input.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | string | yes | — |  |
| `name` | string | yes | — |  |

### `Body_upload_v1_uploads_post`

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `file` | string | yes | — |  |

### `CardHeldDetails`

`server_busy` from the operator door: what holds the card, in the server's words.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `door` | `'operator'` | yes | — |  |
| `fact` | string | yes | — |  |
| `who` | string | yes | — |  |

### `ChoiceAnswer`

A choice question's distribution.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `type` | `'choice'` | no | `'choice'` | `choice`. |
| `choice` | string | yes | — | The most probable option (of those returned, in report mode). |
| `probabilities` | object of number or null | yes | — | Option name to probability, in option order, renormalised over the letters so they sum to 1 (a softmax over the label logits). Null only for an option reported missing. |
| `logprobs` | object of number or null | yes | — | Option name to ln of its `probabilities` entry, in option order; add ln `label_mass` (multiply the probability by `label_mass`) for the un-renormalised mass. NOT calibrated: one forward pass's reading, not a measured frequency. Null where the probability is null, or exactly 0 (`-Infinity` is not JSON). |
| `confidence` | number | yes | — | The largest renormalised probability. |
| `label_mass` | number | yes | — | The raw probability the option letters held together before renormalising (over the letters RETURNED, in report mode). Low means the model wanted to say something that is not an option. A renormalised probability times it is the un-renormalised mass. |
| `missing_labels` | array of string or null | no | — | Present only when the request said `missing: "report"` — absent, not null, otherwise: the options whose letter was not among the top tokens the engine returned, in option order, `[]` when none was. Nothing is invented for them; their `probabilities` and `logprobs` are null. |

### `ChoiceQuestion`

Pick one of named options. Labelled A, B, C… in the order given.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `type` | `'choice'` | yes | — | `choice`. |
| `instructions` | string | yes | — | The question, as a person would ask it: "Which team should handle this?". |
| `options` | object of string | yes | — | Option name to a one-line description, in the order the letters are assigned: the first option is `A`. At least 2; more than 26 is refused as `too_many_options`, because past `Z` there is no one-token label to read. |

### `DecideItem`

One item of the items form: its text, and optionally its own options.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `text` | string | yes | — | The item, quoted verbatim into its own question (never referred to by number): a transcript passage, or a question about the image. |
| `options` | object of string or null | no | — | This item's own options (name to one-line description, letters A, B, C… in the order given, 2 to 26), rendered inline under the item. Absent: the request's `options`. |

### `DecideItemsResponse`

An items-form decision: one choice distribution per item, in item order.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `model` | ModelProvenance | yes | — | Which weights answered (`{id, revision, fingerprint}`). |
| `engine` | string | yes | — | The engine kind that answered: `vllm`, `mlx-lm`, `mlx-vlm`. |
| `answers` | array of ChoiceAnswer | yes | — | One choice answer per item, in the request's item order, in the shape a lone choice question answers with. |
| `timing_ms` | ItemsTiming | yes | — | Crucible's clock. |
| `tokens` | ItemsTokens | yes | — | Prompt sizes. |

### `DecidePairing`

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `id` | string | yes | — |  |
| `user_code` | string | yes | — |  |
| `allow` | boolean | yes | — |  |

### `DecideRequest`

`POST /v1/decide`: one forward pass per question at the resident model, or a list of items about one state; nothing decoded or loaded.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `model` | string | yes | — | The Crucible model id, which must already be resident (`409 model_not_resident` otherwise). An upstream id (`<upstream>/<id>`) is refused `400 decide_needs_logprobs`: no upstream returns a distribution. |
| `state` | State | yes | — | What the questions are about: a string, used verbatim, or any other JSON value, serialised as compact JSON. Required and never null; may be `""` only when `images` carry the state. |
| `questions` | object of ChoiceQuestion or ScoreQuestion or YesNoQuestion or null | no | — | Question name to question. Names are single path members (no `/`, `\`, leading dot) and key the answers. Answers come back in this order. Exactly one of `questions` and `items` is sent. |
| `instructions` | string or null | no | — | The items form's ask, written under every item's text: "Which of the categories listed above does the speaker do in this passage?". Each item's question is `text`, a newline, then this; absent, `text` alone. Refused with `questions`. |
| `options` | object of string or null | no | — | The items form's shared options (name to one-line description, letters A, B, C… in the order given, 2 to 26). An item without its own `options` uses these. |
| `items` | array of DecideItem or null | no | — | The items form: an ordered list of choice questions about ONE state, each answered exactly as a lone choice question would be (it sees the state and its own question, never another item), in one request: on the Mac the shared state runs once and every item continues from its cache. Answers come back as a list in this order. At most 512 (`too_many_items`); token caps in docs/internals/api.md. |
| `images` | array of string or null | no | — | Base64 image files (PNG, JPEG, GIF or WebP; standard alphabet, padded, no whitespace, no `data:` prefix), read as part of the state, after its text. At most 8 (`too_many_images`), and only on a model whose manifest declares `image` (`400 model_text_only` otherwise). `[]` is the same as none. |
| `missing` | `'refuse'` or `'report'` | no | `'refuse'` | What to do when a label is not among the top tokens the engine returned. `refuse` (the default): the decision is `502 label_not_in_probs` naming the question and the letter. `report`: the door never invents a number — that option's probability and log-probability are null, it is named in the answer's `missing_labels`, and the renormalisation, `confidence`, `score` and `label_mass` run over the letters actually returned. A question whose EVERY label is missing is refused in both modes: there is no answer to report. |
| `queue` | QueueRequest or null | no | — | Wait for the model instead of being refused: `{}` or `{"max_wait_s": N}`. While the model is not resident, or every slot on its engine is taken, the request is held open in the server's queue (docs/QUEUE.md) and the model is loaded for it when its turn comes. Absent: refused as before. |

### `DecideResponse`

A decision: one distribution per question.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `model` | ModelProvenance | yes | — | Which weights answered (`{id, revision, fingerprint}`, docs/internals/jobs-runtime.md "Provenance sidecars"). |
| `engine` | string | yes | — | The engine kind that answered: `vllm`, `llama-server`, `mlx-lm`. |
| `answers` | object of ChoiceAnswer or ScoreAnswer or YesNoAnswer | yes | — | Question name to answer, in the request's question order. |
| `timing_ms` | DecideTiming | yes | — | Crucible's clock, per request. |
| `tokens` | DecideTokens | yes | — | Prompt sizes. |

### `DecideTiming`

Where the time went.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `total` | number | yes | — | The whole decision, ms, Crucible's clock. |
| `per_question` | object of ForwardTiming | yes | — | Each question's own request. |
| `prime` | ForwardTiming or null | yes | — | The shared prefix sent alone first — present when the decision had more than one question, null when it had one. |

### `DecideTokens`

How big the prompts were.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `per_question` | object of integer | yes | — | `usage.prompt_tokens` for each question's prompt. |
| `images` | integer | yes | — | How many images every prompt of this decision carried. |

### `ErrorBody`

What every refusal carries: branch on `code`, show a person `message`.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `code` | string | yes | — |  |
| `message` | string | yes | — |  |
| `details` | object or null | no | — |  |

### `ErrorEnvelope`

Every error answer: one `error` object.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `error` | ErrorBody | yes | — | What every refusal carries: branch on `code`, show a person `message`. |

### `ForwardTiming`

One request to the engine, timed by Crucible.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `wall_ms` | number | yes | — | Crucible's wall clock around the request, queueing and the one decoded token included. The OpenAI reply carries no prefill time, so this is the only duration there is. |
| `prompt_tokens` | integer | yes | — | `usage.prompt_tokens`: the whole prompt, cached part included. |
| `cached_tokens` | integer or null | yes | — | `usage.prompt_tokens_details.cached_tokens`, or null when the engine did not say. Never 0 for "unknown": a number nobody measured is not a measurement. |

### `HTTPValidationError`

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `detail` | array of ValidationError | no | — |  |

### `Info`

`GET /v1/info`: who this server is, what it runs on and what it serves.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `server` | object | yes | — |  |
| `role` | string | yes | — |  |
| `managed_by` | object of string or null | yes | — |  |
| `host` | object | yes | — |  |
| `job_types` | array of string | yes | — |  |
| `capabilities` | array of object | yes | — |  |
| `pages_engine` | object | yes | — |  |
| `terminal_states` | TerminalStates | yes | — | The states after which a job or a task never changes again. |
| `voice_sources` | object of VoiceSourceLabel | yes | — |  |
| `service_commands` | array of ServiceCommand | yes | — |  |

### `ItemsTiming`

Where the items form's time went.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `total` | number | yes | — | The whole decision, ms, Crucible's clock. |
| `engine_requests` | integer | yes | — | 1 when the engine read every item in one batched request (mlx-lm, mlx-vlm); otherwise one per item plus the shared prefix sent first. |

### `ItemsTokens`

How big the items form's prompts were.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `shared` | integer or null | yes | — | The tokens every item's prompt shares (system, state, images), run once; null where each item went as its own request. |
| `per_item` | array of integer | yes | — | Each item's whole prompt (shared part included), in item order. |
| `images` | integer | yes | — | How many images every item's prompt carried. |

### `JobBusyDetails`

`server_busy` from the job door: the job that holds the lane.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `door` | `'job'` | yes | — |  |
| `holder` | string or null | yes | — |  |
| `job_id` | string | yes | — |  |
| `type` | string | yes | — |  |
| `model` | string or null | yes | — |  |
| `status` | string | yes | — |  |
| `since` | string | yes | — |  |
| `progress` | integer or number | yes | — |  |
| `message` | string or null | yes | — |  |

### `JobCreate`

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `type` | string | yes | — |  |
| `model` | string or null | no | — |  |
| `params` | object | no | — |  |
| `inputs` | object of JobInput | no | — |  |
| `client_ref` | string or null | no | — | The client's own name for this work, echoed on the job record and never read by the server. |
| `hold` | boolean | no | `False` | Hold the job from creation, as `POST /v1/jobs/{id}/hold` would, so its artifacts outlive being fetched. |
| `queue` | QueueRequest or null | no | — | Opt in to the server's queue: while the lane is busy the job waits (status `queued`) instead of being refused `409 server_busy`. Without it, a busy server refuses as it always has. |

### `JobFailure`

Why a job failed.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `code` | string | yes | — |  |
| `message` | string | yes | — |  |

### `JobInput`

One named input: an uploaded blob, inline base64 bytes, or a previous job's artifact.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `blob_id` | string or null | no | — |  |
| `inline_base64` | string or null | no | — |  |
| `artifact` | ArtifactRef or null | no | — |  |

### `JobRemoval`

Why a job left the queue without running: `removed` is not `failed`.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `reason` | `'operator'` or `'client'` or `'expired'` or `'server_restart'` | yes | — |  |
| `message` | string | yes | — |  |
| `waited_s` | integer or number or null | yes | — |  |
| `at` | string | yes | — |  |

### `JobStatus`

`GET /v1/jobs/{id}`. A job type adds its own `done_extra` keys beside these.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | string | yes | — |  |
| `type` | string | yes | — |  |
| `model` | string or null | yes | — |  |
| `status` | string | yes | — |  |
| `progress` | integer or number | yes | — |  |
| `position` | integer or null | yes | — |  |
| `error` | JobFailure or null | yes | — |  |
| `artifacts` | array of string | yes | — |  |
| `created` | string | yes | — |  |
| `started` | string or null | yes | — |  |
| `finished` | string or null | yes | — |  |
| `client_ref` | string or null | yes | — |  |
| `interrupted_at` | string or null | yes | — |  |
| `held_by` | string or null | yes | — |  |
| `held_since` | string or null | yes | — |  |
| `chunks_done` | array of integer | yes | — |  |
| `chunks_total` | integer or null | yes | — |  |
| `chunk_at` | string or null | yes | — |  |
| `resume_id` | string or null | yes | — |  |
| `resumed` | boolean | yes | — |  |
| `lease_id` | string or null | no | — |  |
| `sampling` | object or null | no | — |  |
| `removal` | JobRemoval or null | no | — |  |

### `LeaseOpen`

`POST /v1/models/{id}/lease`: the act the lease is for and how long it lasts; neither has a default.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `act` | string | yes | — |  |
| `ttl_seconds` | integer | yes | — |  |

### `ModelProvenance`

The weights that made the decision, as an artifact sidecar names them.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `id` | string | yes | — | The Crucible model id. |
| `revision` | string | yes | — | The revision the resident engine was started on. |
| `fingerprint` | string | yes | — | `<id>@<revision>`. |

### `Ping`

`GET /v1/ping`: enough for a client to tell a Crucible from anything else.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `crucible` | `True` | yes | — |  |
| `name` | string | yes | — |  |
| `api_version` | integer | yes | — |  |
| `pairing_version` | integer | yes | — |  |

### `PollPairing`

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `id` | string | yes | — |  |
| `device_code` | string | yes | — |  |

### `QueueItem`

One job waiting in the server's queue.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `position` | integer | yes | — |  |
| `job_id` | string | yes | — |  |
| `type` | string | yes | — |  |
| `model` | string or null | yes | — |  |
| `client` | string or null | yes | — |  |
| `client_ref` | string or null | yes | — |  |
| `submitted` | string | yes | — |  |
| `waited_s` | integer or number | yes | — |  |
| `max_wait_s` | integer | yes | — |  |
| `expires_at` | string | yes | — |  |
| `lease_holder` | boolean | yes | — |  |
| `kind` | `'job'` or `'call'` | yes | — |  |

### `QueueList`

`GET /v1/queue`: the waiting jobs in order, and the queue's limits.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `items` | array of QueueItem | yes | — |  |
| `depth` | integer | yes | — |  |
| `limits` | object | yes | — |  |

### `QueueRemoved`

`DELETE /v1/queue/{job_id}`: the job left the queue.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | string | yes | — |  |
| `status` | `'removed'` | yes | — |  |
| `reason` | `'operator'` | yes | — |  |

### `QueueRequest`

Wait in the server's queue instead of being refused `409 server_busy`.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `max_wait_s` | integer | no | `3600` | How long the job may wait for the lane before it is removed `expired`. |

### `ScoreAnswer`

A score question's distribution and its expected level.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `type` | `'score'` | no | `'score'` | `score`. |
| `score` | number | yes | — | Σ (1-based level index × p) over the levels returned: 1.0 is certainly the lowest level. |
| `level` | string | yes | — | The most probable level (of those returned, in report mode). |
| `probabilities` | object of number or null | yes | — | Level to renormalised probability, lowest level first. Null only for a level reported missing. |
| `logprobs` | object of number or null | yes | — | Level to ln of its `probabilities` entry, lowest first; add ln `label_mass` for the un-renormalised mass. NOT calibrated. Null where the probability is null, or exactly 0 (`-Infinity` is not JSON). |
| `confidence` | number | yes | — | The largest renormalised probability. |
| `label_mass` | number | yes | — | The raw probability the level letters held together before renormalising (over the letters RETURNED, in report mode). Low means the model wanted to say something that is not an option. A renormalised probability times it is the un-renormalised mass. |
| `missing_labels` | array of string or null | no | — | Present only when the request said `missing: "report"` — absent, not null, otherwise: the levels whose letter was not among the top tokens the engine returned, in level order, `[]` when none was. Nothing is invented for them; their `probabilities` and `logprobs` are null. |

### `ScoreQuestion`

Place the state on an ordered scale. `score` is the expected level.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `type` | `'score'` | yes | — | `score`. |
| `instructions` | string | yes | — | The question: "How frustrated is the customer?". |
| `levels` | array of string | yes | — | The scale, lowest first, 2 to 10 unique levels. Level i (1-based) is the value the expected `score` is computed with. |

### `ServerBusy`

The `409 server_busy` envelope.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `error` | ServerBusyBody | yes | — | A `409 server_busy` refusal; `details.door` says which door refused. |

### `ServerBusyBody`

A `409 server_busy` refusal; `details.door` says which door refused.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `code` | `'server_busy'` | yes | — |  |
| `message` | string | yes | — |  |
| `details` | JobBusyDetails or CardHeldDetails | yes | — |  |

### `ServiceCommand`

A command, typed on the server itself, that runs it as a machine service.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `command` | string | yes | — |  |
| `does` | string | yes | — |  |

### `StartPairing`

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `client_name` | string | yes | — |  |

### `StreamOp`

`POST /v1/tts/stream/{id}`: one op. `say` needs `id`, `text` and `take` (no default); `cancel` needs `id`; `cancel_all` and `close` take nothing.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `op` | string | yes | — |  |
| `id` | string or null | no | — |  |
| `text` | string or null | no | — |  |
| `take` | integer or null | no | — |  |

### `StreamOpen`

`POST /v1/tts/stream`: the voice and language of a streaming session; neither has a default.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `voice` | string | yes | — |  |
| `language` | string | yes | — |  |

### `TaskCreate`

`POST /v1/tasks`: one operator task. `type` decides which fields are required and which are refused.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `type` | string | yes | — |  |
| `kind` | string or null | no | — |  |
| `id` | string or null | no | — |  |
| `job_type` | string or null | no | — |  |
| `narrator_engine` | string or null | no | — |  |
| `module` | object or null | no | — |  |
| `target` | string or null | no | — |  |

### `TerminalStates`

The states after which a job or a task never changes again.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `jobs` | array of string | yes | — |  |
| `tasks` | array of string | yes | — |  |

### `ValidationError`

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `loc` | array of string or integer | yes | — |  |
| `msg` | string | yes | — |  |
| `type` | string | yes | — |  |

### `VoiceInfo`

One row of `GET /v1/voices`: a voice and where it stands on this server.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `id` | string | yes | — |  |
| `display` | string | yes | — |  |
| `kind` | string or null | yes | — |  |
| `language` | string or null | yes | — |  |
| `narrator_engine` | string or null | yes | — |  |
| `backend_supported` | boolean | yes | — |  |
| `installed` | boolean | yes | — |  |
| `resident` | boolean | yes | — |  |
| `orphan` | boolean or null | yes | — |  |
| `loadable` | boolean | yes | — |  |
| `reason` | string or null | yes | — |  |
| `revision` | string or null | yes | — |  |
| `fingerprint` | string or null | yes | — |  |
| `source` | string or null | yes | — |  |
| `identity_basis` | string or null | yes | — |  |
| `memory_bytes_estimate` | integer or number or null | yes | — |  |
| `estimate_basis` | string or null | yes | — |  |
| `serving` | object or null | yes | — |  |
| `max_chars` | integer or null | yes | — |  |
| `max_chars_basis` | string or null | yes | — |  |
| `pace_basis` | string or null | yes | — |  |
| `inherited_from` | string or null | yes | — |  |
| `manifest` | string or null | yes | — |  |
| `sample_rate` | integer or null | yes | — |  |
| `takes` | integer | yes | — |  |
| `needs_reference` | boolean | yes | — |  |
| `pace` | object or null | yes | — |  |
| `ref` | string or null | no | — | The tag this voice follows on its repo (`crucible`); null for an exact-sha pin. |
| `latest_revision` | string or null | no | — | The commit the tag named at the last explicit check (`POST /v1/voices/updates`, `crucible voices check-updates`, or a pull); never looked up by this GET. |
| `update_available` | boolean | no | `False` | A pull would move this voice from `revision` to `latest_revision`. |
| `update_checked_at` | string or null | no | — | When the tag was last looked up. |
| `update_error` | string or null | no | — | Why the last look-up failed; the voice stays on the revision it has. |
| `sampling` | object of integer or number or null | no | — | `{temperature, top_p, top_k}` this backend's arm renders take 0 with: the sampling Crucible writes into narrator's voice document. |
| `edge_fade_ms` | object of integer or number or null | no | — | `{in, out}`: raised-cosine fades, in milliseconds, at each chunk edge on this arm. |
| `chunk_gap` | object or null | no | — | The silence to add after each chunk: `inject_s` (net of the model's own tail), `target_join_s`, `model_self_tail_s`, optional `reader_sentence_gap_s` and `model_internal_gap_s`, and `rule`, `method`, `source`, `measured_on`. |
| `reference_seconds_cap` | integer or number or null | no | — | The most reference-clip audio this arm takes, in seconds. |
| `allowed_controls` | array of string or null | no | — | The inline control tokens (`<\|group:name\|>`) this arm allows; `[]` allows none. |

### `VoiceSourceLabel`

How to name a voice row's `manifest` source to a person, and the tone to show it in.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `label` | string | yes | — |  |
| `tone` | `'ok'` or `'warn'` or `'floor'` | yes | — |  |

### `YesNoAnswer`

A yesno question's probability.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `type` | `'yesno'` | no | `'yesno'` | `yesno`. |
| `p` | number | yes | — | Renormalised P(Yes). In report mode with `A` or `B` missing it is the returned one renormalised alone — 1.0 or 0.0, which is honest and useless: gate on `label_mass`. |
| `logprob` | number or null | yes | — | ln `p`; add ln `label_mass` for the un-renormalised mass. NOT calibrated. Null when `p` is exactly 0 (`-Infinity` is not JSON) — a report-mode answer with `Yes` missing. |
| `label_mass` | number | yes | — | The raw probability the letters `A` and `B` held together before renormalising (over the letters RETURNED, in report mode). Low means the model wanted to say something that is not an option. A renormalised probability times it is the un-renormalised mass. |
| `missing_labels` | array of string or null | no | — | Present only when the request said `missing: "report"` — absent, not null, otherwise: `["Yes"]` or `["No"]` when that letter was not among the top tokens the engine returned, `[]` when both were (both missing is refused). |

### `YesNoQuestion`

Is a statement true of the state? `p` is P(Yes).

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `type` | `'yesno'` | yes | — | `yesno`. |
| `instructions` | string | yes | — | The statement to judge: "The message conveys urgency". |
